"""Wall Heating ETL: landing Parquet -> clean facts -> building summary -> reconciled warehouse.

    simulate_sample -> init_warehouse -> discover_batches -> has_new_batches
        -> validate -> [blocks, convergence] -> summary -> reconcile -> commit

Incremental: discover_batches returns only batches whose landing files are new or changed since
their last commit (fingerprint in dw.etl_batch_log). Each Spark stage overwrites exactly those
batch_id partitions, so a retry or a manual re-run never duplicates rows.

The full 4,571-building simulation runs on the DTU GPU nodes (simulation/export_job.sh) and is
copied into data/landing/; simulate_sample only runs the 20 bundled floor plans with Numba on the
CPU so the pipeline can be demonstrated end to end without a GPU.
"""
import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator

PROJECT = os.environ.get('PROJECT_DIR', '/opt/project')
log = logging.getLogger(__name__)


def alert_on_failure(context):
    """Failure alert. Logged only; swap for Slack/e-mail in a real deployment."""
    ti = context['task_instance']
    log.error('ALERT: task %s in run %s failed after %s tries (log: %s)',
              ti.task_id, context['run_id'], ti.try_number, ti.log_url)


def has_batches(ti):
    batches = ti.xcom_pull(task_ids='discover_batches') or '[]'
    log.info('pending batches: %s', batches)
    return batches.strip() != '[]'


default_args = {
    'owner': 'otis',
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'on_failure_callback': alert_on_failure,
    'execution_timeout': timedelta(hours=2),
}

BATCHES = "--batches-json '{{ ti.xcom_pull(task_ids=\"discover_batches\") }}'"
ETL = f'cd {PROJECT} && python -m etl.run'

with DAG(
    dag_id='hpc_temp_etl',
    description='Jacobi simulation results -> Spark ETL -> Hive warehouse',
    start_date=datetime(2026, 10, 1),
    schedule='@hourly',
    catchup=False,
    max_active_runs=1,          # two runs must never write the same batch concurrently
    default_args=default_args,
    params={'simulate_sample': Param(True, type='boolean',
                                     description='run the bundled sample floor plans with Numba first')},
    tags=['hpc', 'spark', 'hive'],
) as dag:
    simulate_sample = BashOperator(
        task_id='simulate_sample',
        bash_command=(
            '{% if params.simulate_sample %}'
            f'cd {PROJECT} && python simulation/export.py --data-dir data/sample '
            '--out-dir "$DATA_ROOT/landing" --batch-size 5 --backend numba --skip-existing'
            '{% else %}echo "skipped: using landing data from the HPC cluster"{% endif %}'
        ),
    )
    init_warehouse = BashOperator(task_id='init_warehouse', bash_command=f'{ETL} init')
    # The last stdout line (a JSON list of batch ids) is pushed to XCom.
    discover = BashOperator(task_id='discover_batches', bash_command=f'set -o pipefail; {ETL} discover 2>/dev/null | tail -n 1')
    gate = ShortCircuitOperator(task_id='has_new_batches', python_callable=has_batches)

    stages = {name: BashOperator(task_id=name, bash_command=f'{ETL} {name} {BATCHES}')
              for name in ('validate', 'blocks', 'convergence', 'summary', 'reconcile', 'commit')}

    simulate_sample >> init_warehouse >> discover >> gate >> stages['validate']
    stages['validate'] >> [stages['blocks'], stages['convergence']] >> stages['summary']
    stages['summary'] >> stages['reconcile'] >> stages['commit']
