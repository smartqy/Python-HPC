"""Paths and SparkSession setup shared by all ETL jobs.

DATA_ROOT layout:
    landing/   written by simulation/export.py (raw layer, Hive-style batch_id=NNNNN dirs)
    lake/      tables managed by the ETL (clean + summary layers) and the warehouse dir
               (override with LAKE_DIR, e.g. to keep the Docker and laptop metastores apart)

Set HIVE_METASTORE_URIS (e.g. thrift://hive-metastore:9083) to use the standalone Hive
Metastore from docker-compose; otherwise an embedded Derby metastore under lake/ is used,
so the same jobs run on a laptop without Docker.
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parent.parent / 'data')).resolve()
LANDING = DATA_ROOT / 'landing'
LAKE = Path(os.environ.get('LAKE_DIR', DATA_ROOT / 'lake')).resolve()
DB = 'dw'


def spark_session(app_name='hpc-temp-etl', hive=True):
    from pyspark.sql import SparkSession

    b = (SparkSession.builder.appName(app_name)
         # local[N,F]: unlike a cluster, plain local[*] never retries a failed task, so one crashed
         # Python worker would fail the whole job; allow 3 attempts per task. N=4: on the M2 dev
         # laptop (4 performance + 4 efficiency cores) 8 workers are slower than 4.
         .master(os.environ.get('SPARK_MASTER', 'local[4,3]'))
         .config('spark.driver.memory', os.environ.get('SPARK_DRIVER_MEMORY', '3g'))
         .config('spark.sql.session.timeZone', 'UTC')
         .config('spark.sql.shuffle.partitions', os.environ.get('SPARK_SHUFFLE_PARTITIONS', '8'))
         # A raw-grid row is 2 MB; keep Arrow batches small so mapInPandas stays within memory.
         .config('spark.sql.execution.arrow.maxRecordsPerBatch', '8')
         .config('spark.sql.files.maxPartitionBytes', str(64 * 1024 * 1024))
         # Overwrite only the partitions being written -> re-running a batch is idempotent.
         .config('spark.sql.sources.partitionOverwriteMode', 'dynamic')
         .config('hive.exec.dynamic.partition', 'true')
         .config('hive.exec.dynamic.partition.mode', 'nonstrict')
         .config('spark.sql.warehouse.dir', str(LAKE / 'warehouse'))
         .config('spark.ui.showConsoleProgress', 'false')
         .config('spark.python.worker.faulthandler.enabled', 'true')
         .config('spark.driver.extraJavaOptions', f'-Dderby.system.home={LAKE}'))
    if hive:
        uris = os.environ.get('HIVE_METASTORE_URIS')
        if uris:
            b = b.config('spark.hadoop.hive.metastore.uris', uris)
        else:
            b = b.config('spark.hadoop.javax.jdo.option.ConnectionURL',
                         f'jdbc:derby:;databaseName={LAKE / "metastore_db"};create=true')
        b = b.enableHiveSupport()
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    return spark
