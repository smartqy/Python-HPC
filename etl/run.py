"""ETL pipeline CLI. Each stage processes a list of batches and is idempotent per batch.

    python -m etl.run init                         create database + tables (warehouse/ddl.sql)
    python -m etl.run discover                     print batches whose landing data is new or changed
    python -m etl.run validate    --batch 0 1      schema / completeness checks on landing
    python -m etl.run blocks      --batch 0 1      raw_grid -> fact_block_temp (+ quarantine)
    python -m etl.run convergence --batch 0 1      landing convergence -> fact_convergence
    python -m etl.run summary     --batch 0 1      fact_block_temp -> dim_building_summary
    python -m etl.run reconcile   --batch 0 1      summary vs simulation summary_stats, tol 1e-6
    python -m etl.run commit      --batch 0 1      record the processed landing version
    python -m etl.run all         [--batch ...]    every stage above for the given (default: pending) batches
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from pyspark.sql import functions as F

from etl import transforms as tf
from etl.blockstats import T_WALL_MIN
from etl.config import DB, LAKE, LANDING, spark_session

LANDING_TABLES = ['raw_grid', 'convergence', 'run_meta']
EXPECTED_SCHEMA = {
    'raw_grid': {'building_id': 'string', 'grid': 'binary', 'mask': 'binary', 'batch_id': 'int'},
    'convergence': {'building_id': 'string', 'iter': 'int', 'residual': 'double', 'mean_temp': 'double',
                    'elapsed_ms': 'double', 'batch_id': 'int'},
    'run_meta': {'building_id': 'string', 'backend': 'string', 'n_iter': 'int', 'converged': 'boolean',
                 'run_seconds': 'double', 'ref_area_cells': 'bigint', 'ref_mean_temp': 'double',
                 'ref_std_temp': 'double', 'ref_pct_above_18': 'double', 'ref_pct_below_15': 'double',
                 'batch_id': 'int'},
}
DDL = Path(__file__).resolve().parent.parent / 'warehouse' / 'ddl.sql'


class DataQualityError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- helpers

def landing_batches():
    d = LANDING / 'raw_grid'
    return sorted(int(p.name.split('=')[1]) for p in d.glob('batch_id=*')) if d.exists() else []


def fingerprint(batch_id):
    h = hashlib.sha256()
    for table in LANDING_TABLES:
        for f in sorted((LANDING / table / f'batch_id={batch_id:05d}').glob('*.parquet')):
            st = f.stat()
            h.update(f'{table}/{f.name}:{st.st_size}:{st.st_mtime_ns};'.encode())
    return h.hexdigest()[:16]


def read_landing(spark, table, batches):
    paths = [str(LANDING / table / f'batch_id={b:05d}') for b in batches]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise DataQualityError(f'missing landing data: {missing}')
    return spark.read.option('basePath', str(LANDING / table)).parquet(*paths)


def overwrite(df, table):
    """INSERT OVERWRITE by position; with dynamic partition overwrite only the batches present
    in df are replaced, so re-running a batch never duplicates rows."""
    cols = df.sparkSession.table(f'{DB}.{table}').columns
    df.select(*cols).write.insertInto(f'{DB}.{table}', overwrite=True)


def write_dq(spark, stage, batches, checks):
    """checks: list of (batch_id, check_name, table_name, failed_rows, total_rows, status, detail)."""
    rows = [dict(zip(['batch_id', 'check_name', 'table_name', 'failed_rows', 'total_rows', 'status', 'detail'], c))
            for c in checks]
    schema = ('batch_id int, check_name string, table_name string, failed_rows bigint, total_rows bigint, '
              'status string, detail string')
    df = (spark.createDataFrame(rows, schema).withColumn('run_ts', F.current_timestamp())
          .withColumn('stage', F.lit(stage)))
    overwrite(df, 'dq_results')
    for c in checks:
        if c[5] != 'PASS':
            print(f'  DQ {c[5]} batch={c[0]} {c[1]}: {c[3]}/{c[4]} {c[6] or ""}')
    failed = [c for c in checks if c[5] == 'FAIL']
    if failed:
        raise DataQualityError(f'{stage}: {len(failed)} failed check(s): {[(c[0], c[1]) for c in failed]}')


def counts_by_batch(df, batches):
    got = {r['batch_id']: r['n'] for r in df.groupBy('batch_id').agg(F.count('*').alias('n')).collect()}
    return {b: got.get(b, 0) for b in batches}


# ---------------------------------------------------------------------------- stages

def stage_init(spark, batches=None):
    sql = DDL.read_text().replace('${LAKE}', LAKE.as_uri().rstrip('/'))
    sql = '\n'.join(l for l in sql.splitlines() if not l.strip().startswith('--'))
    for stmt in filter(str.strip, sql.split(';')):
        spark.sql(stmt)
    print(f'tables in {DB}:', [t.name for t in spark.catalog.listTables(DB)])


def stage_discover(spark, batches=None):
    committed = {}
    if spark.catalog.tableExists(f'{DB}.etl_batch_log'):
        committed = {r['batch_id']: r['landing_fingerprint']
                     for r in spark.table(f'{DB}.etl_batch_log').collect()}
    pending = [b for b in landing_batches() if committed.get(b) != fingerprint(b)]
    print(json.dumps(pending))
    return pending


def stage_validate(spark, batches):
    checks = []
    frames = {}
    for table in LANDING_TABLES:
        df = read_landing(spark, table, batches)
        frames[table] = df
        actual = {f.name: f.dataType.simpleString() for f in df.schema}
        if actual != EXPECTED_SCHEMA[table]:
            for b in batches:
                checks.append((b, 'schema', table, None, None, 'FAIL', f'expected {EXPECTED_SCHEMA[table]}, got {actual}'))
            continue
        for b in batches:
            checks.append((b, 'schema', table, 0, None, 'PASS', None))

    if not any(c[5] == 'FAIL' for c in checks):
        raw, conv, meta = frames['raw_grid'], frames['convergence'], frames['run_meta']
        raw_ids = raw.select('batch_id', 'building_id')
        meta_ids = meta.select('batch_id', 'building_id')
        n_raw = counts_by_batch(raw, batches)
        per_check = {
            'raw_malformed_grid': ('raw_grid', tf.malformed_raw(raw)),
            'raw_duplicate_building': ('raw_grid', raw_ids.groupBy('batch_id', 'building_id').count().where('count > 1')),
            'meta_missing_for_raw': ('run_meta', raw_ids.join(meta_ids, ['batch_id', 'building_id'], 'left_anti')),
            'raw_missing_for_meta': ('raw_grid', meta_ids.join(raw_ids, ['batch_id', 'building_id'], 'left_anti')),
            'convergence_missing_for_raw': ('convergence', raw_ids.join(
                conv.select('batch_id', 'building_id').distinct(), ['batch_id', 'building_id'], 'left_anti')),
        }
        for name, (table, bad) in per_check.items():
            n_bad = counts_by_batch(bad, batches)
            for b in batches:
                checks.append((b, name, table, n_bad[b], n_raw[b], 'FAIL' if n_bad[b] else 'PASS', None))
        not_conv = counts_by_batch(meta.where(~F.col('converged')), batches)
        for b in batches:
            checks.append((b, 'not_converged', 'run_meta', not_conv[b], n_raw[b], 'WARN' if not_conv[b] else 'PASS',
                           'hit max_iter before atol' if not_conv[b] else None))
            if n_raw[b] == 0:
                checks.append((b, 'empty_batch', 'raw_grid', 0, 0, 'FAIL', None))
    write_dq(spark, 'validate', batches, checks)


def stage_blocks(spark, batches):
    read_landing(spark, 'raw_grid', batches)  # fails early if a batch is missing
    files = [(f, b) for b in batches for f in sorted((LANDING / 'raw_grid' / f'batch_id={b:05d}').glob('*.parquet'))]
    blocks = tf.raw_files_to_blocks(spark, files).persist()
    clean, quarantine = tf.split_blocks(blocks)
    # One file per batch partition instead of one per task: avoids the small-files problem.
    overwrite(clean.repartition('batch_id'), 'fact_block_temp')
    overwrite(quarantine.repartition('batch_id'), 'quarantine_block_temp')
    total = counts_by_batch(blocks, batches)
    empty = counts_by_batch(blocks.where('interior_cells = 0'), batches)
    q_buildings = counts_by_batch(quarantine.select('batch_id', 'building_id').distinct(), batches)
    fact = spark.table(f'{DB}.fact_block_temp').where(F.col('batch_id').isin(batches))
    kept = counts_by_batch(fact, batches)
    leaky = counts_by_batch(fact.where(F.col('min_temp') < T_WALL_MIN).select('batch_id', 'building_id').distinct(),
                            batches)
    blocks.unpersist()
    checks = []
    for b in batches:
        checks.append((b, 'buildings_with_mask_leak', 'fact_block_temp', leaky[b], None, 'WARN' if leaky[b] else 'PASS',
                       'interior cells adjacent to outside cells, temperature < 5 degC' if leaky[b] else None))
        checks.append((b, 'empty_blocks_dropped', 'fact_block_temp', empty[b], total[b], 'PASS', None))
        checks.append((b, 'quarantined_buildings', 'quarantine_block_temp', q_buildings[b], None,
                       'WARN' if q_buildings[b] else 'PASS', None))
        checks.append((b, 'rows_written', 'fact_block_temp', 0, kept[b], 'PASS' if kept[b] else 'FAIL', None))
    write_dq(spark, 'blocks', batches, checks)


def stage_convergence(spark, batches):
    conv = read_landing(spark, 'convergence', batches)
    clean, bad = tf.clean_convergence(conv)
    overwrite(clean.repartition('batch_id'), 'fact_convergence')
    total, n_bad = counts_by_batch(conv, batches), counts_by_batch(bad, batches)
    write_dq(spark, 'convergence', batches,
             [(b, 'invalid_rows_dropped', 'fact_convergence', n_bad[b], total[b], 'WARN' if n_bad[b] else 'PASS', None)
              for b in batches])


def stage_summary(spark, batches):
    blocks = spark.table(f'{DB}.fact_block_temp').where(F.col('batch_id').isin(batches))
    meta = read_landing(spark, 'run_meta', batches)
    overwrite(tf.summarize(blocks, meta).repartition('batch_id'), 'dim_building_summary')


def stage_reconcile(spark, batches, tol=1e-6):
    summary = spark.table(f'{DB}.dim_building_summary').where(F.col('batch_id').isin(batches))
    meta = read_landing(spark, 'run_meta', batches)
    rec = tf.reconcile(summary, meta, tol).persist()
    worst = rec.agg(*[F.max(f'diff_{m}').alias(m) for m in tf.RECONCILE_METRICS]).first().asDict()
    total, mismatched = counts_by_batch(rec, batches), counts_by_batch(rec.where(~F.col('match')), batches)
    rec.unpersist()
    print(f'  reconcile max abs diff: { {k: f"{v:.2e}" for k, v in worst.items() if v is not None} }')
    write_dq(spark, 'reconcile', batches,
             [(b, f'summary_vs_reference_tol_{tol:g}', 'dim_building_summary', mismatched[b], total[b],
               'FAIL' if mismatched[b] else 'PASS', json.dumps(worst)) for b in batches])


def stage_commit(spark, batches):
    n = counts_by_batch(spark.table(f'{DB}.dim_building_summary').where(F.col('batch_id').isin(batches)), batches)
    rows = [(b, fingerprint(b), n[b]) for b in batches]
    df = (spark.createDataFrame(rows, 'batch_id int, landing_fingerprint string, n_buildings bigint')
          .withColumn('committed_ts', F.current_timestamp()))
    overwrite(df, 'etl_batch_log')


STAGES = {
    'init': stage_init, 'discover': stage_discover, 'validate': stage_validate, 'blocks': stage_blocks,
    'convergence': stage_convergence, 'summary': stage_summary, 'reconcile': stage_reconcile,
    'commit': stage_commit,
}
PIPELINE = ['validate', 'blocks', 'convergence', 'summary', 'reconcile', 'commit']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=[*STAGES, 'all'])
    p.add_argument('--batch', type=int, nargs='*', help='batch ids (default: pending batches for "all")')
    p.add_argument('--batches-json', help='batch ids as a JSON list (used by Airflow XCom)')
    args = p.parse_args(argv)

    batches = args.batch
    if args.batches_json is not None:
        batches = json.loads(args.batches_json)
    spark = spark_session(f'hpc-temp-etl:{args.stage}')
    if args.stage in ('init', 'discover'):
        STAGES[args.stage](spark)
        return
    if args.stage == 'all':
        stage_init(spark)
        if batches is None:
            batches = stage_discover(spark)
    if not batches:
        print('no batches to process')
        return
    for name in (PIPELINE if args.stage == 'all' else [args.stage]):
        t0 = time.perf_counter()
        STAGES[name](spark, batches)
        print(json.dumps({'stage': name, 'batches': batches, 'seconds': round(time.perf_counter() - t0, 2)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except DataQualityError as e:
        print(f'DATA QUALITY FAILURE: {e}', file=sys.stderr)
        sys.exit(2)
