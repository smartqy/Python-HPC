"""Spark counterpart of etl/pandas_baseline.py: same transforms, plain Parquet output, no metastore,
so both engines do exactly the same work."""
import argparse
import time
from pathlib import Path

from etl import transforms as tf
from etl.config import spark_session


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--landing', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['rows', 'files'], default='files',
                   help='rows: JVM reads grids and streams them to Python; files: Python workers read Parquet')
    a = p.parse_args()

    t_start = time.perf_counter()
    spark = spark_session('bench-spark', hive=False)
    t0 = time.perf_counter()
    meta = spark.read.option('basePath', str(a.landing / 'run_meta')).parquet(str(a.landing / 'run_meta'))
    if a.mode == 'rows':
        raw = spark.read.option('basePath', str(a.landing / 'raw_grid')).parquet(str(a.landing / 'raw_grid'))
        blocks = tf.raw_to_blocks(raw)
    else:
        files = [(f, int(f.parent.name.split('=')[1])) for f in sorted((a.landing / 'raw_grid').glob('batch_id=*/*.parquet'))]
        blocks = tf.raw_files_to_blocks(spark, files)
    blocks = blocks.persist()
    clean, _ = tf.split_blocks(blocks)
    clean.write.mode('overwrite').parquet(str(a.out / 'fact_block_temp'))
    clean = spark.read.parquet(str(a.out / 'fact_block_temp'))
    summary = tf.summarize(clean, meta)
    summary.write.mode('overwrite').parquet(str(a.out / 'dim_building_summary'))
    n_blocks, n_buildings = clean.count(), spark.read.parquet(str(a.out / 'dim_building_summary')).count()
    t1 = time.perf_counter()
    print(f'RESULT engine=spark_{a.mode} buildings={n_buildings} blocks={n_blocks} job_seconds={t1 - t0:.3f} '
          f'startup_seconds={t0 - t_start:.3f}')
    spark.stop()


if __name__ == '__main__':
    main()
