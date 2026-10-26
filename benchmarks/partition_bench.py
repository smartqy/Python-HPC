"""Compare query times on differently partitioned copies of the same tables.

Input: the Parquet output of benchmarks/spark_job.py (fact_block_temp + dim_building_summary).
Layouts:
    fact: none | batch_id | temp_band          dim: none | temp_band | area_band
Each query in warehouse/queries.sql runs `--runs` times per layout after one warm-up; medians and
the number of files each layout has are written to benchmarks/results/partition_bench.csv.

    python benchmarks/partition_bench.py --input data/scaled/_bench_out/spark
"""
import argparse
import csv
import re
import shutil
import statistics
import time
from pathlib import Path

from etl.config import spark_session

REPO = Path(__file__).resolve().parent.parent
FACT_LAYOUTS = [None, 'batch_id', 'temp_band']
DIM_LAYOUTS = [None, 'temp_band', 'area_band']


def load_queries():
    text = (REPO / 'warehouse' / 'queries.sql').read_text()
    out = {}
    for block in text.split('-- name: ')[1:]:
        name, body = block.split('\n', 1)
        sql = '\n'.join(l for l in body.splitlines() if not l.strip().startswith('--')).strip().rstrip(';')
        out[name.strip()] = sql
    return out


def write_layout(df, path, part):
    if path.exists():
        shutil.rmtree(path)
    w = (df.repartition(part) if part else df.coalesce(8)).write
    (w.partitionBy(part) if part else w).parquet(str(path))
    return len(list(path.rglob('*.parquet')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--runs', type=int, default=3)
    a = p.parse_args()

    spark = spark_session('partition-bench', hive=False)
    work = REPO / 'data' / 'scaled' / '_partition_bench'
    dim = spark.read.parquet(str(a.input / 'dim_building_summary'))
    fact = (spark.read.parquet(str(a.input / 'fact_block_temp'))
            .join(dim.select('building_id', 'temp_band', 'area_band'), 'building_id'))
    n_fact, n_dim = fact.count(), dim.count()
    print(f'fact rows {n_fact}, dim rows {n_dim}')

    files = {}
    for kind, df, layouts in (('fact', fact, FACT_LAYOUTS), ('dim', dim, DIM_LAYOUTS)):
        for part in layouts:
            name = f'{kind}_{part or "nopart"}'
            files[name] = write_layout(df, work / name, part)
            spark.read.parquet(str(work / name)).createOrReplaceTempView(name)

    queries = load_queries()
    rows = []
    for qname, sql in queries.items():
        kind = 'fact' if '{fact}' in sql else 'dim'
        for part in (FACT_LAYOUTS if kind == 'fact' else DIM_LAYOUTS):
            view = f'{kind}_{part or "nopart"}'
            q = re.sub(r'\{(fact|dim)\}', view, sql)
            spark.sql(q).collect()  # warm-up (JIT, file listing)
            times = []
            for _ in range(a.runs):
                t0 = time.perf_counter()
                spark.sql(q).collect()
                times.append(time.perf_counter() - t0)
            plan = spark.sql(q)._jdf.queryExecution().executedPlan().toString()
            pruned = 'PartitionFilters: [' in plan and 'PartitionFilters: []' not in plan
            rows.append({'query': qname, 'table': kind, 'partition_by': part or 'none', 'files': files[view],
                         'partition_pruning': pruned, 'median_ms': round(statistics.median(times) * 1e3, 1)})
            print(f'{qname:<34} {view:<16} files={files[view]:<4} pruning={pruned!s:<5} '
                  f'median={rows[-1]["median_ms"]:8.1f} ms', flush=True)

    out = REPO / 'benchmarks' / 'results' / 'partition_bench.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out} (fact rows {n_fact}, dim rows {n_dim})')


if __name__ == '__main__':
    main()
