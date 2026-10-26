"""Run the Spark and Pandas versions of the block/summary ETL on each dataset size, N times each,
in fresh processes, and write the medians to benchmarks/results/spark_vs_pandas.csv.

    python benchmarks/make_scaled.py --factor 10 50 230
    python benchmarks/spark_vs_pandas.py --datasets data/landing data/scaled/x10/landing ... --runs 3
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / 'benchmarks' / 'results'
CMDS = {
    'pandas': [sys.executable, '-m', 'etl.pandas_baseline'],
    'spark_rows': [sys.executable, 'benchmarks/spark_job.py', '--mode', 'rows'],
    'spark_files': [sys.executable, 'benchmarks/spark_job.py', '--mode', 'files'],
}


def run_once(engine, landing, out):
    t0 = time.perf_counter()
    res = subprocess.run(CMDS[engine] + ['--landing', str(landing), '--out', str(out)], cwd=REPO,
                         capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': str(REPO)})
    wall = time.perf_counter() - t0
    line = next((l for l in res.stdout.splitlines() if l.startswith('RESULT')), None)
    if res.returncode or not line:
        raise RuntimeError(f'{engine} failed on {landing}:\n{res.stderr[-3000:]}')
    kv = dict(re.findall(r'(\w+)=([\w.]+)', line))
    return wall, kv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', type=Path, nargs='+', required=True)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--engines', nargs='+', default=list(CMDS))
    a = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    out = RESULTS / 'spark_vs_pandas.csv'
    for landing in a.datasets:
        landing = landing.resolve()
        raw_mb = sum(f.stat().st_size for f in (landing / 'raw_grid').rglob('*.parquet')) / 1e6
        for engine in a.engines:
            walls, jobs = [], []
            for i in range(a.runs):
                wall, kv = run_once(engine, landing, REPO / 'data' / 'scaled' / '_bench_out' / engine)
                walls.append(wall)
                jobs.append(float(kv['job_seconds']))
                print(f'{landing.parent.name:>8} {engine:>11} run {i + 1}: wall {wall:6.1f}s job {jobs[-1]:6.1f}s '
                      f'buildings={kv["buildings"]} blocks={kv["blocks"]}', flush=True)
            rows.append({'dataset': landing.relative_to(REPO).as_posix(), 'buildings': int(kv['buildings']),
                         'blocks': int(kv['blocks']), 'raw_parquet_mb': round(raw_mb, 1), 'engine': engine,
                         'runs': a.runs, 'median_job_s': round(statistics.median(jobs), 2),
                         'median_wall_s': round(statistics.median(walls), 2)})
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
