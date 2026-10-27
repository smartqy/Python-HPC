"""Build larger landing datasets by replicating the simulated sample buildings.

Only 20 floor plans are simulated locally, which is far too little to say anything about Spark vs
Pandas. For the scaling benchmark the landing data is copied K times with building ids
'<id>_r<k>' (synthetic replicas, NOT new simulations), keeping the per-building payload realistic.

    python benchmarks/make_scaled.py --factor 10 50 230
"""
import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent / 'data'


def replicate(src: Path, dst: Path, factor: int):
    if dst.exists():
        shutil.rmtree(dst)
    for table in ('raw_grid', 'run_meta'):
        batches = sorted((src / table).glob('batch_id=*'))
        n_batches = len(batches)
        for k in range(factor):
            for bdir in batches:
                b = int(bdir.name.split('=')[1])
                t = pq.ParquetFile(bdir / 'part-0.parquet').read()  # read_table would add batch_id from the path
                ids = pc.binary_join_element_wise(t.column('building_id'), pa.scalar(f'_r{k}'), '')
                t = t.set_column(t.schema.get_field_index('building_id'), 'building_id', ids)
                out = dst / table / f'batch_id={k * n_batches + b:05d}'
                out.mkdir(parents=True)
                pq.write_table(t, out / 'part-0.parquet', compression='zstd', row_group_size=16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--factor', type=int, nargs='+', required=True)
    p.add_argument('--src', type=Path, default=ROOT / 'landing')
    a = p.parse_args()
    for f in a.factor:
        dst = ROOT / 'scaled' / f'x{f}' / 'landing'
        replicate(a.src, dst, f)
        size = sum(x.stat().st_size for x in dst.rglob('*.parquet'))
        print(f'x{f}: {dst} ({size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
