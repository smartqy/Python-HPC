"""Single-process Pandas/NumPy version of raw_grid -> blocks -> clean -> summary, for benchmarking.

Same logic as etl/transforms.py (and the same per-batch block kernel, _blocks_of_batch). Raw files are streamed one Arrow record batch at a time so
memory stays bounded; the blocks of all buildings are concatenated into one DataFrame.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from etl.blockstats import GRID_BYTES, MASK_BYTES
from etl.transforms import AREA_BANDS, TEMP_BANDS, _blocks_of_batch


def _band(values, bands):
    edges = [u for u, _ in bands[:-1]]
    return np.array([n for _, n in bands])[np.searchsorted(edges, values, side='right')]


def run(landing: Path, out: Path):
    frames = []
    for f in sorted((landing / 'raw_grid').glob('batch_id=*/*.parquet')):
        batch_id = int(f.parent.name.split('=')[1])
        for rb in pq.ParquetFile(f).iter_batches(batch_size=8, columns=['building_id', 'grid', 'mask']):
            pdf = rb.to_pandas()
            ok = pdf['grid'].map(lambda g: g is not None and len(g) == GRID_BYTES) & \
                pdf['mask'].map(lambda m: m is not None and len(m) == MASK_BYTES)
            frames.append(_blocks_of_batch(pdf[ok].assign(batch_id=batch_id)))
    blocks = pd.concat(frames, ignore_index=True)

    # Clean: drop empty blocks, drop every block of a building with an invalid value.
    bad = blocks.loc[blocks.n_invalid > 0, 'building_id'].unique()
    clean = blocks[(blocks.interior_cells > 0) & ~blocks.building_id.isin(bad)].drop(columns='n_invalid')

    g = clean.groupby(['building_id', 'batch_id'])
    agg = g.agg(area_cells=('interior_cells', 'sum'), s=('sum_temp', 'sum'), ss=('sum_sq_temp', 'sum'),
                n18=('n_above_18', 'sum'), n15=('n_below_15', 'sum'), n_blocks=('block_x', 'size')).reset_index()
    mean = agg.s / agg.area_cells
    summary = pd.DataFrame({
        'building_id': agg.building_id, 'batch_id': agg.batch_id, 'area_cells': agg.area_cells,
        'mean_temp': mean,
        'std_temp': np.sqrt(np.maximum(agg.ss / agg.area_cells - mean ** 2, 0.0)),
        'pct_above_18': agg.n18 * 100.0 / agg.area_cells,
        'pct_below_15': agg.n15 * 100.0 / agg.area_cells,
        'n_blocks': agg.n_blocks,
    })
    meta = pd.read_parquet(landing / 'run_meta', columns=['building_id', 'batch_id', 'n_iter', 'converged',
                                                           'backend', 'run_seconds'])
    meta['batch_id'] = meta.batch_id.astype(int)
    summary = summary.merge(meta, on=['building_id', 'batch_id'], how='left')
    summary['temp_band'] = _band(summary.mean_temp, TEMP_BANDS)
    summary['area_band'] = _band(summary.area_cells, AREA_BANDS)

    out.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out / 'fact_block_temp.parquet', index=False)
    summary.to_parquet(out / 'dim_building_summary.parquet', index=False)
    return len(clean), len(summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--landing', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    t0 = time.perf_counter()
    n_blocks, n_buildings = run(a.landing, a.out)
    print(f'RESULT engine=pandas buildings={n_buildings} blocks={n_blocks} job_seconds={time.perf_counter() - t0:.3f}')


if __name__ == '__main__':
    main()
