"""Run the Jacobi simulation for one batch of buildings and land the results as Parquet.

This is the ETL-facing entry point of the simulation. It reuses the solvers from the
course project but adds early stopping, a convergence log and Parquet output:

    landing/raw_grid/batch_id=NNNNN/part-0.parquet      one row per building: final grid + mask
    landing/convergence/batch_id=NNNNN/part-0.parquet   one row per building per LOG_EVERY iterations
    landing/run_meta/batch_id=NNNNN/part-0.parquet      one row per building: run info + reference stats

Batches are defined by position in building_ids.txt (batch_id = index // batch_size), so the
same batch always contains the same buildings and re-running a batch overwrites it in place.

Usage:
    python simulation/export.py --data-dir data/sample --out-dir data/landing --batch-size 5 --backend numba
    python simulation/export.py --data-dir /dtu/.../modified_swiss_dwellings --batch-id 3 --backend cuda
"""
import argparse
import shutil
import time
from os.path import join
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SIZE = 512
MAX_ITER = 20_000
ABS_TOL = 1e-4
LOG_EVERY = 100


def load_data(load_dir, bid):
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def summary_stats(u, interior_mask):
    # Same definition as the reference simulate.py; used as the ground truth for reconciliation.
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'area_cells': int(u_interior.size),
        'mean_temp': u_interior.mean(),
        'std_temp': u_interior.std(),
        'pct_above_18': np.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': np.sum(u_interior < 15) / u_interior.size * 100,
    }


# ---------------------------------------------------------------------------
# Solver backends. Each one exposes run(n) -> (iters_done, last_delta, converged)
# which advances at most n Jacobi iterations, plus grid() to fetch the current state.
# ---------------------------------------------------------------------------

class NumpySolver:
    """The reference algorithm, checking the stopping criterion every iteration."""

    def __init__(self, u, interior_mask, atol):
        self.u = np.copy(u)
        self.mask = interior_mask
        self.atol = atol

    def run(self, n):
        u, mask = self.u, self.mask
        for k in range(1, n + 1):
            u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
            u_new_interior = u_new[mask]
            delta = np.abs(u[1:-1, 1:-1][mask] - u_new_interior).max()
            u[1:-1, 1:-1][mask] = u_new_interior
            if delta < self.atol:
                return k, delta, True
        return n, delta, False

    def grid(self):
        return self.u


class NumbaSolver:
    """Numba JIT on the CPU. Row-major sweep (inner loop over contiguous j) for cache locality."""

    def __init__(self, u, interior_mask, atol):
        global _numba_sweeps
        if _numba_sweeps is None:
            _numba_sweeps = _compile_numba()
        self.u_old = np.copy(u)
        self.u_new = np.copy(u)
        self.mask = interior_mask
        self.atol = atol

    def run(self, n):
        k, delta, swapped = _numba_sweeps(self.u_old, self.u_new, self.mask, n, self.atol)
        if swapped:
            self.u_old, self.u_new = self.u_new, self.u_old
        return k, delta, delta < self.atol

    def grid(self):
        return self.u_old


_numba_sweeps = None


def _compile_numba():
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def sweep(u_old, u_new, mask, row_delta):
        n, m = mask.shape
        for i in prange(n):
            d = 0.0
            for j in range(m):
                if mask[i, j]:
                    # Same summation order as the NumPy reference so results match bit for bit.
                    v = 0.25 * (u_old[i + 1, j] + u_old[i + 1, j + 2] + u_old[i, j + 1] + u_old[i + 2, j + 1])
                    diff = abs(v - u_old[i + 1, j + 1])
                    if diff > d:
                        d = diff
                    u_new[i + 1, j + 1] = v
            row_delta[i] = d

    @njit(cache=True)
    def sweeps(u_old, u_new, mask, n_iter, atol):
        row_delta = np.zeros(mask.shape[0])
        delta = np.inf
        swapped = False
        for k in range(1, n_iter + 1):
            sweep(u_old, u_new, mask, row_delta)
            u_old, u_new = u_new, u_old
            swapped = not swapped
            delta = row_delta.max()
            if delta < atol:
                return k, delta, swapped
        return n_iter, delta, swapped

    return sweeps


class CudaSolver:
    """Numba CUDA kernel from e8.py. The convergence check (a GPU reduction via CuPy)
    only runs at the end of each chunk, so n_iter is rounded up to a multiple of LOG_EVERY."""

    def __init__(self, u, interior_mask, atol):
        import math
        import cupy as cp
        from numba import cuda
        global _cuda_kernel
        if _cuda_kernel is None:
            _cuda_kernel = _compile_cuda()
        self.cp = cp
        self.d_old = cuda.to_device(u)
        self.d_new = cuda.to_device(u)
        self.d_mask = cuda.to_device(interior_mask)
        self.atol = atol
        self.tpb = (16, 16)
        self.bpg = (math.ceil(u.shape[0] / 16), math.ceil(u.shape[1] / 16))

    def run(self, n):
        cp = self.cp
        for _ in range(n):
            _cuda_kernel[self.bpg, self.tpb](self.d_old, self.d_new, self.d_mask)
            self.d_old, self.d_new = self.d_new, self.d_old
        # Zero-copy views of the Numba device arrays; d_new now holds the previous iterate.
        a, b = cp.asarray(self.d_old), cp.asarray(self.d_new)
        delta = float(cp.abs(a - b).max())
        return n, delta, delta < self.atol

    def grid(self):
        return self.d_old.copy_to_host()


_cuda_kernel = None


def _compile_cuda():
    from numba import cuda

    @cuda.jit
    def jacobi_kernel(u_old, u_new, interior_mask):
        i, j = cuda.grid(2)
        n, m = u_old.shape
        if 0 < i < n - 1 and 0 < j < m - 1 and interior_mask[i - 1, j - 1]:
            u_new[i, j] = 0.25 * (u_old[i, j - 1] + u_old[i, j + 1] + u_old[i - 1, j] + u_old[i + 1, j])

    return jacobi_kernel


BACKENDS = {'numpy': NumpySolver, 'numba': NumbaSolver, 'cuda': CudaSolver}


def simulate(u0, interior_mask, backend, max_iter=MAX_ITER, atol=ABS_TOL, log_every=LOG_EVERY):
    """Run one building. Returns the final grid, the iteration count and the convergence log."""
    solver = BACKENDS[backend](u0, interior_mask, atol)
    log = []
    it = 0
    converged = False
    t0 = time.perf_counter()
    while it < max_iter and not converged:
        k, delta, converged = solver.run(min(log_every, max_iter - it))
        it += k
        u = solver.grid()
        log.append((it, float(delta), float(u[1:-1, 1:-1][interior_mask].mean()),
                    (time.perf_counter() - t0) * 1e3))
    return solver.grid(), it, converged, log


def write_table(rows, schema, out_dir, table, batch_id):
    path = Path(out_dir) / table / f"batch_id={batch_id:05d}"
    if path.exists():
        shutil.rmtree(path)  # overwrite the batch: re-running is idempotent
    path.mkdir(parents=True)
    tbl = pa.Table.from_pylist(rows, schema=schema)
    # Raw grids are ~2 MB per row; small row groups keep Spark tasks and Arrow batches small.
    pq.write_table(tbl, path / "part-0.parquet", compression='zstd', row_group_size=16)
    return tbl.num_rows


RAW_SCHEMA = pa.schema([
    ('building_id', pa.string()),
    ('grid', pa.binary()),       # float64[512*512], C order, final temperature field
    ('mask', pa.binary()),       # np.packbits of the 512x512 interior mask
])
CONV_SCHEMA = pa.schema([
    ('building_id', pa.string()),
    ('iter', pa.int32()),
    ('residual', pa.float64()),
    ('mean_temp', pa.float64()),
    ('elapsed_ms', pa.float64()),
])
META_SCHEMA = pa.schema([
    ('building_id', pa.string()),
    ('backend', pa.string()),
    ('n_iter', pa.int32()),
    ('converged', pa.bool_()),
    ('run_seconds', pa.float64()),
    ('ref_area_cells', pa.int64()),
    ('ref_mean_temp', pa.float64()),
    ('ref_std_temp', pa.float64()),
    ('ref_pct_above_18', pa.float64()),
    ('ref_pct_below_15', pa.float64()),
])


def run_batch(data_dir, out_dir, building_ids, batch_id, backend, max_iter, atol):
    raw, conv, meta = [], [], []
    t_batch = time.perf_counter()
    for bid in building_ids:
        u0, mask = load_data(data_dir, bid)
        t0 = time.perf_counter()
        u, n_iter, converged, log = simulate(u0, mask, backend, max_iter, atol)
        run_seconds = time.perf_counter() - t0
        stats = summary_stats(u, mask)

        raw.append({'building_id': bid,
                    'grid': np.ascontiguousarray(u[1:-1, 1:-1], dtype=np.float64).tobytes(),
                    'mask': np.packbits(mask).tobytes()})
        conv.extend({'building_id': bid, 'iter': i, 'residual': r, 'mean_temp': m, 'elapsed_ms': e}
                    for i, r, m, e in log)
        meta.append({'building_id': bid, 'backend': backend, 'n_iter': n_iter, 'converged': converged,
                     'run_seconds': run_seconds, **{f'ref_{k}': v for k, v in stats.items()}})
        print(f"  [{batch_id:05d}] {bid}: {n_iter} iters ({'converged' if converged else 'max_iter'}), "
              f"{run_seconds:.2f} s, mean={stats['mean_temp']:.3f}", flush=True)

    counts = {
        'raw_grid': write_table(raw, RAW_SCHEMA, out_dir, 'raw_grid', batch_id),
        'convergence': write_table(conv, CONV_SCHEMA, out_dir, 'convergence', batch_id),
        'run_meta': write_table(meta, META_SCHEMA, out_dir, 'run_meta', batch_id),
    }
    print(f"batch {batch_id:05d}: {len(building_ids)} buildings in {time.perf_counter() - t_batch:.1f} s, rows={counts}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='/dtu/projects/02613_2025/data/modified_swiss_dwellings/')
    p.add_argument('--out-dir', default='data/landing')
    p.add_argument('--backend', choices=BACKENDS, default='numba')
    p.add_argument('--batch-size', type=int, default=200)
    p.add_argument('--batch-id', type=int, action='append',
                   help='batch(es) to run; default: all batches. Repeatable. '
                        'On LSF job arrays pass $((LSB_JOBINDEX-1)).')
    p.add_argument('--limit', type=int, help='only use the first N building ids')
    p.add_argument('--max-iter', type=int, default=MAX_ITER)
    p.add_argument('--atol', type=float, default=ABS_TOL)
    p.add_argument('--skip-existing', action='store_true', help='do not re-run batches already in --out-dir')
    args = p.parse_args()

    with open(join(args.data_dir, 'building_ids.txt')) as f:
        building_ids = f.read().splitlines()
    if args.limit:
        building_ids = building_ids[:args.limit]

    n_batches = (len(building_ids) + args.batch_size - 1) // args.batch_size
    for batch_id in (args.batch_id if args.batch_id is not None else range(n_batches)):
        ids = building_ids[batch_id * args.batch_size:(batch_id + 1) * args.batch_size]
        if not ids:
            raise SystemExit(f"batch {batch_id} is empty (only {n_batches} batches)")
        if args.skip_existing and (Path(args.out_dir) / 'run_meta' / f'batch_id={batch_id:05d}').exists():
            print(f"batch {batch_id:05d}: already landed, skipping")
            continue
        run_batch(args.data_dir, args.out_dir, ids, batch_id, args.backend, args.max_iter, args.atol)


if __name__ == '__main__':
    main()
