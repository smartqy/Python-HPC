from os.path import join
import sys
import time
import numpy as np
import cupy as cp

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@cp.fuse()
def fused_jacobi_step(top, bottom, left, right):
    return 0.25 * (top + bottom + left + right)

def jacobi(u_np, interior_mask_np, max_iter, atol=1e-6):
    u = cp.array(u_np)
    interior_mask = cp.array(interior_mask_np)

    for _ in range(max_iter):
        u_new = fused_jacobi_step(
            u[:-2, 1:-1], u[2:, 1:-1], u[1:-1, :-2], u[1:-1, 2:]
        )
        u_center = u[1:-1, 1:-1]
        u_new = u_center * (1 - interior_mask) + u_new * interior_mask
        delta = cp.abs(u_center - u_new)[interior_mask].max()
        u[1:-1, 1:-1] = u_new
        if delta < atol:
            break

    return cp.asnumpy(u)

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    start_time = time.time()
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20000
    ABS_TOL = 1e-4
    all_u = np.empty_like(all_u0)

    print(f"Running Jacobi (fused) on {N} buildings...")
    total_start = time.perf_counter()

    for i, (bid, u0, interior_mask) in enumerate(zip(building_ids, all_u0, all_interior_mask), 1):
        start = time.perf_counter()
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        cp.cuda.Device(0).synchronize()
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        all_u[i - 1] = u
        print(f"[{i}/{N}] {bid} finished in {elapsed_ms:.2f} ms")

    total_end = time.perf_counter()
    print(f"\nTotal GPU time: {total_end - total_start:.2f} seconds")

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('\nbuilding_id, ' + ', '.join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
    elapsed = time.time() - start_time
    print(f"\n✅  {len(building_ids)}  floorplans，time {elapsed:.2f} second")