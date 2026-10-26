from os.path import join
import sys
import math
import numpy as np
from numba import cuda
import time


def load_data(load_dir, bid):
    """Load the domain and interior mask for a given building ID."""
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask



@cuda.jit
def jacobi_kernel(u_old, u_new, interior_mask):
    i, j = cuda.grid(2)
    n, m = u_old.shape
    if i < n and j < m:
        if i == 0 or j == 0 or i == n - 1 or j == m - 1:
            u_new[i, j] = u_old[i, j]
        else:
            mask_i = i - 1
            mask_j = j - 1
            if interior_mask[mask_i, mask_j]:
                u_new[i, j] = 0.25 * (u_old[i, j - 1] + u_old[i, j + 1] +
                                      u_old[i - 1, j] + u_old[i + 1, j])
            else:
                u_new[i, j] = u_old[i, j]


def jacobi_cuda(u, interior_mask, max_iter):

    d_u_old = cuda.to_device(u)
    d_u_new = cuda.device_array_like(d_u_old)
    d_mask = cuda.to_device(interior_mask)

    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(d_u_old.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(d_u_old.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](d_u_old, d_u_new, d_mask)
        d_u_old, d_u_new = d_u_new, d_u_old

    cuda.synchronize()
    result_u = d_u_old.copy_to_host()
    return result_u



def summary_stats(u, interior_mask):
    interior_values = u[1:-1, 1:-1][interior_mask]
    mean_temp = interior_values.mean()
    std_temp = interior_values.std()
    pct_above_18 = np.sum(interior_values > 18.0) / interior_values.size * 100.0
    pct_below_15 = np.sum(interior_values < 15.0) / interior_values.size * 100.0
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

    N_buildings = len(building_ids)
    all_u0 = np.empty((N_buildings, 514, 514), dtype=np.float64)
    all_mask = np.empty((N_buildings, 512, 512), dtype=bool)
    for idx, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[idx] = u0
        all_mask[idx] = interior_mask

    MAX_ITER = 20000

    all_u_final = np.empty_like(all_u0)
    for idx, (u0, interior_mask) in enumerate(zip(all_u0, all_mask)):
        u_final = jacobi_cuda(u0, interior_mask, MAX_ITER)
        all_u_final[idx] = u_final

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, u_final, interior_mask in zip(building_ids, all_u_final, all_mask):
        stats = summary_stats(u_final, interior_mask)
        print(f"{bid}, " + ", ".join(str(stats[k]) for k in stat_keys))
    elapsed = time.time() - start_time
    print(f"\n✅  {len(building_ids)}  floorplans，time {elapsed:.2f} second")

