from os.path import join
import sys
import time  
import numpy as np
import visualize as v

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


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
    # Load data
    LOAD_DIR = 'file'
    # with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
    #     building_ids = f.read().splitlines()

    # if len(sys.argv) < 2:
    #     N = 1
    # else:
    #     N = int(sys.argv[1])
    # building_ids = building_ids[:N]

    # Load floor plans
    building_ids = ['694','986','1878','1869','3171','4485','5158','5971','6233','6452','7025','7987','8264','9648','9991','10000','10535','29753','49314','50814']
    N = len(building_ids)
    all_u0 = np.empty((N, 514, 514))

    all_interior_mask = np.empty((N, 512, 512), dtype='bool')

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_00
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
  
    # # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15', 'runtime_s']
    print('building_id, ' + ', '.join(stat_keys))

    # --- Process each building ---
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

        start_time = time.time()
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        runtime_s = time.time() - start_time
        all_u[i] = u

        # v.after_plot_building(u, interior_mask)

        stats = summary_stats(u, interior_mask)
        stats['runtime_s'] = round(runtime_s, 4)

        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
    
v.after_plot_all_buildings(all_u, all_interior_mask, building_ids)

