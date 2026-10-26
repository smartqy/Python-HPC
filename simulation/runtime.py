#!/usr/bin/env python3
import argparse
from os.path import join
import time
import numpy as np

# LOAD_DIR = '../modified_swiss_dwellings/' 
LOAD_DIR =  '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20000
ABS_TOL = 1e-4

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

def load_data_for_building(bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask

def process_building(bid):
    u0, interior_mask = load_data_for_building(bid)
    u_final = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u_final, interior_mask)

    return bid, stats

def main():
    parser = argparse.ArgumentParser(description="Parallel static scheduling simulation")
    parser.add_argument('--n', type=int, default=100, help="Number of floorplans to process")
    args = parser.parse_args()

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    building_ids = building_ids[:args.n]
    print(f"Processing {len(building_ids)} floorplans")

    start_time = time.time()
    results = []
    for bid in building_ids:
        results.append(process_building(bid))
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Time of processing {args.n} floors: {total_time:.2f} seconds." )

    # Output the results
    print("building_id, mean_temp, std_temp, pct_above_18, pct_below_15")
    for bid, stats in results:
        print(f"{bid}, {stats['mean_temp']:.2f}, {stats['std_temp']:.2f}, {stats['pct_above_18']:.2f}, {stats['pct_below_15']:.2f}")

if __name__ == '__main__':
    main()