-- Typical analysis queries. {fact} and {dim} are replaced with a concrete table:
--   dw.fact_block_temp / dw.dim_building_summary in the warehouse, or the layout variants in
--   benchmarks/partition_bench.py. fact tables there also carry temp_band / area_band (denormalised).

-- name: q1_area_by_temp_band
-- Average room area and temperature per temperature band.
SELECT temp_band, COUNT(*) AS buildings, AVG(area_cells) AS avg_area_cells, AVG(mean_temp) AS avg_mean_temp
FROM {dim}
GROUP BY temp_band
ORDER BY temp_band;

-- name: q2_slowest_convergence_top50
-- Buildings that needed the most Jacobi iterations.
SELECT building_id, n_iter, run_seconds, area_cells, mean_temp
FROM {dim}
ORDER BY n_iter DESC, building_id
LIMIT 50;

-- name: q3_cold_blocks_in_warm_buildings
-- Cold spots (< 15 degC block mean) inside buildings whose overall mean is >= 18 degC.
SELECT COUNT(*) AS cold_blocks, COUNT(DISTINCT building_id) AS buildings, AVG(mean_temp) AS avg_block_temp
FROM {fact}
WHERE temp_band = 'ge18' AND mean_temp < 15;

-- name: q4_large_building_heatmap
-- Mean block temperature by block position for large buildings.
SELECT block_y, block_x, SUM(sum_temp) / SUM(interior_cells) AS mean_temp, SUM(interior_cells) AS cells
FROM {fact}
WHERE area_band = 'large'
GROUP BY block_y, block_x
ORDER BY block_y, block_x;

-- name: q5_single_batch_cold_share
-- Share of area below 15 degC for one ingest batch (what an incremental report would query).
SELECT SUM(n_below_15) * 100.0 / SUM(interior_cells) AS pct_below_15, COUNT(DISTINCT building_id) AS buildings
FROM {fact}
WHERE batch_id = 7;
