-- Warehouse tables (Hive external tables over Parquet).
-- ${LAKE} is substituted with the lake directory by `python -m etl.run init`.
-- Every table is partitioned by batch_id: the ETL writes with INSERT OVERWRITE ... PARTITION (batch_id),
-- so re-processing a batch replaces exactly that batch and never duplicates rows.

CREATE DATABASE IF NOT EXISTS dw LOCATION '${LAKE}/dw';

-- Clean layer: one row per building x 32x32 block that contains interior cells.
CREATE EXTERNAL TABLE IF NOT EXISTS dw.fact_block_temp (
    building_id    STRING  COMMENT 'Modified Swiss Dwellings floor plan id',
    block_y        INT     COMMENT 'block row, 0..15 (grid rows 32*block_y .. +31)',
    block_x        INT     COMMENT 'block column, 0..15',
    interior_cells INT     COMMENT 'number of interior (room) cells in the block, 1..1024',
    sum_temp       DOUBLE  COMMENT 'sum of interior temperatures, degC',
    sum_sq_temp    DOUBLE  COMMENT 'sum of squared interior temperatures, degC^2',
    mean_temp      DOUBLE  COMMENT 'mean interior temperature of the block, degC',
    min_temp       DOUBLE  COMMENT 'min interior temperature of the block, degC',
    max_temp       DOUBLE  COMMENT 'max interior temperature of the block, degC',
    n_above_18     INT     COMMENT 'interior cells > 18 degC',
    n_below_15     INT     COMMENT 'interior cells < 15 degC'
)
PARTITIONED BY (batch_id INT)
STORED AS PARQUET
LOCATION '${LAKE}/dw/fact_block_temp';

-- Blocks that failed physical-range / NaN checks, kept for inspection instead of silently dropped.
CREATE EXTERNAL TABLE IF NOT EXISTS dw.quarantine_block_temp (
    building_id    STRING,
    block_y        INT,
    block_x        INT,
    interior_cells INT,
    min_temp       DOUBLE,
    max_temp       DOUBLE,
    n_invalid      INT     COMMENT 'interior cells outside [5, 25] degC or NaN',
    reason         STRING
)
PARTITIONED BY (batch_id INT)
STORED AS PARQUET
LOCATION '${LAKE}/dw/quarantine_block_temp';

-- Clean layer: convergence log, one row per building per 100 Jacobi iterations.
CREATE EXTERNAL TABLE IF NOT EXISTS dw.fact_convergence (
    building_id STRING,
    iter        INT     COMMENT 'iterations completed',
    residual    DOUBLE  COMMENT 'max |u_new - u_old| over interior cells at this iteration, degC',
    mean_temp   DOUBLE  COMMENT 'mean interior temperature at this iteration, degC',
    elapsed_ms  DOUBLE  COMMENT 'wall time since the building started, ms'
)
PARTITIONED BY (batch_id INT)
STORED AS PARQUET
LOCATION '${LAKE}/dw/fact_convergence';

-- Summary layer: aggregated by Spark from fact_block_temp, joined with run metadata.
CREATE EXTERNAL TABLE IF NOT EXISTS dw.dim_building_summary (
    building_id  STRING,
    area_cells   BIGINT  COMMENT 'interior cells (room area in grid cells)',
    mean_temp    DOUBLE  COMMENT 'degC',
    std_temp     DOUBLE  COMMENT 'population std, degC',
    pct_above_18 DOUBLE  COMMENT '% of area above 18 degC',
    pct_below_15 DOUBLE  COMMENT '% of area below 15 degC',
    n_blocks     INT     COMMENT 'blocks with interior cells',
    n_iter       INT     COMMENT 'Jacobi iterations run',
    converged    BOOLEAN COMMENT 'stopped by atol rather than max_iter',
    backend      STRING  COMMENT 'numpy | numba | cuda',
    run_seconds  DOUBLE  COMMENT 'simulation wall time',
    temp_band    STRING  COMMENT 'lt15 | 15_18 | ge18, from mean_temp',
    area_band    STRING  COMMENT 'small | medium | large, fixed thresholds on area_cells',
    etl_ts       TIMESTAMP
)
PARTITIONED BY (batch_id INT)
STORED AS PARQUET
LOCATION '${LAKE}/dw/dim_building_summary';

-- Data-quality results of every check, per batch and pipeline stage.
CREATE EXTERNAL TABLE IF NOT EXISTS dw.dq_results (
    check_name  STRING,
    table_name  STRING,
    failed_rows BIGINT,
    total_rows  BIGINT,
    status      STRING  COMMENT 'PASS | WARN | FAIL',
    detail      STRING,
    run_ts      TIMESTAMP
)
PARTITIONED BY (batch_id INT, stage STRING)
STORED AS PARQUET
LOCATION '${LAKE}/dw/dq_results';

-- Incremental-processing state: which landing version of each batch was last committed.
CREATE EXTERNAL TABLE IF NOT EXISTS dw.etl_batch_log (
    landing_fingerprint STRING  COMMENT 'hash of landing file names, sizes and mtimes',
    n_buildings         BIGINT,
    committed_ts        TIMESTAMP
)
PARTITIONED BY (batch_id INT)
STORED AS PARQUET
LOCATION '${LAKE}/dw/etl_batch_log';
