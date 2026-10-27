# 数据字典

温度单位均为 °C，网格单位为格点（1 格点 ≈ 平面图上的一个像素）。所有表都按 `batch_id` 分区（Hive 风格目录 `batch_id=00007/`）。批次由 `building_ids.txt` 中的位置决定：`batch_id = index // batch_size`，因此同一批次总是包含同一组建筑。

## 原始层（landing，由 `simulation/export.py` 写出）

### landing/raw_grid —— 每栋建筑一行

| 字段 | 类型 | 含义 |
|---|---|---|
| building_id | string | Modified Swiss Dwellings 平面图 ID |
| grid | binary | 收敛后的温度场，512×512 float64，C 顺序，2,097,152 字节 |
| mask | binary | 室内掩码，`np.packbits(512×512 bool)`，32,768 字节 |
| batch_id | int | 分区列 |

### landing/convergence —— 每栋建筑每 100 次迭代一行

| 字段 | 类型 | 含义 |
|---|---|---|
| building_id | string | |
| iter | int | 已完成的迭代次数 |
| residual | double | 当前这一步的 max\|u_new − u_old\|（室内格点），°C |
| mean_temp | double | 当前室内平均温度，°C |
| elapsed_ms | double | 该建筑开始计算以来的耗时，ms |

### landing/run_meta —— 每栋建筑一行

| 字段 | 类型 | 含义 |
|---|---|---|
| building_id | string | |
| backend | string | numpy / numba / cuda |
| n_iter | int | 实际迭代次数（早停，atol = 1e-4；cuda 后端向上取整到 100 的倍数） |
| converged | boolean | 因满足 atol 停止，而非达到 max_iter |
| run_seconds | double | 仿真耗时，s |
| ref_area_cells, ref_mean_temp, ref_std_temp, ref_pct_above_18, ref_pct_below_15 | | 仿真脚本直接用参考 `summary_stats()` 算出的结果，只用于对账 |

## 清洗层（Hive 库 `dw`）

### dw.fact_block_temp —— 每栋建筑 × 含室内格点的 32×32 区块

| 字段 | 类型 | 含义 |
|---|---|---|
| building_id | string | |
| block_y, block_x | int | 区块行 / 列，0–15 |
| interior_cells | int | 区块内室内格点数，1–1024 |
| sum_temp | double | 室内温度之和（可加） |
| sum_sq_temp | double | 室内温度平方和（可加，用于还原标准差） |
| mean_temp, min_temp, max_temp | double | 区块内室内格点的统计量（分析用，不参与聚合） |
| n_above_18, n_below_15 | int | 高于 18 °C / 低于 15 °C 的格点数（可加） |

清洗规则：丢弃 `interior_cells = 0` 的区块（建筑外或纯墙体）；某栋建筑只要有一个格点超出物理范围 [0, 25] °C 或为 NaN，该建筑的所有区块都进入 `dw.quarantine_block_temp`，不进入汇总，避免产生半栋建筑的汇总结果。

### dw.fact_convergence

字段同 landing/convergence。清洗时去掉空值，以及 `iter ≤ 0`、`residual < 0`、`mean_temp ∉ [0, 25]` 的行，并按 `(building_id, iter)` 去重。

## 汇总层

### dw.dim_building_summary —— 每栋建筑一行，由 Spark 从 fact_block_temp 聚合

| 字段 | 类型 | 计算方式 |
|---|---|---|
| area_cells | bigint | Σ interior_cells |
| mean_temp | double | Σ sum_temp / area_cells |
| std_temp | double | sqrt(Σ sum_sq_temp / area_cells − mean²)，总体标准差 |
| pct_above_18, pct_below_15 | double | 100 · Σ n_above_18（或 n_below_15）/ area_cells |
| n_blocks | int | 含室内格点的区块数 |
| n_iter, converged, backend, run_seconds | | 来自 run_meta |
| temp_band | string | mean_temp < 15 → `lt15`；< 18 → `15_18`；否则 `ge18` |
| area_band | string | area_cells < 50,000 → `small`；< 90,000 → `medium`；否则 `large` |
| etl_ts | timestamp | 写入时间 |

分档阈值是固定值而不是分位数：如果用分位数，新批次到来后旧建筑的分档会变化，重跑结果就不再幂等。

## 运维表

| 表 | 粒度 | 用途 |
|---|---|---|
| dw.dq_results | 批次 × 阶段 × 检查项 | 每项数据质量检查的结果：PASS / WARN / FAIL、失败行数、详情 |
| dw.etl_batch_log | 批次 | 最近一次提交的 landing 指纹（文件名 + 大小 + mtime 的哈希），增量处理据此判断哪些批次是新的或有变化的 |
