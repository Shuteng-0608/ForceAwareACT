# ForceAwareACT ±10 mm孔位扰动Rollout实验报告

## 1. 实验背景与目的

本报告对仓库本地 `outputs/peg_hole_100` 下 50 点 x/z ±10 mm 孔位扰动 rollout 记录进行证据化分析。目标是盘点实验完整性，判断哪些实验可以直接比较，并总结任务成功、安全成功、失败、力、完成时间、空间半径与方向分布。

本报告仅使用本地已有实验文件。没有重新运行 rollout，没有使用 GPU，也没有修改模型、训练、推理、rollout、成功判据、checkpoint、normalization 或 dataset 代码。

## 2. 报告范围

搜寻对象限定为目录名包含 `hole_lhs_50_xz_pm10mm` 或 `hole_lhs_50_xz_10mm` 的实验目录。有效实验根目录至少需要包含 `grid_summary.csv`。本报告主表只包含 50 点 ±10 mm x/z 孔位扰动实验，不混入其他扰动范围。

术语定义：

- `task success`：`grid_summary.csv` 中的 `success=True`。
- `safe success`：`grid_summary.csv` 中的 `safe_success=True`；当前 manifest 显示其阈值为 task success 且 `max_force < success_force_threshold=80 N`。
- `hard force stop`：配置中的 `force_stop_threshold=1000 N`，本批实验没有由该项形成的 process error 计数。
- `process error`：manifest run status 为 `process_error` 的运行。

注意：`safe_success` 是本实验配置内的阈值定义，不是通用硬件安全保证。

## 3. 实验记录搜寻与纳入标准

确认测量结果：共发现 6 个有效实验根目录，全部包含 `grid_summary.csv`、`random_position_summary.json`、`grid_manifest.json` 与 `task_points.csv`。所有目录都被纳入分析，没有发现匹配但无效的实验根目录。

机器可读清单写入：`outputs/peg_hole_100/xz_pm10mm_rollout_report/experiment_inventory.csv`。

## 4. 实验完整性检查

确认测量结果：6 个实验均满足标准 50 点实验条件：`requested_points=50`、`completed_points=50`、`process_error_points=0`、`completion_rate=100%`。`random_position_summary.json` 与 `grid_summary.csv` 的成功数、安全成功数和完成数一致，未发现计数差异。

| experiment_name | completed_points | successful_points | failed_task_points | process_error_points | completion_rate | classification |
| --- | --- | --- | --- | --- | --- | --- |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | 50 | 11 | 39 | 0 | 100.0% | complete |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | 50 | 1 | 49 | 0 | 100.0% | complete |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | 50 | 6 | 44 | 0 | 100.0% | complete |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | 50 | 2 | 48 | 0 | 100.0% | complete |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | 50 | 7 | 43 | 0 | 100.0% | complete |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | 50 | 2 | 48 | 0 | 100.0% | complete |

## 5. 实验协议与可比性

所有纳入实验共享相同采样协议：`sampling_mode=latin_hypercube`、`base_seed=20260702`、`x_min=-0.01`、`x_max=0.01`、`z_min=-0.01`、`z_max=0.01`、`y_offset=0`。成功阈值也一致：`success_distance_threshold=0.005`、`success_lateral_threshold=0.006`、`success_force_threshold=80.0`、`success_hold_steps=15`。执行约束一致：`max_delta_q=0.02`、`max_rollout_steps=900`、`force_stop_threshold=1000.0`、`chunk_len=10`、`force_window_len=20`、`force_window_duration=0.25`。

逐点比较要求 `point_index`、`hole_offset_x`、`hole_offset_y`、`hole_offset_z` 完全一致。派生表显示所有两两实验均满足 point-paired 条件。解释时仍需区分：同一 checkpoint 的 mid-vs-temporal 对比主要改变 `action_select_mode`；跨 checkpoint 比较同时改变模型/策略身份，因此属于描述性 paired comparison，不能作为单因素因果结论。

## 6. 纳入实验清单

| experiment_name | policy_variant | action_select_mode | successful_points | failed_task_points | safe_successful_points | task_success_rate | safe_success_rate | success_ci95_lower | success_ci95_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | force_aware_contact_cvae | mid | 11 | 39 | 11 | 22.0% | 22.0% | 12.8% | 35.2% |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | force_aware_contact_cvae | temporal | 1 | 49 | 1 | 2.0% | 2.0% | 0.4% | 10.5% |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | force_aware_motion_cvae | mid | 6 | 44 | 6 | 12.0% | 12.0% | 5.6% | 23.8% |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | force_aware_motion_cvae | temporal | 2 | 48 | 2 | 4.0% | 4.0% | 1.1% | 13.5% |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | force_aware_act_dualzero | mid | 7 | 43 | 7 | 14.0% | 14.0% | 7.0% | 26.2% |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | force_aware_act_dualzero | temporal | 2 | 48 | 2 | 4.0% | 4.0% | 1.1% | 13.5% |

## 7. 总体任务成功与安全成功结果

确认测量结果：最高任务成功率来自 `hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002`，为 11/50，即 22.0%。其次是 dualzero mid，7/50，即 14.0%；Motion-CVAE mid 为 6/50，即 12.0%。Temporal 执行模式下成功数较低：Contact-CVAE temporal 为 1/50，Motion-CVAE temporal 与 dualzero temporal 均为 2/50。

统计结果：Wilson 95% 置信区间采用 `random_position_summary.json` 中与重新计算结果一致的区间。所有实验中 `safe_successful_points` 与 `successful_points` 完全相同，说明当前成功样本均满足 `max_force < 80 N` 的 safe-success 阈值。

![Task success rate](../../outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/task_success_rate_by_experiment.png)

![Safe success rate](../../outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/safe_success_rate_by_experiment.png)

## 8. 各实验的靶心空间分布

所有 target-map 图均显示实测 rollout 点：成功为绿色实心圆，失败为红色实心圆，黑色边缘，名义孔中心在 `(0, 0)`。这些图不插值、不估计连续成功区域。用于比较的图均采用相同尺度：`ring_step_mm=2`、`max_radius_mm=14`。

- `hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/plots/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002_target.png)
- `hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002/plots/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002_target.png)
- `hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900/plots/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900_target.png)
- `hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/plots/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900_target.png)
- `hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900/plots/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900_target.png)
- `hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900/plots/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900_target.png)

## 9. 成功率随孔位半径的变化

半径按 `radius_mm = sqrt(x_mm^2 + z_mm^2)` 计算，并使用固定 bin：`[0,2)`、`[2,4)`、`[4,6)`、`[6,8)`、`[8,10)`、`[10,12)`、`[12,15)`。

结果表明：成功样本主要集中在较小半径。较大半径也有少量成功点，但由于 LHS 样本稀疏，最大成功半径只能称为“采样集合中的最大观测成功半径”，不能称为绝对泛化边界。

| experiment_name | radius_bin_mm | points | successes | failures | task_success_rate | safe_successes | safe_success_rate | mean_radius_mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | [0, 2) | 2 | 2 | 0 | 100.0% | 2 | 100.0% | 1.31 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | [2, 4) | 6 | 4 | 2 | 66.7% | 4 | 66.7% | 3.13 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | [4, 6) | 8 | 4 | 4 | 50.0% | 4 | 50.0% | 5.13 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | [6, 8) | 9 | 1 | 8 | 11.1% | 1 | 11.1% | 7.08 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | [8, 10) | 15 | 0 | 15 | 0.0% | 0 | 0.0% | 9.25 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | [10, 12) | 8 | 0 | 8 | 0.0% | 0 | 0.0% | 11.13 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | [12, 15) | 2 | 0 | 2 | 0.0% | 0 | 0.0% | 12.78 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | [0, 2) | 2 | 0 | 2 | 0.0% | 0 | 0.0% | 1.31 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | [2, 4) | 6 | 1 | 5 | 16.7% | 1 | 16.7% | 3.13 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | [4, 6) | 8 | 0 | 8 | 0.0% | 0 | 0.0% | 5.13 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | [6, 8) | 9 | 0 | 9 | 0.0% | 0 | 0.0% | 7.08 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | [8, 10) | 15 | 0 | 15 | 0.0% | 0 | 0.0% | 9.25 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | [10, 12) | 8 | 0 | 8 | 0.0% | 0 | 0.0% | 11.13 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | [12, 15) | 2 | 0 | 2 | 0.0% | 0 | 0.0% | 12.78 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | [0, 2) | 2 | 2 | 0 | 100.0% | 2 | 100.0% | 1.31 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | [2, 4) | 6 | 3 | 3 | 50.0% | 3 | 50.0% | 3.13 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | [4, 6) | 8 | 1 | 7 | 12.5% | 1 | 12.5% | 5.13 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | [6, 8) | 9 | 0 | 9 | 0.0% | 0 | 0.0% | 7.08 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | [8, 10) | 15 | 0 | 15 | 0.0% | 0 | 0.0% | 9.25 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | [10, 12) | 8 | 0 | 8 | 0.0% | 0 | 0.0% | 11.13 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | [12, 15) | 2 | 0 | 2 | 0.0% | 0 | 0.0% | 12.78 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | [0, 2) | 2 | 1 | 1 | 50.0% | 1 | 50.0% | 1.31 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | [2, 4) | 6 | 1 | 5 | 16.7% | 1 | 16.7% | 3.13 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | [4, 6) | 8 | 0 | 8 | 0.0% | 0 | 0.0% | 5.13 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | [6, 8) | 9 | 0 | 9 | 0.0% | 0 | 0.0% | 7.08 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | [8, 10) | 15 | 0 | 15 | 0.0% | 0 | 0.0% | 9.25 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | [10, 12) | 8 | 0 | 8 | 0.0% | 0 | 0.0% | 11.13 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | [12, 15) | 2 | 0 | 2 | 0.0% | 0 | 0.0% | 12.78 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | [0, 2) | 2 | 2 | 0 | 100.0% | 2 | 100.0% | 1.31 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | [2, 4) | 6 | 3 | 3 | 50.0% | 3 | 50.0% | 3.13 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | [4, 6) | 8 | 2 | 6 | 25.0% | 2 | 25.0% | 5.13 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | [6, 8) | 9 | 0 | 9 | 0.0% | 0 | 0.0% | 7.08 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | [8, 10) | 15 | 0 | 15 | 0.0% | 0 | 0.0% | 9.25 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | [10, 12) | 8 | 0 | 8 | 0.0% | 0 | 0.0% | 11.13 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | [12, 15) | 2 | 0 | 2 | 0.0% | 0 | 0.0% | 12.78 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | [0, 2) | 2 | 2 | 0 | 100.0% | 2 | 100.0% | 1.31 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | [2, 4) | 6 | 0 | 6 | 0.0% | 0 | 0.0% | 3.13 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | [4, 6) | 8 | 0 | 8 | 0.0% | 0 | 0.0% | 5.13 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | [6, 8) | 9 | 0 | 9 | 0.0% | 0 | 0.0% | 7.08 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | [8, 10) | 15 | 0 | 15 | 0.0% | 0 | 0.0% | 9.25 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | [10, 12) | 8 | 0 | 8 | 0.0% | 0 | 0.0% | 11.13 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | [12, 15) | 2 | 0 | 2 | 0.0% | 0 | 0.0% | 12.78 |

关键半径点：

| experiment_name | radius_bin_mm | point_index | x_mm | z_mm | radius_mm | success | safe_success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | largest_radius_success | 18 | -5.889 | -1.715 | 6.134 | True | True |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | smallest_radius_failure | 23 | 0.379 | 3.003 | 3.027 | False | False |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | largest_radius_safe_success | 18 | -5.889 | -1.715 | 6.134 | True | True |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | largest_radius_success | 3 | -3.484 | -0.580 | 3.532 | True | True |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | smallest_radius_failure | 9 | 0.935 | 0.031 | 0.935 | False | False |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | largest_radius_safe_success | 3 | -3.484 | -0.580 | 3.532 | True | True |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | largest_radius_success | 41 | -2.307 | -3.814 | 4.458 | True | True |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | smallest_radius_failure | 23 | 0.379 | 3.003 | 3.027 | False | False |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | largest_radius_safe_success | 41 | -2.307 | -3.814 | 4.458 | True | True |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | largest_radius_success | 26 | 1.986 | -1.264 | 2.354 | True | True |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | smallest_radius_failure | 29 | 0.602 | 1.578 | 1.688 | False | False |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | largest_radius_safe_success | 26 | 1.986 | -1.264 | 2.354 | True | True |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | largest_radius_success | 40 | -5.069 | -0.935 | 5.154 | True | True |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | smallest_radius_failure | 5 | 1.369 | 2.575 | 2.917 | False | False |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | largest_radius_safe_success | 40 | -5.069 | -0.935 | 5.154 | True | True |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | largest_radius_success | 29 | 0.602 | 1.578 | 1.688 | True | True |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | smallest_radius_failure | 26 | 1.986 | -1.264 | 2.354 | False | False |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | largest_radius_safe_success | 29 | 0.602 | 1.578 | 1.688 | True | True |

![Radius binned success rate](../../outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/radius_binned_success_rate.png)

![Largest observed successful radius](../../outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/largest_success_radius_by_experiment.png)

## 10. x/z方向与象限分析

方向与象限分析基于实际计数。由于 LHS 不是专门为方向假设检验构造的平衡实验，以下结果应视为探索性统计，不应仅凭视觉分布断言方向性机制。

| experiment_name | group | points | successes | success_rate | safe_successes | safe_success_rate | mean_radius_mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | x < 0 | 25 | 6 | 24.0% | 6 | 24.0% | 7.60 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | x >= 0 | 25 | 5 | 20.0% | 5 | 20.0% | 7.58 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | z < 0 | 25 | 5 | 20.0% | 5 | 20.0% | 8.01 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | z >= 0 | 25 | 6 | 24.0% | 6 | 24.0% | 7.17 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | |x| >= |z| | 26 | 9 | 34.6% | 9 | 34.6% | 7.55 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | |z| > |x| | 24 | 2 | 8.3% | 2 | 8.3% | 7.64 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | +x +z | 12 | 4 | 33.3% | 4 | 33.3% | 6.66 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | -x +z | 13 | 2 | 15.4% | 2 | 15.4% | 7.64 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | -x -z | 12 | 4 | 33.3% | 4 | 33.3% | 7.55 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | +x -z | 13 | 1 | 7.7% | 1 | 7.7% | 8.44 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | x < 0 | 25 | 1 | 4.0% | 1 | 4.0% | 7.60 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | x >= 0 | 25 | 0 | 0.0% | 0 | 0.0% | 7.58 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | z < 0 | 25 | 1 | 4.0% | 1 | 4.0% | 8.01 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | z >= 0 | 25 | 0 | 0.0% | 0 | 0.0% | 7.17 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | |x| >= |z| | 26 | 1 | 3.8% | 1 | 3.8% | 7.55 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | |z| > |x| | 24 | 0 | 0.0% | 0 | 0.0% | 7.64 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | +x +z | 12 | 0 | 0.0% | 0 | 0.0% | 6.66 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | -x +z | 13 | 0 | 0.0% | 0 | 0.0% | 7.64 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | -x -z | 12 | 1 | 8.3% | 1 | 8.3% | 7.55 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | +x -z | 13 | 0 | 0.0% | 0 | 0.0% | 8.44 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | x < 0 | 25 | 2 | 8.0% | 2 | 8.0% | 7.60 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | x >= 0 | 25 | 4 | 16.0% | 4 | 16.0% | 7.58 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | z < 0 | 25 | 3 | 12.0% | 3 | 12.0% | 8.01 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | z >= 0 | 25 | 3 | 12.0% | 3 | 12.0% | 7.17 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | |x| >= |z| | 26 | 3 | 11.5% | 3 | 11.5% | 7.55 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | |z| > |x| | 24 | 3 | 12.5% | 3 | 12.5% | 7.64 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | +x +z | 12 | 3 | 25.0% | 3 | 25.0% | 6.66 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | -x +z | 13 | 0 | 0.0% | 0 | 0.0% | 7.64 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | -x -z | 12 | 2 | 16.7% | 2 | 16.7% | 7.55 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | +x -z | 13 | 1 | 7.7% | 1 | 7.7% | 8.44 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | x < 0 | 25 | 0 | 0.0% | 0 | 0.0% | 7.60 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | x >= 0 | 25 | 2 | 8.0% | 2 | 8.0% | 7.58 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | z < 0 | 25 | 1 | 4.0% | 1 | 4.0% | 8.01 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | z >= 0 | 25 | 1 | 4.0% | 1 | 4.0% | 7.17 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | |x| >= |z| | 26 | 2 | 7.7% | 2 | 7.7% | 7.55 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | |z| > |x| | 24 | 0 | 0.0% | 0 | 0.0% | 7.64 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | +x +z | 12 | 1 | 8.3% | 1 | 8.3% | 6.66 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | -x +z | 13 | 0 | 0.0% | 0 | 0.0% | 7.64 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | -x -z | 12 | 0 | 0.0% | 0 | 0.0% | 7.55 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | +x -z | 13 | 1 | 7.7% | 1 | 7.7% | 8.44 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | x < 0 | 25 | 4 | 16.0% | 4 | 16.0% | 7.60 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | x >= 0 | 25 | 3 | 12.0% | 3 | 12.0% | 7.58 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | z < 0 | 25 | 4 | 16.0% | 4 | 16.0% | 8.01 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | z >= 0 | 25 | 3 | 12.0% | 3 | 12.0% | 7.17 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | |x| >= |z| | 26 | 5 | 19.2% | 5 | 19.2% | 7.55 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | |z| > |x| | 24 | 2 | 8.3% | 2 | 8.3% | 7.64 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | +x +z | 12 | 2 | 16.7% | 2 | 16.7% | 6.66 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | -x +z | 13 | 1 | 7.7% | 1 | 7.7% | 7.64 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | -x -z | 12 | 3 | 25.0% | 3 | 25.0% | 7.55 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | +x -z | 13 | 1 | 7.7% | 1 | 7.7% | 8.44 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | x < 0 | 25 | 0 | 0.0% | 0 | 0.0% | 7.60 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | x >= 0 | 25 | 2 | 8.0% | 2 | 8.0% | 7.58 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | z < 0 | 25 | 0 | 0.0% | 0 | 0.0% | 8.01 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | z >= 0 | 25 | 2 | 8.0% | 2 | 8.0% | 7.17 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | |x| >= |z| | 26 | 1 | 3.8% | 1 | 3.8% | 7.55 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | |z| > |x| | 24 | 1 | 4.2% | 1 | 4.2% | 7.64 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | +x +z | 12 | 2 | 16.7% | 2 | 16.7% | 6.66 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | -x +z | 13 | 0 | 0.0% | 0 | 0.0% | 7.64 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | -x -z | 12 | 0 | 0.0% | 0 | 0.0% | 7.55 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | +x -z | 13 | 0 | 0.0% | 0 | 0.0% | 8.44 |

## 11. 可配对实验的逐点比较

所有实验共享相同 50 个 task points，因此可做逐点 paired analysis。对同一 checkpoint 的 mid-vs-temporal 对比，McNemar/binomial exact p-value 以 discordant pairs 为基础，作为探索性统计。

| experiment_A | experiment_B | protocol_variables_differing | task_both_success | task_A_only | task_B_only | task_both_fail | task_discordant_pairs | task_mcnemar_exact_p | task_success_rate_difference_A_minus_B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | action_select_mode | 1 | 10 | 0 | 39 | 10 | 0.001953 | 20.0 pp |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | action_select_mode | 2 | 4 | 0 | 44 | 4 | 0.125 | 8.0 pp |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | action_select_mode | 2 | 5 | 0 | 43 | 5 | 0.0625 | 10.0 pp |

结果表明，同 checkpoint 下 mid execution 在该点集上均有更多 task success。可能机制是执行选择模式改变了闭环动作序列稳定性；该解释仍需视频与轨迹验证，不能写成确定物理因果。

![Paired transition counts](../../outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/paired_success_transition_counts.png)

## 12. 力与完成时间统计

可用列包括 `max_force`、`mean_force`、`force_gt_20_steps`、`force_gt_40_steps`、`success_step`、`success_time`、`final_dist`、`final_lateral`、`final_axial`。未发现 `peak_axial_force`、`force_gt_80_steps` 或显式 rollout duration 列，因此这些指标报告为不可用。

全 completed runs 的 `max_force` 统计：

| experiment_name | count | mean | std | median | p95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | 50 | 45.59 | 24.08 | 42.50 | 99.38 | 1.26 | 101.96 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | 50 | 49.17 | 28.38 | 44.98 | 100.08 | 1.22 | 109.68 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | 50 | 69.18 | 28.96 | 80.51 | 92.92 | 1.22 | 102.87 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | 50 | 53.39 | 28.90 | 49.40 | 92.02 | 1.19 | 97.19 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | 50 | 66.44 | 22.44 | 71.88 | 90.01 | 1.22 | 95.19 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | 50 | 62.67 | 29.15 | 60.85 | 103.97 | 1.22 | 108.65 |

成功 runs 的 `success_time` 统计：

| experiment_name | count | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | 11 | 9.07 | 5.56 | 6.70 | 5.74 | 24.75 |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | 1 | 18.61 | 0.00 | 18.61 | 18.61 | 18.61 |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | 6 | 7.48 | 1.32 | 7.64 | 5.58 | 9.17 |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | 2 | 18.55 | 6.58 | 18.55 | 13.89 | 23.20 |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | 7 | 11.16 | 8.97 | 7.39 | 5.38 | 29.67 |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | 2 | 15.54 | 2.19 | 15.54 | 13.99 | 17.09 |

结果表明：成功样本的完成时间存在较大离散性；失败样本不能仅凭 summary 数据解释为某种具体行为失败。力统计显示不同实验的最大力分布不同，但 safe success 只对成功样本应用 80 N 阈值定义，不代表所有失败过程都安全。

## 13. 失败类型分析

确认测量结果：6 个实验均无 process error。失败点主要由 `stop_reason=max_rollout_steps` 表示，即达到最大 rollout 步数仍未满足成功条件。当前 summary-level 数据支持的失败分类包括：

- `process error`：本批为 0。
- `timeout / maximum rollout steps`：达到 `max_rollout_steps=900` 且未成功。
- `unknown task failure`：若缺少可解释 stop reason 或详细轨迹，只能保守归类。

本报告不把失败解释为“未搜索”“未退让”“未释放力”等行为级标签；这些假设需要 `rollout_log.csv`、视频和轨迹验证。

## 14. 代表性成功与失败案例

代表性案例按实测指标选择，包括最大半径成功、最小半径失败、最高力成功/失败、最快/最慢成功，以及 paired comparison 中的 mid-only、temporal-only、both-fail、both-success 点。完整列表见 `representative_cases.csv`。

| experiment_name | category | point_index | x_mm | z_mm | radius_mm | success | safe_success | max_force | success_time | stop_reason | rollout_csv_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | largest-radius success | 18 | -5.889 | -1.715 | 6.134 | True | True | 32.055 | 24.750 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/point_018_x_m005889mm_z_m001715mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | smallest-radius failure | 23 | 0.379 | 3.003 | 3.027 | False | False | 43.128 |  | max_rollout_steps | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/point_023_x_p000379mm_z_p003003mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | highest-force success | 25 | 3.863 | 3.483 | 5.201 | True | True | 43.627 | 6.039 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/point_025_x_p003863mm_z_p003483mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | slowest success | 18 | -5.889 | -1.715 | 6.134 | True | True | 32.055 | 24.750 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/point_018_x_m005889mm_z_m001715mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | fastest success | 26 | 1.986 | -1.264 | 2.354 | True | True | 10.930 | 5.742 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/point_026_x_p001986mm_z_m001264mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | largest-radius success | 3 | -3.484 | -0.580 | 3.532 | True | True | 11.051 | 18.612 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002/point_003_x_m003484mm_z_m000580mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | smallest-radius failure | 9 | 0.935 | 0.031 | 0.935 | False | False | 23.340 |  | max_rollout_steps | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002/point_009_x_p000935mm_z_p000031mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | highest-force success | 3 | -3.484 | -0.580 | 3.532 | True | True | 11.051 | 18.612 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002/point_003_x_m003484mm_z_m000580mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | slowest success | 3 | -3.484 | -0.580 | 3.532 | True | True | 11.051 | 18.612 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002/point_003_x_m003484mm_z_m000580mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | fastest success | 3 | -3.484 | -0.580 | 3.532 | True | True | 11.051 | 18.612 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002/point_003_x_m003484mm_z_m000580mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | largest-radius success | 41 | -2.307 | -3.814 | 4.458 | True | True | 64.151 | 9.174 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900/point_041_x_m002307mm_z_m003814mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | smallest-radius failure | 23 | 0.379 | 3.003 | 3.027 | False | False | 93.677 |  | max_rollout_steps | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900/point_023_x_p000379mm_z_p003003mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | highest-force success | 41 | -2.307 | -3.814 | 4.458 | True | True | 64.151 | 9.174 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900/point_041_x_m002307mm_z_m003814mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | slowest success | 41 | -2.307 | -3.814 | 4.458 | True | True | 64.151 | 9.174 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900/point_041_x_m002307mm_z_m003814mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | fastest success | 26 | 1.986 | -1.264 | 2.354 | True | True | 36.677 | 5.577 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900/point_026_x_p001986mm_z_m001264mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | largest-radius success | 26 | 1.986 | -1.264 | 2.354 | True | True | 29.352 | 13.893 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/point_026_x_p001986mm_z_m001264mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | smallest-radius failure | 29 | 0.602 | 1.578 | 1.688 | False | False | 1.193 |  | max_rollout_steps | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/point_029_x_p000602mm_z_p001578mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | highest-force success | 26 | 1.986 | -1.264 | 2.354 | True | True | 29.352 | 13.893 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/point_026_x_p001986mm_z_m001264mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | slowest success | 9 | 0.935 | 0.031 | 0.935 | True | True | 9.474 | 23.199 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/point_009_x_p000935mm_z_p000031mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | fastest success | 26 | 1.986 | -1.264 | 2.354 | True | True | 29.352 | 13.893 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/point_026_x_p001986mm_z_m001264mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | largest-radius success | 40 | -5.069 | -0.935 | 5.154 | True | True | 58.692 | 29.667 | success | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900/point_040_x_m005069mm_z_m000935mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | smallest-radius failure | 5 | 1.369 | 2.575 | 2.917 | False | False | 95.188 |  | max_rollout_steps | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900/point_005_x_p001369mm_z_p002575mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | highest-force success | 40 | -5.069 | -0.935 | 5.154 | True | True | 58.692 | 29.667 | success | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900/point_040_x_m005069mm_z_m000935mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | slowest success | 40 | -5.069 | -0.935 | 5.154 | True | True | 58.692 | 29.667 | success | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900/point_040_x_m005069mm_z_m000935mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | fastest success | 9 | 0.935 | 0.031 | 0.935 | True | True | 14.660 | 5.379 | success | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900/point_009_x_p000935mm_z_p000031mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | largest-radius success | 29 | 0.602 | 1.578 | 1.688 | True | True | 44.235 | 13.992 | success | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900/point_029_x_p000602mm_z_p001578mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | smallest-radius failure | 26 | 1.986 | -1.264 | 2.354 | False | False | 93.719 |  | max_rollout_steps | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900/point_026_x_p001986mm_z_m001264mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | highest-force success | 9 | 0.935 | 0.031 | 0.935 | True | True | 72.525 | 17.094 | success | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900/point_009_x_p000935mm_z_p000031mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | slowest success | 9 | 0.935 | 0.031 | 0.935 | True | True | 72.525 | 17.094 | success | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900/point_009_x_p000935mm_z_p000031mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | fastest success | 29 | 0.602 | 1.578 | 1.688 | True | True | 44.235 | 13.992 | success | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900/point_029_x_p000602mm_z_p001578mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 vs hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | mid-only success | 18 | -5.889 | -1.715 | 6.134 | True | True | 32.055 | 24.750 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/point_018_x_m005889mm_z_m001715mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 vs hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | both-fail point | 16 | -9.852 | -8.440 | 12.973 | False | False | 98.477 |  | max_rollout_steps | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/point_016_x_m009852mm_z_m008440mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 vs hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | both-success point | 3 | -3.484 | -0.580 | 3.532 | True | True | 27.955 | 6.699 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/point_003_x_m003484mm_z_m000580mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 vs hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | mid-only success | 40 | -5.069 | -0.935 | 5.154 | True | True | 58.692 | 29.667 | success | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900/point_040_x_m005069mm_z_m000935mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 vs hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | both-fail point | 16 | -9.852 | -8.440 | 12.973 | False | False | 71.838 |  | max_rollout_steps | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900/point_016_x_m009852mm_z_m008440mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 vs hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | both-success point | 29 | 0.602 | 1.578 | 1.688 | True | True | 39.251 | 6.270 | success | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900/point_029_x_p000602mm_z_p001578mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 vs hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | mid-only success | 41 | -2.307 | -3.814 | 4.458 | True | True | 64.151 | 9.174 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900/point_041_x_m002307mm_z_m003814mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 vs hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | both-fail point | 16 | -9.852 | -8.440 | 12.973 | False | False | 71.599 |  | max_rollout_steps | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900/point_016_x_m009852mm_z_m008440mm_repeat_001/rollout_log.csv |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 vs hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | both-success point | 26 | 1.986 | -1.264 | 2.354 | True | True | 36.677 | 5.577 | success | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900/point_026_x_p001986mm_z_m001264mm_repeat_001/rollout_log.csv |

## 15. 主要实验发现

1. 确认测量结果：Contact-CVAE zero + mid 在本地 50 点 ±10 mm LHS 实验中成功数最高，为 11/50。
2. 统计结果：同 checkpoint 的 paired comparison 显示 mid-only success 多于 temporal-only success。
3. 空间结果：成功随半径增大总体减少，但仍存在较大半径成功样本；这不是连续成功区域。
4. 安全成功：所有 task success 同时也是 safe success，依据是当前 `max_force < 80 N` 阈值定义。

## 16. 对空间泛化能力的解释

数据支持模型在小半径偏移上更容易成功，在较大半径上成功率下降。可能的机制是较大 x/z 偏移增加了插入定位与接触修正难度，而 action selection mode 影响动作序列的稳定性。该解释仍需视频与轨迹验证；本报告不声称模型已经学习到主动搜索、退让或力释放。

## 17. 实验局限性

1. 每个实验只有 50 个 LHS 点，不能推断连续成功边界。
2. 方向分析是探索性的，样本设计并非方向假设检验。
3. 本报告未系统解析完整视频或逐步轨迹。
4. 跨 checkpoint 比较同时改变模型身份，不能作为单变量因果结果。
5. `safe_success` 是配置阈值，不是硬件安全证明。

## 18. 后续实验建议

1. 对成功/失败转换半径附近做重复 rollout 或局部网格采样。
2. 对代表性点解析 `rollout_log.csv` 与视频，验证接触过程和力峰值来源。
3. 设计方向平衡实验以检验 x/z 或象限不对称。
4. 保持 task point set、成功阈值和 force 阈值一致，以继续进行 paired comparison。
5. 对高潜力策略增加随机种子重复，以估计统计不确定性。

## 19. 结论

本地发现的 6 个 50 点 ±10 mm 孔位扰动 rollout 实验全部完整，并且使用相同 task points，可进行直接逐点比较。最高任务成功率与安全成功率为 Contact-CVAE zero + mid 的 22.0%。同 checkpoint 下，mid execution 在该点集上优于 temporal execution。空间分析显示成功集中于较小半径，并存在方向差异的探索性迹象；这些结果必须限于稀疏 LHS 测量点，不能解释为连续泛化区域或确定物理机制。

## 附录A：实验路径与配置

| experiment_name | relative_path | inferred_policy_variant | inferred_checkpoint | latent_mode | action_select_mode | temporal_agg_decay | max_delta_q | max_rollout_steps | base_seed | sampling_mode | requested_points | x_min | x_max | z_min | z_max | completion_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | force_aware_contact_cvae | outputs/peg_hole_100/forceaware_contact_cvae_betac5e4_lp01_trajectory100k/checkpoint.pt | zero | mid |  | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | force_aware_contact_cvae | outputs/peg_hole_100/forceaware_contact_cvae_betac5e4_lp01_trajectory100k/checkpoint.pt | zero | temporal |  | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | force_aware_motion_cvae | outputs/peg_hole_100/forceaware_motion_cvae_betam5e4_trajectory100k/checkpoint_step_00100000.pt | zero | mid |  | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | force_aware_motion_cvae | outputs/peg_hole_100/forceaware_motion_cvae_betam5e4_trajectory100k/checkpoint_step_00100000.pt | zero | temporal |  | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | force_aware_act_dualzero | outputs/peg_hole_100/forceaware_dualzero_trajectory100k/checkpoint_step_00100000.pt | zero | mid |  | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | force_aware_act_dualzero | outputs/peg_hole_100/forceaware_dualzero_trajectory100k/checkpoint_step_00100000.pt | zero | temporal |  | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |

## 附录B：派生分析文件

- `outputs/peg_hole_100/xz_pm10mm_rollout_report/experiment_inventory.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/experiment_summary.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/paired_transition_summary.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/spatial_radius_summary.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/directional_summary.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/force_timing_summary.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/representative_cases.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/`
