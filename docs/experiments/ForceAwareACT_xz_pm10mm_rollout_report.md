# ForceAwareACT ±10 mm孔位扰动Rollout实验报告

## 1. 实验背景与目的

本报告对仓库本地 `outputs/peg_hole_100` 下 50 点 x/z ±10 mm 孔位扰动 rollout 记录进行证据化分析。目标是盘点实验完整性，判断哪些实验可以直接比较，并总结任务成功、安全成功、失败、力、完成时间、空间半径与方向分布。

报告正文使用短实验标签以减少表格宽度；完整实验目录、最大步数、±10 mm x/z offset、50 个采样点和其他设置保留在附录 A。纳入实验覆盖 Contact-CVAE、Motion-CVAE、DualZero 与 ACT baseline 四类策略，每类均包含 `mid` 与 `temporal` 两种 action selection mode。

本报告仅使用本地已有实验文件和新生成的派生图表。没有修改模型、训练、推理、rollout、成功判据、checkpoint、normalization 或 dataset 代码。

## 2. 报告范围

搜寻对象限定为目录名包含 `hole_lhs_50_xz_pm10mm` 或 `hole_lhs_50_xz_10mm` 的实验目录。有效实验根目录至少需要包含 `grid_summary.csv`。本报告主表只包含 50 点 ±10 mm x/z 孔位扰动实验，不混入其他扰动范围。

术语定义：

- `task success`：`grid_summary.csv` 中的 `success=True`。
- `safe success`：`grid_summary.csv` 中的 `safe_success=True`；当前配置通常表示 task success 且满足 `success_force_threshold=80 N` 的力阈值约束。
- `hard force stop`：配置中的 `force_stop_threshold=1000 N`。
- `process error`：manifest/run summary 中记录为进程错误的运行。

注意：`safe_success` 是本实验配置内的阈值定义，不是通用硬件安全保证。

## 3. 实验记录搜寻与纳入标准

确认测量结果：本地共纳入 8 个有效实验根目录，全部包含 `grid_summary.csv`、`random_position_summary.json`、`grid_manifest.json` 与 `task_points.csv`。这些实验共同构成同一 50 点 ±10 mm LHS 点集上的策略与 action selection mode 对照。

机器可读清单写入：`outputs/peg_hole_100/xz_pm10mm_rollout_report/experiment_inventory.csv`。

## 4. 实验完整性检查

确认测量结果：8 个实验均满足标准 50 点实验条件：`completed_points=50`、`process_error_points=0`、`completion_rate=100%`。未发现本次纳入实验的 process error。

| experiment_label | completed_points | successful_points | failed_task_points | process_error_points | completion_rate | classification |
| --- | --- | --- | --- | --- | --- | --- |
| Contact-CVAE-mid | 50 | 11 | 39 | 0 | 100.0% | complete |
| Contact-CVAE-temporal | 50 | 1 | 49 | 0 | 100.0% | complete |
| Motion-CVAE-mid | 50 | 6 | 44 | 0 | 100.0% | complete |
| Motion-CVAE-temporal | 50 | 2 | 48 | 0 | 100.0% | complete |
| DualZero-mid | 50 | 7 | 43 | 0 | 100.0% | complete |
| DualZero-temporal | 50 | 2 | 48 | 0 | 100.0% | complete |
| ACT baseline-mid | 50 | 14 | 36 | 0 | 100.0% | complete |
| ACT baseline-temporal | 50 | 4 | 46 | 0 | 100.0% | complete |

## 5. 实验协议与可比性

所有纳入实验共享相同空间采样协议：`sampling_mode=latin_hypercube`、`base_seed=20260702`、`x_min=-0.01`、`x_max=0.01`、`z_min=-0.01`、`z_max=0.01`、`y_offset=0`。成功阈值一致：`success_distance_threshold=0.005`、`success_lateral_threshold=0.006`、`success_force_threshold=80.0`、`success_hold_steps=15`。执行约束一致：`max_delta_q=0.02`、`max_rollout_steps=900`、`force_stop_threshold=1000.0`、`chunk_len=10`。

逐点比较要求 `point_index`、`hole_offset_x`、`hole_offset_y`、`hole_offset_z` 完全一致。8 个实验两两共享相同 task points，最大 x/y/z offset 差值为 0，因此可以进行逐点 paired analysis。解释时仍需区分：同一 checkpoint 的 mid-vs-temporal 对比主要改变 `action_select_mode`；跨 checkpoint 比较同时改变模型/策略身份，因此属于描述性 paired comparison，不能作为单因素因果结论。

## 6. 纳入实验清单

| experiment_label | policy_variant | action_select_mode | completed_points | successful_points | failed_task_points | safe_successful_points | task_success_rate | safe_success_rate | success_ci95_lower | success_ci95_upper | largest_radius_success_mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Contact-CVAE-mid | force_aware_contact_cvae | mid | 50 | 11 | 39 | 11 | 22.0% | 22.0% | 12.8% | 35.2% | 6.13 |
| Contact-CVAE-temporal | force_aware_contact_cvae | temporal | 50 | 1 | 49 | 1 | 2.0% | 2.0% | 0.4% | 10.5% | 3.53 |
| Motion-CVAE-mid | force_aware_motion_cvae | mid | 50 | 6 | 44 | 6 | 12.0% | 12.0% | 5.6% | 23.8% | 4.46 |
| Motion-CVAE-temporal | force_aware_motion_cvae | temporal | 50 | 2 | 48 | 2 | 4.0% | 4.0% | 1.1% | 13.5% | 2.35 |
| DualZero-mid | force_aware_act_dualzero | mid | 50 | 7 | 43 | 7 | 14.0% | 14.0% | 7.0% | 26.2% | 5.15 |
| DualZero-temporal | force_aware_act_dualzero | temporal | 50 | 2 | 48 | 2 | 4.0% | 4.0% | 1.1% | 13.5% | 1.69 |
| ACT baseline-mid | act_baseline | mid | 50 | 14 | 36 | 3 | 28.0% | 6.0% | 17.5% | 41.7% | 10.86 |
| ACT baseline-temporal | act_baseline | temporal | 50 | 4 | 46 | 3 | 8.0% | 6.0% | 3.2% | 18.8% | 5.52 |

## 7. 总体任务成功与安全成功结果

确认测量结果：8 个实验的 task success 覆盖 1/50 到 14/50。最高任务成功率来自 `ACT baseline-mid`，为 14/50，即 28.0%；其次是 `Contact-CVAE-mid`，为 11/50，即 22.0%。所有 `mid` 版本的 task success 均高于同策略的 `temporal` 版本。

安全成功结果与任务成功结果并不总是一致。Contact-CVAE、Motion-CVAE 与 DualZero 的 task success 均同时满足 safe success；ACT baseline 的 safe success 明显低于 task success，`ACT baseline-mid` 为 3/50，`ACT baseline-temporal` 为 3/50，说明部分 task success 的最大力超过了本实验 80 N safe-success 阈值。Wilson 95% 置信区间描述的是本 50 点样本上的不确定性，不代表连续空间成功区域。

![Task success rate](../../outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/task_success_rate_by_experiment.png)

![Safe success rate](../../outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/safe_success_rate_by_experiment.png)

## 8. 各实验的靶心空间分布

所有 target-map 图均显示实测 rollout 点：成功为绿色实心圆，失败为红色实心圆，黑色边缘，名义孔中心在 `(0, 0)`。这些图不插值、不估计连续成功区域。用于比较的图均采用相同尺度：`ring_step_mm=2`、`max_radius_mm=14`。

- `Contact-CVAE-mid`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/plots/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002_target.png)
- `Contact-CVAE-temporal`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002/plots/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002_target.png)
- `Motion-CVAE-mid`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900/plots/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900_target.png)
- `Motion-CVAE-temporal`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/plots/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900_target.png)
- `DualZero-mid`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900/plots/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900_target.png)
- `DualZero-temporal`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900/plots/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900_target.png)
- `ACT baseline-mid`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_act_baseline100k_mid_dq002/plots/act_baseline_mid_10mm_target.png)
- `ACT baseline-temporal`: [`target PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_act_baseline100k_temporal_d03_dq002/plots/act_baseline_temporal_10mm_target.png)

另有带 point index 与采样边界的诊断图，便于追查具体样本点：

- `ACT baseline-mid`: [`labeled PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_act_baseline100k_mid_dq002/plots/act_baseline_mid_10mm_target_labeled.png)
- `ACT baseline-temporal`: [`labeled PNG`](../../outputs/peg_hole_100/hole_lhs_50_xz_10mm_act_baseline100k_temporal_d03_dq002/plots/act_baseline_temporal_10mm_target_labeled.png)

## 9. 成功率随孔位半径的变化

半径按 `radius_mm = sqrt(x_mm^2 + z_mm^2)` 计算，并使用固定 bin：`[0,2)`、`[2,4)`、`[4,6)`、`[6,8)`、`[8,10)`、`[10,12)`、`[12,15)`。

结果表明：成功样本总体集中在较小半径，但不同策略的半径分布并不相同。除 `ACT baseline-mid` 外，其余实验的成功点都落在 8 mm 以内；`ACT baseline-mid` 在 6 mm 以上仍有 10 个 task success，最大观测成功半径为 10.86 mm。这里的“最大观测成功半径”仅指本 LHS 采样集合内的成功点，不是绝对泛化边界。

按近中心半径 `[0,6)` 与较大半径 `[6,15)` 汇总如下。完整逐 bin 结果见 `spatial_radius_summary.csv`。

| experiment_label | total_successes | successes_in_[0,6)_mm | successes_in_[6,15)_mm | outermost_success_bin | largest_radius_success_mm |
| --- | --- | --- | --- | --- | --- |
| Contact-CVAE-mid | 11 | 10/16 | 1/34 | [6, 8) | 6.13 |
| Contact-CVAE-temporal | 1 | 1/16 | 0/34 | [2, 4) | 3.53 |
| Motion-CVAE-mid | 6 | 6/16 | 0/34 | [4, 6) | 4.46 |
| Motion-CVAE-temporal | 2 | 2/16 | 0/34 | [2, 4) | 2.35 |
| DualZero-mid | 7 | 7/16 | 0/34 | [4, 6) | 5.15 |
| DualZero-temporal | 2 | 2/16 | 0/34 | [0, 2) | 1.69 |
| ACT baseline-mid | 14 | 4/16 | 10/34 | [10, 12) | 10.86 |
| ACT baseline-temporal | 4 | 4/16 | 0/34 | [4, 6) | 5.52 |

![Radius binned success rate](../../outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/radius_binned_success_rate.png)

![Largest observed successful radius](../../outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/largest_success_radius_by_experiment.png)

关键半径代表点：

| experiment_label | category | point_index | x_mm | z_mm | radius_mm | success | safe_success | max_force | success_time | stop_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Contact-CVAE-mid | largest-radius success | 18 | -5.889 | -1.715 | 6.134 | True | True | 32.05 | 24.75 | success |
| Contact-CVAE-mid | smallest-radius failure | 23 | 0.379 | 3.003 | 3.027 | False | False | 43.13 |  | max_rollout_steps |
| Contact-CVAE-temporal | largest-radius success | 3 | -3.484 | -0.580 | 3.532 | True | True | 11.05 | 18.61 | success |
| Contact-CVAE-temporal | smallest-radius failure | 9 | 0.935 | 0.031 | 0.935 | False | False | 23.34 |  | max_rollout_steps |
| Motion-CVAE-mid | largest-radius success | 41 | -2.307 | -3.814 | 4.458 | True | True | 64.15 | 9.17 | success |
| Motion-CVAE-mid | smallest-radius failure | 23 | 0.379 | 3.003 | 3.027 | False | False | 93.68 |  | max_rollout_steps |
| Motion-CVAE-temporal | largest-radius success | 26 | 1.986 | -1.264 | 2.354 | True | True | 29.35 | 13.89 | success |
| Motion-CVAE-temporal | smallest-radius failure | 29 | 0.602 | 1.578 | 1.688 | False | False | 1.19 |  | max_rollout_steps |
| DualZero-mid | largest-radius success | 40 | -5.069 | -0.935 | 5.154 | True | True | 58.69 | 29.67 | success |
| DualZero-mid | smallest-radius failure | 5 | 1.369 | 2.575 | 2.917 | False | False | 95.19 |  | max_rollout_steps |
| DualZero-temporal | largest-radius success | 29 | 0.602 | 1.578 | 1.688 | True | True | 44.24 | 13.99 | success |
| DualZero-temporal | smallest-radius failure | 26 | 1.986 | -1.264 | 2.354 | False | False | 93.72 |  | max_rollout_steps |
| ACT baseline-mid | largest-radius success | 22 | 8.134 | -7.191 | 10.857 | True | False | 97.57 | 18.78 | success |
| ACT baseline-mid | smallest-radius failure | 5 | 1.369 | 2.575 | 2.917 | False | False | 100.61 |  | max_rollout_steps |
| ACT baseline-temporal | largest-radius success | 15 | 4.328 | -3.434 | 5.525 | True | False | 89.03 | 26.27 | success |
| ACT baseline-temporal | smallest-radius failure | 26 | 1.986 | -1.264 | 2.354 | False | False | 87.24 |  | max_rollout_steps |

## 10. x/z方向与象限分析

方向与象限分析基于实际计数。由于 LHS 不是专门为方向假设检验构造的平衡实验，以下结果应视为探索性统计，不应仅凭视觉分布断言方向性机制。

方向摘要如下。完整方向与象限结果见 `directional_summary.csv`。

| experiment_label | x < 0 | x >= 0 | z < 0 | z >= 0 | highest_success_quadrant |
| --- | --- | --- | --- | --- | --- |
| Contact-CVAE-mid | 6/25 (24.0%) | 5/25 (20.0%) | 5/25 (20.0%) | 6/25 (24.0%) | +x +z: 4/12 (33.3%) |
| Contact-CVAE-temporal | 1/25 (4.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) | -x -z: 1/12 (8.3%) |
| Motion-CVAE-mid | 2/25 (8.0%) | 4/25 (16.0%) | 3/25 (12.0%) | 3/25 (12.0%) | +x +z: 3/12 (25.0%) |
| Motion-CVAE-temporal | 0/25 (0.0%) | 2/25 (8.0%) | 1/25 (4.0%) | 1/25 (4.0%) | +x +z: 1/12 (8.3%) |
| DualZero-mid | 4/25 (16.0%) | 3/25 (12.0%) | 4/25 (16.0%) | 3/25 (12.0%) | -x -z: 3/12 (25.0%) |
| DualZero-temporal | 0/25 (0.0%) | 2/25 (8.0%) | 0/25 (0.0%) | 2/25 (8.0%) | +x +z: 2/12 (16.7%) |
| ACT baseline-mid | 1/25 (4.0%) | 13/25 (52.0%) | 11/25 (44.0%) | 3/25 (12.0%) | +x -z: 10/13 (76.9%) |
| ACT baseline-temporal | 0/25 (0.0%) | 4/25 (16.0%) | 1/25 (4.0%) | 3/25 (12.0%) | +x +z: 3/12 (25.0%) |

确认测量结果：部分实验存在明显方向计数差异，例如 `ACT baseline-mid` 在 `+x -z` 象限有 10/13 task success，而其他象限成功较少。该现象可以作为后续轨迹检查的线索，但机械解释仍需视频与 `rollout_log.csv` 验证，不能仅凭 summary 断言模型学会了某种搜索或接触策略。

## 11. 可配对实验的逐点比较

所有实验共享相同 50 个 task points，因此可做逐点 paired analysis。对同一 checkpoint 的 mid-vs-temporal 对比，McNemar/binomial exact p-value 以 discordant pairs 为基础，作为探索性统计。

| experiment_A | experiment_B | protocol_variables_differing | point_paired | task_both_success | task_A_only | task_B_only | task_both_fail | task_discordant_pairs | task_mcnemar_exact_p | task_success_rate_difference_A_minus_B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Contact-CVAE-mid | Contact-CVAE-temporal | action_select_mode | True | 1 | 10 | 0 | 39 | 10 | 0.001953 | 20.0 pp |
| Motion-CVAE-mid | Motion-CVAE-temporal | action_select_mode | True | 2 | 4 | 0 | 44 | 4 | 0.125000 | 8.0 pp |
| DualZero-mid | DualZero-temporal | action_select_mode | True | 2 | 5 | 0 | 43 | 5 | 0.062500 | 10.0 pp |
| ACT baseline-mid | ACT baseline-temporal | action_select_mode | True | 3 | 11 | 1 | 35 | 12 | 0.006348 | 20.0 pp |

配对结果显示，四类策略在同一 50 点 task set 上均呈现 `mid` 多于 `temporal` 的 task success。`Contact-CVAE` 与 `ACT baseline` 的 discordant pairs 最多，exact two-sided McNemar/binomial p-value 分别为 0.001953 与 0.006348。可能机制是 action selection mode 改变了闭环动作序列的稳定性；该解释仍需视频与轨迹验证，不能写成确定物理因果。

![Paired transition counts](../../outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/paired_success_transition_counts.png)

## 12. 力与完成时间统计

可用列包括 `max_force`、`mean_force`、`force_gt_20_steps`、`force_gt_40_steps`、`success_step`、`success_time`、`final_dist`、`final_lateral`、`final_axial`。未发现 `peak_axial_force`、`force_gt_80_steps` 或显式 rollout duration 列，因此这些指标报告为不可用。

核心力与时间统计如下：

| experiment_label | group | metric | count | mean | std | median | p95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Contact-CVAE-mid | all_completed | max_force | 50 | 45.59 | 24.08 | 42.50 | 99.38 | 1.26 | 101.96 |
| Contact-CVAE-mid | all_completed | success_time | 11 | 9.07 | 5.56 | 6.70 | 18.41 | 5.74 | 24.75 |
| Contact-CVAE-mid | successful | max_force | 11 | 28.78 | 10.38 | 28.18 | 43.38 | 10.93 | 43.63 |
| Contact-CVAE-mid | successful | success_time | 11 | 9.07 | 5.56 | 6.70 | 18.41 | 5.74 | 24.75 |
| Contact-CVAE-temporal | all_completed | max_force | 50 | 49.17 | 28.38 | 44.98 | 100.08 | 1.22 | 109.68 |
| Contact-CVAE-temporal | all_completed | success_time | 1 | 18.61 | 0.00 | 18.61 | 18.61 | 18.61 | 18.61 |
| Contact-CVAE-temporal | successful | max_force | 1 | 11.05 | 0.00 | 11.05 | 11.05 | 11.05 | 11.05 |
| Contact-CVAE-temporal | successful | success_time | 1 | 18.61 | 0.00 | 18.61 | 18.61 | 18.61 | 18.61 |
| Motion-CVAE-mid | all_completed | max_force | 50 | 69.18 | 28.96 | 80.51 | 92.92 | 1.22 | 102.87 |
| Motion-CVAE-mid | all_completed | success_time | 6 | 7.48 | 1.32 | 7.64 | 8.98 | 5.58 | 9.17 |
| Motion-CVAE-mid | successful | max_force | 6 | 41.33 | 16.83 | 36.35 | 62.73 | 19.21 | 64.15 |
| Motion-CVAE-mid | successful | success_time | 6 | 7.48 | 1.32 | 7.64 | 8.98 | 5.58 | 9.17 |
| Motion-CVAE-temporal | all_completed | max_force | 50 | 53.39 | 28.90 | 49.40 | 92.02 | 1.19 | 97.19 |
| Motion-CVAE-temporal | all_completed | success_time | 2 | 18.55 | 6.58 | 18.55 | 22.73 | 13.89 | 23.20 |
| Motion-CVAE-temporal | successful | max_force | 2 | 19.41 | 14.06 | 19.41 | 28.36 | 9.47 | 29.35 |
| Motion-CVAE-temporal | successful | success_time | 2 | 18.55 | 6.58 | 18.55 | 22.73 | 13.89 | 23.20 |
| DualZero-mid | all_completed | max_force | 50 | 66.44 | 22.44 | 71.88 | 90.01 | 1.22 | 95.19 |
| DualZero-mid | all_completed | success_time | 7 | 11.16 | 8.97 | 7.39 | 25.63 | 5.38 | 29.67 |
| DualZero-mid | successful | max_force | 7 | 38.49 | 17.55 | 39.25 | 57.86 | 14.66 | 58.69 |
| DualZero-mid | successful | success_time | 7 | 11.16 | 8.97 | 7.39 | 25.63 | 5.38 | 29.67 |
| DualZero-temporal | all_completed | max_force | 50 | 62.67 | 29.15 | 60.85 | 103.97 | 1.22 | 108.65 |
| DualZero-temporal | all_completed | success_time | 2 | 15.54 | 2.19 | 15.54 | 16.94 | 13.99 | 17.09 |
| DualZero-temporal | successful | max_force | 2 | 58.38 | 20.00 | 58.38 | 71.11 | 44.24 | 72.52 |
| DualZero-temporal | successful | success_time | 2 | 15.54 | 2.19 | 15.54 | 16.94 | 13.99 | 17.09 |
| ACT baseline-mid | all_completed | max_force | 50 | 85.77 | 25.11 | 95.03 | 104.28 | 2.85 | 106.81 |
| ACT baseline-mid | all_completed | success_time | 14 | 12.92 | 7.09 | 10.59 | 24.78 | 5.61 | 25.64 |
| ACT baseline-mid | successful | max_force | 14 | 80.55 | 33.57 | 93.99 | 102.24 | 2.85 | 102.86 |
| ACT baseline-mid | successful | success_time | 14 | 12.92 | 7.09 | 10.59 | 24.78 | 5.61 | 25.64 |
| ACT baseline-temporal | all_completed | max_force | 50 | 72.28 | 31.18 | 86.49 | 103.22 | 1.17 | 106.51 |
| ACT baseline-temporal | all_completed | success_time | 4 | 17.14 | 6.11 | 14.44 | 24.54 | 13.43 | 26.27 |
| ACT baseline-temporal | successful | max_force | 4 | 50.76 | 37.40 | 56.41 | 85.68 | 1.19 | 89.03 |
| ACT baseline-temporal | successful | success_time | 4 | 17.14 | 6.11 | 14.44 | 24.54 | 13.43 | 26.27 |

确认测量结果：不同实验的力分布差异明显。`Contact-CVAE-mid` 的成功样本 `max_force` 均值为 28.78 N，`Motion-CVAE-mid` 为 41.33 N，`DualZero-mid` 为 38.49 N，而 `ACT baseline-mid` 为 80.55 N。由于 safe success 需要满足 80 N 阈值，`ACT baseline-mid` 虽然 task success 最高，但 safe success 只有 3/50。完成时间方面，`mid` 版本的成功样本 median success time 通常短于对应 `temporal` 版本，但样本数较小，不能把该观察写成稳健时间优势。

## 13. 失败类型分析

确认测量结果：8 个实验均无 process error。失败点主要由 `stop_reason=max_rollout_steps` 表示，即达到 `max_rollout_steps=900` 且未满足成功条件。当前 summary-level 数据支持的失败分类包括：

- `process error`：本批为 0。
- `timeout / maximum rollout steps`：达到最大 rollout 步数仍未成功。
- `unknown task failure`：若缺少可解释 stop reason 或详细轨迹，只能保守归类。

本报告不把失败解释为“未搜索”“未退让”“未释放力”等行为级标签；这些假设需要 `rollout_log.csv`、视频和轨迹验证。

## 14. 代表性成功与失败案例

代表性案例按实测指标选择，包括每个实验的最大半径成功、最小半径失败、最高力成功/失败、最快/最慢成功，以及每个 paired comparison 中的 mid-only、temporal-only、both-fail、both-success 点。完整列表见 `outputs/peg_hole_100/xz_pm10mm_rollout_report/representative_cases.csv`。

正文保留最能支撑空间泛化结论的半径代表点，便于追查后续视频和 `rollout_log.csv`：

| experiment_label | largest_radius_success_point | largest_radius_success_mm | smallest_radius_failure_point | smallest_radius_failure_mm |
| --- | --- | --- | --- | --- |
| Contact-CVAE-mid | 18 | 6.13 | 23 | 3.03 |
| Contact-CVAE-temporal | 3 | 3.53 | 9 | 0.94 |
| Motion-CVAE-mid | 41 | 4.46 | 23 | 3.03 |
| Motion-CVAE-temporal | 26 | 2.35 | 29 | 1.69 |
| DualZero-mid | 40 | 5.15 | 5 | 2.92 |
| DualZero-temporal | 29 | 1.69 | 26 | 2.35 |
| ACT baseline-mid | 22 | 10.86 | 5 | 2.92 |
| ACT baseline-temporal | 15 | 5.52 | 26 | 2.35 |

## 15. 主要实验发现

1. 确认测量结果：8 个实验全部完整，无 process error，可在同一 50 点 task set 上逐点比较。
2. 确认测量结果：task success 最高的是 `ACT baseline-mid`，14/50，即 28.0%；最高 safe success 是 `Contact-CVAE-mid`，11/50，即 22.0%。
3. 统计结果：四组同 checkpoint 的 paired comparison 均显示 `mid` 成功数高于 `temporal`；`Contact-CVAE` 与 `ACT baseline` 的 exact p-value 分别为 0.001953 与 0.006348。
4. 安全成功：task success 不能直接替代 safe success，尤其 `ACT baseline-mid` 的 task success 为 14/50，但 safe success 仅为 3/50。
5. 空间结果：成功分布与半径和方向有关，但稀疏 LHS 数据不能支持连续成功边界或确定物理机制。

## 16. 对空间泛化能力的解释

数据支持多数策略在小半径偏移上更容易出现 safe success；较大半径成功样本更少，并且部分较大半径 task success 伴随较高最大力。可能的机制是较大 x/z 偏移增加了插入定位与接触修正难度，而 action selection mode 影响动作序列的稳定性。该解释仍需视频与轨迹验证；本报告不声称模型已经学习到主动搜索、退让或力释放。

## 17. 实验局限性

1. 每个实验只有 50 个 LHS 点，不能推断连续成功边界。
2. 方向分析是探索性的，样本设计并非方向假设检验。
3. 本报告未系统解析完整视频或逐步轨迹。
4. 跨 checkpoint 比较同时改变模型身份，不能作为单变量因果结果。
5. `safe_success` 是配置阈值，不是硬件安全证明。

## 18. 后续实验建议

1. 对较大半径 task success 与高力 success 点进行视频与 `rollout_log.csv` 复核，判断接触过程和力峰值来源。
2. 对每组 mid-vs-temporal 的 discordant points 做逐点轨迹比较。
3. 对成功/失败转换半径附近做重复 rollout 或局部网格采样。
4. 设计方向平衡实验以检验 x/z 或象限不对称。
5. 保持 task point set、成功阈值和 force 阈值一致，以继续进行 paired comparison。

## 19. 结论

本地发现并纳入的 8 个 50 点 ±10 mm 孔位扰动 rollout 实验全部完整，并且使用相同 task points，可进行直接逐点比较。任务成功率最高的是 `ACT baseline-mid`，安全成功率最高的是 `Contact-CVAE-mid`；二者差异表明任务完成和力阈值下的安全成功必须分开报告。同 checkpoint 下，四组策略均表现为 `mid` task success 多于 `temporal`。空间分析显示成功分布与半径、方向有关，但这些结果必须限于稀疏 LHS 测量点，不能解释为连续泛化区域或确定物理机制。

## 附录A：实验路径与配置

| experiment_label | relative_path | inferred_policy_variant | action_select_mode | max_delta_q | max_rollout_steps | base_seed | sampling_mode | requested_points | x_min | x_max | z_min | z_max | completion_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Contact-CVAE-mid | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002 | force_aware_contact_cvae | mid | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| Contact-CVAE-temporal | outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_temporal_d03_dq002 | force_aware_contact_cvae | temporal | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| Motion-CVAE-mid | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_mid_dq002_maxsteps900 | force_aware_motion_cvae | mid | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| Motion-CVAE-temporal | outputs/peg_hole_100/hole_lhs_50_xz_10mm_motion_cvae100k_temporal_d03_dq002_maxsteps900 | force_aware_motion_cvae | temporal | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| DualZero-mid | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_mid_dq002_maxsteps900 | force_aware_act_dualzero | mid | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| DualZero-temporal | outputs/peg_hole_100/hole_lhs_50_xz_pm10mm_dualzero100k_temporal_d03_dq002_maxsteps900 | force_aware_act_dualzero | temporal | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| ACT baseline-mid | outputs/peg_hole_100/hole_lhs_50_xz_10mm_act_baseline100k_mid_dq002 | act_baseline | mid | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |
| ACT baseline-temporal | outputs/peg_hole_100/hole_lhs_50_xz_10mm_act_baseline100k_temporal_d03_dq002 | act_baseline | temporal | 0.02 | 900 | 20260702 | latin_hypercube | 50 | -0.01 | 0.01 | -0.01 | 0.01 | complete |

## 附录B：派生分析文件

- `outputs/peg_hole_100/xz_pm10mm_rollout_report/experiment_inventory.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/experiment_summary.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/paired_transition_summary.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/spatial_radius_summary.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/directional_summary.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/force_timing_summary.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/representative_cases.csv`
- `outputs/peg_hole_100/xz_pm10mm_rollout_report/report_assets/`
- `outputs/peg_hole_100/act_baseline_xz_10mm_mid_vs_temporal_analysis/paired_results.csv`
- `outputs/peg_hole_100/act_baseline_xz_10mm_mid_vs_temporal_analysis/paired_summary.json`
- `outputs/peg_hole_100/act_baseline_xz_10mm_mid_vs_temporal_analysis/radius_summary.csv`
