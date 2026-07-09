# ForceAwareACT ±4 mm孔位扰动Rollout实验报告

## 1. 实验背景与目的

本报告整理仓库本地 `outputs/peg_hole_100` 下 50 点 x/z ±4 mm 孔位扰动 rollout 结果。纳入 Contact-CVAE zero、Contact-CVAE prior、Motion-CVAE、DualZero 与 ACT baseline 五种策略/latent 配置，每种配置均包含 `mid` 和 `temporal` 两种 action selection mode，共 10 个实验、500 次 rollout。

分析目标包括：

1. 核验实验完整性和协议一致性；
2. 汇总 task success、safe success、力和完成时间；
3. 分析成功结果随半径、方向和象限的变化；
4. 对同一配置的 `mid` 与 `temporal` 进行逐点配对；
5. 在相同 Contact-CVAE checkpoint 下比较 `z_contact=zero` 与 `z_contact=prior`。

本报告仅使用本地 CSV、JSON、manifest、逐点 summary 和已有 target map。没有修改任何原始实验结果。

## 2. 报告范围与术语

纳入目录为 `hole_lhs_50_xz_4mm_*` 且包含 `grid_summary.csv` 的实验根目录。

- `task success`：`grid_summary.csv` 中 `success=True`。
- `safe success`：`safe_success=True`。本批配置的安全成功要求 task success 同时满足 `success_force_threshold=80 N`；它不是通用硬件安全保证。
- `task success, not safe`：任务成功，但未满足上述 safe-success 定义。
- `process error`：rollout 进程没有正常完成。
- `maximum observed successful radius`：本次离散 LHS 点集中最远的成功点，不是连续空间泛化边界。

所有空间结论仅描述实测点，不对稀疏样本之间的区域进行插值。

## 3. 实验协议

10 个实验共享以下协议：

| setting | value |
| --- | --- |
| sampling mode | `latin_hypercube` |
| points / repeats | 50 / 1 |
| base seed | `20260702` |
| x range | `[-0.004, 0.004] m` |
| z range | `[-0.004, 0.004] m` |
| y offset | `0 m` |
| action mode | `action` |
| chunk length | 10 |
| policy rate | 30 Hz |
| maximum rollout steps | 900 |
| maximum delta q | 0.02 rad |
| force stop threshold | 1000 N |
| force window | 20 steps / 0.25 s |
| success distance threshold | 0.005 m |
| success lateral threshold | 0.006 m |
| success force threshold | 80 N |
| success hold steps | 15 |
| hole site / body | `hole_goal_site` / `wall_task` |
| offset frame | `world` |
| normalization | `normalization_stats_action_all100.pt` |

Contact-CVAE zero 与 prior 使用相同 checkpoint。`zero` 将 `z_contact` 设为零向量；`prior` 使用当前观测条件下 contact-prior 网络预测的确定性均值。两者之间的主要协议差异是 `contact_latent_mode`。

## 4. 完整性与可比性

确认测量结果：

- 10 个实验均包含 `grid_summary.csv`、`random_position_summary.json`、`grid_manifest.json` 与 `task_points.csv`。
- 每组均完成 50/50 points，`completion_rate=100%`。
- 10 组 `process_error_runs` 均为 0。
- 所有 task failure 的 `stop_reason` 均为 `max_rollout_steps`。
- 10 份 task points 的 `point_index` 与 x/y/z offset 完全一致，最大绝对差为 0。
- 每组均已有 safe-success publication PNG/PDF 和 labeled diagnostic PNG。

因此，同配置 `mid/temporal` 以及 Contact-CVAE `zero/prior` 均可进行直接逐点配对比较。跨 checkpoint 的比较仍属于描述性比较，不能作为单因素因果结论。

## 5. 总体任务成功与安全成功

95% CI 为 task-success rate 的 Wilson 区间，所有比率分母均为 50。

| experiment | task success | task rate | Wilson 95% CI | safe success | safe rate | task failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero-mid | 32/50 | 64.0% | 50.1%-75.9% | 32/50 | 64.0% | 18 |
| Contact-CVAE-zero-temporal | 14/50 | 28.0% | 17.5%-41.7% | 14/50 | 28.0% | 36 |
| Contact-CVAE-prior-mid | 24/50 | 48.0% | 34.8%-61.5% | 24/50 | 48.0% | 26 |
| Contact-CVAE-prior-temporal | 11/50 | 22.0% | 12.8%-35.2% | 6/50 | 12.0% | 39 |
| Motion-CVAE-mid | 29/50 | 58.0% | 44.2%-70.6% | 29/50 | 58.0% | 21 |
| Motion-CVAE-temporal | 7/50 | 14.0% | 7.0%-26.2% | 7/50 | 14.0% | 43 |
| DualZero-mid | 22/50 | 44.0% | 31.2%-57.7% | 20/50 | 40.0% | 28 |
| DualZero-temporal | 17/50 | 34.0% | 22.4%-47.8% | 14/50 | 28.0% | 33 |
| ACT baseline-mid | 28/50 | 56.0% | 42.3%-68.8% | 17/50 | 34.0% | 22 |
| ACT baseline-temporal | 20/50 | 40.0% | 27.6%-53.8% | 12/50 | 24.0% | 30 |

确认测量结果：`Contact-CVAE-zero-mid` 的 task success 与 safe success 均最高，为 32/50（64.0%）。Motion-CVAE-mid 的 safe success 为 29/50（58.0%），排名第二。

Contact-CVAE-zero 和 Motion-CVAE 的所有 task success 均为 safe success。Contact-CVAE-prior-temporal 有 11 个 task success，但仅 6 个 safe success；DualZero-mid、DualZero-temporal、ACT baseline-mid、ACT baseline-temporal 分别有 2、3、11、8 个成功点超过 safe-success 条件。

## 6. 靶心空间分布

所有 target map 使用相同的 ±6 mm 显示范围与 2 mm rings。绿色为 safe success，橙色为 task success 但非 safe success，红色为 failure；标题显示 safe-success 数量和比率。

| experiment | publication map | labeled diagnostic map |
| --- | --- | --- |
| Contact-CVAE-zero-mid | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success_labeled.png) |
| Contact-CVAE-zero-temporal | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_zero_temporal_d03_dq002_maxsteps900/plots/contact_cvae100k_zero_temporal_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_zero_temporal_d03_dq002_maxsteps900/plots/contact_cvae100k_zero_temporal_4mm_target_safe_success_labeled.png) |
| Contact-CVAE-prior-mid | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_prior_mid_dq002_maxsteps900/plots/contact_cvae100k_prior_mid_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_prior_mid_dq002_maxsteps900/plots/contact_cvae100k_prior_mid_4mm_target_safe_success_labeled.png) |
| Contact-CVAE-prior-temporal | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_prior_temporal_d03_dq002_maxsteps900/plots/contact_cvae100k_prior_temporal_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_prior_temporal_d03_dq002_maxsteps900/plots/contact_cvae100k_prior_temporal_4mm_target_safe_success_labeled.png) |
| Motion-CVAE-mid | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_4mm_target_safe_success_labeled.png) |
| Motion-CVAE-temporal | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/plots/motion_cvae100k_temporal_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/plots/motion_cvae100k_temporal_4mm_target_safe_success_labeled.png) |
| DualZero-mid | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_4mm_target_safe_success_labeled.png) |
| DualZero-temporal | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_dualzero100k_temporal_d03_dq002_maxsteps900/plots/dualzero100k_temporal_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_dualzero100k_temporal_d03_dq002_maxsteps900/plots/dualzero100k_temporal_4mm_target_safe_success_labeled.png) |
| ACT baseline-mid | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success_labeled.png) |
| ACT baseline-temporal | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_act_baseline100k_temporal_d03_dq002_maxsteps900/plots/act_baseline100k_temporal_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_4mm_act_baseline100k_temporal_d03_dq002_maxsteps900/plots/act_baseline100k_temporal_4mm_target_safe_success_labeled.png) |

目视检查确认图中名义孔居中、x/z 比例一致、采样点未裁切，safe/task/failure 图例计数与 CSV 一致。

## 7. 成功率随半径的变化

定义：

```text
radius_mm = 1000 * sqrt(hole_offset_x^2 + hole_offset_z^2)
```

共享点集中 `[0,2)`、`[2,4)`、`[4,6)` mm 分别有 11、29、10 个点。下表为 `task successes / points`，括号内为 safe successes。

| experiment | [0,2) mm | [2,4) mm | [4,6) mm | largest task-success radius |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero-mid | 11/11 (11) | 21/29 (21) | 0/10 (0) | 3.98 mm |
| Contact-CVAE-zero-temporal | 8/11 (8) | 6/29 (6) | 0/10 (0) | 3.98 mm |
| Contact-CVAE-prior-mid | 11/11 (11) | 13/29 (13) | 0/10 (0) | 3.84 mm |
| Contact-CVAE-prior-temporal | 8/11 (4) | 3/29 (2) | 0/10 (0) | 3.15 mm |
| Motion-CVAE-mid | 10/11 (10) | 19/29 (19) | 0/10 (0) | 3.98 mm |
| Motion-CVAE-temporal | 4/11 (4) | 3/29 (3) | 0/10 (0) | 3.81 mm |
| DualZero-mid | 11/11 (11) | 11/29 (9) | 0/10 (0) | 3.91 mm |
| DualZero-temporal | 11/11 (10) | 6/29 (4) | 0/10 (0) | 3.98 mm |
| ACT baseline-mid | 9/11 (4) | 16/29 (10) | 3/10 (3) | 5.04 mm |
| ACT baseline-temporal | 7/11 (5) | 10/29 (7) | 3/10 (0) | 5.04 mm |

结果表明，除 ACT baseline 外，其余配置在 `[4,6)` mm 没有 task success。ACT baseline-mid 与 temporal 均在该 bin 有 3/10 task success；其中 mid 的 3 个也是 safe success，temporal 的 3 个均不是 safe success。

所有实验在近中心区域也存在失败，因此半径不能单独解释结果。最大观测成功半径只描述本次点集，不能称为绝对泛化极限。

## 8. 方向与象限分析

### 8.1 半轴结果

| experiment | x < 0 | x >= 0 | z < 0 | z >= 0 |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero-mid | 19/25 | 13/25 | 17/25 | 15/25 |
| Contact-CVAE-zero-temporal | 10/25 | 4/25 | 9/25 | 5/25 |
| Contact-CVAE-prior-mid | 13/25 | 11/25 | 14/25 | 10/25 |
| Contact-CVAE-prior-temporal | 7/25 | 4/25 | 7/25 | 4/25 |
| Motion-CVAE-mid | 18/25 | 11/25 | 14/25 | 15/25 |
| Motion-CVAE-temporal | 4/25 | 3/25 | 3/25 | 4/25 |
| DualZero-mid | 14/25 | 8/25 | 9/25 | 13/25 |
| DualZero-temporal | 12/25 | 5/25 | 8/25 | 9/25 |
| ACT baseline-mid | 8/25 | 20/25 | 19/25 | 9/25 |
| ACT baseline-temporal | 4/25 | 16/25 | 12/25 | 8/25 |

ACT baseline 的方向不对称在 ±4 mm 下仍然存在。Fisher exact test：

- ACT baseline-mid：x 方向 `p=0.001442`，z 方向 `p=0.009595`；
- ACT baseline-temporal：x 方向 `p=0.001209`，z 方向 `p=0.386845`。

其他配置的 x/z 半轴 Fisher 检验在本批均未达到 `p<0.05`。

### 8.2 象限结果

| experiment | +x +z | -x +z | -x -z | +x -z |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero-mid | 7/12 | 8/13 | 11/12 | 6/13 |
| Contact-CVAE-zero-temporal | 4/12 | 1/13 | 9/12 | 0/13 |
| Contact-CVAE-prior-mid | 5/12 | 5/13 | 8/12 | 6/13 |
| Contact-CVAE-prior-temporal | 3/12 | 1/13 | 6/12 | 1/13 |
| Motion-CVAE-mid | 7/12 | 8/13 | 10/12 | 4/13 |
| Motion-CVAE-temporal | 2/12 | 2/13 | 2/12 | 1/13 |
| DualZero-mid | 6/12 | 7/13 | 7/12 | 2/13 |
| DualZero-temporal | 4/12 | 5/13 | 7/12 | 1/13 |
| ACT baseline-mid | 7/12 | 2/13 | 6/12 | 13/13 |
| ACT baseline-temporal | 8/12 | 0/13 | 4/12 | 8/13 |

ACT baseline-mid 在 `+x,-z` 仍为 13/13 task success，而 `-x,+z` 仅为 2/13；temporal 在 `-x,+z` 为 0/13。该现象是实测方向计数，但其机制仍需镜像点重复实验和 trajectory/video 分析。

## 9. Mid与Temporal逐点比较

### 9.1 Task success

| configuration | both success | mid only | temporal only | both fail | rate difference | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 14 | 18 | 0 | 18 | +36 pp | 0.000008 |
| Contact-CVAE-prior | 11 | 13 | 0 | 26 | +26 pp | 0.000244 |
| Motion-CVAE | 7 | 22 | 0 | 21 | +44 pp | <0.000001 |
| DualZero | 14 | 8 | 3 | 25 | +10 pp | 0.226562 |
| ACT baseline | 17 | 11 | 3 | 19 | +16 pp | 0.057373 |

### 9.2 Safe success

| configuration | both safe | mid only | temporal only | neither safe | rate difference | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 14 | 18 | 0 | 18 | +36 pp | 0.000008 |
| Contact-CVAE-prior | 6 | 18 | 0 | 26 | +36 pp | 0.000008 |
| Motion-CVAE | 7 | 22 | 0 | 21 | +44 pp | <0.000001 |
| DualZero | 12 | 8 | 2 | 28 | +12 pp | 0.109375 |
| ACT baseline | 9 | 8 | 3 | 30 | +10 pp | 0.226562 |

统计结果：五种配置的 `mid` task/safe-success rate 均高于对应 temporal。Contact-CVAE-zero、Contact-CVAE-prior 和 Motion-CVAE 的配对差异达到 `p<0.05`；DualZero 与 ACT baseline 在本样本上未达到该阈值。

## 10. Contact-CVAE Zero与Prior比较

该比较使用相同 checkpoint、相同 task points 和相同 action selection mode，主要改变 `contact_latent_mode`。

| action mode | metric | both | zero only | prior only | neither | zero-prior difference | exact p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mid | task success | 23 | 9 | 1 | 17 | +16 pp | 0.021484 |
| mid | safe success | 23 | 9 | 1 | 17 | +16 pp | 0.021484 |
| temporal | task success | 10 | 4 | 1 | 35 | +6 pp | 0.375000 |
| temporal | safe success | 6 | 8 | 0 | 36 | +16 pp | 0.007812 |

确认测量结果：本批中 prior 没有提高 Contact-CVAE 的总体成功数。mid 下 zero 为 32/50，prior 为 24/50；temporal 下 zero 为 14/50，prior 为 11/50。prior-temporal 的 11 个 task success 中有 5 个不满足 safe success，使 safe rate 降至 6/50。

统计结果支持 zero-mid 在本点集上优于 prior-mid，并支持 zero-temporal 的 safe success 多于 prior-temporal。可能原因包括 prior latent 与在线观测、训练 posterior 或动作解码之间存在分布偏差，但这只是机制假设，仍需分析 `z_contact`、`prior_vs_zero_action_mean_abs_diff` 与逐步力/动作轨迹。

## 11. 力与完成时间

下表报告所有 rollout、成功 rollout 的 `max_force` 均值/中位数/最大值，以及成功时间均值/中位数/最大值。

| experiment | all max force (N) | success max force (N) | success time (s) |
| --- | --- | --- | --- |
| Contact-CVAE-zero-mid | 28.72 / 27.99 / 47.18 | 24.66 / 26.28 / 41.25 | 8.31 / 6.81 / 29.63 |
| Contact-CVAE-zero-temporal | 42.40 / 34.06 / 108.17 | 37.28 / 28.41 / 64.66 | 17.78 / 17.54 / 25.64 |
| Contact-CVAE-prior-mid | 54.46 / 50.08 / 103.64 | 37.28 / 37.67 / 78.42 | 6.65 / 6.39 / 9.60 |
| Contact-CVAE-prior-temporal | 71.36 / 74.48 / 108.84 | 68.39 / 59.09 / 108.26 | 19.44 / 19.80 / 27.06 |
| Motion-CVAE-mid | 47.12 / 52.10 / 90.63 | 40.96 / 47.23 / 72.81 | 9.97 / 7.43 / 25.54 |
| Motion-CVAE-temporal | 36.94 / 35.10 / 102.30 | 39.97 / 41.47 / 62.66 | 21.83 / 21.81 / 25.67 |
| DualZero-mid | 63.33 / 68.09 / 140.36 | 38.85 / 31.33 / 91.70 | 8.95 / 6.73 / 22.67 |
| DualZero-temporal | 59.34 / 59.15 / 107.17 | 39.36 / 32.13 / 88.45 | 17.52 / 16.86 / 23.89 |
| ACT baseline-mid | 72.43 / 98.85 / 107.12 | 51.03 / 31.66 / 106.55 | 9.29 / 6.20 / 22.70 |
| ACT baseline-temporal | 81.65 / 87.23 / 106.01 | 60.96 / 75.02 / 104.31 | 18.53 / 16.32 / 28.88 |

确认测量结果：

- Contact-CVAE-zero-mid 的 all-run mean max force 最低，为 28.72 N。
- Contact-CVAE-prior-temporal 的成功样本 mean max force 为 68.39 N，5/11 task success 超过 safe-success 条件。
- DualZero-mid 的单次最大力达到 140.36 N，但仍远低于 1000 N hard-stop threshold；该点不是安全成功。
- 五组 temporal 的成功时间中位数均高于对应 mid，但部分 temporal 的成功样本很少，应谨慎解释。

## 12. 失败类型

10 组实验共有 296 个 task failures，全部为 `stop_reason=max_rollout_steps`，即正常执行到 900 步但未持续满足成功条件。本批没有：

- process error；
- hard force stop；
- 非有限值 termination；
- 缺失 stop reason 的 unknown failure。

summary-level 数据不足以把失败进一步解释为“未搜索”“未退让”或“卡住”。这些行为级标签必须由 `rollout_log.csv` 与视频支持。

## 13. 代表性案例

| experiment | largest-radius success | smallest-radius failure | highest-force success |
| --- | --- | --- | --- |
| Contact-CVAE-zero-mid | point 45, 3.98 mm | point 30, 2.29 mm | point 36, 41.25 N |
| Contact-CVAE-zero-temporal | point 45, 3.98 mm | point 26, 0.94 mm | point 23, 64.66 N |
| Contact-CVAE-prior-mid | point 12, 3.84 mm | point 25, 2.08 mm | point 12, 78.42 N |
| Contact-CVAE-prior-temporal | point 50, 3.15 mm | point 23, 1.21 mm | point 28, 108.26 N |
| Motion-CVAE-mid | point 45, 3.98 mm | point 49, 1.52 mm | point 36, 72.81 N |
| Motion-CVAE-temporal | point 43, 3.81 mm | point 9, 0.37 mm | point 34, 62.66 N |
| DualZero-mid | point 11, 3.91 mm | point 30, 2.29 mm | point 27, 91.70 N |
| DualZero-temporal | point 45, 3.98 mm | point 25, 2.08 mm | point 35, 88.45 N |
| ACT baseline-mid | point 2, 5.04 mm | point 24, 1.80 mm | point 36, 106.55 N |
| ACT baseline-temporal | point 2, 5.04 mm | point 47, 1.27 mm | point 28, 104.31 N |

优先建议检查：

- Contact-CVAE-prior-temporal point 28：task success、`max_force=108.26 N`，不是 safe success；
- Contact-CVAE zero/prior 的 paired transition 点：用于判断 prior 对动作与力预测的影响；
- ACT baseline-mid point 2：本批最大观测成功半径 5.04 mm；
- Motion-CVAE-temporal point 9：距中心仅 0.37 mm 仍失败，说明近中心不保证成功。

## 14. 主要发现

1. **确认测量结果**：10 个实验全部完成，共 500 次 rollout，0 process error，且 task points 完全一致。
2. **总体结果**：Contact-CVAE-zero-mid 的 task/safe success 最高，均为 32/50（64.0%）。
3. **执行模式**：五种配置的 mid 均高于对应 temporal；其中 Contact-CVAE zero/prior 与 Motion-CVAE 的配对差异最明确。
4. **Contact latent**：在本批数据中，prior 未优于 zero；prior-temporal 还出现 5 个 task success 但非 safe success。
5. **空间结果**：除 ACT baseline 外，其他配置在 `[4,6)` mm 无成功点。
6. **方向结果**：ACT baseline 的 `+x,-z` 优势在 ±4 mm 仍存在，说明此前 ±6 mm 观察到的不对称并未消失。
7. **安全结果**：task success 与 safe success 必须分开报告，尤其是 Contact-CVAE-prior-temporal 和 ACT baseline。

## 15. 解释与局限性

数据支持 mid、contact-latent mode、半径和方向与结果变化有关，但不能单独确定物理机制。prior 表现较低可能与 prior latent 的估计偏差有关；ACT baseline 的方向差异可能来自训练分布、初始几何、运动学或动作执行约束。这些解释都需要 trajectory、latent 和视频验证。

主要局限性：

1. 每个条件仅有一组 50 点 LHS，没有跨 seed 重复；
2. 稀疏点集不能推断连续成功区域；
3. 方向检验是探索性的；
4. prior/zero 虽共享 checkpoint，但逐步 latent 会改变动作和力预测，仍需过程分析；
5. 跨模型比较同时改变 checkpoint 与 policy variant；
6. safe success 是 80 N 阈值下的实验定义，不是硬件安全证明。

## 16. 后续建议

1. 对 Contact-CVAE zero/prior 的 discordant points 分析 `prior_vs_zero_action_mean_abs_diff` 和 `prior_vs_zero_force_mean_abs_diff`；
2. 保存并比较 `mu_contact_prior`、`logvar_contact_prior` 与 `z_contact` 的时序变化；
3. 对 ACT baseline 进行严格镜像点和多 seed 重复实验；
4. 对 task success 但非 safe success 的点定位峰值力发生阶段；
5. 保持相同 task points，对 4/6/10 mm 结果进行分层比较，不把不同点集误当作同点配对。

## 17. 结论

本批 ±4 mm 实验完整且可逐点比较。Contact-CVAE-zero-mid 获得最高 task/safe-success rate；Contact-CVAE prior 在当前 checkpoint 与协议下没有带来成功率提升，prior-temporal 还表现出更明显的 task/safe-success 分离。mid 整体优于 temporal，ACT baseline 的方向不对称在较小 offset 范围内仍然存在。

所有结论仅适用于当前 50 个 LHS 实测点，不代表连续空间成功区域或确定泛化边界。

## 附录A：实验路径

| experiment | experiment root |
| --- | --- |
| Contact-CVAE-zero-mid | `outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900` |
| Contact-CVAE-zero-temporal | `outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_zero_temporal_d03_dq002_maxsteps900` |
| Contact-CVAE-prior-mid | `outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_prior_mid_dq002_maxsteps900` |
| Contact-CVAE-prior-temporal | `outputs/peg_hole_100/hole_lhs_50_xz_4mm_contact_cvae100k_prior_temporal_d03_dq002_maxsteps900` |
| Motion-CVAE-mid | `outputs/peg_hole_100/hole_lhs_50_xz_4mm_motion_cvae100k_mid_dq002_maxsteps900` |
| Motion-CVAE-temporal | `outputs/peg_hole_100/hole_lhs_50_xz_4mm_motion_cvae100k_temporal_d03_dq002_maxsteps900` |
| DualZero-mid | `outputs/peg_hole_100/hole_lhs_50_xz_4mm_dualzero100k_mid_dq002_maxsteps900` |
| DualZero-temporal | `outputs/peg_hole_100/hole_lhs_50_xz_4mm_dualzero100k_temporal_d03_dq002_maxsteps900` |
| ACT baseline-mid | `outputs/peg_hole_100/hole_lhs_50_xz_4mm_act_baseline100k_mid_dq002_maxsteps900` |
| ACT baseline-temporal | `outputs/peg_hole_100/hole_lhs_50_xz_4mm_act_baseline100k_temporal_d03_dq002_maxsteps900` |

## 附录B：数据字段

所有 `grid_summary.csv` 使用一致 schema：

```text
point_index, sampling_mode, base_seed,
hole_offset_x, hole_offset_y, hole_offset_z,
radial_offset, quadrant, success, safe_success,
success_step, success_time, stop_reason,
final_dist, final_lateral, final_axial,
max_force, mean_force, force_gt_20_steps, force_gt_40_steps,
output_dir, summary_json, rollout_log_csv
```

报告中的关键计数均由 `grid_summary.csv` 计算，并与 `random_position_summary.json` 交叉核对。
