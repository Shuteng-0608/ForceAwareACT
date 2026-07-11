# ForceAwareACT ±4 mm孔位扰动Rollout实验报告（New Goal）

## 1. 实验背景与目的

本报告分析 `outputs/peg_hole_100/new_goal` 下的新一批 50 点 x/z ±4 mm 孔位扰动 rollout。实验覆盖 Contact-CVAE zero、Contact-CVAE prior、Motion-CVAE、DualZero 与 ACT baseline 五种策略/latent 配置，每种配置包含 `mid` 和 `temporal` 两种 action selection mode，共 10 个实验、500 次 rollout。

本批实验与此前 ±4 mm 实验的已知主要差异是 MuJoCo 模型中的 `hole_goal_site` 距离发生了调整。

> **待用户补充的几何配置**
>
> - 原 `hole_goal_site` 距离：`-0.016`
> - 新 `hole_goal_site` 距离：`-0.021`
> - 原 `peg_size` 距离：`0.010`
> - 新 `peg_size` 距离：`0.011`
> - 距离变化量及方向：`[待补充]`
> - 对应 XML 参数或 site position：`[待补充]`

manifest 只保存模型 XML 路径，不保存 XML 内容快照，因此无法从 rollout 结果可靠恢复上述具体数值。本报告不会猜测距离变化量。

## 2. 数据范围与术语

纳入目录为：

```text
outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_*
```

术语定义：

- `task success`：`grid_summary.csv` 中 `success=True`。
- `safe success`：`safe_success=True`，即 task success 同时满足本批配置的 `success_force_threshold=80 N`。
- `task success, not safe`：任务成功，但未满足 safe-success 定义。
- `process error`：rollout 进程没有正常完成。
- `old goal`：此前保存在 `outputs/peg_hole_100/hole_lhs_50_xz_4mm_*` 的实验。
- `new goal`：本报告分析的 `outputs/peg_hole_100/new_goal` 实验。

`safe success` 是本实验阈值定义，不是通用硬件安全保证。

## 3. 实验协议与已知差异

10 个 new-goal 实验共享：

| setting | value |
| --- | --- |
| sampling mode | `latin_hypercube` |
| points / repeats | 50 / 1 |
| base seed | `20260702` |
| x/z bounds | `[-0.004, 0.004] m` |
| y offset | `0 m` |
| action mode | `action` |
| chunk length | 10 |
| policy rate | 30 Hz |
| maximum rollout steps | 900 |
| maximum delta q | 0.02 rad |
| force stop threshold | 1000 N |
| success distance threshold | 0.005 m |
| success lateral threshold | 0.006 m |
| success force threshold | 80 N |
| success hold steps | 15 |
| hole site / body | `hole_goal_site` / `wall_task` |
| offset frame | `world` |

与旧 4 mm 实验比较时，已知改变项为 `hole_goal_site` 距离。由于没有保存旧/新 XML 快照及精确 site position，本报告不能独立证明除此以外所有模型几何参数都完全相同。后续应将 XML 版本或 hash 写入 manifest。

## 4. 完整性与可比性

确认测量结果：

- 10 个实验均包含 `grid_summary.csv`、`random_position_summary.json`、`grid_manifest.json` 和 `task_points.csv`。
- 每个实验均完成 50/50 points，completion rate 为 100%。
- 共 500 次 rollout，process error 为 0。
- 所有 task failure 的 `stop_reason` 均为 `max_rollout_steps`。
- new-goal 10 组实验使用完全相同的 50 个 task points。
- new goal 与 old goal 的 `point_index` 和 x/y/z offsets 也完全相同。
- 每组均有 safe-success publication PNG/PDF 和 labeled PNG。

因此，new-goal 内部以及相同配置的 old/new goal 可以逐点配对。old/new 的性能变化与 goal-site 几何调整同时发生，但不应在缺少额外控制实验时写成确定因果。

## 5. New Goal总体结果

95% CI 为 task-success rate 的 Wilson 区间。

| experiment | task success | task rate | Wilson 95% CI | safe success | safe rate | task failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero-mid | 18/50 | 36.0% | 24.1%-49.9% | 18/50 | 36.0% | 32 |
| Contact-CVAE-zero-temporal | 7/50 | 14.0% | 7.0%-26.2% | 7/50 | 14.0% | 43 |
| Contact-CVAE-prior-mid | 20/50 | 40.0% | 27.6%-53.8% | 18/50 | 36.0% | 30 |
| Contact-CVAE-prior-temporal | 9/50 | 18.0% | 9.8%-30.8% | 5/50 | 10.0% | 41 |
| Motion-CVAE-mid | 21/50 | 42.0% | 29.4%-55.8% | 17/50 | 34.0% | 29 |
| Motion-CVAE-temporal | 6/50 | 12.0% | 5.6%-23.8% | 5/50 | 10.0% | 44 |
| DualZero-mid | 16/50 | 32.0% | 20.8%-45.8% | 15/50 | 30.0% | 34 |
| DualZero-temporal | 7/50 | 14.0% | 7.0%-26.2% | 7/50 | 14.0% | 43 |
| ACT baseline-mid | 22/50 | 44.0% | 31.2%-57.7% | 10/50 | 20.0% | 28 |
| ACT baseline-temporal | 9/50 | 18.0% | 9.8%-30.8% | 4/50 | 8.0% | 41 |

确认测量结果：

- task success 最高的是 `ACT baseline-mid`：22/50（44.0%）。
- safe success 最高的是 `Contact-CVAE-zero-mid` 与 `Contact-CVAE-prior-mid`：均为 18/50（36.0%）。
- Contact-CVAE-zero 的所有 task success 均为 safe success。
- ACT baseline-mid 虽有 22 个 task success，但仅 10 个 safe success。
- 10 组共计 135 个 task success、106 个 safe success和 365 个 task failures。

## 6. Target Map

所有图采用相同的 ±6 mm 显示范围和 2 mm rings。绿色为 safe success，橙色为 task success 但非 safe success，红色为 failure。

| experiment | publication map | labeled map |
| --- | --- | --- |
| Contact-CVAE-zero-mid | [PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success_labeled.png) |
| Contact-CVAE-zero-temporal | [PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_contact_cvae100k_zero_temporal_d03_dq002_maxsteps900/plots/contact_cvae100k_zero_temporal_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_contact_cvae100k_zero_temporal_d03_dq002_maxsteps900/plots/contact_cvae100k_zero_temporal_4mm_target_safe_success_labeled.png) |
| Contact-CVAE-prior-mid | [PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_contact_cvae100k_prior_mid_dq002_maxsteps900/plots/contact_cvae100k_prior_mid_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_contact_cvae100k_prior_mid_dq002_maxsteps900/plots/contact_cvae100k_prior_mid_4mm_target_safe_success_labeled.png) |
| Contact-CVAE-prior-temporal | [PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_contact_cvae100k_prior_temporal_d03_dq002_maxsteps900/plots/contact_cvae100k_prior_temporal_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_contact_cvae100k_prior_temporal_d03_dq002_maxsteps900/plots/contact_cvae100k_prior_temporal_4mm_target_safe_success_labeled.png) |
| Motion-CVAE-mid | [PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_4mm_target_safe_success_labeled.png) |
| Motion-CVAE-temporal | [PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/plots/motion_cvae100k_temporal_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/plots/motion_cvae100k_temporal_4mm_target_safe_success_labeled.png) |
| DualZero-mid | [PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_4mm_target_safe_success_labeled.png) |
| DualZero-temporal | [PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_dualzero100k_temporal_d03_dq002_maxsteps900/plots/dualzero100k_temporal_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_dualzero100k_temporal_d03_dq002_maxsteps900/plots/dualzero100k_temporal_4mm_target_safe_success_labeled.png) |
| ACT baseline-mid | [PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success_labeled.png) |
| ACT baseline-temporal | [PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_act_baseline100k_temporal_d03_dq002_maxsteps900/plots/act_baseline100k_temporal_4mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/new_goal/hole_lhs_50_xz_4mm_act_baseline100k_temporal_d03_dq002_maxsteps900/plots/act_baseline100k_temporal_4mm_target_safe_success_labeled.png) |

这些图只显示离散实测点，不表示连续成功区域。

## 7. 半径分析

共享点集中 `[0,2)`、`[2,4)`、`[4,6)` mm 分别有 11、29、10 个点。表中为 `task successes / points`，括号内为 safe successes。

| experiment | [0,2) mm | [2,4) mm | [4,6) mm | largest task-success radius |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero-mid | 10/11 (10) | 8/29 (8) | 0/10 (0) | 3.46 mm |
| Contact-CVAE-zero-temporal | 6/11 (6) | 1/29 (1) | 0/10 (0) | 2.06 mm |
| Contact-CVAE-prior-mid | 8/11 (8) | 12/29 (10) | 0/10 (0) | 3.84 mm |
| Contact-CVAE-prior-temporal | 7/11 (4) | 2/29 (1) | 0/10 (0) | 3.15 mm |
| Motion-CVAE-mid | 10/11 (10) | 11/29 (7) | 0/10 (0) | 3.39 mm |
| Motion-CVAE-temporal | 3/11 (3) | 3/29 (2) | 0/10 (0) | 3.81 mm |
| DualZero-mid | 11/11 (11) | 5/29 (4) | 0/10 (0) | 3.15 mm |
| DualZero-temporal | 6/11 (6) | 1/29 (1) | 0/10 (0) | 2.06 mm |
| ACT baseline-mid | 6/11 (1) | 13/29 (7) | 3/10 (2) | 5.04 mm |
| ACT baseline-temporal | 5/11 (0) | 4/29 (4) | 0/10 (0) | 3.72 mm |

除 ACT baseline-mid 外，其他配置在 `[4,6)` mm 均无 task success。多个配置在 `[0,2)` mm 仍有失败点，因此半径不是唯一决定因素。

## 8. 方向与象限

### 8.1 半轴结果

| experiment | x < 0 | x >= 0 | z < 0 | z >= 0 |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero-mid | 10/25 | 8/25 | 11/25 | 7/25 |
| Contact-CVAE-zero-temporal | 5/25 | 2/25 | 3/25 | 4/25 |
| Contact-CVAE-prior-mid | 12/25 | 8/25 | 13/25 | 7/25 |
| Contact-CVAE-prior-temporal | 6/25 | 3/25 | 5/25 | 4/25 |
| Motion-CVAE-mid | 12/25 | 9/25 | 9/25 | 12/25 |
| Motion-CVAE-temporal | 4/25 | 2/25 | 5/25 | 1/25 |
| DualZero-mid | 9/25 | 7/25 | 7/25 | 9/25 |
| DualZero-temporal | 4/25 | 3/25 | 4/25 | 3/25 |
| ACT baseline-mid | 3/25 | 19/25 | 16/25 | 6/25 |
| ACT baseline-temporal | 3/25 | 6/25 | 4/25 | 5/25 |

ACT baseline-mid 仍呈现显著方向不对称：x 方向 Fisher exact `p=0.000010`，z 方向 `p=0.009595`。其 `+x,-z` 象限为 13/13 task success，`-x,+z` 为 0/13。其他配置的 x/z 半轴检验在本批没有达到 `p<0.05`。

该方向性是实测结果，但不能仅凭 summary 判定其物理机制。

## 9. Mid与Temporal逐点比较

### 9.1 Task success

| configuration | both success | mid only | temporal only | both fail | rate difference | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 7 | 11 | 0 | 32 | +22 pp | 0.000977 |
| Contact-CVAE-prior | 8 | 12 | 1 | 29 | +22 pp | 0.003418 |
| Motion-CVAE | 4 | 17 | 2 | 27 | +30 pp | 0.000729 |
| DualZero | 7 | 9 | 0 | 34 | +18 pp | 0.003906 |
| ACT baseline | 6 | 16 | 3 | 25 | +26 pp | 0.004425 |

### 9.2 Safe success

| configuration | both safe | mid only | temporal only | neither safe | rate difference | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 7 | 11 | 0 | 32 | +22 pp | 0.000977 |
| Contact-CVAE-prior | 5 | 13 | 0 | 32 | +26 pp | 0.000244 |
| Motion-CVAE | 3 | 14 | 2 | 31 | +24 pp | 0.004181 |
| DualZero | 7 | 8 | 0 | 35 | +16 pp | 0.007812 |
| ACT baseline | 3 | 7 | 1 | 39 | +12 pp | 0.070312 |

new goal 下五种配置的 mid task-success rate 均高于对应 temporal。除 ACT baseline safe success 外，其余表中配对差异达到 `p<0.05`。

## 10. Contact-CVAE Zero与Prior

| action mode | metric | both | zero only | prior only | neither | zero-prior difference | exact p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mid | task success | 14 | 4 | 6 | 26 | -4 pp | 0.753906 |
| mid | safe success | 14 | 4 | 4 | 28 | 0 pp | 1.000000 |
| temporal | task success | 7 | 0 | 2 | 41 | -4 pp | 0.500000 |
| temporal | safe success | 3 | 4 | 2 | 41 | +4 pp | 0.687500 |

本批没有证据表明 zero 与 prior 的总体差异显著。prior-mid 的 task success 多 2 点，但 safe success 与 zero-mid 相同；prior-temporal 的 task success 多 2 点，但 safe success 少 2 点。

## 11. 力与完成时间

数值格式为均值 / 中位数 / 最大值。

| experiment | all max force (N) | success max force (N) | success time (s) |
| --- | --- | --- | --- |
| Contact-CVAE-zero-mid | 32.46 / 31.97 / 59.74 | 27.89 / 28.61 / 36.36 | 7.23 / 7.24 / 9.47 |
| Contact-CVAE-zero-temporal | 38.67 / 33.16 / 104.42 | 45.86 / 46.56 / 66.64 | 18.69 / 17.62 / 29.27 |
| Contact-CVAE-prior-mid | 60.72 / 64.50 / 99.29 | 49.73 / 48.41 / 85.64 | 7.82 / 6.81 / 20.66 |
| Contact-CVAE-prior-temporal | 70.43 / 73.88 / 109.11 | 68.97 / 75.48 / 95.84 | 19.61 / 20.36 / 28.48 |
| Motion-CVAE-mid | 52.41 / 59.24 / 91.54 | 42.96 / 40.21 / 91.54 | 9.92 / 8.15 / 23.36 |
| Motion-CVAE-temporal | 35.77 / 29.71 / 103.78 | 35.61 / 25.21 / 103.78 | 21.59 / 21.12 / 26.53 |
| DualZero-mid | 71.47 / 77.76 / 99.31 | 55.22 / 66.80 / 91.92 | 8.34 / 7.11 / 27.75 |
| DualZero-temporal | 64.39 / 56.78 / 107.99 | 43.46 / 48.79 / 76.40 | 16.34 / 16.33 / 25.94 |
| ACT baseline-mid | 90.20 / 99.23 / 326.34 | 68.34 / 93.94 / 107.96 | 11.57 / 6.44 / 23.20 |
| ACT baseline-temporal | 91.60 / 94.11 / 108.16 | 87.89 / 96.78 / 107.15 | 20.34 / 23.33 / 28.61 |

ACT baseline-mid 的 all-run 最大力达到 326.34 N，但仍低于 1000 N hard-stop threshold；该最大力点不是 task success。task success 与 safe success 的明显分离说明不能只报告任务完成率。

## 12. 与旧4 mm实验的直接比较

old/new goal 使用相同 task points，可以逐点比较。下表聚焦 safe success：

| experiment | old safe | new safe | change | old only | new only | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero-mid | 32 | 18 | -28 pp | 15 | 1 | 0.000519 |
| Contact-CVAE-zero-temporal | 14 | 7 | -14 pp | 8 | 1 | 0.039062 |
| Contact-CVAE-prior-mid | 24 | 18 | -12 pp | 8 | 2 | 0.109375 |
| Contact-CVAE-prior-temporal | 6 | 5 | -2 pp | 3 | 2 | 1.000000 |
| Motion-CVAE-mid | 29 | 17 | -24 pp | 13 | 1 | 0.001831 |
| Motion-CVAE-temporal | 7 | 5 | -4 pp | 4 | 2 | 0.687500 |
| DualZero-mid | 20 | 15 | -10 pp | 5 | 0 | 0.062500 |
| DualZero-temporal | 14 | 7 | -14 pp | 7 | 0 | 0.015625 |
| ACT baseline-mid | 17 | 10 | -14 pp | 7 | 0 | 0.015625 |
| ACT baseline-temporal | 12 | 4 | -16 pp | 8 | 0 | 0.007812 |

确认测量结果：new goal 的 safe-success count 在 10 组中均未高于 old goal。降幅最大的配置是 Contact-CVAE-zero-mid（-28 pp），其次是 Motion-CVAE-mid（-24 pp）。

这些变化与 `hole_goal_site` 距离调整共同出现。数据支持“新几何条件下本批 safe-success 表现整体更低”，但不能在没有 XML 差异记录和重复 seed 的情况下把下降完全归因于 goal-site 距离。

## 13. 失败与代表性案例

500 次 rollout 中有 365 个 task failure，全部以 `max_rollout_steps` 结束，无 process error 或 hard force stop。

| experiment | largest-radius success | smallest-radius failure | highest-force success |
| --- | --- | --- | --- |
| Contact-CVAE-zero-mid | point 4, 3.46 mm | point 49, 1.52 mm | point 23, 36.36 N |
| Contact-CVAE-zero-temporal | point 40, 2.06 mm | point 26, 0.94 mm | point 29, 66.64 N |
| Contact-CVAE-prior-mid | point 12, 3.84 mm | point 29, 0.68 mm | point 12, 85.64 N |
| Contact-CVAE-prior-temporal | point 50, 3.15 mm | point 5, 1.17 mm | point 40, 95.84 N |
| Motion-CVAE-mid | point 37, 3.39 mm | point 41, 1.78 mm | point 36, 91.54 N |
| Motion-CVAE-temporal | point 43, 3.81 mm | point 9, 0.37 mm | point 34, 103.78 N |
| DualZero-mid | point 50, 3.15 mm | point 15, 2.21 mm | point 25, 91.92 N |
| DualZero-temporal | point 40, 2.06 mm | point 29, 0.68 mm | point 9, 76.40 N |
| ACT baseline-mid | point 2, 5.04 mm | point 47, 1.27 mm | point 29, 107.96 N |
| ACT baseline-temporal | point 20, 3.72 mm | point 29, 0.68 mm | point 47, 107.15 N |

行为级失败机制仍需 `rollout_log.csv` 和视频验证。

## 14. 主要发现

1. 10 组 new-goal 实验全部完整，共 500 次 rollout、0 process error。
2. task success 最高为 ACT baseline-mid 的 44%；safe success 最高为 Contact-CVAE-zero/prior-mid 的 36%。
3. 五种配置的 mid 均高于对应 temporal。
4. ACT baseline-mid 的明显方向不对称仍然存在。
5. new goal 的 safe-success count 在全部 10 组中均未高于 old goal。
6. 已知主要协议差异是 `hole_goal_site` 距离调整，但具体距离和 XML 差异仍待补充。

## 15. 局限性与后续建议

1. 每个条件仅有一个 seed，无法分离点集与执行随机性。
2. 没有保存 old/new XML 快照或 hash，不能完整审计几何差异。
3. goal-site 距离具体数值尚未写入报告。
4. 50 个 LHS 点不能定义连续成功区域。
5. 方向和 paired p-value 均属于探索性分析。

建议：

- 补充本报告开头的 old/new distance 与 XML site position；
- 将模型 XML 复制或 hash 写入后续 experiment manifest；
- 使用 `run_xz_multiseed_rollout_suite.py` 在 new-goal 条件下运行至少 5 个 seeds；
- 对 old/new discordant points 检查视频、力峰值和成功判据时序；
- 保持 checkpoint、task points、阈值和动作执行参数不变。

## 16. 结论

new-goal ±4 mm 实验完整、可与旧实验逐点比较。新几何条件下，所有配置的 safe-success count 均未高于旧 4 mm 结果，且 task/safe-success 分离在 ACT baseline 和部分 prior/temporal 配置中较明显。由于 `hole_goal_site` 距离变化量尚未记录，当前结论应限定为结果关联，不作确定的几何因果解释。

## 附录：实验根目录

所有实验位于：

```text
outputs/peg_hole_100/new_goal/
```

其中包含 10 个 `hole_lhs_50_xz_4mm_*` 实验根目录及各自的 `plots/` 子目录。
