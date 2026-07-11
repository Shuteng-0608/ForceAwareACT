# Contact-CVAE-zero与ACT baseline十组孔位种子的40N safe-success实验报告

## 1. 实验范围

本报告整理 `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets` 下的 10 组 point-set seed 实验。每组 point-set 包含两个直接可配对配置：

- `Contact-CVAE-zero`，`mid` action selection；
- `ACT baseline`，`mid` action selection。

所有纳入实验均为 `latin_hypercube` 采样、`100` 个 x/z 孔位点、`+/-4 mm` 范围、`rollout_seed_base=31000`。本报告仅使用本地 `grid_summary.csv`、`random_position_summary.json`、`grid_manifest.json` 和 `task_points.csv` 中的实测记录，不插值连续成功区域。

## 2. 40N成功与安全成功定义

本批 manifest 记录的成功阈值为：`success_distance_threshold=0.005 m`、`success_lateral_threshold=0.006 m`、`success_force_threshold=40.0 N`、`success_hold_steps=15`。

报告中：

- `task success`：rollout 达到上述距离、横向误差和瞬时力保持条件；
- `safe success <40N`：`task success=True` 且完整 rollout 历史 `max_force < 40 N`；
- `unsafe task success`：任务成功，但完整 rollout 历史最大力不低于 40N；
- `task failure`：未达到 task success；
- `safe success` 是本实验的操作性指标，不是通用硬件安全保证。

所有汇总均按 `max_force < 40 N` 重新校验，并确认与 CSV 中 `safe_success` 字段一致。

## 3. 完整性检查

- point-set seeds：`20260707, 20260708, 20260709, 20260710, 20260711, 20260712, 20260713, 20260714, 20260715, 20260716`。
- 有效实验根目录：`20` 个。
- 每个实验 `100` 点，共 `2000` 条 rollout 记录。
- `random_position_summary.json` 中 process errors 合计为 `0`。
- 所有 Contact/ACT 成对实验的 `point_index, hole_offset_x, hole_offset_y, hole_offset_z` 完全一致，可进行逐点配对比较。

## 4. 总体结果

| configuration | point sets | points | task success | task Wilson 95% CI | safe success <40N | safe Wilson 95% CI | safe seed mean +/- SD | unsafe task successes | task failures | process errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT baseline | 10 | 1000 | 595/1000 (59.5%) | 56.4%--62.5% | 295/1000 (29.5%) | 26.8%--32.4% | 29.5% +/- 3.7 pp | 300 | 405 | 0 |
| Contact-CVAE-zero | 10 | 1000 | 614/1000 (61.4%) | 58.3%--64.4% | 590/1000 (59.0%) | 55.9%--62.0% | 59.0% +/- 3.2 pp | 24 | 386 | 0 |

确认测量结果：Contact-CVAE-zero 的 pooled 40N safe-success rate 为 `59.0%`，ACT baseline 为 `29.5%`。Contact 的 task-success rate 也高于 ACT，但二者更大的差异体现在 safe-success 口径和高力成功点数量上。

## 5. 每个point-set seed的safe-success结果

下表为每 100 点中的 40N safe-success 数量。

| point_set_seed | Contact-CVAE-zero | ACT baseline | Contact-ACT safe diff |
| --- | --- | --- | --- |
| 20260707 | 62 | 29 | 33 |
| 20260708 | 62 | 33 | 29 |
| 20260709 | 61 | 32 | 29 |
| 20260710 | 63 | 33 | 30 |
| 20260711 | 60 | 29 | 31 |
| 20260712 | 56 | 26 | 30 |
| 20260713 | 60 | 24 | 36 |
| 20260714 | 55 | 29 | 26 |
| 20260715 | 56 | 25 | 31 |
| 20260716 | 55 | 35 | 20 |

结果表明，10 个 point-set seed 上 Contact-CVAE-zero 的 safe-success 数均高于 ACT baseline；最小差距为 `20` 点，最大差距为 `36` 点。

## 6. 逐点配对比较

| metric | paired_points | both_success | contact_only | act_only | both_fail | contact rate | act rate | Contact-ACT | exact McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task | 1000 | 422 | 192 | 173 | 213 | 61.4% | 59.5% | 1.9 pp | 0.346 |
| safe40 | 1000 | 216 | 374 | 79 | 331 | 59.0% | 29.5% | 29.5 pp | 5.97e-47 |

统计结果：在 1000 个配对孔位点上，40N safe-success 的 discordant pairs 为 Contact-only `374`、ACT-only `79`，双侧 exact McNemar/binomial `p=5.97e-47`。这支持 Contact-CVAE-zero 在这批点集上有更多 40N safe-success 点。

## 7. 半径分析

表格单元为 `task successes / safe successes / points`。

| radius_bin_mm | ACT baseline | Contact-CVAE-zero |
| --- | --- | --- |
| [0,2) | 176/99/196 | 191/187/196 |
| [2,4) | 350/161/603 | 394/375/603 |
| [4,6) | 69/35/201 | 29/28/201 |

结果表明，两个模型的成功都主要集中在较小半径区间；但 Contact-CVAE-zero 在 `[0,2)`、`[2,4)` 和 `[4,6)` 三个半径区间均保持更高的 40N safe-success 数量。最大观测成功半径只是离散 LHS 样本中的最大值，不能解释为连续空间泛化边界。

## 8. 方向与象限分析

表格单元为 `task successes / safe successes / points`。

| group | ACT baseline | Contact-CVAE-zero |
| --- | --- | --- |
| +x,+z | 177/80/253 | 126/114/253 |
| +x,-z | 247/199/247 | 144/144/247 |
| -x,+z | 34/1/247 | 140/130/247 |
| -x,-z | 137/15/253 | 204/202/253 |
| x<0 | 171/16/500 | 344/332/500 |
| x>=0 | 424/279/500 | 270/258/500 |
| z<0 | 384/214/500 | 348/346/500 |
| z>=0 | 211/81/500 | 266/244/500 |

确认测量结果：ACT baseline 仍表现出明显方向不对称，尤其在 `x<0` 与部分负 x 象限中 safe-success 数量很低。Contact-CVAE-zero 的方向分布更均衡，但仍需轨迹、接触力方向和视频验证才能解释具体物理机制。

## 9. 力与时间统计

下表为 `max_force` 的 `mean / median / P95 / max`。

| model | subset | count | max_force mean/median/P95/max |
| --- | --- | --- | --- |
| Contact-CVAE-zero | all | 1000 | 28.15 / 28.22 / 46.76 / 92.20 |
| Contact-CVAE-zero | task_success | 614 | 22.99 / 22.44 / 38.30 / 92.20 |
| Contact-CVAE-zero | safe_success_40n | 590 | 22.02 / 22.01 / 34.85 / 39.90 |
| Contact-CVAE-zero | task_failure | 386 | 36.36 / 38.19 / 50.34 / 54.92 |
| Contact-CVAE-zero | unsafe_task_success_40n | 24 | 46.84 / 43.02 / 83.74 / 92.20 |
| ACT baseline | all | 1000 | 71.54 / 97.48 / 105.20 / 108.89 |
| ACT baseline | task_success | 595 | 52.65 / 40.89 / 105.33 / 108.24 |
| ACT baseline | safe_success_40n | 295 | 14.71 / 9.21 / 38.43 / 39.79 |
| ACT baseline | task_failure | 405 | 99.30 / 99.88 / 104.88 / 108.89 |
| ACT baseline | unsafe_task_success_40n | 300 | 89.95 / 100.63 / 106.35 / 108.24 |

结果表明，ACT baseline 中存在更多任务成功但不满足 40N safe-success 的点。该结果是完整 rollout 历史最大力的测量事实；不能单凭汇总表断言模型产生了某种具体接触策略。

## 10. 靶心图

已有 target-map 图位于各实验根目录的 `plots/` 子目录。图中绿色为 40N safe success，橙色为 task success 但非 safe success，红色为 task failure。

示例图：

- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/pointset_20260707/rollout_31000/hole_lhs_100_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success.png`
- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/pointset_20260707/rollout_31000/hole_lhs_100_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success.png`

这些图只显示离散实测 rollout 点，不估计、不插值连续成功区域。

## 11. 代表性案例

代表性案例表已写入：`outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/representative_cases_40n.csv`。该表包括最大半径 safe success、最小半径 task failure、最高力成功/失败、最快/最慢成功，以及 paired safe-success 转移案例。

## 12. 主要发现

1. **确认测量结果：**10 组 point-set seed、每模型 1000 个 rollout 均完成，未观察到 process error。
2. **确认测量结果：**Contact-CVAE-zero 的 40N safe-success rate 高于 ACT baseline。
3. **统计结果：**逐点 exact McNemar 检验支持 Contact-CVAE-zero 在 40N safe-success 上优于 ACT baseline。
4. **确认测量结果：**ACT baseline 的 task success 中有较多 `max_force >= 40 N` 的 unsafe task success。
5. **确认测量结果：**ACT baseline 的方向不对称在 10 个 point-set seed pooled 后仍然明显。
6. **解释限制：**方向差异和高力暴露的机制仍需 rollout trajectory、接触力方向、相机观测和视频证据验证。

## 13. 局限性

- 本报告只比较 `Contact-CVAE-zero-mid` 与 `ACT baseline-mid`，不包含 temporal 模式和其他模型。
- 所有结果来自 MuJoCo 仿真；40N safe-success 不是硬件安全保证。
- point-set seed 改变了孔位集合，因此 pooled 结果同时反映点集采样差异与策略表现。
- 未对视频或逐步轨迹做行为级分类，不能声称某模型学会了主动搜索、撤退或力释放。

## 14. 派生文件

- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/experiment_inventory.csv`
- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/aggregate_summary_40n.csv`
- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/per_pointset_summary_40n.csv`
- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/paired_overall_40n.csv`
- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/paired_transition_summary_40n.csv`
- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/radius_summary_40n.csv`
- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/directional_summary_40n.csv`
- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/force_timing_summary_40n.csv`
- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/representative_cases_40n.csv`
- `outputs/peg_hole_100/contact_zero_vs_act_10_pointsets/summary/target_map_inventory.csv`

## 15. 结论

在 `+/-4 mm`、10 个独立 point-set seed、每模型 1000 个 `mid` rollout 的实验范围内，Contact-CVAE-zero 在 40N safe-success 指标上显著优于 ACT baseline。ACT baseline 的 task-success 表现不能直接代表安全成功表现，因为其成功样本中存在更多完整 rollout 最大力超过 40N 的情况。当前数据支持将 Contact-CVAE-zero 作为该设置下更稳健的低力成功配置；机制层面的解释仍需进一步轨迹和视频分析。
