# ForceAwareACT +/-4 mm Multi-seed Mid Rollout 实验报告

## 1. 实验目的与范围

本报告分析 `outputs/peg_hole_100/multiseed` 下 5 个 base seed 的 x/z +/-4 mm 孔位扰动 rollout，只纳入 `action_select_mode=mid`。纳入 seed 为 `20260702`、`20260703`、`20260704`、`20260705` 和 `20260706`。

实验覆盖 4 类策略架构、5 个可执行配置：Contact-CVAE 使用 `z_contact=zero` 与 `z_contact=prior` 两种 latent mode，另包括 Motion-CVAE、DualZero 和 ACT baseline。每个配置在每个 seed 上测试 50 个 Latin hypercube 点，因此每个配置有 250 次 rollout，总计 1250 次。

目录中已有的 `temporal` 结果不属于本报告范围，未进入任何统计量。

## 2. 数据来源与术语

主要数据来自每个实验根目录中的：

- `grid_summary.csv`：逐点任务结果、力、完成时间和最终误差；
- `random_position_summary.json`：实验级成功数、完成率与 process error；
- `grid_manifest.json`：采样、策略及成功判据配置；
- `task_points.csv`：逐 seed 的实际采样点；
- `plots/`：safe-success target map。

本文使用以下定义：

- `task success`：距离、横向误差和当前力连续 15 步满足成功条件；
- `safe success`：已经 task success，且该次完整 rollout 的历史 `max_force < 80 N`；
- `task success, not safe`：达到 task success，但历史最大力不满足上述严格不等式；
- `task failure`：未达到 task success；本批数据中的 task failure 均以 `max_rollout_steps` 结束；
- `process error`：rollout 进程或结果写入未正常完成。

`safe success` 只是本实验基于最大力阈值定义的指标，不代表通用硬件安全保证。

## 3. 实验协议

25 个纳入实验的 manifest 在以下字段上完全一致，checkpoint/latent 配置是有意设置的模型间差异。

| setting | value |
| --- | --- |
| sampling mode | `latin_hypercube` |
| points / repeats | 50 / 1 |
| base seeds | `20260702`--`20260706` |
| x/z bounds | `[-0.004, 0.004] m` |
| y offset | `0 m` |
| action mode / selection | `action` / `mid` |
| chunk length | 10 |
| policy rate | 30 Hz |
| maximum rollout steps | 900 |
| maximum delta q | 0.02 rad |
| force window | 20 samples / 0.25 s |
| force stop threshold | 1000 N |
| success distance threshold | 0.005 m |
| success lateral threshold | 0.006 m |
| success force threshold | 80 N |
| success hold steps | 15 |
| hole site / body | `hole_goal_site` / `wall_task` |
| model XML path | `../arm_teleop/model/pangu_all_right.xml` |
| normalization stats | `normalization_stats_action_all100.pt` |

每个 seed 内，5 个配置的 `point_index` 与 x/y/z offsets 完全相同，因此同一 seed 内可以逐点配对。不同 seed 的 LHS 点集不同，正是本实验用于估计采样波动的设计。

XML 文件当前 mtime 为 2026-07-09 15:07 UTC，早于最早纳入 rollout 的开始时间 15:09 UTC；所有纳入运行均在此后完成。这支持本批运行期间使用同一路径下模型文件的判断，但 manifest 没有保存 XML hash 或内容快照，因此不能从结果文件独立证明文件内容在整个期间绝对未变化。

## 4. 完整性检查

| item | result |
| --- | ---: |
| seeds | 5/5 |
| mid experiment roots | 25/25 |
| planned rollouts | 1250 |
| completed rollouts | 1250 |
| completion rate | 100% |
| process errors | 0 |
| task successes | 686 |
| safe successes | 610 |
| task failures | 564 |
| task success, not safe | 76 |

所有实验均有 `grid_summary.csv`、`random_position_summary.json`、`grid_manifest.json`、`task_points.csv`，并各自生成 publication PNG、PDF 和 labeled diagnostic PNG。seed `20260705` 的 4 个复用实验在当前 manifest 中没有逐 run 的 `start_time/end_time`，但其 50 个逐点 summary、聚合 CSV 与完成计数均存在且相互一致；这是时间元数据缺口，不影响结果计数。

## 5. 逐 Seed 结果

表格单元为 `task successes / safe successes`，分母均为 50。

| configuration | 20260702 | 20260703 | 20260704 | 20260705 | 20260706 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 32 / 32 | 29 / 28 | 29 / 29 | 32 / 32 | 30 / 30 |
| Contact-CVAE-prior | 24 / 24 | 27 / 27 | 28 / 27 | 24 / 23 | 24 / 22 |
| Motion-CVAE | 29 / 29 | 26 / 26 | 27 / 27 | 30 / 29 | 28 / 26 |
| DualZero | 22 / 20 | 25 / 25 | 22 / 21 | 24 / 23 | 22 / 21 |
| ACT baseline | 28 / 17 | 33 / 17 | 30 / 18 | 32 / 20 | 29 / 17 |

各配置的 safe-success rate 在 5 个 seed 间范围为：Contact-CVAE-zero 56%--64%，Contact-CVAE-prior 44%--54%，Motion-CVAE 52%--58%，DualZero 40%--50%，ACT baseline 34%--40%。这说明当前 50 点 LHS 采样下，模型排序没有被某一个 seed 单独决定。

## 6. 跨 Seed 汇总

`pooled rate` 使用 5 个 seed 的 250 个实测点；95% CI 为 pooled Bernoulli 结果的 Wilson 区间。`seed mean +/- SD` 把每个 seed 的成功率作为一个观测，展示点集变化带来的波动。由于每个 seed 恰好都是 50 点，seed mean 与 pooled rate 数值相同。

| configuration | task success | task rate (95% CI) | safe success | safe rate (95% CI) | safe seed mean +/- SD | not-safe successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 152/250 | 60.8% (54.6%--66.6%) | 151/250 | **60.4%** (54.2%--66.3%) | 60.4% +/- 3.6 pp | 1 |
| Motion-CVAE | 140/250 | 56.0% (49.8%--62.0%) | 137/250 | **54.8%** (48.6%--60.9%) | 54.8% +/- 3.0 pp | 3 |
| Contact-CVAE-prior | 127/250 | 50.8% (44.6%--56.9%) | 123/250 | **49.2%** (43.1%--55.4%) | 49.2% +/- 4.6 pp | 4 |
| DualZero | 115/250 | 46.0% (39.9%--52.2%) | 110/250 | **44.0%** (38.0%--50.2%) | 44.0% +/- 4.0 pp | 5 |
| ACT baseline | 152/250 | 60.8% (54.6%--66.6%) | 89/250 | **35.6%** (29.9%--41.7%) | 35.6% +/- 2.6 pp | 63 |

确认测量结果：Contact-CVAE-zero 的 pooled safe-success rate 最高，为 60.4%；Motion-CVAE 次之，为 54.8%。ACT baseline 的 task-success rate 与 Contact-CVAE-zero 同为 60.8%，但 safe-success rate 只有 35.6%，两者相差 25.2 个百分点。该差距来自 ACT baseline 的 152 个 task success 中有 63 个 rollout 的历史最大力达到或超过 80 N。

Wilson 区间描述 pooled 点级二项不确定性，没有显式建模 seed 内相关性；5 个 seed 的 SD 是对此的重要补充，但 5 个 seed 仍不足以精确估计更高层级的随机效应。

## 7. 配对比较

同一 seed 内的点位完全一致，因此下表使用 250 个配对点比较 safe success。`A only` 和 `B only` 是 discordant pairs，`p` 为双侧 exact McNemar/binomial 检验。该检验是点级探索性结果，未对 10 次两两比较做多重检验校正，也未显式处理 seed 分层。

| A vs B | both safe | A only | B only | neither | rate difference | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-zero vs Motion | 115 | 36 | 22 | 77 | +5.6 pp | 0.086949 |
| Contact-zero vs Contact-prior | 120 | 31 | 3 | 96 | +11.2 pp | <0.000001 |
| Contact-zero vs DualZero | 102 | 49 | 8 | 91 | +16.4 pp | <0.000001 |
| Contact-zero vs ACT baseline | 59 | 92 | 30 | 69 | +24.8 pp | <0.000001 |
| Motion vs Contact-prior | 98 | 39 | 25 | 88 | +5.6 pp | 0.103422 |
| Motion vs DualZero | 84 | 53 | 26 | 87 | +10.8 pp | 0.003183 |
| Motion vs ACT baseline | 51 | 86 | 38 | 75 | +19.2 pp | 0.000019 |
| Contact-prior vs DualZero | 91 | 32 | 19 | 108 | +5.2 pp | 0.091915 |
| Contact-prior vs ACT baseline | 49 | 74 | 40 | 87 | +13.6 pp | 0.001867 |
| DualZero vs ACT baseline | 46 | 64 | 43 | 97 | +8.4 pp | 0.052668 |

统计结果支持 Contact-CVAE-zero 相对 Contact-CVAE-prior、DualZero 和 ACT baseline 有更多 safe-success 点，也支持 Motion-CVAE 相对 DualZero 和 ACT baseline 有更多 safe-success 点。Contact-CVAE-zero 与 Motion-CVAE 的差异在本样本中未达到 `p<0.05`；这不等于二者性能相同。

任务成功层面，Contact-CVAE-zero 与 ACT baseline 都是 152/250，且逐点转移恰为各自独有 48 点，exact `p=1.0`。safe-success 层面的明显分离因此主要来自完整 rollout 最大力，而不是任务完成数量。

## 8. 半径分析

定义 `radius_mm = 1000 * sqrt(x^2 + z^2)`。5 个 seed 合计在 `[0,2)`、`[2,4)`、`[4,6)` mm 分别有 52、149 和 49 个点。表格为 `task successes / safe successes / points`。

| configuration | [0,2) mm | [2,4) mm | [4,6) mm | largest observed task / safe radius |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 51 / 51 / 52 | 96 / 95 / 149 | 5 / 5 / 49 | 4.83 / 4.83 mm |
| Contact-CVAE-prior | 49 / 49 / 52 | 75 / 71 / 149 | 3 / 3 / 49 | 4.83 / 4.83 mm |
| Motion-CVAE | 47 / 47 / 52 | 86 / 84 / 149 | 7 / 6 / 49 | 4.83 / 4.68 mm |
| DualZero | 51 / 51 / 52 | 62 / 57 / 149 | 2 / 2 / 49 | 4.15 / 4.15 mm |
| ACT baseline | 46 / 24 / 52 | 81 / 47 / 149 | 25 / 18 / 49 | 5.04 / 5.04 mm |

结果表明，所有配置的成功率总体随半径增大而下降，但半径不是唯一解释变量。ACT baseline 在 `[4,6)` mm 有最多 task/safe successes，却在 `[0,2)` mm 只有 24/52 safe successes；这与其明显方向依赖和较高历史最大力共同出现。最大成功半径只是 5 个离散 LHS 点集中的最大观测值，不能解释为连续空间的绝对泛化边界。

## 9. 方向与象限分析

下表为 pooled `task successes / safe successes / points`。

| configuration | x < 0 | x >= 0 | z < 0 | z >= 0 |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 86 / 85 / 125 | 66 / 66 / 125 | 84 / 84 / 125 | 68 / 67 / 125 |
| Contact-CVAE-prior | 72 / 69 / 125 | 55 / 54 / 125 | 72 / 71 / 125 | 55 / 52 / 125 |
| Motion-CVAE | 79 / 77 / 125 | 61 / 60 / 125 | 71 / 71 / 125 | 69 / 66 / 125 |
| DualZero | 67 / 62 / 125 | 48 / 48 / 125 | 48 / 48 / 125 | 67 / 62 / 125 |
| ACT baseline | 41 / 3 / 125 | 111 / 86 / 125 | 103 / 58 / 125 | 49 / 31 / 125 |

ACT baseline 的方向差异在 5 个 seed pooled 后仍很突出：`x<0` 只有 3/125 safe success，而 `x>=0` 为 86/125。象限计数进一步显示：

| quadrant | Contact-zero | Contact-prior | Motion | DualZero | ACT baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| +x +z | 32 / 32 / 57 | 26 / 26 / 57 | 34 / 33 / 57 | 33 / 33 / 57 | 43 / 31 / 57 |
| -x +z | 36 / 35 / 68 | 29 / 26 / 68 | 35 / 33 / 68 | 34 / 29 / 68 | 6 / 0 / 68 |
| -x -z | 50 / 50 / 57 | 43 / 43 / 57 | 44 / 44 / 57 | 33 / 33 / 57 | 35 / 3 / 57 |
| +x -z | 34 / 34 / 68 | 29 / 28 / 68 | 27 / 27 / 68 | 15 / 15 / 68 | 68 / 55 / 68 |

其中 ACT baseline 在 `-x,+z` 象限为 0/68 safe success，在 `+x,-z` 象限为 55/68。该结果确认了采样结果中的方向不对称，但不能单凭终态表格断言其物理机制。可能相关的因素包括训练数据方向分布、机器人构型与接触几何、相机视角、归一化分布或动作执行误差；这些解释仍需镜像点实验、逐步 trajectory、接触力方向和视频验证。

## 10. 力与完成时间

表中 force 单位为 N；`all max-force` 统计全部 250 个 rollout，`success max-force` 统计 task-success rollout，时间只在 task success 上定义。

| configuration | all max-force mean / median / P95 | success max-force mean / median | safe-success max-force mean | failure max-force mean | success time mean / median / P90 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 29.70 / 29.54 / 47.22 | 24.41 / 25.31 | 23.97 | 37.90 | 7.75 / 6.81 / 8.84 |
| Contact-CVAE-prior | 56.74 / 53.03 / 94.52 | 44.79 / 43.10 | 43.48 | 69.08 | 6.84 / 6.57 / 7.72 |
| Motion-CVAE | 50.13 / 53.12 / 90.55 | 43.02 / 45.03 | 41.99 | 59.19 | 9.34 / 7.62 / 15.91 |
| DualZero | 60.82 / 63.60 / 95.17 | 39.48 / 43.18 | 37.11 | 79.00 | 8.70 / 6.67 / 15.80 |
| ACT baseline | 71.17 / 97.10 / 105.21 | 53.20 / 40.46 | 19.07 | 99.05 | 9.91 / 6.17 / 22.37 |

Contact-CVAE-zero 同时具有最低的总体 max-force 均值、P95 和最高 safe-success rate。ACT baseline 的 task-success max-force 中位数为 40.46 N，但其均值为 53.20 N，且 63 个成功 rollout 的历史峰值不低于 80 N，说明只看成功时刻的条件或中位数会低估其高力尾部。这里是关联性描述，不足以证明某种 latent 或策略结构直接导致特定接触行为。

## 11. Target Map

现有 target map 使用相同 +/-6 mm 显示范围和 2 mm rings。绿色表示 safe success，橙色表示 task success 但非 safe success，红色表示 task failure。每张图只显示离散实测点，不插值连续成功区域。

| configuration | seed 20260702 | seed 20260703 | seed 20260704 | seed 20260705 | seed 20260706 |
| --- | --- | --- | --- | --- | --- |
| Contact-CVAE-zero | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260702/hole_lhs_50_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260703/hole_lhs_50_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260704/hole_lhs_50_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260705/hole_lhs_50_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260706/hole_lhs_50_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success.png) |
| Contact-CVAE-prior | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260702/hole_lhs_50_xz_4mm_contact_cvae100k_prior_mid_dq002_maxsteps900/plots/contact_cvae100k_prior_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260703/hole_lhs_50_xz_4mm_contact_cvae100k_prior_mid_dq002_maxsteps900/plots/contact_cvae100k_prior_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260704/hole_lhs_50_xz_4mm_contact_cvae100k_prior_mid_dq002_maxsteps900/plots/contact_cvae100k_prior_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260705/hole_lhs_50_xz_4mm_contact_cvae100k_prior_mid_dq002_maxsteps900/plots/contact_cvae100k_prior_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260706/hole_lhs_50_xz_4mm_contact_cvae100k_prior_mid_dq002_maxsteps900/plots/contact_cvae100k_prior_mid_4mm_target_safe_success.png) |
| Motion-CVAE | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260702/hole_lhs_50_xz_4mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260703/hole_lhs_50_xz_4mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260704/hole_lhs_50_xz_4mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260705/hole_lhs_50_xz_4mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260706/hole_lhs_50_xz_4mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_4mm_target_safe_success.png) |
| DualZero | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260702/hole_lhs_50_xz_4mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260703/hole_lhs_50_xz_4mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260704/hole_lhs_50_xz_4mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260705/hole_lhs_50_xz_4mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260706/hole_lhs_50_xz_4mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_4mm_target_safe_success.png) |
| ACT baseline | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260702/hole_lhs_50_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260703/hole_lhs_50_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260704/hole_lhs_50_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260705/hole_lhs_50_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success.png) | [PNG](../../outputs/peg_hole_100/multiseed/seed_20260706/hole_lhs_50_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success.png) |

每个同名 stem 还存在 `.pdf` 和 `_labeled.png` 版本。

## 12. 主要发现

1. **确认测量结果：**25 个 `mid` 实验均完整，1250/1250 rollout 完成，process error 为 0。
2. **确认测量结果：**Contact-CVAE-zero 的 safe-success rate 最高，为 151/250（60.4%）；Motion-CVAE 为 137/250（54.8%）。
3. **统计结果：**Contact-CVAE-zero 与 Motion-CVAE 的 5.6 pp safe-rate 差异在当前逐点 exact 配对检验中未达到 `p<0.05`。
4. **确认测量结果：**Contact-CVAE-zero 相比 prior latent 的 safe-success rate 高 11.2 pp；逐点配对转移为 zero-only 31、prior-only 3。
5. **确认测量结果：**ACT baseline 具有并列最高 task-success rate（60.8%），却具有最低 safe-success rate（35.6%），说明 task success 不能替代完整 rollout 的最大力评估。
6. **确认测量结果：**ACT baseline 的空间结果高度不对称，尤其 `-x,+z` 为 0/68 safe success，而 `+x,-z` 为 55/68。
7. **机制解释边界：**现有结果支持“性能与方向、半径和历史最大力有关”，但不支持直接声称模型学会了搜索、退让或力释放。此类解释需要轨迹和视频证据。

## 13. 局限性与后续建议

- 只有 5 个 seed。当前结果明显强于单 seed 结论，但 seed-level 方差估计仍较粗糙。
- 每个 seed 内是 50 个 LHS 点，而不是同一点的重复动力学试验；因此本实验同时改变空间采样点和 rollout seed，不能把 seed 波动拆分成纯动力学随机性与纯点集差异。
- pooled Wilson CI 与 McNemar 检验按点处理，没有构建 seed 分层或混合效应模型；统计 p 值应视为探索性结果。
- 10 次模型两两检验没有多重比较校正。正式论文推断应预先指定主要比较，并考虑 Holm 等校正或分层模型。
- `safe success` 只由 task success 与历史 `max_force < 80 N` 定义，没有覆盖冲量、持续接触时间、关节负载或硬件安全边界。
- manifest 未保存 XML、checkpoint 和 normalization 文件 hash。建议后续把这些 hash 与 Git revision 写入每个实验 manifest。
- 对 ACT baseline 的方向不对称，建议增加 x/z 镜像成对点，并比较 `qpos/qvel`、三轴力、接触状态、动作裁剪次数和视频；在这些证据出现前，不应将不对称归因于单一机制。

## 14. 结论

在 5 个 seed、每配置 250 个 +/-4 mm LHS 点和统一 `mid` action selection 下，Contact-CVAE-zero 获得最高 pooled safe-success rate（60.4%），Motion-CVAE 次之（54.8%）。Contact-CVAE-zero 的 seed 间 safe rate 为 56%--64%，说明这一结果在当前 5 个点集上较稳定。ACT baseline 的 task-success rate 同样达到 60.8%，但其 safe-success rate 仅为 35.6%，并表现出强烈方向不对称和较高的完整 rollout 最大力。

因此，本批结果支持将 Contact-CVAE-zero 作为当前 +/-4 mm、`mid` 模式下 safe-success 表现最好的配置；但 Contact-CVAE-zero 与 Motion-CVAE 的差异尚不足以形成显著的逐点统计分离。后续最有价值的工作是增加预先设计的独立 seeds、保存不可变配置 hash，并围绕 ACT baseline 的镜像方向点进行轨迹与接触力验证。

## 附录：汇总文件

- [Per-seed summary](../../outputs/peg_hole_100/multiseed/summary/per_seed_summary.csv)
- [Aggregate summary](../../outputs/peg_hole_100/multiseed/summary/aggregate_summary.csv)
- 实验根目录：`outputs/peg_hole_100/multiseed/seed_<seed>/`

