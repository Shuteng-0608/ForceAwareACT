# ForceAwareACT ±6 mm孔位扰动Rollout实验报告

## 1. 实验背景与目的

本报告对仓库本地 `outputs/peg_hole_100` 下 50 点 x/z ±6 mm 孔位扰动 rollout 结果进行证据化整理。分析覆盖 Contact-CVAE、Motion-CVAE、DualZero 与 ACT baseline 四类策略，以及每类策略的 `mid`、`temporal` 两种 action selection mode，共 8 个实验、400 次 rollout。

报告目标包括：

1. 核验实验记录与图表是否完整；
2. 确认各实验的协议和 task points 是否可直接比较；
3. 汇总 task success、safe success、力、完成时间及失败原因；
4. 分析成功结果随半径、方向和象限的变化；
5. 对同一模型的 `mid` 与 `temporal` 进行逐点配对比较。

本报告仅使用已保存的 `grid_summary.csv`、`random_position_summary.json`、`grid_manifest.json`、`task_points.csv`、逐点 `summary.json` 和已有 target map。未修改任何实验原始文件。

## 2. 报告范围与术语

纳入范围为目录名匹配 `hole_lhs_50_xz_6mm_*`、且包含 `grid_summary.csv` 的实验根目录。

- `task success`：`grid_summary.csv` 中 `success=True`。
- `safe success`：`grid_summary.csv` 中 `safe_success=True`。本批配置使用 `success_force_threshold=80 N`；它表示成功 rollout 同时满足该实验定义的力阈值约束，不是通用硬件安全保证。
- `process error`：rollout 进程未正常完成或 manifest 记录为执行错误。
- `task failure`：进程正常完成，但未满足成功判据。
- `maximum observed successful radius`：本次 50 点 LHS 样本中最远的成功点，不是连续空间中的泛化边界。

所有空间结论只描述实测采样点。本报告不插值连续成功区域，也不从稀疏点集拟合成功边界。

## 3. 实验协议

8 个实验共享以下协议：

| setting | value |
| --- | --- |
| sampling mode | `latin_hypercube` |
| points / repeats | 50 / 1 |
| base seed | `20260702` |
| x range | `[-0.006, 0.006] m` |
| z range | `[-0.006, 0.006] m` |
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
| normalization | `normalization_stats_action_all100.pt` |

同一模型的 `mid` 与 `temporal` 实验仅在 `action_select_mode` 上有意不同。不同模型之间还改变了 checkpoint/policy variant，因此跨模型差异只能作描述性比较，不能归因于单一网络设计因素。

## 4. 完整性与可比性检查

确认测量结果：

- 8 个实验均包含 `grid_summary.csv`、`random_position_summary.json`、`grid_manifest.json` 和 `task_points.csv`。
- 每个实验均有 50 个逐点输出目录、50 个 completed runs 和 50 条 summary 记录。
- 8 个实验的 `completion_rate` 均为 100%，`process_error_runs` 均为 0。
- 所有失败记录的 `stop_reason` 均为 `max_rollout_steps`，未发现 hard force stop 或非有限值进程错误。
- 8 份 task points 的 `point_index`、`hole_offset_x`、`hole_offset_y`、`hole_offset_z` 完全相同，最大绝对差为 0。
- 每个实验均已有 publication PNG、publication PDF 和带 point index 的 labeled PNG。

因此，8 个实验可以在相同空间点集上进行逐点描述性比较；其中同模型 `mid` 与 `temporal` 是本报告的主要 paired comparison。

## 5. 纳入实验与总体结果

成功率的 95% CI 使用 Wilson 区间。所有比率分母均为 50 个 completed points。

| experiment | task success | task rate | Wilson 95% CI | safe success | safe rate | task failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-mid | 16/50 | 32.0% | 20.8%-45.8% | 16/50 | 32.0% | 34 |
| Contact-CVAE-temporal | 5/50 | 10.0% | 4.3%-21.4% | 5/50 | 10.0% | 45 |
| Motion-CVAE-mid | 14/50 | 28.0% | 17.5%-41.7% | 14/50 | 28.0% | 36 |
| Motion-CVAE-temporal | 2/50 | 4.0% | 1.1%-13.5% | 2/50 | 4.0% | 48 |
| DualZero-mid | 14/50 | 28.0% | 17.5%-41.7% | 14/50 | 28.0% | 36 |
| DualZero-temporal | 8/50 | 16.0% | 8.3%-28.5% | 7/50 | 14.0% | 42 |
| ACT baseline-mid | 22/50 | 44.0% | 31.2%-57.7% | 12/50 | 24.0% | 28 |
| ACT baseline-temporal | 10/50 | 20.0% | 11.2%-33.0% | 6/50 | 12.0% | 40 |

确认测量结果：`ACT baseline-mid` 的 task success 最高，为 22/50（44.0%）；`Contact-CVAE-mid` 的 safe success 最高，为 16/50（32.0%）。Contact-CVAE 和 Motion-CVAE 的所有 task success 都属于 safe success；DualZero-temporal 有 1 个成功点超过 80 N；ACT baseline-mid 和 ACT baseline-temporal 分别有 10 个和 4 个 task success 未达到 safe-success 定义。

因此，task success 排名不能直接替代本实验定义下的 safe-success 排名。

## 6. 靶心空间分布图

所有 target map 使用相同的 ±10 mm 显示范围和 2 mm 环间距；x/z 比例相等，名义孔中心为 `(0, 0)`。绿色圆点为 safe success，橙色圆点为 task success 但非 safe success，红色圆点为失败；标题报告 safe-success 数量和比率。图中只显示实测 rollout 结果。

| experiment | publication map | labeled diagnostic map |
| --- | --- | --- |
| Contact-CVAE-mid | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_6mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_6mm_target_safe_success_labeled.png) |
| Contact-CVAE-temporal | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_contact_cvae100k_zero_temporal_d03_dq002_maxsteps900/plots/contact_cvae100k_zero_temporal_6mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_contact_cvae100k_zero_temporal_d03_dq002_maxsteps900/plots/contact_cvae100k_zero_temporal_6mm_target_safe_success_labeled.png) |
| Motion-CVAE-mid | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_6mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_6mm_target_safe_success_labeled.png) |
| Motion-CVAE-temporal | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/plots/motion_cvae100k_temporal_6mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_motion_cvae100k_temporal_d03_dq002_maxsteps900/plots/motion_cvae100k_temporal_6mm_target_safe_success_labeled.png) |
| DualZero-mid | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_6mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_6mm_target_safe_success_labeled.png) |
| DualZero-temporal | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_dualzero100k_temporal_d03_dq002_maxsteps900/plots/dualzero100k_temporal_6mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_dualzero100k_temporal_d03_dq002_maxsteps900/plots/dualzero100k_temporal_6mm_target_safe_success_labeled.png) |
| ACT baseline-mid | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_6mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_6mm_target_safe_success_labeled.png) |
| ACT baseline-temporal | [PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_act_baseline100k_temporal_d03_dq002_maxsteps900/plots/act_baseline100k_temporal_6mm_target_safe_success.png) | [labeled PNG](../../outputs/peg_hole_100/hole_lhs_50_xz_6mm_act_baseline100k_temporal_d03_dq002_maxsteps900/plots/act_baseline100k_temporal_6mm_target_safe_success_labeled.png) |

目视检查确认图中名义孔居中、x/z 尺度一致、点位未变形或裁切，图例计数与 CSV 一致。labeled 图可用于定位具体 point index，但标签较密集，因此 publication 图更适合正文展示。

## 7. 成功率随半径的变化

半径定义为：

```text
radius_mm = 1000 * sqrt(hole_offset_x^2 + hole_offset_z^2)
```

50 个共享点在 `[0,2)`、`[2,4)`、`[4,6)`、`[6,8)` mm 中分别有 6、15、19、10 个点。由于方形 ±6 mm 采样的最大理论半径约为 8.49 mm，本次实际点集中没有 `[8,10)` mm 样本。

下表为每个半径 bin 的 `task successes / points`：

| experiment | [0,2) mm | [2,4) mm | [4,6) mm | [6,8) mm | largest successful radius |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-mid | 5/6 | 10/15 | 1/19 | 0/10 | 4.73 mm |
| Contact-CVAE-temporal | 3/6 | 2/15 | 0/19 | 0/10 | 3.68 mm |
| Motion-CVAE-mid | 5/6 | 7/15 | 2/19 | 0/10 | 5.71 mm |
| Motion-CVAE-temporal | 2/6 | 0/15 | 0/19 | 0/10 | 1.75 mm |
| DualZero-mid | 6/6 | 7/15 | 1/19 | 0/10 | 4.73 mm |
| DualZero-temporal | 5/6 | 3/15 | 0/19 | 0/10 | 3.09 mm |
| ACT baseline-mid | 5/6 | 6/15 | 8/19 | 3/10 | 7.56 mm |
| ACT baseline-temporal | 4/6 | 6/15 | 0/19 | 0/10 | 3.97 mm |

结果表明：除 `ACT baseline-mid` 外，其他实验在 `[6,8)` mm 均无成功点；`ACT baseline-mid` 在该 bin 有 3/10 成功，并具有本批最大的观测成功半径 7.56 mm。另一方面，每个实验在较小半径也都有失败点，说明半径不是成功与否的唯一解释变量。

这里的数据支持“成功率总体随半径增大而下降”的描述，但不支持把任一最大成功半径解释为确定的泛化极限。

## 8. 方向与象限分析

下表给出 x/z 半轴上的 `successes / points`：

| experiment | x < 0 | x >= 0 | z < 0 | z >= 0 |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-mid | 9/25 | 7/25 | 8/25 | 8/25 |
| Contact-CVAE-temporal | 3/25 | 2/25 | 2/25 | 3/25 |
| Motion-CVAE-mid | 10/25 | 4/25 | 8/25 | 6/25 |
| Motion-CVAE-temporal | 0/25 | 2/25 | 1/25 | 1/25 |
| DualZero-mid | 7/25 | 7/25 | 6/25 | 8/25 |
| DualZero-temporal | 4/25 | 4/25 | 3/25 | 5/25 |
| ACT baseline-mid | 3/25 | 19/25 | 16/25 | 6/25 |
| ACT baseline-temporal | 2/25 | 8/25 | 4/25 | 6/25 |

象限结果：

| experiment | +x +z | -x +z | -x -z | +x -z |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-mid | 5/12 | 3/13 | 6/12 | 2/13 |
| Contact-CVAE-temporal | 2/12 | 1/13 | 2/12 | 0/13 |
| Motion-CVAE-mid | 3/12 | 3/13 | 7/12 | 1/13 |
| Motion-CVAE-temporal | 1/12 | 0/13 | 0/12 | 1/13 |
| DualZero-mid | 6/12 | 2/13 | 5/12 | 1/13 |
| DualZero-temporal | 4/12 | 1/13 | 3/12 | 0/13 |
| ACT baseline-mid | 6/12 | 0/13 | 3/12 | 13/13 |
| ACT baseline-temporal | 6/12 | 0/13 | 2/12 | 2/13 |

探索性 Fisher exact test 显示，`ACT baseline-mid` 的 x 方向差异为 `p=0.000010`，z 方向差异为 `p=0.009595`；其 `+x,-z` 象限为 13/13 成功，而 `-x,+z` 象限为 0/13。其他实验的 x/z 半轴 Fisher 检验均未达到 `p<0.05`。

该方向性是实测计数结果，但 LHS 设计并非预注册的方向假设检验，且同一物理系统可能存在坐标、初始姿态或控制动态的不对称。可能机制仍需逐步轨迹和视频验证，不能仅凭 target map 断言模型学习了特定搜索、退让或接触策略。

## 9. Mid与Temporal逐点配对比较

每对实验使用完全相同的 50 个点。`mid only` 表示该点仅在 `mid` 成功，`temporal only` 同理。p-value 为 discordant pairs 上的 exact two-sided McNemar/binomial test。

### 9.1 Task success

| model | both success | mid only | temporal only | both fail | mid-temporal rate difference | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE | 5 | 11 | 0 | 34 | +22 pp | 0.000977 |
| Motion-CVAE | 1 | 13 | 1 | 35 | +24 pp | 0.001831 |
| DualZero | 7 | 7 | 1 | 35 | +12 pp | 0.070312 |
| ACT baseline | 8 | 14 | 2 | 26 | +24 pp | 0.004181 |

### 9.2 Safe success

| model | both safe | mid only | temporal only | neither safe | mid-temporal rate difference | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE | 5 | 11 | 0 | 34 | +22 pp | 0.000977 |
| Motion-CVAE | 1 | 13 | 1 | 35 | +24 pp | 0.001831 |
| DualZero | 6 | 8 | 1 | 35 | +14 pp | 0.039062 |
| ACT baseline | 5 | 7 | 1 | 37 | +12 pp | 0.070312 |

统计结果：四个模型的 `mid` task success 均高于对应 `temporal`。在本 50 点样本上，Contact-CVAE、Motion-CVAE 和 ACT baseline 的 task-success 配对差异达到 `p<0.05`；DualZero 的 task-success 差异未达到该阈值。safe-success 配对结果中，Contact-CVAE、Motion-CVAE 和 DualZero 达到 `p<0.05`，ACT baseline 未达到。

这些检验是单批 50 点实验上的探索性结果。它们支持 action selection mode 与结果变化相关，但不单独证明 temporal aggregation 导致了某种具体物理行为。

## 10. 力与完成时间

下表报告 `max_force` 和成功 rollout 的 `success_time`。`max_force` 的 all-run 数量均为 50；时间只对成功点定义。

| experiment | all max force mean / median / max (N) | success max force mean / median / max (N) | success time mean / median / max (s) |
| --- | --- | --- | --- |
| Contact-CVAE-mid | 35.15 / 35.47 / 62.67 | 25.50 / 28.53 / 35.98 | 6.80 / 6.75 / 8.55 |
| Contact-CVAE-temporal | 43.81 / 36.53 / 108.55 | 19.16 / 10.91 / 57.99 | 16.90 / 17.49 / 20.72 |
| Motion-CVAE-mid | 59.88 / 65.41 / 92.29 | 40.42 / 45.17 / 68.78 | 8.66 / 7.77 / 20.03 |
| Motion-CVAE-temporal | 52.13 / 54.83 / 100.41 | 32.62 / 32.62 / 55.48 | 19.60 / 19.60 / 24.85 |
| DualZero-mid | 68.70 / 75.33 / 97.62 | 40.72 / 40.90 / 76.05 | 8.96 / 6.93 / 28.18 |
| DualZero-temporal | 67.12 / 66.94 / 108.14 | 45.90 / 36.41 / 106.95 | 16.16 / 16.60 / 21.58 |
| ACT baseline-mid | 79.95 / 96.90 / 104.24 | 56.50 / 43.82 / 104.01 | 9.07 / 6.24 / 24.92 |
| ACT baseline-temporal | 81.84 / 88.21 / 107.87 | 62.53 / 71.44 / 102.05 | 17.25 / 14.73 / 28.28 |

确认测量结果：

- Contact-CVAE-mid 的 all-run mean max force 最低，为 35.15 N。
- ACT baseline-mid 的 task success 最多，但 22 个成功中只有 12 个满足 safe success；成功样本 mean max force 为 56.50 N，最大值为 104.01 N。
- 四组 `temporal` 成功样本的 median success time 均长于对应 `mid`。不过 temporal 成功样本数量仅为 2-10，时间比较应谨慎解释。
- 1000 N hard force stop 从未触发。观察到的最大力约为 108.55 N，远低于 hard-stop threshold，但超过 80 N safe-success threshold 的成功仍不能记为 safe success。

## 11. 失败类型

8 个实验共有 309 个 task failures，全部记录为 `stop_reason=max_rollout_steps`。这表示 rollout 正常执行到 900 步但未满足持续 15 步的成功判据。

本批没有：

- process error；
- hard force stop；
- 非有限值 termination；
- 缺失 stop reason 的 unknown failure。

仅凭最终 summary 无法进一步确认失败是由距离、横向误差、接触状态还是动作轨迹中的其他过程造成。诸如“未搜索”“未退让”“卡住”等行为级分类，需要结合 `rollout_log.csv` 和视频验证。

## 12. 代表性案例

| experiment | largest-radius success (point, radius) | smallest-radius failure (point, radius) | highest-force success (point, N) |
| --- | --- | --- | --- |
| Contact-CVAE-mid | 50, 4.73 mm | 23, 1.82 mm | 49, 35.98 N |
| Contact-CVAE-temporal | 18, 3.68 mm | 26, 1.41 mm | 29, 57.99 N |
| Motion-CVAE-mid | 43, 5.71 mm | 5, 1.75 mm | 43, 68.78 N |
| Motion-CVAE-temporal | 5, 1.75 mm | 9, 0.56 mm | 5, 55.48 N |
| DualZero-mid | 50, 4.73 mm | 49, 2.28 mm | 40, 76.05 N |
| DualZero-temporal | 40, 3.09 mm | 26, 1.41 mm | 5, 106.95 N |
| ACT baseline-mid | 2, 7.56 mm | 47, 1.90 mm | 23, 104.01 N |
| ACT baseline-temporal | 42, 3.97 mm | 26, 1.41 mm | 3, 102.05 N |

建议优先检查以下案例的 `rollout_log.csv`：

- `ACT baseline-mid`, point 2：本批最大观测成功半径，`x=5.647 mm`、`z=-5.021 mm`、`max_force=93.75 N`，属于 task success 但不是 safe success。
- `ACT baseline-mid`, point 20：最快成功之一，`success_time=5.58 s`、`max_force=35.39 N`。
- `Motion-CVAE-temporal`, point 9：距中心仅 0.56 mm 仍失败，说明近中心不保证成功。
- `DualZero-temporal`, point 5：task success 但 `max_force=106.95 N`，是该实验中 task/safe-success 分离的唯一成功点。

## 13. 主要发现

1. **确认测量结果**：8 个实验全部完整，400 次 rollout 无 process error，且使用完全相同的 50 点 LHS 点集。
2. **确认测量结果**：`ACT baseline-mid` 的 task success 最高，为 44.0%；`Contact-CVAE-mid` 的 safe success 最高，为 32.0%。
3. **统计结果**：四个模型的 `mid` task success 均高于对应 `temporal`；其中 Contact-CVAE、Motion-CVAE 和 ACT baseline 的配对 exact p-value 小于 0.05。
4. **空间结果**：成功总体集中在较小半径；只有 `ACT baseline-mid` 在 `[6,8)` mm 仍有成功点。
5. **方向结果**：`ACT baseline-mid` 呈现显著方向不对称，尤其 `+x,-z` 为 13/13、`-x,+z` 为 0/13；该现象需要轨迹层面验证。
6. **力结果**：task success 与 safe success 必须分开。ACT baseline 的较高 task success 伴随较多超过 80 N 的成功点。

## 14. 解释与局限性

数据支持 `mid` 在本批 ±6 mm、50 点协议下比 `temporal` 获得更多成功点，并且成功概率与孔位半径、方向有关。可能的机制包括动作选择方式改变闭环轨迹平滑性、响应速度或接触后的修正过程，但这些属于待验证假设。

本报告不声称模型已经学习主动搜索、退让或力释放。该类机制解释仍需视频与逐步轨迹验证。

主要局限性：

1. 每个条件只有一组 50 点 LHS 样本，没有跨 seed 重复；
2. 点集稀疏，不能推断连续成功区域；
3. 方向检验是探索性的，并非专门设计的平衡方向实验；
4. temporal 成功样本较少，力与时间统计不稳定；
5. 跨模型比较同时改变 checkpoint 和 policy variant，不能解释为单因素因果；
6. `safe success` 仅是本实验 80 N 阈值下的操作定义。

## 15. 后续实验建议

1. 对 paired transition 中的 `mid only`、`temporal only` 点检查 rollout trajectory 和视频；
2. 对 `ACT baseline-mid` 的 `+x,-z` 与 `-x,+z` 象限进行重复 seed 实验，验证方向差异是否稳定；
3. 在相同点集上增加重复 rollout，估计单点随机性；
4. 对成功但不安全的点分析峰值力出现时刻、持续时间和接触几何；
5. 保持 task points 与阈值不变，再比较其他 offset 范围，避免把点集变化误当作模型变化。

## 16. 结论

本批 ±6 mm 实验完整且可逐点比较。总体上，`mid` 在四类策略中均获得更高 task success；`ACT baseline-mid` 的任务成功率最高，而 `Contact-CVAE-mid` 的 safe-success rate 最高。成功结果随半径增大总体下降，同时存在明显的模型与方向差异。

这些结论严格限于当前 50 个 LHS 实测点。target map 展示的是离散测量结果，不代表连续空间中的成功区域或确定泛化边界。

## 附录A：实验路径

| experiment | experiment root |
| --- | --- |
| Contact-CVAE-mid | `outputs/peg_hole_100/hole_lhs_50_xz_6mm_contact_cvae100k_zero_mid_dq002_maxsteps900` |
| Contact-CVAE-temporal | `outputs/peg_hole_100/hole_lhs_50_xz_6mm_contact_cvae100k_zero_temporal_d03_dq002_maxsteps900` |
| Motion-CVAE-mid | `outputs/peg_hole_100/hole_lhs_50_xz_6mm_motion_cvae100k_mid_dq002_maxsteps900` |
| Motion-CVAE-temporal | `outputs/peg_hole_100/hole_lhs_50_xz_6mm_motion_cvae100k_temporal_d03_dq002_maxsteps900` |
| DualZero-mid | `outputs/peg_hole_100/hole_lhs_50_xz_6mm_dualzero100k_mid_dq002_maxsteps900` |
| DualZero-temporal | `outputs/peg_hole_100/hole_lhs_50_xz_6mm_dualzero100k_temporal_d03_dq002_maxsteps900` |
| ACT baseline-mid | `outputs/peg_hole_100/hole_lhs_50_xz_6mm_act_baseline100k_mid_dq002_maxsteps900` |
| ACT baseline-temporal | `outputs/peg_hole_100/hole_lhs_50_xz_6mm_act_baseline100k_temporal_d03_dq002_maxsteps900` |

## 附录B：数据字段

所有 `grid_summary.csv` 使用同一 schema：

```text
point_index, sampling_mode, base_seed,
hole_offset_x, hole_offset_y, hole_offset_z,
radial_offset, quadrant, success, safe_success,
success_step, success_time, stop_reason,
final_dist, final_lateral, final_axial,
max_force, mean_force, force_gt_20_steps, force_gt_40_steps,
output_dir, summary_json, rollout_log_csv
```

分析中的成功、力、时间和空间数值均直接由上述字段计算，并与 `random_position_summary.json` 中的总体计数交叉核对。
