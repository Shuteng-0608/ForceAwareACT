# ForceAwareACT +/-4 mm 100点Rollout 40 N Safe-success评估报告

## 1. 评估目的

本报告将 safe-force boundary 设为 40 N，重新评估 Contact-CVAE-zero 与 ACT baseline 在同一组 100 个 +/-4 mm 孔位上的结果，并生成按 40 N 重新分类的 target maps。

本次不重跑 rollout，也不修改原始 CSV。派生规则为：

```text
safe_success_40n = task success AND max_force < 40 N
```

规则使用严格小于号；`max_force = 40 N` 不属于 safe success。原始 `safe_success` 字段继续保留 80 N 定义。

## 2. 数据范围

```text
point_set_seed = 20260702
rollout_seed_base = 31000
action_select_mode = mid
num_points = 100
x/z range = [-4, 4] mm
max_rollout_steps = 900
```

纳入 Contact-CVAE-zero 与 ACT baseline。其他 9 个 rollout-seed bases 的逐点科学结果与本组完全一致，因此只作为确定性复现证据，不增加有效样本量。每个模型的有效样本量为 100 个唯一空间点。

## 3. 完整性与可比性

- 两个实验均完成 100/100 points；
- process error 均为 0；
- 两个模型的 `point_index` 和 x/y/z offsets 完全相同；
- 模型 XML、动作选择、成功判据和控制参数一致；
- 可以进行逐点配对比较。

本报告只重算完成后的 safe 分类。原 task-success 判据中的瞬时 80 N 条件没有重跑；但任何历史 `max_force < 40 N` 的 task success 必然全程低于 40 N。

## 4. 40 N总体结果

| model | task success | safe success <40 N | task success >=40 N | task failure |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 57/100 | **56/100 (56%)** | 1 | 43 |
| ACT baseline | 61/100 | **24/100 (24%)** | 37 | 39 |

Wilson 95% CI：

| metric | rate | Wilson 95% CI |
| --- | ---: | ---: |
| Contact safe <40 N | 56% | 46.2%--65.3% |
| ACT safe <40 N | 24% | 16.7%--33.2% |

Contact 的 40 N safe-success rate 比 ACT 高 32 个百分点。Contact 的 57 个 task successes 中只有 1 个达到或超过 40 N；ACT 的 61 个 task successes 中有 37 个达到或超过 40 N。

## 5. 逐点配对比较

| transition | points |
| --- | ---: |
| both safe | 17 |
| Contact only | 39 |
| ACT only | 7 |
| neither safe | 37 |

双侧 exact McNemar/binomial `p=0.00000183`。在这个固定 100 点集合上，数据支持 Contact-CVAE-zero 有更多 40 N safe-success 点。

task success 本身为 Contact 57/100、ACT 61/100，配对 `p=0.6271`。因此主要差异来自任务完成过程中的历史峰值力，而不是 task-success 数量。

## 6. 与50 N和80 N结果比较

| threshold | Contact safe | ACT safe | Contact minus ACT |
| ---: | ---: | ---: | ---: |
| 80 N | 57/100 | 35/100 | +22 pp |
| 50 N | 57/100 | 33/100 | +24 pp |
| 40 N | 56/100 | 24/100 | **+32 pp** |

从 50 N 降到 40 N：

- Contact 仅减少 point 73，其 `max_force=40.14 N`；
- ACT 减少 9 个 safe points，它们的 `max_force` 位于 41.41--48.43 N；
- 模型差异由 24 pp 扩大到 32 pp。

Contact 被重新分类的唯一成功点：

| point | x (mm) | z (mm) | radius (mm) | max force | mean force | >40 N steps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 73 | +2.601 | +0.463 | 2.641 | 40.14 N | 4.49 N | 1 |

该点仅有一个记录 step 高于 40 N，说明仅使用峰值阈值会对短暂越界非常敏感。是否将这种短暂越界与长时间高力暴露同等处理，应由任务安全规范决定。

## 7. 力分布背景

### 7.1 Task-success Rollout

| model | n | max-force mean | median | P95 | maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 57 | 23.25 N | 24.37 N | 35.77 N | 40.14 N |
| ACT baseline | 61 | 55.24 N | 47.26 N | 103.52 N | 105.86 N |

Contact 成功样本基本全部位于 40 N 以下。ACT 成功样本分布更宽，既有低力成功，也有 40--50 N、中间力和接近 100 N 的高力成功。

### 7.2 Task-failure Rollout

| model | failures | max-force mean | median | P95 | maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 43 | 35.72 N | 37.03 N | 49.64 N | 53.58 N |
| ACT baseline | 39 | 98.98 N | 100.12 N | 104.30 N | 110.93 N |

40 N safe success 仍然要求 task success；低力 task failure 不会被计为 safe。Contact point 63 的 `max_force=1.17 N`，但未完成任务，是这一点的直接例子。

## 8. 半径分析

表格为 `40 N safe successes / points`。

| model | [0,2) mm | [2,4) mm | [4,6) mm | total |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 18/18 | 35/61 | 3/21 | 56/100 |
| ACT baseline | 8/18 | 12/61 | 4/21 | 24/100 |

Contact 在 `[0,2)` mm 的 18 个点全部为 40 N safe success。ACT 在近中心区域只有 8/18 safe，说明 ACT 的高力 task success 并不局限于较大半径。

## 9. 方向与象限

### 9.1 半轴

| model | x < 0 | x >= 0 | z < 0 | z >= 0 |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 31/50 | 25/50 | 33/50 | 23/50 |
| ACT baseline | 1/50 | 23/50 | 19/50 | 5/50 |

### 9.2 象限

| model | +x +z | -x +z | -x -z | +x -z |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 10/23 | 13/27 | 18/23 | 15/27 |
| ACT baseline | 4/23 | 1/27 | 0/23 | 19/27 |

ACT 在 `x<0` 只有 1/50 个 40 N safe success，在 `-x,-z` 象限为 0/23；其 safe points 高度集中于 `+x,-z`。40 N 分类比 50 N 更明显地显示了这种方向集中，但其机制仍需轨迹、接触力方向与视频验证。

## 10. 40 N Target Maps

图采用相同的 +/-6 mm 范围、2 mm rings、marker size 和 300 DPI。绿色表示 `task success AND max_force < 40 N`，橙色表示 task success 但 `max_force >= 40 N`，红色表示 task failure。

### Contact-CVAE-zero

- [Publication PNG](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe40n.png)
- [Publication PDF](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe40n.pdf)
- [Labeled PNG](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe40n_labeled.png)

### ACT baseline

- [Publication PNG](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe40n.png)
- [Publication PDF](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe40n.pdf)
- [Labeled PNG](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe40n_labeled.png)

这些图只显示离散实测点，不插值连续成功区域。原 50 N 和 80 N 图片均未被覆盖。

## 11. 解释与限制

确认测量结果表明，40 N 下 Contact-CVAE-zero 的 safe-success rate 为 56%，ACT baseline 为 24%，差异为 32 pp。在已检查的候选阈值中，40 N 附近对这两个模型提供了较强区分度。

但不应因为差异更大就直接选择 40 N。阈值应由硬件容许载荷、任务接触要求、材料风险、传感器标定和真实机器人安全验证预先确定。特别是 Contact point 73 只有一个 step 略高于 40 N，说明峰值阈值与持续暴露指标可能给出不同安全解释。

主要局限性：

- 只有一个 100 点 point set；
- 40 N 是观察数据后的候选阈值；
- 10 个 rollout seeds 是相同的确定性重复；
- 没有对多阈值扫描进行统计校正；
- `max_force` 不包含力方向、冲量和接触位置；
- safe success 不是通用硬件安全保证。

## 12. 后续建议

1. 若工程上有独立依据支持 40 N，应在下一轮实验前固定该阈值。
2. 使用多个独立 point-set seeds 验证 56% 对 24% 的差异是否稳定。
3. 同时报告 40 N peak-force 分类和 `force_gt_40_steps`，区分短暂峰值与持续高力。
4. 对 Contact point 73 和 ACT 的 40--50 N 成功点检查完整力轨迹。
5. 正式安全分析应增加力冲量、轴向/横向分量和真实硬件载荷边界。

## 13. 结论

使用严格的 40 N safe-force boundary 后，Contact-CVAE-zero 为 56/100 safe success，ACT baseline 为 24/100，模型差异为 32 pp；逐点 exact McNemar `p=0.00000183`。Contact 只有 1 个 task success 因超过 40 N 被重新分类，ACT 则有 37 个。

与 50 N 相比，40 N 对模型区分更强，但这一结果属于单 point-set 的事后敏感性分析。是否正式采用 40 N，仍应由独立安全依据和多 point-set seed 验证决定。
