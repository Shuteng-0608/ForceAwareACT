# ForceAwareACT +/-4 mm 100点Rollout 50 N Safe-success评估报告

## 1. 评估目的

本报告使用 50 N 作为 safe-force boundary，重新评估 Contact-CVAE-zero 与 ACT baseline 在同一组 100 个 +/-4 mm 孔位上的结果，并生成按 50 N 重新分类的 target maps。

本次评估不重跑 rollout，也不修改原始 `grid_summary.csv`。50 N safe success 由原始实测字段重新计算：

```text
safe_success_50n = task success AND max_force < 50 N
```

这里使用严格小于号；`max_force = 50 N` 不属于 safe success。原实验 `safe_success` 字段仍保留其 80 N 定义。

## 2. 数据范围

选取一个有效的 100 点实验：

```text
point_set_seed = 20260702
rollout_seed_base = 31000
action_select_mode = mid
x/z range = [-4, 4] mm
num_points = 100
max_rollout_steps = 900
```

比较模型：

- Contact-CVAE-zero；
- ACT baseline。

其他 9 个 rollout-seed bases 的逐点科学结果与本组完全一致，因此不重复计入样本量。每个模型的有效样本量为 100 个唯一空间点。

## 3. 完整性与可比性

- 两个模型均完成 100/100 points；
- process error 均为 0；
- `point_index` 和 x/y/z offsets 完全相同；
- task-success 判据、模型 XML、动作模式、最大步数和控制参数一致；
- 模型间可以进行逐点配对比较。

本报告只改变派生的 safe-force 分类阈值。原始 task-success 判据中的瞬时 80 N 条件没有被重新执行；但任何 `max_force < 50 N` 的 task success 必然在完整 rollout 中都低于 50 N。

## 4. 50 N总体结果

| model | task success | safe success <50 N | task success >=50 N | task failure |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 57/100 | **57/100 (57%)** | 0 | 43 |
| ACT baseline | 61/100 | **33/100 (33%)** | 28 | 39 |

基于 100 个唯一点的 Wilson 95% CI：

| metric | rate | Wilson 95% CI |
| --- | ---: | ---: |
| Contact safe <50 N | 57% | 47.2%--66.3% |
| ACT safe <50 N | 33% | 24.6%--42.7% |

确认测量结果：Contact 的 safe-success rate 比 ACT 高 24 个百分点。Contact 的 57 个 task successes 全部满足 50 N；ACT 的 61 个 task successes 中有 28 个达到或超过 50 N。

## 5. 逐点配对比较

50 N safe-success 转移为：

| transition | points |
| --- | ---: |
| both safe | 25 |
| Contact only | 32 |
| ACT only | 8 |
| neither safe | 35 |

双侧 exact McNemar/binomial `p=0.000182`。在这个预先固定的 100 点集合上，数据支持 Contact-CVAE-zero 具有更多 50 N safe-success 点。

task success 本身为 Contact 57/100、ACT 61/100，配对检验 `p=0.6271`，没有支持 task-success 数量存在差异。因此主要模型差异仍然来自完整 rollout 最大力，而不是任务完成数。

## 6. 与80 N结果比较

| model | safe at 80 N | safe at 50 N | change |
| --- | ---: | ---: | ---: |
| Contact-CVAE-zero | 57/100 | 57/100 | 0 |
| ACT baseline | 35/100 | 33/100 | -2 |
| Contact minus ACT | +22 pp | +24 pp | +2 pp |

将阈值从 80 N 降到 50 N 后，模型差异只扩大 2 个百分点。原因是 ACT 的 task-success `max_force` 在 50--80 N 范围内只有两个点：

| point | x (mm) | z (mm) | max force |
| ---: | ---: | ---: | ---: |
| 6 | +3.491 | +2.715 | 58.43 N |
| 65 | +1.747 | +2.195 | 51.91 N |

Contact 的最高 task-success `max_force` 为 40.14 N，因此从 80 N 降至 50 N 不改变其任何成功点分类。

## 7. 力分布

### 7.1 全部Rollout

| model | max-force mean | median | P95 | maximum |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 28.61 N | 29.28 N | 48.05 N | 53.58 N |
| ACT baseline | 72.30 N | 97.17 N | 103.80 N | 110.93 N |

### 7.2 Task-success Rollout

| model | n | max-force mean | median | P95 | maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 57 | 23.25 N | 24.37 N | 35.77 N | 40.14 N |
| ACT baseline | 61 | 55.24 N | 47.26 N | 103.52 N | 105.86 N |

ACT 成功样本同时包含低力成功和接近 100 N 的高力成功，分布范围明显更宽。50 N 将 ACT 的中高力成功归为橙色，但仍保留 33 个低于 50 N 的绿色成功点。

## 8. 半径分析

表格为 `50 N safe successes / points`。

| model | [0,2) mm | [2,4) mm | [4,6) mm | total |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 18/18 | 36/61 | 3/21 | 57/100 |
| ACT baseline | 8/18 | 20/61 | 5/21 | 33/100 |

Contact 在近中心 `[0,2)` mm 的 18 个点全部为 50 N safe success。ACT 在该区域只有 8/18 safe，说明 ACT 的高力成功并不只发生在大半径位置。

## 9. 方向与象限

### 9.1 半轴

| model | x < 0 | x >= 0 | z < 0 | z >= 0 |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 31/50 | 26/50 | 33/50 | 24/50 |
| ACT baseline | 1/50 | 32/50 | 19/50 | 14/50 |

### 9.2 象限

| model | +x +z | -x +z | -x -z | +x -z |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 11/23 | 13/27 | 18/23 | 15/27 |
| ACT baseline | 13/23 | 1/27 | 0/23 | 19/27 |

ACT 在 `x<0` 只有 1/50 个 50 N safe success，在 `-x,-z` 象限为 0/23；其 safe points 主要集中在 `x>=0`。50 N target map 因此仍显示明显方向不对称。该空间分布是实测结果，其机制仍需轨迹、接触力方向和视频验证。

## 10. 50 N Target Maps

所有图使用相同的 +/-6 mm 显示范围、2 mm rings、marker size 和 300 DPI。绿色表示 `task success AND max_force < 50 N`，橙色表示 task success 但 `max_force >= 50 N`，红色表示 task failure。

### Contact-CVAE-zero

- [Publication PNG](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe50n.png)
- [Publication PDF](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe50n.pdf)
- [Labeled PNG](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe50n_labeled.png)

### ACT baseline

- [Publication PNG](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe50n.png)
- [Publication PDF](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe50n.pdf)
- [Labeled PNG](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe50n_labeled.png)

图中只显示离散实测点，不插值连续成功区域。

## 11. 解释与限制

确认测量结果表明，50 N 下 Contact-CVAE-zero 的 safe-success rate 为 57%，ACT baseline 为 33%，差异为 24 pp。相比原 80 N 结果，差异只增加 2 pp，而不是大幅扩大。

可能的解释是 ACT 成功样本具有明显的低力和高力分群；但这只是对分布形态的描述，不能据此断言 ACT 发生了某种具体接触行为。行为机制仍需检查逐步力、动作、接触状态和视频。

局限性包括：

- 只有一个 100 点 point set；
- 50 N 是看过数据后的候选阈值；
- 其他 rollout seeds 是完全相同的确定性重复；
- safe success 只基于历史最大力，不覆盖冲量、方向、持续时间和真实硬件载荷；
- 50 N 不是未经硬件依据即可采用的通用安全边界。

## 12. 结论

使用严格的 50 N safe-force boundary 后，Contact-CVAE-zero 为 57/100 safe success，ACT baseline 为 33/100，逐点配对差异具有探索性统计证据（`p=0.000182`）。Contact 的所有 task successes 都保持 safe；ACT 有 28 个 task successes 因峰值达到或超过 50 N 被重新分类为非 safe。

50 N 相比 80 N 仅多剔除 ACT 的两个成功点，因此模型差异从 22 pp 小幅扩大到 24 pp。该阈值可作为后续多 point-set seed 实验中的预先声明候选值，但正式采用仍应由独立的硬件与任务安全依据决定。
