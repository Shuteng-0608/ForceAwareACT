# ForceAwareACT +/-4 mm 100点Rollout力阈值敏感性实验报告

## 1. 实验目的

本报告分析同一组 100 个 x/z +/-4 mm 孔位上 Contact-CVAE-zero 与 ACT baseline 的力表现，并回答：若将当前 `80 N` safe-force threshold 调低，两种模型的 safe-success 差异是否会进一步扩大。

这是基于既有 rollout 结果的事后阈值敏感性分析，不修改现有 `safe_success` 字段、成功判据、rollout 数据或模型行为。候选阈值统一定义为：

```text
safe success at threshold T
= task success AND max_force < T
```

`max_force` 是完整 rollout 历史中的最大力范数，不是成功时刻的瞬时力。

## 2. 数据范围

选取：

```text
point_set_seed = 20260702
rollout_seed_base = 31000
num_points = 100
action_select_mode = mid
x/z bounds = [-4, 4] mm
max_rollout_steps = 900
```

纳入 Contact-CVAE-zero 与 ACT baseline，实验根目录为：

```text
outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/
pointset_20260702/rollout_31000/
```

另外 9 个 rollout-seed bases 在所有非 provenance `grid_summary.csv` 字段上与本组逐点完全一致，连续指标最大绝对差为 0；抽查的第 1、25、50、75、100 点完整 rollout log 的 SHA256 也一致。因此其他 seed 只构成确定性复现证据，不作为额外独立样本。本报告的有效样本量是每个模型 100 个空间点，而不是 1000。

## 3. 实验协议与完整性

两个实验均完成 100/100 points，process error 为 0，并使用完全相同的 `point_index` 与 x/y/z offsets。

| setting | value |
| --- | --- |
| sampling | `latin_hypercube` |
| action mode / selection | `action` / `mid` |
| chunk length | 10 |
| policy rate | 30 Hz |
| max delta q | 0.02 rad |
| force stop threshold | 1000 N |
| task-success distance | <0.005 m |
| task-success lateral error | <0.006 m |
| task-success instantaneous force | <80 N |
| success hold | 15 steps |
| reported safe success | task success and historical `max_force < 80 N` |

需要区分：task success 要求成功条件中的瞬时力连续满足阈值；safe success 进一步要求整个 rollout 的历史最大力低于候选阈值。

## 4. 80 N原始结果

| model | task success | safe success at 80 N | task success, not safe | task failure |
| --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 57/100 | **57/100** | 0 | 43 |
| ACT baseline | 61/100 | **35/100** | 26 | 39 |

ACT baseline 的 task-success rate 比 Contact-CVAE-zero 高 4 个百分点，但 safe-success rate 低 22 个百分点。Contact 的 57 个 task success 全部满足 80 N safe-force 定义；ACT 的 61 个 task success 中有 26 个完整 rollout 峰值不低于 80 N。

| metric | rate | Wilson 95% CI |
| --- | ---: | ---: |
| Contact task/safe | 57% | 47.2%--66.3% |
| ACT task | 61% | 51.2%--70.0% |
| ACT safe at 80 N | 35% | 26.4%--44.7% |

逐点 task-success 转移为 both 40、Contact-only 17、ACT-only 21、both-fail 22，双侧 exact McNemar `p=0.6271`。safe-success 转移为 both 26、Contact-only 31、ACT-only 9、neither 34，`p=0.000680`。这支持两者的主要差异来自完整 rollout 力暴露，而不是 task-success 数量。

## 5. 最大力分布

### 5.1 全部Rollout

| model | mean | SD | Q25 | median | Q75 | P90 | P95 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 28.61 | 12.08 | 22.07 | 29.28 | 36.93 | 45.26 | 48.05 | 1.17 | 53.58 |
| ACT baseline | 72.30 | 38.75 | 41.63 | 97.17 | 100.98 | 102.94 | 103.80 | 2.54 | 110.93 |

单位均为 N。ACT baseline 的总体 `max_force` 中位数接近 100 N，明显高于其均值，反映结果中同时存在低力 rollout 和接近 100 N 的高力 rollout。

### 5.2 Task-success Rollout

| model | n | mean | SD | Q25 | median | Q75 | P90 | P95 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 57 | 23.25 | 8.60 | 16.24 | 24.37 | 29.92 | 33.46 | 35.77 | 7.55 | 40.14 |
| ACT baseline | 61 | 55.24 | 41.27 | 8.56 | 47.26 | 99.86 | 101.84 | 103.52 | 2.54 | 105.86 |

Contact 的成功样本峰值力连续分布在约 7.5--40.1 N。ACT 的成功样本则明显分群：21 个低于 20 N，随后有一组约 29--58 N 的样本，以及 26 个不低于 80 N 的高力成功样本。这种分群导致阈值敏感性不是单调的。

### 5.3 Task-failure Rollout

| model | n | max-force mean | median | P95 | min | max | mean-force mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 43 | 35.72 | 37.03 | 49.64 | 1.17 | 53.58 | 11.60 |
| ACT baseline | 39 | 98.98 | 100.12 | 104.30 | 84.13 | 110.93 | 63.12 |

ACT 的 39 个 task failures 全部具有 `max_force > 80 N`。其失败样本平均有 757.2 steps 高于 20 N、753.5 steps 高于 40 N；Contact failure 对应均值为 190.8 和 58.0 steps。这是高力暴露的直接测量结果，但不能仅凭汇总值断言具体接触机制。

## 6. Safe-force阈值敏感性

下表重新应用不同候选阈值，不修改原 CSV。差值定义为 Contact safe rate 减 ACT safe rate。95% CI 按每个模型 100 个唯一点计算。

| threshold | Contact safe | 95% CI | ACT safe | 95% CI | difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 N | 3% | 1.0%--8.5% | 19% | 12.5%--27.8% | -16 pp |
| 15 N | 13% | 7.8%--21.0% | 21% | 14.2%--30.0% | -8 pp |
| 20 N | 19% | 12.5%--27.8% | 21% | 14.2%--30.0% | -2 pp |
| 25 N | 30% | 21.9%--39.6% | 21% | 14.2%--30.0% | +9 pp |
| 30 N | 45% | 35.6%--54.8% | 22% | 15.0%--31.1% | +23 pp |
| 35 N | 53% | 43.3%--62.5% | 22% | 15.0%--31.1% | +31 pp |
| 40 N | 56% | 46.2%--65.3% | 24% | 16.7%--33.2% | **+32 pp** |
| 45 N | 57% | 47.2%--66.3% | 29% | 21.0%--38.5% | +28 pp |
| 50 N | 57% | 47.2%--66.3% | 33% | 24.6%--42.7% | +24 pp |
| 60 N | 57% | 47.2%--66.3% | 35% | 26.4%--44.7% | +22 pp |
| 80 N | 57% | 47.2%--66.3% | 35% | 26.4%--44.7% | +22 pp |

将阈值从 80 N 调低到 40 N，观察到的 safe-success 差异从 22 pp 扩大到 32 pp。40 N 下逐点转移为 both 17、Contact-only 39、ACT-only 7、neither 37，双侧 exact McNemar `p=0.00000183`。

但是，“阈值越低，差异越大”不成立。阈值降到 20 N 时两者只有 2 pp 差异；低于 15 N 时 ACT 的低力成功数反而更多。40 N 是本次事后扫描中观察到差异最大的附近阈值，而不是由独立安全规范预先确定的最优阈值。

ACT 在 58.43--87.76 N 之间没有 task-success `max_force`，所以 60--85 N 阈值产生相同的 35/100 safe-success 结果。阈值在这个区间内移动不会改变分类。

## 7. 40 N与80 N的空间影响

### 7.1 半径分组

| model / threshold | [0,2) mm | [2,4) mm | [4,6) mm | total |
| --- | ---: | ---: | ---: | ---: |
| Contact, 80 N | 18/18 | 36/61 | 3/21 | 57/100 |
| Contact, 40 N | 18/18 | 35/61 | 3/21 | 56/100 |
| ACT, 80 N | 8/18 | 21/61 | 6/21 | 35/100 |
| ACT, 40 N | 8/18 | 12/61 | 4/21 | 24/100 |

从 80 N 降到 40 N 时，Contact 只减少 1 个 `[2,4)` mm safe success；ACT 减少 11 个，其中 `[2,4)` mm 减少 9 个、`[4,6)` mm 减少 2 个。因此差异扩大并不是最外圈点单独驱动。

### 7.2 方向分组

| model / threshold | x < 0 | x >= 0 | z < 0 | z >= 0 |
| --- | ---: | ---: | ---: | ---: |
| Contact, 80 N | 31/50 | 26/50 | 33/50 | 24/50 |
| Contact, 40 N | 31/50 | 25/50 | 33/50 | 23/50 |
| ACT, 80 N | 1/50 | 34/50 | 19/50 | 16/50 |
| ACT, 40 N | 1/50 | 23/50 | 19/50 | 5/50 |

ACT 在 `x<0` 的 safe success 在两个阈值下都只有 1/50。降低阈值主要剔除了 ACT 的 `x>=0` 和 `z>=0` 高力成功点，因此 40 N 分类会进一步强化 target map 上的方向集中现象。该方向结果是测量事实；其成因仍需动作、接触力方向、机器人构型及视频验证。

## 8. 代表性点位

| model | case | point | x (mm) | z (mm) | radius (mm) | max force (N) | mean force (N) | >20 N steps | >40 N steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact | highest-force success | 73 | +2.601 | +0.463 | 2.641 | 40.14 | 4.49 | 6 | 1 |
| Contact | lowest-force success | 45 | -0.064 | +0.057 | 0.085 | 7.55 | 1.85 | 0 | 0 |
| Contact | highest-force failure | 35 | +2.127 | +3.911 | 4.452 | 53.58 | 19.92 | 350 | 324 |
| Contact | lowest-force failure | 63 | +2.743 | +2.061 | 3.431 | 1.17 | 1.13 | 0 | 0 |
| ACT | highest-force success | 53 | -0.822 | +0.493 | 0.959 | 105.86 | 46.19 | 381 | 358 |
| ACT | lowest-force success | 61 | +0.219 | +0.102 | 0.242 | 2.54 | 1.18 | 0 | 0 |
| ACT | highest-force failure | 86 | -3.357 | -3.749 | 5.032 | 110.93 | 55.45 | 660 | 634 |
| ACT | lowest-force failure | 72 | -2.461 | +3.238 | 4.067 | 84.13 | 63.88 | 786 | 784 |

Contact point 63 说明低力不自动意味着任务成功；它在全程低力情况下仍未满足任务条件。ACT point 53 则说明 task success 与低历史峰值力是不同维度。这些点适合进一步结合 rollout log 和视频检查，但当前报告不对物理行为作未经验证的分类。

## 9. Target Map

现有 80 N safe-success target maps：

- Contact-CVAE-zero：[PNG](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success.png)
- ACT baseline：[PNG](../../outputs/peg_hole_100/contact_zero_vs_act_100points_10rolloutseeds/pointset_20260702/rollout_31000/hole_lhs_100_xz_4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success.png)

这些图仍使用原始 80 N 定义。本报告没有生成或覆盖 40 N 图，以避免将事后候选阈值与正式实验判据混淆。

## 10. 解释与阈值选择原则

确认测量结果支持：

1. Contact 的 task-success 峰值力主要位于 7.5--40.1 N，57 个成功点中 56 个低于 40 N。
2. ACT 的成功峰值力呈宽范围和明显分群，61 个成功点中只有 24 个低于 40 N，26 个不低于 80 N。
3. 将阈值从 80 N 降到 40 N 会使本点集上的 safe-success 差异由 22 pp 扩大到 32 pp。
4. 阈值效应非单调；阈值过低时 ACT 的低力成功分群占优，不能概括为阈值越严格越有利于 Contact。

不应为了获得更大的模型差异而选择 40 N。safe-force threshold 应优先来自硬件允许载荷、任务接触规范、材料风险、传感器校准和真实机器人验证，并在观察测试结果前预先固定。若工程上有独立依据支持 40 N，则当前数据表明它会更强地区分两种模型；如果没有该依据，40 N 只能作为探索性候选值。

## 11. 局限性

- 只有一个 100 点 point set，不能估计 point-set 间变异。
- 10 个 rollout seeds 是完全相同的确定性重复，不能增加有效样本量。
- 候选阈值是在看到数据后扫描得到，没有独立验证集，也没有为多阈值搜索做统计校正。
- `max_force` 只保留峰值，不区分峰值方向、冲量、接触位置和持续时间。
- `force_gt_20_steps` 与 `force_gt_40_steps` 提供持续暴露线索，但不能替代完整力轨迹分析。
- safe success 不是通用硬件安全保证。

## 12. 后续建议

1. 在工程安全依据允许时，将 `40 N` 作为预先声明的候选阈值，而不是根据模型排名事后确定。
2. 使用至少 5--10 个独立 point-set seeds 重复 100 点测试；rollout seed 固定即可。
3. 同时固定报告 40 N、60 N、80 N 三个预先指定阈值，或者报告完整 empirical safe-success curve，避免只呈现最有利阈值。
4. 对 ACT 的高力成功点和 Contact point 63 这类低力失败点检查完整力轨迹、动作裁剪、接触状态与视频。
5. 若目标是实际安全评估，应进一步加入峰值持续时间、力冲量、轴向/横向力和真实机器人载荷限制。

## 13. 结论

在这个固定的 100 点 +/-4 mm 测试集上，将 safe-force threshold 从 80 N 降到 40 N，Contact-CVAE-zero 与 ACT baseline 的 safe-success 差异确实从 22 pp 扩大到 32 pp。该变化来自 Contact 成功样本几乎全部低于 40 N，而 ACT 有大量中高峰值力成功样本。

但阈值敏感性是非单调的：20 N 时两者接近，10--15 N 时 ACT 反而更高。因此更准确的结论是：**40 N 附近在本点集上提供了更强的区分度，而不是阈值越低差异越大。**是否采用更低阈值必须由独立安全依据和更多 point-set seeds 验证，不能仅根据本次事后模型差异决定。
