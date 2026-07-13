# ForceAwareACT 五模型十组孔位种子 ±4 mm rollout 实验报告

## 1. 实验目的与范围

本报告整理 `outputs/peg_hole_100/pointsets_10` 下五种策略配置在 x/z 孔位偏移任务上的 rollout 结果，比较任务成功率、40 N 安全成功率、跨 point-set 稳定性、力峰值和空间方向差异。

纳入配置如下：

| 报告名称 | suite model key | contact latent | action selection |
| --- | --- | --- | --- |
| Contact-CVAE-zero | `contact_cvae` | zero | mid |
| Contact-CVAE-prior | `contact_cvae_prior` | prior sample | mid |
| Motion-CVAE | `motion_cvae` | zero | mid |
| DualZero | `dualzero` | zero | mid |
| ACT baseline | `act_baseline` | zero | mid |

所有配置使用相同实验协议：

- point-set seeds：`20260707` 至 `20260716`，共 10 组；
- 每组 100 个 Latin hypercube x/z 孔位点，共 1000 个点/模型；
- x/z 偏移范围：各轴 `[-4, +4] mm`，y 偏移为 0；
- rollout seed base：`31000`；
- action selection：`mid`；
- `max_rollout_steps=900`，`max_delta_q=0.02`；
- 每点仅运行一次，因此总计 `10 × 100 × 5 = 5000` 次 rollout。

本报告只使用本地 `suite_plan.json`、`grid_manifest.json`、`task_points.csv` 和 `grid_summary.csv` 中的实测记录，不对离散点之间的连续成功区域进行插值。

## 2. 指标定义

manifest 中记录的任务成功条件为：

- `success_distance_threshold=0.005 m`；
- `success_lateral_threshold=0.006 m`；
- `success_force_threshold=40 N`；
- `success_hold_steps=15`。

报告采用以下口径：

- **task success**：rollout 达到任务成功条件；
- **safe success <40 N**：`task success=True` 且完整 rollout 历史的 `max_force < 40 N`；
- **unsafe task success**：任务成功，但完整历史 `max_force >= 40 N`；
- **task failure**：900 步内未达到任务成功条件。

这里的 safe success 是本仿真实验的操作性指标，不等价于真实机器人上的通用安全保证。

## 3. 数据完整性

- `suite_plan.json` 包含预期的 10 个 point-set 配置和 5 个模型；
- 50/50 个实验均存在 `grid_summary.csv`、`grid_manifest.json` 和 `task_points.csv`；
- 5000/5000 个点均完成，完成率 100%；
- 10/10 个 seed configuration 均为 complete；
- process error 合计为 0；
- 所有未成功点的 stop reason 均为 `max_rollout_steps`，没有异常停止类型；
- 同一 point-set seed 内五个模型的孔位坐标完全一致，可以进行逐点配对比较。

## 4. 总体结果

| configuration | task success | task Wilson 95% CI | safe success <40 N | safe Wilson 95% CI | safe seed mean ± SD | unsafe task success | task failure | safe/task |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Contact-CVAE-zero | 614/1000 (61.4%) | 58.3%–64.4% | 590/1000 (59.0%) | 55.9%–62.0% | 59.0% ± 3.2 pp | 24 | 386 | 96.1% |
| ACT baseline | 595/1000 (59.5%) | 56.4%–62.5% | 295/1000 (29.5%) | 26.8%–32.4% | 29.5% ± 3.7 pp | 300 | 405 | 49.6% |
| Contact-CVAE-prior | 549/1000 (54.9%) | 51.8%–58.0% | 266/1000 (26.6%) | 24.0%–29.4% | 26.6% ± 2.8 pp | 283 | 451 | 48.5% |
| DualZero | 502/1000 (50.2%) | 47.1%–53.3% | 238/1000 (23.8%) | 21.3%–26.5% | 23.8% ± 4.1 pp | 264 | 498 | 47.4% |
| Motion-CVAE | 589/1000 (58.9%) | 55.8%–61.9% | 223/1000 (22.3%) | 19.8%–25.0% | 22.3% ± 2.7 pp | 366 | 411 | 37.9% |

按 task success 排名为 Contact-CVAE-zero、ACT baseline、Motion-CVAE、Contact-CVAE-prior、DualZero。按 40 N safe success 排名为 Contact-CVAE-zero、ACT baseline、Contact-CVAE-prior、DualZero、Motion-CVAE。

最关键的差异不只是 Contact-CVAE-zero 的 task success 最高，而是其 614 次任务成功中有 590 次满足完整轨迹小于 40 N，safe/task 比例达到 96.1%。其他四种配置的 safe/task 比例只有 37.9%–49.6%。

## 5. 跨 point-set 稳定性

下表单元格为每个 100 点 point set 的 `task success / safe success` 数量。

| point-set seed | Contact-zero | Contact-prior | Motion-CVAE | DualZero | ACT baseline |
| --- | --- | --- | --- | --- | --- |
| 20260707 | 67 / 62 | 57 / 27 | 65 / 23 | 52 / 31 | 60 / 29 |
| 20260708 | 65 / 62 | 56 / 31 | 56 / 17 | 43 / 21 | 59 / 33 |
| 20260709 | 62 / 61 | 56 / 26 | 60 / 21 | 53 / 21 | 56 / 32 |
| 20260710 | 65 / 63 | 56 / 29 | 55 / 22 | 51 / 25 | 63 / 33 |
| 20260711 | 63 / 60 | 53 / 28 | 57 / 21 | 47 / 21 | 54 / 29 |
| 20260712 | 57 / 56 | 57 / 28 | 59 / 25 | 48 / 24 | 67 / 26 |
| 20260713 | 63 / 60 | 57 / 23 | 54 / 21 | 48 / 18 | 59 / 24 |
| 20260714 | 57 / 55 | 53 / 28 | 59 / 27 | 56 / 28 | 60 / 29 |
| 20260715 | 59 / 56 | 54 / 24 | 56 / 22 | 49 / 21 | 55 / 25 |
| 20260716 | 56 / 55 | 50 / 22 | 68 / 24 | 55 / 28 | 62 / 35 |

Contact-CVAE-zero 在全部 10 个 point sets 上都获得最高的 safe-success 数量，范围为 55–63/100。其最差 point set 的 safe-success 率仍为 55%，高于其他模型各自最好的 point set：ACT 35%、Contact-prior 31%、DualZero 31%、Motion-CVAE 27%。

## 6. 逐点配对比较

每组比较包含相同的 1000 个孔位点。表中 `A-only/B-only` 表示配对结果不一致的点；p 值为基于 discordant pairs 的双侧 exact McNemar/binomial 检验。该统计检验用于描述本批固定协议下的差异。

### 6.1 以 Contact-CVAE-zero 为 A

| B | metric | both success | A-only | B-only | both fail | A−B rate | exact p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-prior | task | 526 | 88 | 23 | 363 | +6.5 pp | 3.80e-10 |
| Contact-CVAE-prior | safe40 | 248 | 342 | 18 | 392 | +32.4 pp | 9.40e-79 |
| Motion-CVAE | task | 464 | 150 | 125 | 261 | +2.5 pp | 0.148 |
| Motion-CVAE | safe40 | 196 | 394 | 27 | 383 | +36.7 pp | 1.11e-84 |
| DualZero | task | 446 | 168 | 56 | 330 | +11.2 pp | 3.44e-14 |
| DualZero | safe40 | 221 | 369 | 17 | 393 | +35.2 pp | 2.45e-87 |
| ACT baseline | task | 422 | 192 | 173 | 213 | +1.9 pp | 0.346 |
| ACT baseline | safe40 | 216 | 374 | 79 | 331 | +29.5 pp | 5.97e-47 |

Contact-CVAE-zero 与 ACT baseline、Motion-CVAE 的 task-success 差异在该配对检验下不显著，但 safe-success 差异很大。换言之，主要优势来自任务成功过程中的低力表现，而不只是终态任务完成数量。

Contact-CVAE-zero 与 Contact-CVAE-prior 使用同一 Contact-CVAE checkpoint，主要推理配置差异是 contact latent 使用 zero 还是 prior sample。在当前协议下，zero 配置比 prior 高 6.5 pp task success 和 32.4 pp safe success。结果支持 zero latent 在本任务中更稳健，但不能自动推广到其他 seed、偏移范围或 action-selection 模式。

### 6.2 其他模型相对 ACT baseline

| A | metric | A-only | ACT-only | A−ACT rate | exact p |
| --- | --- | ---: | ---: | ---: | ---: |
| Contact-CVAE-prior | task | 176 | 222 | −4.6 pp | 0.0240 |
| Contact-CVAE-prior | safe40 | 119 | 148 | −2.9 pp | 0.0864 |
| Motion-CVAE | task | 191 | 197 | −0.6 pp | 0.800 |
| Motion-CVAE | safe40 | 147 | 219 | −7.2 pp | 1.98e-4 |
| DualZero | task | 167 | 260 | −9.3 pp | 7.86e-6 |
| DualZero | safe40 | 112 | 169 | −5.7 pp | 8.06e-4 |

## 7. 偏移半径分析

表格单元为 `task successes / safe successes / points`。每个模型在各半径区间使用同一批点。

| radial offset | Contact-zero | Contact-prior | Motion-CVAE | DualZero | ACT baseline |
| --- | --- | --- | --- | --- | --- |
| [0, 2) mm | 191 / 187 / 196 | 189 / 91 / 196 | 177 / 119 / 196 | 196 / 131 / 196 | 176 / 99 / 196 |
| [2, 4) mm | 394 / 375 / 603 | 326 / 159 / 603 | 366 / 94 / 603 | 283 / 103 / 603 | 350 / 161 / 603 |
| [4, 6) mm | 29 / 28 / 201 | 34 / 16 / 201 | 46 / 10 / 201 | 23 / 4 / 201 | 69 / 35 / 201 |

Contact-CVAE-zero 在三个半径区间的 safe-success 数均最高。ACT baseline 在最外层 `[4, 6) mm` 的 task-success 数最高，但其中 69 次 task success 只有 35 次满足 40 N 安全成功。最大可达径向偏移约为 `sqrt(4²+4²)=5.66 mm`；表中的最大观测点不能解释为连续空间的成功边界。

## 8. 象限与方向分析

表格单元为 `task successes / safe successes / points`。

| x/z quadrant | Contact-zero | Contact-prior | Motion-CVAE | DualZero | ACT baseline |
| --- | --- | --- | --- | --- | --- |
| +x, +z | 126 / 114 / 253 | 117 / 96 / 253 | 138 / 38 / 253 | 144 / 76 / 253 | 177 / 80 / 253 |
| +x, −z | 144 / 144 / 247 | 126 / 92 / 247 | 125 / 48 / 247 | 92 / 54 / 247 | 247 / 199 / 247 |
| −x, +z | 140 / 130 / 247 | 120 / 21 / 247 | 141 / 88 / 247 | 137 / 61 / 247 | 34 / 1 / 247 |
| −x, −z | 204 / 202 / 253 | 186 / 57 / 253 | 185 / 49 / 253 | 129 / 47 / 253 | 137 / 15 / 253 |

ACT baseline 表现出最强的方向不对称：在 `+x,−z` 象限几乎全部 task success，但在 `−x,+z` 象限仅 34/247 task success、1/247 safe success。Contact-CVAE-zero 的四象限 safe-success 更均衡，且在两个负 x 象限显著高于 ACT。Contact-prior、Motion-CVAE 和 DualZero 也存在不同程度的方向差异。具体物理原因仍需结合轨迹、接触力方向与视频判断。

## 9. 力峰值与成功时间

下表的力统计为全部 1000 个 rollout 的 `max_force mean / median / P95 / max`。成功时间为 task-success 子集的中位数。

| configuration | max force mean / median / P95 / max (N) | task-success median time (s) |
| --- | --- | ---: |
| Contact-CVAE-zero | 28.15 / 28.22 / 46.76 / 92.20 | 6.732 |
| Contact-CVAE-prior | 53.11 / 47.70 / 94.50 / 106.43 | 6.600 |
| Motion-CVAE | 50.62 / 55.06 / 90.86 / 98.56 | 8.481 |
| DualZero | 59.42 / 60.90 / 94.72 / 102.33 | 6.732 |
| ACT baseline | 71.54 / 97.48 / 105.20 / 108.89 | 6.270 |

Contact-CVAE-zero 的全体 max-force 均值、median 和 P95 均为最低。ACT baseline 的成功中位时间最短，但全体 max-force median 达 97.48 N；因此不能只用完成速度评价策略质量。Motion-CVAE 的成功中位时间最长，同时 safe-success 最低。

任务成功子集中的 unsafe-success 数量进一步说明差异：Contact-zero 仅 24 次，而 Contact-prior、Motion-CVAE、DualZero、ACT 分别为 283、366、264、300 次。

## 10. 主要发现

1. **完整性结果：**五模型共 5000 次 rollout 全部完成，未观察到 process error，模型间点位严格配对。
2. **总体结果：**Contact-CVAE-zero 同时取得最高 task success（61.4%）和最高 40 N safe success（59.0%）。
3. **低力优势：**Contact-CVAE-zero 的 safe/task 比例为 96.1%，显著高于其他配置的 37.9%–49.6%；其优势主要体现在成功轨迹的力峰值控制。
4. **稳健性结果：**Contact-CVAE-zero 在全部 10 个 point sets 和全部三个半径区间上均取得最高 safe-success 数量。
5. **latent 对比：**同一 Contact-CVAE checkpoint 下，zero latent 明显优于 prior sampling，尤其是 safe success（59.0% 对 26.6%）。
6. **task 与 safety 排名不同：**Motion-CVAE 的 task success 接近 ACT baseline，但其 safe success 为五种配置最低，说明任务完成率不能替代安全成功率。
7. **方向差异：**ACT baseline 对 x/z 象限高度敏感；Contact-CVAE-zero 的安全成功分布更均衡。

## 11. 局限性

- 所有结果来自 MuJoCo 仿真，40 N 指标不是硬件安全认证或真实机器人安全保证；
- 仅测试 `mid` action selection，不能推断 temporal 模式下的相对表现；
- 每个点只有一次 rollout；虽然有 10 组 point-set seeds，但只有一个 rollout-seed base，尚未充分分离策略随机性与孔位采样变化；
- pooled Wilson 区间把 rollout 当作二项观测；同一 point set 内样本和共享 rollout seed 结构可能引入相关性，因此区间和 exact p 值应作为本实验协议下的描述性/探索性统计；
- 五个模型并非全部只改变一个因素。除 Contact-zero 与 Contact-prior 外，跨 checkpoint 比较同时包含架构、训练目标或训练过程差异，不能解释为单一组件的因果效应；
- 尚未对逐步轨迹、接触法向力、动作序列或视频进行行为级分类，因此不能据此断言某模型学会了具体的搜索、撤退或力释放机制；
- 本实验只覆盖各轴 ±4 mm 的离散 LHS 样本，不能外推到更大偏移范围或连续工作空间。

## 12. 复现实验与数据位置

实验协议：

- `outputs/peg_hole_100/pointsets_10/suite_plan.json`

suite 汇总：

- `outputs/peg_hole_100/pointsets_10/summary/aggregate_summary.csv`
- `outputs/peg_hole_100/pointsets_10/summary/per_seed_summary.csv`

重新聚合现有结果可执行：

```bash
python scripts/run_xz_multiseed_rollout_suite.py \
  --point-set-seeds \
    20260707 20260708 20260709 20260710 20260711 \
    20260712 20260713 20260714 20260715 20260716 \
  --rollout-seed-bases 31000 \
  --models contact_cvae contact_cvae_prior motion_cvae dualzero act_baseline \
  --action-select-modes mid \
  --num-points 100 \
  --offset-mm 4 \
  --max-rollout-steps 900 \
  --output-base outputs/peg_hole_100/pointsets_10 \
  --aggregate-only
```

## 13. 结论

在本次 ±4 mm、10 个独立 point sets、每模型 1000 次 mid rollout 的仿真实验范围内，**Contact-CVAE-zero 是综合表现最好的配置**。它不仅 task-success 最高，而且几乎所有任务成功都满足完整轨迹 `max_force < 40 N`，并在不同 point set、半径区间和空间象限上保持更稳定的安全成功表现。

ACT baseline 和 Motion-CVAE 的 task-success 与 Contact-CVAE-zero 相对接近，但二者存在大量高于 40 N 的任务成功点。因而，本批结果最明确的结论是：Contact-CVAE-zero 的优势主要是**低力完成任务的稳健性**，而不是单纯扩大任务成功数量。下一步最有价值的验证是增加多个独立 rollout-seed bases，并对配对分歧点进行轨迹和视频级分析。
