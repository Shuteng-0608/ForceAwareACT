# ForceAwareACT 固定 Fibonacci 圆盘 100 点五模型 Rollout 实验报告

## 1. 实验目的

本实验在同一组固定的 100 个 x/z 孔位偏移点上比较五种策略配置。点位位于半径 4 mm 的圆盘内，使用 deterministic Fibonacci disk 方法生成。实验同时评价任务完成能力、完整轨迹 40 N 安全成功率、峰值力、成功时间以及空间方向差异。

纳入配置如下：

| 报告名称 | suite model key | contact latent | action selection |
| --- | --- | --- | --- |
| Contact-CVAE-zero | `contact_cvae` | zero | mid |
| Contact-CVAE-prior | `contact_cvae_prior` | prior sample | mid |
| Motion-CVAE | `motion_cvae` | zero | mid |
| DualZero | `dualzero` | zero | mid |
| ACT baseline | `act_baseline` | zero | mid |

本报告只使用本地 `grid_manifest.json`、`task_points.csv` 和 `grid_summary.csv` 中的实测记录，不对离散点之间的连续成功区域进行插值。

## 2. 固定 100 点实验设计

第 `i=0,...,99` 个点使用下式生成：

```text
r_i     = 4 mm × sqrt((i + 0.5) / 100)
theta_i = i × pi × (3 - sqrt(5))
x_i     = r_i × cos(theta_i)
z_i     = r_i × sin(theta_i)
```

平方半径位于 100 个等面积单元的中心，方位角按 golden angle 递增。因此该点集在圆盘面积上近似均匀，同时固定、可复现，不依赖随机 seed。

- 点数：100；
- 圆盘半径上限：4 mm；
- 最小观测半径：0.2828 mm；
- 最大观测半径：3.98999 mm；
- `[0,2) mm`：25 点；
- `[2,4) mm`：75 点；
- y 偏移：0；
- 点位文件：`configs/experiments/fibonacci_disk_100_r4mm.csv`。

由于采用 cell-centered 半径，本点集不含精确中心点或恰好位于 4 mm 边界的点。

### 2.1 100 点位图

颜色表示径向偏移，数字为固定 `point_index`。

![固定 Fibonacci 圆盘 100 点位图](../../outputs/peg_hole_100/fibonacci_disk_100_r4mm/fibonacci_disk_100_r4mm_points_labeled.png)

## 3. Rollout 协议与指标

五种配置使用相同协议：

```text
sampling_mode = file
task_points_csv = configs/experiments/fibonacci_disk_100_r4mm.csv
rollout_seed_base = 31000
action_select_mode = mid
max_rollout_steps = 900
max_delta_q = 0.02
success_distance_threshold = 0.005 m
success_lateral_threshold = 0.006 m
success_force_threshold = 40 N
success_hold_steps = 15
```

每个模型对每个点运行一次，共 `100 × 5 = 500` 次 rollout。point `i` 使用 `rollout_seed=31000+i-1`。

报告采用以下指标：

- **task success**：rollout 达到任务成功条件；
- **safe success <40 N**：task success，且完整 rollout 历史的 `max_force < 40 N`；
- **unsafe task success**：任务成功，但完整历史 `max_force >= 40 N`；
- **task failure**：900 步内未达到任务成功条件。

这里的 safe success 是本次 MuJoCo 实验的操作性指标，不等价于真实机器人上的通用安全保证。

## 4. 数据完整性与可比性

- 五个实验均完成 100/100 点，共 500/500 次 rollout；
- 未观察到 process error；
- 所有 task failure 的 stop reason 均为 `max_rollout_steps`；
- 五个模型的 `point_index` 和 x/y/z 坐标逐行完全一致；
- checkpoint、contact latent 配置和动作选择方式均记录在各自 manifest 中；
- Contact-CVAE-zero 的前两个点在续跑时标为 `skipped_existing`，对应 `summary.json` 和逐步日志完整，已正常纳入统计；
- 五个模型可以进行逐点配对比较。

## 5. 总体结果

| configuration | task success | task Wilson 95% CI | safe success <40 N | safe Wilson 95% CI | unsafe task success | task failure | safe/task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 69/100 (69%) | 59.4%–77.2% | **67/100 (67%)** | 57.3%–75.4% | 2 | 31 | **97.1%** |
| Contact-CVAE-prior | 62/100 (62%) | 52.2%–70.9% | 26/100 (26%) | 18.4%–35.4% | 36 | 38 | 41.9% |
| Motion-CVAE | **72/100 (72%)** | 62.5%–79.9% | 30/100 (30%) | 21.9%–39.6% | 42 | 28 | 41.7% |
| DualZero | 65/100 (65%) | 55.3%–73.6% | 31/100 (31%) | 22.8%–40.6% | 34 | 35 | 47.7% |
| ACT baseline | 67/100 (67%) | 57.3%–75.4% | 33/100 (33%) | 24.6%–42.7% | 34 | 33 | 49.3% |

按 task success 排名为 Motion-CVAE、Contact-CVAE-zero、ACT baseline、DualZero、Contact-CVAE-prior。按 40 N safe success 排名则为 Contact-CVAE-zero、ACT baseline、DualZero、Motion-CVAE、Contact-CVAE-prior。

Motion-CVAE 的 task success 最高，为 72%，但其中 42 次超过完整轨迹 40 N 峰值限制，safe success 只有 30%。Contact-CVAE-zero 的 task success 为 69%，其中只有 2 次属于 unsafe task success，因此 safe success 达到 67%。本实验最明显的差异是低力完成能力，而不是单纯的任务完成数量。

## 6. 偏移半径分析

表格单元为 `task successes / safe successes / points`。

| radial offset | Contact-zero | Contact-prior | Motion-CVAE | DualZero | ACT baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[0,2) mm` | 24 / 24 / 25 | 24 / 9 / 25 | 24 / 15 / 25 | 25 / 18 / 25 | 24 / 14 / 25 |
| `[2,4) mm` | 45 / 43 / 75 | 38 / 17 / 75 | 48 / 15 / 75 | 40 / 13 / 75 | 43 / 19 / 75 |

Contact-CVAE-zero 在内层 25 个点中取得 24 次 task success，且全部 24 次均为 safe success；在外层 75 个点中取得 43 次 safe success。其他四种配置虽然在内层也有 24–25 次 task success，但只有 9–18 次满足完整轨迹 40 N 限制。

## 7. 象限与方向分析

表格单元为 `task successes / safe successes / points`。point 1 位于 `+x` 轴，单独列为 axis。

| x/z region | Contact-zero | Contact-prior | Motion-CVAE | DualZero | ACT baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| `+x,+z` | 14 / 13 / 24 | 12 / 9 / 24 | 13 / 4 / 24 | 17 / 10 / 24 | 19 / 10 / 24 |
| `-x,+z` | 16 / 15 / 25 | 13 / 2 / 25 | 20 / 13 / 25 | 20 / 6 / 25 | 6 / 0 / 25 |
| `-x,-z` | 22 / 22 / 24 | 20 / 4 / 24 | 24 / 6 / 24 | 16 / 7 / 24 | 15 / 2 / 24 |
| `+x,-z` | 16 / 16 / 26 | 16 / 10 / 26 | 14 / 7 / 26 | 11 / 7 / 26 | 26 / 20 / 26 |
| axis | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 0 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |

ACT baseline 呈现最强的方向不对称：在 `+x,-z` 区域为 26/26 task success、20/26 safe success，而在 `-x,+z` 区域仅 6/25 task success、0/25 safe success。Contact-CVAE-zero 的四象限 safe success 更均衡，在 `-x,-z` 和 `+x,-z` 区域的 task successes 全部满足 40 N 安全条件。

## 8. 逐点配对比较

以 Contact-CVAE-zero 为 A。`A-only/B-only` 表示相同点位上两模型结果不一致的数量。p 值为 discordant pairs 上的双侧 exact McNemar/binomial 检验。

| B | metric | both positive | A-only | B-only | both negative | A−B rate | exact p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-prior | task | 61 | 8 | 1 | 30 | +7 pp | 0.0391 |
| Contact-CVAE-prior | safe40 | 26 | 41 | 0 | 33 | +41 pp | 9.09e-13 |
| Motion-CVAE | task | 59 | 10 | 13 | 18 | −3 pp | 0.678 |
| Motion-CVAE | safe40 | 26 | 41 | 4 | 29 | +37 pp | 9.33e-9 |
| DualZero | task | 55 | 14 | 10 | 21 | +4 pp | 0.541 |
| DualZero | safe40 | 29 | 38 | 2 | 31 | +36 pp | 1.49e-9 |
| ACT baseline | task | 49 | 20 | 18 | 13 | +2 pp | 0.871 |
| ACT baseline | safe40 | 26 | 41 | 7 | 26 | +34 pp | 6.24e-7 |

在这个固定 100 点集合上，Contact-CVAE-zero 与 Motion-CVAE、DualZero、ACT baseline 的 task-success 差异很小，逐点检验未显示显著差异；但 Contact-CVAE-zero 的 safe-success 数量均高出 34–37 个百分点。Contact-CVAE-zero 与 Contact-CVAE-prior 使用同一 Contact-CVAE checkpoint，zero latent 比 prior sample 高 7 pp task success 和 41 pp safe success。

其余四个模型之间的 safe-success 两两差异均未在本 100 点固定集合上显示明确证据。因此，当前结果最稳健的相对结论是 Contact-CVAE-zero 的 safe-success 优势，而不是其余四种配置之间的精细排名。

## 9. 力峰值与成功时间

力统计使用全部 100 次 rollout；成功时间仅统计 task-success 子集。

| configuration | max-force mean / median / P95 / max (N) | task-success median time (s) |
| --- | --- | ---: |
| Contact-CVAE-zero | **26.90 / 25.93 / 44.84 / 52.70** | 6.831 |
| Contact-CVAE-prior | 53.41 / 45.89 / 96.24 / 107.95 | 6.584 |
| Motion-CVAE | 46.87 / 50.26 / 89.92 / 96.24 | 8.052 |
| DualZero | 55.54 / 55.65 / 94.43 / 98.71 | 6.963 |
| ACT baseline | 69.85 / 98.01 / 104.87 / 107.09 | **6.369** |

Contact-CVAE-zero 的全体 max-force mean、median、P95 和 maximum 均最低。ACT baseline 的成功中位时间最短，但其全体 max-force median 为 98.01 N，因此完成速度不能代替力安全指标。Motion-CVAE 的 task success 最高，但成功中位时间最长，且存在 42 次 unsafe task success。

## 10. 五模型空间结果图

以下图片均使用一致的 `[-5,+5] mm` x/z 显示范围、2 mm rings、marker size 和颜色定义。绿色表示 `task success AND max_force < 40 N`，橙色表示 task success 但 `max_force >= 40 N`，红色表示 task failure。采样点本身仍全部位于半径 4 mm 圆盘内；5 mm 只是统一的可视化范围。

### 10.1 Contact-CVAE-zero

![Contact-CVAE-zero 固定圆盘 100 点结果](../../outputs/peg_hole_100/fibonacci_disk_100_r4mm/hole_fibonacci_disk_100_r4mm_contact_cvae100k_zero_mid_dq002_maxsteps900/plots/contact_cvae100k_zero_mid_4mm_target_safe_success_view5mm.png)

- task success：69/100；
- safe success：67/100；
- unsafe task success：2/100；
- task failure：31/100。

### 10.2 Contact-CVAE-prior

![Contact-CVAE-prior 固定圆盘 100 点结果](../../outputs/peg_hole_100/fibonacci_disk_100_r4mm/hole_fibonacci_disk_100_r4mm_contact_cvae100k_prior_mid_dq002_maxsteps900/plots/contact_cvae100k_prior_mid_4mm_target_safe_success_view5mm.png)

- task success：62/100；
- safe success：26/100；
- unsafe task success：36/100；
- task failure：38/100。

### 10.3 Motion-CVAE

![Motion-CVAE 固定圆盘 100 点结果](../../outputs/peg_hole_100/fibonacci_disk_100_r4mm/hole_fibonacci_disk_100_r4mm_motion_cvae100k_mid_dq002_maxsteps900/plots/motion_cvae100k_mid_4mm_target_safe_success_view5mm.png)

- task success：72/100；
- safe success：30/100；
- unsafe task success：42/100；
- task failure：28/100。

### 10.4 DualZero

![DualZero 固定圆盘 100 点结果](../../outputs/peg_hole_100/fibonacci_disk_100_r4mm/hole_fibonacci_disk_100_r4mm_dualzero100k_mid_dq002_maxsteps900/plots/dualzero100k_mid_4mm_target_safe_success_view5mm.png)

- task success：65/100；
- safe success：31/100；
- unsafe task success：34/100；
- task failure：35/100。

### 10.5 ACT baseline

![ACT baseline 固定圆盘 100 点结果](../../outputs/peg_hole_100/fibonacci_disk_100_r4mm/hole_fibonacci_disk_100_r4mm_act_baseline100k_mid_dq002_maxsteps900/plots/act_baseline100k_mid_4mm_target_safe_success_view5mm.png)

- task success：67/100；
- safe success：33/100；
- unsafe task success：34/100；
- task failure：33/100。

各图同时提供 PDF 和带 `point_index` 的诊断 PNG，位于对应实验目录的 `plots/` 下。

## 11. 主要发现

1. **task success 与 safe success 排名不同。**Motion-CVAE 的 task success 最高，为 72%，但 safe success 只有 30%；Contact-CVAE-zero 的 task success 为 69%，safe success 则达到 67%。
2. **Contact-CVAE-zero 的优势主要来自低力完成。**其 safe/task 比例为 97.1%，其他四种配置只有 41.7%–49.3%。
3. **Contact-CVAE-zero 的峰值力整体最低。**其 max-force median 为 25.93 N；其他四种配置为 45.89–98.01 N。
4. **zero latent 明显优于 prior sample。**同一 Contact-CVAE checkpoint 下，zero 比 prior 高 7 pp task success、41 pp safe success。
5. **方向不对称仍然存在。**ACT baseline 的 safe successes 强烈集中于 `+x,-z`，在 `-x,+z` 为 0/25；Contact-CVAE-zero 的空间分布更均衡。
6. **固定点结果与随机 LHS 结果方向一致。**Contact-CVAE-zero 保持明显的 40 N safe-success 优势，但本报告只描述当前固定圆盘点集，不将其自动推广为连续空间成功边界。

## 12. 局限性

- 本实验只有一组固定点，固定点用于严格配对和空间诊断，但不能估计跨 point-set 变异；
- 每点仅运行一次，只有一个 rollout-seed base，尚未充分评价策略推理随机性；
- Fibonacci 点集不含精确中心和恰好 4 mm 的边界点；
- Wilson 区间和 exact p 值是当前固定协议下的描述性统计，不应解释为对所有连续孔位或真实硬件条件的普遍保证；
- 五模型比较并非全部为单因素消融；除 Contact-zero 与 Contact-prior 外，不同 checkpoint 同时包含架构、训练目标或训练过程差异；
- `max_force` 不包含力方向、冲量、持续暴露和接触位置；
- 所有结果来自 MuJoCo 仿真，40 N safe success 不是硬件安全认证；
- 本报告没有分析逐步动作、接触法向力或视频，因此不对策略行为机制做因果解释。

## 13. 数据位置与复现命令

固定点生成：

```bash
python scripts/generate_fibonacci_disk_points.py \
  --num-points 100 \
  --radius-mm 4 \
  --output configs/experiments/fibonacci_disk_100_r4mm.csv
```

完整五模型 rollout：

```bash
python scripts/run_xz_rollout_suite.py \
  --models \
    contact_cvae contact_cvae_prior motion_cvae dualzero act_baseline \
  --action-select-modes mid \
  --task-points-csv configs/experiments/fibonacci_disk_100_r4mm.csv \
  --offset-mm 4 \
  --rollout-seed-base 31000 \
  --max-rollout-steps 900 \
  --max-delta-q 0.02 \
  --success-force-threshold 40 \
  --target-map-max-radius-mm 5 \
  --output-base outputs/peg_hole_100/fibonacci_disk_100_r4mm
```

实验根目录：

```text
outputs/peg_hole_100/fibonacci_disk_100_r4mm/
```

每个模型目录包含：

```text
task_points.csv
grid_manifest.json
grid_summary.csv
random_position_summary.json
plots/
point_*/summary.json
point_*/rollout_log.csv
```

## 14. 结论

在本次半径 4 mm 固定 Fibonacci 圆盘 100 点、`mid` action selection、每模型 100 次 rollout 的实验范围内，Motion-CVAE 获得最高 task success（72%），但 Contact-CVAE-zero 获得明显最高的 40 N safe success（67%），比第二名 ACT baseline 的 33% 高 34 个百分点。

Contact-CVAE-zero 的 69 次 task successes 中有 67 次满足完整轨迹 `max_force < 40 N`，safe/task 比例为 97.1%，且全体峰值力统计最低。其他四种配置虽然可以达到相近甚至更高的 task success，但存在大量高于 40 N 的成功轨迹。因此，本批固定点实验最明确的结论是：**Contact-CVAE-zero 的主要优势是更稳定地以较低峰值力完成任务，而不是单纯获得更多 task success。**
