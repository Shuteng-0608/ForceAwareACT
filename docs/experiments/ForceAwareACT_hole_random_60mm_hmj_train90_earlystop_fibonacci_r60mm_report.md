# ForceAwareACT `hole_random_60mm_hmj` 五模型训练与固定 60 mm Rollout 实验报告

## 1. 实验概述

本实验使用人工筛查后保留的 100 条 `hole_random_60mm_hmj` MuJoCo 示教数据，按照固定随机种子划分为 90 条训练集和 10 条验证集，在相同数据与训练预算下训练五种策略配置。模型选择使用以部署模式验证损失为监控指标的 early stopping。训练完成后，统一加载各模型的 `checkpoint_best.pt`，在半径 60 mm 的固定 100 点 Fibonacci 圆盘上进行逐点 rollout。

实验主要回答两个问题：

1. 五种配置在 held-out 验证集上分别需要多少 epoch 才停止，最佳 checkpoint 出现在哪个 epoch；
2. 最佳 checkpoint 在完全相同的 100 个孔位上，任务成功率、40 N 安全成功率和峰值力表现如何。

正式训练于 2026-07-16 21:01:28 开始，于 2026-07-17 21:06:56 完成，总流水线时间约 24 小时 05 分 28 秒。正式 rollout 于 2026-07-17 21:46:30 开始，于 22:35:20 完成，总计约 48 分 50 秒。训练与 rollout 均使用 NVIDIA GeForce RTX 3070 和 CUDA；rollout 使用 MuJoCo EGL 后端。

## 2. 数据集与划分

数据目录为：

```text
mujoco_data/hole_random_60mm_hmj/
```

人工筛查后共保留 100 条 episode。自动质量检查也将 100 条全部判为 `good`，平均质量分为 99.1。数据划分使用种子 `20260716`：

| 集合 | episode 数 | 训练样本数 | 用途 |
| --- | ---: | ---: | --- |
| train | 90 | 35,886 | 参数优化与归一化统计 |
| validation | 10 | 3,977 | 每个 epoch 后的部署模式验证与 early stopping |

归一化统计只由 90 条训练 episode 计算，未使用验证集。动作模式为 `action`；数据检查确认该字段等价于绝对关节位置命令 `joint_pos_command`。用于复现的数据清单及统计文件为：

```text
outputs/hole_random_60mm_hmj/train90_val10_seed20260716/train90.txt
outputs/hole_random_60mm_hmj/train90_val10_seed20260716/val10.txt
outputs/hole_random_60mm_hmj/train90_val10_seed20260716/normalization_stats_action_train90.pt
```

对应 SHA-256：

```text
train90.txt: 2ab5a3e178d67622acbaf4a67ce006476b948d59dcfe870f66f1741f97392fc6
val10.txt:   c342ea8d8ede9e6d40538c837b9e88581f856d00f1af32f442c940d342ea1d69
stats:       9d6c9298be9b046b8c116dcfbcda9b8a7cca0439907051e837e0684c475d46ba
```

## 3. 五种模型配置

五种配置共享相同的图像、动作、训练集和优化预算。主要差异如下：

| 配置 | policy variant | 训练 latent | 部署/验证 latent | 力预测 | 主要差异 |
| --- | --- | --- | --- | --- | --- |
| Contact-CVAE-zero | `force_aware_contact_cvae` | contact posterior | zero | 有 | `lambda_prior=0` |
| Contact-CVAE-prior | `force_aware_contact_cvae` | contact posterior | deterministic prior | 有 | `lambda_prior=0.1`, `prior_loss_mode=mse_mu` |
| Motion-CVAE | `force_aware_motion_cvae` | motion posterior | zero | 有 | motion latent，`beta_motion_max=5e-4` |
| DualZero | `force_aware_act` | zero | zero | 有 | 双 latent 均以 zero 方式训练/部署 |
| ACT baseline | `act_baseline` | motion posterior | zero | 无 | 不输入力、不预测力、无 contact latent |

Contact-CVAE-prior 在验证和 rollout 时使用确定性的 prior 均值，不进行随机 prior 采样。因此，当前 Contact-zero 与 Contact-prior 的比较同时包含训练目标和部署 latent 的差异，不能解释为只改变推理 latent 的单因素消融。

## 4. 共同训练协议

| 参数 | 设置 |
| --- | --- |
| action mode | `action` |
| camera | `ee_cam`, `base_top_cam` |
| image size | 224 × 224 |
| action chunk length | 10 |
| force window | 20 帧 / 0.25 s |
| batch size | 16 |
| steps per epoch | 2,243 |
| learning rate | `1e-4` |
| training seed | 0 |
| dataloader seed | 1 |
| device | CUDA |
| validation frequency | 每个 epoch 结束后一次 |
| maximum epochs | 100 |
| maximum steps | 200,000 |
| early-stop minimum epochs | 20 |
| early-stop patience | 10 |
| early-stop relative min delta | 0.005，即相对改善至少 0.5% |
| monitored metric | `deploy_loss` |
| checkpoint used for rollout | `checkpoint_best.pt` |

对四个带力预测的模型，验证指标为：

```text
deploy_loss = action_l1 + 0.1 × force_l1
```

ACT baseline 不预测力，其 `deploy_loss = action_l1`。因此 ACT baseline 的验证损失与前四个模型不是完全相同的目标，不能仅按该数值对五个模型进行排序。early stopping 的“改善”要求相对下降至少 0.5%；小于该阈值的数值下降不会重置 patience。

## 5. Early-stopping 结果

五个模型均正常完成，退出码均为 0；全部由 early stopping 终止，没有模型运行到 100 epoch 或 200,000 step 的硬上限。

| 模型 | 停止 epoch | 消耗 step | 最佳 epoch | 最佳 step | 最佳 deploy loss | 最佳 action L1 | 最佳 force L1 | 停止时 deploy loss | patience | 训练耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 47 | 105,421 | 37 | 82,991 | 0.079150 | 0.053735 | 0.254146 | 0.079127 | 10/10 | 5:07:14 |
| Contact-CVAE-prior | 75 | 168,225 | 65 | 145,795 | 0.074055 | 0.053661 | 0.203945 | 0.075310 | 10/10 | 8:12:00 |
| Motion-CVAE | 34 | 76,262 | 24 | 53,832 | 0.074384 | 0.054497 | 0.198865 | 0.075007 | 10/10 | 3:45:56 |
| DualZero | 29 | 65,047 | 17 | 38,131 | 0.074560 | 0.054068 | 0.204918 | 0.076057 | 10/10 | 3:09:34 |
| ACT baseline | 37 | 82,991 | 27 | 60,561 | 0.059984 | 0.059984 | — | 0.063552 | 10/10 | 3:50:44 |

五个模型合计完成 222 个 epoch、497,946 个优化 step。

### 5.1 各模型停止状态说明

- **Contact-CVAE-zero**：最佳 checkpoint 位于 epoch 37。此后连续 10 个 epoch 没有达到相对 0.5% 的改善要求，于 epoch 47 停止。epoch 47 的 `deploy_loss=0.079127` 虽在绝对数值上略低于 epoch 37 的 0.079150，但下降幅度不足 0.5%，因此不替换最佳 checkpoint。
- **Contact-CVAE-prior**：训练时间和 epoch 数最多。最佳 checkpoint 位于 epoch 65，之后连续 10 个 epoch 未出现满足阈值的改善，于 epoch 75 停止。
- **Motion-CVAE**：最佳 checkpoint 位于 epoch 24，于 epoch 34 停止，相比两个 Contact-CVAE 配置更早收敛。
- **DualZero**：最佳 checkpoint 最早出现在 epoch 17。由于设置了最少训练 20 epoch，训练没有在 epoch 20 前停止，最终在 epoch 29 达到 patience 10 并终止。
- **ACT baseline**：最佳 checkpoint 位于 epoch 27，于 epoch 37 停止。其验证指标不含 force L1，不能用 0.059984 直接宣称优于四个 force-aware 模型。

所有正式训练目录均保留最终 checkpoint、最佳 checkpoint、训练日志、验证日志和实际训练命令。后续部署统一选用最佳验证 checkpoint，而不是停止时的最终 checkpoint。

## 6. 固定 60 mm Fibonacci Rollout 协议

每个模型在同一组 100 个固定孔位上各运行一次，共 500 次 rollout。点集位于 x/z 平面半径 60 mm 的圆盘内，最大实际采样半径为 59.849812 mm。

| 参数 | 设置 |
| --- | --- |
| point file | `configs/experiments/fibonacci_disk_100_r60mm.csv` |
| point count | 100 |
| point file SHA-256 | `e696287c3afafccbfb9da07bdeceb5a25fbf76d98a1d10469c47b00693279ae5` |
| rollouts per model | 100 |
| action selection | `mid` |
| policy rate | 30 Hz |
| maximum rollout steps | 900，约 30 s |
| maximum joint-command delta | 0.02 |
| success distance threshold | 0.005 m |
| success lateral threshold | 0.006 m |
| success force threshold | 40 N |
| success hold | 连续 15 step |
| hard force-stop threshold | 1,000 N |
| rollout seeds | 31,000–31,099；同一点在五模型间配对 |
| rendering backend | EGL |
| video/snapshot | 不保存 |

固定点文件决定孔位；rollout seed 固定 NumPy/PyTorch 随机状态。当前部署模式均为确定性路径，但仍记录并在模型间配对 seed，以保证协议完整和后续可复现。

本报告使用两个主要 rollout 指标：

- **task success**：在 900 step 内满足距离、横向误差、当前力和连续保持条件；
- **safe success <40 N**：task success，且整条 rollout 的历史峰值力 `max_force < 40 N`。

`safe success` 是本次仿真协议下的操作性指标，不等价于真实机器人安全认证。1,000 N 的 hard force-stop 仅用于防止异常仿真发散，并不是安全阈值。

## 7. Rollout 完整性

- 500/500 次计划 rollout 均执行完成；
- 500/500 份 `summary.json` 有效；
- 五个模型均为 100/100 个有效点；
- process error 为 0；
- hard force stop 为 0；
- 五模型使用完全相同的 `point_index`、孔位坐标和逐点 seed；
- 每个模型均加载各自 early stopping 选出的 `checkpoint_best.pt`。

因此，本批结果可进行同点的横向对比。

## 8. Rollout 总体结果

| 模型 | task success | safe success <40 N | 成功但不安全 | task failure | safe / task | mean max force | 全局 max force | 成功时间中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact-CVAE-zero | 42/100 (42%) | 42/100 (42%) | 0 | 58 | **100.0%** | **20.00 N** | **38.92 N** | **8.37 s** |
| Contact-CVAE-prior | 55/100 (55%) | 29/100 (29%) | 26 | 45 | 52.7% | 35.07 N | 70.40 N | 9.97 s |
| Motion-CVAE | **69/100 (69%)** | **54/100 (54%)** | 15 | **31** | 78.3% | 31.01 N | 66.64 N | 9.21 s |
| DualZero | 47/100 (47%) | 23/100 (23%) | 24 | 53 | 48.9% | 45.17 N | 78.46 N | 8.94 s |
| ACT baseline | 45/100 (45%) | 38/100 (38%) | **7** | 55 | 84.4% | 32.14 N | 98.61 N | 10.59 s |

### 8.1 任务完成能力

Motion-CVAE 的 task success 最高，为 69%，比第二名 Contact-CVAE-prior 高 14 个百分点。Contact-CVAE-prior、DualZero、ACT baseline 和 Contact-CVAE-zero 分别为 55%、47%、45% 和 42%。在当前固定点集上，Motion-CVAE 对较大工作空间偏移的覆盖最好。

### 8.2 40 N 安全成功

Motion-CVAE 的 safe success 也是最高，为 54%。Contact-CVAE-zero 的 safe success 为 42%，数量低于 Motion-CVAE，但其 42 次 task success 全部满足整轨迹峰值力小于 40 N，且全部 100 次 rollout 的最高峰值也只有 38.92 N，表现出最保守的接触行为。

Contact-CVAE-prior 获得 55 次 task success，但其中 26 次峰值力不低于 40 N，safe success 降至 29%。DualZero 的 47 次成功中有 24 次不满足 40 N 条件。ACT baseline 只有 7 次成功属于高力成功，因此其成功条件下的安全比例为 84.4%，但总 task success 仅为 45%。

### 8.3 受力与速度

Contact-CVAE-zero 的全体 rollout 平均峰值力和全局最大力均最低，并且是唯一一个全局最大力仍低于 40 N 的模型。ACT baseline 的平均峰值力不算最高，但出现了五个模型中最大的单次峰值 98.61 N，说明只看均值会掩盖尾部风险。DualZero 的平均峰值力最高，为 45.17 N。

成功时间方面，Contact-CVAE-zero 的中位成功时间最短，为 8.37 s；ACT baseline 最慢，为 10.59 s。该时间只在 task-success 子集上统计，不能脱离成功率单独解释。

## 9. 空间结果图

图中绿色为整轨迹峰值力小于 40 N 的安全成功，黄色为任务成功但峰值力不低于 40 N，红色为失败，黑色五角星为标称孔位。五张图使用相同的 ±60 mm 坐标范围和 10 mm 靶环。

### 9.1 Contact-CVAE-zero

![Contact-CVAE-zero target map](../../outputs/hole_random_60mm_hmj/earlystop_train90_splitseed20260716_run1/rollouts/fibonacci_disk_100_r60mm_mid/target_maps/contact_cvae_zero_target_map.png)

### 9.2 Contact-CVAE-prior

![Contact-CVAE-prior target map](../../outputs/hole_random_60mm_hmj/earlystop_train90_splitseed20260716_run1/rollouts/fibonacci_disk_100_r60mm_mid/target_maps/contact_cvae_prior_target_map.png)

### 9.3 Motion-CVAE

![Motion-CVAE target map](../../outputs/hole_random_60mm_hmj/earlystop_train90_splitseed20260716_run1/rollouts/fibonacci_disk_100_r60mm_mid/target_maps/motion_cvae_target_map.png)

### 9.4 DualZero

![DualZero target map](../../outputs/hole_random_60mm_hmj/earlystop_train90_splitseed20260716_run1/rollouts/fibonacci_disk_100_r60mm_mid/target_maps/dualzero_target_map.png)

### 9.5 ACT baseline

![ACT baseline target map](../../outputs/hole_random_60mm_hmj/earlystop_train90_splitseed20260716_run1/rollouts/fibonacci_disk_100_r60mm_mid/target_maps/act_baseline_target_map.png)

Motion-CVAE 的成功表现存在明显方向差异：`z<0` 点的 task success 为 92%，`z>0` 为 44.9%。其他模型也存在不同程度的方向不对称。这说明总体成功率不能完整表达工作空间覆盖，target map 应与汇总数字共同使用。

## 10. 主要结论

1. **Motion-CVAE 是本次协议下综合 rollout 表现最好的模型。**它同时取得最高 task success（69%）和最高 safe success（54%），说明其优势并非只来自高力插入。
2. **Contact-CVAE-zero 的接触最保守。**其全部 100 次 rollout 的最大峰值只有 38.92 N，42 次成功全部为 safe success；代价是 task success 仅为 42%，空间覆盖不足。
3. **Contact prior 提高任务完成数，但引入较多高力成功。**与 Contact-zero 相比，task success 从 42% 提升到 55%，safe success 却从 42% 降到 29%。由于两个模型的训练目标也不同，该结果不能完全归因于部署 latent。
4. **DualZero 的安全表现最弱。**safe success 仅为 23%，平均峰值力为五模型最高的 45.17 N。
5. **ACT baseline 的典型成功较安全，但存在高峰值尾部。**其 safe/task 为 84.4%，但 task success 只有 45%，并出现 98.61 N 的全局最大力。
6. **验证损失与 rollout 排名并不相同。**held-out imitation validation 用于模型内的 checkpoint 选择；真正的任务完成和力安全仍需通过闭环 rollout 评价。

如果只以本次固定 100 点实验为依据，推荐优先选择 Motion-CVAE 作为任务成功与安全成功之间的综合方案；如果应用更重视峰值力约束并允许较低覆盖率，则 Contact-CVAE-zero 更合适。

## 11. 局限性

- 训练和 rollout 都只有一个训练 seed，尚未量化模型初始化带来的方差；
- 每个固定点只运行一次，没有对同一点做多 rollout-seed 重复；
- 验证集只有 10 条 episode，early-stopping 曲线可能对该小验证集敏感；
- Fibonacci 点集适合严格配对和空间诊断，但不能直接代表连续圆盘上的真实成功面积；
- 五种模型并非全部是单因素消融，架构、latent、训练目标和是否使用力信息存在组合差异；
- `max_force` 没有描述力方向、冲量、持续时间和接触位置；
- 所有结果来自 MuJoCo，40 N safe success 不是实体机器人安全保证。

后续若需要更强统计结论，应至少增加多个训练 seed，并在同一固定点集上使用多个 rollout seed 重复测试，再进行逐点配对统计。

## 12. 结果与复现位置

训练根目录：

```text
outputs/hole_random_60mm_hmj/earlystop_train90_splitseed20260716_run1/formal/
```

每个模型目录包含：

```text
checkpoint_best.pt
checkpoint.pt
train_log.csv
validation_log.csv
training_command.sh
console.log
```

rollout 根目录：

```text
outputs/hole_random_60mm_hmj/earlystop_train90_splitseed20260716_run1/
  rollouts/fibonacci_disk_100_r60mm_mid/
```

关键 rollout 文件包括：

```text
suite_plan.json
<model>/grid_summary.csv
<model>/random_position_summary.json
<model>/grid_manifest.json
<model>/point_*/summary.json
<model>/point_*/rollout_log.csv
target_maps/*.png
target_maps/*.pdf
```

正式五模型训练入口：

```bash
bash scripts/run_hole_random_5model_earlystop.sh train
```

固定 60 mm 五模型 rollout 入口：

```bash
python scripts/run_hole_random_5model_fibonacci_r60_rollouts.py run
```

实际的单模型训练参数以各模型目录中的 `training_command.sh` 为准，实际 rollout 协议以 `suite_plan.json` 为准。
