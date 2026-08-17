# `peg_hole_100` Paired-50：Official ACT 与高频力 Contact-CVAE 阶段实验报告

更新日期：2026-08-17

## 1. 报告范围

本文只记录 `mujoco_data/peg_hole_100` 上的 paired-50 实验链：从 100 条示教中固定
选择 50 条，使用其中 40 条训练、10 条验证，分别训练 Official ACT 和 high-rate
Contact-CVAE；随后在保留的 50 条示教、固定点 MuJoCo rollout 及 4 mm Fibonacci
圆盘 100 点上进行评估和 action temporal `k` 扫描。

本文不混入其他训练数据或早期 100k-step 五模型实验。方法细节仍以
`docs/methods/` 下的模型、训练和 rollout 文档为准；本文固定本阶段实际使用的
数据、配置、产物和结果。

截至本报告：

- 两个 10,000-step 正式训练均完成并通过 checkpoint reload；
- 50 条 demonstration holdout 的 teacher-forced 离线评估已完成；
- validation 上的 query interval 扫描和固定 Q=1 signed-`k` 扫描已完成；
- 固定中心粗扫、中尺度扫描和 `0.070~0.078` 窄区间扫描已完成；
- 五个 `k`、两个模型、100 个点的 1,000 次 Fibonacci rollout 已全部完成；
- 十组配置的 target map 均已生成。

## 2. 实验问题

本阶段依次回答：

1. 两个模型在完全相同的 40/10 训练验证划分和相同 optimizer-step 预算下能否稳定训练？
2. Contact-CVAE 对未参与训练和选参的 50 条示教是否具有更低动作误差，其 500 Hz
   力输入是否真实进入网络并影响输出？
3. action chunk 长度固定为 100、每帧重新规划 `Q=1` 时，新旧 chunk 的融合权重
   `k` 如何改变离线动作误差与闭环行为？
4. 固定中心发现的 `k` 成功区域能否迁移到 4 mm 空间扰动？
5. task success、40 N safe success 和 100 N hard stop 是否给出一致结论？

## 3. 数据集与固定拆分

### 3.1 数据来源和规模

```text
data root:           mujoco_data/peg_hole_100
manifest:            configs/experiments/peg_hole_paired50_seed0.json
dataset fingerprint: e6fc49b8521e2e52c59ca0d09892da0c4ae153ebf6a0ad4fde086cd7bc7bb07c
all episodes:        100
all state timesteps: 30,977
```

每条 episode 包含两个 RGB 相机 `ee_cam`、`base_top_cam`，7D joint state、7D
absolute joint action，以及 6D force/torque。图像、joint state 和 action 约为 30 Hz；
原生 wrench 约为 500 Hz。

### 3.2 选择与拆分方法

manifest 使用 `chronological_strata_distribution_match_v1`：按时间顺序分成 10 个
strata，每个 stratum 从 10 条中选择 5 条，使选中集合在轨迹长度、动作变化、接触
比例、接触时刻和力分布等特征上尽量接近全部 100 条；每个 stratum 再固定取 1 条
validation，其余 4 条作为 train。

| split | episodes | state windows | 用途 |
| --- | ---: | ---: | --- |
| train | 40 | 12,427 | 两模型训练与 normalization stats |
| validation | 10 | 3,047 | checkpoint 选择、执行器离线扫描 |
| holdout | 50 | 15,503 | 训练和 `k` 选择后的一次性离线评估 |
| total | 100 | 30,977 | 完整数据集 |

selected 50 条共 15,474 windows，另外 50 条全部保留为 holdout。两个模型读取同一
manifest，禁止各自重新随机拆分。qpos、action 和 wrench 的 normalization statistics
只由 40 条 train episodes 计算，validation 与 holdout 不参与。

### 3.3 输入预处理

- 图像保持数据原生 `480 x 640`，不 resize；转为 `[0,1]` 后做 ImageNet normalization；
- qpos 与 action 按 7 个维度使用 train mean/std 标准化，std 下限为 `0.01`；
- Contact wrench 对 6 个分量分别使用全部 train 原生 500 Hz 样本计算 mean/std；
- 力的前三维单位为 N，后三维为 N·m，归一化和物理指标均按分量处理，不把两种单位
  混成一个共享标量统计量；
- padding 在标准化后置零，并由显式 mask 排除 loss。

## 4. 两个模型

### 4.1 共享 ACT 主干

| 配置 | Official ACT | Contact-CVAE |
| --- | ---: | ---: |
| cameras / image size | 2 / `480 x 640` | 同左 |
| qpos / action dim | 7 / 7 | 7 / 7 |
| action chunk | 100 | 100 |
| `d_model` / heads | 512 / 8 | 512 / 8 |
| feed-forward dim | 3,200 | 3,200 |
| posterior layers | 4 | 4 |
| policy encoder layers | 4 | 4 |
| decoder layers | 7 | 7 |
| latent dim | 32 | 32 |
| vision backbone | ImageNet ResNet18 + FrozenBN | 同左 |
| dropout | 0.1 | 0.1 |
| parameter count | 83,908,744 | 104,038,597 |

Official ACT 的 posterior 输入是 `[CLS, qpos, 100-step action]`，学习 `z_motion`；
部署使用精确零 latent。它没有力输入、力 token、力预测头或 conditional prior。

Contact-CVAE 保持相同 ACT 主干深度，增加：

- 共享 4-layer local temporal convolution encoder，编码每个原生 500 Hz 力区间；
- 4-layer online-force Transformer，编码当前时刻前最后 100 个 500 Hz wrench；
- force-conditioned visual cross-attention；
- 4-layer contact posterior 和 conditional contact prior；
- 100-step endpoint wrench head 与 `[100,20,6]` native-rate force head。

在线 100 点力窗口覆盖约 `0.198 s`，按最多 7 个 state intervals 打包为
`[7,20,6]`；未来每个 30 Hz action interval 保留最多 20 个原生力点及独立 mask。
训练 decoder 使用 posterior `z_contact`；本阶段所有部署评估和 rollout 均使用精确
`z_contact=0`。conditional prior 只作为训练辅助和诊断路径。

## 5. 训练配置与训练语义

### 5.1 共同 optimizer 配置

| 配置 | 数值 |
| --- | ---: |
| optimizer | AdamW |
| main learning rate | `1e-5` |
| ResNet18 body learning rate | `1e-5` |
| weight decay | `1e-4` |
| betas / epsilon | `(0.9,0.999)` / `1e-8` |
| batch size | 8 |
| scheduler | 无 |
| gradient clipping | 无；只记录 gradient norm |
| seed / workers | 0 / 2 |

### 5.2 Official ACT

Objective：

```text
L_official = masked_action_L1 + 10 * KL(q_motion || N(0,I))
```

训练严格采用 Official ACT 的 episode-balanced sampling：每个 epoch 从每条 train
episode 均匀采一个 timestep，因此每个 epoch 有 40 个样本：

```text
40 samples / batch 8 = 5 optimizer steps / epoch
2,000 epochs * 5 steps = 10,000 optimizer steps
total sampled windows = 80,000
```

validation 在每个 epoch 的训练前执行；`best_policy.pt` 按
`official_sampled_validation_loss` 保存。这里的 2,000 epochs 不是遍历 12,427 个
train windows 2,000 次，而是进行 2,000 轮“每 episode 采一个随机时刻”。

### 5.3 High-rate Contact-CVAE v3

Objective：

```text
L_contact = 1.0 * masked_action_L1
          + 1.0 * masked_interval_endpoint_force_L1
          + 0.5 * interval_balanced_native_500Hz_force_L1
          + 10.0 * KL(q_contact || N(0,I))
          + 1.0 * KL(stopgrad(q_contact) || p_contact_conditional)
```

prior-match 项中的 posterior 参数及 prior 条件输入 detach，所以该项只训练 prior，
不会反向拖动 posterior、视觉主干或共享高频力编码器。

Contact dataset 暴露全部 train windows：

```text
12,427 windows / batch 8 = 1,554 steps per full data epoch
10,000 steps = 6 full data epochs + 676 steps
             = approximately 6.44 data epochs
total sampled windows = approximately 80,000
```

`official_reference_epochs=2000` 用于把 40 条 episode、batch 8、2,000 reference
epochs 换算成与 Official 相同的 10,000 optimizer-step / 80,000 sample 预算。两者
训练预算相同，但采样分布不同：Official 每个 epoch 每 episode 一个时刻，Contact
对全部 windows shuffle 并重复遍历。Contact 每个完整 data epoch 做一次全 validation，
`best.pt` 按 deployment-zero action L1 保存。

## 6. 训练结果与产物

### 6.1 Official ACT

```text
output:       runs/paired50_official_act_formal_e2000_b8_seed0
epochs:       2,000
global step:  10,000
best epoch:   1,608
best sampled validation metric: 0.0549519
reload audit: passed at epoch 2,000 / step 10,000
```

训练结束时 sampled validation loss 为 `0.128800`，deployment-zero action L1
为 `0.131719`（标准化空间）。best 与 final 的差异说明后期 sampled validation
存在明显波动，因此后续离线评估和 rollout 使用 `best_policy.pt`。

| 产物 | 大小 | 含义 |
| --- | ---: | --- |
| `best_policy.pt` | 321 MiB | sampled-validation best；后续实验使用 |
| `final.pt` | 约 1.3 GiB | epoch 2,000 的完整恢复点 |
| `latest.pt` | 约 1.3 GiB | 最近覆盖式恢复点 |
| `metrics.jsonl` | 约 4.0 MiB | 逐步和逐 epoch 指标 |
| `run_metadata.json` / `training_summary.json` | 约 11 KiB | 配置、拆分和 reload 证明 |

### 6.2 High-rate Contact-CVAE

```text
output:       runs/paired50_highrate_contact_v3_formal_s10000_b8_seed0
global step:  10,000
position:     epoch 6, step 676 / 1,554
best metric:  0.164401 deployment-zero action L1
reload audit: passed; optimizer parameter entries = 361
```

最终 validation（标准化空间）：

| 指标 | 数值 |
| --- | ---: |
| zero action L1 | 0.16440145 |
| posterior action L1 | 0.16440187 |
| zero endpoint-force L1 | 0.35557304 |
| zero native-force L1 | 0.36404929 |
| posterior KL | 0.00227846 |
| posterior-prior match KL | 0.00244837 |
| posterior-zero action delta | `5.56e-6` |

10,000 steps 是本次最低 zero-latent validation action metric，所以 `best.pt` 与最终
训练位置一致。

| 产物 | 大小 | 含义 |
| --- | ---: | --- |
| `best.pt` | 约 1.2 GiB | zero-latent action metric best；后续实验使用 |
| `final.pt` / `last.pt` | 各约 1.2 GiB | 10,000-step 完整恢复点 |
| `step_00001000.pt` ... `step_00009000.pt` | 各约 1.2 GiB | 1,000-step milestones |
| `metrics.jsonl` / `console.log` | 各约 0.8 MiB | 训练记录 |
| `rollout_force_intervention_t135.json` | 约 8 KiB | 部署力输入干预审计 |

### 6.3 Latent 与力输入审计

两模型最终 posterior KL 都接近零，posterior 与 zero-latent action 差异为微小量；
这表明当前条件下 latent 使用很弱。它不阻止 zero-latent policy rollout，但本阶段
不能把行为差异解释为 latent 多模态采样效果。

Contact 的部署干预审计在 validation episode 的 timestep 135 使用真实 100 点、
500 Hz、严格因果窗口：实测采样率约 500 Hz、跨度 0.198 s、最新样本 age 为 0。
在保持图像、qpos 和 `z_contact=0` 不变时：

- 将 wrench 置零使 action chunk 相对真实力输入的 L2 变化为 `0.30945`；
- 反转力窗口时间顺序使 action L2 变化为 `0.02952`；
- 延迟 10 个力样本（20 ms）使 action L2 变化为 `0.000730`；
- `z_F_online`、force-vision cross-attention、endpoint force 和 native force 输出均
  随干预变化。

因此可确认 500 Hz 力历史已进入 online force encoder、视觉交叉注意力和 decoder，
而不是只被读取后丢弃。该单样本敏感性审计证明“路径有效”，不证明力输入必然提升
任务成功率。

## 7. 50 条 holdout demonstration 离线评估

评估目录：

```text
runs/paired50_holdout_official_vs_contact_all50
```

评估覆盖全部 50 条 holdout、15,503 个时刻。两个模型在每个真实时刻接收专家轨迹
输入；Contact 使用因果 500 Hz 力历史和 `z_contact=0`。这是 teacher-forced 离线
评估，不执行预测动作，也不形成闭环状态转移。

| physical action metric | Official ACT | Contact-CVAE | Contact 相对改善 |
| --- | ---: | ---: | ---: |
| complete 100-step chunk L1 | 0.0388044 | 0.0105500 | 72.81% |
| horizon 1 L1 | 0.0526218 | 0.00571859 | 89.13% |
| horizon 10 L1 | 0.0493702 | 0.00634696 | 87.14% |
| horizon 25 L1 | 0.0443466 | 0.00840985 | 81.04% |
| horizon 50 L1 | 0.0364682 | 0.0117422 | 67.80% |
| horizon 100 L1 | 0.0286759 | 0.0138698 | 51.63% |
| temporal-executed action L1 | 0.0477322 | 0.00958185 | 79.93% |

Contact 还得到：

```text
endpoint linear-force L1 = 1.7003 N
endpoint torque L1       = 0.06470 N*m
native-force L1          = 1.7386 N
native-torque L1         = 0.06627 N*m
```

50/50 holdout episodes 上 Contact 的 temporal action L1 都低于 Official。该结果说明
Contact 对未用于训练的示教轨迹具有更好的 teacher-forced 动作拟合，但不能单独预测
闭环成功率。

## 8. Action executor 与 `k` 扫描

### 8.1 第一轮：query interval 与执行器族筛查

`runs/paired50_validation_action_executor_sweep_b1` 在 10 条 validation、3,047 个时刻
缓存每个模型的预测，然后重放 11 种执行器。实验比较：

- legacy temporal 的 `k=0,0.01,0.03`；
- recency temporal 的 `k=0.01,0.03,0.1`；
- receding chunk 的 `Q=1,5,10,25,100`。

这一步证明 query interval 与 temporal 新旧权重是不同变量。为了单独研究融合方式，
后续固定 `Q=1`，即每个 30 Hz frame 都重新预测一个 100-step chunk。

### 8.2 Signed temporal 定义

查询时刻 `tau` 的 chunk 对绝对执行时刻 `t` 给出候选
`A_tau[t-tau]`。Q=1 时最多有 100 个对齐候选：

```text
prediction_age = t - tau
weight = softmax(-signed_k * prediction_age)
executed_action = weighted sum of aligned candidates
```

因此：

- `k < 0` 偏向更旧预测，即旧 chunk 中更远期的动作；
- `k = 0` 对全部对齐预测等权；
- `k > 0` 偏向更新预测；
- `latest_only` 只用当前 chunk 的 action[0]；
- Official ACT 原始 candidate-index `k=0.01` 按此 age 定义等价于 signed `k=-0.01`。

### 8.3 Q=1 validation 全范围离线扫描

`runs/paired50_validation_signed_temporal_sweep_b1` 扫描 21 个有限 `k`：

```text
-1,-0.5,-0.3,-0.2,-0.1,-0.05,-0.03,-0.02,-0.01,-0.005,
 0,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.5,1
```

另加 `latest_only` 和 `oldest_only`，共 23 个执行器、两个模型、46 条 aggregate
结果。所有配置重放同一份缓存预测，holdout 未被读取。

| 模型 | 最低 validation action L1 | 对应 `k` | 官方等价 `k=-0.01` L1 | latest-only L1 |
| --- | ---: | ---: | ---: | ---: |
| Official ACT | 0.0474616 | -0.05 | 0.0478495 | 0.0521636 |
| Contact-CVAE | 0.00547922 | +0.30 | 0.00994996 | 0.00554851 |

Official 的离线最优区浅而偏旧：`k=-0.05` 相对 `-0.01` 只改善约 0.81%。Contact
则明显偏向新预测，`k=0.2~0.5` 形成低误差平台；`k=0.3` 比 `-0.01` 改善 44.93%。
但自由/低力样本占 validation 的 81.06%，所以全局最优主要由自由空间主导。

这是 teacher-forced 扫描，只能筛选候选，不能把最低动作误差等价为闭环最优。

### 8.4 固定中心闭环粗扫

`runs/paired50_q1_temporal_rollout_pilot_b1` 使用同一原点孔位、seed 0、600 steps
（20 s）、Q=1，扫描 14 个 `k` 加 latest-only，共 30 次 rollout：

```text
-0.01,0,0.01,0.03,0.04,0.045,0.05,0.055,0.06,0.07,
0.08,0.10,0.20,0.30,latest_only
```

共同协议为 30 Hz policy、100 N physics-step hard stop、40 N safe-success 阈值、
3 mm distance + 0.1 s dwell、`max_delta_q=0.02 rad`、EMA alpha 1。Contact 使用
严格因果 500 Hz 力窗口和 zero latent；Official 不接收力。

主要结果：

- Contact 在已测 `k=0.04~0.10` 的 8 个点全部几何成功；
- Official 15/15 均未成功；
- Contact 的 8 次成功全部超过 40 N，因此 safe success 为 0/8；
- `latest_only` 几乎不前进，说明 action[0] 接近当前 qpos，旧 chunk 的远期动作对
  推进至关重要；
- Contact 的局部最佳行为出现在约 `k=0.07~0.08`，但不是单调“越偏新越好”。

代表结果：

| `k` | steps / time | peak force | `>40 N` duration | final distance |
| ---: | ---: | ---: | ---: | ---: |
| 0.040 | 541 / 18.00 s | 88.96 N | 6.967 s | 1.961 mm |
| 0.050 | 428 / 14.23 s | 65.89 N | 0.367 s | 1.918 mm |
| 0.070 | 306 / 10.17 s | 60.50 N | 0.267 s | 1.893 mm |
| 0.080 | 325 / 10.80 s | 62.01 N | 0.233 s | 1.985 mm |
| 0.100 | 449 / 14.93 s | 85.45 N | 3.679 s | 1.938 mm |

### 8.5 中尺度和窄区间细扫

中尺度扫描：

```text
runs/paired50_q1_contact_dynamic_k006_k009_step0025_b1
k = 0.060 ... 0.090, step 0.0025, 13 configurations
```

13/13 全部几何成功，0/13 safe success。最快为 `k=0.0675`（298 steps）；最低
峰值为 `k=0.075`（60.05 N）。`k>=0.0825` 的峰值重新升至约 85--87 N。

窄区间扫描：

```text
runs/paired50_q1_contact_dynamic_k007_k0078_step0005_b1
k = 0.070 ... 0.078, step 0.0005, 17 configurations
```

17/17 全部几何成功，0/17 safe success。最快为 `k=0.071`（300 steps）；最低峰值
为 `k=0.0735`（56.84 N）。值得注意的是：

```text
k=0.0730   peak 58.95 N
k=0.0735   peak 56.84 N
k=0.0740   peak 85.33 N
k=0.0745   peak 85.71 N
k=0.0750   peak 60.05 N
k=0.0755   peak 85.85 N
```

因此固定点存在连续几何成功区，但力响应包含非常窄、非单调的恢复岛；只在一个点位
继续细化 `k` 容易过拟合执行器参数。由此预注册 `0.01,0.05,0.071,0.0735,0.075`
进入 100 点空间扰动测试，而不是继续按固定点结果挑选单个“最优”值。

## 9. Fibonacci 4 mm 圆盘 100 点配对 rollout

### 9.1 实验设计

点集：

```text
configs/experiments/fibonacci_disk_100_r4mm.csv
SHA256: a26e2a5a3e215b6cbb7696c82e5154a19b978139b4696840c652ceed0861df94
```

100 个点按 Fibonacci disk 在 x/z 平面的半径 4 mm 圆盘内确定性铺设，y offset 为 0。
每个点对两个模型使用完全相同 offset 和 seed；五个 `k`、两个模型共：

```text
100 points * 5 k * 2 models = 1,000 rollouts
```

锁定协议：

| 配置 | 数值 |
| --- | ---: |
| action chunk / query interval | 100 / Q=1 |
| action selection | signed temporal |
| signed `k` | `0.01,0.05,0.071,0.0735,0.075` |
| policy rate / max length | 30 Hz / 600 steps = 20 s |
| action mode | absolute joint position |
| EMA / max target delta | 1.0 / 0.02 rad |
| success | 3 mm + 0.1 s dwell |
| safe success | task success 且整条轨迹不超过 40 N |
| hard stop | raw force 超过 100 N，在每个 physics step 检查 |
| video | 关闭；保留逐步日志与 raw/compensated force metrics |

Contact 的 predicted endpoint/native force 只写入诊断，不参与 temporal action
selection，也不触发 hard stop；两个模型共用的 100 N hard stop 只读取 MuJoCo 实测
sensor force。

这里五个值都是当前 signed-age 定义下的正 `k`。`k=+0.01` 是成功区外侧的弱偏新
参考点，并不等价于 Official ACT 原始 temporal `k=0.01`；后者应映射为 signed
`k=-0.01`。

### 9.2 完整性与 20 个失败任务的修复

第一轮有 20 个进程在动作执行前由 argparse 拒绝：point 20 的负科学计数法 x
offset 和 point 73 的负科学计数法 z offset 被误判为 option。受影响的是
`2 points * 5 k * 2 models = 20` 个命令，不是策略 rollout 失败。

commit `4043254` 将 offset 和 decay 改为 `--option=<repr(float)>`，并增加负科学计数
法解析测试。随后只续跑缺失的 20 个任务。最终：

```text
planned/completed: 1,000/1,000
final failed rollouts: 0
task successes: 268
max-step timeouts: 534
100 N hard stops: 198
```

### 9.3 十组配置结果

`safe raw/comp` 分别使用 raw 和 gravity-compensated 40 N 全轨迹阈值；100 N hard
stop 始终使用在线 raw sensor force。

| `k` | model | task success | safe raw / comp | hard stop | median success time | mean comp peak |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.0100 | Official | 16/100 | 11 / 11 | 63 | 8.97 s | 93.79 N |
| 0.0100 | Contact | 31/100 | **24 / 24** | 68 | 8.43 s | 82.39 N |
| 0.0500 | Official | 14/100 | 5 / 5 | 3 | 12.63 s | 56.27 N |
| 0.0500 | Contact | 39/100 | 18 / 18 | 15 | 9.20 s | 74.26 N |
| 0.0710 | Official | 17/100 | **12 / 12** | 2 | 11.77 s | 43.46 N |
| 0.0710 | Contact | 38/100 | 10 / 10 | 15 | 10.72 s | 77.16 N |
| 0.0735 | Official | 17/100 | 11 / 11 | 4 | 11.77 s | 45.67 N |
| 0.0735 | Contact | **41/100** | 10 / 9 | 13 | 10.97 s | 76.36 N |
| 0.0750 | Official | 16/100 | 11 / 11 | 2 | 11.70 s | **43.19 N** |
| 0.0750 | Contact | 39/100 | 9 / 9 | 13 | 10.80 s | 76.72 N |

Contact 在五个 `k` 上的名义 task success 都高于 Official，最高为 `k=0.0735` 的
41%。但 Contact 的 safe success 在较大的 `k` 上没有随 task success 同步提高；
`k=0.071~0.075` 的任务成功主要包含超过 40 N 的轨迹。

### 9.4 逐点配对统计

下表为每个固定 `k`、同一 100 点上的 exact McNemar 检验。`O-only/C-only` 表示
同一空间点只有 Official 或 Contact 达到该指标。p 值未做多重比较校正，应与差值
方向和效应量一起作为描述性证据；若对下表 10 次比较采用 Bonferroni `0.05/10`，
`k=0.05` 的 safe-comp 差异不再达到该保守阈值。

| `k` | metric | both | O-only | C-only | neither | exact p |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.0100 | task | 12 | 4 | 19 | 65 | 0.00260 |
| 0.0100 | safe-comp | 9 | 2 | 15 | 74 | 0.00235 |
| 0.0500 | task | 11 | 3 | 28 | 58 | `4.65e-6` |
| 0.0500 | safe-comp | 0 | 5 | 18 | 77 | 0.0106 |
| 0.0710 | task | 9 | 8 | 29 | 54 | 0.000753 |
| 0.0710 | safe-comp | 1 | 11 | 9 | 79 | 0.824 |
| 0.0735 | task | 12 | 5 | 29 | 54 | `3.86e-5` |
| 0.0735 | safe-comp | 0 | 11 | 9 | 80 | 0.824 |
| 0.0750 | task | 12 | 4 | 27 | 57 | `3.40e-5` |
| 0.0750 | safe-comp | 1 | 10 | 8 | 81 | 0.815 |

在当前固定点集上，Contact 的 task-success 优势在五个 `k` 上方向一致；40 N safe
success 的未校正 paired 证据只出现在 `k=0.01` 和 `0.05`，保守 Bonferroni 后只保留
`k=0.01`。`k>=0.071` 时没有证据显示 safe-success 存在差异，而 Official 的名义
safe 数还略高。由此不能用 task-success 结果替代安全结论。

### 9.5 跨 `k` 空间覆盖

以下 union 是“同一模型五个 `k` 中至少一个成功”的事后诊断，不是可部署单策略的
成功率，因为部署前并不知道每个点应选择哪个 `k`。

| radius | points | Official task/safe union | Contact task/safe union |
| --- | ---: | ---: | ---: |
| 0--1 mm | 6 | 1 / 0 | 5 / 4 |
| 1--2 mm | 19 | 6 / 6 | 16 / 11 |
| 2--3 mm | 31 | 12 / 11 | 16 / 13 |
| 3--4 mm | 44 | 4 / 4 | 12 / 2 |
| all | 100 | 23 / 21 | 49 / 30 |

两个模型五个 `k` 的 task-success 并集共有 51 个点：21 点共同成功、2 点仅 Official
成功、28 点仅 Contact 成功、49 点全部失败。Contact 将几何成功覆盖扩展到更多外圈
点，但在 3--4 mm 外圈 12 个 task-success 点中只有 2 个满足 40 N；更大的几何覆盖
伴随明显的高力代价。

### 9.6 `k` 的总体影响

固定中心扫描显示 `0.07~0.08` 是 Contact 的连续几何成功带，并在极窄范围内出现
较低峰值点；Fibonacci 扰动则表明：

1. 固定中心的低峰值 `k=0.0735` 没有迁移成 100 点最低力配置；
2. Contact task success 从 `k=0.01` 的 31% 上升到 `k=0.0735` 的 41%，但
   compensated safe success 从 24% 降到 9%；
3. Official 在较大的正 `k` 上 task success 仍只有 14--17%，但 hard stop 和平均峰值
   大幅低于 `k=0.01`；
4. `k` 实际控制的是推进、计划连续性与接触风险之间的交换，不是普通的“越平滑越好”
   或“越新越好”参数；
5. 固定点单次 success 和单次 peak-force 最优值不能直接作为空间扰动下的最终参数。

### 9.7 Target maps

十组配置均已生成 PNG、PDF 和绘图数据：

```text
runs/paired_fibonacci_r4mm_temporal_formal_b1/target_maps/
```

文件命名为：

```text
official_act__signed_kp<k>_target_map.{png,pdf}
highrate_contact_v3__signed_kp<k>_target_map.{png,pdf}
target_maps/data/<configuration>.csv
```

图只显示 100 个实际 rollout 点，不对离散点之间插值，也不代表连续空间成功边界。

## 10. 综合结论

1. **训练流程有效且预算配对。** 两模型均完成 10,000 optimizer steps、约 80,000
   个 batch samples，并通过完整 checkpoint reload；但 Official 的 episode-balanced
   随机时刻采样与 Contact 的全 window 遍历不是相同的数据顺序。
2. **Contact 在离线未见示教上动作拟合更好。** 50 条 holdout 的 complete-chunk
   physical L1 降低 72.81%，temporal action L1 降低 79.93%，且 50/50 episodes
   均优于 Official。
3. **高频力路径确实生效。** 真实、置零、反序和延迟 500 Hz 力窗口会改变 online
   force feature、cross-attention、action 与两类 force prediction；部署 latent 始终为零。
4. **离线最优 `k` 与闭环最优不同。** Contact 离线偏好 `k≈0.3`，固定中心闭环
   成功区在 `0.04~0.10`，而 100 点最高 task success 在 `0.0735`；三者不能互换。
5. **Contact 提高空间扰动 task success，但未解决力安全。** 五个 `k` 下 Contact
   task success 的逐点差值方向一致且未校正检验均显著；safe-success 的未校正优势
   只在 `0.01/0.05` 出现，较大 `k` 下多数 Contact 成功超过 40 N。
6. **当前最重要的矛盾是任务推进与高力接触。** `k=0.0735` 取得最高 task success
   41%，但 compensated safe success 只有 9%；按安全优先，不能将其直接称为最佳部署
   配置。
7. **latent 使用很弱。** 本阶段主要比较两个近确定性 zero-latent policy 的结构、
   输入与监督差异，不能把结果归因于 learned latent 多模态行为。

## 11. 结论边界与下一步

- demonstration holdout 是 teacher-forced 离线测试，不是闭环任务成功证据；
- Fibonacci 每个 point/model/`k` 只有一次 deterministic rollout，没有重复 seed；
- 五个 `k` 已在同一 100 点集上比较，该点集现在属于 executor development/evaluation
  set；若据此选择 `k`，最终确认应使用新的点集或新 seed；
- Contact 与 Official 不只是多了力输入，还存在 posterior、prior、力监督和额外网络
  模块差异，因此当前结果不能把收益单独归因于高频力；
- 40 N 是实验评价阈值，不是硬件安全认证；当前策略没有利用 predicted force 做在线
  退让，100 N hard stop 只是外部安全机制；
- 下一轮应固定少量候选（例如兼顾 task/safe 的 `0.01/0.05` 与 task-oriented
  `0.0735`），在新空间点或重复随机条件上确认，并增加去力输入、动态 `k` 或 40 N
  dwell 软干预消融。

## 12. 主要产物索引

训练：

```text
configs/experiments/peg_hole_paired50_seed0.json
runs/paired50_official_act_formal_e2000_b8_seed0/
runs/paired50_highrate_contact_v3_formal_s10000_b8_seed0/
```

离线评估与扫描：

```text
runs/paired50_holdout_official_vs_contact_all50/
runs/paired50_validation_action_executor_sweep_b1/
runs/paired50_validation_signed_temporal_sweep_b1/
docs/experiments/PAIRED50_Q1_SIGNED_TEMPORAL_VALIDATION_REPORT.md
```

固定中心闭环：

```text
runs/paired50_q1_temporal_rollout_pilot_b1/
runs/paired50_q1_contact_hud_k004_k010_b1/
runs/paired50_q1_contact_dynamic_k006_k009_step0025_b1/
runs/paired50_q1_contact_dynamic_k007_k0078_step0005_b1/
docs/experiments/PAIRED50_Q1_TEMPORAL_ROLLOUT_PILOT_REPORT.md
```

Fibonacci 100 点：

```text
configs/experiments/fibonacci_disk_100_r4mm.csv
runs/paired_fibonacci_r4mm_temporal_preflight_b1/
runs/paired_fibonacci_r4mm_temporal_formal_b1/
runs/paired_fibonacci_r4mm_temporal_formal_b1/target_maps/
docs/experiments/PAIRED_FIBONACCI_R4MM_TEMPORAL_ROLLOUT_PLAN.md
docs/experiments/PAIRED_FIBONACCI_R4MM_TEMPORAL_PREFLIGHT_AUDIT.md
```
