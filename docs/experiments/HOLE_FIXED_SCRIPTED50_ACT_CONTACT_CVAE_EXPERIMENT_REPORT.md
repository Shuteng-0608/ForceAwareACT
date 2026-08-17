# Hole-Fixed Scripted-50：Official ACT 与高频力 Contact-CVAE 实验报告

更新日期：2026-08-17

## 1. 报告范围与当前状态

本文记录以新采集的 `hole_fixed_scripted` 50 条 scripted demonstration 为唯一训练
数据，分别训练 Official ACT 与 native-500-Hz Contact-CVAE，并在相同 MuJoCo
环境、相同 action executor 和相同安全判据下完成的离线评估、固定中心 rollout、
temporal aggregation 消融和 4 mm Fibonacci 圆盘 100 点配对 rollout。

本文只总结这条新数据集实验链。旧 `peg_hole_100`、旧 paired-50 checkpoint 和
早期正 `k` 成功区间属于不同数据与模型，不与本文主结果合并。为避免遗漏参数扫描
如何影响后续设计，第 7.1 节单独保留其扫描范围、关键结果和解释边界；完整历史结果
见 `PAIRED50_Q1_SIGNED_TEMPORAL_VALIDATION_REPORT.md`、
`PAIRED50_Q1_TEMPORAL_ROLLOUT_PILOT_REPORT.md` 和
`PAIRED_FIBONACCI_R4MM_TEMPORAL_ROLLOUT_PLAN.md`。

截至本文更新时：

- 两个 25,000-step 正式训练均已完成并通过 checkpoint reload audit；
- `k=-0.02/-0.01/0` 的 600 次 Fibonacci-100 rollout 已全部完成；
- 固定中心 `k in [-1,1]`、步长 `0.01` 的 402 次全范围扫描正在服务器运行；
- 全范围扫描尚无最终结果，本文不预写其结论。

训练完成后服务器记录的代码版本为 `669d9bb`，分支为
`feat/rollout-force-hud`。后续 rollout runner、EGL 批处理与分析脚本又经过独立提交；
因此复现实验时应同时保留 checkpoint 内配置、每个 rollout 的 `summary.json` 和
运行脚本版本，不能只记录 checkpoint 文件名。

## 2. 研究问题

本轮实验回答四个相互独立的问题：

1. 在同一批 fixed-hole demonstrations 上，Official ACT 与加入原生 500 Hz 力输入
   和力监督的 Contact-CVAE 能否充分拟合动作？
2. 离线动作误差是否能转化为固定点闭环插入和空间扰动鲁棒性？
3. Q=1 temporal aggregation 中新旧预测的相对权重如何改变推进、停滞和碰撞？
4. Contact-CVAE 是否同时改善任务成功与实测力安全，而不只是减少某一种失败？

实验必须分别报告 action error、task success、safe success、100 N hard stop 和持续
高力暴露；其中任一指标不能代替其他指标。

## 3. 数据集与拆分

### 3.1 数据来源

服务器数据根目录：

```text
/home/stw/work/stw/hole_fixed_scripted
```

数据由 `scripted_two_stage_replay` 采集。采集状态
`scripted_replay_success` 已人工确认表示该条示教成功。最终数据集包含 50 个 episode，
其 dataset fingerprint 为：

```text
8092674aadc32ed27104184cc173b61ade066d15d0934d5aa8612eab36111f8d
```

最初有一条 episode 出现 727 个 state timestamp 对 721 个 image timestamp 的长度
不一致；该条数据没有通过放宽读取契约进入训练，而是重新采集替换。替换后 50 条
全部通过 strict image contract。

### 3.2 质量门禁

最终质量审计结果为 50/50 `good`，平均质量分 100。主要门禁包括：

| 项目 | 设置 |
| --- | ---: |
| success status | `scripted_replay_success` |
| duration | 3--30 s |
| maximum force screening | 60 N |
| maximum joint speed | 1 rad/s |
| maximum command step | 0.05 rad |
| image samples per episode | 8 |
| minimum image std | 2.0 |
| minimum frame change | 0.1 |

质量门禁证明录制满足当前结构和工程筛查规则，不等价于证明数据覆盖足够的空间扰动
或策略一定安全。

### 3.3 固定拆分

实验 manifest：

```text
configs/experiments/hole_fixed_scripted50_seed0.json
```

拆分版本与结果：

| 项目 | 数值 |
| --- | ---: |
| algorithm | `chronological_stratified_validation_all50_v1` |
| train episodes | 40 |
| validation episodes | 10 |
| holdout episodes | 0 |
| train state windows | 29,432 |
| validation state windows | 7,359 |
| seed | 0 |

十个时间分层中各选择一条 validation episode，剩余 40 条用于训练。两个模型严格
使用同一 train/validation episode 集合。归一化统计只由 40 条 train episodes
计算，validation 不参与；对于 Official ACT，这一点是为了本次配对公平性而采用的
manifest 模式，与官方默认“全部 episode 计算统计量”不同。

本数据集没有独立 demonstration holdout。后续 MuJoCo 100 点位是闭环几何扰动测试，
不是未见 HDF5 episode 测试，二者不能混为同一种泛化证据。

### 3.4 采样率和监督语义

| 数据 | 采样率/形状 | 用途 |
| --- | --- | --- |
| `ee_cam`, `base_top_cam` | 约 30 Hz，原生 480 x 640 RGB | 两模型视觉输入 |
| joint state | 约 30 Hz，7 DoF | 当前 qpos |
| `/action` | 约 30 Hz，7 DoF absolute joint command | 100-step action target |
| raw wrench | 500 Hz，6D force/torque | Contact 在线输入与未来力监督 |

图像在原生尺寸已经为 480 x 640 时不 resize，先转 `[0,1]`，再做 ImageNet
normalization。训练和 rollout 共用相同 checkpoint adapter 预处理。

qpos 与 action 分别按 7 个维度使用 train split 的 population mean/std 标准化。
Contact 的 wrench 统计直接遍历 train split 的全部原生 500 Hz 样本，并对 6 个
分量分别标准化；这既避免先降到 30 Hz，也避免把 N 与 N·m 当作同一量纲求统计量。
高频 padding 在标准化后严格置零。loss 在标准化空间计算，rollout 的 action 与
力诊断再使用 checkpoint 统计量恢复到原始物理单位。

高频力审计确认：

- 在线窗口为当前时刻之前最后 100 个真实 500 Hz wrench，覆盖约
  `0.197--0.199 s`；
- 100 点被按 state timestamp 打包到最多 7 个历史区间；
- 每个 action interval 通常对应 16 或 17 个原生力点，全集观察范围为 8--18；
- 每个未来区间容量为 20，保留原生点和独立 mask，不先降采样到 30 Hz。

## 4. 模型与训练配置

### 4.1 共享的 ACT 主干配置

| 配置 | Official ACT | Contact-CVAE |
| --- | ---: | ---: |
| image cameras | 2 | 2 |
| image resolution | 480 x 640 | 480 x 640 |
| qpos/action dimension | 7 / 7 | 7 / 7 |
| action chunk length | 100 | 100 |
| model width | 512 | 512 |
| attention heads | 8 | 8 |
| feed-forward width | 3,200 | 3,200 |
| policy encoder layers | 4 | 4 |
| decoder layers | 7 | 7 |
| latent dimension | 32 | 32 |
| ResNet18 | ImageNet pretrained, FrozenBN | 同左 |
| parameters | 83,908,744 | 104,038,597 |

Official ACT 使用 4-layer action posterior、4-layer policy encoder 和 7-layer
decoder，训练 latent 为 `z_motion`，部署时为精确零向量。它没有任何力输入、力
token、力预测头或 conditional prior。

Contact-CVAE 保持相同 ACT 层数，并增加：

- 共享 4-layer local temporal convolution encoder，对每个原生 500 Hz 区间编码；
- 4-layer online-force Transformer；
- force-conditioned visual cross-attention；
- 4-layer contact posterior；
- conditional contact prior；
- endpoint force head 和 native-rate force head。

训练时 decoder 使用 posterior `z_contact`；本文全部在线 rollout 使用精确
`z_contact=0`。conditional prior 仅作为训练辅助和独立诊断模式，不参与本文闭环
动作执行。

### 4.2 共同 optimizer 配置

| 配置 | 数值 |
| --- | ---: |
| optimizer | AdamW |
| main learning rate | `1e-5` |
| ResNet18 body learning rate | `1e-5` |
| weight decay | `1e-4` |
| betas | `(0.9, 0.999)` |
| epsilon | `1e-8` |
| batch size | 8 |
| scheduler | 无 |
| gradient clipping | 无，只记录 norm |
| seed | 0 |
| workers | 2 |

### 4.3 Official ACT 目标和采样

Official ACT objective：

```text
L_official = official_masked_action_L1
           + 10 * KL(q_motion || N(0,I))
```

每个 epoch 从每条 train episode 均匀采一个 timestep：

```text
40 samples / epoch / batch 8 = 5 optimizer steps / epoch
5000 epochs * 5 steps = 25,000 optimizer steps
```

validation 每个 epoch 在训练前执行，best metric 是
`official_sampled_validation_loss`。这种每 episode 单点采样是 Official ACT 的
训练语义，不等于遍历全部 29,432 个 window。

### 4.4 Contact-CVAE 目标和采样

Contact-CVAE objective：

```text
L_contact = 1.0 * L_action
          + 1.0 * L_force_endpoint
          + 0.5 * L_force_native_500Hz
          + 10.0 * KL(q_contact || N(0,I))
          + 1.0 * KL(stopgrad(q_contact) || p_contact_conditional)
```

prior matching 中 posterior 参数及 prior 的条件输入均 detach，使该项只训练 prior，
不反向拖动 posterior、视觉主干或共享高频力编码器。

本轮训练栈标识为 `contact_conditional_cvae_native_500hz_v3`。

Contact dataset 暴露每个 train state timestep：

```text
29,432 windows / batch 8 = 3,679 optimizer steps / data epoch
25,000 steps = 6 full data epochs + 2,926 steps
                   approximately 6.80 data epochs
```

命令中的 `official_reference_epochs=5000` 只用于导出与 Official ACT 相同的训练预算：

```text
ceil(40 reference episodes / 8) * 5000 = 25,000 steps
```

所以两模型拥有相同 optimizer-step 和近似相同 batch-sample 预算，但采样分布不同：
Official 每 epoch 每 episode 随机取一点；Contact 对全部窗口 shuffle 并反复遍历。

### 4.5 正式运行配置

| 项目 | Official ACT | Contact-CVAE |
| --- | --- | --- |
| output | `runs/hole_fixed_scripted50_official_act_formal_e5000_b8_seed0` | `runs/hole_fixed_scripted50_highrate_v3_formal_s25000_b8_seed0` |
| duration argument | 5,000 epochs | 5,000 reference epochs |
| optimizer steps | 25,000 | 25,000 |
| checkpoint interval | 100 epochs | 2,000 steps |
| selection metric | sampled validation loss | zero-latent action L1 |

两条训练在服务器两张 RTX 6000 Ada GPU 上并行完成。训练前先通过 dataset readiness、
500 Hz force contract、10-step 与 100-step burn-in、optimizer coverage、梯度、部署输入
隔离和 checkpoint reload 检查。

服务器共有 4 张 49,140 MiB RTX 6000 Ada；训练环境记录为 PyTorch
`2.13.0+cu126`、torchvision `0.28.0+cu126`、cuDNN `91002`。`runs/` 在服务器上
链接到 `/home/stw/work/stw/runs`，训练产物不由 Git 跟踪。

## 5. 训练结果与最终产物

### 5.1 Official ACT

```text
epochs                 5,000
global_step            25,000
best_epoch             4,336
best sampled metric    0.0304963
reload audit           passed
```

产物：

| 文件 | 大小 | 含义 |
| --- | ---: | --- |
| `final.pt` | 1.251 GiB | 最终模型、optimizer 与恢复状态；本文 rollout 使用 |
| `latest.pt` | 1.251 GiB | 最近覆盖式恢复点 |
| `best_policy.pt` | 0.313 GiB | sampled-validation best 的轻量 policy |

### 5.2 Contact-CVAE

```text
global_step                         25,000
position                            epoch 6, step_in_epoch 2,926
final deployment-zero action L1    0.0701074
final zero endpoint-force L1       0.1082575
final zero native-force L1         0.1359300
final posterior KL                 0.0005560
final prior-match KL               0.0005737
checkpoint reload audit            passed
optimizer parameter entries        361
```

产物：

| 文件 | 大小 | 含义 |
| --- | ---: | --- |
| `best.pt` | 1.163 GiB | zero-latent action metric best；本文 rollout 使用 |
| `final.pt` | 1.163 GiB | 25,000-step 最终恢复点 |
| `last.pt` | 1.163 GiB | 最近恢复点 |
| `step_00002000.pt` ... `step_00024000.pt` | 各 1.163 GiB | 2,000-step milestones |

### 5.3 Latent 诊断

两条训练最终都呈现 posterior KL 接近零、posterior/zero action 差异极小。Contact
的 zero、prior 和 posterior action L1 也几乎相同。这是明显的 latent-use collapse
信号，说明当前 demonstration 对 decoder 而言高度条件可预测。

该现象不等于 deployable policy 必然失败，但意味着本轮 rollout 主要比较的是两个
近确定性策略，而不是依靠 latent 多模态采样产生的行为差异。

### 5.4 统一离线动作评估

使用全部 validation windows 和同一物理动作误差口径比较候选 checkpoint：

| checkpoint | action L1 | action-step L2 |
| --- | ---: | ---: |
| Official `best_policy.pt` | 0.0116608 | 0.0454033 |
| Official `final.pt` | 0.0111017 | 0.0435808 |
| Contact `best.pt` | 0.00250646 | 0.00952540 |

因此后续 rollout 选择 Official `final.pt` 和 Contact `best.pt`。Contact 离线 action
L1 约为 Official final 的 22.6%，但后续闭环结果表明这一离线优势没有直接转化为
更高空间扰动成功率。

## 6. Rollout 公平协议

### 6.1 共同输入和部署差异

- 两模型接收同一时刻的 `ee_cam`、`base_top_cam` 和 7D qpos；
- Official ACT 不接收力；
- Contact 接收因果的最近 100 个 500 Hz 实测 wrench；
- Contact deployment latent 固定为精确零；
- 每个 policy query 都输出 100 个 future actions；
- 模型预测力只用于诊断，不参与当前 action selection 或 hard stop。

### 6.2 Action executor

本文主要使用 Q=1 signed temporal aggregation，即每个 30 Hz policy step 都重新
预测，并只聚合同一绝对执行时刻的对齐候选。定义：

```text
weight = softmax(-signed_k * prediction_age)
```

因此：

- `k < 0`：偏向更旧 query 对当前时刻的预测，即旧 chunk 中更远期的动作；
- `k = 0`：所有对齐预测等权；
- `k > 0`：偏向最新 query 的当前/近期动作；
- Official ACT 原始 temporal `k=0.01` 在此 signed-age 约定下等价于 `k=-0.01`。

### 6.3 控制、安全与成功判据

| 配置 | 数值 |
| --- | ---: |
| policy rate | 30 Hz |
| Contact force-history sampling | 500 Hz |
| force safety observation | 每个 MuJoCo physics step；本实验为 1 kHz |
| max rollout length | 1,800 policy steps = 60 s simulated time |
| EMA alpha | 1.0 |
| max joint command delta | 0.02 rad |
| success distance | 3 mm |
| success dwell | 0.1 s |
| safe-success force threshold | 40 N |
| hard stop | 100 N |

两模型使用相同后处理、阈值、初始关节状态、MuJoCo XML 和 hole offset。`safe=40 N`
只用于评价；当前实现不会因持续超过 40 N 自动减速或退让。100 N hard stop 使用
每个 physics step 的传感器实测力。

## 7. 已完成实验

### 7.1 `k` 扫描实验链与数据边界

本项目先后对两代 checkpoint 做过 Q=1 signed-temporal 扫描。下表将全部扫描集中
列出，目的是说明候选参数从何而来，同时防止把不同训练集、模型和 rollout 上限下的
结果误当作同一条响应曲线。

| 阶段 | checkpoint / 数据 | 扫描范围 | 关键结果 |
| --- | --- | --- | --- |
| paired-50 离线 validation | 旧 `peg_hole_100` 40/10 split | 21 个有限 `k`：`-1` 至 `+1`，另加 latest/oldest-only | Official 最低 teacher-forced action L1 在 `k=-0.05`；Contact 在 `k=+0.3`，但该结果不包含闭环状态转移 |
| paired-50 固定中心粗扫 | 旧 paired-50 两模型，600 steps | 14 个 `k` 加 latest-only | Contact 在已测 `k=+0.04~+0.10` 的 8 个点全部几何成功；Official 15/15 失败；Contact 8 次成功均超过 40 N |
| paired-50 Contact 中尺度细扫 | 旧 Contact，600 steps | `+0.060~+0.090`，步长 `0.0025` | 13/13 几何成功、0/13 safe success；最快为 `k=0.0675`、298 steps，最低峰值为 `k=0.075`、60.05 N |
| paired-50 Contact 窄区间细扫 | 旧 Contact，600 steps | `+0.070~+0.078`，步长 `0.0005` | 17/17 几何成功、0/17 safe success；最快为 `k=0.071`、300 steps，最低峰值为 `k=0.0735`、56.84 N；`0.074~0.0745` 和 `0.0755~0.078` 又升至约 85--87 N，存在窄而非单调的恢复区 |
| paired-50 Fibonacci-100 | 旧 paired-50 两模型，600 steps | `+0.01,+0.05,+0.071,+0.0735,+0.075` | 1,000/1,000 rollout 完成；Contact task success 为 `31/39/38/41/39`，Official 为 `16/14/17/17/16`；但 Contact safe success 仅 `24/18/10/9/9`，不能把任务成功解释为力安全 |
| scripted-50 固定中心 pilot | 本文新两模型，600 steps | `-0.01,0,+0.074` 加 latest-only | 8/8 均未在 20 s 内成功，但 temporal 模式显著推进，latest-only 几乎不动 |
| scripted-50 固定中心粗扫 | 本文新两模型，1,800 steps | 19 个 `k`：`-0.20` 至 `+0.10` | 两模型成功带移到 `-0.02~0` 附近；正 `k>=0.01` 大范围停滞 |
| scripted-50 固定中心全扫 | 本文新两模型，1,800 steps | `-1.00~+1.00`，步长 `0.01` | 402 个配置正在服务器执行；尚未完成，不进入本文结果结论 |

旧 paired-50 细扫最重要的工程发现是：固定点成功并非由单个浮点 `k` 偶然触发，
但其连续几何成功区仍全部不满足 40 N safe-success。随后在同一 4 mm Fibonacci 点集
上的 1,000 次测试表明，正 `k` 的 Contact task-success 优势仍伴随大量高力轨迹。

新 scripted-50 checkpoint 的有效区域反而落在非正 `k`。这说明 `k` 不是可跨数据、
checkpoint 和训练流程直接迁移的“模型常数”，而是模型输出的 chunk 时间结构与执行器
共同形成的部署参数。另需特别指出：按本文统一的 signed-age 公式，官方 ACT 的
candidate-index `k=0.01` 等价于 **signed `k=-0.01`**；旧 Fibonacci 计划中把
signed `+0.01` 称作 official anchor 的文字不严谨，本文已用 `-0.01` 作为修正后的
官方等价参考点。

### 7.2 600-step 固定中心执行器 pilot

最初以 600 steps（20 s simulated）测试 `k=-0.01,0,+0.074,latest-only`。两模型
八种配置全部失败。

- `latest-only` 几乎停在初始位置，证明 action[0] 接近当前 qpos，不能只执行最新
  chunk 的第一项；
- 三种 temporal 配置把 peg 从约 162 mm 推进到约 46--49 mm，但 600 steps 不足以
  完成任务；
- 因此正式固定点和空间扰动实验统一扩展到 1,800 steps，而不是把 600-step 失败
  解释成模型失败。

### 7.3 1,800-step 固定中心粗扫描

粗扫描 19 个 signed `k`，每个 `k` 对两个模型各运行一次：

```text
-0.20, -0.10, -0.05, -0.03, -0.02, -0.015, -0.010,
-0.0075, -0.005, -0.0025, 0,
+0.0025, +0.005, +0.010, +0.020, +0.030, +0.050,
+0.074, +0.100
```

符号：`S`=safe success，`U`=task success 但不安全，`H`=100 N hard stop，
`T`=1,800-step timeout。

| k | Official | Contact | 主要观察 |
| ---: | :---: | :---: | --- |
| -0.20 | H | H | 过度偏旧，双方剧烈碰撞 |
| -0.10 | S | U | Contact 成功但峰值约 83.5 N |
| -0.05 | T | H | Contact 出现安全悬崖 |
| -0.03 | T | S | 两模型转变边界不同 |
| -0.02 | S | S | 双方成功，Contact 终距约 1.49 mm |
| -0.015 | S | S | 双方成功 |
| -0.010 | S | S | 官方等价基线 |
| -0.0075 | S | S | 双方成功 |
| -0.005 | S | S | 双方成功 |
| -0.0025 | S | S | 双方成功 |
| 0 | S | S | 双方成功但更慢 |
| +0.0025 | S | T | Contact 开始停滞 |
| +0.005 | S | T | Official 接近 1,800-step 上限才成功 |
| >= +0.01 | T | T | 双方大范围停滞 |

代表结果：

| k | 模型 | steps | final distance | max raw force |
| ---: | --- | ---: | ---: | ---: |
| -0.02 | Official | 817 | 2.22 mm | 13.80 N |
| -0.02 | Contact | 811 | 1.49 mm | 12.83 N |
| -0.01 | Official | 868 | 2.42 mm | 13.02 N |
| -0.01 | Contact | 867 | 1.67 mm | 13.30 N |
| 0 | Official | 1,116 | 2.62 mm | 11.43 N |
| 0 | Contact | 1,407 | 1.53 mm | 17.09 N |

粗扫描说明 signed `k` 不是普通平滑系数，而是在“执行旧规划中的远期推进”与
“依赖最新预测而停滞”之间改变策略。它据此产生三个预先保留的 100 点设置：

- `k=-0.01`：Official 原始 temporal 的 signed 等价基线；
- `k=-0.02`：更激进、固定中心更快；
- `k=0`：等权、相对保守。

### 7.4 服务器渲染性能审计

服务器默认 `MUJOCO_GL` 未设置时，一条 rollout 的平均耗时为：

```text
model inference             20.72 ms
full policy compute        188.66 ms
non-inference compute      167.94 ms
wall-clock throughput        5.30 Hz
```

同一服务器显式使用 EGL 后：

```text
model inference             11.81 ms
full policy compute         17.75 ms
compute throughput          56.34 Hz
deadline miss fraction       0.67%
```

本机 RTX 3070 对照为 20.59 ms full policy compute。该审计确认慢速来自服务器默认
OpenGL 路径，不是模型、CCTV HUD 或 swap。正式 Fibonacci 实验统一设置：

```text
MUJOCO_GL=egl
MUJOCO_EGL_DEVICE_ID=1
CUDA_VISIBLE_DEVICES=1
```

三组 600-rollout 总墙钟约 5 小时 57 分，无运行失败。

### 7.5 Fibonacci 4 mm 圆盘 100 点配对 rollout

点集：

```text
configs/experiments/fibonacci_disk_100_r4mm.csv
```

每个点对两个模型使用相同 offset 和 seed；每个 `k` 为 200 次 rollout，三组共
600 次。所有结果均为 200/200 completed、0 process failure。

#### 总体结果

| k | 模型 | task success | safe success | hard stop | timeout |
| ---: | --- | ---: | ---: | ---: | ---: |
| -0.02 | Official | 20/100 | 19/100 | 30 | 50 |
| -0.02 | Contact | 14/100 | 14/100 | 22 | 64 |
| -0.01 | Official | 16/100 | 16/100 | 27 | 57 |
| -0.01 | Contact | 12/100 | 12/100 | 14 | 74 |
| 0 | Official | 15/100 | 14/100 | 23 | 62 |
| 0 | Contact | 10/100 | 10/100 | 9 | 81 |

此表的 raw-force 与 compensated-force safe-success 计数在六组配置中恰好相同，
因此合并显示为 `safe success`；后文的 peak、40 N 持续时间和 exposure 分析使用
重力补偿后的力，100 N hard stop 则由在线原始传感器力触发。

`k=-0.02` 在两个模型上都得到最高名义 task success，但也产生最多 hard stop。
从 `k=0 -> -0.01 -> -0.02`，Contact 的 task success 为 `10 -> 12 -> 14`，hard
stop 为 `9 -> 14 -> 22`，显示清晰的推进能力与碰撞风险交换。

#### 模型逐点配对

| k | 共同成功 | 仅 Official | 仅 Contact | 共同失败 | exact McNemar p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -0.02 | 11 | 9 | 3 | 77 | 0.1460 |
| -0.01 | 9 | 7 | 3 | 81 | 0.3438 |
| 0 | 8 | 7 | 2 | 83 | 0.1797 |

Official 在三个 `k` 上名义成功率均较高，但 paired difference 均未达到 0.05。
因此当前证据支持“观察到一致方向”，不支持“已经证明 Official 成功率更高”。

Contact 显著减少某些设置的 100 N hard stop：

```text
k=-0.01: Official 27, Contact 14, paired p=0.00443
k=0:     Official 23, Contact  9, paired p=0.00258
k=-0.02: Official 30, Contact 22, paired p=0.0963
```

#### 空间鲁棒性

| radius | points | Official 最好结果 | Contact 最好结果 |
| --- | ---: | ---: | ---: |
| 0--1 mm | 6 | 6/6 | 4/6 |
| 1--2 mm | 19 | 13/19 | 10/19 |
| 2--3 mm | 31 | 2/31 | 0/31 |
| 3--4 mm | 44 | 0/44 | 0/44 |

六种 model/executor 配置的成功并集也只有 25 个点，75 个点全部失败。Official
三个 `k` 的成功并集为 23 点，单组最好 20 点；Contact 成功并集为 15 点，单组
最好 14 点。继续细调 `k` 只能小幅移动边界，不能解决 2 mm 外大面积失败。

Contact 还存在明显方向不对称。按 CSV 坐标，Contact 在 `z>=0` 半区的成功数为
`11/12/10`，在 `z<0` 半区仅为 `1/2/0`（依次对应 `k=-0.01/-0.02/0`）。该现象
需要结合 target map、相机视角和机械方向解释，不能只归因于某个网络模块。

#### 持续高力与 hard stop 不是同一安全现象

Contact 的平均 compensated peak 较低、100 N hard stop 较少，但其典型接触力
中位数约 12--13 N，高于 Official 的约 5--6 N；两者的力分布形态不同。

Contact 存在超过 40 N 但未触发 100 N hard stop 的长时间失败：

| k | non-hard-stop >40 N cases | conditional mean duration |
| ---: | ---: | ---: |
| -0.02 | 5 | 3.42 s |
| -0.01 | 7 | 4.23 s |
| 0 | 7 | 2.82 s |

例如 `k=-0.01, point 33` 最大 compensated force 约 98.4 N，高于 40 N 累计约
7.38 s，最终仍因 max steps 失败。因此“hard stop 更少”不能直接写成“整体更安全”。
当前策略没有把预测力转化为在线退让，也没有对持续超过 40 N 做软干预。

#### 失败形态

Official 的 timeout 中每组有 4--7 个最终距离小于 5 mm，属于接近阈值但未满足
3 mm/0.1 s dwell。Contact 的 timeout 大多停在约 48 mm：

```text
k=-0.01 timeout final-distance minimum 44.7 mm
k=0     timeout final-distance minimum 45.2 mm
```

Contact 因而更接近“进入有效轨迹后成功/碰撞，否则在约 48 mm 阶段停滞”的二元
行为。它的低成功率不是简单放宽成功阈值就能修复。

## 8. 综合结论

1. **训练有效但 latent 使用很弱。** 两模型均稳定完成 25,000 steps；Contact 能
   学习动作和两类力重建，但 posterior、prior 与 zero 行为趋同。
2. **离线动作拟合不等于闭环任务成功。** Contact 的 validation action L1 明显
   更低，但 Fibonacci task success 在三个 `k` 下均低于 Official 的名义值。
3. **`k` 调节的是推进与接触风险。** 更负的 `k` 更强调旧规划中的远期动作，通常
   推进更快，同时提高 hard stop；正 `k` 强调最新动作，容易停滞。
4. **当前空间鲁棒范围主要小于 2 mm。** 3--4 mm 外圈全部失败，继续只优化执行器
   无法补足训练数据和策略本身的空间泛化。
5. **Contact 改变了力失败分布，而非单向改善安全。** 它减少部分瞬时 100 N
   hard stop，却产生数秒的 40--100 N 持续受力失败。
6. **单一 `k` 的选择取决于目标。** `k=-0.02` 名义成功率最高，`k=0` hard stop
   最少，`k=-0.01` 是官方等价且较平衡的共享比较点。三者都不应被称为已验证的
   实机最优参数。

## 9. 正在运行的全范围固定中心扫描

当前服务器实验：

```text
output:
runs/hole_fixed_scripted50_fixed_center_q1_signed_k_m1_p1_step001_n201_s1800_b1_egl

k:                -1.00 ... +1.00
step:              0.01
k count:           201
models:            2
planned rollouts:  402
hole offset:       [0,0,0]
Q:                 1
max steps:         1800
video:             disabled
renderer:          EGL
```

执行顺序为 `0,-0.01,+0.01,...,-1,+1`，使正负值相邻。该实验的目的不是估计
成功概率，而是在一个确定性固定点上绘制完整的 executor response curve，并定位：

- 成功、near-miss、停滞和 hard-stop 区间；
- 权重从多预测融合过渡到近端点选择的位置；
- Official 与 Contact 的参数敏感性和不连续边界；
- steps、最小距离和最大实测力随 `k` 的变化。

因为同一点每个配置只运行一次，该扫描不能替代多点或多 seed 鲁棒性实验。结果
完成后应形成带结果日期的后续报告或附录，不应反向修改本快照中已经完成的
Fibonacci test 数值。

## 10. 解释边界与后续实验

### 10.1 当前结果的限制

- 50 条数据全部用于 40/10 train/validation，没有独立 HDF5 holdout；
- demonstration 均来自 fixed-hole scripted replay，未直接覆盖 4 mm 扰动圆盘；
- Fibonacci 每个 point/model/k 只有一次确定性 rollout，没有模拟噪声或重复 seed；
- 同一 100 点已经用于比较三个 `k`，因此属于 executor ablation/development set；
- 若根据该集合选出 `k`，不能再用同一集合报告无偏最终成功率；
- 40 N 只是评价阈值，尚未形成独立的软停止、退让或力限制控制器。

### 10.2 推荐顺序

1. 完成 `[-1,1]` 固定点扫描并绘制完整曲线，不只列成功 `k`；
2. 为六组 Fibonacci 结果生成 target maps 和 model-difference maps；
3. 选择 common success、model-discordant、48 mm stall、near-miss、持续高力和 hard
   stop 代表点，重新保存 CCTV force-HUD 视频；
4. 在 500 Hz 独立安全线程中设计 40 N dwell 后的减速/保持/退让消融，不让模型
   推理线程承担 hard real-time safety；
5. 若目标是 4 mm 鲁棒性，采集或合成具有孔位扰动、不同接触方向和恢复轨迹的训练
   数据，而不是继续只调 `k`；
6. 固定模型、executor 和安全策略后，用新的点集旋转、随机 seed 或新采集 holdout
   做确认性测试。

## 11. 主要实验产物

### 11.1 训练

```text
runs/hole_fixed_scripted50_official_act_formal_e5000_b8_seed0/
runs/hole_fixed_scripted50_highrate_v3_formal_s25000_b8_seed0/
runs/hole_fixed_scripted50_readiness/
```

### 11.2 固定中心

```text
runs/hole_fixed_scripted50_fixed_center_q1_executor_pilot_b1/
runs/hole_fixed_scripted50_fixed_center_q1_signed_k_coarse_s1800_b1/
runs/hole_fixed_scripted50_fixed_center_q1_signed_k_m1_p1_step001_n201_s1800_b1_egl/
```

### 11.3 Fibonacci 100 点

```text
runs/hole_fixed_scripted50_fibonacci100_q1_km0p02_s1800_b1_egl/
runs/hole_fixed_scripted50_fibonacci100_q1_km0p01_s1800_b1_egl/
runs/hole_fixed_scripted50_fibonacci100_q1_k0p0_s1800_b1_egl/
```

三组汇总归档：

```text
fibonacci100_three_k_results.tar.gz
```

方法细节以 `docs/methods/CONTACT_CVAE_MODEL_METHOD.md`、
`CONTACT_CVAE_TRAINING_METHOD.md` 和 `CONTACT_CVAE_ROLLOUT_METHOD.md` 为准；本文
只固定本次数据、训练产物、实验协议和观测结果。
