# Contact-CVAE 训练配置与训练逻辑

本文说明当前高频力 Contact-CVAE 的数据构造、优化目标、训练循环、验证、模型
选择与 checkpoint 逻辑。本文是方法说明，不包含任何具体实验结果。

适用版本：

```text
architecture: act_aligned_contact_cvae_highrate_force_v2
training:     act_aligned_highrate_conditional_cvae_training_v3
```

训练入口：

```text
scripts/train_act_aligned_high_rate_contact_cvae.py
```

## 1. 标准训练配置

### 1.1 模型相关配置

| 配置 | 标准值 |
| --- | ---: |
| action chunk | 100 |
| policy/state rate | 30 Hz |
| wrench rate | 500 Hz |
| online raw wrench count | 100 |
| online interval capacity | 7 |
| native samples per interval capacity | 20 |
| image size | 480 x 640 |
| latent dimension | 32 |
| encoder/decoder depth | 4 / 7 |

正式训练使用 canonical model config。`--smoke` 只缩小 width、图像尺寸和
chunk 以做结构测试；encoder 仍为 4 层、decoder 仍为 7 层，避免用一个层数
不同的简化网络错误代替正式结构验证。

### 1.2 optimizer 配置

| 配置 | 标准值 |
| --- | ---: |
| optimizer | AdamW |
| main learning rate | `1e-5` |
| ResNet18 body learning rate | `1e-5` |
| weight decay | `1e-4` |
| betas | `(0.9, 0.999)` |
| epsilon | `1e-8` |
| scheduler | 无 |
| gradient clipping | 默认无 |
| batch size | 8 |
| seed | 0 |

参数按模块身份严格分成两个互斥 group：`vision_backbone.body` 属于 backbone
group，其余可训练参数属于 main group。构造 optimizer 时会检查所有可训练参数
恰好出现一次，既不能遗漏也不能重复。当前两组学习率相同，但保留分组接口是
为了与 ACT backbone 的独立学习率配置方式一致。

## 2. Episode split 与训练样本定义

### 2.1 Episode-disjoint split

默认流程先发现并校验所有合法 episode，再按 seed 对 episode 做确定性 shuffle，
以 episode 为单位划分 train/validation；同一个 episode 不会同时出现在两边。
也可以通过 experiment manifest 显式指定 train、validation 和未使用 episode。
manifest 的 provenance 会写入 checkpoint。

拆分发生在统计量计算和 window 构造之前，避免同一轨迹的相邻时间步跨越训练集
与验证集。

### 2.2 Dataset 长度

当前高频力 dataset 会把每个训练 episode 的每个 state timestep 都暴露为一个
确定性样本：

```text
len(dataset) = sum(num_steps of selected episodes)
```

因此这里的“一个 data epoch”是对所有 state window 的一次遍历，不等于 Official
ACT 中“每个 episode 随机抽一个 timestep”的 reference epoch。训练 DataLoader
对全部 window shuffle；validation DataLoader 不 shuffle。

### 2.3 单个 timestep 的输入与目标

对 episode 中当前索引 `t`：

1. 读取当前 `qpos[t]`；
2. 用 state timestamp 因果选择不晚于当前时刻的图像；
3. 选择当前时刻之前最后 100 个原生 500 Hz wrench；
4. 按真实 state timestamp 把在线 wrench 打包为最多 7 个区间；
5. 读取 `action[t:t+100]`，episode 尾部右侧补零并生成独立 action mask；
6. 为每个 action step `j` 收集 `(t_j,t_{j+1}]` 内全部原生 wrench；
7. 每个未来区间最多放 20 点，生成 sample mask 与 interval mask；
8. 取每个有效未来区间最后一个原生 wrench，构成 endpoint-force target。

最后一个 action 如果没有下一 state timestamp，动作标签仍然有效，但不存在合法
future-force interval；该时间步只参与 action loss，不参与两类 force loss。

## 3. 时间戳与高频力数据契约

### 3.1 在线历史

在线窗口只允许选择 `force_timestamp <= current_state_timestamp` 的最后 100 点。
episode 开头不足 100 点时使用 mask 左侧补空位，不复制首个实测值。被选中的
每个原生样本必须恰好进入一个在线区间；容量不足时抛错，而不是静默截断。

### 3.2 未来区间

第 `j` 个动作对应的力响应区间为：

```text
(state_time[j], state_time[j+1]]
```

区间是“随后一个 state/control segment 内的高频响应”，不是声称演示采集器在
整个区间内把 `action[j]` 做了严格 zero-order hold。训练方法保留原始采样信息，
但不改变演示控制器在物理步上的真实更新方式。

### 3.3 Mask 独立性

训练 batch 中维护三类彼此独立的 mask：

- `action_padding_mask [B,100]`：动作是否存在；
- `future_force_interval_padding_mask [B,100]`：对应未来力区间是否存在；
- `future_force_sample_padding_mask [B,100,20]`：区间内部哪些原生点存在。

不能用 action mask 代替 force mask，因为 episode 尾部可能有合法 action，却没有
后续时间边界或原生力样本。

## 4. 图像、qpos、action 与 force 的标准化

所有统计量只从 training episodes 计算，validation 与未使用 episode 均不参与。

### 4.1 qpos 与 action

qpos 使用训练 episode 中所有 state-rate `observations/joint_pos`；action 使用
所有 state-rate `action`。分别按维计算 population mean/std：

```text
x_normalized = (x - mean) / max(std, 1e-6)
```

### 4.2 wrench

force mean/std 使用训练 split 中 `observations/ft_wrench` 的每一个原生 500 Hz
样本，而不是先对齐到 state rate 再统计。这样标准化本身不会丢弃模型准备利用的
高频变化。六维 wrench 使用各自的 mean/std；有效样本标准化后，padding 位置
严格置零。

### 4.3 图像

图像转 `[0,1]` float，只有尺寸不匹配时才 resize 到 checkpoint 的
`480 x 640`，随后执行 ImageNet normalization。训练与 rollout 共享相同处理
顺序，checkpoint 记录图像尺寸、是否预训练和是否 ImageNet normalize。

## 5. Batch 张量契约

标准 batch 形状为：

| 张量 | 形状 |
| --- | --- |
| images | `[B,2,3,480,640]` |
| qpos | `[B,7]` |
| action chunk | `[B,100,7]` |
| action mask | `[B,100]` |
| online force intervals | `[B,7,20,6]` |
| online relative time | `[B,7,20]` |
| online sample mask | `[B,7,20]` |
| online interval mask | `[B,7]` |
| future force intervals | `[B,100,20,6]` |
| future relative time | `[B,100,20]` |
| future sample mask | `[B,100,20]` |
| future interval mask | `[B,100]` |
| endpoint force target | `[B,100,6]` |

每个区间内有效原生样本必须左对齐，后面才是 padding。模型与 criterion 在每步
训练前都会验证 shape、dtype、device、mask 一致性及有效值的 finiteness。

## 6. Objective loss

总目标为：

```text
L = 1.0 * L_action
  + 1.0 * L_force_endpoint
  + 0.5 * L_force_native
  + 10.0 * KL(q_contact || N(0,I))
  + 1.0 * KL(stopgrad(q_contact) || p_contact_conditional)
```

### 6.1 Action reconstruction

`L_action` 是 `[B,100,7]` 上的 masked L1。只对有效 action step 和 7 个标量维度
求平均：

```text
sum(abs(pred_action-target) * valid_action)
/ (valid_action_step_count * 7)
```

### 6.2 区间末端力 reconstruction

`L_force_endpoint` 比较 `[B,100,6]` 输出与每个有效未来区间的最后一个原生
wrench。它只在 valid force interval 上平均。

末端力头提供每个 action-scale 区间的紧凑监督：在局部编码器的 mean/max/last
统计中，last 与下一状态边界附近的接触状态直接对应。它与 native-force loss
不是重复的：前者强调区间结尾，后者约束区间内部完整轨迹。

### 6.3 原生 500 Hz force reconstruction

`L_force_native` 比较 `[B,100,20,6]` prediction 和完整 future force interval。
reduction 分三步：

1. 对 6 个 wrench 维度求平均，得到每个原生样本的 L1；
2. 在每个 interval 内只对有效原生样本求平均；
3. 对所有有效 interval 等权求平均。

因此含 17 个原生点的区间不会仅因样本多于 16 点就获得更高总权重。这样的
reduction 以“一个 action/state interval”为基本监督单位，同时保留区间内所有
500 Hz 变化。

如果一个 batch item 没有有效 force interval，两类 force loss 对该 item 不产生
监督；实现返回可微的零，而其有效 action 仍可训练。

### 6.4 Posterior KL

posterior 为对角高斯 `q=N(mu_q,diag(var_q))`。标准正态 KL 对 32 个 latent 维度
求和，再对 batch 求平均：

```text
L_KL = mean_batch sum_latent KL(q || N(0,I))
```

权重 10 与 ACT 的标准 KL 权重保持同一量级。该项约束训练 posterior 接近部署
可用的简单 latent 区域，但不保证模型一定使用 latent；latent 使用程度需通过
单独诊断，而不是修改本方法定义。

### 6.5 Asymmetric conditional-prior matching

prior loss 是两个对角高斯之间的：

```text
KL(stopgrad(q_contact) || p_contact_conditional)
```

posterior 的 `mu_q` 与 `logvar_q` 在 loss 内 detach；同时 prior 的 qpos、在线力、
力—视觉特征和 visual summary 输入在模型 forward 中也 detach。因此这项 loss：

- 更新 conditional prior trunk 与两个 prior heads；
- 不更新 posterior；
- 不更新共享高频力编码器；
- 不更新视觉、在线力或 policy 主干。

这种非对称设计使 prior 把当前 posterior 当作稳定教师，避免 prior 尚未学好时通过
匹配项反向拖动 posterior。posterior 仍会从 reconstruction loss 和 standard-normal
KL 获得正常梯度。

## 7. 单个 optimizer step

一个训练 step 的严格顺序为：

1. batch 移到 device 并验证全部契约；
2. `model.train()`；
3. `optimizer.zero_grad(set_to_none=True)`；
4. 调用训练专用 `forward_train`；
5. 用 reparameterization 随机采样 posterior `z_contact`；
6. decoder 使用 posterior latent 做三类 reconstruction；
7. 计算五项 loss 与加权总 loss；
8. 检查总 loss finite；
9. `backward()`；
10. 收集所有实际获得梯度的可训练参数并计算全局 L2 gradient norm；
11. 如果显式配置 gradient clipping，则以同一阈值裁剪；默认只记录不裁剪；
12. 再次检查 gradient norm finite；
13. `optimizer.step()`。

没有 learning-rate scheduler、gradient accumulation 或 automatic mixed precision
的隐式逻辑。一个 DataLoader batch 对应一个 optimizer step。

## 8. 训练长度：最低训练量、收敛判断与安全上限

canonical config 仍用 Official ACT 的 reference epoch 定义初始训练预算：

```text
steps_per_reference_epoch = ceil(reference_train_episodes / batch_size)
max_optimizer_steps       = steps_per_reference_epoch * official_reference_epochs
```

标准 `official_reference_epochs=2000`。它提供可复现的初始 update 尺度，不代表
两个模型都已充分训练，也不要求 Official ACT 与 Contact-CVAE 在同一步停止。

Contact-CVAE 的 data epoch 定义为：

```text
steps_per_data_epoch = ceil(total_train_windows / batch_size)
```

它决定何时完成一次全 window 遍历并触发验证。formal run 支持三个独立控制量：

- `--target-optimizer-steps`：绝对 global-step 安全上限，不是追加步数；
- `--minimum-optimizer-steps`：达到此步数前不允许因平台期停止；
- `--early-stop-patience-validations`：达到 minimum 后，连续多少次 full validation
  无有效改善才停止。

`--early-stop-min-relative-improvement` 默认 `0.01`，即相对下降至少 1% 才清空
patience。小于阈值但严格更低的指标仍保存为 best，因此模型选择与平台判断相互
独立。不传 patience 时保持固定预算；传 patience 但省略 minimum 时，minimum
取 checkpoint 原始 canonical step limit。

## 9. Formal、burn-in 与 smoke

- `formal`：使用 canonical model，训练到验证收敛或绝对 step 上限；
- `burn_in`：使用同一目标和数据契约，但必须显式给定不超过正式预算的绝对
  `--max-train-steps`；
- `--smoke`：使用小宽度、小图像、短 chunk 和少量样本，只验证结构和流程。

burn-in 与 formal 应使用不同目录。所有 step 参数都是绝对 global step。

## 10. Validation 路径与模型选择

validation 使用 `model.eval()` 和 `torch.no_grad()`，并计算三条路径：

1. posterior mean：可见 future label 的确定性 oracle 诊断；
2. deployment zero：只用在线输入，`z_contact` 严格为零；
3. deployment prior：只用在线输入，使用 conditional-prior mean。

三条路径分别计算 action、endpoint force、native-force L1，并记录 KL、latent
统计与相对 zero 输出差异。validation 前后保存并恢复 Python、NumPy、Torch 和
CUDA RNG，保证验证不改变后续训练随机序列。

高频模型的 best selection metric 固定为：

```text
deployment_zero_action_l1_physical
```

该路径枚举 episode-disjoint validation split 的每个 `(episode,timestep)` window，
用 zero contact latent 确定性推理；prediction 与 label 反标准化到物理关节单位，
再对所有有效 action scalar 做全局加权 L1。posterior/prior 指标只作诊断。

fresh formal run 在训练前执行 full validation 建立基准。续训也在 run start 复评；
旧 checkpoint 若没有当前物理指标和可定位 best，会重新建立基准并写出迁移记录。

## 11. 日志、checkpoint 与精确恢复

`metrics.jsonl` 的 `step` 记录 loss、latent、gradient、LR 与显存；
`epoch_segment` 记录完整或部分 data epoch、validation 和 convergence state；formal
run 还写 `full_validation(stage=run_start)`。

保存文件包括：

- `step_XXXXXXXX.pt`：周期性完整 checkpoint；
- `last.pt`：每个 data-epoch segment 后更新；
- `best.pt`：物理 selection metric 严格改善时更新；
- `final.pt`：达到 step 上限或验证平台期停止；
- `burn_in.pt`：burn-in 达到 step 上限。

checkpoint 使用临时文件加 `os.replace` 原子保存，并包含 model、optimizer、
progress、normalization、split、experiment provenance、全部 RNG、DataLoader
generator、runtime horizon、convergence monitor、父 checkpoint 内容指纹、best
artifact 描述和可能的指标迁移记录。

resume 会严格检查 config 并恢复 epoch 内 shuffle 起点及已完成 batch 数。最终
checkpoint 立即执行 reload audit。新目录续训且父 best 未被超越时，`best.pt` 是到
父 best 的符号链接；出现新 best 后，原子保存替换链接本身，不覆盖父文件。

## 12. 充分训练的推荐 continuation 协议

以 40 个 train episodes、batch 8、已训练到 25,000 steps 为例，可把 50,000
设为新 minimum、100,000 设为安全上限：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python \
  scripts/train_act_aligned_high_rate_contact_cvae.py \
  /path/to/data \
  --output-dir runs/contact_converged_b8_seed0 \
  --device cuda --batch-size 8 --num-workers 2 --seed 0 \
  --run-mode formal \
  --resume runs/contact_previous/final.pt \
  --target-optimizer-steps 100000 \
  --minimum-optimizer-steps 50000 \
  --early-stop-patience-validations 5 \
  --early-stop-min-relative-improvement 0.005 \
  --checkpoint-interval-steps 2000 --log-interval 10
```

100,000 只是故障保护上限。每个模型按自己的 validation 曲线独立停止，实际训练
量可以不同。minimum、patience、relative threshold 和上限必须在启动前固定。

## 13. 方法边界

1. 训练 decoder 使用 posterior、默认部署使用 zero latent，是有意保留的非对称。
2. conditional prior 可独立评估，但 prior matching 不应更新共享在线编码器或
   posterior。
3. native-force reconstruction 不等价于 500 Hz 闭环控制。
4. validation L1 与 latent 诊断不是任务成功率；成功语义属于 rollout 方法。
5. 修改 loss、采样、训练边界或选模指标时必须升级或明确记录协议。

## 14. 源码索引

- 训练入口：`scripts/train_act_aligned_high_rate_contact_cvae.py`
- 通用编排：`scripts/train_act_aligned_contact_cvae.py`
- dataset：`src/force_aware_act/act_aligned_training/high_rate_data.py`
- criterion：`src/force_aware_act/act_aligned_training/high_rate_losses.py`
- train/validation：`src/force_aware_act/act_aligned_training/high_rate_trainer.py`
- checkpoint：`src/force_aware_act/act_aligned_training/checkpoint.py`
- horizon：`src/force_aware_act/training/horizon.py`
- convergence：`src/force_aware_act/training/convergence.py`
- artifact lineage：`src/force_aware_act/training/artifacts.py`
