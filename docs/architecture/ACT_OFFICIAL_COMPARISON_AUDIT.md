# ACT 官方实现对比审计

## 1. 审计基准与结论

审计基准是 `/home/stw/act` 的已提交版本 `742c753`，不是该目录当前的
工作树。当前工作树包含两项未提交修改：

- `constants.py` 的本地数据目录；
- `detr/models/detr_vae.py` 将 decoder 输出由官方的 `[0]` 改为
  `[-1]`。

因此，“官方行为”以 `git show 742c753:<file>` 为准。结论如下：

1. 新实现的主干宽度、Transformer 深度、attention/FFN 形式、视觉
   backbone、chunk/query 数和默认 zero latent 已与官方训练示例对齐。
2. 本次审计修正了两个此前未完全对齐的默认项：时间序列位置编码由
   learned 改为官方的一维固定正弦编码；主学习率由 `1e-4` 改为官方
   README 示例的 `1e-5`。
3. action/force/contact 的增加属于任务扩展，不是假装等价的官方复刻；
   它们仍遵守相同宽度、四层 encoder、七层 decoder 和一致的时序编码
   规则。
4. 新实现有意不复现官方已知缺陷，包括使用第一层 decoder 输出、错误
   的 masked-L1 分母、验证集随机采样、全数据统计量泄漏和不完整
   checkpoint。

审计后未发现阻止进入完整训练预检的结构性差异。

## 2. 官方 ACT 的精确基线

官方 README 给出的 ACT 训练示例是：

```text
KL weight=10, chunk_size=100, hidden_dim=512, batch_size=8,
dim_feedforward=3200, num_epochs=2000, lr=1e-5, seed=0
```

`imitate_episodes.py` 固定 ResNet18、backbone LR `1e-5`、encoder 4 层、
decoder 7 层、8 heads、双臂 qpos/action 维度 14。模型实际配置为：

```text
D=512, heads=8, head_dim=64, FFN=3200
posterior encoder=4, policy encoder=4, decoder=7
latent_dim=32, queries/chunk=100
dropout=0.1, ReLU, post-norm
```

官方 posterior 的数据流为：

```text
qpos [B,14]                 -> Linear(14,512) -> [B,1,512]
actions [B,100,14]          -> Linear(14,512) -> [B,100,512]
CLS [1,512]                 -> repeat         -> [B,1,512]
concat [CLS,qpos,actions]                    -> [B,102,512]
fixed 1D sine positions                     -> [1,102,512]
4-layer encoder; take CLS                   -> [B,512]
Linear(512,64); split                        -> mu/logvar [B,32]
reparameterize                               -> z [B,32]
```

部署时不运行 posterior，直接令 `z=zeros([B,32])`。图像先 `/255`，
再做 ImageNet mean/std normalization。每个 camera 共用一个 pretrained
ResNet18 + FrozenBatchNorm，取 layer4 `[B,512,H/32,W/32]`，经过
`Conv2d(512,512,1)`；不同相机沿 width 拼接。视觉位置是 DETR 二维正弦
编码，官方没有 camera identity embedding。

policy memory 为：

```text
latent [B,512], qpos [B,512], visual spatial tokens
-> [2 + sum_c(Hc*Wc), B, 512]
-> 4-layer policy encoder
```

100 个 learned query positions 与全零 decoder targets 进入七层 decoder。
各 decoder 层均为 self-attention、cross-attention、FFN，且使用 post-norm；
position 只加到 query/key，value 保持内容张量。decoder 有最终 LayerNorm
并返回七层 intermediate stack。提交版本随后错误地取 `[0]`，即第一层
结果；本地未提交修补才改为 `[-1]`。官方仅有
`Linear(512,14)` action head；`is_pad_head` 被计算但不参与 loss。

官方训练目标是：

```text
L = L1(action, pred_action) + 10 * KL(q(z|qpos,action) || N(0,I))
```

优化器为 AdamW、weight decay `1e-4`，按参数名是否包含 `"backbone"`
分组。README 示例中主干与非主干 LR 都是 `1e-5`。没有 scheduler，
parser 中的 clip norm 也未用于训练。

官方 Dataset 每个 episode 每次只随机抽一个起点，取从该点到 episode
末尾的 action 并补零；训练和验证 DataLoader 都 shuffle，验证起点也会
变化。qpos/action 统计量在划分后仍由全部 episode 计算，且实现要求各
episode 等长。masked L1 使用整个补零张量的 `.mean()`，所以 padding
比例会改变 loss 尺度。每个 epoch 先验证再训练；checkpoint 主要保存
model state_dict，不保存 optimizer、RNG 或完整配置。

## 3. 新实现逐项对照

| 项目 | 官方 ACT `742c753` | ACT-aligned contact-CVAE | 判定 |
|---|---|---|---|
| 宽度/heads/FFN | 512 / 8 / 3200 | 512 / 8 / 3200 | 一致 |
| encoder/decoder | posterior 4，policy 4，decoder 7 | posterior 4，force 4，policy 4，decoder 7 | 同类一致，新增 force encoder |
| latent/chunk | 32 / 100 | 32 / 100 | 一致 |
| Transformer | ReLU、dropout 0.1、post-norm | 相同 | 一致 |
| posterior position | 固定一维正弦 | 固定一维正弦 | 一致 |
| visual position | 二维正弦 | 二维正弦 + camera identity | 后者为多相机扩展 |
| query | 100 learned positions，zero targets | 相同 | 一致 |
| decoder 输出 | 错取 intermediate `[0]` | final normalized layer | 有意修复 |
| backbone | pretrained ResNet18、FrozenBN | 相同 | 一致 |
| 输入图像 | RGB、CHW、`/255`、ImageNet normalize | RGB、CHW、resize 224、同一 normalize | 对齐并固定尺寸 |
| qpos/action | 双臂 14/14 | 单臂 7/7 | 任务维度差异 |
| force | 无 | history `[B,20,6]`；future `[B,100,6]` | 新增任务模态 |
| prediction heads | `Linear(512,14)` action | 并行 `Linear(512,7/6)` action/force | 同型扩展 |
| 部署 latent | zero | 默认 zero；可选 prior mean/sample | 默认一致 |
| 主/backbone LR | `1e-5` / `1e-5` | `1e-5` / `1e-5` | 一致 |
| AdamW/WD | AdamW / `1e-4` | AdamW / `1e-4` | 一致 |
| epoch/batch/KL weight | 2000 / 8 / 10 | 2000 / 8 / 10 | 一致 |

新 posterior 的序列为：

```text
[CLS, qpos, action_t + future_force_t for t=0..99]
-> [B,102,512] -> 4-layer encoder
-> mu_contact/logvar_contact/z_contact [B,32]
```

同一 future timestep 的 action `[7]` 和 wrench `[6]` 分别投影到 512
后相加，避免把一个 timestep 拆成两个互相竞争的序列位置。在线 force
序列为 `[CLS_F, force_0..force_19] -> [B,21,512]`，同样采用固定一维
正弦位置和四层 encoder，输出 `z_F_online [B,512]`。

策略 memory 明确为：

```text
[z_contact, qpos, z_F_online, z_VF, visual_0..visual_97]
-> [B,102,512]  # 两个 224x224 camera，各 7x7
-> 4-layer encoder
-> 7-layer decoder with 100 queries
-> hidden [B,100,512]
-> pred_action [B,100,7], pred_force [B,100,6]
```

action 与 force head 都直接读取同一个最终 decoder hidden。force head
不再额外拼接 latent，因此层数和数据流与 ACT head 保持同型。

## 4. 条件 prior 与非对称训练

这是仓库的核心扩展：

```text
q_contact = q(z | qpos, future_action, future_force)
p_contact = p(z | qpos, force_history, current_images)
```

训练使用 posterior sample；部署默认保留官方 ACT 的 zero latent。
conditional-prior 是可选部署模式，同时通过以下非对称目标稳定学习：

```text
L_total =
    masked_L1(action)
  + lambda_force * masked_L1(force)
  + 10 * KL(q_contact || N(0,I))
  + lambda_prior * KL(stop_gradient(q_contact) || p_contact)
```

最后一项对 posterior 的 mean/logvar 显式 detach，所以 prior matching
只更新 conditional prior，不会反向拖动 posterior。posterior 仍由重构
项和标准正态 KL 学习。这一设计不是官方 ACT 原有内容，但与“训练使用
posterior、默认部署使用 zero”的要求兼容。

## 5. 有意保留的工程改进

- action/force masked L1 只除以有效 scalar 数，不复现官方 padding
  稀释 loss 的问题。
- episode-disjoint deterministic split；统计量只使用 train episodes。
- 验证窗口固定，不在每次验证时重新随机选择 timestep。
- 图像与 force 都按不晚于 decision timestamp 的因果规则对齐。
- optimizer 通过模块 identity 分组，不依赖参数名字字符串。
- checkpoint 原子写入 model、optimizer、epoch、配置、统计量、manifest
  和 RNG，可准确 resume。
- validation 分别报告 posterior mean、zero deployment 和 conditional
  prior mean，默认以 zero-deployment action L1 选择 best checkpoint。

## 6. 审计后的进入条件

进入正式长训练前仍需完成下一阶段预检：

1. 用 canonical `D=512/K=100/224x224` 真实构型执行一次 forward/backward；
2. 输出各模块参数量、optimizer 分组覆盖和实际 LR；
3. 验证 posterior、prior、force encoder、fusion、四层 policy encoder、
   七层 decoder 及两个 prediction heads 的梯度均符合各 loss 路径预期；
4. 记录峰值显存与单步耗时，为 batch size 选择提供依据。

这些是运行资源与梯度连通性的验证，不再改变已审计的模型定义。
