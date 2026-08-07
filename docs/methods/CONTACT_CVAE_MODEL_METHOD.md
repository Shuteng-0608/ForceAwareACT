# Contact-CVAE 模型框架与编码方法

本文只说明当前分支 `refactor/act-aligned-contact-cvae` 中正式使用的
高频力 Contact-CVAE 模型，不记录任何训练或 rollout 实验结果。

当前架构版本是：

```text
act_aligned_contact_cvae_highrate_force_v2
```

对应实现入口为
`ACTAlignedHighRateContactCVAEPolicy`。文中的“当前模型”均指这个版本，
不是较早的 state-rate Contact-CVAE，也不是 motion-CVAE 或 Official ACT。

## 1. 设计目标与总体数据流

模型保留 ACT 的基本骨架：多相机视觉编码、Transformer policy encoder、
learned action queries 和 Transformer decoder；在此基础上加入两类力信息：

1. 部署时可获得的过去 500 Hz 力窗口，用于构造在线接触上下文；
2. 仅训练时可获得的未来 500 Hz 力区间，用于学习
   `q(z_contact | qpos, future actions, future force)`。

总体数据流为：

```text
两路 RGB 图像 -> 600 个视觉 token ------------------------+
                                                        |
过去 100 个 500 Hz wrench -> 7 个区间 token -> z_F_online |-> policy encoder
                                      |                 |      -> 100 个 action query
                                      +-> 视觉交叉注意力 -> z_VF |      -> policy decoder
                                                        |      -> 三类预测头
当前 qpos ----------------------------------------------+
训练: future action + future 500 Hz force -> posterior z_contact
部署: z_contact = 0（默认）或 conditional-prior mean
```

这种拆分的原则是：部署前向只允许使用在线可见数据；未来动作和未来力只能进入
训练专用 posterior，防止无意中的未来信息泄漏。

## 2. 统一符号与标准配置

| 符号 | 含义 | 标准值 |
| --- | --- | ---: |
| `B` | batch size | 运行时决定 |
| `K` | action chunk 长度 | 100 |
| `D` | Transformer 隐空间 `d_model` | 512 |
| `D_z` | `z_contact` 维数 | 32 |
| `D_q` | 关节位置维数 | 7 |
| `D_a` | 动作维数 | 7 |
| `D_f` | wrench 维数 | 6 |
| `S` | 每个状态区间容纳的原生力样本数 | 20 |
| `I_online` | 在线历史最多包含的状态区间数 | 7 |
| `N_cam` | 相机数 | 2 |

所有 ACT 风格 Transformer 使用同一套核心配置：

| 配置 | 数值 |
| --- | ---: |
| attention heads | 8 |
| 每头维度 | 64 |
| encoder layers | 4 |
| policy decoder layers | 7 |
| feed-forward width | 3200 |
| activation | ReLU |
| dropout | 0.1 |
| normalization | post-norm |

单个 encoder layer 的顺序是：8-head self-attention、残差与 LayerNorm、
`512 -> 3200 -> 512` FFN、残差与 LayerNorm。位置编码只加到 attention 的
query/key，value 保持原 token。四层 encoder 后不再额外添加 stack-level
LayerNorm。

单个 decoder layer 的顺序是：8-head self-attention、8-head memory
cross-attention、`512 -> 3200 -> 512` FFN；三个子层均采用残差、dropout 和
post-norm。七层 decoder 之后有一个共享的最终 LayerNorm。

标准模型共有 `104,038,597` 个参数，所有参数默认可训练。ResNet18 body
单独进入 backbone optimizer group，但并未冻结。

## 3. 输入、训练专用信息与输出

### 3.1 部署可见输入

| 输入 | 进入模型前的形状 | 含义 |
| --- | --- | --- |
| `images` | `[B,2,3,480,640]` | 两路同步 RGB 图像 |
| `qpos` | `[B,7]` | 当前关节位置 |
| online force intervals | `[B,7,20,6]` | 过去 100 个原生 500 Hz wrench 的区间表示 |
| relative time | `[B,7,20]` | 每个力样本在所属区间内的相对时间 |
| sample padding mask | `[B,7,20]` | 原生样本 padding 标记 |
| interval padding mask | `[B,7]` | 整个区间是否无效 |

这里的 wrench 六维顺序沿用数据集：前三维为力，后三维为力矩。模型不会改变
物理单位；标准化只改变数值尺度，rollout 输出再按训练统计量反标准化。

### 3.2 仅训练 posterior 可见的输入

| 输入 | 形状 |
| --- | --- |
| normalized action chunk | `[B,100,7]` |
| future force intervals | `[B,100,20,6]` |
| future relative time | `[B,100,20]` |
| action padding mask | `[B,100]` |
| future sample padding mask | `[B,100,20]` |
| future interval padding mask | `[B,100]` |

未来第 `j` 个力区间严格定义为 `(t_j, t_{j+1}]`。左开保证它与时刻
`t_j` 可见的在线历史不重叠；右闭保证相邻区间不重复也不遗漏边界样本。

### 3.3 模型输出

| 输出 | 形状 | 含义 |
| --- | --- | --- |
| `pred_action` | `[B,100,7]` | 100 步绝对关节目标 |
| `pred_force` | `[B,100,6]` | 每个未来区间末端 wrench |
| `pred_force_highrate` | `[B,100,20,6]` | 每个未来区间的原生 500 Hz wrench 序列 |

三个输出均直接由最后一层 decoder hidden state 产生。`pred_force` 不再额外
拼接 `z_contact`；latent 已经作为 policy memory token 影响 decoder。

## 4. 图像编码

### 4.1 输入准备

HDF5 图像先从 `uint8 HWC` 转成 `float32 CHW` 并除以 255。图像索引采用
相对于当前 state timestamp 的因果对齐，不选择未来图像。标准输入保持数据集
原生 `480 x 640`；只有实际尺寸不符合 checkpoint 配置时才做 bilinear resize，
并使用 `antialias=True`。因为使用 ImageNet 预训练权重，随后执行 ImageNet
mean/std normalization。

这样设计有两个原因：一是训练与 rollout 使用同一预处理；二是保留原生画面
比例和空间细节，不再无条件压缩为正方形 224 x 224。

### 4.2 ResNet18 tokenization

两台相机共享一个 ResNet18，而不是各自一套 backbone：

```text
[B,2,3,480,640]
-> reshape [2B,3,480,640]
-> ResNet18 去掉 avgpool 和 fc
-> [2B,512,15,20]
-> 1x1 Conv: 512 -> 512
-> flatten spatial grid
-> [B,2*15*20,512] = [B,600,512]
```

ResNet18 使用 ImageNet 默认权重和 FrozenBatchNorm2d。FrozenBatchNorm 固定
统计量和 affine buffers，但卷积权重仍参与训练。1x1 projection 即使输入输出
同为 512 也保留，以维持 ACT/DETR 风格的显式 backbone-to-transformer 接口。

### 4.3 视觉位置编码

每个 `15 x 20` feature grid 使用 DETR 风格二维 sine position encoding；
此外为两台相机分别加入 learned camera embedding。token 排列顺序是
camera-major，再按每个 feature grid 的 row-major 排列。因此输出为：

```text
visual_tokens   : [B,600,512]
visual_position : [B,600,512]
visual_summary  : mean over 600 tokens -> [B,512]
```

二维位置编码使注意力知道像素区域，camera embedding 使相同网格坐标在不同
视角下可区分。`visual_summary` 只进入 conditional prior，不替代空间 token。

## 5. 当前关节状态编码

normalized qpos 使用单层线性适配器：

```text
[B,7] -> Linear(7,512) -> [B,512] -> unsqueeze -> [B,1,512]
```

这里不使用额外 MLP，是为了让 qpos 与 ACT posterior 中的输入适配方式一致，
并把跨模态非线性组合交给后续 Transformer。该 token 同时用于 policy memory；
去掉长度维后得到 `[B,512]` 的 `qpos_feature`，供 conditional prior 使用。

## 6. 原生 500 Hz 力区间的共享局部编码

### 6.1 为什么先划分区间

在线历史是当前时刻之前最后 100 个原生力样本，约覆盖 0.2 秒。它们根据真实
state timestamp 被装入最多 7 个状态区间，每区间最多 20 点。future force
则按 100 个动作时间区间装入。区间成员关系由 timestamp 计算，不假设
`500/30` 是整数，也不先降采样到 30 Hz。

在线和未来区间使用同一个已注册的 `ACTAlignedHighRateForceEncoder`，共享全部
权重。共享编码使“过去某段 500 Hz 接触变化”和“未来某段 500 Hz 接触响应”
处在同一个 512 维表示空间。

### 6.2 单个区间的四层 TCN

输入 `[B,I,20,6]` 先把 batch 与 interval 合并。每个 6D wrench 经
`Linear(6,128)`，再加入 128 维 sinusoidal relative-time encoding。时间坐标
为 `relative_time * 500`，因此编码直接以原生采样步为尺度。

随后经过 4 个 mask-safe residual temporal convolution blocks，dilation 依次为
`1, 2, 4, 8`。每一层结构为：

```text
LayerNorm(128)
-> Conv1d(128,128,kernel=3,dilation=d,padding=d,bias=False)
-> ReLU
-> LayerNorm(128)
-> Conv1d(128,128,kernel=3,dilation=d,padding=d,bias=False)
-> Dropout(0.1)
-> residual add
-> ReLU
```

每次卷积前后都重新应用 valid-sample mask，防止 padding 值通过卷积传播到有效
位置。四层 dilation 在仅 20 点的局部区间内同时覆盖短时尖峰和较长变化趋势。

### 6.3 区间池化

四层输出为 `[B*I,20,128]`。只对有效样本分别计算：

- mean pooling：区间平均状态；
- max pooling：幅值突变或峰值模式；
- last pooling：区间末端状态。

三者拼接为 384 维，再执行：

```text
[B*I,384] -> Linear(384,512) -> LayerNorm(512)
           -> reshape [B,I,512]
```

这三种统计互补：mean 表示整体水平，max 保留冲击，last 与末端力监督及下一时刻
状态更直接相关。整个区间无有效样本时输出严格置零。

## 7. 在线力历史编码

共享局部编码器先把 7 个历史区间变为 `[B,7,512]`。随后在线聚合器添加一个
learned force CLS token：

```text
[CLS_F, interval_0, ..., interval_6] -> [B,8,512]
```

序列加入固定 sinusoidal sequence position encoding 和两类 learned token-type
embedding（CLS/interval），再进入 4 层 ACT Transformer encoder。padding
interval 通过 `[B,8]` mask 排除，最终取 CLS 输出：

```text
z_F_online : [B,512]
```

局部 TCN 负责区间内 500 Hz 模式，四层 Transformer 负责跨多个状态区间组合；
二者分工避免直接在 100 个原生样本上使用一个没有时间分层的全局编码器。

## 8. 力—视觉交叉注意力

`z_F_online` 作为唯一 query，600 个视觉 token 作为 key/value：

```text
query : z_F_online -> [B,1,512]
key   : visual_tokens + visual_position -> [B,600,512]
value : visual_tokens -> [B,600,512]
MHA(embed=512, heads=8, dropout=0.1)
-> z_VF [B,512]
```

这里是一个独立的 multi-head cross-attention 层，不是四层 Transformer。
设计含义是“用当前接触历史去询问哪些视觉区域与接触状态相关”，同时仍保留
完整视觉 token 供主 policy encoder 使用。

## 9. Contact posterior：训练时学习 `z_contact`

posterior 接收 normalized qpos、未来动作和经过共享局部力编码器得到的未来力
token。具体步骤为：

```text
qpos [B,7]       -> Linear(7,512)   -> [B,1,512]
actions [B,100,7]-> Linear(7,512)   -> [B,100,512]
future force intervals              -> shared local encoder
                                     -> [B,100,512]
contact step_j = LayerNorm(action_token_j + force_token_j)
```

然后构造 102-token 序列：

```text
[CLS_contact, qpos, contact_step_0, ..., contact_step_99]
-> [B,102,512]
```

序列加入固定 sinusoidal position encoding 和三类 learned type embedding
（CLS、qpos、contact-step），进入 4 层 ACT Transformer encoder。action padding
mask 前补两个永远有效的位置，用于屏蔽尾部无效时间步。取 CLS 输出后使用两个
独立线性头：

```text
Linear(512,32) -> mu_contact
Linear(512,32) -> logvar_contact
z_contact = mu + exp(0.5*logvar) * epsilon
```

训练时使用 reparameterization sample；确定性验证时使用 `mu_contact`。

action 和 force 在同一时间步相加而不是先后拼成两个长序列，是为了显式建立
“动作区间—接触响应区间”的逐步对应关系，同时把 posterior token 数保持为
`K+2`，与 ACT motion posterior 的结构尺度一致。

## 10. Conditional contact prior

prior 只使用部署时可获得的四个 512 维特征：

```text
qpos_feature
z_F_online
z_VF
visual_summary
-> concat [B,2048]
-> Linear(2048,512) -> ReLU -> Dropout(0.1)
-> Linear(512,32) for prior mean
-> Linear(512,32) for prior log-variance
```

训练时这四个输入在进入 prior 前全部 `detach()`；prior matching loss 中的
posterior mean/log-variance 也 `detach()`。因此 prior 拟合 posterior 分布时，
梯度只更新 prior 网络，不会反向拖动 posterior、视觉编码器、在线力编码器或
共享局部力编码器。

prior 并不替代默认部署策略。默认部署仍传入严格全零 `z_contact`；prior mean
是可显式选择的部署/诊断模式，prior sample 主要保留为模型能力接口。

## 11. Policy memory encoder

32 维 contact latent 先经 `Linear(32,512)`。四个 special token 按固定顺序为：

```text
[z_contact, qpos, z_F_online, z_VF]
```

每个 special token 加独立 learned type/position embedding，然后在其后拼接
600 个带二维空间与相机位置编码的视觉 token：

```text
[B,4,512] + [B,600,512] -> [B,604,512]
```

这 604 个 token 进入 4 层 ACT Transformer encoder，得到同形状 policy
memory。四类在线上下文保留为独立 token，而不是先拼成一个 MLP 向量，使
self-attention 可以决定它们与各视觉区域之间的关系。

## 12. Action-query decoder 与预测头

模型维护 100 个 learned query embeddings，形状 `[100,512]`。对 batch 扩展后，
以全零 target 作为 decoder 初始内容，learned query 作为 query position；
七层 decoder 对 `[B,604,512]` policy memory 做 cross-attention：

```text
zero target + 100 learned query positions
-> 7-layer ACT decoder
-> final LayerNorm
-> decoder_hidden [B,100,512]
```

默认只使用最后一层 decoder 输出。三个预测头为：

```text
action head        : Linear(512,7)
endpoint force head: Linear(512,6)
high-rate force head: Linear(512,120) -> reshape [B,100,20,6]
```

100 个 query 与 100 个动作/力时间区间一一对应。高频力头在每个 decoder time
step 内同时预测最多 20 个 500 Hz 点，用于保留区间内动态监督；它不是独立的
500 Hz 闭环控制器。

## 13. 训练路径与部署路径的结构性区别

### 13.1 训练路径

训练调用 `forward_train`：

1. 编码图像、qpos 与过去高频力；
2. 用未来 action 和未来高频力构造 posterior；
3. 采样 posterior `z_contact`；
4. 同时计算只更新 prior 的 conditional-prior 输出；
5. decoder 始终以 posterior latent 做重建；
6. 输出动作、区间末端力和原生高频力预测。

### 13.2 部署路径

部署调用 `forward`，函数签名不接受未来 action 或未来 force：

1. 只编码当前图像、qpos 和因果在线力历史；
2. 仍可计算 conditional prior 供诊断；
3. 默认创建精确的 `[B,32]` 全零 `z_contact`；
4. 也可显式选择确定性 prior mean；
5. decoder 生成同样的 100 步输出。

这种 API 层面的分离比仅靠调用约定更可靠：部署路径无法误传 future label。

## 14. 主要子模块参数规模

| 子模块 | 参数量 |
| --- | ---: |
| ResNet18 vision backbone + 1x1 projection | 11,430,592 |
| shared local high-rate force encoder | 594,304 |
| online force 4-layer Transformer | 17,334,272 |
| force-vision cross-attention | 1,050,624 |
| high-rate contact posterior | 17,376,832 |
| conditional prior | 1,081,920 |
| policy 4-layer encoder | 17,332,736 |
| policy 7-layer decoder + query embeddings | 37,746,048 |
| 其余 adapters、positions 与三个 heads | 91,269 |
| **总计** | **104,038,597** |

## 15. 单独说明：修改前 `main` 主分支的 Contact-CVAE

本节只用于交代历史变化，不参与上面对当前模型的定义。

修改前的主要类是 `ForceAwareACTContactCVAEPolicy`，默认配置为
`chunk_len=50`、2 层 policy encoder、2 层 decoder、FFN 2048。图像无条件
resize 到 `224 x 224`，共享 ResNet18 后每相机产生 `7 x 7` token；没有当前
版本明确的二维 sine position 与 camera embedding。

旧 qpos 编码为 `Linear(7,512) -> ReLU -> Linear(512,512)`。在线力窗口默认
取 50 点、覆盖 0.25 秒，采用按均匀目标时刻取最近过去样本的方式；随后执行
`Linear(6,512)`、learned CLS/position 和 2 层 Transformer。未来力不是保留
每个状态区间内的原生 500 Hz 序列，而是为每个 state timestamp 选择一个最近
force 样本，得到 `[B,K,6]`。

旧 posterior 将序列构造成：

```text
[CLS, qpos, K action tokens, K force tokens]
```

即动作和力分成两段，长度为 `2K+2`，经过 2 层 Transformer；当前版本则把
每个 action 与其完整原生力区间先融合成一个 time-aligned token，再使用
`K+2` 的 4 层 posterior。

旧 policy memory 顺序为视觉 token 后接 `z_VF`、`z_q`、`z_F_online`、
`z_contact`，没有当前版本统一的 special-token 类型位置编码，且主 encoder /
decoder 都只有 2 层。旧 force head 还会把 32 维 `z_contact` 直接拼到每个
decoder hidden 后再预测单个 6D force；当前三个预测头全部只读最终的 512 维
decoder hidden，并新增 `[B,K,20,6]` 原生力重建头。

核心变化可概括为：ACT 层数与 token 规范对齐、chunk 从 50 统一到 100、图像
恢复原生 480 x 640、力信息由 state-rate 单点表示升级为原生 500 Hz 区间表示、
online/future 共享局部力编码器、posterior 变为逐时间步 action-force 融合、
训练/部署 API 严格拆分，以及 prior 的梯度隔离被明确固化。

## 16. 源码索引

- 配置：`src/force_aware_act/models/act_aligned/config.py`
- 总模型：`src/force_aware_act/models/act_aligned/high_rate_policy.py`
- 图像：`src/force_aware_act/models/act_aligned/backbone.py`
- 高频力局部编码：`src/force_aware_act/models/act_aligned/high_rate_force.py`
- 在线力聚合：`src/force_aware_act/models/act_aligned/online_force.py`
- posterior/prior：`src/force_aware_act/models/act_aligned/contact_latent.py`
- 力—视觉融合：`src/force_aware_act/models/act_aligned/fusion.py`
- Transformer：`src/force_aware_act/models/act_aligned/transformer.py`
