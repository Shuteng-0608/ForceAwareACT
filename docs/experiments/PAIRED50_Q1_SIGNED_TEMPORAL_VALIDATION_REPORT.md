# Paired-50 固定 Q=1 Signed Temporal Aggregation 实验报告

```text
实验日期：2026-08-06
实验性质：validation-only teacher-forced执行器超参数筛选
实现commit：27d69ce
```

## 1. 报告摘要

本实验固定策略查询间隔为 `Q=1`：每个 30 Hz 观测时刻都重新预测一个长度为
100 的 action chunk；唯一改变的执行变量是，当多个历史 chunk 都对当前时刻
给出动作预测时，聚合器对新预测和旧预测的权重偏好。

实验在 paired-50 manifest 的 10 条 validation episodes、3047 个时刻上进行。
Official ACT 与高频力 Contact-CVAE 在每个时刻各自只前向一次并缓存预测，随后对
同一份缓存结果重放 21 个 signed temporal 参数和 2 个精确端点。实验没有读取
剩余 50 条 holdout episodes，Contact-CVAE 部署时严格使用 `z_contact=0`。

主要发现如下：

1. Official ACT 的离线动作误差在偏旧区域最低，最优点为 `k=-0.05`，但相对
   官方等价基线 `k=-0.01` 只改善 `0.81%`，属于浅而宽的最优区。
2. Contact-CVAE 明确偏好新预测；`k=0.2~0.5`形成低误差平台，全局最优为
   `k=0.3`，相对 `k=-0.01` 改善 `44.93%`。
3. Contact-CVAE 的 `k=0.3`只比 `latest_only`低 `1.25%`动作误差，但命令
   逐步变化量降低约 `24.57%`，表明“最近几帧软融合”可能优于完全不聚合。
4. Contact-CVAE 在自由空间的最优点为 `k=0.5`，在 `5~20 N`与`>=20 N`
   接触阶段分别为 `k=0.05`与`k=0.1`。全局最优受占比 `81.06%`的自由空间
   样本主导，不能直接解释为接触阶段最优。
5. 本实验是 teacher-forced 离线筛选，不包含策略动作引起的状态转移，不能据此
   宣称某个配置具有更高闭环成功率。

---

## 2. 实验问题与设计初衷

### 2.1 要回答的问题

ACT 类策略每次输出未来 100 步动作，而每个当前时刻可能同时收到多份预测：

```text
较早 chunk 对当前时刻的预测
稍早 chunk 对当前时刻的预测
当前最新 chunk 对当前时刻的预测
```

动作执行结果不仅取决于模型本身，也取决于如何组合这些相互重叠的预测。因此，
固定点位 rollout 失败时，至少存在两种可能：

1. 模型没有预测出有效动作；
2. 模型预测包含有效信息，但新旧 chunk 的执行和融合方式不合适。

本实验希望在进入昂贵且有安全风险的闭环 rollout 前，先隔离第二个因素，回答：

- 每帧重新规划时，模型更适合依赖新预测还是旧预测？
- 对新旧预测的偏好应当多强？
- 完全只采用最新预测是否优于软融合？
- Official ACT 与带高频力输入的 Contact-CVAE 是否呈现相同偏好？
- 自由空间和不同接触力阶段是否需要不同程度的时间聚合？

### 2.2 为什么固定 Q=1

这里的 `Q=1`指策略查询间隔为一个观测帧，不是 action chunk 长度变成 1：

```text
action chunk length = 100
policy query interval Q = 1
```

因此每个时刻都会生成新的 100 步预测。固定 `Q=1`以后，“多久重新规划一次”不再
变化，实验只改变 temporal aggregation 的新旧权重。这样可以避免把查询频率、
chunk 顺序执行和 temporal 融合三种影响混在一起。

### 2.3 为什么先做离线扫描

两个模型在每个 validation 时刻各前向一次，输出被缓存到内存。23 种执行器随后
重放完全相同的缓存 action chunks。因此执行器数量增加不会增加模型前向次数，且
不同参数之间不存在模型采样、DataLoader 顺序或输入差异造成的混杂。

这个设计适合低成本筛除明显不合适的权重区域，但它只负责候选筛选，不能替代
闭环任务验证。

---

## 3. Signed Temporal Aggregation 定义

模型在查询时刻 `tau`预测：

```text
A_tau = [a(tau|tau), a(tau+1|tau), ..., a(tau+99|tau)]
```

在当前时刻 `t`，所有仍覆盖 `t`的历史 chunk 提供对齐候选：

```text
a_hat(t|tau) = A_tau[t - tau]
prediction_age = t - tau
```

本实验统一使用：

```text
log_weight(tau) = -k * prediction_age
weight(tau) = softmax(log_weight(tau))
a_exec(t) = sum_tau weight(tau) * a_hat(t|tau)
```

参数含义为：

| signed `k` | 权重含义 |
|---:|---|
| `< 0` | 偏向更旧的预测 |
| `= 0` | 所有有效预测等权 |
| `> 0` | 偏向更新的预测 |

权重在指数运算前减去最大 log weight，以保证 `k=±1`等极端配置下仍保持数值
稳定。

### 3.1 与官方 ACT 参数的对应关系

官方 ACT 候选按“最旧到最新”排列，并使用：

```text
weight_official(i) proportional to exp(-0.01 * candidate_index_i)
```

若当前共有 `n`个候选，则：

```text
prediction_age = n - 1 - candidate_index
```

归一化后，官方 `candidate-index k=0.01`与本实验的`signed-age k=-0.01`
严格等价。因此本报告以 `k=-0.01`作为官方参考基线。

### 3.2 精确端点

除 21 个有限 `k`外，实验还包含：

| 端点 | 定义 |
|---|---|
| `latest_only` | 只使用当前时刻刚生成的预测，等价于 Q=1 且不聚合 |
| `oldest_only` | 只使用仍覆盖当前时刻的最旧预测，仅作离线诊断 |

`oldest_only`可能采用接近 100 帧前生成的动作，不应未经安全审计直接进入闭环。

---

## 4. 实验对象与数据契约

### 4.1 数据划分

实验 manifest：

```text
configs/experiments/peg_hole_paired50_seed0.json
```

数据集共 100 条 episode：

| 划分 | episode数 | 用途 |
|---|---:|---|
| selected train | 40 | 两模型训练 |
| selected validation | 10 | 本次执行参数筛选 |
| holdout | 50 | 保留评估，本实验未读取 |

数据集指纹：

```text
e6fc49b8521e2e52c59ca0d09892da0c4ae153ebf6a0ad4fde086cd7bc7bb07c
```

本次 validation 共 10 条 episode、3047 个策略时刻。

### 4.2 固定模型

Official ACT：

```text
runs/paired50_official_act_formal_e2000_b8_seed0/best_policy.pt
```

高频力 Contact-CVAE：

```text
runs/paired50_highrate_contact_v3_formal_s10000_b8_seed0/best.pt
```

两者共同使用训练契约规定的图像和关节状态输入。Official ACT 不使用力信息；
Contact-CVAE 额外使用因果高频力区间，部署 latent 严格固定为：

```text
contact_latent_mode = zero
z_contact = 0
```

两个 checkpoint 在扫描前检查 action chunk 长度、相机数量、图像尺寸、qpos维度、
动作维度和 ImageNet normalization 契约一致。

### 4.3 时间与动作契约

```text
策略/图像/关节时间基准：30 Hz
高频力原生采样：约500 Hz
action chunk length：100
policy query interval：1个策略帧
```

模型输出在各自 normalization stats 下反归一化到物理动作空间后再执行重放和计算
误差。每条 episode 开始时都会清空 temporal executor，禁止跨 episode 使用历史
预测。

---

## 5. 实验变量与扫描范围

唯一主要自变量为 signed `k`：

```text
-1, -0.5, -0.3, -0.2, -0.1, -0.05, -0.03, -0.02, -0.01, -0.005,
 0,
 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1
```

另加 `latest_only`和`oldest_only`，合计 23 种配置。每种配置对 Official ACT
和 Contact-CVAE 都执行，得到 46 条 aggregate 结果和 460 条 per-episode 结果。

所有配置满足：

```text
policy_query_fraction = 1.0
```

即每个 validation 时刻都进行了模型查询，不存在因查询频率不同带来的比较偏差。

---

## 6. 评估过程

对 validation 的每个真实时刻执行：

1. 读取专家轨迹中的真实图像、qpos和当前动作标签；
2. 为 Contact-CVAE构造相同时间基准下的因果高频力输入；
3. Official ACT预测一个 100 步 action chunk；
4. Contact-CVAE以`z_contact=0`预测一个 100 步 action chunk；
5. 将两模型预测反归一化并按 episode、时间顺序缓存；
6. 对每个执行器配置，从缓存 chunk 中提取对齐到当前时刻的动作候选；
7. 按 signed `k`聚合，或使用精确端点选择；
8. 与专家在当前时刻的物理动作标签比较；
9. 在每条 episode 边界重置执行器，再汇总全局和分阶段指标。

这个过程是 teacher-forced：时刻 `t+1`的输入仍来自专家数据，而不是执行
`a_exec(t)`后形成的新状态。

---

## 7. 指标定义

### 7.1 动作模仿误差

`action_l1_physical_global`为反归一化动作在全部时刻和动作维度上的平均绝对误差：

```text
mean(abs(a_exec - a_expert))
```

这是本次排序主指标。它衡量执行器生成的当前命令与示教命令的接近程度，但不是
任务成功率。

### 7.2 命令变化量

`command_step_delta_l2_mean`为相邻执行命令之间 L2 距离的均值：

```text
mean(norm(a_exec[t] - a_exec[t-1], 2))
```

它是动作平滑性的代理指标；数值较低表示相邻命令变化较小，但过低也可能意味着
响应迟缓。

### 7.3 加权预测年龄

`prediction_age_mean_steps`表示形成当前命令的预测，其加权平均生成年龄。按30 Hz
估算：

```text
prediction_age_seconds = prediction_age_steps / 30
```

它度量计划陈旧程度，而不是传感器硬件时间戳延迟。

### 7.4 力阶段指标

当前时刻力范数取因果在线高频力区间中最后一个有效原生样本的前三维线性力范数：

| 阶段 | 阈值 | 时刻数 | 占比 |
|---|---:|---:|---:|
| 自由/低力 | `<5 N` | 2470 | 81.06% |
| 中等接触 | `5~20 N` | 461 | 15.13% |
| 高接触 | `>=20 N` | 116 | 3.81% |

该划分使用示教轨迹的观测力，只用于诊断不同接触阶段的动作误差。

---

## 8. 完整全局结果

下表按 signed `k`从偏旧到偏新排列。年龄对两个模型相同，因为它由执行器、
chunk长度和episode长度决定。

| 配置 | 平均年龄/帧 | Official L1 | Contact L1 |
|---|---:|---:|---:|
| `k=-1.0` | 82.176 | 0.048465 | 0.016972 |
| `k=-0.5` | 81.231 | 0.048409 | 0.016783 |
| `k=-0.3` | 79.951 | 0.048274 | 0.016507 |
| `k=-0.2` | 78.365 | 0.048103 | 0.016132 |
| `k=-0.1` | 73.773 | 0.047721 | 0.015002 |
| `k=-0.05` | 65.781 | **0.047462** | 0.013214 |
| `k=-0.03` | 58.648 | 0.047504 | 0.011825 |
| `k=-0.02` | 53.671 | 0.047627 | 0.010933 |
| `k=-0.01` | 47.800 | 0.047849 | 0.009950 |
| `k=-0.005` | 44.626 | 0.048001 | 0.009445 |
| `k=0` | 41.377 | 0.048176 | 0.008954 |
| `k=0.005` | 38.129 | 0.048372 | 0.008488 |
| `k=0.01` | 34.955 | 0.048582 | 0.008056 |
| `k=0.02` | 29.083 | 0.049029 | 0.007327 |
| `k=0.03` | 24.106 | 0.049448 | 0.006772 |
| `k=0.05` | 16.974 | 0.050101 | 0.006103 |
| `k=0.1` | 8.982 | 0.050882 | 0.005635 |
| `k=0.2` | 4.390 | 0.051357 | 0.005498 |
| `k=0.3` | 2.804 | 0.051545 | **0.005479** |
| `k=0.5` | 1.523 | 0.051737 | 0.005486 |
| `k=1.0` | 0.578 | 0.051952 | 0.005516 |
| `latest_only` | 0.000 | 0.052164 | 0.005549 |
| `oldest_only` | 82.755 | 0.048364 | 0.017099 |

---

## 9. Official ACT 结果分析

### 9.1 最优区域

Official ACT 全局最低L1出现在：

```text
k = -0.05
action_l1_physical_global = 0.047461643
prediction_age_mean_steps = 65.780679
```

相对官方等价基线 `k=-0.01`：

```text
baseline L1 = 0.047849459
relative improvement = 0.81%
```

跨 episode 比较：

- `k=-0.05`在 10 条中的 8 条优于`k=-0.01`；
- `k=-0.05`只在 10 条中的 6 条优于`k=-0.03`；
- `k=-0.05`与`k=-0.03`的全局绝对差只有约`4.2e-5`。

因此不能把`-0.05`解释为精确且稳定的唯一最优超参数。更合理的结论是：

```text
Official ACT在本验证集上存在约k=-0.02~-0.10的浅偏旧低误差区域。
```

### 9.2 误差与平滑性并不一致

| 配置 | Official L1 | 命令变化量 | 平均年龄/帧 |
|---|---:|---:|---:|
| `k=-0.05` | **0.047462** | 0.000961 | 65.781 |
| `k=-0.01` | 0.047849 | 0.000852 | 47.800 |
| `k=0` | 0.048176 | 0.000824 | 41.377 |
| `k=0.05` | 0.050101 | **0.000756** | 16.974 |
| `latest_only` | 0.052164 | 0.002809 | 0.000 |

偏旧降低了离线模仿误差，却没有降低本实验定义的命令逐步变化量；`k=-0.05`
反而比官方基线变化更大。不能用“偏旧一定更平滑”解释这个结果。它更可能反映：

- 较早 chunk 对专家轨迹的长期计划在该数据上更一致；
- 较新预测在接触阶段可能发生系统性偏移；
- teacher-forced误差可能奖励具有时间滞后的轨迹匹配。

这些解释都需要闭环轨迹和模型输出时序进一步验证。

### 9.3 分阶段偏好

Official ACT 分阶段最低误差为：

| 阶段 | 最低误差配置 | L1 |
|---|---|---:|
| `<5 N` | `k=-0.01` | 0.048636 |
| `5~20 N` | `oldest_only` | 0.041104 |
| `>=20 N` | `oldest_only` | 0.027659 |

接触阶段对最旧预测的偏好不能直接转化为部署建议。Official ACT没有力输入，且
`oldest_only`平均使用约82.75帧前形成的计划；离线误差较低可能是示教轨迹
平滑、接触动作速度降低或teacher-forced时间对齐共同造成的，而闭环中如此陈旧
的命令可能无法响应新的接触偏差。

---

## 10. Contact-CVAE 结果分析

### 10.1 从偏旧到偏新的清晰趋势

Contact-CVAE 全局最低L1出现在：

```text
k = 0.3
action_l1_physical_global = 0.005479225
prediction_age_mean_steps = 2.803645
```

相对官方等价基线`k=-0.01`：

```text
baseline L1 = 0.009949959
relative improvement = 44.93%
```

`k=0.3`在全部10条episode中都优于`k=-0.01`，说明Contact-CVAE不适合直接
继承官方偏旧的时间权重。随着`k`从负值增加到正值，误差整体连续下降，并在：

```text
k = 0.2~1.0
```

形成低误差平台。

### 10.2 平台内部不存在强唯一最优点

| 配置 | Contact L1 | 相对`k=-0.01`改善 | 命令变化量 |
|---|---:|---:|---:|
| `k=0.1` | 0.005635 | 43.36% | **0.003620** |
| `k=0.2` | 0.005498 | 44.74% | 0.003724 |
| `k=0.3` | **0.005479** | 44.93% | 0.003805 |
| `k=0.5` | 0.005486 | 44.87% | 0.003950 |
| `k=1.0` | 0.005516 | 44.56% | 0.004273 |
| `latest_only` | 0.005549 | 44.24% | 0.005044 |

跨 episode 结果进一步说明平台内差异较弱：

- `k=0.3`只在 6/10 条中优于`k=0.2`；
- `k=0.3`只在 4/10 条中优于`k=0.5`；
- `k=0.3`只在 6/10 条中优于`latest_only`；
- 10条episode的单独最优配置分布在多个正`k`和`latest_only`上。

因此，`k=0.3`是全局点估计最优，而不是已经得到统计证明的唯一最优。

### 10.3 最近几帧软融合具有实践价值

相对`latest_only`，`k=0.3`：

```text
动作L1降低约1.25%
命令逐步变化量降低约24.57%
加权平均预测年龄约2.80帧，即约0.093秒
```

这说明完全抛弃历史预测不是必要条件。保留最近几帧并给予快速指数衰减，可能在
基本保持新鲜力反馈的同时降低命令抖动。

`k=0.1`则提供另一个折中：其动作误差比`k=0.3`高约2.85%，但命令变化量低
约4.85%，且加权年龄约8.98帧。因此`k=0.1`和`k=0.3`都值得进入闭环。

### 10.4 力阶段差异

Contact-CVAE分阶段最低误差为：

| 阶段 | 最低误差配置 | L1 | 对应平均年龄/帧 |
|---|---|---:|---:|
| `<5 N` | `k=0.5` | 0.005792 | 1.523 |
| `5~20 N` | `k=0.05` | 0.003533 | 16.974 |
| `>=20 N` | `k=0.1` | 0.004022 | 8.982 |

由此得到一个有价值但尚未证实的启发：

```text
自由空间可以更强地依赖最新预测；进入接触后，适度融合历史计划可能更稳定。
```

但全局样本中自由空间占`81.06%`，高力阶段仅有116步。当前证据不足以立即实现
按力阶段自适应切换`k`；这应当作为后续独立消融，而不是本轮闭环比较中的新增
变量。

---

## 11. 两模型差异带来的启发

本实验呈现出方向相反的离线偏好：

```text
Official ACT：偏旧区域更低误差，点估计k=-0.05
Contact-CVAE：明显偏新，低误差平台k=0.2~1.0
```

一种合理假设是：Contact-CVAE的新预测包含最新高频力窗口编码，过度采用旧chunk
会稀释新的接触信息；Official ACT没有力输入，较早形成的视觉—关节运动计划可能
在示教轨迹上具有更好的长期一致性。

但这不是因果证明。两模型还存在训练目标、模型结构和checkpoint质量等差异。要
证明差异来自高频力反馈，需要额外的受控模型消融，而不能只比较这两个checkpoint。

另一个重要启发是：统一后处理不等于统一最优超参数。若研究问题是公平比较模型，
必须让两模型在相同`k`下配对测试；若研究问题是每种模型的最佳可部署性能，则可在
validation上分别选`k`，但最终必须使用未参与选择的测试条件评估，并明确这是
“各自调优后”的比较。

---

## 12. 实验局限

### 12.1 不是闭环 rollout

这是最重要的限制。模型动作没有驱动MuJoCo进入下一状态，因此没有测试：

- 误差累积和轨迹偏离；
- 接触状态因动作变化产生的反馈；
- 力反馈能否纠正新的闭环扰动；
- 插孔成功率、完成时间和hard stop；
- action postprocess、安全裁剪和控制器动力学。

### 12.2 动作L1不是任务成功率

离专家动作更近通常是有价值的诊断，但插孔任务对接触几何、动作方向和少量关键
时刻可能高度敏感。全局平均L1会被大量自由空间时刻主导。

### 12.3 validation用于超参数选择

本实验有意使用validation选择`k`。剩余50条holdout未被读取，因此未来仍可用于
独立离线评估。但一旦根据本报告选择了`k`，就不能再把同一validation结果报告为
无偏测试性能。

### 12.4 样本和训练随机性有限

当前只有10条validation episodes、一个manifest split和每个模型的一个训练
checkpoint。尤其`>=20 N`只有116个时刻，分阶段最优可能对episode组成敏感。

### 12.5 扫描耗时不能代表闭环实时性

本次CUDA扫描总耗时约99.48秒，其中包含数据读取、两个模型前向、缓存和23种
离线重放。该吞吐不能直接等价为MuJoCo闭环中的稳定30 Hz推理能力；闭环必须单独
记录模型推理耗时、总循环耗时和deadline miss。

---

## 13. 结论与下一阶段建议

### 13.1 可以从本实验得出的结论

1. 固定每帧查询后，temporal新旧权重对两个模型的离线动作误差有实质影响；
2. Official ACT官方等价偏旧权重处于合理区域，但`k=-0.05`相对基线提升很小；
3. Contact-CVAE不应沿用官方偏旧权重，`k=0.1~0.5`是更合理的闭环候选；
4. Contact的`k=0.3`兼顾最低全局误差和明显低于`latest_only`的命令变化；
5. 接触阶段可能需要比自由空间更强的历史融合，但当前只应作为后续假设。

### 13.2 不能从本实验得出的结论

1. 不能宣布Contact-CVAE闭环性能优于Official ACT；
2. 不能宣布`k=0.3`具有最高插孔成功率；
3. 不能把Official ACT对旧预测的离线偏好解释为闭环中越旧越安全；
4. 不能基于本validation再次选择参数后，将其作为独立测试结果。

### 13.3 推荐闭环筛选配置

为了覆盖旧、官方基线、等权、新预测平台和无聚合端点，同时避免测试大量冗余
参数，建议对两个模型使用完全相同的七组配置：

```text
k=-0.05
k=-0.01
k= 0.00
k=+0.05
k=+0.10
k=+0.30
latest_only
```

按30 Hz估计，它们对应的加权平均计划年龄为：

| 配置 | 平均年龄/帧 | 约合时间/秒 |
|---|---:|---:|
| `k=-0.05` | 65.781 | 2.193 |
| `k=-0.01` | 47.800 | 1.593 |
| `k=0` | 41.377 | 1.379 |
| `k=0.05` | 16.974 | 0.566 |
| `k=0.1` | 8.982 | 0.299 |
| `k=0.3` | 2.804 | 0.093 |
| `latest_only` | 0.000 | 0.000 |

第一轮应在相同固定点位各执行一次安全screening，统一使用40 N safe-success阈值、
100 N hard stop和相同控制后处理，记录：

- task success与safe success；
- 最大/平均接触力和hard stop；
- 完成时间、最终距离和最小距离；
- 命令变化量和安全裁剪次数；
- 实际推理频率、循环耗时和deadline miss；
- 每步聚合候选数与加权预测年龄。

筛除明显失败或不安全配置后，再选择少量候选进行每组至少5次、正式比较建议10次
的重复闭环实验。

---

## 14. 复现信息

执行器与扫描实现commit：

```text
27d69ce expand signed temporal action sweep
```

执行命令：

```bash
PYTHONPATH=src python scripts/sweep_paired50_validation_action_execution.py \
  mujoco_data/peg_hole_100 \
  --experiment-manifest configs/experiments/peg_hole_paired50_seed0.json \
  --official-checkpoint runs/paired50_official_act_formal_e2000_b8_seed0/best_policy.pt \
  --contact-checkpoint runs/paired50_highrate_contact_v3_formal_s10000_b8_seed0/best.pt \
  --output-dir runs/paired50_validation_signed_temporal_sweep_b1 \
  --sweep-profile expanded_signed_temporal \
  --device cuda \
  --batch-size 1 \
  --num-workers 2 \
  --log-interval 100
```

原始输出：

```text
runs/paired50_validation_signed_temporal_sweep_b1/summary.json
runs/paired50_validation_signed_temporal_sweep_b1/aggregate.csv
runs/paired50_validation_signed_temporal_sweep_b1/per_episode.csv
```

运行审计：

```text
sweep_version = paired50_validation_action_executor_sweep_v2
sweep_profile = expanded_signed_temporal
passed = true
partial_evaluation = false
teacher_forced = true
closed_loop_rollout = false
holdout_used = false
fixed_policy_query_interval = 1
validation_episode_count = 10
validation_timestep_count = 3047
model_forward_samples_per_model = 3047
elapsed_seconds = 99.4771823529154
```
