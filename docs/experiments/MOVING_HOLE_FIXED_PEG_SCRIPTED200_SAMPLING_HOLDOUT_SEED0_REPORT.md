# Scripted200：四模型 seed-0 正式 holdout 离线评估

评估日期：2026-09-06。执行代码：`ba9d4f5`，其中评估器修复提交为
`f218a1b`。本次完成第一次正式 holdout 性能评估，未根据结果修改模型、
checkpoint、控制参数或成功阈值。

## 1. 评估范围与验收

- 数据：moving-hole / fixed-peg scripted200，训练/验证/holdout 为 80/20/100 条。
- 本次完整读取 holdout 的 100 条 episode：clean 50、collision 50。
- 全部 63,175 个 timestep，40,757,500 个有效 action-chunk 标量。
- 四模型使用各自验证集选出的 best checkpoint，同一 episode/timestep 顺序。
- 协议：`scripted200_sampling_ablation_all_timesteps_v2`。
- 部署 latent：zero；batch=16，workers=0，CUDA；temporal decay=0.01。
- 主要指标：有效 action chunk 的物理单位 L1，排除 padding。
- 配对推断单位：episode；bootstrap 10,000 次，基础 seed=0。
- 用时：6,982.72 秒（约 116.4 分钟）；进程退出码 0。

四个 checkpoint 的 SHA-256 与修正后的完整 validation 完全一致；数据指纹为
`0f407682f52aa431a4ccc3e9bcf26deda935c0c3440b8a9d81f03bc5630bb5dc`。
100 条输出 episode 的身份和长度逐项匹配 manifest，无遗漏或重复，全部全局
指标有限。由 per-episode CSV 重新按有效标量数加权，复现了四项全局主要指标；
也复现了整体和两类分层的全部 18 组配对差异、胜负计数及 bootstrap 区间。

本次评估调用冻结的推理代码，没有产生新的实现修改。之前该代码版本的全仓
测试为 `746 passed`；本次验收针对实际 holdout 产物，没有重复训练或全仓测试。

## 2. 整体结果

L1 越低越好。全局 action-chunk L1 按所有有效动作标量加权；temporal L1 是
teacher-forced 轨迹上 action temporal aggregation 的次要指标。

| 模型 / 训练采样 | Action chunk L1 | Temporal L1 | 对应 validation chunk L1 |
| --- | ---: | ---: | ---: |
| Official ACT / episodic | 0.0023445795 | 0.0019269380 | 0.0023619782 |
| Official ACT / dense | 0.0026229468 | 0.0022690634 | 0.0025896796 |
| Contact-CVAE / dense | 0.0026551019 | 0.0025531458 | 0.0026402107 |
| Contact-CVAE / episodic | 0.0027261960 | 0.0026136016 | 0.0026680049 |

Official ACT episodic 的整体 chunk L1 比 Official dense 低 10.61%；
Contact dense 比 Contact episodic 低 2.61%。四组 holdout 相对 validation 的
主要指标变化分别为 -0.74%、+1.28%、+0.56%、+2.18%。

这些是同分布示教轨迹上的离线预测结果；模型接收真实示教状态，并没有执行
预测动作，不能由这些 L1 推导闭环成功率或接触安全性。

### 整体配对差异

以下差值均为“左侧模型误差减去右侧模型误差”，先在每条 episode 内计算，
再对 episode 等权平均。因此与上表全局加权指标直接相减会略有差别。

| 差值 | episode 等权平均 | bootstrap 95% CI |
| --- | ---: | --- |
| Official dense − Official episodic | +0.00027536 | [+0.00021182, +0.00034486] |
| Contact episodic − Contact dense | +0.00007475 | [+0.00001149, +0.00013369] |
| Contact dense − Official dense | +0.00004279 | [-0.00008159, +0.00015871] |
| Contact dense − Official episodic | +0.00031815 | [+0.00021629, +0.00041711] |
| Contact episodic − Official dense | +0.00011755 | [-0.00004223, +0.00026705] |
| Contact episodic − Official episodic | +0.00039291 | [+0.00025803, +0.00051882] |

Official episodic 在 91/100 条 episode 上优于 Official dense。
Contact dense 在 69/100 条上优于 Contact episodic。两项架构内比较的区间
均不跨零，但 Contact 的优势幅度较小。

模型与采样的交互定义为
`(Contact dense − Official dense) − (Contact episodic − Official episodic)`。
均值为 -0.00035012，95% CI 为 [-0.00042104, -0.00028017]。在这四个固定
checkpoint 上，改变采样方式对两种架构的影响方向不同：Official 受益于
episodic，Contact 整体上略受益于 dense。

所有区间均为预设的逐比较 percentile bootstrap 区间，未做多重比较校正。
它们描述这批 episode 的不确定性，不包含不同训练 seed 的方差。

## 3. Clean / collision 分层

下表为 episode 等权的 action-chunk L1，各类 50 条 episode。

| 模型 / 训练采样 | Clean | Collision |
| --- | ---: | ---: |
| Official ACT / episodic | 0.00089619 | 0.00372412 |
| Official ACT / dense | 0.00104037 | 0.00413066 |
| Contact-CVAE / dense | 0.00151843 | 0.00373819 |
| Contact-CVAE / episodic | 0.00175907 | 0.00364706 |

Clean 中两个 Official 模型都优于两个 Contact 模型；Contact dense 在全部
50 条 clean episode 上优于 Contact episodic。Collision 中，Contact
episodic 的平均值最低，但不能把这个数值排名直接解释为稳定优于
Official episodic：

| Collision 配对差值 | 平均值 | bootstrap 95% CI |
| --- | ---: | --- |
| Contact dense − Official dense | -0.00039247 | [-0.00056115, -0.00022468] |
| Contact episodic − Official dense | -0.00048361 | [-0.00069586, -0.00027185] |
| Contact dense − Official episodic | +0.00001407 | [-0.00014344, +0.00017364] |
| Contact episodic − Official episodic | -0.00007706 | [-0.00027431, +0.00010456] |
| Contact episodic − Contact dense | -0.00009114 | [-0.00019345, +0.00000802] |

两组 Contact 相对 Official dense 的 collision 优势得到逐比较区间支持；
相对 Official episodic 的区间均跨零。Contact 的 dense 整体优势主要来自
clean 分层；collision 中 episodic 的平均误差更低，但这项差异区间跨零。

## 4. 预测 horizon 与力指标

| 模型 | h1 | h10 | h25 | h50 | h100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official episodic | 0.00098703 | 0.00118066 | 0.00177666 | 0.00255458 | 0.00348049 |
| Official dense | 0.00146412 | 0.00149420 | 0.00202765 | 0.00281406 | 0.00377041 |
| Contact dense | 0.00171391 | 0.00178648 | 0.00200198 | 0.00270795 | 0.00362936 |
| Contact episodic | 0.00207164 | 0.00200358 | 0.00223845 | 0.00279239 | 0.00333735 |

| Contact 模型 | Endpoint wrench component L1 | Native-rate wrench component L1 |
| --- | ---: | ---: |
| Dense | 0.01834701 | 0.02631251 |
| Episodic | 0.01845100 | 0.02646418 |

Wrench 指标混合了力与力矩分量的原始单位，不能标注为纯 N，也不是闭环
峰值力或安全指标。两种采样的力预测误差数值接近，本评估没有给这些力指标
计算 episode 配对置信区间。

## 5. 结论边界与后续工作

本次 seed-0 的主要发现是：Official ACT episodic 整体动作预测最好；
Contact dense 比 Contact episodic 有小幅整体优势；Contact 对 Official
dense 的 collision 优势没有推广为对 Official episodic 的明确优势。
不能据此宣称高频力 Contact-CVAE 全面优于 ACT。

训练遵循共同的早停规则，各自 best checkpoint 步数不同（97,960、75,816、
72,657、44,240）。这里比较的是该早停协议下的训练结果，不是固定相同步数
截取的模型。当前只有一个训练 seed；采样交互需要完整 2×2 多 seed 实验
进一步检验，不能只给表现较好的一个分支增加 seed。

100 条 holdout 现已完成正式性能评估。后续不应据此重新调参后再把同一批
holdout 称为未使用的最终测试集。若下一阶段重点是接触恢复能力，应先在
validation 或独立仿真条件上制定固定目标下的可复现接触扰动协议，并保证
四模型采用相同接管状态与评估规则。已有 ±4 mm 固定目标平移测试属于另一种
目标位置分布偏移，详见 rollout 文档。

## 6. 产物与执行命令

产物位于
`outputs/moving_hole_fixed_peg_scripted200/sampling_ablation_b16_earlystop/evaluation/`：

- [完整 summary](../../outputs/moving_hole_fixed_peg_scripted200/sampling_ablation_b16_earlystop/evaluation/holdout_full_v2/summary.json)
- [逐 episode CSV](../../outputs/moving_hole_fixed_peg_scripted200/sampling_ablation_b16_earlystop/evaluation/holdout_full_v2/per_episode.csv)
- [执行日志](../../outputs/moving_hole_fixed_peg_scripted200/sampling_ablation_b16_earlystop/evaluation/holdout_full_v2.execution.log)

以下记录本次命令。输出目录已经存在，评估器会拒绝覆盖；无需再次执行。

```bash
CHECKPOINT_ROOT=outputs/moving_hole_fixed_peg_scripted200/sampling_ablation_b16_earlystop
PYTHONPATH=src /home/stw/miniconda3/envs/forceact/bin/python -u scripts/evaluate_sampling_ablation.py \
  /home/stw/ForceAwareACT_workspace/arm_teleop/data/moving_hole_fixed_peg \
  --experiment-manifest configs/experiments/moving_hole_fixed_peg_scripted200_seed0.json \
  --official-episodic-checkpoint "$CHECKPOINT_ROOT/01_official_act_episodic_seed0/best_policy.pt" \
  --official-dense-checkpoint "$CHECKPOINT_ROOT/02_official_act_dense_seed0/best_policy.pt" \
  --contact-dense-checkpoint "$CHECKPOINT_ROOT/03_contact_cvae_dense_seed0/best.pt" \
  --contact-episodic-checkpoint "$CHECKPOINT_ROOT/04_contact_cvae_episodic_seed0/best.pt" \
  --split holdout --confirm-holdout-evaluation \
  --output-dir "$CHECKPOINT_ROOT/evaluation/holdout_full_v2" \
  --device cuda --batch-size 16 --num-workers 0 \
  --bootstrap-resamples 10000 --bootstrap-seed 0 \
  --temporal-decay 0.01 --log-interval 25
```
