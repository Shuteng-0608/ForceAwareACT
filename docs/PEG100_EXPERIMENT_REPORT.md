# Peg-in-Hole 100-Episode Experiment Report

## 1. 实验目的

本轮实验的目标是使用新采集的 100 条 peg-in-hole 示教数据，对 ForceAwareACT 的完整训练与验证流程进行一次较正式的小规模实验闭环验证。

重点验证以下问题：

1. 100 条 HDF5 episode 是否能够稳定读取、对齐和训练；
2. `z_contact` posterior latent 是否能够在 Stage 1 中被有效使用；
3. conditional contact prior 是否能够在 Stage 2 中学习 posterior contact latent；
4. 在可部署推理模式下，`prior inference` 是否优于 `zero inference`；
5. 在 held-out validation/test episodes 上，force prediction 是否获得稳定提升。

本轮实验采用 80/10/10 episode split：

```text
train: 80 episodes
val:   10 episodes
test:  10 episodes
```

其中 normalization statistics 只使用 train80 计算，val10/test10 只用于评估。

---

## 2. 数据集概况

数据目录：

```text
/Users/wangshuteng/Desktop/ForceAwareACT/mujoco_data/peg_in_hole_hdf5_100
```

每个 episode 目录包含：

```text
episode.hdf5
metadata.json
```

HDF5 episode 主要字段包括：

```text
observations/ee_pose
observations/joint_pos
observations/joint_vel
observations/joint_torque
observations/ft_wrench
observations/images/ee_cam
observations/images/base_top_cam
timestamps/state_episode
timestamps/force_episode
timestamps/image_episode
episode_metadata/
```

数据集检查结果：

```text
Episodes: 100
Strict valid episodes: 78
Tolerant valid episodes: 100
Truly invalid episodes: 0
Episodes requiring trimming: 22

State frames: 42,741
Force frames: 711,175
Duration min/mean/max: 8.529 / 14.211 / 22.697 seconds
Dataset length min/mean/max: 246 / 416.41 / 671
```

其中 22 条 episode 存在 one-frame mismatch：

```text
Image group mismatch: 19 episodes
Force group mismatch: 2 episodes
State group mismatch: 1 episode
```

所有 mismatch 都只是 1 frame，判断为录制停止阶段的异步写入 artifact，不属于数据损坏。因此更新了 dataset reader，默认允许 one-frame mismatch，并按 safe length 进行 trim。

相关测试结果：

```text
77 passed
```

---

## 3. 数据读取与路径问题

最初使用 project-root-relative split 文件时，`compute_normalization_stats.py` 将：

```text
mujoco_data/peg_in_hole_hdf5_100/xxx/episode.hdf5
```

错误解析为：

```text
configs/splits/mujoco_data/peg_in_hole_hdf5_100/xxx/episode.hdf5
```

导致 file not found。

临时解决方式是生成 absolute path split：

```text
/Users/wangshuteng/Desktop/ForceAwareACT/mujoco_data/peg_in_hole_hdf5_100/xxx/episode.hdf5
```

后续建议统一修改所有 episode-list 解析逻辑：

```text
1. absolute path: use directly
2. project-root-relative path: prefer this
3. episode-list-parent-relative path: fallback
```

注意：absolute path split 文件不应提交到 GitHub。

---

## 4. Train/Val/Test Split

生成了以下 split 文件：

```text
configs/splits/peg_in_hole_100_all.txt
configs/splits/peg_in_hole_100_train80.txt
configs/splits/peg_in_hole_100_val10.txt
configs/splits/peg_in_hole_100_test10.txt
```

当前实验实际使用的是 absolute path 版本。

split 规模检查：

```text
100 configs/splits/peg_in_hole_100_all.txt
 80 configs/splits/peg_in_hole_100_train80.txt
 10 configs/splits/peg_in_hole_100_val10.txt
 10 configs/splits/peg_in_hole_100_test10.txt
```

---

## 5. Normalization Statistics

使用 train80 计算 normalization statistics：

```bash
PYTHONPATH=src .venv/bin/python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/peg_in_hole_100_train80.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output outputs/peg100/normalization_stats_train80.pt
```

运行结果：

```text
resolved_episode_count=80
dataset_length=32374
saved_stats=outputs/peg100/normalization_stats_train80.pt
qpos_mean: shape=(7,)
qpos_std: shape=(7,)
action_mean: shape=(7,)
action_std: shape=(7,)
force_mean: shape=(6,)
force_std: shape=(6,)
```

说明 train80 的数据读取、trim、统计计算流程均正常。

---

## 6. Stage 1: Policy Training

### 6.1 Smoke Test

首先运行 100-step smoke test：

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py \
  --episode-list configs/splits/peg_in_hole_100_train80.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --normalization-stats outputs/peg100/normalization_stats_train80.pt \
  --max-steps 100 \
  --batch-size 4 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --output-dir outputs/peg100/stage1_smoke \
  --log-csv outputs/peg100/stage1_smoke/train_log.csv
```

结果：

```text
resolved_episode_count=80
dataset_length=32374
normalization_enabled=True
saved_checkpoint=outputs/peg100/stage1_smoke/checkpoint.pt
```

说明 DataLoader、normalization、model forward/backward、loss computation、checkpoint saving 全部正常。

### 6.2 Full Stage 1 Training

随后运行正式 Stage 1：

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py \
  --episode-list configs/splits/peg_in_hole_100_train80.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --normalization-stats outputs/peg100/normalization_stats_train80.pt \
  --max-steps 10000 \
  --batch-size 4 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --output-dir outputs/peg100/stage1 \
  --log-csv outputs/peg100/stage1/train_log.csv
```

训练日志分析结果：

```text
loss_total:  88.21% reduction
loss_action: 93.66% reduction
loss_force:  76.13% reduction
loss_prior:  22.11% reduction
```

### 6.3 Stage 1 训练现象

#### Action loss

`loss_action` 从约 0.7～0.9 快速下降到约 0.04～0.06，并保持稳定。

这说明：

```text
action reconstruction 收敛充分；
ACT-style transformer decoder 与 ActionHead 正常；
vision/joint/force/contact tokens 输入链路有效。
```

#### Force loss

`loss_force` 从约 0.8～1.1 降到约 0.12～0.18。虽然 raw force loss 仍有 spike，但这是 contact-rich manipulation 中正常现象，通常对应强接触、卡滞、接触突变或高力样本。

这说明：

```text
ForceHead 学到了未来力趋势；
force prediction 没有发散；
接触状态差异导致 batch-level force loss 波动较大。
```

#### Prior loss

`loss_prior` 下降幅度较小，主要稳定在约 0.15～0.30 区间。这说明 Stage 1 中 prior 分支已经学习到部分 posterior latent 映射，但并未充分贴合 posterior。

因此 Stage 2 prior-only distillation 是必要的。

#### KL terms

`kl_motion` 和 `kl_contact` 随训练明显上升：

```text
kl_motion: 约 5 -> 50+
kl_contact: 约 5 -> 35~40
```

这说明 posterior latent 正在承载较多信息。由于 Stage 1 中 KL 权重采用 warmup，早期 posterior 可以较自由地编码 action/contact 信息。只要 `loss_total`、`loss_action` 和 `loss_force` 没有发散，KL 上升并不代表训练失败，而是说明 latent 被有效使用。

---

## 7. Analyze Train Log Script Update

为了避免 raw losses 重叠导致难以观察趋势，更新了：

```text
scripts/analyze_train_log.py
```

新增功能：

```text
--output-dir
--window
--last-fraction
individual raw + moving-average plots
combined moving-average-only plots
graceful missing-column handling
backward-compatible --plot support
```

生成图包括：

```text
loss_total.png
loss_action.png
loss_force.png
loss_prior.png
kl_motion.png
kl_contact.png
loss_moving_average_combined.png
loss_moving_average_last_fraction.png
```

随后进一步增强，使 Stage 2 prior metrics 也能单独画图：

```text
prior_mu_mse.png
prior_mu_l2.png
prior_mu_cosine_similarity.png
```

---

## 8. Stage 2: Contact Prior Distillation

Stage 2 使用 Stage 1 checkpoint，冻结 posterior target，专门训练 conditional contact prior：

```bash
PYTHONPATH=src .venv/bin/python scripts/train_contact_prior_stage2.py \
  --episode-list configs/splits/peg_in_hole_100_train80.txt \
  --checkpoint outputs/peg100/stage1/checkpoint.pt \
  --normalization-stats outputs/peg100/normalization_stats_train80.pt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --max-steps 10000 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --prior-loss-mode mse_mu \
  --output-dir outputs/peg100/stage2 \
  --log-csv outputs/peg100/stage2/train_log.csv
```

结果：

```text
saved_checkpoint=outputs/peg100/stage2/checkpoint.pt
```

Stage 2 log summary：

```text
logged_steps=10000
metric_columns=[
  'loss_prior',
  'prior_mu_mse',
  'prior_mu_l2',
  'prior_mu_cosine_similarity'
]
prior_loss_mode_values=['mse_mu']
```

Stage 2 统计：

```text
loss_prior / prior_mu_mse:
  first100_mean = 0.186487
  last100_mean  = 0.166173
  reduction     = 10.8928%

prior_mu_l2:
  first100_mean = 1.04711
  last100_mean  = 0.937477
  reduction     = 10.4702%

prior_mu_cosine_similarity:
  first100_mean = 0.951147
  last100_mean  = 0.964691
  increase      = 1.424%
```

### Stage 2 结论

Stage 2 是有效的，但提升幅度不大。原因可能是 Stage 1 中 prior 已经学到了一部分 posterior latent 映射，Stage 2 主要是在做进一步细化。

最终 prior-posterior 对齐状态：

```text
MSE 下降；
L2 下降；
cosine similarity 上升；
cosine similarity 最后一百步均值约 0.965。
```

说明 conditional contact prior 没有 collapse，也没有发散，可以进入 deployable prior-mode evaluation。

---

## 9. Inference Modes

本轮实验比较三种 inference mode：

### zero mode

```text
z_contact = 0
```

这是可部署 baseline，表示没有使用 contact prior。

### prior mode

```text
z_contact = mu_contact_prior
```

这是可部署的 contact-prior inference 模式。它只使用当前视觉、关节状态和过去力窗口，不使用未来标签。

### posterior mode

```text
z_contact = posterior latent
```

这是 oracle/debug 模式。posterior encoder 使用 future action chunk 和 future force chunk，因此不可部署，只作为上界参考。

真正的可部署比较是：

```text
zero mode vs prior mode
```

---

## 10. Validation Evaluation: Val10

评估命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_inference_modes.py \
  --episode-list configs/splits/peg_in_hole_100_val10.txt \
  --checkpoint outputs/peg100/stage2/checkpoint.pt \
  --normalization-stats outputs/peg100/normalization_stats_train80.pt \
  --batch-size 8 \
  --max-batches 500 \
  --output-csv outputs/peg100/stage2/inference_eval_val10.csv
```

结果：

```text
dataset_length=4446
evaluated_batches=500
```

Aggregate metrics：

```text
action_l1_zero:      mean=0.176825 median=0.148249
action_l1_prior:     mean=0.153156 median=0.112856
action_l1_posterior: mean=0.0418503 median=0.0355974

force_l1_zero:       mean=0.595141 median=0.327291
force_l1_prior:      mean=0.275052 median=0.0827141
force_l1_posterior:  mean=0.150442 median=0.037647

mu_prior_to_mu_posterior_cosine:
  mean=0.945832
  median=0.983194
```

Improvement ratios：

```text
action_prior_improvement_vs_zero:
  mean=0.129723
  median=0.115684

force_prior_improvement_vs_zero:
  mean=0.618713
  median=0.722226
```

### Val10 结论

在 held-out val10 上，prior inference 相比 zero baseline：

```text
action error 平均降低约 13.0%
force error 平均降低约 61.9%
```

其中 force improvement 非常明显，说明 conditional contact prior 对未来力预测贡献显著。

---

## 11. Test Evaluation: Test10

评估命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_inference_modes.py \
  --episode-list configs/splits/peg_in_hole_100_test10.txt \
  --checkpoint outputs/peg100/stage2/checkpoint.pt \
  --normalization-stats outputs/peg100/normalization_stats_train80.pt \
  --batch-size 8 \
  --max-batches 500 \
  --output-csv outputs/peg100/stage2/inference_eval_test10.csv
```

结果：

```text
dataset_length=4821
evaluated_batches=500
```

Aggregate metrics：

```text
action_l1_zero:      mean=0.199471 median=0.17039
action_l1_prior:     mean=0.171127 median=0.143646
action_l1_posterior: mean=0.0445377 median=0.0408011

force_l1_zero:       mean=0.770793 median=0.341332
force_l1_prior:      mean=0.422972 median=0.0764673
force_l1_posterior:  mean=0.306942 median=0.0368801

mu_prior_to_mu_posterior_cosine:
  mean=0.934059
  median=0.987691
```

Improvement ratios：

```text
action_prior_improvement_vs_zero:
  mean=0.144817
  median=0.119579

force_prior_improvement_vs_zero:
  mean=0.578754
  median=0.733213
```

### Test10 结论

在 held-out test10 上，prior inference 相比 zero baseline：

```text
action error 平均降低约 14.5%
force error 平均降低约 57.9%
```

这个结果和 val10 的趋势高度一致，说明该效果不是 validation split 的偶然现象。

---

## 12. Val/Test 对比

| Split  | Action zero | Action prior | Action improvement | Force zero | Force prior | Force improvement |
| ------ | ----------: | -----------: | -----------------: | ---------: | ----------: | ----------------: |
| Val10  |      0.1768 |       0.1532 |              13.0% |     0.5951 |      0.2751 |             61.9% |
| Test10 |      0.1995 |       0.1711 |              14.5% |     0.7708 |      0.4230 |             57.9% |

主要观察：

```text
1. prior inference 在 val10 和 test10 上均稳定优于 zero inference；
2. force improvement 明显大于 action improvement；
3. posterior oracle 仍然明显更强，说明 posterior latent 包含大量未来信息；
4. prior-posterior cosine median 在 val/test 上均接近 0.98；
5. 少数困难 batch 中 prior 可能比 zero 更差，需要后续 worst-case 分析。
```

---

## 13. 当前实验结论

本轮 100 条数据实验表明：

1. 100 条 peg-in-hole HDF5 数据整体质量可用；
2. one-frame mismatch 可以通过 safe trimming 处理，不需要丢弃对应 episode；
3. Stage 1 policy training 成功收敛；
4. action reconstruction 和 future force prediction 均显著下降；
5. posterior contact latent 被有效使用；
6. Stage 2 prior-only distillation 能够进一步改善 prior-posterior 对齐；
7. 在 held-out val10/test10 上，deployable prior inference 明显优于 zero-latent baseline；
8. conditional contact prior 对 future force prediction 的提升尤其显著；
9. prior latent 也能改善 action prediction，说明接触状态隐变量对未来动作选择有正向作用。

可以记录为正式结论：

```text
On the 100-episode peg-in-hole dataset, we used an 80/10/10 train/validation/test episode split for offline evaluation. Compared with the zero-contact-latent baseline, deterministic contact-prior inference consistently improved both action and future-force prediction on held-out validation and test episodes. On the test split, prior inference reduced the action L1 error by 14.5% and the future-force L1 error by 57.9% on average. The predicted contact latent also remained well aligned with the posterior oracle latent, with a median cosine similarity of 0.988 on the test split. These results suggest that the conditional contact prior can infer useful contact-dynamics latent variables from online observations and substantially improve deployable force-aware prediction.
```

中文总结：

```text
在 100 条 peg-in-hole 示教数据上，我们采用 80/10/10 的 train/validation/test episode split 进行离线评估。结果表明，在可部署的 deterministic contact-prior inference 模式下，模型相比 z_contact=0 的 zero-latent baseline 在 held-out validation 和 test episodes 上均取得稳定提升。在 test10 上，prior inference 将动作预测误差平均降低约 14.5%，将未来力预测误差平均降低约 57.9%。同时，prior 预测的 contact latent 与 posterior oracle latent 保持较高方向一致性，test10 上 cosine similarity 的 median 达到 0.988。这说明 conditional contact prior 能够从当前视觉、关节状态和历史力窗口中推断有效的接触动力学隐变量，并显著改善可部署推理条件下的接触力建模能力。
```

---

## 14. 需要注意的问题

### 14.1 Posterior oracle 和 prior 仍有明显差距

posterior mode 明显优于 prior mode，尤其是 action prediction。这是正常的，因为 posterior encoder 使用了 future action 和 future force，属于 oracle/debug 模式。

这说明：

```text
当前 prior 仍然没有完全捕捉未来接触模式；
当前观测与过去力窗口无法完全确定未来接触状态；
posterior latent 的信息容量仍然较高。
```

### 14.2 少数 batch 中 prior 会比 zero 更差

在 val/test 中，improvement ratio 的最小值为负：

```text
val10:
  action min improvement = -0.299296
  force min improvement  = -1.56192

test10:
  action min improvement = -0.394092
  force min improvement  = -2.70114
```

说明少数困难样本中 prior latent 会误判接触模式。这些样本可能对应：

```text
强接触
卡滞
接触突变
视觉遮挡
历史力窗口不足以判断未来
采集异常或 episode 分布差异
```

后续需要做 worst-case analysis。

### 14.3 Absolute path split 不应提交

本轮实验为了快速修复路径解析问题，使用了 absolute path split。后续应统一 episode-list path resolver，恢复 project-root-relative split 文件。

---

## 15. 后续工作

建议下一步做以下工作：

### 15.1 修复 episode-list 路径解析

统一所有脚本中的 episode path resolution：

```text
absolute path -> use directly
project-root-relative path -> prefer
episode-list-parent-relative path -> fallback
```

这样可以避免 `configs/splits/mujoco_data/...` 这类错误路径。

### 15.2 Worst-case analysis

为 `evaluate_inference_modes.py` 增加：

```text
--save-worst-cases
--top-k-worst
```

记录 prior 比 zero 差最多的 batch，包括：

```text
episode path
global index
state index
timestamp
action_l1_zero/prior/posterior
force_l1_zero/prior/posterior
improvement ratio
mu prior-posterior mse/l2/cosine
force window statistics
future force statistics
```

这样可以定位哪些接触状态最难建模。

### 15.3 Latent PCA on train/val/test

对 100 条数据重新做 force-balanced latent PCA，并分别检查：

```text
posterior contact latent structure
prior-posterior overlay
episode identity effect
future force magnitude coloring
force delta coloring
time coloring
```

### 15.4 Evaluate closed-loop rollout

当前评估是 offline supervised prediction。下一阶段需要在 MuJoCo 或真实控制链路中验证：

```text
zero mode vs prior mode
```

对实际插孔成功率、接触力峰值、轨迹平滑性和卡滞恢复能力的影响。

### 15.5 Ablation study

后续可加入 ablation：

```text
without force window
without force-vision cross-attention
without z_contact
zero prior vs conditional prior
Stage 1 only vs Stage 1 + Stage 2
different force window length
different lambda_force / lambda_prior
```

---

## 16. 当前实验状态

截至本轮实验结束，项目状态如下：

```text
Data inspection: passed with tolerant trimming
Train80 normalization: completed
Stage 1 policy training: completed
Stage 2 prior distillation: completed
Val10 inference evaluation: completed
Test10 inference evaluation: completed
Main conclusion: conditional contact prior improves deployable action and future-force prediction, especially force prediction
```

本轮 100 条演示实验可以视为 ForceAwareACT 当前阶段的第一个完整有效实验闭环。
