# ForceAwareACT 五模型规范化训练与早停手册

本文档规定本仓库五种实验配置从数据划分、归一化、smoke、正式训练、epoch 早停到 checkpoint 选择的统一流程。命令以 `peg_hole_100` 的 train80/val10/test10 划分为当前实例，均从项目根目录执行。

```bash
cd ~/ForceAwareACT_workspace/ForceAwareACT
conda activate forceact
```

## 1. “五模型”的准确含义

仓库当前实现了 4 种 policy architecture，但实验中通常比较 5 种训练/部署配置：

| 实验名称 | `policy_variant` | 训练 latent | 部署/验证 latent | prior 是否训练 |
|---|---|---|---|---|
| Contact-CVAE-zero | `force_aware_contact_cvae` | contact posterior | contact zero | 否，`lambda_prior=0` |
| Contact-CVAE-prior | `force_aware_contact_cvae` | contact posterior | deterministic contact prior | 是，`lambda_prior=0.1` |
| Motion-CVAE | `force_aware_motion_cvae` | motion posterior | motion zero | 无 contact prior |
| DualZero | `force_aware_act` | motion/contact 均为 zero | motion/contact 均为 zero | 否 |
| ACT baseline | `act_baseline` | motion posterior | motion zero | 结构中没有 contact prior |

因此：

- Contact-zero 和 Contact-prior 使用同一种网络结构，但规范的五配置实验把它们作为两次独立训练；
- 历史五模型 rollout 曾让 Contact-zero 与 Contact-prior 共用同一个 prior-trained Contact-CVAE checkpoint，只在 rollout 时切换 latent mode；这种做法只有 4 个训练 checkpoint；
- 新实验若要回答“完全不训练 prior 是否更好”，必须单独训练 Contact-CVAE-zero，不能直接把 prior-trained checkpoint 的 rollout mode 改成 zero 后称为 pure-zero training。

## 2. 统一训练原则

五种配置共享以下规则：

1. 以 episode 为单位固定 train/validation/test；
2. normalization stats 只使用 train split；
3. 每个 epoch 结束后在完整 validation split 上执行确定性部署路径；
4. 训练 loss 只用于诊断，不用于手工选择 epoch；
5. 使用预先声明的 validation 指标和 patience 自动早停；
6. 使用 `checkpoint_best.pt` 进入 test 和 rollout，不默认使用最后的 `checkpoint.pt`；
7. test 只在配置和 checkpoint 选择规则冻结后使用；
8. future action 和 future force 只能作为训练标签或 posterior oracle 标签，不能进入部署推理输入；
9. 每个模型使用独立输出目录，禁止复用目录混写日志；
10. 修改模型、数据语义或超参数后必须重新 smoke。

统一流程：

```text
固定 episode 划分
  -> 仅使用 train 计算 normalization stats
  -> 五配置逐一 smoke
  -> CUDA/磁盘预检
  -> 五配置分别正式训练
  -> 每 epoch 运行各自真实部署路径的 validation
  -> 自动早停并保存 checkpoint_best.pt
  -> 固定协议的闭环 rollout
  -> 最后一次 test 评估
  -> 多 seed 汇总与实验归档
```

## 3. 数据划分与归一化

### 3.1 当前划分

| 用途 | 列表 | episode 数 | 用途限制 |
|---|---|---:|---|
| Train | `configs/splits/peg_hole_100_train80.txt` | 80 | 训练和 stats |
| Validation | `configs/splits/peg_hole_100_val10.txt` | 10 | 早停和 checkpoint 选择 |
| Test | `configs/splits/peg_hole_100_test10.txt` | 10 | 配置冻结后的最终评估 |

检查数量：

```bash
wc -l \
  configs/splits/peg_hole_100_train80.txt \
  configs/splits/peg_hole_100_val10.txt \
  configs/splits/peg_hole_100_test10.txt
```

使用固定种子重新生成相同划分：

```bash
python scripts/split_episode_list.py \
  --input outputs/peg_hole_100/all100.txt \
  --train-output configs/splits/peg_hole_100_train80.txt \
  --val-output configs/splits/peg_hole_100_val10.txt \
  --test-output configs/splits/peg_hole_100_test10.txt \
  --train-count 80 \
  --val-count 10 \
  --seed 20260701
```

训练器会自动拒绝 train/validation 重叠，并验证 stats 中记录的 episode 与 train split 完全一致。test 隔离仍应通过实验流程管理，不能拿 test 曲线调参。

### 3.2 统一 stats

五个配置都使用 `action_mode=action`。当前 stats：

```text
outputs/peg_hole_100/normalization_stats_action_train80.pt
```

重新计算：

```bash
PYTHONPATH=src python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --action-mode action \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --camera-names ee_cam base_top_cam \
  --output outputs/peg_hole_100/normalization_stats_action_train80.pt
```

ACT baseline 只读取其中的 qpos/action 统计，不读取 force；另外四个 force-aware 配置同时使用 force 统计。改变 action mode、chunk、force window、相机顺序、图像预处理或 train split 时必须重新计算 stats。

## 4. 统一 epoch 与早停规则

本轮建议所有模型先使用同一停止预算：

| 参数 | 规范值 | 作用 |
|---|---:|---|
| `--max-steps` | 200000 | 防止异常长运行的安全上限 |
| `--max-epochs` | 100 | epoch 硬上限 |
| `--val-every-epochs` | 1 | 每个 epoch 完整验证一次 |
| `--early-stop-min-epochs` | 20 | 前 20 epochs 不因 patience 停止 |
| `--early-stop-patience` | 10 | 连续 10 次无有效改善则停止 |
| `--early-stop-min-delta` | 0.005 | 至少相对改善 0.5% 才刷新 best |
| `--early-stop-metric` | `deploy_loss` | 用部署路径的综合离线误差选模型 |

当前 train80 有 23934 个有效样本，batch size 16：

```text
steps_per_epoch = ceil(23934 / 16) = 1496
20 epochs       = 29920 steps
100 epochs      = 149600 steps
```

`max_steps` 和 `max_epochs` 同时生效，先达到者结束。当前配置通常由 early stopping 或 100 epochs 先结束。

### 4.1 有效改善定义

设历史最佳指标为 `best`，当前指标为 `current`：

```text
current < best - abs(best) * 0.005
```

才视为有效改善并更新 `checkpoint_best.pt`。小于 0.5% 的变化视为平台波动，在 epoch 不小于 20 后计入 patience。最极端的无改善情况下，最早约在 epoch 29 停止；每次有效改善都会清零计数。

### 4.2 各模型的 deploy loss

Force-aware 四配置：

```text
deploy_loss = action_l1 + lambda_force * force_l1
            = action_l1 + 0.1 * force_l1
```

ACT baseline 没有 force head：

```text
deploy_loss = action_l1
```

这些指标在 normalization 后的空间计算。它们适合自动筛选 checkpoint，但不能替代闭环任务成功率、峰值力和安全停止分析。

## 5. 五种配置的关键差异

| 配置 | 训练入口 | 主要 KL | `lambda_prior` | validation mode | 应出现的训练日志 |
|---|---|---:|---:|---|---|
| Contact-zero | `train_minimal.py` | `beta_contact=5e-4` | 0 | zero | `kl_contact>0`，`loss_prior=0` |
| Contact-prior | `train_minimal.py` | `beta_contact=5e-4` | 0.1 | prior | `kl_contact>0`，prior loss 有限 |
| Motion-CVAE | `train_minimal.py` | `beta_motion=5e-4` | 0 | zero | `kl_motion>0`，`kl_contact=0` |
| DualZero | `train_minimal.py` | 实际均不进入 loss | 0 | zero | `kl_motion=0`，`kl_contact=0` |
| ACT baseline | `train_act_baseline.py` | `beta_motion=5e-4` | 不存在 | zero | action loss 与 motion KL，无 force loss |

下面五节给出每一种配置的完整正式命令。

## 6. Contact-CVAE-zero：posterior 训练，zero 部署，不训练 prior

### 6.1 研究语义

- contact posterior 使用 future action/force 标签学习；
- contact prior 模块为结构兼容而保留，但不参与 loss；
- validation 和 rollout 都固定 `z_contact=0`；
- 该配置用于回答“不学习 prior 的 Contact-CVAE”表现如何。

关键参数：

```text
--train-latent-mode posterior
--train-contact-latent-mode posterior
--lambda-prior 0
--validation-deployment-mode zero
```

### 6.2 正式训练

当前已准备的启动脚本：

```bash
bash outputs/peg_hole_100/contact_cvae_zero_earlystop_train80/run_formal_seed0.sh
```

等价核心命令：

```bash
OUT="outputs/peg_hole_100/earlystop_train80/contact_cvae_zero_seed0"
mkdir -p "$OUT"

PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --val-episode-list configs/splits/peg_hole_100_val10.txt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --policy-variant force_aware_contact_cvae \
  --action-mode action \
  --train-latent-mode posterior \
  --train-contact-latent-mode posterior \
  --chunk-len 10 \
  --image-size 224 224 \
  --camera-names ee_cam base_top_cam \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --lambda-force 0.1 \
  --lambda-prior 0 \
  --prior-loss-mode mse_mu \
  --beta-contact-max 5e-4 \
  --warmup-steps 2000 \
  --max-steps 200000 \
  --max-epochs 100 \
  --val-every-epochs 1 \
  --early-stop-min-epochs 20 \
  --early-stop-patience 10 \
  --early-stop-min-delta 0.005 \
  --early-stop-metric deploy_loss \
  --validation-deployment-mode zero \
  --save-every 10000 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --seed 0 \
  --device cuda \
  --output-dir "$OUT" \
  --log-csv "$OUT/train_log.csv" \
  --validation-log "$OUT/validation_log.csv" \
  2>&1 | tee "$OUT/console.log"
```

验收重点：训练日志中的 `lambda_prior` 和 `loss_prior` 始终为 0，validation 始终为 `mode=zero`。当前配置已通过 1-step smoke，并验证 8 个 `contact_prior` 状态张量训练前后完全不变。

## 7. Contact-CVAE-prior：posterior 训练，prior 部署

### 7.1 研究语义

- contact posterior 作为监督 teacher；
- conditional contact prior 从在线输入学习匹配 posterior；
- validation 使用 deterministic prior，避免随机采样噪声干扰早停；
- rollout 若评价 prior，也必须显式使用 prior mode。

关键参数：

```text
--train-latent-mode posterior
--train-contact-latent-mode posterior
--lambda-prior 0.1
--validation-deployment-mode prior
```

### 7.2 正式训练

```bash
OUT="outputs/peg_hole_100/earlystop_train80/contact_cvae_prior_seed0"
mkdir -p "$OUT"

PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --val-episode-list configs/splits/peg_hole_100_val10.txt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --policy-variant force_aware_contact_cvae \
  --action-mode action \
  --train-latent-mode posterior \
  --train-contact-latent-mode posterior \
  --chunk-len 10 \
  --image-size 224 224 \
  --camera-names ee_cam base_top_cam \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --lambda-force 0.1 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --beta-contact-max 5e-4 \
  --warmup-steps 2000 \
  --max-steps 200000 \
  --max-epochs 100 \
  --val-every-epochs 1 \
  --early-stop-min-epochs 20 \
  --early-stop-patience 10 \
  --early-stop-min-delta 0.005 \
  --early-stop-metric deploy_loss \
  --validation-deployment-mode prior \
  --save-every 10000 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --seed 0 \
  --device cuda \
  --output-dir "$OUT" \
  --log-csv "$OUT/train_log.csv" \
  --validation-log "$OUT/validation_log.csv" \
  2>&1 | tee "$OUT/console.log"
```

验收重点：`lambda_prior=0.1`、prior loss 有限、validation 为 `mode=prior`。如果最终主要部署 zero，应把 validation mode 改为 zero 并把实验名称写清楚，不能在训练完成后根据结果临时切换选择规则。

## 8. Motion-CVAE：motion posterior 训练，zero 部署

### 8.1 研究语义

- 结构中只有 motion latent，没有 contact latent 和 contact prior；
- 训练使用 future action 形成 motion posterior；
- 部署时 motion latent 固定为 zero；
- 仍读取在线 force history，并保留 future force 辅助预测。

关键参数：

```text
--policy-variant force_aware_motion_cvae
--train-latent-mode posterior
--lambda-prior 0
--validation-deployment-mode zero
```

### 8.2 正式训练

```bash
OUT="outputs/peg_hole_100/earlystop_train80/motion_cvae_seed0"
mkdir -p "$OUT"

PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --val-episode-list configs/splits/peg_hole_100_val10.txt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --policy-variant force_aware_motion_cvae \
  --action-mode action \
  --train-latent-mode posterior \
  --chunk-len 10 \
  --image-size 224 224 \
  --camera-names ee_cam base_top_cam \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --lambda-force 0.1 \
  --lambda-prior 0 \
  --beta-motion-max 5e-4 \
  --warmup-steps 2000 \
  --max-steps 200000 \
  --max-epochs 100 \
  --val-every-epochs 1 \
  --early-stop-min-epochs 20 \
  --early-stop-patience 10 \
  --early-stop-min-delta 0.005 \
  --early-stop-metric deploy_loss \
  --validation-deployment-mode zero \
  --save-every 10000 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --seed 0 \
  --device cuda \
  --output-dir "$OUT" \
  --log-csv "$OUT/train_log.csv" \
  --validation-log "$OUT/validation_log.csv" \
  2>&1 | tee "$OUT/console.log"
```

验收重点：motion KL 有限，contact KL 和 prior loss 为 0，validation 为 zero。

## 9. DualZero：双 latent 结构，但训练和部署均为 zero

### 9.1 研究语义

- 使用完整 `force_aware_act` 双 latent 结构；
- 训练 forward 中 motion/contact latent 都固定为 zero；
- posterior KL 和 prior loss 不进入总 loss；
- 模型主要学习图像、qpos、在线 force 到 action/future force 的确定性映射。

关键参数：

```text
--policy-variant force_aware_act
--train-latent-mode zero
--lambda-prior 0
--validation-deployment-mode zero
```

### 9.2 正式训练

```bash
OUT="outputs/peg_hole_100/earlystop_train80/dualzero_seed0"
mkdir -p "$OUT"

PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --val-episode-list configs/splits/peg_hole_100_val10.txt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --policy-variant force_aware_act \
  --action-mode action \
  --train-latent-mode zero \
  --chunk-len 10 \
  --image-size 224 224 \
  --camera-names ee_cam base_top_cam \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --lambda-force 0.1 \
  --lambda-prior 0 \
  --beta-motion-max 1e-4 \
  --beta-contact-max 1e-4 \
  --warmup-steps 2000 \
  --max-steps 200000 \
  --max-epochs 100 \
  --val-every-epochs 1 \
  --early-stop-min-epochs 20 \
  --early-stop-patience 10 \
  --early-stop-min-delta 0.005 \
  --early-stop-metric deploy_loss \
  --validation-deployment-mode zero \
  --save-every 10000 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --seed 0 \
  --device cuda \
  --output-dir "$OUT" \
  --log-csv "$OUT/train_log.csv" \
  --validation-log "$OUT/validation_log.csv" \
  2>&1 | tee "$OUT/console.log"
```

这里保留的 beta 值与历史 DualZero 配置一致，但 `train_latent_mode=zero` 时它们不会进入 loss。验收重点：`uses_zero_latent=True`，motion/contact KL 和 prior loss 都为 0，validation 为 zero。

## 10. ACT baseline：force-free Motion-CVAE

### 10.1 研究语义

- 不读取在线 force；
- 没有 force head、contact latent 或 contact prior；
- 训练使用 motion posterior；
- validation/部署使用 zero motion latent；
- early-stop `deploy_loss` 只等于 action L1。

ACT baseline 使用独立入口 `scripts/train_act_baseline.py`，不能用 `train_minimal.py --policy-variant act_baseline` 替代。

### 10.2 正式训练

```bash
OUT="outputs/peg_hole_100/earlystop_train80/act_baseline_run0"
mkdir -p "$OUT"

PYTHONPATH=src python scripts/train_act_baseline.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --val-episode-list configs/splits/peg_hole_100_val10.txt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --action-mode action \
  --chunk-len 10 \
  --image-size 224 224 \
  --camera-names ee_cam base_top_cam \
  --beta-motion-max 5e-4 \
  --warmup-steps 2000 \
  --max-steps 200000 \
  --max-epochs 100 \
  --val-every-epochs 1 \
  --early-stop-min-epochs 20 \
  --early-stop-patience 10 \
  --early-stop-min-delta 0.005 \
  --early-stop-metric deploy_loss \
  --save-every 10000 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --d-model 128 \
  --z-dim 16 \
  --nhead 4 \
  --num-encoder-layers 1 \
  --num-decoder-layers 1 \
  --dim-feedforward 256 \
  --dropout 0.0 \
  --device cuda \
  --output-dir "$OUT" \
  --log-csv "$OUT/train_log.csv" \
  --validation-log "$OUT/validation_log.csv" \
  2>&1 | tee "$OUT/console.log"
```

验收重点：`uses_force=False`、`uses_contact_latent=False`、validation 为 zero，并且 validation CSV 不包含 force L1。

注意：当前 `train_act_baseline.py` 没有 `--seed` 参数，而 `train_minimal.py` 有。这是统一多 seed 复现前需要补齐的代码差异；在该功能实现前，ACT 输出目录使用 `run0` 而不是声称为可控的 `seed0`。

## 11. Smoke 规范

每种配置在正式训练前都应单独 smoke。以对应正式命令为基础，只修改：

```text
--max-steps 1
--batch-size 1
--num-workers 0
--early-stop-min-epochs 1
--early-stop-patience 1
--device cpu
--output-dir <独立的 smoke 目录>
```

保留 train/val split、stats、policy variant、latent mode、loss 权重和 validation deployment mode，确保 smoke 验证的就是正式语义。

统一验收条件：

- episode 和 stats 正常解析；
- forward、backward、optimizer step 和 validation 全部完成；
- 所有 loss 和 validation 指标有限；
- 生成 `train_log.csv`、`validation_log.csv`、`checkpoint.pt`、`checkpoint_best.pt`；
- validation mode 与该模型的真实部署 mode 一致；
- 日志中的 KL/prior/force 分项符合第 5 节表格。

如果更改 batch size 只是为了 smoke，不应拿 smoke 数值与正式训练曲线比较。

## 12. 正式训练前检查

### 12.1 CUDA

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('count=', torch.cuda.device_count())"
```

只有 CUDA 为 `True` 且 device count 大于 0 时才启动正式训练。

### 12.2 磁盘和工作区记录

```bash
df -h outputs
git rev-parse HEAD
git status --short
```

保存 commit、dirty status、环境、启动命令和运行日期。不要为了开始实验而清理用户已有改动。

### 12.3 输出目录

每个模型、seed 和超参数组合必须使用独立目录：

```text
outputs/peg_hole_100/earlystop_train80/
  contact_cvae_zero_seed0/
  contact_cvae_prior_seed0/
  motion_cvae_seed0/
  dualzero_seed0/
  act_baseline_run0/
```

启动前如目录已包含日志或 checkpoint，应换一个新目录，不覆盖旧实验。

## 13. 训练过程监控

以某个输出目录为例：

```bash
OUT="outputs/peg_hole_100/earlystop_train80/contact_cvae_zero_seed0"
tail -f "$OUT/console.log"
tail -n 12 "$OUT/validation_log.csv"
```

重点检查：

- `dataset_length`、`steps_per_epoch`、`effective_max_steps` 符合预期；
- validation mode 与第 1 节一致；
- loss、KL、action L1、force L1 没有 `NaN/Inf`；
- `checkpoint_best.pt` 在有效改善时更新；
- `epochs_without_improvement` 正常清零和累计；
- GPU 显存稳定，没有持续增长或 OOM；
- pure-zero、Motion-CVAE、DualZero 的 prior loss 始终为 0。

训练结束后读取停止原因：

```bash
python -c "import torch; p=torch.load('$OUT/checkpoint.pt', map_location='cpu', weights_only=False); print({k: p.get(k) for k in ('step','epoch','best_metric','best_epoch','best_step','epochs_without_improvement','stop_reason')})"
```

常见 `stop_reason`：

| 值 | 含义 |
|---|---|
| `early_stopping` | patience 已耗尽 |
| `max_epochs` | 达到 epoch 上限 |
| `max_steps` | 达到 step 上限 |
| `best_validation_metric` | best checkpoint 的保存原因 |

## 14. Checkpoint 选择和最终评估

### 14.1 文件职责

- `checkpoint_best.pt`：validation 指标最佳，进入 test/rollout；
- `checkpoint.pt`：停止时最后状态，用于审计；
- `checkpoint_step_XXXXXXXX.pt`：周期快照，用于曲线诊断或邻近 checkpoint 对照。

不要因为最后训练 loss 更低而覆盖 validation 选择结果。正式报告至少记录 best epoch、best step、best metric 和 stop reason。

### 14.2 离线 evaluator 对应关系

| 配置 | evaluator | 可部署主结果 |
|---|---|---|
| Contact-zero | `evaluate_contact_cvae_modes.py` | zero；prior 未训练，不报告为有效结果 |
| Contact-prior | `evaluate_contact_cvae_modes.py` | prior；zero 可作为预先声明的补充消融 |
| Motion-CVAE | `evaluate_motion_cvae_modes.py` | zero |
| DualZero | `evaluate_inference_modes.py` | zero |
| ACT baseline | `evaluate_act_baseline_modes.py` | zero |

任何 posterior 指标都属于离线 oracle，不能称为部署性能。

### 14.3 Rollout mode

| 配置 | rollout 关键参数 |
|---|---|
| Contact-zero | `--contact-latent-mode zero` |
| Contact-prior | `--contact-latent-mode prior` |
| Motion-CVAE | zero motion latent，由模型部署路径固定 |
| DualZero | `--contact-latent-mode zero` |
| ACT baseline | zero motion latent，由模型部署路径固定 |

五模型闭环比较必须使用相同任务点、rollout seed、安全阈值、动作选择方式、最大步数和控制裁剪参数。不能为某个模型单独修改协议后继续称为配对比较。

test10 只能在配置、checkpoint 选择和 rollout 协议冻结后使用一次。若 test 暴露问题，下一轮应创建新实验，不得回头用 test 选择旧运行的 checkpoint。

## 15. 多随机种子与最终 epoch

对支持 `--seed` 的四个 `train_minimal.py` 配置，正式比较建议至少运行：

```text
seed = 0, 1, 2
```

每个 seed 独立早停，报告 test/rollout 的均值和离散程度，不从多个 seed 中只挑最优者。ACT baseline 在补齐 seed CLI 前无法与这四个配置形成完全一致的受控多 seed 流程，应在报告中明确这一限制。

如果配置已通过 train80/val10 选择，之后希望使用 all100 训练最终部署模型：

1. 使用多个 seed 的 `best_epoch` 中位数作为固定 epoch；
2. 使用 all100 重新计算 stats；
3. 每个模型在 all100 上训练相同的已选 epoch 规则；
4. 按 all100 的 `steps_per_epoch × selected_epoch` 重新计算 step，不照搬 train80 step；
5. all100 模型不再拥有同一数据集上的未见 test10，不能继续声明 test10 泛化性能。

论文或泛化结论保留 train80/val10/test10 模型；all100 模型只作为最终部署版本。

## 16. 异常处理

### CUDA 不可用

不要用 CPU 无意启动完整训练。恢复 GPU 后重新运行 CUDA 预检。

### CUDA OOM

在新目录中降低 batch size 并重新训练。batch size 改变会改变优化过程和 `steps_per_epoch`，需要重新计算 epoch/step 对应关系，不能与原配置混为同一次实验。

### NaN/Inf

停止运行并保留日志，定位首个异常 step，检查数据、stats、学习率和 KL。不要用异常运行的最后 checkpoint。

### 早停过早

先检查 action/force 分项和多个 seed。只有多个独立运行都显示相同问题时，才在新实验中调整 `min_epochs` 或 `patience`。不能在运行途中改变规则。

### 进程中断

当前训练 CLI 没有 resume 功能。保留中断目录用于审计，使用新目录从头启动，避免日志/checkpoint 混写。

## 17. 归档与验收清单

每次正式运行至少归档：

- 模型名称、policy variant、训练/部署 latent 语义；
- 完整启动命令和输出目录；
- Git commit、dirty status、日期和环境；
- train/val/test 列表与 split seed；
- normalization stats 与 episode provenance；
- seed；ACT baseline 当前需记录为未受 CLI 控制；
- `console.log`、`train_log.csv`、`validation_log.csv`；
- `checkpoint_best.pt`、`checkpoint.pt` 和必要的周期 checkpoint；
- best epoch、best step、best metric、stop reason；
- test 输出；
- rollout 任务点、seed、安全参数和汇总结果。

最终验收必须同时满足：

- 五配置使用相同 train/val/test 和数据语义；
- 每种配置的训练 latent 与 validation 部署路径一致于第 1 节定义；
- epoch 由预先声明的 validation 规则选择；
- 使用 `checkpoint_best.pt`；
- test 没有参与调参；
- posterior oracle 未被误报为部署性能；
- Contact-zero 的 prior 确实未训练；
- 五模型闭环比较使用完全相同的 rollout 协议；
- 实验可由命令、split、stats、checkpoint 和 seed 复现。

## 18. 当前五模型执行顺序

建议按下面顺序推进，避免长训练前遗漏语义错误：

```text
1. Contact-CVAE-zero smoke -> formal
2. Contact-CVAE-prior smoke -> formal
3. Motion-CVAE smoke -> formal
4. DualZero smoke -> formal
5. ACT baseline smoke -> formal
6. 汇总五个 validation_log.csv 和 best checkpoint 元数据
7. 冻结 checkpoint 与 rollout 协议
8. 五模型固定点闭环 rollout
9. 最后执行 test10 离线评估
10. 多 seed 汇总并形成实验报告
```
