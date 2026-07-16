# ForceAwareACT 新数据集训练实验手册

本文档规定从一批新的 HDF5 episode 到完成 ForceAware Contact-CVAE 训练的标准流程，已于 2026-07-16 按当前源码复核。命令均从项目根目录、已激活的 Python 环境中执行，并以 `mujoco_data/new_dataset` 作为新数据集示例。

本流程包含两条可选路线：

- **泛化评估路线**：划分 `train/val/test`，只用 `train` 计算归一化统计和训练，用 `val` 选择配置，最后在 `test` 上报告结果。
- **全量最终训练路线**：不划分数据，使用 `all.txt` 计算归一化统计并训练。历史 Contact-CVAE 100k 模型采用的就是这条路线。它适合充分利用数据训练部署模型，但同数据集离线结果不能作为未见数据泛化证据。

标准流程如下：

```text
HDF5 episodes
  -> all.txt
  -> 采集质量门禁 + 数据读取器一致性检查
  -> 确定 action_mode 和数据语义参数
  -> [可选] train/val/test 划分
  -> 使用训练列表计算 normalization stats
  -> smoke training
  -> 正式训练
  -> [可选] validation/test 离线评估
  -> 归档命令、日志、stats 和 checkpoints
```

## 1. 环境和实验变量

进入项目并激活环境：

```bash
cd ~/ForceAwareACT_workspace/ForceAwareACT
conda activate forceact
```

为新实验统一定义路径和关键数据参数：

```bash
DATA_DIR="mujoco_data/new_dataset"
EXP_ROOT="outputs/new_dataset"
ALL_LIST="$EXP_ROOT/all.txt"

ACTION_MODE="action"
CHUNK_LEN=10
FORCE_WINDOW_LEN=20
FORCE_WINDOW_DURATION=0.25
IMAGE_HEIGHT=224
IMAGE_WIDTH=224

mkdir -p "$EXP_ROOT"
```

参数说明：

| 参数 | 示例 | 说明 |
|---|---:|---|
| `DATA_DIR` | `mujoco_data/new_dataset` | 包含 episode 子目录的新数据集根目录。 |
| `EXP_ROOT` | `outputs/new_dataset` | 本实验的列表、统计量、日志和模型输出根目录。 |
| `ACTION_MODE` | `action` | 动作监督来源；确定后必须在统计、训练、评估和 rollout 中保持一致。 |
| `CHUNK_LEN` | `10` | 每个样本监督的未来动作/未来力步数。 |
| `FORCE_WINDOW_LEN` | `20` | 在线输入包含的历史力采样点数。 |
| `FORCE_WINDOW_DURATION` | `0.25` | 历史力窗口覆盖的时间，单位为秒。 |
| `IMAGE_HEIGHT/WIDTH` | `224/224` | 输入模型前的图像尺寸。 |

如需启用 ImageNet 图像归一化，应从计算 stats 开始就在所有后续命令中加入 `--imagenet-normalize`。不要只在其中某一步加入。

## 2. 生成全集数据列表

项目列表文件每行包含一个 HDF5 episode 路径。推荐保存相对于项目根目录的路径，以便仓库移动后仍可解析。

当数据目录本身是项目内的相对路径时，生成排序稳定的全集列表：

```bash
find "$DATA_DIR" -type f -name episode.hdf5 | sort > "$ALL_LIST"
```

检查数量和前几行：

```bash
wc -l "$ALL_LIST"
sed -n '1,5p' "$ALL_LIST"
```

检查重复路径；正常情况应无输出：

```bash
sort "$ALL_LIST" | uniq -d
```

检查列表中所有文件是否存在；正常情况应无输出：

```bash
while IFS= read -r path; do
  test -f "$path" || echo "missing: $path"
done < "$ALL_LIST"
```

`find` 参数说明：

| 参数 | 说明 |
|---|---|
| `-type f` | 只选择普通文件。 |
| `-name episode.hdf5` | 只选择标准 episode 文件。 |
| `sort` | 固定列表顺序，避免文件系统遍历顺序变化。 |
| `> "$ALL_LIST"` | 将结果写入本实验的全集列表。 |

验收条件：列表非空、数量符合采集记录、无重复项、所有路径存在，并且没有混入别的数据集。

## 3. 数据集质量检测（必须）

质量检测分为单 episode 抽查、全集结构检查和动作标签检查。任何一项失败，都应先修复或明确排除坏 episode，再重新生成最终 `all.txt`。不要直接修改原始 HDF5 数据。

### 3.1 单 episode 抽查

取列表第一条进行结构和样本形状检查：

```bash
FIRST_EPISODE="$(sed -n '1p' "$ALL_LIST")"

PYTHONPATH=src python scripts/inspect_real_hdf5.py \
  "$FIRST_EPISODE" \
  --chunk-len "$CHUNK_LEN" \
  --force-window-len "$FORCE_WINDOW_LEN"
```

参数说明：

| 参数 | 说明 |
|---|---|
| `episode_path` | 必填位置参数，要检查的一条 `episode.hdf5`。 |
| `--chunk-len` | 使用计划中的未来监督长度计算有效样本和形状。 |
| `--force-window-len` | 使用计划中的历史力窗口长度检查样本形状。 |

重点确认图像、`qpos`、`force_window`、`action_chunk` 和 `future_force_chunk` 的 shape 符合预期。

### 3.2 采集质量门禁

对当前 command-labelled 录制格式，先运行只读质量门禁：

```bash
PYTHONPATH=src python scripts/evaluate_dataset_quality.py \
  "$DATA_DIR" \
  --output-csv "$EXP_ROOT/quality_report.csv" \
  --output-json "$EXP_ROOT/quality_summary.json" \
  2>&1 | tee "$EXP_ROOT/quality_report.log"
```

它检查采集 `status`、当前录制字段、严格递增时间戳、非有限数、平移力/力矩、关节速度、命令步长与跟踪误差，并抽样检查空白或冻结图像。默认要求 `timestamps/*_episode` 和 `actions/joint_pos_command`，因此它针对当前新录制格式，比通用 `ContactForceHDF5Dataset` 更严格。

`quality=reject` 的 episode 不应进入最终列表；`quality=review` 必须人工查看具体 warning。阈值是工程筛查规则，不是接触安全或数据正确性的数学证明。若数据来自旧格式，不能简单忽略缺字段错误；应使用下一节的通用读取器检查，并明确记录为什么不适用该门禁。

### 3.3 全集结构与时间对齐检查

```bash
QUALITY_CSV="$EXP_ROOT/dataset_inspection.csv"

PYTHONPATH=src python scripts/inspect_episode_collection.py \
  --episode-list "$ALL_LIST" \
  --chunk-len "$CHUNK_LEN" \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --force-window-duration "$FORCE_WINDOW_DURATION" \
  --output-csv "$QUALITY_CSV" \
  2>&1 | tee "$EXP_ROOT/dataset_inspection.log"
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--episode-list` | 待检查的 episode 列表。 |
| `--chunk-len` | 决定 episode 能否产生足够长的动作/未来力监督。 |
| `--force-window-len` | 历史力窗口采样点数。 |
| `--force-window-duration` | 历史力窗口覆盖秒数。 |
| `--output-csv` | 将每条 episode 的长度、持续时间、力统计和问题写入 CSV。 |
| `tee` | 同时在终端显示并保存检查日志。 |

当前检查器验证：

- `ee_pose`、`joint_pos`、`joint_vel`、`joint_torque` 为 `[N, 7]`；
- `ft_wrench` 为 `[N, 6]`，且没有 `NaN/Inf`；
- `ee_cam`、`base_top_cam` 图像为 `[N, H, W, 3]`；
- state/image/force 时间戳存在、有限且单调不减；
- 同类数据长度是否一致；
- episode 在指定窗口参数下是否能产生有效训练样本；
- 每条 episode 的力均值、标准差、最小值和最大值。

注意：该脚本当前固定检查 `ee_cam` 和 `base_top_cam`。如果新数据集使用其他相机名称，应先扩展检查脚本或额外人工检查，不能把检查通过理解为自定义相机已经被验证。

若数据采集允许最多一帧的已知长度偏差，可显式运行容错检查：

```bash
PYTHONPATH=src python scripts/inspect_episode_collection.py \
  --episode-list "$ALL_LIST" \
  --chunk-len "$CHUNK_LEN" \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --force-window-duration "$FORCE_WINDOW_DURATION" \
  --tolerate-length-mismatch \
  --max-length-mismatch 1 \
  --output-csv "$QUALITY_CSV"
```

只有在已确认一帧偏差来自已知采集同步行为时才使用：

| 参数 | 说明 |
|---|---|
| `--tolerate-length-mismatch` | 允许检查器按安全长度裁齐数据流。 |
| `--max-length-mismatch 1` | 最多允许一帧差异；不要为了让坏数据通过而随意增大。 |

### 3.4 动作标签语义检查

集合检查器不负责确认 `/action` 与不同 action mode 的语义，因此必须单独运行：

```bash
PYTHONPATH=src python scripts/inspect_action_modes.py \
  --data-dir "$DATA_DIR" \
  --camera-names ee_cam base_top_cam \
  --chunk-len "$CHUNK_LEN" \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --sample-index 0 \
  2>&1 | tee "$EXP_ROOT/action_mode_inspection.log"
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--data-dir` | 包含 `*/episode.hdf5` 的数据目录；该脚本检查排序后的第一条 episode。 |
| `--camera-names` | 实际使用的相机名称。 |
| `--chunk-len` | 动作块长度。 |
| `--force-window-len` | 历史力窗口长度。 |
| `--sample-index` | 在第一条 episode 中抽查的样本索引。 |

预期至少看到：

```text
action_equals_joint_pos_command=True
delta_joint_cmd_matches_action_minus_current_qpos=True
finite_tensor_check=passed
```

可选的 action mode：

| 模式 | 监督含义 | 使用建议 |
|---|---|---|
| `action` | HDF5 根 `/action` 的绝对执行命令 | 新的 command-labelled 数据首选。 |
| `joint_pos_command` | `/actions/joint_pos_command` | `/action` 的语义副本。 |
| `delta_joint_cmd` | `/action - 当前 qpos` | 训练增量命令策略时使用。 |
| `delta_joint_pos_command` | joint position command 相对当前 qpos 的增量 | 仅在数据字段和部署解释一致时使用。 |
| `joint_pos` | 未来实测关节位置 | 旧实验或 state-as-action 对照，不等同于可执行命令。 |

质量检测通过标准：所有计划使用的 episode 可读、关键 shape 正确、没有非有限力/动作张量、时间戳有效、长度偏差符合已知采集约定、动作标签语义与 `ACTION_MODE` 一致。

## 4. 可选：划分 train/validation/test

是否划分由实验目标决定。

### 4.1 何时划分

选择泛化评估路线，如果需要：

- 比较模型或超参数；
- 选择 checkpoint；
- 报告模型在未见 episode 上的性能；
- 形成论文或正式实验结论。

选择全量最终训练路线，如果：

- 配置已经通过其他实验确定；
- 目标是最大化最终部署模型可用数据；
- 不把同数据集离线误差宣称为泛化结果。

### 4.2 示例：固定种子的 80/10/10 随机划分

仓库内的标准工具会先排序、再按固定 seed 洗牌，拒绝重复项，并给每个输出文件写入 provenance header。100 条 episode 的 80/10/10 示例：

```bash
PYTHONPATH=src python scripts/split_episode_list.py \
  --input "$ALL_LIST" \
  --train-output "$EXP_ROOT/train.txt" \
  --val-output "$EXP_ROOT/val.txt" \
  --test-output "$EXP_ROOT/test.txt" \
  --train-count 80 \
  --val-count 10 \
  --seed 20260701
```

输出示例（100 条 episode）：

```text
outputs/new_dataset/train.txt  # 80
outputs/new_dataset/val.txt    # 10
outputs/new_dataset/test.txt   # 10
```

每个文件第一行记录 split 名、seed 和 episode 数，空行与 `#` 开头的 header 会被 episode-list 解析器忽略。

注意事项：

- 必须按完整 episode 划分，不能把同一 episode 的帧分到不同集合；
- 如果数据按操作者、任务目标、采集批次或初始位置分组，优先按组或分层划分，避免近重复轨迹泄漏；
- 随机划分并不自动保证各种工况分布均衡；
- `test` 在模型和超参数确定前不应用于调参。

检查三个集合没有重叠：

```bash
comm -12 <(sort "$EXP_ROOT/train.txt") <(sort "$EXP_ROOT/val.txt")
comm -12 <(sort "$EXP_ROOT/train.txt") <(sort "$EXP_ROOT/test.txt")
comm -12 <(sort "$EXP_ROOT/val.txt") <(sort "$EXP_ROOT/test.txt")
```

三条命令均应无输出。

### 4.3 选择后续训练列表

泛化评估路线：

```bash
TRAIN_LIST="$EXP_ROOT/train.txt"
VAL_LIST="$EXP_ROOT/val.txt"
TEST_LIST="$EXP_ROOT/test.txt"
STATS="$EXP_ROOT/normalization_stats_action_train.pt"
```

全量最终训练路线：

```bash
TRAIN_LIST="$ALL_LIST"
STATS="$EXP_ROOT/normalization_stats_action_all.pt"
```

## 5. 计算归一化统计（必须）

必须使用实际参与训练的列表计算统计量：泛化路线只用 `train.txt`，全量路线使用 `all.txt`。不要用 `all.txt` 的统计量训练 `train.txt` 模型，否则 validation/test 信息会进入预处理。

```bash
PYTHONPATH=src python scripts/compute_normalization_stats.py \
  --episode-list "$TRAIN_LIST" \
  --output "$STATS" \
  --action-mode "$ACTION_MODE" \
  --chunk-len "$CHUNK_LEN" \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --force-window-duration "$FORCE_WINDOW_DURATION" \
  --image-size "$IMAGE_HEIGHT" "$IMAGE_WIDTH" \
  --camera-names ee_cam base_top_cam \
  --batch-size 64 \
  --num-workers 0 \
  --eps 1e-6 \
  2>&1 | tee "$EXP_ROOT/normalization.log"
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--episode-list` | 用于计算统计量的训练 episode 列表。 |
| `--output` | 保存的 PyTorch stats 字典。 |
| `--action-mode` | 决定动作标签来源及 `action_mean/std` 的语义。 |
| `--chunk-len` | 与训练相同的未来监督长度。 |
| `--force-window-len` | 与训练相同的历史力采样点数。 |
| `--force-window-duration` | 与训练相同的历史力时间跨度。 |
| `--image-size` | 与训练相同的图像高、宽。 |
| `--camera-names` | 与训练相同且顺序一致的相机列表。 |
| `--batch-size` | 统计计算的批量大小；可按内存调整，不改变统计语义。 |
| `--num-workers` | DataLoader worker 数；`0` 最稳妥，可按机器调整。 |
| `--eps` | 标准差的数值下限，防止除零。 |

输出文件包含 `qpos/action/force` 的 mean/std，以及 action mode、窗口、相机、图像尺寸和 episode 路径元数据。

当前训练器会检查 stats 的 `action_mode`；在提供 validation list 时，还会要求 stats 记录的 episode 路径集合与训练集完全一致。但 chunk/window/camera/image/Imagenet 元数据并非被所有消费者自动逐项拒绝，因此仍须按本手册人工保持一致。

验收条件：命令成功退出，输出没有 `NaN/Inf`，保存路径正确，日志中的 episode 数与 `TRAIN_LIST` 一致。

## 6. Smoke training（必须）

Smoke training 用极少步数验证完整训练链路。它不用于判断模型最终性能。除了 `batch-size`、`max-steps`、保存频率和输出目录外，模型及数据语义参数应与正式训练一致。

Contact-CVAE 示例：

```bash
SMOKE_OUT="$EXP_ROOT/contact_cvae_smoke"
mkdir -p "$SMOKE_OUT"

PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list "$TRAIN_LIST" \
  --normalization-stats "$STATS" \
  --policy-variant force_aware_contact_cvae \
  --action-mode "$ACTION_MODE" \
  --train-latent-mode posterior \
  --train-contact-latent-mode posterior \
  --chunk-len "$CHUNK_LEN" \
  --image-size "$IMAGE_HEIGHT" "$IMAGE_WIDTH" \
  --camera-names ee_cam base_top_cam \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --force-window-duration "$FORCE_WINDOW_DURATION" \
  --lambda-force 0.1 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --beta-contact-max 5e-4 \
  --warmup-steps 20 \
  --max-steps 50 \
  --save-every 25 \
  --batch-size 2 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --seed 0 \
  --deterministic \
  --torch-num-threads 4 \
  --torch-num-interop-threads 1 \
  --device cuda \
  --output-dir "$SMOKE_OUT" \
  --log-csv "$SMOKE_OUT/train_log.csv" \
  2>&1 | tee "$SMOKE_OUT/console.log"
```

Smoke 专用参数说明：

| 参数 | 示例 | 说明 |
|---|---:|---|
| `--max-steps` | `50` | 足以覆盖取 batch、forward、backward、优化和保存流程。 |
| `--batch-size` | `2` | 降低首次运行的显存风险。 |
| `--warmup-steps` | `20` | 在短运行中实际覆盖 KL warmup；正式训练使用正式值。 |
| `--save-every` | `25` | 验证中间 checkpoint 保存逻辑。 |
| `--num-workers` | `0` | 降低首次定位 DataLoader 问题的复杂度。 |

其余训练参数的含义见下一节。

Smoke 验收条件：

- 日志显示 episode 能全部解析且 `dataset_length > 0`；
- `loss_total`、`loss_action`、`loss_force`、`kl_contact`、`loss_prior` 均为有限值；
- loss 可以反向传播并完成 50 个 optimizer steps；
- `train_log.csv`、中间 checkpoint 和最终 `checkpoint.pt` 均成功生成；
- 没有 normalization metadata mismatch、CUDA OOM 或数据读取异常。

如果 smoke 失败，应先修复问题，不要直接启动正式训练。

## 7. 正式 Contact-CVAE 训练

历史 100k Contact-CVAE 配置的标准化示例：

```bash
OUT="$EXP_ROOT/forceaware_contact_cvae_betac5e4_lp01_trajectory100k"
mkdir -p "$OUT"

PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list "$TRAIN_LIST" \
  --normalization-stats "$STATS" \
  --policy-variant force_aware_contact_cvae \
  --action-mode "$ACTION_MODE" \
  --train-latent-mode posterior \
  --train-contact-latent-mode posterior \
  --chunk-len "$CHUNK_LEN" \
  --image-size "$IMAGE_HEIGHT" "$IMAGE_WIDTH" \
  --camera-names ee_cam base_top_cam \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --force-window-duration "$FORCE_WINDOW_DURATION" \
  --lambda-force 0.1 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --beta-contact-max 5e-4 \
  --warmup-steps 2000 \
  --max-steps 100000 \
  --save-every 10000 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --seed 0 \
  --deterministic \
  --torch-num-threads 4 \
  --torch-num-interop-threads 1 \
  --device cuda \
  --output-dir "$OUT" \
  --log-csv "$OUT/train_log.csv" \
  2>&1 | tee "$OUT/console.log"
```

泛化评估路线应在上述命令中额外加入：

```bash
  --val-episode-list "$VAL_LIST" \
  --max-epochs 100 \
  --val-every-epochs 1 \
  --early-stop-min-epochs 10 \
  --early-stop-patience 8 \
  --early-stop-min-delta 0.005 \
  --early-stop-metric deploy_loss
```

此时 `--max-steps` 是安全上限，`--max-epochs` 是 epoch 上限，两者先到者结束训练。验证默认使用确定性部署路径；当前 Contact-CVAE 配置因 `lambda_prior > 0` 自动使用 prior。最佳验证模型写入 `checkpoint_best.pt`，最终状态仍写入 `checkpoint.pt`。全量训练路线不提供 validation list，而是使用前一阶段确定的最佳 epoch 数。

完整参数说明：

| 参数 | 示例 | 说明 |
|---|---:|---|
| `--episode-list` | `$TRAIN_LIST` | 实际参与训练的 episode 列表。 |
| `--normalization-stats` | `$STATS` | 必须由同一训练集合和同一数据参数计算。 |
| `--policy-variant` | `force_aware_contact_cvae` | 构造只使用 contact latent 的 ForceAware Contact-CVAE。 |
| `--action-mode` | `action` | 动作监督来源，必须与 stats 一致。 |
| `--train-latent-mode` | `posterior` | 训练时使用监督信息形成后验路径；Contact-CVAE 保持此设置。 |
| `--train-contact-latent-mode` | `posterior` | contact latent 训练模式；当前 CLI 只接受 `posterior`。 |
| `--chunk-len` | `10` | 未来动作和未来力监督长度。 |
| `--image-size` | `224 224` | 输入图像高、宽。 |
| `--camera-names` | `ee_cam base_top_cam` | 使用的相机及其固定顺序。 |
| `--force-window-len` | `20` | 历史力输入采样点数。 |
| `--force-window-duration` | `0.25` | 历史力覆盖秒数。 |
| `--lambda-force` | `0.1` | 未来力预测损失权重。 |
| `--lambda-prior` | `0.1` | contact prior 蒸馏/匹配损失权重。 |
| `--prior-loss-mode` | `mse_mu` | prior loss 使用先验和后验均值的 MSE；另一选项为 `kl_q_to_p`。 |
| `--beta-contact-max` | `5e-4` | contact latent KL 损失 warmup 后的最大权重。 |
| `--warmup-steps` | `2000` | KL beta 从零增加到最大值所用 optimizer steps。 |
| `--max-steps` | `100000` | optimizer 更新总步数，不是 epoch 数。 |
| `--save-every` | `10000` | 每 10k steps 保存一个中间 checkpoint；`0` 可禁用周期保存。 |
| `--batch-size` | `16` | 每次更新的样本数；需按显存调整。 |
| `--num-workers` | `0` | DataLoader worker 数；提高前先验证 HDF5 多进程读取稳定性。 |
| `--learning-rate` | `1e-4` | 优化器学习率。 |
| `--seed` | `0` | 模型与训练随机种子；DataLoader 使用 `seed + 1`。 |
| `--deterministic` | 开启 | 请求 PyTorch/cuDNN/cuBLAS 严格确定性；如算子不支持会显式报错。 |
| `--torch-num-threads` | `4` | PyTorch CPU intra-op 线程数；应按机器固定并记录。 |
| `--torch-num-interop-threads` | `1` | PyTorch CPU inter-op 线程数。 |
| `--device` | `cuda` | 训练设备；没有 CUDA 时可用 `cpu`，但会很慢。 |
| `--output-dir` | `$OUT` | 最终和中间 checkpoints 的目录。 |
| `--log-csv` | `$OUT/train_log.csv` | 每步 loss 指标的 CSV。 |
| `2>&1 \| tee ...` | `console.log` | 合并标准输出/错误并保存完整控制台记录。 |

`train_minimal.py` 还支持：

- `--beta-motion-max`：motion latent KL 最大权重；Contact-CVAE 不使用 motion posterior，通常无需设置；
- `--save-steps 3000 10000`：除周期保存外，在指定 steps 精确保存；
- `--imagenet-normalize`：启用 ImageNet 图像归一化，但必须与 stats 和评估保持一致；
- `--policy-variant force_aware_act` 或 `force_aware_motion_cvae`：训练其他模型变体，需要重新确认相应 latent 和 loss 配置。

上述 seed、deterministic 和线程参数只存在于 `train_minimal.py`。`train_act_baseline.py` 与 `train_contact_prior_stage2.py` 当前没有对应 CLI；跨 trainer 对比必须记录这一不对称，不能把输出目录中的 `seed0` 当作已经受控的证据。

正式训练预期输出：

```text
$OUT/checkpoint_step_00010000.pt
$OUT/checkpoint_step_00020000.pt
...
$OUT/checkpoint_step_00100000.pt
$OUT/checkpoint.pt
$OUT/train_log.csv
$OUT/console.log
```

当前 trainer 没有 resume CLI。启动长训练前应确认运行时长、保存空间和输出目录，避免覆盖已有实验记录。

### 7.1 训练过程监控

训练启动后，先查找 `train_minimal.py` 进程：

```bash
pgrep -af "python.*scripts/train_minimal.py"
```

输出中第一列是 PID，例如：

```text
411698 python scripts/train_minimal.py ...
```

将该 PID 传给训练监控脚本，并每 10 秒刷新一次：

```bash
TRAIN_PID=411698

watch -n 10 -d "python scripts/forceact_eta.py \
  --log $OUT/train_log.csv \
  --max-steps 100000 \
  --pid $TRAIN_PID \
  --recent-window 100"
```

参数说明：

| 参数 | 说明 |
|---|---|
| `watch -n 10` | 每 10 秒重新运行一次监控命令。 |
| `watch -d` | 高亮相邻两次输出发生变化的部分。 |
| `--log` | 正式训练正在写入的 `train_log.csv`。 |
| `--max-steps` | 训练目标步数，必须与正式训练的 `--max-steps` 相同。 |
| `--pid` | 当前训练进程的 PID，用于读取进程运行时间并估算 ETA。 |
| `--recent-window` | 计算近期 loss 均值所使用的最近行数，默认值为 100。 |

监控页面会显示进程状态、当前 step、完成比例、平均单步耗时、吞吐率、预计剩余时间、预计结束时间，以及主要 loss 的最新值和近期均值。

如果 `pgrep` 返回多个训练进程，应根据完整命令中的输出目录确认正确 PID，不要随意使用第一条。监控脚本只读取日志和进程状态，不会控制、暂停或终止训练；按 `Ctrl+C` 只会退出 `watch`，不会停止训练进程。

如需直接查看原始训练输出，可在另一个终端运行：

```bash
tail -f "$OUT/console.log"
```

## 8. 可选：validation 和 test 离线评估

只有选择泛化评估路线时，`val/test` 才具有独立集合含义。Contact-CVAE 可比较 zero、prior 和 posterior-oracle 三种 contact latent 模式。

先在 validation 上评估最终或候选 checkpoint：

```bash
PYTHONPATH=src python scripts/evaluate_contact_cvae_modes.py \
  --episode-list "$VAL_LIST" \
  --checkpoint "$OUT/checkpoint.pt" \
  --normalization-stats "$STATS" \
  --action-mode "$ACTION_MODE" \
  --batch-size 16 \
  --max-batches 500 \
  --chunk-len "$CHUNK_LEN" \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --force-window-duration "$FORCE_WINDOW_DURATION" \
  --image-size "$IMAGE_HEIGHT" "$IMAGE_WIDTH" \
  --camera-names ee_cam base_top_cam \
  --device cuda \
  --num-workers 0 \
  --posterior-mode mean \
  --seed 0 \
  --output-csv "$OUT/contact_mode_eval_val.csv" \
  2>&1 | tee "$OUT/contact_mode_eval_val.log"
```

配置和 checkpoint 确定后，保持同一参数，仅替换列表和输出文件进行 test：

```bash
PYTHONPATH=src python scripts/evaluate_contact_cvae_modes.py \
  --episode-list "$TEST_LIST" \
  --checkpoint "$OUT/checkpoint.pt" \
  --normalization-stats "$STATS" \
  --action-mode "$ACTION_MODE" \
  --batch-size 16 \
  --max-batches 500 \
  --chunk-len "$CHUNK_LEN" \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --force-window-duration "$FORCE_WINDOW_DURATION" \
  --image-size "$IMAGE_HEIGHT" "$IMAGE_WIDTH" \
  --camera-names ee_cam base_top_cam \
  --device cuda \
  --num-workers 0 \
  --posterior-mode mean \
  --seed 0 \
  --output-csv "$OUT/contact_mode_eval_test.csv" \
  2>&1 | tee "$OUT/contact_mode_eval_test.log"
```

评估参数说明：

| 参数 | 说明 |
|---|---|
| `--episode-list` | validation 或 test 列表。 |
| `--checkpoint` | 要评估的 Contact-CVAE checkpoint。 |
| `--normalization-stats` | 仍使用训练集合计算的 stats，不为 val/test 重算。 |
| `--max-batches` | 最多评估的 batch 数；完整评估时应确保覆盖全部样本。 |
| `--posterior-mode mean` | posterior oracle 使用后验均值，得到确定性结果；`sample` 会采样。 |
| `--seed` | 固定涉及随机性的评估结果。 |
| `--output-csv` | 保存逐样本 zero/prior/posterior 指标。 |

解释边界：zero 和 prior 是部署可用模式；posterior oracle 使用未来标签，只用于离线分析，不能宣称为可部署性能。

## 9. 可选：用全部数据重新训练最终部署模型

如果先通过 train/val/test 确定了模型与超参数，之后可以用 `all.txt` 重新计算全量 stats，并从头训练最终部署模型：

```bash
TRAIN_LIST="$ALL_LIST"
STATS="$EXP_ROOT/normalization_stats_action_all.pt"
```

然后重新执行第 5、6、7 节。必须明确标记这是一个新的模型；原来的 test 结果不能直接当作这个全量重训模型的独立 test 结果。

## 10. 实验归档和复现清单

每次正式实验至少保留：

- 原始 `all.txt`；
- 若使用划分：`train.txt`、`val.txt`、`test.txt` 和 `split_seed.txt`；
- `dataset_inspection.csv`、质量检测日志和动作模式检查日志；
- normalization stats 文件及其日志；
- 完整训练命令；
- `console.log` 和 `train_log.csv`；
- 最终及关键中间 checkpoints；
- validation/test 评估命令、日志和 CSV；
- Git commit、Python/PyTorch/CUDA 版本、GPU 型号和随机种子。

训练前最终一致性检查：

```text
[ ] episode 列表数量正确、无重复、路径全部存在
[ ] 数据结构、时间戳、长度和有限值检查通过
[ ] action_mode 的字段来源和物理语义已确认
[ ] 是否划分 train/val/test 已根据实验目的决定并记录
[ ] stats 只使用实际训练列表计算
[ ] action_mode/chunk/window/cameras/image preprocessing 在各步骤一致
[ ] smoke training 完整通过
[ ] 正式输出目录是新的或已明确确认可写
[ ] 日志、checkpoint 和剩余磁盘空间满足长训练要求
```

## 11. 历史 Contact-CVAE 100k 对照示例

历史模型使用：

```text
数据目录: mujoco_data/peg_hole_100
全集列表: outputs/peg_hole_100/all100.txt
episode 数: 100
训练样本位置数: 29977
划分: 无，全部 100 条用于 stats 和训练
stats: outputs/peg_hole_100/normalization_stats_action_all100.pt
policy: force_aware_contact_cvae
action_mode: action
chunk_len: 10
force_window: 20 points / 0.25 s
cameras: ee_cam, base_top_cam
image_size: 224 x 224
training: 100000 steps, batch size 16, learning rate 1e-4
```

它对应本文档的“全量最终训练路线”。新增的质量检测和 smoke training 是标准流程要求，用来降低数据问题和长训练失败风险，但不改变正式模型的数据语义配置。
