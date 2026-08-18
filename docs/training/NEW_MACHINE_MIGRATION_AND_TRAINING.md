# ForceAwareACT 新电脑迁移与训练指南

本文档用于把 ForceAwareACT 从已有电脑迁移到一台新的 Linux/WSL 训练机，并完成一次可验证、可复现的 GPU 训练，已于 2026-08-18 按当前源码复核。命令均从仓库根目录执行。源码和命令行 `--help` 是最终依据；新数据的语义检查见 [`NEW_DATASET_TRAINING_MANUAL.md`](../data/NEW_DATASET_TRAINING_MANUAL.md)，五种受控训练配置见 [`MODEL_TRAINING_AND_EARLY_STOPPING_MANUAL.md`](MODEL_TRAINING_AND_EARLY_STOPPING_MANUAL.md)。

## 1. 先明确哪些内容必须迁移

仅执行 `git clone` 不足以恢复训练环境。本仓库的 `.gitignore` 会排除 HDF5 数据、checkpoint、归一化统计、CSV、图像和整个 `outputs/`，因此迁移应分为三部分：

| 内容 | 必需性 | 推荐方式 |
| --- | --- | --- |
| Git 仓库及准确 commit | 必需 | 远端仓库 clone；未提交修改另存 patch 或提交到专用分支 |
| HDF5 数据集 | 必需 | `rsync`、移动硬盘或对象存储；传输后校验 SHA-256 |
| episode 列表、实验命令、日志 | 必需 | 列表优先使用仓库根目录相对路径；保留完整命令 |
| normalization stats | 建议留档，但换路径后应重算 | 复制用于审计，在新电脑按 train split 重新生成 |
| checkpoints、已有输出 | 继续评估时必需 | 单独复制 `outputs/` 或选定模型目录 |
| 原环境版本清单 | 强烈建议 | 保存 Python、驱动、PyTorch 和 `pip freeze` 输出 |

在旧电脑记录实验身份：

```bash
git rev-parse HEAD
git status --short
python --version
python -m pip freeze > environment.freeze.txt
nvidia-smi
```

若工作区有未提交修改，`git rev-parse HEAD` 不能代表完整源码。应先确认这些修改是否属于本次实验，再将其提交到实验分支，或用下面的方式单独保存：

```bash
git diff --binary > forceawareact-local-changes.patch
git diff --binary --cached > forceawareact-staged-changes.patch
```

新建但未跟踪的文件不会进入上述 patch，需要单独复制。不要把数据和 checkpoint 强行提交到 Git。

## 2. 在旧电脑生成数据校验清单

以下示例假设数据位于 `mujoco_data/peg_hole_100`。在旧电脑执行：

```bash
find mujoco_data/peg_hole_100 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > peg_hole_100.sha256
```

复制仓库外或被 Git 忽略的内容。示例中的目标地址需替换为实际用户名、主机和目录：

```bash
rsync -a --info=progress2 \
  mujoco_data/peg_hole_100/ \
  USER@NEW_HOST:/path/to/ForceAwareACT/mujoco_data/peg_hole_100/

rsync -a --info=progress2 \
  outputs/ \
  USER@NEW_HOST:/path/to/ForceAwareACT/outputs/
```

如果只准备从头训练，可以不复制旧 checkpoint，但仍应保留旧实验的 commit、split、命令和统计元数据。

## 3. 新电脑的系统与硬件前提

推荐条件：

- 64 位 Linux，或具备 NVIDIA CUDA 支持的 WSL2；
- Python 3.9 或更高版本；
- NVIDIA GPU 及能被 `nvidia-smi` 正确识别的驱动；
- 内存和显存能够容纳两路 `224x224` 图像、ResNet18 与训练 batch；
- 数据集和输出盘留有足够空间。当前仓库中的 `mujoco_data` 可能达到十几 GB，正式 checkpoint 和日志还会继续增长。

先做只读检查：

```bash
nvidia-smi
df -h .
free -h
```

CUDA Toolkit 的系统级安装不是本项目的固定依赖步骤；PyTorch wheel/conda 包通常自带所需 CUDA runtime，但主机 NVIDIA 驱动必须兼容。不要根据旧电脑的 CUDA 字样盲目安装相同 toolkit，应使用 PyTorch 官方安装选择器生成与新电脑驱动/平台相符的安装命令。

## 4. 获取完全一致的源码

有远端仓库时：

```bash
REPOSITORY_URL="https://example.com/your-org/ForceAwareACT.git"
RECORDED_COMMIT_SHA="replace-with-the-recorded-commit"

git clone "$REPOSITORY_URL" ForceAwareACT
cd ForceAwareACT
git switch --detach "$RECORDED_COMMIT_SHA"
```

先把两个变量值替换为实际仓库地址和第 1 节记录的 commit。

如果要继续开发而不是只复现实验，可从该 commit 新建分支。若旧电脑保留了 patch，在确认基准 commit 一致后应用并检查：

```bash
git apply --check /path/to/forceawareact-local-changes.patch
git apply /path/to/forceawareact-local-changes.patch
git status --short
```

验收：`git rev-parse HEAD` 与记录值一致；若应用了本地修改，`git diff` 与旧电脑的差异一致。

## 5. 创建隔离的 Python 环境

仓库没有锁定 Python/PyTorch 的精确版本，`pyproject.toml` 只声明最低层面的运行依赖。因此有两个合理目标：

1. **严格复现旧实验**：尽量使用旧电脑记录的 Python、PyTorch、torchvision、CUDA runtime 和依赖版本；
2. **在新硬件上重新训练**：选择新电脑支持的 PyTorch/torchvision 配对版本，再运行项目测试。此路线功能可用，但不能承诺浮点结果逐位一致。

conda 示例：

```bash
conda create -n forceact python=3.10 -y
conda activate forceact
python -m pip install --upgrade pip
```

或使用标准虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装彼此匹配、且适合新电脑 CUDA/CPU 平台的 `torch` 和 `torchvision`，再安装本项目：

```bash
python -m pip install -e ".[test]"
```

仅训练基础策略时，项目声明的核心依赖是 `h5py`、`numpy`、`torch` 和 `torchvision`。绘图/分析通常还需要 `pandas matplotlib scipy`；MuJoCo rollout 另需 `mujoco imageio imageio-ffmpeg`。这些扩展包不是开始训练的必要条件，可按任务安装：

```bash
python -m pip install pandas matplotlib scipy
python -m pip install mujoco imageio imageio-ffmpeg
```

## 6. 环境验收

确认解释器、核心包和 GPU 来自预期环境：

```bash
which python
python - <<'PY'
import h5py
import numpy
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
print("h5py:", h5py.__version__)
print("numpy:", numpy.__version__)
PY
```

GPU 训练要求 `CUDA available: True`。随后执行测试：

```bash
PYTHONPATH=src python -m pytest -q
```

至少再确认训练 CLI 可加载：

```bash
PYTHONPATH=src python scripts/train_minimal.py --help
PYTHONPATH=src python scripts/train_act_baseline.py --help
```

## 7. 放置和校验数据

建议仍将数据放在仓库内的 `mujoco_data/<dataset_name>/`，但不要提交到 Git。复制完成后，在仓库根目录使用旧电脑生成的清单校验：

```bash
sha256sum --check /path/to/peg_hole_100.sha256
```

所有行应显示 `OK`。如果迁移后数据目录名称改变，校验清单中的相对路径也要在对应父目录执行，或重新生成两端清单再比较。

episode 列表应优先写仓库根目录相对路径，例如：

```text
mujoco_data/peg_hole_100/run_000/episode.hdf5
mujoco_data/peg_hole_100/run_001/episode.hdf5
```

本项目会先按仓库根目录解析相对路径，再按列表文件所在目录解析。旧电脑的绝对路径在新电脑不存在时不会自动重定位，因此应重新生成列表，或将列表内容改为正确的相对路径。不要只做字符串替换后就跳过数据检查。

先检查列表：

```bash
wc -l configs/splits/peg_hole_100_train80.txt \
  configs/splits/peg_hole_100_val10.txt \
  configs/splits/peg_hole_100_test10.txt

while IFS= read -r episode; do
  case "$episode" in ''|'#'*) continue ;; esac
  test -f "$episode" || echo "missing: $episode"
done < configs/splits/peg_hole_100_train80.txt
```

再用项目实际读取器验证结构、时间戳、shape 和有效样本数：

```bash
PYTHONPATH=src python scripts/inspect_episode_collection.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output-csv outputs/migration_check/train_inspection.csv
```

新数据或来源不明的数据还必须执行采集质量和 action 语义检查，不能仅凭 HDF5 可打开就开始训练。

## 8. 在新电脑重新计算 normalization stats

这是迁移中最容易遗漏的一步。当前统计文件会记录用于计算它的**绝对 episode 路径**，训练器会验证这些路径与当前 train split 完全一致。即使数据内容相同，只要新电脑的仓库绝对路径不同，直接复用旧 `.pt` 就可能触发 `normalization stats episode_paths do not match the training split`。

因此推荐在新电脑用同一个 train split 和完全相同的数据语义参数重算：

```bash
mkdir -p outputs/peg_hole_100

PYTHONPATH=src python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --action-mode action \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --image-size 224 224 \
  --camera-names ee_cam base_top_cam \
  --output outputs/peg_hole_100/normalization_stats_action_train80.pt
```

以下参数必须在 stats、训练、评估和 rollout 中保持一致：train split、`action-mode`、`chunk-len`、force window、相机名称及顺序、图像尺寸，以及是否使用 `--imagenet-normalize`。stats 只能由 train split 计算，不能混入 validation/test。

## 9. 先运行最小 smoke training

以下命令使用 Contact-CVAE prior 配置，只跑 2 step，用来检查数据加载、前向、反向、CUDA 和 checkpoint 写入。它不是有效实验：

```bash
SMOKE_OUT="outputs/migration_smoke/contact_cvae_prior"
mkdir -p "$SMOKE_OUT"

PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
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
  --batch-size 2 \
  --num-workers 0 \
  --max-steps 2 \
  --save-every 1 \
  --device cuda \
  --output-dir "$SMOKE_OUT" \
  --log-csv "$SMOKE_OUT/train_log.csv"
```

smoke 验收条件：

- 终端出现有限的 `loss_total`、`loss_action`、`loss_force` 和 latent loss；
- 没有 CUDA OOM、NaN、路径或 HDF5 错误；
- 输出目录包含 `checkpoint.pt`、step checkpoint 和 `train_log.csv`；
- `nvidia-smi` 能观察到对应 Python 进程占用 GPU。

若显存不足，先将 `--batch-size` 降到 1。`--num-workers 0` 最适合首轮排错；确认稳定后再逐步增加 worker。改变 batch size 会改变优化过程，正式对照实验应统一并记录。

## 10. 正式训练示例

通过 smoke 后，为正式运行使用全新的输出目录，不要复用 smoke 目录。下面示例启用 train80/val10、epoch 验证和早停：

```bash
OUT="outputs/peg_hole_100/contact_cvae_prior_seed0"
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
  --learning-rate 1e-4 \
  --batch-size 16 \
  --num-workers 0 \
  --seed 0 \
  --deterministic \
  --max-steps 200000 \
  --max-epochs 100 \
  --val-every-epochs 1 \
  --early-stop-min-epochs 20 \
  --early-stop-patience 10 \
  --early-stop-min-delta 0.005 \
  --early-stop-metric deploy_loss \
  --validation-deployment-mode prior \
  --save-every 10000 \
  --device cuda \
  --output-dir "$OUT" \
  --log-csv "$OUT/train_log.csv" \
  --validation-log "$OUT/validation_log.csv" \
  2>&1 | tee "$OUT/console.log"
```

正式 batch size 应根据显存通过短运行确定。对比实验中不要只对某一个模型改变 batch、数据 split、stats 或预处理。其他四种规范配置及 ACT baseline 的完整参数以五模型训练手册为准。

训练会写入：

- `checkpoint.pt`：训练结束时的最终状态；
- `checkpoint_best.pt`：验证指标最优状态，存在 validation 时优先用于 test/rollout；
- `checkpoint_step_XXXXXXXX.pt`：按保存间隔生成；
- `train_log.csv`、`validation_log.csv` 和本例中的 `console.log`。

当前 `train_minimal.py` 和 `train_act_baseline.py` **没有 resume CLI**。已有 checkpoint 可用于评估/rollout，但不能通过命令行无缝续训。因此应使用稳定终端会话或作业调度器，并确保输出盘可靠；不要把频繁 step checkpoint 当作已经实现断点续训。

## 11. 训练期间与结束后的检查

运行期间关注：

```bash
nvidia-smi
tail -f outputs/peg_hole_100/contact_cvae_prior_seed0/console.log
df -h outputs
```

结束后应确认：

1. 日志中有明确的 `stop_reason` 对应正常上限或 early stopping；
2. loss 和验证指标均为有限值；
3. 有 validation 时 `checkpoint_best.pt` 存在；
4. Git commit、dirty status、精确命令、split 文件、stats、种子和依赖版本已经归档；
5. test 只在配置和 checkpoint 选择规则冻结后运行。

建议将下面的信息复制到每个实验目录的说明文件中：

```text
git_commit=
git_status=
python_version=
torch_version=
torchvision_version=
cuda_runtime=
gpu_model=
train_list=
val_list=
normalization_stats=
seed=
command=
```

## 12. 常见问题

### `torch.cuda.is_available()` 为 `False`

先确认 `nvidia-smi` 是否正常。如果正常但 PyTorch 看不到 CUDA，通常是安装了 CPU 版 PyTorch，或 PyTorch CUDA runtime 与驱动不兼容。重新使用 PyTorch 官方选择器安装匹配版本，不要在项目代码中绕过检测。

### `CUDA out of memory`

降低 `--batch-size`，关闭其他 GPU 进程，确认没有同时启动重复训练。必要时降低图像尺寸需要重新计算 stats，并把该变化视为新的实验配置。

### episode 路径不存在

检查列表是否含旧电脑绝对路径。推荐改为从仓库根目录可解析的相对路径，然后重新运行 collection inspection。

### normalization stats 与 train split 不匹配

迁移后路径发生变化时，在新电脑使用当前 train list 重算 stats。不要为了绕过保护而直接删改 `.pt` 内的 provenance。

### `No module named force_aware_act`

确认已在正确环境执行 `python -m pip install -e ".[test]"`，或者从仓库根目录使用文档中的 `PYTHONPATH=src` 命令。

### HDF5 worker 崩溃或训练卡住

先用 `--num-workers 0` 复现并运行 `inspect_episode_collection.py`。确认数据和单进程读取稳定后再增加 worker；网络文件系统上的 HDF5 并发读取尤其需要谨慎验证。

### 新电脑与旧电脑结果不完全一致

相同 seed 不保证跨 GPU、驱动、CUDA、PyTorch 和 torchvision 的逐位一致性。`--deterministic` 会加强 `train_minimal.py` 的确定性控制，但硬件/软件栈变化仍应被记录。ACT baseline 当前也没有与 `train_minimal.py` 完全对称的 seed/deterministic CLI，跨模型比较必须注明该限制。

## 13. 最终验收清单

- [ ] 源码 commit 和预期一致，未提交修改已说明；
- [ ] 数据集已独立复制并通过 checksum；
- [ ] episode 列表不含失效的旧电脑绝对路径；
- [ ] Python、PyTorch、torchvision、驱动和 GPU 检查通过；
- [ ] `pytest` 与训练 CLI 加载通过；
- [ ] 数据读取器检查和 action 语义检查通过；
- [ ] stats 在新电脑仅用 train split 重算；
- [ ] 2-step GPU smoke 产生 checkpoint 和有限 loss；
- [ ] 正式训练使用新输出目录并保存完整控制台日志；
- [ ] validation 使用真实部署 latent mode，最终选择 `checkpoint_best.pt`；
- [ ] commit、环境、命令、split、stats、seed 和输出已归档。

完成以上项目，即可认为“新电脑上的框架迁移、训练启动和可复现性证据链”成立。
