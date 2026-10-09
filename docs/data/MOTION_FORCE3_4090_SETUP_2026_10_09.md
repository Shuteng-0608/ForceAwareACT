# Motion-CVAE 三维力训练：4090 环境准备

## 目标与训练配置

- 目标主机：`shutiao@192.168.8.105`。
- 仓库：`/home/shutiao/ForceAwareACT_workspace/ForceAwareACT`。
- 同步分支：`codex/real-robot-sdk-deployment`，保留原 `feat/rollout-force-hud` 分支。
- Conda：`/home/shutiao/miniconda3/envs/forceact`；依赖只在该环境安装。
- 数据：`/home/shutiao/Downloads/1mm-gap-10points_hdf5_raw_v2`，50条HDF5，固定30/10/10划分。
- 模型：Motion-CVAE，历史法兰坐标系三维力输入；动作和三维力辅助监督；motion posterior仅使用qpos和未来动作，验证/推理时motion latent置零。
- batch8，episodic_uniform，seed0，chunk100，双相机640×480，训练上限20000epoch。
- 每200次更新（50个采样epoch）验证，最低500epoch，patience8次验证，相对改善阈值0.005。
- 动作L1权重1，力端点L1权重1，高频力L1权重0.5，motion KL权重10；Motion-CVAE没有contact prior matching项。
- 重新计算训练集三维力归一化；不载入旧七维力矩归一化或Contact-CVAE权重。

## 手动启动

以下命令在4090电脑上执行。准备过程只进行有限步骤的环境验证，不自动启动正式训练。

```bash
cd /home/shutiao/ForceAwareACT_workspace/ForceAwareACT
bash scripts/training/train_gap50_motion_force3.sh
```

另一终端监控：

```bash
cd /home/shutiao/ForceAwareACT_workspace/ForceAwareACT
bash scripts/training/monitor_gap50_motion_force3.sh
```

输出：`outputs/1mm_gap_50_10points_v1/runs/motion_force3_seed0_b8`。

启动脚本根据当前用户HOME定位Conda和数据目录，也可通过`FORCEACT_DATA_ROOT`显式指定数据路径。正式目录已有训练记录时入口会拒绝覆盖。

## 准备记录

初步检查：RTX4090 24564MiB，驱动595.91.07；Python位于forceact环境，PyTorch2.10.0+cu128、torchvision0.25.0+cu128、h5py3.16.0、NumPy2.2.6，CUDA可用；ResNet18权重已缓存。

HDF5不进入Git仓库。数据传输后逐文件比对SHA-256；最终环境检查、数据完整性及完整网络batch8短测试结果另存于`outputs/4090_motion_force3_preparation_20261009/`。该记录目录不会作为正式训练输出。
