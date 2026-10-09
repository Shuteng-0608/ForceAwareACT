# Motion-CVAE 三维力训练：4090 环境准备

## 目标与训练配置

- 目标主机：`shutiao@192.168.8.105`。
- 仓库：`/home/shutiao/ForceAwareACT_workspace/ForceAwareACT`。
- 同步分支：`codex/real-robot-sdk-deployment`，保留原 `feat/rollout-force-hud` 分支。
- Conda：`/home/shutiao/miniconda3/envs/forceact`；依赖只在该环境安装。
- 数据：`/home/shutiao/Downloads/1mm-gap-10points_hdf5_raw_v2`，50条HDF5，固定30/10/10划分。
- 模型：Motion-CVAE，历史法兰坐标系三维力输入；动作和三维力辅助监督；motion posterior仅使用qpos和未来动作，验证/推理时motion latent置零。
- batch8，episodic_uniform，seed0，chunk100，双相机640×480，训练上限20000epoch。
- 每200次更新（50个采样epoch）验证，最低3000epoch，patience20次验证，相对改善阈值0.001。
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

输出：`outputs/1mm_gap_50_10points_v1/runs/motion_force3_seed0_b8_es3000_p20_r001`。

启动脚本根据当前用户HOME定位Conda和数据目录，也可通过`FORCEACT_DATA_ROOT`显式指定数据路径。正式目录已有训练记录时入口会拒绝覆盖。

## 准备记录

### 早停配置更正

最初准备的脚本及此前命令继承了旧50条实验的500/8/0.005，偏离用户此前确认的3000/20/0.001。已将后续Motion训练恢复为宽松早停，并使用新输出目录；同时更新了Contact三维力的从头训练示例。batch8、数据划分、每50epoch验证、损失权重及最大20000epoch不变。

本机`contact_force3_seed0_b8`错误地使用500/8/0.005，已按用户要求删除整个训练输出目录，不保留日志或checkpoint，也不纳入后续模型对比。若需要同一宽松训练控制下的Contact/Motion对照，应使用更正后的新目录重新训练Contact。这里未自动启动训练，也未更改已创建训练的配置。

初步检查：RTX4090 24564MiB，驱动595.91.07；Python位于forceact环境，PyTorch2.10.0+cu128、torchvision0.25.0+cu128、h5py3.16.0、NumPy2.2.6，CUDA可用；ResNet18权重已缓存。

HDF5不进入Git仓库。环境检查及完整网络batch8短测试结果保存于`outputs/4090_motion_force3_preparation_20261009/`。该记录目录不会作为正式训练输出。

### 已完成检查

- 本机336项训练/推理/部署相关回归测试通过；新增Motion三维力CLI保存、加载和续训检查也通过。
- 4090上148项训练/推理相关回归测试通过，日志为`pytest.log`。
- 4090完整Motion-CVAE、双相机640×480、chunk100、batch8、三维力，合成输入完成3次优化器更新；loss及梯度全部有限，力编码器和motion posterior梯度非零，推理latent为全零。
- 张量：images `[8,2,3,480,640]`，历史力 `[8,20,8,3]`，未来力监督 `[8,100,8,3]`，动作 `[8,100,7]`。
- CUDA峰值allocated 4,451,516,416字节（约4.15GiB），reserved 5,651,824,640字节（约5.26GiB）。此数据仅用于确认完整模型可运行，不代表真实数据训练速度或模型效果。
- GPU结果文件为`gpu_synthetic_smoke.json`，记录代码提交`a7a351c1c28a9579be7249470a407ad59adc99eb`。随后仅补充本说明和数据校验清单。
- 已有forceact可编辑安装直接指向当前仓库，无需安装或更新依赖。两个shell入口的语法及CLI帮助检查通过。

### 数据由用户传输

用户要求停止代理的数据传输并自行完成。停止时目标目录约2.4GiB，包含未完成文件；真实50条数据尚未齐备，未进行真实数据训练或完整性验收。

将本机整个`/home/stw/Downloads/1mm-gap-10points_hdf5_raw_v2/`复制到4090的`/home/shutiao/Downloads/1mm-gap-10points_hdf5_raw_v2/`。必需内容为固定划分中50个episode子目录及各自的`episode.hdf5`；原始MCAP、旧训练输出和旧模型权重不需要传输。侧文件可随整目录复制，但训练归一化仍从训练集重新计算。

仓库提供`configs/datasets/1mm_gap_50_10points_v2/hdf5.sha256`，来源为最近50条Contact-CVAE训练记录中的HDF5 SHA-256，已确认本机现有50文件大小与该记录一致。传完后可在4090执行：

```bash
cd /home/shutiao/Downloads/1mm-gap-10points_hdf5_raw_v2
sha256sum -c /home/shutiao/ForceAwareACT_workspace/ForceAwareACT/configs/datasets/1mm_gap_50_10points_v2/hdf5.sha256
```

应为50个文件全部`OK`。未完成文件需要续传或校验后覆盖，不能按“存在即跳过”处理。校验通过后执行上方手动启动命令。
