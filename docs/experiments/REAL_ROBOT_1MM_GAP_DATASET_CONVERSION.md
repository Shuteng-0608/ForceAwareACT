# 1 mm gap：20 条实机数据的转换与对齐记录

日期：2026-09-28。本记录固化已经完成的数据准备，范围为原始提取、读取时对齐、缺失 mask、训练侧归一化与记录级划分；训练实现、运行结果及早停分析另行记录。

## 数据与结果

| 项目 | 固定结果 |
| --- | --- |
| 原始数据 | `/home/stw/Downloads/1mm_gap_mcap`，20 条 MCAP，继续保留 |
| 当前派生数据 | `/home/stw/Downloads/1mm_gap_hdf5_raw_v2`，20 个 `episode.hdf5` |
| Schema | `real_robot_right_arm_raw_v2` |
| 任务标签 | 全部成功插入，由用户确认；不是从 MCAP 自动判定 |
| 条件分组 | x13y08、x14y08、x15y08、x16y08，每组 5 条 |
| 派生文件大小 | 26,702,509,706 bytes，约 26.70 GB |
| 状态 / 外力消息 | 201,542 / 201,463；两个流不等长 |
| 有效锚点 | 67,193 / 67,233，约 99.94% |
| 记录划分 | train 12 / validation 4 / test 4，seed `20260928` |
| 各侧有效锚点 | train 40,719 / validation 12,690 / test 13,784 |

每组固定分配 3 条训练、1 条验证、1 条测试。划分单位为完整采集记录，相邻窗口没有拆到不同侧。没有执行新的数据下载或转换；本次提交归档现有结果。

## 外力来源更正

外力估计使用 `/tianji_marvin_teleop_bridge/estimated_joint_disturbance` 的 `effort`，按每条消息的 `name` 提取 `Joint1_R`～`Joint7_R`，单位 N·m。保存为 `observations/joint_external_torque`，独立时间轴为 `timestamps/disturbance`。

`io_teleop/joint_states.effort` 是机械臂内部关节力矩，保存到 `observations/joint_internal_torque`，当前不参与模型输入、标签或归一化。`joint_cmd.effort` 仅保留。重力／摩擦补偿细节尚未核实，不额外补偿，也不把内部力矩补到缺失的外力估计中。

旧 HDF5 v1 和原始 MCAP 保留供追溯；v2 读取器拒绝旧 schema 或错误的外力话题、字段与时间组。旧内部力矩来源下的 Contact/Motion 训练结果不能作为外力估计模型结果。

## 处理链路

1. `scripts/convert_mcap_right_arm.py` 提取选定右臂消息、两个相机及五个原生时间流。原始数值 float64、JPEG 字节、RGB uint8 和整数纳秒时间戳保留；不排序、插值、重采样、补帧或过滤。HDF5 不是完整 MCAP 的镜像。
2. 转换器第二次完整读取选定源消息，核对数值字节、全部时钟、来源索引、JPEG 字节和 RGB 像素，确认源 SHA-256 不变后才发布 HDF5。20/20 条通过，外力无 NaN/Inf。
3. 读取器用 MCAP `log_ns` 作为可用时间近似，保留 ROS header 和 publish 时钟用于诊断。每段开头 2 秒 action 时间戳的圆均值确定 100 Hz 网格相位，并保存每段相位及集中度，供所有模型复用。
4. 状态和相机分别因果选取最近观测，年龄上限为 20/100 ms；动作最近邻容差 5 ms；外力固定 300 Hz 槽容差 1.666666 ms。历史为真实 200 ms disturbance 样本；未来动作与力矩只用于监督。
5. 未匹配槽与记录末尾保持 mask，不缩短缺失区间、不持有末条值；外力样本、区间和端点分别标记。归一化只统计 12 条训练记录的原生真实消息，每条一次，标准差下限 0.01。

完整字段、边界和张量布局见 [原始 HDF5 与对齐合同](../data/REAL_ROBOT_RAW_HDF5.md)。历史 `[B,20,8,7]`、动作 `[B,K,7]`、未来力矩 `[B,K,8,7]`；mask 的 `True` 表示无效。网格锚点生成与训练时随机选择窗口是两个不同步骤。

稳定相位审核原先使用首条指令相位时，记录了 64,189/67,243 个有效锚点；稳定相位结果为 67,193/67,233。本次正确外力来源版本的逐段相位和有效锚点数与已审核结果一致。

下一段 10 ms 的外力固定槽匹配率为 94.0083%，端点为 95.4087%。该统计每个有效锚点只检查紧随的一个区间，包含记录边界，不能解释成整段 100 步未来标签覆盖率或丢包率，也不同于 99.94% 的锚点可用率。

## 归档与身份校验

小型机器可读记录纳入 Git；MCAP、HDF5、图像及权重保留在本地：

| 文件 | 内容 |
| --- | --- |
| [`preparation_record.json`](../../configs/datasets/1mm_gap_20_v2/preparation_record.json) | 当前正确数据合同、20 条成功标签、汇总、归档来源及 SHA-256 |
| [`conversion_manifest.json`](../../configs/datasets/1mm_gap_20_v2/conversion_manifest.json) | 原转换 manifest 的逐字节副本；每条 MCAP/HDF5 指纹、验证、相位、划分和排除原因 |
| [`action_phase_audit_20260925.json`](../../configs/datasets/1mm_gap_20_v2/action_phase_audit_20260925.json) | 原相位审核的逐字节副本，保留旧网格对照 |
| [`force_source_quality_20260928.json`](../../configs/datasets/1mm_gap_20_v2/force_source_quality_20260928.json) | 外力来源更正完成时的逐字节质量记录 |
| [`dataset.json`](../../configs/experiments/1mm_gap_20_v1/dataset.json) | 本机 v2 数据位置及转换 manifest SHA-256 |
| [`split.json`](../../configs/experiments/1mm_gap_20_v1/split.json) | 固定 12/4/4 记录清单 |

当前转换 manifest SHA-256：`dc2b71d778ab66f17ef70dae053f688193f98612e69a13fe3d2a55486c2ce1ed`。

相位审核于外力来源更正前生成，其旧 `raw_v1` 力语义及 `gravity_compensated=true` 已作废；只使用其中动作相位、源身份和网格对照。当前外力合同以 v2 conversion manifest 为准。质量快照里的 `formal_training_started=false` 描述当时数据准备状态，不代表当前训练状态。配置目录名 `1mm_gap_20_v1` 保留已有路径兼容性，里面的 dataset 指向正确 v2 数据。

本次没有改写已有 HDF5 或历史 manifest。读取器和对齐算法的文件 SHA-256 与实际转换时记录一致；转换器仅清理一处尾随空格，文件 SHA-256 变化但转换逻辑不变，历史和提交时身份分别记录。准备脚本去掉训练/checkpoint 模块依赖，并停止生成 `optimization.json`，数据指纹算法及分组规则保持原样。

## 执行与验证

批量准备要求源目录含 `dataset_labels.csv` 与已审核的 `alignment_processing_manifest_v2.json`。脚本比对源 SHA-256、标签和相位，转换缺少的派生文件并生成 manifest；它会写入准备记录，不能作为只读检查命令。不要对当前完成数据重新执行以更新历史记录；复现时指定新的输出和配置目录。

```bash
conda activate forceact
python scripts/prepare_1mm_gap_dataset.py \
  --source-root /path/to/labeled_mcap \
  --output-root /path/to/new_raw_v2 \
  --experiment-dir /path/to/new_dataset_config \
  --workers 2
```

依赖为 numpy、h5py、opencv-python、mcap、mcap-ros2-support；测试还需 pytest、torch 和 torchvision。需要安装时只在 `forceact` conda 环境中安装。此次验证使用环境已有依赖，并以 `PYTHONPATH` 引用已有 `/tmp/1mm_gap_python_deps` 的 MCAP 解码库，没有安装新包。

```bash
PYTHONPATH=src:/tmp/1mm_gap_python_deps \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/stw/miniconda3/envs/forceact/bin/python -m pytest \
  tests/test_real_robot_conversion.py \
  tests/test_real_robot_data.py \
  tests/test_prepare_1mm_gap_dataset.py -q
```

测试覆盖真正编码的 MCAP 来源核验、独立外力消息数及时间轴、错误来源拒绝、缺失外力不回退、因果观测、稳定相位、固定缺失槽与端点 mask、训练侧原生统计、固定分组及准备入口独立性。提交前还在仅包含暂存内容的独立目录运行这些测试，检查提交自身不依赖工作区未提交的训练改动。

本次上述数据专项测试 35 项通过（2.54 秒）。独立暂存副本进一步运行原有 `tests/test_act_aligned_high_rate_training.py` 与 `tests/test_high_rate_force.py`，合计 57 项通过（5.83 秒）。归档 JSON 逐字节／SHA-256、成功标签、实际 12/4/4 划分、相位配置、总数及提交时代码身份核对通过；没有重跑真实数据转换或启动训练。
