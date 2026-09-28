# 右臂实机原始 HDF5：real_robot_right_arm_raw_v2

此格式保存从 Tianji MCAP 提取的原生异步右臂数据，由实机读取器构建训练视图。

2026-09-28 用户纠正信号来源：`/tianji_marvin_teleop_bridge/estimated_joint_disturbance` 的 `effort` 是外力估计对应的七维关节力矩；`io_teleop/joint_states.effort` 是内部关节力矩，单独保存供后续研究，当前不参与模型输入、监督及归一化。重力／摩擦补偿细节标记为未核实，不从 ROS JointState 类型推断。v1 曾错误地将内部力矩作为外部估计；v2 读取器拒绝旧 schema，旧 Contact/Motion 结果不可作为外力感知模型结果。

## 转换原则

- 依每条消息的 `name`，按 `Joint1_R` 到 `Joint7_R` 提取右臂。
- 排除同一 `joint_cmd` topic 中不包含右臂机械臂关节的手部命令。
- 各流保留 MCAP 文件中的记录顺序及实际采样点。
- 不排序、不重采样、不插值、不补零、不补帧、不因采样间隔变长删除数据。
- 不调整时钟偏差；header/log/publish 三类时间戳均保留整数纳秒。
- 不改变数值精度：ROS float64 字段存为 HDF5 float64。
- 图像保存原始压缩字节，同时保存解码后的 RGB uint8；RGB 使用无损 LZF 压缩，不缩放、不重新 JPEG 编码。
- 缺少必需数组、右臂关节不完整或图像解码失败时停止转换，不静默跳过或制造替代值。
- 采样间隔异常和非有限数仅在质量报告中标记，不自动修复。
- 原始 MCAP 继续保留；输出只包含选定训练信号，不是整份 MCAP 的完整镜像。

## 字段

`N_state`、`N_disturbance`、`N_action`、`N_d455`、`N_d405` 相互独立，不要求相等。

| 路径 | 类型 / 形状 | 含义 |
|---|---|---|
| `observations/joint_pos` | float64 `[N_state, 7]` | 右臂实际关节位置，rad |
| `observations/joint_vel` | float64 `[N_state, 7]` | 右臂实际关节速度，rad/s |
| `observations/joint_external_torque` | float64 `[N_disturbance, 7]` | `estimated_joint_disturbance.effort` 外部关节力矩估计，N·m |
| `observations/joint_internal_torque` | float64 `[N_state, 7]` | `joint_states.effort` 内部关节力矩，仅保留，N·m |
| `action` | float64 `[N_action, 7]` | 右臂关节位置指令，rad |
| `actions/source_velocity` | float64 `[N_action, 7]` | 指令消息的原始 velocity 字段，保留用于追溯 |
| `actions/source_effort` | float64 `[N_action, 7]` | 指令消息的原始 effort 字段，保留用于追溯，不作为力矩观测 |
| `observations/images/<camera>/rgb` | uint8 `[N_camera, H, W, 3]` | RGB 图像 |
| `observations/images/<camera>/compressed` | 变长 uint8 `[N_camera]` | 原始 JPEG 字节 |
| `observations/images/<camera>/format` | UTF-8 `[N_camera]` | 原始 ROS 图像 format 字段 |
| `metadata/joint_names` | UTF-8 `[7]` | 固定右臂关节顺序 |
| `metadata/camera_names` | UTF-8 `[2]` | d455、d405 |
| `metadata/topic_inventory_json` | UTF-8 标量 | 源文件所有 topic 的名称、类型和消息数 |

每个 `timestamps/<stream>` group 对应 `state`、`disturbance`、`action`、`d455`、`d405` 之一：

- `header_ns`：int64 原始消息 header 时间戳。
- `log_ns`、`publish_ns`：uint64 MCAP 原始时间戳。
- `sequence`：MCAP 原始消息序号；未假定其是硬件帧计数器。
- `source_topic_message_index`：在对应 topic 按 MCAP 文件顺序读取时的零基序号，包含筛选前的手部消息位置。
- `frame_id`：原始 header.frame_id。
- group 属性记录 `source_topic` 和用户提供的 `nominal_rate_hz`。

位置、速度和内部力矩共用 `timestamps/state`；外部力矩独立使用 `timestamps/disturbance`，禁止按状态行号配对。两个相机各自独立，不提前配对。标称频率分别为图像 30 Hz、action 100 Hz、状态和扰动力矩 300 Hz。

外部力矩 dataset 必须记录 `source_topic`、`source_field=effort`、`timestamp_group=/timestamps/disturbance`；读取器核对这些来源信息及时间组来源。checkpoint 数据合同同样记录源话题、字段和独立时间轴，拒绝把旧模型重新解释为新信号模型。

根属性还记录源文件路径、字节数、SHA-256、schema、转换时间、补偿语义来源，以及 `resampled=False`、`imputed=False`、`training_ready=False`。

## 执行与验证

转换工具：`scripts/convert_mcap_right_arm.py`。

依赖为 numpy、h5py、opencv-python、mcap、mcap-ros2-support。新增依赖只安装到 `forceact` conda 环境；仓库不要求修改系统 Python。

在具备这些依赖的环境中执行：

```bash
python scripts/convert_mcap_right_arm.py /path/to/input.mcap /path/to/episode.hdf5
```

转换器拒绝覆盖已有输出。先写入 `episode.hdf5.partial`，第二次完整读取源文件，逐条核对以下内容后才发布正式 HDF5：

1. 全部选中消息的计数、时间戳、frame_id、sequence、原始 topic 序号。
2. 全部选中数值字段的 float64 字节一致性。
3. 全部原始 JPEG 字节和 format 一致性。
4. 全部 RGB 图像与重新解码源 JPEG 的逐像素一致性。
5. 转换前后源文件 SHA-256 未改变。

同时生成 `quality_report.json`，包含逐流频率、间隔分位数、重复/倒退时间戳、较长间隔的位置、数值形状和非有限数统计。间隔超过标称周期 1.5 倍只表示需要检查，并不证明传输丢包，也不触发数据修改。

MCAP 读取启用所读 chunk 的可用 CRC 检查；这不等于检查所有无关 topic 的解码或整个 DataEnd CRC。

## 读取阶段的时间对齐与训练视图

HDF5 保留原始异步记录，以下规则由读取器生成视图，不改写数据文件：

1. 使用各流 `log_ns` 作为记录器可用时间的近似。ROS `header_ns` 和 MCAP `publish_ns` 用于保留与诊断；尚未建立多设备时钟同步及真实在线到达时间映射。
2. 取每段开头 2 秒的 action `log_ns`，相对首条指令求模 10 ms 后计算圆均值，建立整段 100 Hz 网格。记录相位、样本数、圆均值集中度、首末锚点；集中度低于 0.5 时拒绝自动建立网格。该步骤属于离线准备，需要先取得初始窗口；实时冷启动应预先固定相位或等待此窗口完成。
3. 当前状态和两个相机分别选取 `log_ns <= anchor_ns` 的最近观测，状态最大年龄 20 ms、相机最大年龄 100 ms。原生观测不按行号配对，也不读取锚点之后的观测。
4. 每个动作槽最近邻匹配，容差 5 ms，每条真实指令最多使用一次。缺失槽保持原位置及 mask，不挤压时间，不复制末条动作补齐记录结尾。
5. 历史外力保留 `(anchor-200 ms, anchor]` 的全部实际 disturbance 样本，分为 20 个 10 ms 区间，每区间容量 8；超过容量明确报错，不静默截断。
6. 未来外力仅为监督：每个动作区间保留三个固定 300 Hz 槽，容差 1.666666 ms，且来源时间必须严格晚于锚点。缺失样本槽、空区间及区间端点分别保存 mask。内部力矩不作为缺失估计的替代值。
7. 有效锚点要求当前动作、状态、两个相机有效，并且历史外力非空。排除原因、动作匹配残差及逐样本原始索引可追溯。
8. 归一化仅统计显式训练段的原生 qpos、action、disturbance，每条消息一次；标准差下限 0.01。验证、测试、内部力矩和 mask 占位值不参与统计。

`RealRobotHighRateDataset` 构建张量视图：图像 `[B,2,3,H,W]`、当前位置 `[B,7]`、动作 `[B,K,7]`、历史力矩 `[B,20,8,7]`、未来力矩 `[B,K,8,7]`、端点力矩 `[B,K,7]`。转换层数值仍为 float64；模型张量使用 float32。mask 中 `True` 表示无效，张量中的零为配套 mask 的占位值，不是插补后的真实测量。

上述网格对齐不决定训练时如何随机选择窗口。20 条数据的已完成转换、固定划分和逐段来源见 [数据准备记录](../experiments/REAL_ROBOT_1MM_GAP_DATASET_CONVERSION.md)。
