# 右臂模型的纯 SDK 部署开发计划

日期：2026-10-06（Asia/Shanghai）。状态：准备阶段，实时设备适配器尚未实现。

训练与离线推理基线：`5d54762b515939ae0892c33a205632b845fa59ee`。
开发分支：`codex/real-robot-sdk-deployment`。

## 1. 本阶段目标

用户计划在另一台 Linux、RTX 4090 电脑上部署；相机和机械臂可以有线连接。
首先完成环境检查、实时数据获取与审核，再接入模型生成预测日志。
设备 SDK 直接接入作为开发方向，ROS 不作为推理程序的必需依赖。
本计划没有启动设备、训练、推理或机器人运动；硬件控制是后续独立阶段。

复用现有组件：

- `RealRobotObservationBuffer`：独立图像、关节位置和外力矩缓存。
- `RealRobotPolicyAdapter`：checkpoint、冻结统计量及统一预处理。
- `TimestampActionChunkBuffer`：100 Hz 动作时间网格与迟到结果处理。
- `scripts/replay_real_robot.py`：离线数据与在线缓存输入一致性检查。

上述实现见 [real_robot.py](../../src/force_aware_act/inference/real_robot.py)。
数据语义见 [实机训练指南](../data/REAL_ROBOT_TRAINING.md) 和
[实机模型架构](../architecture/REAL_ROBOT_MODEL_ARCHITECTURE.md)。

## 2. 已核实的采集实现

以下是当前本机源码的检查结果；当前文件不能独立证明采集当时的全部运行参数。

### 2.1 相机

参考：`/home/stw/tianji_ws/src/dual_realsense_launch/launch/dual_rs.launch.py`。

| 项目 | 当前配置 |
| --- | --- |
| D455 | 序列号 `261822304024`，全局相机 |
| D405 | 序列号 `260322275961`，右侧腕部相机 |
| 图像 | 两路 RGB8，640×480，30 Hz；深度 Z16 同规格同时开启 |
| 曝光 | 自动曝光开启；D455 在 RGB sensor，D405 在 depth module 配置 |
| 启动 | 先 D455，3 秒后启动 D405 |
| 处理 | 深度对齐、点云、空间/时间滤波和 IMU 关闭；`enable_sync=false` |
| 记录 | 彩色图保存为 compressed 图像；模型使用 JPEG 解码后的 RGB |

相机 README 仍引用 D405 `260322275294`，与当前 launch 中的右侧设备不一致。
新电脑必须枚举并按实物序列号绑定，不依赖枚举次序。
直接 SDK 读取原始 RGB 会改变 JPEG 编解码环节，需要对照画面和模型输出；
首轮复现先保持 RGBD stream profile，关闭无用流作为单独可记录的后续优化。
元数据 JSON bridge 只转存元数据，不处理像素；新接入应直接记录设备元数据。

### 2.2 机械臂与外力

参考包：`/home/stw/tianji_ws/src/tianji_io_teleop_bridge`。
SDK 本机来源：`/home/stw/TJ_FX_ROBOT_CONTRL_SDK/DEMO_C++`。

`MarvinSdkAdapter` 通过 `OnLinkTo` 连接、`OnGetBuf(DCSS*)` 读取反馈。
右臂为 `m_Out[1]`，模型关节顺序为 `Joint1_R` 到 `Joint7_R`。

| 信号 | SDK 字段及转换 | 模型用途 |
| --- | --- | --- |
| qpos | `m_FB_Joint_Pos * pi / 180` | 输入，rad |
| qvel | `m_FB_Joint_Vel * pi / 180` | 仅诊断，rad/s |
| 内部力矩 | `m_FB_Joint_SToq` | 仅诊断，不代替外力 |
| 原始扰动 | `m_EST_Joint_Force` | 保存来源值 |
| 外力输入 | `m_EST_Joint_Force - joint_bias` | 七维外部关节力矩，N·m |

桥接的 `JointTorqueZero` 从静止窗口计算固定残余偏置，需人工确认无接触；
它不在运动中自动去零，也不额外做重力/阻抗补偿。SDK 内部估计算法未在此实现。
重连、对应臂控制状态或反馈工具参数变化等情况会清除零偏，必须保留这些失效规则。

配置差异需在实施前确认：`bridge.yaml` 为 1 秒窗口、0.05 N·m 标准差阈值；
`bridge_both_test.yaml` 为 2 秒、0.15 N·m。二者的速度和位置稳定阈值分别为
0.01 rad/s、0.005 rad。实际采用值需要来自采集启动命令/参数记录。

本次核对全部 50 条转换文件的源 topic 清单，均含最终扰动估计，
均没有原始扰动、零偏和校零状态话题。不能用这些文件完整还原历史校零过程。
新程序应同时保存 raw、bias、corrected、校零状态、修订号及工具/控制模式上下文。

### 2.3 帧号、时钟与连接所有权

- 当前桥接配置为 300 Hz 读取/发布，控制更新上限 100 Hz。
- `feedback_frames_are_fresh` 允许重复发布未超时的控制器帧；300 Hz 消息频率
  不证明 300 Hz 独立测量。记录 `m_OutFrameSerial`，统计重复率和真实帧更新率。
- 当前关节消息 header 在 SDK 读取后使用本机 ROS `now()`，状态与扰动同帧同戳；
  训练按各消息 MCAP `log_ns` 构造可用性时间轴。
- 直接接入保留源时间戳及其时钟域、帧号和本机单调接收时间。解码/IPC/推理耗时
  分别记录；不同设备时钟未经验证不能直接比较。
- 实时接入需明确记录到现有缓存的时间映射及其版本，保持因果可用性规则；
  不把设备时间简单改名为 `log_ns`，不改写 checkpoint 中冻结的数据合同。
- SDK 文档要求独占连接。由一个进程管理连接，后续采集与控制共享该所有者。
  对照 ROS/SDK 数据时，从同一份反馈分流，不能同时启动两个 SDK 客户端。

## 3. 计划中的结构与输入合同

```text
RealSense SDK -> 相机采集器 ----------------------+
                                                +-> 观测缓存 -> 推理进程 -> 预测日志
Marvin SDK -> 右臂采集器 -> 单位转换/零偏校正 ----+
                       -> 原始反馈及质量日志
```

先用 C++ 复用已验证的 Marvin SDK 结构体和 ABI 检查，抽离 ROS 消息类型依赖；
再选择 Python binding 或本机 IPC 接入模型。实现必须使设备接收不被模型前向阻塞。
相机保留最新可用帧；外力历史必须保留窗口内真实样本，不能只留最后一个数值。

| 张量/数据 | 现有合同 |
| --- | --- |
| 两路图像 | 顺序 `d455, d405`；单路 uint8 RGB `[480,640,3]` |
| 模型图像 batch | `[1,2,3,480,640]`，复用 checkpoint 指定预处理 |
| qpos | 七维 rad；最大接收年龄 20 ms |
| 图像时效 | 最大接收年龄 100 ms；另记录物理采集/传输延迟诊断 |
| 外力历史 | 最近 200 ms，`[1,20,8,7]`，保留真实时间与缺失 mask |
| 输出动作 | `[100,7]`，绝对关节位置 rad，每格 10 ms |
| 推理 | batch=1、FP32、Contact `zero` latent，先与现有评估保持一致 |

查询频率根据预热后的端到端延迟确定，与动作执行 100 Hz 分别调度。
无新反馈、过期数据、非有限值、队列溢出和校零失效必须显式记录；
不把未知外力填成有效零值，不通过无记录的插值或去重改变训练输入分布。

## 4. 分阶段开发与验收

| 阶段 | 交付 | 验收 |
| --- | --- | --- |
| P0 环境与迁移 | 只读环境检查脚本、依赖清单、源码/模型指纹清单 | 确认 Linux 发行版、CPU/驱动、4090 型号；SDK ABI/版本兼容；模型可 strict load |
| P1 设备只读接入 | 两路相机、Marvin 反馈采集器与版本化数据包 | 绑定正确设备；角度/单位正确；能追踪帧号、时钟、原始扰动与零偏 |
| P2 实时数据审核 | 连续 10–15 分钟报告和短段原始记录 | 统计频率、独立帧率、缺失、P50/P95/P99 延迟、历史覆盖与积压；满足模型合同 |
| P3 实时预测 | 冻结 checkpoint 的在线推理与预测日志 | 回放输入等价；记录预热后准备、GPU、IPC及端到端延迟和超时比例 |
| P4 执行集成 | 控制接口、关节约束、退出条件和人工接管 | 单独设计审查及实机授权后开展；任务成功率需闭环验证 |

P1/P2 不提供运动发布入口。单元测试使用模拟 SDK 和录制样本，
不得通过 import 或测试 fixture 自动连接控制器、修改工具参数或切换模式。
新增模块覆盖异常帧、重复帧、断连/重连、校零失效、异步消息和迟到预测。
GPU 回放/实时性能属于新电脑验收，已有 CPU 回归不能替代该验收。

## 5. 迁移清单与待确认项

- 完整源码及明确 Git commit；开发基线已包含此前未跟踪的真机模块。
- Contact 候选：`outputs/1mm_gap_50_10points_v1/runs/contact_seed0_b8/best.pt`。
- ACT 对照：`outputs/1mm_gap_50_10points_v1/runs/act_seed0_b8/best.pt`。
- 模型文件、HDF5、运行日志和厂商动态库单独复制并校验 SHA-256，不提交 Git。
- 推理沿用 checkpoint 内归一化参数；新电脑不重新估计归一化。
- 初期复制少量代表性 HDF5 供回放，无需迁移全部训练集。
- Marvin SDK 头文件/动态库、ABI 与版本校验、工具配置和必要的依赖库。
- RealSense SDK/固件、相机序列号、当前传感器选项和安装姿态。
- 新电脑 Linux 具体发行版及版本、NVIDIA 驱动、CPU/USB 控制器/网卡情况待确认。
- 采集时实际桥接配置、校零步骤、控制模式和工具参数待确认；默认值不视为运行证据。

Contact 训练完成不等于真机验收完成。已有验证中其最佳动作 MAE 尚未超过保持当前
关节位置的基线；P3 用于接口、时序及性能检查，插入能力须另行评估。

## 6. 基线验证记录

2026-10-06 在原电脑对本次提交的训练/离线推理代码运行：

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src:/tmp/1mm_gap_python_deps \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/stw/miniconda3/envs/forceact/bin/python -m pytest tests -q --import-mode=importlib --maxfail=5
```

结果：**913 passed，2 skipped，64.52 秒**；暂存差异格式检查通过。
这里的临时 MCAP 依赖路径仅用于记录本机复现方式，新电脑应建立自己的依赖环境。
本开发计划只增加文档，不新增类、函数或张量布局，也不改变模型、数据或控制代码。
