# 动作执行、安全保护与目标监测：阶段 1–3 交付

日期：2026-10-07。用户授权：实现、离线回放、软件校零和 2–3 分钟只读联调；禁止任何机械臂运动控制指令。用户再次确认静止、无接触，并确认 10points 是 **10 个不同孔位**。

## 交付边界

本版本只有 `replay` 和 `shadow`，`allow_motion` 必须严格为 `false`。没有机器人动作发送、使能、模式切换、清错、工具写入或急停后端。所有动作都是内存中的提议，所有停止都是日志事件。`physical_stop_confirmed=false`，不能以当前结果证明机器人能安全停止或模型能成功插入。

C++ 程序独占 SDK 并读取反馈；Python 模型进程进行 50 Hz 推理；独立 CPU 进程完成 100 Hz 聚合、规则检查和记录。此阶段使用 Python 规则实现验证行为，尚未实现后续运动阶段所需的 C++ 最终发送门和物理停止后端。Linux/Python 调度不提供硬实时保证。

新 PC 最终代码目录：`/home/shutiao/tianji_deployment/readonly_controller_20261007_v3`，环境 `tianji`。实测运行来自保留的 `readonly_controller_20261007_v2`；最终版另补充故障升级记录、异常退出清理及完整日志回放，使用同一段实测日志进行验证，未再次连接机械臂。

## 文件与接口

| 文件 | 新接口与职责 |
| --- | --- |
| `src/force_aware_act/control/contracts.py` | `Limits`、`Goal`、`ShadowConfig`、`Feedback`、`Pose`；单位、形状、配置及只读模式检查 |
| `src/force_aware_act/control/execution.py` | `TemporalExecutor`；绝对时间网格、可用性、过期动作跳过、三种聚合。`ActionLimiter`；位置、首动作误差、速度、加速度、跟踪误差及持续裁剪检查 |
| `src/force_aware_act/control/supervision.py` | `SafetySupervisor`；反馈/外力新鲜度、帧号、校准上下文、关节力矩保护。`GoalMonitor`；实际关节或 TCP 插入几何、稳定时间、进展与深度上限 |
| `src/force_aware_act/control/runtime.py` | `ShadowController` 状态机；`ShadowProcess` 独立进程、有界 IPC、故障锁定、后台文件缓冲 |
| `src/force_aware_act/inference/real_robot.py` | 新增 `prepare_snapshot`，把图像预处理、力历史打包、设备复制移出接收锁；原 `prepare_observation` 与 `predict` 接口保留 |
| `scripts/deployment/shadow_contact_readonly.py` | 可选 `--controller-config`，固定 50 Hz 查询与共同 100 Hz 时间网格；程序导入不连接设备 |
| `scripts/deployment/replay_controller_readonly.py` | 全数据数值审核、因果命令基线回放、真实预测聚合对比、完整实时事件/决策回放 |
| `configs/deployment/contact_right_shadow.json` | 默认只读配置；正式力矩/动作限值及物理目标未配置 |
| `configs/deployment/goals/*.json` | 十个孔位的独立配置模板，默认关闭目标完成判定，等待物理标定 |
| `tests/test_shadow_controller.py`、`tests/test_controller_replay.py` | 虚拟时钟故障注入与回放测试 |

张量合同保持不变：图像 `[1,2,3,480,640]`，qpos `[1,7]`，外力历史 `[1,20,8,7]`，模型动作 `[100,7]`。关节位置 rad，关节速度 rad/s，外部与内部关节力矩 N·m。每个执行时刻的提议 `[7]`。

外力输入及主要保护信号为 `m_EST_Joint_Force - fixed_bias`，对应 `estimated_joint_disturbance`。`m_FB_Joint_SToq` 单独保存，只有配置其独立限值时才参与补充保护。

## 执行与安全语义

- 100 Hz 网格共用单调时钟原点；预测锚点取 50 Hz 查询格。计算完成后不会重新从动作块第 0 项执行。
- 同一个执行时刻只融合已可用、仍在有效年龄内的预测；全零关节角是合法值。运行编号阻止上一轮的迟到预测进入新一轮。
- 默认影子聚合：按年龄指数衰减，时间常数 80 ms，最大预测年龄 200 ms。这两个值是软件比较起点，尚未通过接触任务验证。
- 支持 `newest`、`official`、`recency`。`official` 按旧到新候选次序应用 `exp(-0.01*i)`，偏重较早预测；`recency` 按真实年龄衰减。
- 正式动作约束缺失时，仍记录未约束提议，但 `constrained_proposal.action_rad=null`，不会伪造“已安全限幅”的结果。
- 每关节硬阈值即时判断；达到 soft 启动持续计时，下降到 release 才清零，回差区内保持计时。重复帧或重复外力时间不能刷新/推进相应计时。
- 开始前等待新鲜观测和预测；准备阶段的过期反馈丢弃并记录。已运行后，20 ms 状态/外力超时仍锁定故障，禁止自动恢复。
- 校准失效、控制器错误、上下文变更与硬超限在准备阶段同样会锁定。
- 故障时清除动作缓存，拒收普通预测；更严重故障可升级停止原因，并保留首次原因。人工停止优先级最高。
- 成功条件必须来自实际反馈，必须连续满足低速/稳定条件；未复核目标只输出 `UNVERIFIED_CANDIDATE`。真实 TCP 缺失、坐标系不匹配、位姿过期均不能完成任务。
- 软件报告正常完成最多为 `COMPLETED_SHADOW`，没有物理停止确认。模型输出和接触力增大均不能单独证明插入成功。

## 阶段 1：测试

最终相关测试共 **87 项通过**，包括：未来信息隔离、预测迟到/乱序、零动作、过期块、重复查询、限速/限加速度及约束不可行、硬/持续力矩、回差、时间倒退、帧号回绕、校准失效、内部与外部力矩分离、目标稳定/速度/坐标系/插入深度、任务停滞、故障优先于完成、启动数据门槛、运行故障不恢复、故障升级、独立进程看门狗和异常退出清理。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/stw/miniconda3/envs/forceact/bin/python -m pytest \
  tests/test_shadow_controller.py tests/test_controller_replay.py \
  tests/test_readonly_deployment.py tests/test_real_robot_inference.py -q
```

这组测试使用模拟 SDK，不访问硬件。未修改训练、数据集或归一化参数。

## 阶段 2：50 条示教审核与回放

读取全部 50 条 raw_v2 HDF5。按原有训练划分中的 30 条生成统计候选，在全部 50 条上回放；验证/测试数据没有用于确定候选限值和目标中心。原文件与标签保持不变。

累计检查 **519,194 条外力消息**；集成回放窗口处理 **515,262 条关节状态**。回放中的动作源明确为“查询时刻已经记录的最后一条指令保持”，不是模型预测，也没有读取未来动作作为推理输入。人为模拟 12 ms 返回延迟用于检查索引。真实模型回放来自阶段 3 的独立日志。

诊断候选公式见 `diagnostic_limits.json`，仅用于软件敏感性检查；没有安装为实机限值。

| 首次拟中止原因 | 记录数 |
| --- | ---: |
| 持续外部力矩超过 P99 统计候选 | 43 |
| 超出训练数据推得的速度包络 | 2 |
| 录制时间轴上的外力年龄超过 20 ms | 2 |
| 未触发拟中止 | 3 |

**这些计数不是数据成功率，也不能据此把成功示教判为失败。** 43 条触发说明当前统计候选会中止大量已知成功动作；需要结合机器人/工具/工装承受能力设计正式限值。两条时间问题基于 MCAP log_time 可用性代理，不能直接推断控制器当时的物理传感延迟。

全部数据七关节外部力矩绝对峰值约为 `[17.043, 5.846, 4.343, 9.939, 0.723, 5.794, 1.828] N·m`。这些是观测结果，不是安全建议值。

十个孔位分别提取训练记录最后 0.5 秒的终态候选，并保存源记录与关节范围。**30 个末段窗口都存在速度超过 0.01 rad/s 的样本**，因此不能把文件末段自动认定为稳定插入终态。候选没有转为正式物理目标；需标注真实完成时段或现场测量 TCP/孔位。

历史数据未记录原始外力、零偏有效性和控制器上下文。回放把已记录的校正扰动作为输入，并明确标注校准状态不可追溯，不能验证采集当时的校零是否始终有效。

```bash
/home/stw/miniconda3/envs/forceact/bin/python \
  scripts/deployment/replay_controller_readonly.py dataset \
  --data-root /home/stw/Downloads/1mm-gap-10points_hdf5_raw_v2 \
  --split configs/datasets/1mm_gap_50_10points_v2/split.json \
  --output outputs/controller_readonly_new_audit
```

## 阶段 3：新 PC 实时只读联调

使用已确认的 Contact checkpoint，FP32、batch=1、zero latent。直接读取 D455/D405 RGB，无额外 JPEG；沿用 checkpoint 的预处理。机器人 SDK 指纹仍为 `47ab2ab6e35cf899eae3978dfb50ee219e05b10f72aedd6bc9ba6a338a8b4011`。

链接符号检查确认实际 C++ 可执行程序的 SDK 引用只有 `CheckSDKTypeCompat`、`OnLinkTo`、`OnGetBuf`、`OnRelease`。Python 另调用只读版本查询 `OnGetSDKVersion`；没有机械臂写入接口。

第一轮 v1 保留为失败证据：模型完成 6,000 次预测，但监测器在准备阶段因约 78.7 ms 的旧反馈锁定，全部预测被拒收。随后缩短接收锁范围、增加明确的准备阶段门槛，再测试相同 120 秒。

| v2 指标 | 实测 |
| --- | ---: |
| 正式预测窗口 | 120.005 s |
| 成功预测 | 6,000 |
| 实际预测频率 | 49.998 Hz |
| 观测跳过 / 查询超期 | 0 / 0 |
| 完整查询循环 P99 | 14.004 ms |
| 模型前向平均 / P99 | 7.798 / 8.213 ms |
| 100 Hz 动作提议 | 12,000 |
| 执行漏拍 / IPC 队列溢出 | 0 / 0 |
| 监测调度延迟 P99 / 最大 | 1.073 / 1.094 ms |
| 反馈交付延迟 P99 | 0.492 ms |
| 正式窗口独立机器人帧 | 36,002 |
| 两路相机正式窗口帧数 | 3,596 / 3,600 |
| 机器人读取失败 / SDK 错误 | 0 / 0 |
| 机械臂控制调用 | 0 |

准备阶段另记录 1 条约 74.7 ms 的过期反馈和 1 次反馈间隙，没有把它们计作有效新鲜状态。正式窗口未触发故障；正常结束为 `ENDED_SHADOW`。SDK `release_ok=true`，相机与读取进程已退出，软件校零在断连后失效。

本轮正式力矩/动作限值未配置，所以 **12,000 条是未约束动作提议，完整限幅提议数为 0**。物理目标未选择/标定，实时目标结果为 `UNCONFIGURED`。力矩阈值和目标逻辑的验证来自合成故障与离线回放，不把这轮静止实验算作物理保护或目标成功验收。

真实事件日志重放了 **12,881 个监测时间格（含准备阶段）**，状态、动作提议、约束结果和目标结果 **0 项不一致**。

| 同一批真实预测的聚合比较 | 最大单步变化 rad | P99 单步变化 rad | 平均预测年龄 ms |
| --- | ---: | ---: | ---: |
| newest | 0.015283 | 0.009486 | 25.003 |
| official | 0.002662 | 0.001249 | 111.423 |
| recency，80 ms | 0.003353 | 0.001721 | 75.836 |

比较开始时有 3 个查询网格尚无已返回预测，未生成动作；正式运行仍为 12,000 个提议。该表仅展示静止观测下的提议平滑度与年龄，不能据此判定插入控制效果或最佳衰减参数。

## 新 PC 的复现方式

仅在操作者再次确认静止、无接触后开始软件校零；输出目录必须不存在：

```bash
conda activate tianji
cd /home/shutiao/tianji_deployment/readonly_controller_20261007_v3
PYTHONPATH=src python scripts/deployment/shadow_contact_readonly.py \
  --checkpoint /home/shutiao/tianji_deployment/models/contact_seed0_b8/best.pt \
  --reader ./robot_feedback_readonly --seconds 120 --query-hz 50 \
  --operator-confirmed-no-contact \
  --controller-config configs/deployment/contact_right_shadow.json \
  --output reports/shadow_new_run
```

回放已有日志无需连接设备，也不需要校零：

```bash
PYTHONPATH=src python scripts/deployment/replay_controller_readonly.py journal \
  --events ../readonly_controller_20261007_v2/reports/shadow_2min_v2/controller/events.jsonl \
  --config configs/deployment/contact_right_shadow.json \
  --output reports/replay_new_result.json
```

各孔位配置位于 `configs/deployment/goals/`。标定前保持 `kind=disabled`、`verified=false`。`joint` 判据需要真实目标关节与容差；`insertion` 判据需要孔口、插入轴、深度上下限、姿态、坐标系和实际 TCP 数据源。目前实现可接收带时间戳的 `Pose`，尚未接入/验证本机运动学库和工装标定。

## 原始证据与下一阶段边界

本地证据目录：`outputs/controller_readonly_20261007/`，包含测试 XML、各次数据回放、源码归档指纹、远端报告副本和交付清单。远端完整日志保留在 v1、v2 各自的 `reports/`，包括完整 `[100,7]` 模型输出、约 300 Hz 反馈和 100 Hz 拟执行记录。

运动阶段仍需确定并验证：七关节力矩及动作限值、十个孔位的物理目标、实际 TCP 数据源、控制模式与阻抗、SDK 停止语义、控制器通信看门狗。不得直接给当前影子程序添加一个 SDK 发送调用就开始运动；最终发送门与物理停止确认属于阶段 4，需新的明确实机运动授权。
