# 新 PC：观测、推理、FK 与球形目标监测只读联调

日期：2026-10-07。范围：用户批准的同步、离线验证及约两分钟实时只读验证；用户本轮已确认右臂静止、无接触，可以软件校零。

## 结论

**本次 120 秒联调通过。** 新 PC 同时运行双相机观测、右臂状态/外力读取、Contact 模型 50 Hz 推理、实际关节反馈 FK、球形目标监测及 100 Hz 动作提议记录。6,000 次预测全部完成，实际频率 49.9966 Hz；观测跳过、推理超期、监测漏周期均为 0。

全程 `commands_sent=false`、`motor_control_api_calls=0`、`allow_motion=false`，动作后端为 `journal_only_no_motion_api`。本次未发送运动、模式、使能或停止指令。结束时 SDK 正常释放，采集/推理进程均已退出；软件校零在断开读取连接后按设计失效，不能作为下次运行的有效校零复用。

这验证了当前只读链路和计算预算。真实运动中的目标到达、负载变化和物理停止行为仍未验证；正式关节力矩与运动限值仍为 `unconfigured`。

## 部署位置和版本

- 新 PC：`shutiao@192.168.8.105`，Ryzen 9 9950X、RTX 4090。
- 环境：`/home/shutiao/miniconda3/envs/tianji`，Python 3.10.22、NumPy 1.26.4、Torch 2.12.1、torchvision 0.27.1、pyrealsense2 2.56.5.9235。
- 本次目录：`/home/shutiao/tianji_deployment/readonly_controller_20261007_v4`。此前版本保留。
- 既有权重：`/home/shutiao/tianji_deployment/models/contact_seed0_b8/best.pt`。
- 权重 SHA256：`a5999aa40c8a39111366fbbcd9ae99301e70b302a435cffd5b1f63bfc194a855`。
- 验证过的 SDK SHA256：`47ab2ab6e35cf899eae3978dfb50ee219e05b10f72aedd6bc9ba6a338a8b4011`；运行版本 `100343013`。
- 只读参数沿用 `bridge_both_test.yaml`，源文件 SHA256：`32c04d3f49a8e2ff19ac266d291fdfb5806f266b3ad20d6a29d6d31a3a2a7c88`。
- D455：`261822304024`；D405：`260322275961`。
- 机械臂网络：`192.168.1.190`，新 PC 经 `enp8s0`、源地址 `192.168.1.156` 访问。

新 IP 的 SSH 主机公钥与原已信任的 `192.168.1.107` ED25519 公钥一致，使用原主机别名严格验证后连接；没有关闭主机密钥验证，也没有修改用户 `known_hosts`。

本轮在 `tianji` 环境内增加 `scipy==1.15.3`，用于已有离线对照测试；没有更换 Torch/CUDA。在线 FK 只依赖 NumPy。读取器重新编译后，动态符号检查中的 SDK 依赖仅为类型兼容检查、连接、读取和释放；Python 启动时另做只读 SDK 版本检查。

## 实际配置和输入语义

- 推理 50 Hz，动作提议/保护调度 100 Hz，实际关节反馈约 300 Hz，相机各约 30 Hz。
- 外力取 SDK `m_EST_Joint_Force`，减去本轮静止窗口均值；与训练用 `estimated_joint_disturbance` 的语义一致。内部关节力矩未替代外力输入。
- 保持既有原始 RGB 读取和训练对应的模型预处理；没有增加 JPEG 重编码。
- 当前部署的单调接收时钟映射和输入 `log_ns` 语义保持不变。
- FK 在独立 CPU 监测进程中按实际反馈计算，输入关节角 `[7]` rad，生成法兰 Pose 的 position `[3]` m、quaternion `[4]`。没有使用预测关节角判断实际到达。
- 同步十个孔位配置。本次静止实时检查显式选用 `configs/deployment/goals/x13y07.json`：球心 `[0.5171237737981482, -0.31671363124769164, 0.25007723345747657]` m，半径 3 mm，连续占据 500 ms。
- 坐标系为 `marvin_m6_ccs40_right_base`，参考点为法兰；这是已接受的粗略完成区，`coarse_completion_accepted=true`、`verified=false`。
- 20 ms 状态/外力/位姿新鲜度、100 ms 图像新鲜度以及采样间断检查保持启用。关节力矩和运动限值尚未填入，因此 12,000 个输出均为未获物理执行许可的日志提议。

## 校零

610 个样本，跨度 2.0301 秒。最大关节速度 0.001795 rad/s，最大位置范围约 1.5581e-6 rad，最大外力标准差 0.08917 Nm；均通过既有静止校零门槛（0.01 rad/s、0.005 rad、0.15 Nm）。

七关节零偏 Nm：

```text
[-0.10104772, 0.32122292, 0.62216177, 1.20681997,
  0.24122087, 0.43895727, -0.12815033]
```

`calibration_valid_for_completed_run=true`；结束断连后 `calibration_valid_at_end=false`、原因为 `reader_disconnect`，这是预期的有效期管理。

## 实时测量

有效推理段 120.0081 秒；监测进程总时长约 128.81 秒包含加载/预热等待。表中的推理/数据流频率取有效推理段，读取器频率另含校零阶段。

| 项目 | 本次结果 |
|---|---:|
| 模型预测 | 6,000 次，49.9966 Hz |
| 模型前向 | 平均 7.802 ms，P99 8.454 ms |
| 完整推理循环 | P50 12.515 ms，P99 13.737 ms |
| 观测跳过 / 推理超期 | 0 / 0 |
| 有效段右臂新鲜反馈 | 36,003 条，约 300.00 Hz |
| D455 / D405 帧数 | 3,595 / 3,599，约 29.96 / 29.99 Hz |
| 读取器整体频率 | 299.9505 Hz |
| 读取失败 / 非有限值 / 帧号倒退 | 0 / 0 / 0 |
| 100 Hz 提议 | 12,000 条；漏周期 0 |
| 监测调度迟到 | P99 1.031 ms，最大 2.099 ms |
| 正式运行反馈交付延迟 | P99 1.375 ms，最大 2.535 ms |
| FK，36,076 次 | 平均 0.1245 ms，P99 0.2555 ms，最大 0.4969 ms |
| 目标判断，47,998 次 | 平均 0.02243 ms，P99 0.05620 ms，最大 0.2216 ms |
| IPC 队列溢出 / 首个保护故障 | 无 / 无 |
| 结束状态 | `ENDED_SHADOW`，`normal_readonly_end` |

目标判断既在反馈到达时检查，也在调度周期内检查，因此次数多于 FK。两项 P99 来自不同样本集合，不相加当作组合 P99。

完整统计保留了一次启动阶段 76.427 ms 的反馈交付延迟；该过期样本被启动保护丢弃，记录为 `startup_stale_feedback_dropped=1`，启动间断记录 1 次。以首次 `RUNNING_SHADOW` 为界，运行阶段 35,998 条监测反馈中超过 20 ms 的为 0。这个分阶段口径与采集线程的 36,003 条有效段计数边界略有不同。

静止期间实际法兰距目标中心 **203.332587～203.332984 mm**。12,000 个运行周期均为 `RUNNING`、球内周期数为 0，符合当前机械臂远离目标的实际情况。本次没有让机器人进入球内，不能将其当作真实插入成功或物理停止测试。

作为历史参考，先前未配置球形目标的 v2 只读运行：49.998 Hz，完整循环 P99 14.004 ms；本次为 49.997 Hz、13.737 ms。两次不是受控 A/B 实验，不能据此声称 FK 使推理更快，只能说本次加入 FK 与目标判断后仍达到 50 Hz 验收要求。

## 离线与回放验证

### 测试

本地与新 PC 的相关测试集均 **122 项通过**，包含 FK、球形判据、控制器、日志回放、只读接入和推理接口。首次远端测试因同步包遗漏 `scripts/replay_real_robot.py` 出现 1 项导入失败；补齐该文件后重跑全部通过，初次日志保留。

实时测试后修正回放报告对 tuple/JSON list 的比较问题：原 `same_config=false` 是球心容器类型差异，运行参数并未变化。增加球形配置序列化及真实配置差异的回归断言后，本地/远端各 **7 项回放测试通过**。实时运行代码未因这个报告修正发生变化，无需重新接入硬件。

### 十孔位真实记录的跨主机回放

从每孔最后命名的一条示教提取最后五秒的原始 q/dq，共 10 段、14,990 个样本，打包为 `recorded_goal_samples.json.gz`。包含源记录标识与哈希，不修改 HDF5；使用部署代码的在线 FK 与 GoalMonitor 逐样本重算。

新 PC 与本地的首触发时刻、状态计数、末状态完全一致。9 段在该五秒窗口内满足 500 ms 球内持续规则；`x16y08` 的所选末段窗口没有满足，双方都正确复现该结果。这项短窗口一致性检查不替代上一轮全 50 条完整轨迹回放，也不要求每条记录结束时仍留在球内。回放中的力/校零上下文为明确标记的占位值，仅验证目标算法。

### 本次完整实时日志回放

按记录的事件接收时刻和 tick 评估时刻，重新运行实际反馈、相机事件、模型预测、FK、目标与控制器逻辑。**12,881 个周期（881 个等待＋12,000 个运行）的状态、动作提议、受限提议和目标结果全部一致，差异数 0。**

最终 `same_config=true`、`final_state_matches=true`、`exact_decisions_reproduced=true`。正常退出请求的最终墙钟采样时间不是逐周期相等断言；回放不模拟机器人动力学或物理执行。

### 独立计算基准

新 PC 预热后各测 20,000 次，FK 平均 0.05498 ms；FK＋目标判断平均 0.07395 ms、P99 0.07937 ms、最大 0.32117 ms。基准与上述实时并发计时口径不同，后者包含实际系统调度干扰；均不构成硬实时最坏上界。

## 本轮代码增量

| 文件 | 改动 |
|---|---|
| `src/force_aware_act/control/runtime.py` | 增加有界 FK/目标判断耗时样本；`_record_compute`、`_goal_update` 和 `report().compute_timing`；诊断不参与判断 |
| `scripts/replay_spherical_goals.py` | `replay_sample_fixture`：读取便携真实关节样本，使用在线 FK/GoalMonitor 跨主机复现 |
| `tests/test_spherical_goals.py` | 计时字段与便携样本回放回归 |
| `scripts/deployment/replay_controller_readonly.py` | 将原始 JSON 配置解析回 `ShadowConfig` 后比较，消除 list/tuple 误报 |
| `tests/test_controller_replay.py` | 增加球形配置序列化一致性，确保不同参数仍被识别 |

无新增模型张量维度；q `[7]`、图像/模型输入及预测动作 chunk 的既有形状保持不变。本轮未修改权重、训练集、归一化、采样或正式保护限值。

## 证据与复现

本地证据目录：`/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/sphere_live_1007`。

- `tests_local_final.xml`：本地 122 项测试。
- `replay_config_regression.xml`：本地补充回放回归。
- `source_hashes_final.json`：实时运行时的 266 文件代码版本。
- `source_hashes_delivery.json`：含回放报告修正的交付版本，远端逐文件核对 266/266 一致。
- `runtime_v4_delivery.tar.gz`：完整源码交付包；SHA256 `2468fd16b99b9ced8d35ac1fb6b5e2de18dbeb7847bb3770ac899a1240e6e57a`。没有打包权重/SDK/原始数据。
- `remote_reports/`：同步回本地的紧凑远端报告，包括校零、FK 基准、数据回放结果、实时摘要、完整控制器回放与交付检查。
- `remote_reports/shadow_sphere_2min/integration_summary.json`：实时主要数值。
- `remote_reports/shadow_sphere_2min/feedback_delivery_by_phase.json`：启动/运行反馈延迟拆分。
- `remote_reports/shadow_sphere_2min/controller_replay_final.json`：完整回放最终结果。
- `remote_reports/delivery_verification.json`：源码校验、依赖版本、退出检查与原始实时日志 SHA256。

大体积原始事件、执行及预测日志保留在新 PC 的 `readonly_controller_20261007_v4/reports/shadow_sphere_2min/`，合计约 242 MB；没有重复传回笔记本。早期失败日志、修正前回放报告也保留。

只运行无硬件测试，在新 PC：

```bash
cd /home/shutiao/tianji_deployment/readonly_controller_20261007_v4
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/shutiao/miniconda3/envs/tianji/bin/python -m pytest \
  tests/test_spherical_goals.py tests/test_shadow_controller.py \
  tests/test_controller_replay.py tests/test_real_robot_endpoints.py \
  tests/test_readonly_deployment.py tests/test_real_robot_inference.py -q
```

完整日志复现可调用 `scripts.deployment.replay_controller_readonly.replay_control_events`，参数为 `Path('reports/shadow_sphere_2min/controller/events.jsonl')` 与 `ShadowConfig.read('configs/deployment/goals/x13y07.json')`，不会连接硬件。

## 下一阶段边界

可以继续制定七关节外力软/硬阈值、迟滞和持续时间、运动范围/速度/加速度/跟随误差限制，以及经过确认的停止行为，并以已保存日志和故障注入验收。这些参数和硬件执行/停止链仍是实际运动前的待办。此次交付保持只读，不构成允许模型驱动机器人运动的授权。
