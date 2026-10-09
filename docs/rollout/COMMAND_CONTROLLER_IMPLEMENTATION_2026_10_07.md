# Contact 右臂启动、发送与停止流程交付

日期：2026-10-07。范围：代码实现、离线检查、新 PC 编译。**交付配置禁止硬件控制；本轮没有连接机器人 SDK，没有发送参数、模式、位置、停止或下伺服指令。** SSH 仅用于文件同步、编译和无设备检查。

## 1. 本轮采用的参数

| 项目 | 当前值 |
|---|---|
| 控制器速度 / 加速度 | **10% / 10%** |
| 右臂 K，N·m/degree | **[2.5, 2.5, 2.5, 2.0, 1.5, 1.2, 1.0]** |
| D | [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]，保留 |
| 首目标 / 后续目标与反馈误差 | 每关节 0.05 rad |
| 相邻已提交目标变化 | 每关节 0.03 rad，超出拒绝，不裁剪 |
| 推理 / 动作网格 | 50 / 100 Hz；实际发送率和时延仍需实机测量 |
| 聚合 | recency，80 ms；只聚合同一动作时刻的候选 |
| 孔位 | 每次必填，例如 x13y07；本次运行锁定 |
| 到达 | 真实关节反馈做法兰 FK，进入指定 3 mm 球连续 500 ms |
| 停止 | cease_targets；可选 servo_off_after_standstill=false |
| 软件零偏 | 最终控制模式下静止、无接触采样 2 s；每次重新计算 |

外力 soft/release/hard 和 100 ms 持续判据保持既有用户配置。力输入仍是 `m_EST_Joint_Force` 减有效软件零偏，内部关节力矩只记录。模型、归一化、RGB 数据和过去外力窗口不变。孔位编号选择目标监测，不是新加给模型的条件输入。

低 K 会改变实际跟随表现，不能凭 v7 只读通过就认定新 K 下插入性能已验证。v7 的 75 Hz 推理结果也不包含本轮 SDK 发送开销。

## 2. 代码分工与接口

| 文件 | 内容 |
|---|---|
| `src/force_aware_act/control/command_session.py` | SessionOptions、CommandSession、SubmittedActionLimiter；启动、逐帧安全/目标监测、提交与停止状态机 |
| `src/force_aware_act/control/sdk_transport.py` | RightArmTransport；版本/指纹核对、显式连接、结构化右臂反馈与包接口 |
| `src/force_aware_act/control/command_worker.py` | CommandProcess、run_loop；独立 CPU 进程独占 SDK 连接，读取、保护和发送不依赖 GPU 线程继续运行 |
| `scripts/deployment/right_arm_transport.cpp` | 窄 C ABI；只调用右臂 B 的 setter；rad→degree；提交前检查截止时间 |
| `scripts/deployment/build_right_arm_transport.py` | 编译并记录源文件、头文件、SDK、产物 SHA256；构建过程不加载 SDK、不连接机器人 |
| `scripts/deployment/run_contact_controller.py` | 默认离线预览；显式启用后可选择 prepare-only 或模型执行；相机/模型准备、记录及退出 |
| `configs/deployment/command_session.json` | 硬件和可选下伺服开关，均为 false |

动作 chunk 是绝对关节角 **[100,7] rad**，每个执行目标 **[7] rad**；关节、外力、内部力矩均为右臂 **[7]**。原始 RGB 为每路 **[480,640,3] uint8**。没有引入未来动作或未来外力输入。

既有 `shadow_contact_readonly.py` / `robot_feedback_readonly.cpp` 继续是独立只读入口，没有加入启用运动开关。新的会话内部复用只读计算核心，但发送许可、SDK 连接和提交历史在新的模块中管理。

## 3. 启动顺序

1. 默认入口只校验并输出孔位、保护配置、预期控制器参数和流程；不加载相机、CUDA 或 SDK。
2. 未来显式启用时，同时要求会话配置 `allow_hardware=true`、`--enable-control` 和本次无接触确认。参数设置和切换模式也属于控制操作。
3. 核对 Contact 权重、SDK 与构建指纹；CUDA 预热；两路相机有有效 RGB 帧后才创建机器人连接进程。同一 SDK 目录下的新入口通过文件锁防止重复运行。
4. 读取实际关节、工具、模式和错误码，检查任务关节范围；只接受 IDLE 或已有右臂关节阻抗模式作为起始模式。不清错、不改工具、不自动回零、不自动重连。
5. 至少 500 ms 新鲜反馈显示静止（速度≤0.01 rad/s，窗口位置范围≤0.005 rad）后，单独提交速度/加速度及 K/D 参数包。
6. 至少间隔 250 ms，确认实际参数回显一致，再提交纯模式包：`OnSetTargetState_B(ARM_STATE_TORQ)` + `OnSetImpType_B(1)`。
7. 模式和参数均回显一致、再经过 250 ms 后，开始最终模式下的 2 s 软件校零。工具改变、模式/参数变化或不稳定窗口会取消运行。
8. 校零后，等待因果有效的关节/外力、两路图像和预测，再经首动作门槛提交目标。`--prepare-only` 在此阶段不产生模型目标，观察最多 5 s 后走停止流程。

初始准备总时限 15 s；参数/模式回显各 2 s。静止阈值用于启动校零和停止确认，不是额外的执行速度或加速度整形。

## 4. 每次发送的约束

- 独立 CPU 进程在 100 Hz 网格上做聚合、外力保护、反馈误差与单步检查；保护或目标完成优先于发送。
- SDK `OnClearSet` 忙时不覆盖前一包，不缓存这一时刻目标。下一时刻重新聚合；持续忙超过 100 ms 锁定停止。
- 只有 `OnSetSend` 返回成功，才把目标记为“上一次成功提交的目标”。提交成功不是控制器已执行、更不是末端已到达。
- 按已验证 SDK 头文件要求，API 调用之间至少隔 1 ms；最终 `OnSetSend` 前检查动作、反馈、力、相机的有效截止时间。这个间隔的实际调度开销需在新 PC 上后续测量。
- 记录查询时刻、可用时刻、参与聚合的 query、各自权重、候选/检查结果、提交结果、提交截止及结束时刻。不会补发错过的动作网格。
- 相机、反馈或预测过期、进程输入溢出、外力/位置/跟随超限，都锁定撤销发送许可。恢复需要新会话；原运行不会自动恢复。
- 停止观察期间仍记录控制器错误和同一模式/工具上下文中的硬外力超限，不把这类运行标记为正常结束。

## 5. “停止”准确指什么

### 默认：停止新目标，观察实际状态

发生到达、超限、超时、人工停止或正常时长结束时，立即清空动作 chunk、锁定发送许可，并在 C++ 传输层禁止后续目标包；不反复发送保持目标，不自动撤退。

**采集桥接代码明确写明：不发送新目标时，控制器仍保持上一次目标。** 因此它可能继续追踪最后目标、保持接触力，停止发送本身不能保证瞬时停止，也不能保证外力解除。

随后用连续新鲜反馈观察至少 500 ms 的静止窗口。满足后记录 `STOPPED_HOLDING_LAST_TARGET`；这是当时观察到停稳，伺服仍可能保持最后目标。无法在 2 s 内确认、通信失联或 SDK 异常时，记录未确认状态，不声称物理停止。

### 可选：观察停稳后下伺服

`servo_off_after_standstill=true` 时，先完成上述静止窗口，再单独提交 `ARM_STATE_IDLE`，并等待新反馈确认为 IDLE。没有新鲜反馈、仍运动、存在控制器错误时不会盲发下伺服。

厂家 `FxRtCSDef.h` 将 IDLE 标为“下伺服”。它不是切断整机电源；本轮没有接入电源控制。该选项当前为 false，现场验证应关注下伺服后的支撑/制动行为，再决定是否长期使用。

SDK 还有 `OnSetStopRunning("B")`，头文件写明是中止运动、非急停、非 IDLE；本轮没有把它当成已经在关节阻抗模式验证过的停止动作，也没有调用它。`OnRelease()` 只断开连接，不能作为停止证明。

GPU 挂起时，独立 CPU 进程仍可因预测过期撤销发送；若主机、控制进程或通信本身失效，则无法依靠该软件确认或保证物理停止。控制器自身断线/看门狗行为仍待现场核实。用户此前不要求故障注入，本轮没有做故障注入或硬件异常试验。

依据：采集桥 `marvin_teleop_bridge_node.cpp` 的 COMMAND_WAIT 分支与 `marvin_sdk_adapter.cpp` 的包顺序；新 PC `sdk_validated/MarvinSDK.h`、`FxRtCSDef.h`。新旧 SDK 文件不混用。

## 6. 当前可以运行的命令：完全离线

本地仓库根目录：

```bash
PYTHONPATH=src:. /home/stw/miniconda3/envs/forceact/bin/python \
  scripts/deployment/run_contact_controller.py --goal-id x13y07
```

换成 `--goal-id x14y08` 会选择对应目标球。默认命令即使填写了模型或 SDK 路径，也只预览；原始配置下单加 `--enable-control` 也会拒绝启动。

新 PC 交付目录：`/home/shutiao/tianji_deployment/command_controller_20261007_v8`，使用 `tianji` 环境。

```bash
cd /home/shutiao/tianji_deployment/command_controller_20261007_v8
PYTHONPATH=src:. /home/shutiao/miniconda3/envs/tianji/bin/python \
  scripts/deployment/run_contact_controller.py --goal-id x13y07
```

编译本身不连接设备；产物已存在时应换输出路径，不覆盖：

```bash
PYTHONPATH=src:. /home/shutiao/miniconda3/envs/tianji/bin/python \
  scripts/deployment/build_right_arm_transport.py \
  --sdk-directory /home/shutiao/tianji_deployment/sdk_validated \
  --output build/right_arm_transport.so
```

## 7. 未来实机授权后的调用方式（本轮未启用）

须先另存会话配置为 `configs/deployment/command_session.enabled.json`，经现场确认再将 `allow_hardware` 改为 true；当前交付不创建该启用文件。停止已有采集/桥接/只读 SDK 进程，避免多个连接或控制来源；文件锁只能约束新入口，不能约束厂商界面和旧采集程序。

第一次建议先使用下面的 **prepare-only**：会设置参数、切换模式并可能上伺服，但不会发送模型位置目标。这不是只读命令。

```bash
PYTHONPATH=src:. /home/shutiao/miniconda3/envs/tianji/bin/python \
  scripts/deployment/run_contact_controller.py \
  --goal-id x13y07 \
  --session-config configs/deployment/command_session.enabled.json \
  --checkpoint /home/shutiao/tianji_deployment/models/contact_seed0_b8/best.pt \
  --transport build/right_arm_transport.so \
  --sdk /home/shutiao/tianji_deployment/sdk_validated/libMarvinSDK.so \
  --enable-control --operator-confirmed-no-contact \
  --prepare-only --seconds 5 --output reports/x13y07_prepare_001
```

该阶段回读、校零和停止行为得到现场确认后，去掉 `--prepare-only`，使用新的输出目录及短时长（例如 `--seconds 3`）才会进入模型动作测试。Ctrl+C 请求撤销发送并观察停稳，不是硬件急停。到达目标球或保护触发会提前停止。

## 8. 验证与待确认事项

离线测试命令：

```bash
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/stw/miniconda3/envs/forceact/bin/python -m pytest \
  tests/test_operator_profile_config.py tests/test_goal_selection.py \
  tests/test_command_session.py tests/test_right_arm_transport.py -q
```

新增测试使用内存记录器和无网络 SDK 替身，覆盖正常启动、正常提交及角度单位转换、FK 球停留触发停止、手动停止锁定、可选 IDLE 回读、默认入口禁用及构建指纹。测试替身结果不是机械臂物理行为验证。

尚待实际授权后确认：10%/10% 和新 K/D 的实际回显、最终模式的无接触零偏、加入真实 SDK 发送后的 50/100 Hz 时延、低刚度跟随情况、停止发送后的最后目标行为、可选 IDLE 后的制动/支撑，以及控制器断线行为。**本轮交付不把这些标记为通过。**

### 本次实际交付记录

- 本地 `forceact` 环境：上述四份测试 **46 项通过**。
- 新 PC `tianji` 环境：相同四份测试 **46 项通过**。
- 新 PC 已编译 `build/right_arm_transport.so`，使用 `-Wall -Wextra -Werror` 并确认链接至已验证 SDK 的绝对路径。未加载真实传输库、未调用真实 SDK 连接或控制 API。
- SDK SHA256：`47ab2ab6e35cf899eae3978dfb50ee219e05b10f72aedd6bc9ba6a338a8b4011`。
- 新 PC 构建指纹与头文件指纹：`build/right_arm_transport.json`；离线预览及测试记录：`reports/default_preview.json`、`reports/offline_checks.xml`。
- 本地证据目录：`outputs/command_controller_20261007_v8/`；最终源文件与远端一致性见该目录的清单和验收 JSON。
- `allow_hardware=false`、`servo_off_after_standstill=false`；没有创建启用配置，没有启动真实控制进程，没有修改控制器参数。
