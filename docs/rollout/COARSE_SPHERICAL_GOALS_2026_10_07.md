# 十点位球形粗略完成区、关节空间对照与 FK 耗时

日期：2026-10-07。

后续更新：同日已完成新 PC 同步与 120 秒实时只读联调，见 [新 PC 联调报告](SPHERICAL_GOALS_NEW_PC_INTEGRATION_2026_10_07.md)。下文保留本轮本地验证时的状态与测量口径。

## 用户选择与交付范围

用户同意用统计出的十个点位作为粗略到达/完成点，并以三维球作为完成区域；随后询问是否可以直接判断关节空间，以及 FK 是否会影响推理。

本次已在**本地工作区**实现球形判据及纯数学 FK 接入，更新十个点位配置，并完成全数据离线回放、关节空间对照与耗时测试。未连接机械臂或相机、未发送运动/模式/使能/停止指令。没有将本次更新同步至新 PC，也没有启动新 PC 的实时监测。

半径 **3 mm**、持续 **500 ms** 为本次明确采用的初始参数，尚非用户逐项指定的数值；均可在单点配置调整。球心使用上次报告的 `last_0.5s` 五条示教等权重组中心，保持所选十点不变；没有以 25 mm 网格覆盖中心，也没有为提高覆盖率重新拟合。

## 规则

```text
本轮明确选择一个 goal_id
    ↓
实际右臂关节反馈 q[7] → 纯数学 FK → 基座内法兰位置 p[3]
    ↓
||p − center[goal_id]|| ≤ radius
    ↓
新鲜实际样本连续满足 500 ms → COARSE_REACHED
    ↓
独立保护无故障 → COMPLETED_SHADOW，清除并停止后续动作提议
```

- 只判断选定球体，不会任意靠近其他九个球就完成。
- 球心单位 m，`radius_m=0.003`；坐标系 `marvin_m6_ccs40_right_base`，参考点是**法兰**，不是尚未标定的插入件尖端。
- 判据是纯位置占据和持续时间，不额外要求姿态或关节速度。七轴冗余运动可能使关节速度较高但末端仍留在球内，因此未暗中沿用原关节目标的 `max_joint_velocity_rad_s=0.01` 门槛；该旧字段对 sphere 不生效。
- 逐个约 300 Hz 的有效反馈检查，离球就重置计时；重复样本不推进，倒退/过期/缺位姿/坐标系错误不能完成，超过 20 ms 间断重置计时。两个 100 Hz tick 之间的短暂离球也会被捕获。
- FK 位姿严格继承该条实际反馈时间戳。不使用模型预测关节角计算到达；外部 pose 消息不能覆盖自动 FK 来源。
- 到达在监测进程中锁存，100 Hz tick 先检查保护，再决定正常完成。硬力矩故障、数据超时等仍优先，完成后到来的更严重故障也能覆盖正常完成。
- `coarse_completion_accepted=true` 表示用户接受此粗略定义；`verified=false` 继续保留，避免把粗略到达说成经过物理 TCP/深度标定的插入成功。
- `allow_motion=false` 及“无动作后端”继续受代码强制。`COMPLETED_SHADOW` 不等于已给机器人下发停止命令。

默认 `contact_right_shadow.json` 仍为未选择点位；应显式选择十个配置中的一个，防止默认为错误孔位。

## 修改与接口

| 文件 | 内容 |
|---|---|
| `src/force_aware_act/control/contracts.py` | `Goal.kind=sphere`，球心、半径、FK 来源和独立的粗略完成接受标记；旧 joint/insertion 配置仍可读取 |
| `src/force_aware_act/control/kinematics.py` | `right_flange_transform`、`right_flange_pose`、`quaternion_xyzw`；q `[7]` rad → T `[4,4]` mm → Pose 的 position `[3]` m、quaternion `[4]` |
| `src/force_aware_act/control/supervision.py` | `GoalMonitor` 的球内/球外、连续时间、粗略完成状态；倒退反馈重置 |
| `src/force_aware_act/control/runtime.py` | 实际反馈 FK、逐帧检测、待处理完成锁存、保护优先及影子正常完成 |
| `configs/deployment/goals/x*.json` | 十个各自选择的球形目标，3 mm/500 ms |
| `configs/deployment/spherical_goals_manifest.json` | 中心来源、统计文件 SHA、模型 SHA、配置 SHA 与语义 |
| `scripts/replay_spherical_goals.py` | `prepare_configs`、`goal_only_replay`、`space_comparison`；全轨迹完成检测及两种空间对照 |
| `scripts/benchmark_goal_monitor.py` | 无硬件的 FK 与目标判断微基准，输出均值、P99、最大值和主机信息 |
| `tests/test_spherical_goals.py` | 几何、单位、时间、间断、来源、接受状态、实际反馈、故障优先、配置及脚本辅助函数测试 |

运动学计算无需加载厂家 SDK，也不调用 ROS 服务、网络或 GPU。固定链对应 SHA `ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52` 的 M6 CCS 4.0 配置。150 个真实数据姿态与前一轮已通过厂家库对照的离线实现交叉检查，最大位置差 **6.36e-14 mm**。

## 50 条全轨迹回放

证据目录：`outputs/spherical_goals_20261007/`。

- 采用十个已固定组中心、半径 3 mm、持续 500 ms；全 50 条均曾满足粗略完成规则。
- 第一次触发距录制结束 **3.72～11.02 秒**，中位数 **5.40 秒**；34 条在最后五秒之前已经触发。
- 这与上一轮“其他四条中心、仅回看最后五秒、附姿态条件”的 **47/50** 是不同统计口径，不能直接当作算法准确率提升。
- 本次球心由全部 50 条产生，且缺乏首次成功时刻标注；50/50 仅说明已知成功演示曾进入配置区域。**不能排除插入尚未完成时就进入球体的情况，也不是新模型实机成功率。**
- 这是 GoalMonitor 的独立回放：没有用假设的力矩阈值、相机或模型门槛替代实际保护验收。完整 ShadowController 的完成锁存和故障优先由合成测试验证。

逐条首触发时刻、原始数据 SHA、末段状态见 `episodes.json`；汇总见 `summary.json`。

## 是否可以只用关节空间

可以，但“每关节误差小”与“末端落在毫米级球内”不是同一个约束。对于七轴机械臂，关节构型重复性会影响阈值选择。

这批记录中 `x17y08` 的 **J3** 在五条末段代表姿态间相差 **0.2123 rad，约 12.16°**。因此不能因为是同一孔，就默认七个关节都会接近一个固定关节向量。

公平对照采用：末五秒、连续 ≥0.5 秒、同孔位另外四条的参考、不含被检查示教自身；两种规则都只判断几何，不加速度/姿态门槛。关节规则要求七关节分别满足统一误差范围，球形规则判断三维距离。额外检查九个非本次目标，描述区域之间是否容易混淆。

| 方法 | 自身目标覆盖 | 同时也匹配过其他孔位的示教数 |
|---|---:|---:|
|各关节 ±1°|15/50|0/50|
|各关节 ±2°|29/50|0/50|
|各关节 ±3°|41/50|19/50|
|各关节 ±5°|45/50|33/50|
|FK 球半径 3 mm|47/50|0/50|
|FK 球半径 5 mm|50/50|0/50|

“匹配其他孔位”是对照区域存在歧义的统计，不代表实际控制器会自动换目标或已测得的误停止率。当前控制器只认明确选择的 `goal_id`。

这项对照没有穷尽逐关节不同阈值、关节模板集合或统计距离等设计；这些方法可以继续研究。但当前数据和下述耗时测量**没有显示为节约 FK 开销而更换成简单关节阈值的必要性**。

## FK 会不会影响 50 Hz 推理

### 本机实测

主机：`Legion-Y9000P`，Intel Core Ultra 9 275HX，Python 3.10.20、NumPy 2.2.6。使用 50 条记录的实际关节构型循环采样，预热后各测 20,000 次。时钟为 `perf_counter_ns`。

测试结束后单独运行的结果，包含四元数转换、数值/单位与 Pose 校验：

| 运算 | 平均 | P99 | 本次最大 |
|---|---:|---:|---:|
|FK → 完整 Pose|0.0547 ms|0.0694 ms|1.5543 ms|
|FK ＋ GoalMonitor.update|0.0704 ms|0.1068 ms|0.1247 ms|

两行是各自独立采样，最大值不能直接相减。另一次与单元测试并发执行的测量中，FK 单项有 **6.671 ms** 的偶发长耗时；FK＋目标判断均值 0.0894 ms、P99 0.1294 ms。保留两次原始结果，未丢弃长尾。

这些计时包含操作系统抢占等影响，不证明某次峰值全由 FK 算术引起，也不是硬实时上界。原始结果：`fk_benchmark.json`、`fk_benchmark_sequential.json`。

### 计算预算与放置位置

按单独运行的 FK＋判定均值估算：

- 50 Hz：约占一个 CPU 核心 **0.35%**。
- 100 Hz：约 **0.70%**。
- 300 Hz：约 **2.11%**。

50 Hz 推理周期为 **20 ms**。这里的 FK 不做迭代求解，只按七关节链相乘。它运行在现有独立 CPU `ShadowProcess`，不占用 GPU，也不加入模型预处理/前向的同步等待路径。共享 CPU 调度、日志和 IPC 的总体影响仍需完整程序联调衡量，不能据微基准宣称实时推理绝对零影响。

保留状态/外力/位姿新鲜度检查和时间间断处理；不能用平均耗时替代超时保护。若后续联调出现调度长尾，应先优化监测进程的资源和日志，而不是仅为了省掉几十微秒放弃三维目标定义。

### 新 PC 状态

本次尝试 SSH 读取新 PC 主机信息时，连接返回 `Host key verification failed`，未绕过主机密钥校验。**上述性能数值来自笔记本，不是 4090 主机实测。** 后续在新 PC 的 `tianji` 环境中可先单独运行相同无硬件微基准，再在只读联调中比较推理周期/反馈交付延迟；该基准本身不要求机械臂静止或软件校零，因为完全不读取硬件。

## 复现

下列命令从仓库根目录运行，输出目录需使用新名称。

```bash
# 当前十点配置已生成；重放不再次覆盖配置。
/home/stw/miniconda3/envs/forceact/bin/python scripts/replay_spherical_goals.py \
  --data-root /home/stw/Downloads/1mm-gap-10points_hdf5_raw_v2 \
  --statistics-dir outputs/endpoint_fk_20261007 \
  --model /home/stw/TJ_FX_ROBOT_CONTRL_SDK/CommonConfig/ccs_m6_40.MvKDCfg \
  --goals-dir configs/deployment/goals \
  --output outputs/spherical_goals_new_replay

/home/stw/miniconda3/envs/forceact/bin/python scripts/benchmark_goal_monitor.py \
  --config configs/deployment/goals/x13y07.json \
  --episodes outputs/endpoint_fk_20261007/episodes.json \
  --iterations 20000 \
  --output outputs/spherical_goals_new_benchmark.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/stw/miniconda3/envs/forceact/bin/python -m pytest \
  tests/test_spherical_goals.py tests/test_shadow_controller.py \
  tests/test_controller_replay.py tests/test_real_robot_endpoints.py \
  tests/test_readonly_deployment.py tests/test_real_robot_inference.py -q
```

相关测试 **121 项通过**，包括新球形逻辑、旧控制器、事件回放、只读接入与推理接口回归。未修改模型、训练采样、归一化或数据集。

将本次 `control/` 源码及配置同步到新 PC 后，已有只读推理脚本通过 `--controller-config configs/deployment/goals/x13y07.json` 选择具体点位；不能只复制新配置而继续使用不认识 `sphere` 的旧版代码。实时采集与校零仍按现有只读程序流程进行，本次未启动该流程。
