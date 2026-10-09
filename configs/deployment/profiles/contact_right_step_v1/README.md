# Contact 右臂参数配置 v1

日期：2026-10-07。状态：**首动作及后续反馈误差门槛均为每关节 0.05 rad，单步跳变为每关节 0.03 rad；v7 两分钟完整只读重测通过，不启用实机运动。**

## 文件

- `base.json`：公共保护/动作参数，未选择孔位。
- `goals/x13y07.json` ～ `goals/x17y08.json`：十个显式选择孔位的完整影子配置，均可通过 `ShadowConfig.read()` 读取。
- `controller_baseline.json`：右臂阻抗、速度 10% / 加速度 10% 的控制器预期参数；仅保存，不在加载时应用。速度由用户在本轮指定从 30% 降为 10%。
- `manifest.json`：用户决策、参数来源、配置文件 SHA256 和配套读取器源码 SHA256。

原 `configs/deployment/contact_right_shadow.json`、`configs/deployment/goals/` 及历史报告均保留。使用下述 `--goal-id` 入口或显式指向本目录下的文件即可选择本配置，旧配置不会自动转为新配置。

## 每次运行选择孔位

启动脚本支持 `--goal-id x14y08`，自动加载本目录的 `goals/x14y08.json`。
可选编号：`x13y07`、`x13y08`、`x14y07`、`x14y08`、`x15y07`、`x15y08`、`x16y07`、`x16y08`、`x17y07`、`x17y08`。
编号是已有孔位的符号标签，不能直接按数字解释为米或毫米；球心仍来自示教统计的法兰 FK。

在仓库根目录离线预览（不需要模型、校零确认或硬件连接）：

```bash
PYTHONPATH=src:. /home/stw/miniconda3/envs/forceact/bin/python \
  scripts/deployment/shadow_contact_readonly.py --goal-id x14y08 --describe-goal
```

在已有只读运行命令中，用 `--goal-id x14y08 --query-hz 50` 替代 `--controller-config ...`。
`--controller-profile /absolute/path/to/contact_right_step_v1` 可指定另一份完整配置目录，必须同时提供 `--goal-id`。
二者不能与 `--controller-config` 混用；不合法或不存在的编号会在加载 SDK 前退出，不会默认选择其它孔位。
历史纯预测入口仍可不启用目标监测；历史显式 `--controller-config` 用法继续支持。

程序启动时校验配置与 manifest 的 SHA256、内部孔位编号及公共保护参数，打印 `GOAL_SELECTION`。
运行目录写入 `goal_selection.json`，`shadow_report.json` 同样保存选定编号、球心、半径、停留时间、配置路径/指纹、完整配置快照及控制器预期参数。只读反馈样本分别记录实际速度/加速度比例，不能将预期配置视为已在硬件生效。
本次目标在启动时固定，不在循环里重复读文件；下一次改变编号会创建全新的监测状态和停留计时。
到达其它孔位不算当前任务完成。孔位标签仅选择目标监测区域，不作为现有 Contact 模型的新输入，也不改变其动作预测。

## 七关节外力参数

顺序 `Joint1_R` ～ `Joint7_R`，单位 N·m。信号为 `estimated_joint_disturbance`；新 PC 对应有效软件校零后的 `m_EST_Joint_Force`。内部关节力矩不参与此阈值判断。

| 关节 | soft | release | hard |
|---|---:|---:|---:|
| J1 | 17.5 | 16.625 | 20.0 |
| J2 | 6.5 | 6.175 | 6.75 |
| J3 | 5.0 | 4.750 | 5.5 |
| J4 | 11.0 | 10.450 | 12.0 |
| J5 | 1.0 | 0.950 | 1.2 |
| J6 | 6.5 | 6.175 | 7.0 |
| J7 | 2.0 | 1.900 | 2.5 |

- release = **95% soft**；持续时间 **100 ms**。
- 绝对值达到 soft 开始计时，回落至 release 或以下才清零；迟滞带内保留计时。
- 任一新鲜样本达到 hard，立即锁定影子中止请求。软件发现故障不等于已经物理停止。
- 运行故障不自动恢复；没有修改已有安全优先级与外力有效性检查。
- 这些参数全部高于现有 50 条示教的对应绝对峰值，因此该数据集不会触发这组外力阈值；这不构成异常接触检测效果的验证。

## 动作与控制器分工

| 项目 | 配置 |
|---|---|
| 推理 / 影子执行网格 | 50 / 100 Hz |
| 时间聚合 | `recency`，衰减时间 80 ms |
| 控制器速度 / 加速度 | 10% / 10%，仅记录预期值，各次实际回显见验收报告 |
| 模式 | `joint_impedance` |
| K，N·m/deg | `[2.5,2.5,2.5,2.0,1.5,1.2,1.0]` |
| D | `[0.3,0.3,0.3,0.2,0.2,0.2,0.2]` |
| 上位机动作保护 | `motion_guard=step_reject` |
| 单步变化 | 每关节 `0.03 rad`，超限拒绝，不截断目标；2026-10-07 用户从 0.01 rad 调整 |
| 首动作对齐 | 每关节相对实际反馈 `0.05 rad`，作为独立首动作门槛；2026-10-07 用户从 0.02 rad 调整 |
| 上位机速度/加速度限幅及实际速度阈值 | 不配置、不执行 |
| 连续限幅 200 ms 中止 | 新模式不使用，字段为 null |

单步检查位于时间聚合后。首动作通过独立对齐门槛；后续与上一个接受的目标相比。浮点比较只保留 `1e-12 rad` 的数值容差。

首动作对齐门槛为每关节 0.05 rad。用户随后将后续目标—实际反馈及前一接受目标—实际反馈的门槛也统一为每关节 0.05 rad（`following_error_rad`）；单步跳变另按用户补充要求统一为每关节 0.03 rad。只读状态下实际机器人不会跟随影子目标，超出任何适用门槛仍会被拒绝。

**当前影子程序的参考是上一次接受的影子目标，没有任何硬件发送。**未来真实发送端必须另行使用“上一次成功提交的目标”作比较，并且只有发送成功后才能推进参考；新增独立 command_session 实现提交成功后推进参考，现有影子入口仍保持只读。不能把影子目标当作已发送目标。

参数 `velocity_rad_s`、`acceleration_rad_s2`、`max_consecutive_limited_ms` 在新模式中必须为 null。旧配置继续使用默认 `velocity_acceleration` 模式，行为不因本次新增配置而切换。

## 位置与跟随误差

| 关节 | 任务下限 rad | 任务上限 rad | 跟随误差 rad |
|---|---:|---:|---:|
| J1 | −2.02 | −0.97 | 0.050 |
| J2 | −1.64 | −1.11 | 0.050 |
| J3 | 1.00 | 1.79 | 0.050 |
| J4 | −2.01 | −0.83 | 0.050 |
| J5 | −0.10 | 0.51 | 0.050 |
| J6 | −0.30 | 0.09 | 0.050 |
| J7 | −0.35 | 0.22 | 0.050 |

该任务范围来自当前十孔示教附近，不用于所有回零/搬运/撤离流程。保留目标与实际反馈的误差检查，以及前一接受目标与实际反馈的跟随检查；每步很小仍可能累计跟随误差。

## 数据、目标与授权

- 状态/外力新鲜度：20 ms；图像：100 ms；预测最大年龄：200 ms。
- 执行迟到门槛：10 ms；任务超时：180 s。
- 十孔目标：沿用现有实际关节反馈 FK、法兰位置球、半径 3 mm、连续 500 ms。默认公共配置不选择孔位。
- `allow_motion=false`、此影子入口没有硬件动作后端、`physical_stop_verified=false`。
- `authority=operator_reviewed` 仅表示这套参数已由用户选择，不表示运动授权或物理安全验证。
- v5、v6 配置和运行日志保留。本次首动作/后续误差均为 0.05 rad 的配置使用独立的 `/home/shutiao/tianji_deployment/readonly_controller_20261007_v7` 目录。
- v5 只读读取器从 `m_In[arm].m_Joint_Vel_Ratio` / `m_Joint_Acc_Ratio` 记录速度、加速度比例。配置预期值不写入控制器；本次实际回显为 10% / 10%，与加速度预期 30% 不一致，未标记整套控制器参数核对通过。
- 本次选择 `x13y07`；实时验收细节见 `docs/rollout/SELECTED_GOAL_NEW_PC_V5_2026_10_07.md`。影子保护锁定中止后不再生成可执行目标，不能将后续持续观测和推理解释为动作链路通过。
- v6 历史重测见 `docs/rollout/FIRST_ACTION_005_NEW_PC_V6_2026_10_07.md`：首条目标通过 0.05 rad 门槛；第二条 J5 目标—反馈误差 0.016506 rad 超过当时的 0.015 rad 后续门槛，锁定影子中止。
- v7 初次配置变更和离线检查见 `docs/rollout/FOLLOWING_ERROR_005_PROFILE_V7_2026_10_07.md`，该阶段没有硬件测试。
- 随后的两分钟实际只读重测见 `docs/rollout/V7_X13Y07_READONLY_RETEST_2026_10_07.md`：6,000 次预测、12,000 条影子目标全部完成，无保护故障；最大反馈误差 0.033972 rad，最大连续单步 0.003032 rad。模型和机械臂之间仍没有动作发送后端。

## 离线检查

从仓库根目录运行，不会连接机械臂：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/stw/miniconda3/envs/forceact/bin/python -m pytest \
  tests/test_operator_profile_config.py tests/test_goal_selection.py -q
```

本轮 14 项通过：11 个配置的数值、往返读取和禁用运动状态；manifest/控制器基线一致性；正常单步动作计算；旧配置默认模式及正常限幅公式兼容。没有运行故障注入测试。

孔位选择补充检查：十孔映射、配置快照、启动参数校验、命中指定球而非相邻球、新运行清空计时、离线预览及旧入口兼容。均不接触硬件。

新增 `control.goal_selection.select_goal()` / `GoalSelection`，在启动脚本 `parse_arguments()` 中完成选择，`main()` 将其交给已有 `ShadowProcess`。本次两份测试共 36 项通过；结果保存在 `outputs/operator_profile_20261007/goal_selection_checks.xml`。

实现增量：`Limits.motion_guard`、`Limits.max_command_step_rad`、按模式判断的 `missing_motion()`；`ActionLimiter.apply()` 增加原样接受或拒绝的 step 分支。输入仍是目标/反馈 `[7]`，模型动作 chunk `[100,7]` 和图像接口不变。

## 发送流程实现（2026-10-07，尚未实机启用）

用户随后确定速度/加速度均为 10%，K 为 `[2.5,2.5,2.5,2,1.5,1.2,1]`，D 保留。
独立入口 `scripts/deployment/run_contact_controller.py` 默认只输出计划，硬件开关位于 `configs/deployment/command_session.json`，交付为 false。
新的发送会话、构建方法、停止语义及待验证事项见 `docs/rollout/COMMAND_CONTROLLER_IMPLEMENTATION_2026_10_07.md`。
v7 历史测试和部署副本未使用这一版刚度；本地当前配置变更尚未应用于控制器。
