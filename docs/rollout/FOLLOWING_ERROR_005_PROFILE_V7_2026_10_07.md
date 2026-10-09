# v7：七关节反馈误差 0.05 rad、单步跳变 0.03 rad

日期：2026-10-07。用户针对 v6 第二条影子目标的 J5 误差超限，要求所有关节都采用 0.05 rad 的后续误差门槛，随后补充要求单步也统一改为 0.03 rad。

本文记录初次配置交付和离线检查阶段。随后用户授权的两分钟实际只读重测已完成，见 [独立重测报告](V7_X13Y07_READONLY_RETEST_2026_10_07.md)。

## 本轮改动

公共配置及十个孔位配置共 11 个文件中：

```text
limits.following_error_rad
原值：[0.035, 0.025, 0.015, 0.040, 0.015, 0.020, 0.015]
新值：[0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050]
```

该字段同时用于第二条及以后的“新目标—实际反馈”和“上一条接受目标—实际反馈”两项检查。当前全部七关节采用同一个 0.05 rad 门槛，约 2.865°。

首动作 `first_action_error_rad` 保持每关节 0.05 rad。单步 `max_command_step_rad` 按补充要求由每关节 0.01 rad 改为每关节 0.03 rad（约 1.719°），仍在时间聚合后比较连续接受目标，超限拒绝、不截断。目标球、七关节外力、任务关节范围、时序及聚合参数未改变。

速度预期仍为 10%，加速度预期仍为 30%。v6 最近一次实际回显为速度 10%、加速度 10%；本轮没有读写控制器，不提供新的实机回显结论。

## 同步与核对

- 新 PC：`/home/shutiao/tianji_deployment/readonly_controller_20261007_v7`，conda `tianji`。
- v5、v6 的配置及实时日志保留。
- 第一轮同步与 v6 比较，仅修改 `limits.following_error_rad`；随后在 v7 内追加 `limits.max_command_step_rad` 的 0.01 → 0.03 rad 变更。两次均覆盖公共配置及十孔位配置，分别保存前后指纹。
- 本地与新 PC 各 **36 项离线检查通过**，包含十孔配置、准确数值、配置指纹、读取往返及正常计算。
- 正常计算检查的第二条目标与实际反馈相差 0.045 rad，上一条目标与反馈相差 0.04 rad；在新的统一门槛下通过，仍独立检查单步变化。
- 单步补充检查：第三条目标相对上一条接受目标变化 0.03 rad，按新配置原样接受，不做截断；读取检查同时断言所有七关节首动作/后续误差为 0.05 rad、单步为 0.03 rad。
- 用 v6 已记录的前两条目标和各自实际关节角，再经过 v7 的 `ActionLimiter.apply()` 计算，两条均被接受。这只验证这两条历史输入；没有运行完整实时链路或模拟机械臂运动。

只读程序已在 v7 重新编译，动态 SDK 依赖仍是兼容检查、连接、读取和释放。离线 `--goal-id x13y07 --describe-goal` 的最新配置快照显示首动作/后续反馈误差均为 0.05 rad、单步 0.03 rad。

**本轮交付为配置修改、同步及离线检查。没有重新校零、连接机械臂/相机、运行实时预测、设置控制器参数或发送电机控制指令。**

## 文件与复现

- 配置目录：`configs/deployment/profiles/contact_right_step_v1/`。
- 本地记录：`outputs/goal_x13y07_v7_20261007/`。
- `parameter_changes.json`：逐文件的原值、新值及修改前后 SHA256。
- `step003_changes.json`：单步 0.01 → 0.03 rad 的逐文件变更。`source_hashes_feedback005_before_step003.json` 保存单步补充前版本，最终清单继续使用 `source_hashes_delivery.json`。
- `previous_two_targets_check.json`：v6 两条历史目标的输入摘要来源 SHA256 和重新计算结果。
- `source_hashes.json` / `source_hashes_delivery.json`：同步及交付源码/配置清单。
- 新 PC `reports/preflight.json`、`reports/tests_remote_delivery.xml`、`reports/delivery_verification.json`：远端配置、测试和最终一致性检查。

本轮未增加控制器函数/类或修改执行算法；仅修改配置、说明及已有测试的期望值和正常计算输入。关节/外力形状仍为 `[7]`，预测动作仍为 `[100,7]`。

离线复核（新 PC，不访问设备）：

```bash
cd /home/shutiao/tianji_deployment/readonly_controller_20261007_v7
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/shutiao/miniconda3/envs/tianji/bin/python -m pytest \
  tests/test_operator_profile_config.py tests/test_goal_selection.py -q
PYTHONPATH=src:. /home/shutiao/miniconda3/envs/tianji/bin/python \
  scripts/deployment/shadow_contact_readonly.py --goal-id x13y07 --describe-goal
```
