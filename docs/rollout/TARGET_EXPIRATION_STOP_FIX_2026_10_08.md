# 200 ms 聚合边界与下使能锁存修复

日期：2026-10-08。依据为操作者在新 PC 运行的 `x13y07_rollout_3s_5f939c629d86`。本轮只读取日志、离线回放、修改源代码、编译和同步；未连接真实 SDK 或发送机械臂指令。

## 本次实际结果

- 校零状态 `valid`，模型预测 12 次，参数/模式包各 1 个，模型目标提交 18 个。
- 约 180 ms 的目标发送后，以 `sdk_submission_error` 中止；没有触发关节误差、单步跳变或力矩保护。
- 停止路径观察到静止后尝试下使能，但得到 `servo_off_submission_failed`，下使能提交数为 0，最终为 `STOP_UNCONFIRMED`。
- SDK 的 `Robot released` 表示连接释放；不能作为下使能或物理停止的确认。现场应由操作人员确认停稳与下使能后再重新测试。

## 原因一：仍参与聚合的旧预测已经到期

失败 tick 为 `160863329510937` ns。其 10 条预测年龄依次为 200、180、160、140、120、100、80、60、40、20 ms。旧实现按逻辑 tick 判断年龄，允许年龄恰好 200 ms 的预测参与聚合。

发送截止时间取执行迟到、状态、外力、图像和全部参与预测的最早有效期限。这条 200 ms 旧预测把截止时间压到了 tick 本身。计算时刻为 `160863329932771` ns，已过期 0.421834 ms；适配器返回时为 `160863332803110` ns，已过期 3.292173 ms。SDK 提交错误这一分类同时包含了适配器的截止时间拒绝，不能据此判定 SDK 通信故障。

失败时图像与关节反馈仍然新鲜，其余 9 条预测也有效。该目标通过了动作与力矩保护。

### 处理

- `TemporalExecutor.sample()` 用实际计算时刻检查预测是否仍未到期；预测完成可用性仍按逻辑 tick 检查，动作索引仍取同一个 100 Hz 网格时间点。
- `CommandSession.tick()` 记录各截止时间来源，并在已经过期时直接丢弃本轮目标。
- 原生适配器仍在调用 `OnSetSend()` 前检查时间，防止计算或 SDK API 排队过程中再次过期。
- 过期返回独立的 `expired`，没有目标发送，不推进上一成功提交目标，不重试这条旧目标；下一 tick 使用新聚合。原生已暂存的过期目标必须经 `OnClearSet()` 清除后才可继续，清除失败仍作为 SDK 错误中止。
- 不扩大 200 ms 预测有效期、10 ms 执行迟到、图像/反馈超时或动作保护门槛。图像、状态、预测持续失效仍会触发已有保护。

## 原因二：发送失败锁存同时封住了下使能

旧原生代码将过期统一置为 `failed=true`，而 `begin_packet(true)` 仍无条件拒绝 `failed`。因此即使 Python 已撤销目标、观察到连续 500 ms 静止，IDLE 包也无法进入 SDK。这与本次零下使能包、`servo_off_submission_failed` 的记录相符。

### 处理

- 真实 setter/`OnSetSend()` 失败仍锁存运动发送权限；后续目标、参数与模式包保持被拒绝。
- 目标已经撤销后，独立的 IDLE 包可经清除暂存缓冲再尝试提交。Python 仍要求新鲜、无控制器错误的模式反馈，以及原有连续 500 ms 静止检查。
- SDK 拒绝、忙超时、反馈未确认均继续报告未确认状态，不自动恢复动作，不用连接释放代替停稳/下使能回读。
- `submission_detail`、`submit_deadline_sources_ns`、`stop.submission_failure`、`stop.servo_off_submission` 与 `expired_target_ticks` 为新增诊断字段。

## 版本与参数

原生 C ABI 从 1 升到 **2**，增加过期返回值和 `fa_last_packet_detail()`。新 Python 读取器拒绝旧 ABI；必须使用 v9 重新编译的 `build/right_arm_transport.so`。厂家 SDK 本身未修改：版本 100343013，SHA256 `47ab2ab6e35cf899eae3978dfb50ee219e05b10f72aedd6bc9ba6a338a8b4011`。

v9 目录：`/home/shutiao/tianji_deployment/command_controller_20261008_v9`。原 v8 与其构建文件、运行日志保留。默认配置仍禁止硬件；操作者已建立的会话配置按原样复制，不重新打开或执行。

- 七关节首动作误差 0.05 rad、后续反馈误差 0.05 rad、单步跳变 0.03 rad。
- 控制器速度/加速度 10%/10%，右臂刚度 `[2.5,2.5,2.5,2.0,1.5,1.2,1.0]` N·m/°。
- 预测 50 Hz，逻辑执行 100 Hz，原有 temporal aggregation 方式与权重规则保留。
- 模型输入/输出结构不变，动作 chunk `[100,7]` rad，单次目标 `[7]` rad。

## 验证与实机限制

真实日志的纯离线边界回放排除 query 1，保留 query 2–10；对应原网格的动作索引仍为 18、16、14、12、10、8、6、4、2。新发送截止时间为 `160863339510937` ns，计算时剩余 9.578166 ms；相对原发送结束时刻仍剩 6.707827 ms。聚合目标最大变化约 0.000310519 rad。

回归检查覆盖相同 200 ms 边界、过期不推进提交参考、原生暂存前/发送前过期不调用 `OnSetSend()`、真实提交失败锁存后仅允许已撤销目标的下使能路径。所有原生回归使用无网络实现的模拟 SDK，不进行硬件故障注入。

离线回放不能证明修复后的实机动作或停止已经成功。下一次仍由操作者执行短时测试，检查 `expired_target_ticks`、原生提交详情、下使能包数量，以及最终反馈是否为 `SERVO_OFF_CONFIRMED`。

本地记录在 `outputs/target_expiration_fix_20261008/`，远端编译/测试/源码指纹验收在 v9 的 `reports/target_expiration_fix_20261008/`。

```bash
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/stw/miniconda3/envs/forceact/bin/python -m pytest \
  tests/test_operator_profile_config.py tests/test_goal_selection.py \
  tests/test_command_session.py tests/test_right_arm_transport.py \
  tests/test_shadow_controller.py::test_late_chunk_indices_and_expiry \
  tests/test_shadow_controller.py::test_temporal_weights_and_completion_order \
  tests/test_shadow_controller.py::test_no_future_availability_duplicates_or_wrong_run -q
```
