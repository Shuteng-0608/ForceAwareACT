# 1 mm gap：外力来源更正记录

日期：2026-09-28。来源由用户纠正，正式训练继续保持未启动。

## 正确映射

| ROS源 | HDF5 | 时间组 | 当前用途 |
| --- | --- | --- | --- |
| `/tianji_marvin_teleop_bridge/estimated_joint_disturbance.effort` | `observations/joint_external_torque`，float64 `[N_disturbance,7]` | `timestamps/disturbance` | Contact/Motion历史输入、未来高频及端点监督、force归一化 |
| `io_teleop/joint_states.effort` | `observations/joint_internal_torque`，float64 `[N_state,7]` | `timestamps/state` | 原始保留，等待后续使用设计 |
| `io_teleop/joint_cmd.effort` | `actions/source_effort` | `timestamps/action` | 原始保留 |

三者均按每条消息的关节名称提取 `Joint1_R`～`Joint7_R`；力矩单位N·m。补偿算法细节尚未核实，因此新版本不宣称已确认重力或摩擦补偿，也不额外执行补偿。

## 修正范围

- 转换器增加原生disturbance流，独立保存header/log/publish纳秒时间戳、消息索引、sequence及frame_id。
- 原始MCAP及旧v1转换文件保持原样；新输出位于 `/home/stw/Downloads/1mm_gap_hdf5_raw_v2`，schema升级为 `real_robot_right_arm_raw_v2`。
- 外部估计的消息数与状态流不同，禁止按同一行号配对。当前位置仍来自state；力矩历史及未来标签分别使用disturbance的原生log_ns。
- 原100 Hz动作网格、开头2秒相位算法、200 ms历史、300 Hz未来固定槽和缺失mask规则保留。缺失外部估计时保持mask，不以内部力矩替代。
- 归一化仅统计训练集原生disturbance样本，不混入内部力矩、不使用验证或测试记录。
- 离线回放的状态与力矩使用独立游标；缓冲新增 `append_external_torque`，按本流到达时间控制可见性。组合式旧API仅保留显式传入同时间戳外部估计的兼容用途。
- dataset元数据、运行合同及checkpoint记录外力来源和时间组；读取器拒绝旧schema或错误话题、字段、时间组。旧Contact/Motion权重不能直接续训为正确的外力模型。
- ACT没有力输入，原结果不涉及力输入混淆；仍按修正后的共同数据合同重新建立可追溯的测试结果。

原错误来源的Contact/Motion短测试及72.6%损失下降结果已标记作废。原始日志及权重仅保留供追溯；更正通知在 `outputs/1mm_gap_20_v1/FORCE_SOURCE_CORRECTION.json`。

## 配置与测试

仍保留12/4/4划分、episodic_uniform、batch=2、100步chunk、20,000采样epoch上限及原早停配置，排除Dual-Zero。正式队列状态为 `prepared_not_started_per_user_request`；当前工作仅执行数据转换和短测试。

短测试使用 `--mode test`，要求显式 `--steps` 且上限300次更新；不接受epoch预算，不自动接续正式训练。完整网络10步测试涵盖Contact/ACT/Motion；Contact/ACT另做双轨迹300步拟合检查；推理仅离线回放，不发送机器人命令。

回归命令：

```bash
PYTHONPATH=src:/tmp/1mm_gap_python_deps /home/stw/miniconda3/envs/forceact/bin/python -m pytest tests/test_real_robot*.py -q
```

新增测试覆盖：MCAP数值与原生时间逐条来源核验、独立消息数、异步到达、内部力矩改变不影响输入/标签/统计、缺失估计禁止回退、旧来源拒绝，以及短测试预算限制。

实现文件：`scripts/convert_mcap_right_arm.py`、`scripts/prepare_1mm_gap_dataset.py`、`real_robot/config.py`、`reader.py`、`data.py`、`inference/real_robot.py`、`scripts/replay_real_robot.py`、`scripts/train_real_robot.py`及正式队列入口。新增 `_TorqueMessage` 和 `RealRobotObservationBuffer.append_external_torque`；`RawEpisode.torque_timestamps_ns` 暴露独立外力时间轴。训练张量布局仍为动作 `[B,K,7]`、历史 `[B,20,8,7]`、未来力矩 `[B,K,8,7]`，本轮 `K=100`。内部力矩没有新增输入张量。

历史结果在 `outputs/1mm_gap_20_v1/`，更正后的转换、测试日志和机器可读报告在 `outputs/1mm_gap_20_v2/`。处理manifest保存源及输出SHA-256、代码SHA-256、原始来源、更正原因、分组划分、相位及有效anchor变化。

## 完成结果

- 20/20条新版转换通过两次源核验；原始MCAP SHA-256均未变化。新版HDF5共26,702,509,706 bytes（26.70 GB）；内部力矩、其他数值及原有四个时间流与v1逐字节一致。
- 外力估计201,463条、状态201,542条；外部力矩无NaN/Inf。总有效anchor 67,193，训练/验证/测试为40,719/12,690/13,784；相位及anchor数逐段与原审核一致。
- 每个有效anchor紧随的一个10 ms区间：外部力矩固定监督槽匹配率94.0083%，端点95.4087%；相同规则下旧内部力矩流槽匹配率93.5301%。这项统计包含记录边界，区别于99.94%的观测/动作anchor可用率，不代表传输丢包率；未匹配槽保持mask。
- 116项不同的相关测试通过。完整网络Contact/ACT/Motion各10次更新及checkpoint往返通过；各回放10个真实时刻，在线/离线输入相同，未发布机器人命令。

| 模型 | 前30步平均动作L1 | 后30步平均动作L1 | 下降 |
| --- | ---: | ---: | ---: |
| Contact | 0.7271 | 0.1860 | 74.4% |
| Act | 0.7724 | 0.1706 | 77.9% |

两条训练轨迹各300次更新，均使用修正后的数据合同。上述为归一化训练损失，仅证明短测试中优化有效；不能判断泛化、正式收敛或实机插入成功率。

![正确来源下的短拟合损失](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_20_v2/short_fit_action_loss.png)

正式配置仍为20,000采样epoch上限及原早停规则，**正式训练未启动**。机器可读完整报告：`/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_20_v2/training_test_report.json`；命令、split、每步采样记录、来源及代码SHA-256全部保存。
