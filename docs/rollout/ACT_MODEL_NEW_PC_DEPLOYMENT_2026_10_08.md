# Gap50 ACT 新 PC 部署准备（2026-10-08）

## 模型与部署位置

- 本地检查点：`outputs/1mm_gap_50_10points_v1/runs/act_seed0_b8/best.pt`。
- 新 PC 检查点：`/home/shutiao/tianji_deployment/models/act_seed0_b8/best.pt`。
- SHA256：`a4d6abb6fa8b1674cca088086d10523ff279050c0d113371c67b7f87495a6989`，1,007,588,781 字节。
- 新 PC 独立工作目录：`/home/shutiao/tianji_deployment/act_controller_20261008_v11`。
- 最优检查点训练 step 3600，模型架构 `official_act_single_arm_v1`。不是训练终止时 step 5200 的 `last.pt`。

本次未连接机器人，没有提交参数、模式、位置目标或下使能指令。新 PC 原 Contact v10 工作目录保留。

## 实现

`scripts/deployment/run_act_controller.py` 调用共用启动流程，以固定 `act` 类型和上述检查点指纹运行。Contact 入口仍默认绑定原 Contact 检查点。指纹以分块方式检查，发生在相机、SDK 初始化前。共享流程另检查 100 帧动作、100 Hz 动作时间基准及 D455/D405 相机顺序。

`check_act_checkpoint.py` 复用 GPU 检查脚本，以合成观测做严格权重加载、有限值和输出形状检查，不导入相机或 SDK。新增入口均有单元测试。三个现有控制 profile 的 manifest 只更新运行代码指纹和变更记录，保护数值保持原值。

模型输入：RGB `[1,2,3,480,640]`、归一化关节位置 `[1,7]`。输出反归一化关节目标 `[100,7]`，单位 rad。ACT 使用评估模式和官方零潜变量推理路径，没有动作真值或未来外力输入。

ACT 不接收外力历史；实时外力读取、静止无接触软件校零、外力矩保护仍保留。测试验证同一图像/关节位置下将外力从 0 改成 20 N·m，ACT 输出完全一致。对应外力仅为合成测试值，未进入真实控制器。

## 与 Contact 的对照条件

两个检查点的数据配置、数据契约、完整归一化字典均逐项相等；共同的图像维度、ImageNet 归一化、关节/动作维度、chunk 长度也相等。各模型仍读取自己检查点的归一化统计。

为了比较模型，第一次 ACT 使用最近 Contact 测试的同一配置：

- 目标 `x13y07`，FK 三维目标球半径 3 mm、持续 500 ms。
- `configs/deployment/profiles/contact_right_collection_error010_v1` 是共用控制与保护配置，名字不决定策略类型。
- 采集时刚度 `[12,12,12,10,9,9,9]` N·m/degree，阻尼 `[0.3,0.3,0.3,0.2,0.2,0.2,0.2]`。
- 控制器速度/加速度 10%；首动作、后续反馈误差各关节 0.1 rad；单步跳变各关节 0.03 rad。
- 50 Hz 模型推理，100 Hz temporal aggregation 与目标提交；聚合权重和时间逻辑与 Contact 共用。
- 沿用 v10 图像 GC 修复、数据时效保护、静止后自动下使能路径。

`--goal-id` 选择目标监测点，不是 ACT 的额外输入。ACT 能否消除已观察到的运动卡顿，需要真实对照日志和现场观察验证；现有 GPU 检查不作这个结论。

## 验证结果

本地和新 PC（conda `tianji`）分别运行以下同一组测试，均 79 passed：

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/test_act_deployment.py tests/test_real_robot_inference.py \
  tests/test_command_session.py tests/test_operator_profile_config.py \
  tests/test_goal_selection.py tests/test_realtime_gc.py
```

RTX 4090 严格加载通过；输出 `[100,7]` 且全为有限值，外力输入张量为 `None`。预热后 5 次合成输入推理（包含预处理）平均 6.985 ms、最大 7.139 ms。这是小样本设备检查，未测真实 rollout 的长期频率、平滑性或成功率。

新 PC 报告：`reports/act_checkpoint_gpu_validation.json`、`reports/act_disabled_preview.json`。本地审计目录：`outputs/act_deployment_preparation_20261008/`，包含元数据、兼容性比较、GPU 检查结果及默认关闭发送的预览。

运输库及操作人员原会话配置与 v10 指纹一致：

- `build/right_arm_transport.so`：`eb9819c3f0111ddcebb89af0f89ff5adf6510ea8fef5777ba54c7c4fb09d6a14`。
- `configs/deployment/command_session_20261008_142911_98c76a.json`：`ce87176baf12c0fba446d71533eda60ed620544a4c0fd0e9920a8819e105d671`。

该个人会话允许硬件、静止后自动下使能。默认 `command_session.json` 仍关闭硬件。ACT 无 `--enable-control` 时只打印预览，不接设备。

## 操作人员首次 3 秒对照

仅由操作人员执行。在机械臂复位、静止、末端无接触且无外力时按现有流程重新校零。实际命令会设置控制参数、切换模式并发送动作；3 秒从首次预测开始计时，初始化、校零和停机时间另计。

1. 进入目录，使用已有 `tianji` 环境：

```bash
cd /home/shutiao/tianji_deployment/act_controller_20261008_v11
```

2. 初始化本次终端和独立输出路径：

```bash
conda activate tianji
export PYTHONPATH=src:.
export TIANJI_SESSION_CONFIG="configs/deployment/command_session_20261008_142911_98c76a.json"
export TIANJI_RUN_OUTPUT="reports/x13y07_act_3s_$(python -c 'import uuid; print(uuid.uuid4().hex[:12])')"
```

3. 操作人员确认本次无接触条件后运行：

```bash
python scripts/deployment/run_act_controller.py \
  --goal-id x13y07 \
  --controller-profile configs/deployment/profiles/contact_right_collection_error010_v1 \
  --session-config "$TIANJI_SESSION_CONFIG" \
  --checkpoint /home/shutiao/tianji_deployment/models/act_seed0_b8/best.pt \
  --transport build/right_arm_transport.so \
  --sdk /home/shutiao/tianji_deployment/sdk_validated/libMarvinSDK.so \
  --robot-ip 192.168.1.190 \
  --enable-control \
  --operator-confirmed-no-contact \
  --seconds 3 \
  --output "$TIANJI_RUN_OUTPUT"
```

本次交付没有执行以上实机命令。实机运行后应对比目标跳变、反馈跟随误差、实际关节速度、目标距离、图像接收间隔及停止确认，先判断差异是否来自策略。
