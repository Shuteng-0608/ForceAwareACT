# 模型部署力 HUD（2026-10-09）

## 交付状态与位置

新 PC：`/home/shutiao/tianji_deployment/deployment_controller_20261009_v12`，环境为 `tianji`。
同一目录支持已有 Contact-CVAE 和 ACT 检查点，原 v10/v11 目录保留。
本轮仅安装环境依赖、同步代码和分析保存日志，没有连接机器人、启动相机或发送控制指令。
真实设备软件校零和约两分钟只读联调需要本次静止、无接触确认，目前尚未开始。

## 画面与数据含义

复用 `tianji_io_teleop_bridge/scripts/force_hud_core.py` 的纯 OpenCV 绘制核心。
原文件 SHA256：`c442a51dad0e5d9a75135de607e336aa8dca1b2a1b591343420a38a845c4a8ab`。
除新增来源注释外，复制的正文保持一致；原 ROS 工作空间未修改。

- 右臂圆环、横向接触侧箭头、轴向力、合力。
- 左侧七关节外力矩及当前运行状态、校零状态、指定孔位。
- 右侧右臂基坐标系六维分量、原值合力/合力矩、残差、条件数、FK 目标距离。
- 浏览器支持 `Ring` / `Six-axis bars`、`Raw / filtered`，这些按钮只改变显示。
- 默认 1280×720、30 Hz、因果低通时间常数 80 ms；不是离线中心均值滤波。
- 同一实际反馈包的 q(rad) 与 `external_raw_nm`，减去本次 `calibration.bias_nm` 后再换算。
  不使用 `internal_torque_nm`，不额外做一组显示零偏。
- 参考点为法兰，分量表达在右臂基坐标系。任务轴来自原插孔预览配置的固定轴，仍为未标定方向预览。
  箭头表示横向外力的反方向，即原 HUD 的接触侧语义，不是运动指令。
- 有限且新鲜的高残差样本完整显示，同时提示 `HIGH RESIDUAL`。
  数据过期、非有限估计或校零时的状态/工具上下文改变时隐藏旧数值，明确提示失效。
  “完全显示”不等于“真实针尖力已经标定”。

## 接入结构

SDK 连接仍由原控制进程单独持有。新增可选 `HudTelemetry`，每约 33 ms 写入最新反馈与
生命周期快照，采用固定大小共享缓冲区和非阻塞锁；显示忙时允许跳过快照，不积累队列。
发布位于原控制与保护评估之后，不执行 FK/SVD、图像编码或文件写入。

HUD 为独立 CPU 进程，仅计算 FK/雅可比、滤波和 OpenCV 绘制，提供
`http://127.0.0.1:8787`。浏览器访问的是本机页面，没有机器人控制接口。
测试确认显示进程没有加载 Torch、相机库或机器人运输模块。
HUD 生命周期、错误和显示帧另存本次输出目录的 `hud/`，显示故障不更改控制决策。
开启运行前若 HUD 启动失败，则在相机/SDK连接前退出。

原动作聚合、100 Hz 执行、50 Hz 推理、关节力矩/位置保护及停止路径保持原逻辑。
三个 profile 只更新新增和修改代码的指纹及修订记录，数值配置未修改。

## 如何使用

在新 PC 本地终端进入新目录并激活环境：

```bash
cd /home/shutiao/tianji_deployment/deployment_controller_20261009_v12
conda activate tianji
export PYTHONPATH=src:.
```

以后由操作人员执行既有实机命令时，在 ACT 或 Contact 命令末尾添加 `--hud`。
待终端打印 `Force HUD` 后，在新 PC 浏览器打开 `http://127.0.0.1:8787`。
默认未添加此参数时不创建 HUD；没有 `--enable-control` 时仍仅打印关闭发送的预览。
`--hud` 不使能机械臂，不绕过任何运行授权和配置门槛。
本轮没有执行实机命令。

### 仅查看保存日志（无设备连接）

```bash
python scripts/deployment/replay_force_hud.py \
  --run /home/shutiao/tianji_deployment/command_controller_20261008_v10/reports/x13y07_error010_10s_4bfec9807ce8 \
  --output "reports/hud_replay_$(python -c 'import uuid; print(uuid.uuid4().hex[:12])')" \
  --browser
```

浏览器打开上述本机地址，画面明确标记 `LOG REPLAY`。
播放结束后显示 `STALE`，保留页面直到终端 Ctrl-C；不会保持旧数字为新鲜数据。
不加 `--browser` 则快速输出日志回放报告及 PNG，不启动服务。

### 准备好的实时只读工具

`observe_force_hud_readonly.py` 只启动已验证的 `robot_feedback_readonly` 二进制，
该二进制动态链接的机器人接口只有 `CheckSDKTypeCompat / OnLinkTo / OnGetBuf / OnRelease`。
执行前核对二进制 SHA256，要求本次 `--operator-confirmed-no-contact`，然后读取完整的
至少 2 秒稳定窗口；不使能、不设置参数、不切换模式、不发送目标或下使能。
单次观测校零只用于这次 HUD 观测，后续实机 rollout 会按原流程重新校零。

经本次现场确认后，使用以下工具联调；本轮尚未执行：

```bash
python scripts/deployment/observe_force_hud_readonly.py \
  --reader /home/shutiao/tianji_deployment/robot_feedback_readonly \
  --goal-id x13y07 \
  --seconds 120 \
  --operator-confirmed-no-contact \
  --output "reports/hud_readonly_$(python -c 'import uuid; print(uuid.uuid4().hex[:12])')"
```

## 验证结果与范围

本地及新 PC：相同 75 项测试通过，包括共享快照、SDK外力来源/同一零偏、高残差不屏蔽、
因果滤波、重复帧不更新数据年龄、校零上下文、HTTP模式切换、控制生命周期、关闭发送预览、
力换算几何和 profile 指纹回归。

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/test_deployment_force_hud.py tests/test_rollout_external_wrench.py \
  tests/test_command_session.py tests/test_act_deployment.py \
  tests/test_operator_profile_config.py tests/test_goal_selection.py tests/test_realtime_gc.py
```

- 最后一次 Contact 10 秒日志含准备和停机共 4364 帧反馈；回放输出 416 张显示时刻。
- 回放中高残差结果保留，108 个显示时刻出现残差超过 50%。
- 285 个运行期原值与此前离线曲线逐帧核对，六维分量最大差为 0。
- 新 PC 回放绘制 P99 约 3.17 ms（不含真实相机接收和 GPU 推理竞争）。
- 本地发布 5000 个快照，序列化＋共享缓冲写入 P99 约 0.042 ms；调度偶发最大值约 3.22 ms。
  非阻塞锁保证不等待显示读取，但不是对整机调度延迟的上界保证。
- 实际相机＋推理＋控制开启 HUD 后的长期时序尚未实测，需要后续只读及操作员 rollout 验收。

新 PC仅在 `tianji` 环境增加 `opencv-python-headless==4.11.0.86`，`--no-deps`安装，
原 NumPy 1.26.4 保留。HUD不需要 ROS、桌面 OpenCV 窗口或额外机器人连接。

## 文件、主要对象和形状

| 文件 | 主要对象/函数 | 输入与输出 |
|---|---|---|
| `control/hud_telemetry.py` | `HudTelemetry`, `session_snapshot` | 实际反馈包＋生命周期 → 一个最新 JSON 快照 |
| `control/wrench_estimation.py` | `flange_jacobian`, `equivalent_wrench` | q `[7]`、tau `[7]` → J `[6,7]`、wrench `[6]`及诊断 |
| `visualization/tianji_hud_render.py` | 原 `HudModel`, `RingMapping`, `render_hud` | 显示状态 → BGR `uint8 [720,1280,3]` |
| `visualization/deployment_hud.py` | `DeploymentHud`, `HudProcess`, `export_replay` | 最新反馈或日志 → HUD/JPEG、状态、回放报告 |
| `scripts/deployment/replay_force_hud.py` | `main` | 保存日志 → PNG或本机浏览器回放 |
| `scripts/deployment/observe_force_hud_readonly.py` | `calibration_candidate`, `main` | 只读实际反馈 → 观测校零、HUD、报告 |

现有 `command_worker.py` 添加可选遥测参数；共用 `run_contact_controller.py` 添加 `--hud`参数，
ACT通过原共用入口自动支持。原离线曲线脚本改为导入同一纯数学换算函数，保持其导入接口。
配置为 `configs/deployment/force_hud.json`；新增测试为 `tests/test_deployment_force_hud.py`。
本轮没有提交 Git commit。
