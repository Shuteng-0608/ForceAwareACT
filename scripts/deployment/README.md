# 新 PC 的只读设备检查

动作执行、安全保护和目标监测的阶段 1–3 交付见
[`READONLY_CONTROLLER_PHASES_1_3_2026_10_07.md`](../../docs/rollout/READONLY_CONTROLLER_PHASES_1_3_2026_10_07.md)。
新增 `--controller-config` 可以在独立进程中记录 100 Hz 拟执行动作及保护事件。
只支持 replay/shadow，没有运动发送后端；默认真实限值和目标未配置。

## 每次运行显式选择孔位（2026-10-07 新 PC v5）

使用新的 `contact_right_step_v1` 配置时，在只读预测命令中传入
`--goal-id x14y08 --query-hz 50`，即可启用对应孔位的 FK 球形到达监测。
下一次换成如 `--goal-id x16y07` 会重新加载目标并清空停留计时。
支持 x13～x17 与 y07/y08 的十个已知组合；符号编号仅用于查表，不进行坐标单位换算。
本次只检查选中的球（半径 3 mm、连续停留 500 ms），不会把到达其它孔位计为完成。

```bash
# 在仓库根目录离线预览，不连接设备
PYTHONPATH=src:. python scripts/deployment/shadow_contact_readonly.py \
  --goal-id x14y08 --describe-goal
```

可用 `--controller-profile /absolute/path/to/profile` 指向完整配置目录，必须同时传入 `--goal-id`。
新参数不能与旧的 `--controller-config` 混用；错误编号直接报错，没有默认孔位。
选择在 SDK 加载前校验，在本次运行内固定，并保存至 `goal_selection.json` 和 `shadow_report.json`；
记录包含孔位、完整目标/保护配置及文件 SHA256。模型输入张量和预测方式不变，孔位 ID 不输入网络。

详见 [参数及十孔配置说明](../../configs/deployment/profiles/contact_right_step_v1/README.md)。
配套代码和配置已同步至 `/home/shutiao/tianji_deployment/readonly_controller_20261007_v5`。
当前速度预期值 10%、加速度预期值 30%；只读反馈中的比例另行记录，不自动应用参数。
本次 x13y07 验收和实际回显见 [v5 报告](../../docs/rollout/SELECTED_GOAL_NEW_PC_V5_2026_10_07.md)。
后续将首动作门槛调整至 0.05 rad，并部署到独立的 `readonly_controller_20261007_v6` 目录；
两分钟重测见 [v6 报告](../../docs/rollout/FIRST_ACTION_005_NEW_PC_V6_2026_10_07.md)。后续跟随误差和单步门槛保持原值。
v7 随后按用户要求将后续反馈误差也统一为每关节 0.05 rad，并按补充要求将单步跳变统一为每关节 0.03 rad。
配置交付及离线检查见 [v7 记录](../../docs/rollout/FOLLOWING_ERROR_005_PROFILE_V7_2026_10_07.md)；本轮没有实时硬件测试。
该配置随后完成了两分钟完整只读重测，6,000 次预测、12,000 条影子目标通过检查；
结果见 [v7 实时重测报告](../../docs/rollout/V7_X13Y07_READONLY_RETEST_2026_10_07.md)。没有动作发送或控制器参数写入。

## 基础只读工具

基础检查工具用于环境和数据流确认；`shadow_contact_readonly.py` 可加载模型进行预测，均不发送电机控制指令。
程序导入不连接硬件，机器人 reader 的 SDK 调用白名单为
`CheckSDKTypeCompat`、`OnLinkTo`、`OnGetBuf`、`OnRelease`。
连接会占用 SDK 会话并产生连接/状态通信，但不发送使能、清错、模式、工具或动作指令。
不得用厂商 `showcase_link_check` 替代，它包含自动清错操作。

新 PC：`shutiao@192.168.8.105`，工作目录 `/home/shutiao/tianji_deployment`。
Python 环境 `/home/shutiao/miniconda3/envs/tianji`。
新 PC 的源码放在 `src/`，本仓库中的源码放在 `scripts/deployment/`。
厂商动态库放在新 PC 的 `sdk_validated/`，不提交 Git。
采集端库指纹：`47ab2ab6e35cf899eae3978dfb50ee219e05b10f72aedd6bc9ba6a338a8b4011`。
新 PC 原有 SDK 指纹不同，保留原目录不变。

## 运行（新 PC）

```bash
conda activate tianji
cd /home/shutiao/tianji_deployment
python src/check_tianji_environment.py --output reports/environment_check.json
python src/audit_realsense_readonly.py --seconds 120 --output reports/camera_check
./robot_feedback_readonly 192.168.1.190 120 > reports/robot_check.jsonl
python -m pytest tests/test_readonly_deployment.py -q
```

首次编译：

```bash
g++ -std=c++17 -O2 -Wall -I sdk_validated src/robot_feedback_readonly.cpp \
  -L sdk_validated -lMarvinSDK \
  -Wl,-rpath,/home/shutiao/tianji_deployment/sdk_validated \
  -pthread -o robot_feedback_readonly
```

SDK header 必需文件：MarvinSDK.h、PointSet.h、FXDG.h、FxRtCSDef.h、FxType.h。
默认约 300 Hz 轮询。独立帧率是轮询时观察到的更新率，不代表控制器仅运行在 300 Hz。
输出每秒保存一次原始状态样本，全部轮询用于统计，JSONL 中需跳过 SDK 的文本诊断行。
双臂均保存，模型后续使用 `arms[1]` 的右臂七维数据。
角度/速度转为 rad、rad/s；内力矩和外力原始估计分开保存。
未人工校零：`bias_valid=false`，不构造虚假的有效校正外力。

相机：绑定 D455 `261822304024`、D405 `260322275961`；RGB8 与 Z16
均为 640×480@30Hz，自动曝光打开，启动间隔 3 秒。
保存一张 RGB PNG 和接收时间记录，不保存连续视频或训练数据。
SDK pipeline 返回 frameset；本工具用于接收审核，尚未实现生产环境的异步图像缓存。
图像实际采样比训练 compressed/JPEG 解码少了一次有损编解码，后续模型接入需对照。
设备时间戳保留其时钟域；对齐统计使用主机单调接收时间，不把设备时间当作 log_time。
`causal_age_audit` 在 100 Hz 网格上只选择过去已收到的帧，首秒不纳入稳态统计。
接收年龄不是经过独立标定的物理采集延迟。

## 测试（原电脑）

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/stw/miniconda3/envs/forceact/bin/python -m pytest tests/test_readonly_deployment.py -q
```

测试包括模拟 SDK 编译和运行、调用白名单、rad 转换、读失败退出与连接释放、
相机帧号缺失统计及因果时间采样。测试不连接实际设备。
输入图像 shape `[480,640,3]`、qpos/外力 shape `[7]`；本轮没有模型 batch 张量。

## Contact 权重与实时预测日志

权重路径：`/home/shutiao/tianji_deployment/models/contact_seed0_b8/best.pt`。
SHA256：a5999aa40c8a39111366fbbcd9ae99301e70b302a435cffd5b1f63bfc194a855。
已安装本仓库源码为 editable 包，位置 `code/src/force_aware_act`。
源版本、配置和零偏记录必须随模型保存，实时测试不发送任何控制指令。

只在操作者新近确认右臂静止、无接触后执行：

```bash
conda activate tianji
cd /home/shutiao/tianji_deployment
python src/shadow_contact_readonly.py \
  --checkpoint models/contact_seed0_b8/best.pt \
  --reader ./robot_feedback_readonly --seconds 120 --query-hz 5 \
  --operator-confirmed-no-contact --output reports/shadow_new_run
```

使用 `configs/readonly_profile.json` 记录并核对 bridge_both_test.yaml 来源。
软件校零窗口约 2 秒，允许标准差 0.15 N·m，速度上限 0.01 rad/s、位置范围 0.005 rad。
右臂、反馈 300 Hz、SDK version 100343013；只读程序不订阅机器人指令。
不应用阻抗、初始姿态或工具参数；反馈中的控制模式/工具上下文变化立即使校准失效并结束。
不保存 bias 作为以后自动复用的默认值；每次运行重新人工确认并计算。

RGB 直接来自 RealSense SDK，不增加 JPEG 编解码；模型的 /255、resize 和归一化沿用 checkpoint。
设备读取在后台运行，仅选取/复制观测快照时持锁，预处理、张量复制及 GPU 前向均在锁外；首次 GPU 预热在连接设备前完成。
C++ 通过 --stream 输出每次轮询的反馈；Python 按帧号排除重复帧，保留原始与校正外力。
超过 20 ms 的 IPC 样本丢弃并统计；超过 200 ms 使校准失效并结束。
模型仍要求状态接收年龄 <=20 ms、图像 <=100 ms；缺失观测记录 skip，不生成伪造输入。
采集时间使用一次性 epoch/monotonic 偏移映射，record timestamp 与 availability 分开，
作为在线接收可用性时钟适配 v1，不改 checkpoint 的 log_ns 合同。

输出：calibration.json、feedback_samples.jsonl、predictions.jsonl、shadow_report.json。
动作日志单位为绝对关节位置 rad，shape [100,7]；外力输入 [1,20,8,7]，
图像 [1,2,3,480,640]、qpos [1,7]。实时日志不代表控制器已接受动作。

## 稳定查询频率测试

操作者重新确认静止、无接触后运行：

```bash
python src/shadow_contact_readonly.py \
  --checkpoint models/contact_seed0_b8/best.pt --reader ./robot_feedback_readonly \
  --seconds 10 --rate-sweep 10,30,50,60,70,80,100 --confirm-seconds 60 \
  --operator-confirmed-no-contact --output reports/rate_limit_new_run
```

逐档升频，在第一个未通过档位与之前通过档位之间追加一次中点测试，
再对最高通过档位持续验证 60 秒。日志保存首个动作和完整输出形状，不保存全部动作点，
避免大量磁盘写入主导测量；所有模型输出仍实际计算，并通过有限值检查。
判定要求：实际预测频率 >=98% 目标；完整循环 P99 不超周期、超期率 <=1%；
观测跳过 <=0.5%；有效机器人反馈 >=270 Hz、两路相机 >=27 Hz；IPC 过期丢弃 <=1%。
完整循环含取观测、预处理、GPU、输出转 CPU 及日志写入，不含主动等待到下次查询的时间。
结果仅代表当前模型、FP32、batch=1、当前设备和桌面负载、当前日志策略；
超出相机 30 Hz 的查询会使用同一图像搭配新的关节/外力历史，不能产生新的视觉帧。
