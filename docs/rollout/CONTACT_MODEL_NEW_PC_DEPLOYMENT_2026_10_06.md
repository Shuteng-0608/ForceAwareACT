# Contact 模型迁移与只读实时预测记录

日期：2026-10-06（Asia/Shanghai）。用户授权迁移模型、软件校零和只记录预测，禁止电机控制指令。

## 文件与来源

新 PC 工作目录 `/home/shutiao/tianji_deployment`，conda 环境 `tianji`。
模型 `/home/shutiao/tianji_deployment/models/contact_seed0_b8/best.pt`，完整训练 checkpoint，未剪裁或改写。
原路径 `/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_50_10points_v1/runs/contact_seed0_b8/best.pt`。
SHA256：a5999aa40c8a39111366fbbcd9ae99301e70b302a435cffd5b1f63bfc194a855，迁移前后匹配。
model_metadata.json 保存 checkpoint 自带模型/数据配置、合同和冻结归一化参数。
model_deployment_manifest.json 记录来源 Git 基线、开发分支、未提交改动及文件指纹。
源码安装位置 `code/src/force_aware_act`，采用 editable 安装；没有迁移训练数据集。
Git 基线 f55f9b5419992f5259bb1c2473df29c10d6b8634，分支 codex/real-robot-sdk-deployment。

## 参数沿用

用户指定参考 `/home/stw/tianji_ws/src/tianji_io_teleop_bridge/config/bridge_both_test.yaml`。
已将原文件及源 SHA256 保存在新 PC configs/。
只读程序从 readonly_profile.json 核对右臂、IP 和 SDK 版本，采用其软件校零参数：

- 机械臂 192.168.1.190，右臂 m_Out[1]，SDK version 100343013，已实测。
- 反馈轮询 300 Hz；校零约 2 秒，std 上限 0.15 N·m，速度 0.01 rad/s、位置范围 0.005 rad。
- 外力 m_EST_Joint_Force，校正值 raw minus fixed bias；内部力矩单独记录，不输入模型。
- 人工确认右臂静止无接触后才计算零偏；每次新运行重新确认，不自动复用旧校准。
- 不应用 YAML 中的控制模式、阻抗、初始姿态或工具写入；不订阅 joint_cmd。
- 模型的状态年龄 20 ms、图像年龄 100 ms 限制保持不变。

RGB 为 SDK 原始 RGB，不增加 JPEG 编解码；仍复用 checkpoint 的 /255、resize 和归一化。
输入张量 images [1,2,3,480,640]、qpos [1,7]、force intervals [1,20,8,7]。
输出绝对关节角 pred_action [100,7] rad；Contact latent_mode=zero；未把任何输出发给机器人。

## 验证与修复

严格加载 checkpoint 所有权重通过，global_step=4000。
4090 上 8 次合成输入检查通过，排除前三次后的 5 个计时样本均值约 10.204 ms，最大 10.379 ms。
该检查没有连接机器人，使用合成图像、冻结统计量构造的状态与外力历史，不代表实际任务表现。

首次真实数据检查成功软件校零并完成一次预测；首次 GPU 计算约 177 ms 导致反馈 IPC 积压，
时效检查终止了程序，退出时因 stdout 未持续排空而 SDK 释放等待超时。
失败记录保留在 model_setup/shadow_2min/，没有发送电机控制或清错指令。
修复内容：设备连接前预热 GPU，退出时持续排空 reader stdout，
丢弃 IPC 年龄 >20 ms 的样本并计数，>200 ms 使校准失效并退出。
模型仍使用原有 20 ms 状态时效限制；不通过放宽输入合同绕过过期数据。

RealRobotPolicyAdapter 新增 predict_prepared，GPU 前向在获取观测快照后执行，
数据接收线程不被 GPU 前向占用同一个锁。原 predict 接口保留并委托新方法。
测试覆盖两个入口输出一致、无效 latent_mode、因果观测、校零静止和帧唯一性。
原电脑相关测试 25 passed；新 PC 部署工具 6 passed。

## 在线时钟与日志

记录设备时间域及主机单调接收时间；启动时固定 epoch-minus-monotonic 偏移。
机器人 timestamp 是 SDK 读取后的主机时间，availability 是 Python 收到并可用于观测的时间。
不同源只按同一主机的接收可用性时钟构造因果输入，不直接比较未经验证的设备时钟。
checkpoint 的 log_ns 字段保持不变，在线映射记录为 fixed_epoch_offset_from_host_monotonic_receive_v1。

实际测试记录在 `/home/shutiao/tianji_deployment/reports/2026-10-06/model_setup/shadow_2min_v2/`：
calibration.json、feedback_samples.jsonl、predictions.jsonl、shadow_report.json。
未运行正式训练，未启用电机或执行插入动作；本轮验证输入/预测接口和运行时序。
复现方式见 [只读部署工具说明](../../scripts/deployment/README.md)。

## 修复后正式结果

真实数据运行 120 秒、查询 5 Hz：599 次预测、1 次过期观测跳过，无错误。
预热后 GPU 前向平均 8.582 ms，P95 9.294 ms；不含观测准备、采集及 IPC。
反馈 reader 实际运行 122.354 秒，36,700 次轮询，299.950 Hz，无重复帧、非有限值或错误码。
Python 接收 36,684 个有效反馈，16 条 IPC 年龄 >20 ms 的样本丢弃并计数，未放宽状态时效。
两路相机接收分别 3,667/3,671 帧。软件零偏通过，校准 J1 标准差约 0.1 N·m，符合 0.15 上限。
reader returncode=0、release_ok=true，两条相机 pipeline 正常停止。

原始 shadow_report.json 中 calibration_valid_at_end=true 表示该次采样没有发现校准失效；
会话释放后这份零偏不再有效，不能自动复用于下一次连接。已修正后续程序的字段：
calibration_valid_for_completed_run 表示本次审核通过，calibration_valid_at_end=false 表示退出失效。
原始测试记录保留，追加 result_summary.json 明确退出后的校准状态。

权重、配置、来源、零偏及预测日志均已保存；没有电机使能、控制模式切换、清错、
工具写入、初始姿态或关节目标发送。当前完成模型迁移与预测链路验证，尚未进行闭环执行。
