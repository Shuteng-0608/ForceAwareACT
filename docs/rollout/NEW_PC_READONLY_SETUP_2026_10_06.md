# 新 PC 环境与只读设备接入记录

日期：2026-10-06（Asia/Shanghai）。范围：环境准备和真实数据流确认。
用户禁止发送任何电机控制指令；本次未进行模型推理、训练或运动测试。
用户将连续检查时长调整为 2–3 分钟，原长检查提前终止。

## 主机与环境

- 新 PC：shutiao-MS-7E92，SSH `shutiao@192.168.8.105`。
- Ubuntu 22.04.5 LTS，Linux 6.8.0-138-generic。
- RTX 4090，约 24 GiB 显存，驱动 595.91.07。
- conda：`/home/shutiao/miniconda3/envs/tianji`，Python 3.10.22。
- numpy 1.26.4，torch 2.12.1，torchvision 0.27.1，h5py 3.16.0。
- PyTorch CUDA runtime 13.0，GPU 矩阵计算通过；torchvision、h5py 和 NumPy/Tensor 转换通过。
- pyrealsense2 2.56.5.9235，pytest 8.3.5；pip check 无依赖冲突。
- 仅在 tianji 环境安装包，未使用 sudo，未改动系统驱动、固件或全局 Python。
- pip 安装将 tianji 的 setuptools 从 83.0.0 调整至 81.0.0，详见完整 freeze 和安装日志。

## 设备来源与设置

参考代码来自原电脑：
`/home/stw/tianji_ws/src/dual_realsense_launch`、
`/home/stw/tianji_ws/src/tianji_io_teleop_bridge`。
这些原目录未修改。

D455 序列号 261822304024，固件 5.17.3.10；D405 序列号 260322275961，固件 5.15.1.55。
初次 USB 枚举只看到 D455，用户重新插拔后两台均被 USB 与 RealSense SDK 识别，USB 3.2。
两台设备实际 USB 链路均为 5000M，分处 Bus 04 和 Bus 06。
图像 RGB8 和深度 Z16 均为 640×480@30Hz；自动曝光启用，两台启动间隔 3 秒。
已保存首帧 RGB，画面能看到插入工装和绿色插入件，没有人为旋转图像。

机械臂 `192.168.1.190`，新 PC 有线 `192.168.1.156/24`；Wi-Fi `192.168.8.105/24`。
两网段已经分开。网络配置保持用户当前设置，有线仍存在较低优先级默认路由。

新 PC 原有 SDK SHA256 为 dd6b7afdbda8de3bafb6797c6f94547302ae60bf285ee8daa9762de45be636d6，
与采集端不同。单独复制采集端已验证库及配套 header 至
`/home/shutiao/tianji_deployment/sdk_validated/`，原 SDK 未覆盖。
使用库 SHA256：47ab2ab6e35cf899eae3978dfb50ee219e05b10f72aedd6bc9ba6a338a8b4011。
ABI 检查通过，动态链接引用只有 CheckSDKTypeCompat、OnLinkTo、OnGetBuf、OnRelease。
厂商 showcase_link_check 包含清错，本次没有执行。

## 机械臂检查结果

- 10 秒短检查通过，随后连续检查在约 111.647 秒提前终止并正常释放连接。
- 33,488 次轮询，约 299.946 Hz；双臂观察到的独立更新数均为 33,488。
- 读失败、重复轮询、无效帧号、非有限值、错误码和异常帧号倒退计数均为 0。
- 接收更新间隔 P50/P95/P99：3.333/3.335/3.347 ms，最大约 3.547 ms。
- SDK OnGetBuf 耗时 P99 约 0.00055 ms，只表示缓存读取成本，不表示网络/物理采集延迟。
- 本次状态样本中双臂 state=0、error_code=0；未尝试改变状态。
- 单位：qpos rad、qvel rad/s、内部力矩和外力原始估计 N·m。
- 外力字段 m_EST_Joint_Force，内部力矩 m_FB_Joint_SToq 单独保留。
- 无人工无接触静止确认，未校零；bias_valid=false，未将原始外力当作有效校正输入。

## 文件与复现

新 PC 工作目录 `/home/shutiao/tianji_deployment`，源码 `src/`，检查记录 `reports/2026-10-06/`。
记录包含安装日志、完整依赖 freeze、SDK/源码指纹、GPU 检查、USB/路由和数据流报告。
机械臂日志文件 robot_10min.jsonl 保留启动时的名称，实际检查已提前停止，时长以 summary 为准。
旧 camera_10min 是中断的记录，正式相机结论采用 camera_2min。

仓库新增工具见 [只读工具说明](../../scripts/deployment/README.md)。
四项测试在原电脑与新 PC 的 tianji 环境均通过；测试采用模拟 SDK，不连接设备。
新增函数包括相机接收审核、summarize_frames、causal_age_audit、环境 collect；未新增模型类。
图像 shape [480,640,3]，关节/外力 shape [7]；没有更改训练和模型输入合同。

后续仍需人工无接触静止确认后的软件零偏记录、实时缓存接入、训练图像预处理对照。
当前没有加载 checkpoint，不构成完整模型部署或实机成功率验收。

## 正式相机检查结果（120 秒）

- D455：3,592 帧，29.933 Hz；帧号跳过 4 帧，无重复/倒退。
- D405：3,595 帧，29.958 Hz；帧号跳过 2 帧，无重复/倒退。
- RGB 与 depth 均收到；RGB 实际 shape [480,640,3]、uint8。
- 接收间隔 P99：D455 33.810 ms、D405 33.798 ms。
- 全段最大间隔：D455 188.753 ms、D405 91.581 ms；首帧保存 PNG 在采集线程中执行，
  启动统计包含此诊断开销，不能直接归因为硬件故障。
- 排除首秒，以 100 Hz 因果网格审核 100 ms 接收年龄限制：
  D455 11,898/11,898、D405 11,888/11,888，有效率均为 100%。
- 稳态最大接收年龄约 33.764/33.776 ms；不是独立验证的物理采集延迟。
- 设备报告时间域 global_time，保留原始时间及主机单调接收时间。
- 相机错误列表为空，两条 pipeline 均正常停止；机械臂会话已释放。

本轮可确认新 PC 的环境与只读设备通信可用。尚未完成零偏校准、生产缓存接入、
模型输入等价验证或预测测试，当前不宣称完整模型部署通过。
