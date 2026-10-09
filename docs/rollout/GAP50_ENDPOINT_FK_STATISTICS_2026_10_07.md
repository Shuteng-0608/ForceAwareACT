# 十点位示教末段 FK：方法、运行与证据

日期：2026-10-07。

## 范围

用户要求从 50 条已成功插入的示教末段统计十个不同点位，并通过 FK 分析末端。用户补充：相邻点位间距固定为 25 mm。本次交付是本地离线统计，无任何机器人连接、动作发送、训练任务或部署配置修改。

完整中文报告：[REPORT.md](../../outputs/endpoint_fk_20261007/REPORT.md)。所有数值与图表保存在 `outputs/endpoint_fk_20261007/`。该目录被 Git 的 outputs 规则忽略；保留在当前工作区，不能假定只克隆代码就会带上报告。

## 新增代码

- `scripts/analyze_real_robot_endpoints.py`：可直接运行的离线分析程序。
- `tests/test_real_robot_endpoints.py`：合成几何、时间间断、均值跨角度边界、等示教权重、末段稳定区间和图像时间选择测试。

主要函数及数据形状：

| 函数 / 类 | 输入 | 输出 |
|---|---|---|
| `read_episode` | 只读 raw_v2 HDF5 | `q,dq: [N,7]`，`tau: [M,7]`，各自时间流 |
| `load_model` | 有校验和的 CCS 4.0 配置 | 右臂参数 `[8,4]` |
| `flange_fk` | `q: [N,7]` rad | 齐次位姿 `[N,4,4]`，平移 mm |
| `VendorFK` | 已校验 libKine、同一模型 | 厂家纯数学 FK 参考；输入内部转 deg |
| `pose_summary` | 位姿 `[N,4,4]` | XYZ 中位数、SO(3) 均值、位置/姿态离散范围 |
| `lag_speed` | 相对时间 `[N]`、位姿 `[N,4,4]` | 50 ms 割线线速度/角速度 `[N]`，跨间断无效 |
| `latest_stable_window` | 单条完整历史轨迹 | 最近的满足规则的 0.5 秒窗口或空 |
| `group_stats` | 每条一个末段代表位姿 | 十个点位等示教权重的中心与离散范围 |
| `grid_stats` | 分组中心 | 不受约束仿射拟合、25 mm 刚体拟合、13 条邻接边 |
| `endpoint_checks` | 示教代表位姿、训练划分 | 留一示教/训练中心的端点归属，非完成判据准确率 |
| `continuous_intervals` | 时间 `[N]`、满足条件 `[N]` | 不跨坏样本或 >20 ms 间断的连续区间 |
| `analyze_lookback` | 各条示教、FK 模型 | 最后五秒相对其他四条目标的占据区间及候选窗口 |

## FK 的可追溯性

ROS bridge 路径：`/home/stw/tianji_ws/src/tianji_io_teleop_bridge`。

- `config/bridge_both_test.yaml` 的 `kinematics_config_file` 为空，包默认使用校验过的 `ccs_m6_40.MvKDCfg`。
- `CMakeLists.txt` 固定了模型和纯运动学库 SHA。脚本加载前做相同检查。
- `src/marvin_kinematics_adapter.cpp` 对右臂使用序号 1，并移除工具变换，计算法兰处量。
- NumPy 链参考 `/home/stw/TJ_FX_ROBOT_CONTRL_SDK/kinematicsSDK/FxRobot.cpp` 的 `FX_Init_Robot_Kine_Pilot_CCS` 和 `FX_Robot_Kine_Piolt`；它是特定 CCS 链，不能套为任意 DH 模型。
- 对照使用 `DEMO_C++/libKine.so`，不是源码目录中 SHA 不同的另一份 `libKine.so`。
- 纯数学库的函数包括初始化数学参数、移除数学工具偏移及 FK；没有加载 `libMarvinSDK.so`，没有控制器连接 API。
- 实际末端尖端 TCP 未标定：`MarvinPlatform_EN/tools_cfg.json` 中当前右工具 `tool-1.kine` 全零。

模型文件与 libKine 数值一致不代表已做物理绝对精度标定。统计结果限定在同一机械臂、工具与工装安装条件。

## 运行

从仓库根目录运行，环境使用现有 `forceact`。数据只读，输出应指向仓库内部。

```bash
/home/stw/miniconda3/envs/forceact/bin/python scripts/analyze_real_robot_endpoints.py \
  --data-root /home/stw/Downloads/1mm-gap-10points_hdf5_raw_v2 \
  --model /home/stw/TJ_FX_ROBOT_CONTRL_SDK/CommonConfig/ccs_m6_40.MvKDCfg \
  --kinematics-library /home/stw/TJ_FX_ROBOT_CONTRL_SDK/DEMO_C++/libKine.so \
  --split configs/datasets/1mm_gap_50_10points_v2/split.json \
  --output outputs/endpoint_fk_20261007

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/stw/miniconda3/envs/forceact/bin/python -m pytest \
  tests/test_real_robot_endpoints.py -q
```

依赖 numpy、scipy、h5py、matplotlib，当前环境已有，未安装全局包。关闭 pytest 外部插件自动加载是为了避开现有 ROS `launch_testing` 插件缺少 `lark` 的环境问题，不影响这些纯离线测试。

数据分析脚本重跑会刷新其 JSON/CSV/图表；解释性 `REPORT.md` 是本次人工审阅后的记录，不由脚本自动重写。参数、算法或输入变化后应同步复核报告。

## 核心结果与后续约束

- 全 50 条，10 组×5 条，519,376 条实际关节反馈。
- 1,300 帧与厂家 FK 对照，最大位置差 3.93e-7 mm。
- 未强制网格的拟合步距 24.74/24.27 mm；固定 25 mm 网格 RMS 残差 1.31 mm。
- 末 0.5 秒窗口内位置半径 P95 的中位数 0.42 mm；46/50 满足探索性规则，其余四条前移窗口后满足。
- 同组中心最大偏差 1.95–8.66 mm；最大项主要沿法兰 Z 轴。插入深度/停止记录时机的含义尚需任务定义，不能直接取球形 1 mm 阈值。
- 训练集中心对原验证/测试 20 条末段的孔位归属为 20/20，但固定 3 mm 球只覆盖 14/20。端点归属不等于完成判据准确率。
- 报告中的稳定窗口筛选是离线回顾性分析；不得在在线推理中使用未来数据或“距离记录结束的时间”。
- 用户补充“可能结束记录时机不同，往前看五秒”后，追加了逐条留一示教参考的末五秒检查。最大偏差示教在结束前 3.30～2.80 秒的窗口中心距离参考仅 1.32 mm，2 mm 内连续占据约 0.847 秒，随后离开。这支持不用录制终点直接作为任务终点。
- 全五秒检查：3 mm/1° 内 ≥0.2 秒为 49/50，≥0.5 秒为 47/50；5 mm/1° 内 ≥0.5 秒为 50/50。未增加速度、外力或视觉成功条件，这些覆盖数不能解释为完成检测准确率。
- 新模块 10 项测试通过；状态时间间断不会被算成连续停留。
- 本轮没有修改 `configs/deployment/goals`，没有启用动作后端。下一步应先明确成功事件/插入轴，再编写并回放 FK 目标检测。
