# 随机目标与物理接触扰动 rollout：seed-0 报告

执行日期：2026-09-07。执行代码冻结于 `571ccc9`，协议见
[测试协议](MOVING_HOLE_FIXED_PEG_RANDOM_RECOVERY_PROTOCOL.md)。没有重新训练，
没有根据 rollout 结果选择 checkpoint 或调整成功阈值。

## 工程试测验收

- 全仓回归：`767 passed in 300.48s`；新增的象限覆盖测试单独复测 `4 passed`。
- 16 次工程试测均正常退出并通过产物检查：16 个 summary、16 份 rollout
  CSV、32 个双视角视频，以及 8 份物理扰动高频 trace。
- 中心点对照四模型均成功。步数、成功时间、峰值补偿力及终点距离与旧版
  无扰动结果完全一致（下表列出时间和步数）。
- 首个随机目标为 `(X, Z) = (+0.2019, +1.4310) mm`，四模型均失败：
  Official episodic、Official dense 和 Contact episodic 力停；Contact dense
  到达 40 s 上限。未将这些有效失败当作软件故障。
- 孔口附近 +X 2 N 脉冲：四模型完整接受 0.500 N·s 冲量，之后均成功插入；
  但脉冲期间实际插销—套筒接触力均为零，**不能计作接触恢复成功**。
- 插入中段 +X 6 N 脉冲：四模型完整接受 1.500 N·s 冲量，均检测到真实
  插销—套筒接触，并在释放后完成插入及 collector 稳定保持验收。
- 对 8 份扰动 trace 逐值检查数值有限；完整脉冲的积分与理论冲量相差小于
  `1e-7 N·s`；逐 rollout 的固定插销位置在日志中保持不变。
- 逐视频帧数与 rollout 步数匹配；抽看中段接触视频，场景及相机正常。

| 模型 | 无扰动成功步数 | 无扰动成功时间 s | 中段 +X 6 N 脉冲期间腕部补偿力峰值 N |
| --- | ---: | ---: | ---: |
| Official ACT / episodic | 916 | 30.516 | 14.413 |
| Official ACT / dense | 1097 | 36.552 | 16.532 |
| Contact-CVAE / dense | 654 | 21.773 | 15.300 |
| Contact-CVAE / episodic | 656 | 21.860 | 13.820 |

腕部传感器的补偿力与 MuJoCo 逐接触力的模长之和不是同一个量。例如
Official episodic 的这条中段脉冲中，后者峰值为 19.530 N；不要将二者混用。

工程验收通过后继续执行其余 84 个预定案例，复用上述 16 个完成案例。
完整矩阵的最终结果将在完成验收后补充；本节不是 100 次 rollout 的结论。

## 产物与代码阶段

- 输出目录：`runs/moving_hole_fixed_peg_random_recovery_v1_seed0/`。
- `suite_plan.json`：冻结点位、命令、模型和代码指纹。
- `aggregate.csv`、`suite_summary.json`：逐案例结果及分组计数。
- 每个 `case/model/`：`summary.json`、`rollout_log.csv`、两个视频；扰动触发
  后另有 `disturbance_trace.csv`。过程日志在对应 case 目录。
- `b11212b`：物理侧向力脉冲接口与测试。
- `d0d47aa`：四模型配对矩阵、续跑与验收、协议。
- `571ccc9`：执行前点位审查后，确保四个 XZ 象限都被覆盖。

随机目标任务属于训练时未变化的目标位置的局部 OOD；物理脉冲也不是
采集器位移命令扰动的逐字复刻。所有模型只有一个训练 seed，不据此宣称
架构的普遍优越性。恢复时间指释放至最终任务验收，而非单纯力下降时间。
