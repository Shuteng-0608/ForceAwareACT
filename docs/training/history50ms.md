# 历史 4×50 ms、未来力监督 20×10 ms

开发分支：`codex/force-history-50ms`。基于多指标早停版本 `7a87d0d`。
实现日期：2026-10-21。此实验仅改变历史力的分组；历史总长度仍为 200 ms。

## 时间与张量定义

对于当前观测时刻 `t`，四个历史区间按旧到新排列：

```
(t−200ms, t−150ms]
(t−150ms, t−100ms]
(t−100ms, t−50ms]
(t−50ms, t]
```

每个区间保留全部原始有效样本、时间顺序和相对区间起点的实际时间戳。
采用原有 `log_ns` 可用时间语义。没有降采样、插值或预平均；缺失位置由 mask 表示。
区间超过容量时明确报错，不静默丢弃数据。观测锚点仍每 10 ms 滑动一次，
不需要等待未来 50 ms 的数据。

| 项目 | 原实验 | 本实验 |
| --- | --- | --- |
| 历史总长度 | 200 ms | 200 ms |
| 历史分组 | 20×10 ms | 4×50 ms |
| 每个历史区间名义样本数 | 3 | 15 |
| 每个历史区间填充容量 | 8 | 20 |
| 历史力张量 | `[B,20,8,F]` | `[B,4,20,F]` |
| 历史相对时间、样本 mask | `[B,20,8]` | `[B,4,20]` |
| 历史区间 mask | `[B,20]` | `[B,4]` |
| 历史编码器序列 | CLS + 20 个区间 token | CLS + 4 个区间 token |
| 未来力监督 | 20×10 ms | 20×10 ms |
| 未来力张量 | `[B,20,8,F]` | `[B,20,8,F]` |
| endpoint 输出 | `[B,20,F]` | `[B,20,F]` |
| 动作输出 | `[B,20,7]` | `[B,20,7]` |

`F=6` 使用传感器 `ee_wrench` 全六维；`F=3` 只使用同一传感器的 Fx/Fy/Fz。
20 和 8 是填充容量，不代表每个区间实际采集了这么多点。
未来每个 10 ms 区间仍使用 3 个名义 300 Hz 监督位置，其他槽位保持屏蔽；
endpoint 监督目标仍取第三个位置，未来监督仍严格限制在 200 ms 内。

## 编码和隐变量

共享局部卷积编码器现在分别接受历史容量 20、未来容量 8。
卷积层、真实时间编码、区间位置编码和网络可训练参数数量均不变。
池化仍为卷积后有效位置的 **mean、max、last**，128×3 拼接后映射为 512 维 token。
历史张量减少的是区间 token 数量，不是原始力样本数量。

Contact 后验仍以 20 个未来 10 ms 力区间与 20 步动作逐步融合，
仍为 `[CLS, qpos, C0, …, C19]` 共 22 个 token。
Motion 后验仍仅使用动作序列。此次改变的是历史力条件编码，
没有把 Contact 的未来后验序列缩成 4 个 token。
部署前向只使用当前可用观测与历史力，未来动作/力标签不参与推理。

更宽区间会让局部卷积覆盖更长的实际时间、减少 Transformer 的历史 token 数量；
是否改善预测需要后续训练比较，本次测试只证明实现和数据契约一致。

## 配置、入口及记录

新增数据配置：

- `configs/real_robot/right_arm_wrench6_history50ms.json`
- `configs/real_robot/right_arm_sensor_force3_history50ms.json`

两者相对各自原配置只增加：

```json
{
  "history_interval_ms": 50,
  "history_max_samples_per_interval": 20
}
```

新增入口 `scripts/training/train_x14y08_history50ms.sh`，参数依次为：
`[wrench6|sensor_force3] [contact|motion] [test|train] [额外 CLI 参数]`。
入口调用原三维或六维脚本，仅覆盖数据配置与输出目录；此次没有执行此入口。
新结果默认放在独立目录：

```
outputs/1mm_x14y08_55_wrench6_history50ms_multimetric_v1/
outputs/1mm_x14y08_55_sensor_force3_history50ms_multimetric_v1/
```

保留原始数据与划分 40/10/5、episodic_uniform、batch=8、seed=0、
动作 20 步、未来力 20 步、归一化、优化器、学习率与全部损失权重。
训练上限仍为 20000 epoch，最低 3000 epoch，patience=20，
相对改善阈值 0.001，验证每 250 updates，stride=10。
多指标早停及各指标分别保存最佳 checkpoint 的规则沿用原实现。

`run_manifest.json` 和 checkpoint 记录新的历史数据契约；
`prediction_horizons` 明确保存历史区间时长、数量、容量以及未来区间时长、容量。
样本 trace 记录每个历史区间实际有效样本数。
历史分组新契约版本为 `right_arm_available_clock_100hz_torque300hz_history_v3`。

旧数据配置省略新字段时仍按原分组读取，旧 checkpoint 的缺省配置仍可解析。
新旧历史契约不能直接混合续训，恢复检查会拒绝不一致的数据或模型配置。
此次调整了离线实机观测适配器的契约检查；未修改机器人控制脚本，
也未进行实时传感器或机器人运行验证。

## 验证记录

在 `forceact` 环境中运行 CPU 测试；未安装新依赖，未启动真实数据训练。

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/test_real_robot_history50ms.py -q
```

新增测试覆盖边界归属、10 ms 滑动、因果性、空区间、容量溢出、
旧配置解析、新旧采样锚点及归一化一致、未来标签一致、Contact/Motion × 三/六维
前向与反向计算、参数数量不变、推理不读未来标签、离线观测打包一致、
checkpoint 契约恢复和启动脚本参数委托。CLI 恢复测试使用合成数据和标记模型。

主回归运行 195 项测试全部通过（38.08 秒），覆盖实机数据、三/六维输入、
高频力模型及训练、离线推理、CLI、多指标早停和数据准备。
随后补充审核器区间容量测试，运行新分组与 MCAP 审核测试共 25 项，全部通过
（4.10 秒；其中 18 项与主回归重叠，两轮共覆盖 202 项不同测试）。
启动脚本语法及 `git diff --check` 检查通过。

实数据只读核查结果（`outputs/history50ms_review/packing_audit.json`）：

- 本地 raw_v3 数据集 55 条记录，146,546 个有效锚点。
- 所有有效锚点的历史原始样本总数与原 20×10 ms 分组相同。
- 50 ms 区间最多 17 个样本，小于预留容量 20；溢出区间为 0。
- 每条记录抽查首、中、末有效锚点，共 165 个窗口：历史样本值及顺序完全一致，
  未来标签、时间和 mask 完全一致。
- 空区间继续保留 mask，不补造信号；原始数据集未改动。

此次开发在独立工作树完成，运行中的 4090 和服务器训练代码未同步、进程未改动。
