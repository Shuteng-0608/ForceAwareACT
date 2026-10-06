# 右臂实机原始 HDF5 训练

本入口适配 `real_robot_right_arm_raw_v2`，支持新版高频 Contact-CVAE、Motion-CVAE、Dual-Zero，以及 Official ACT 网络基线。原有模拟训练入口和六维末端 wrench 配置仍独立保留。

完整的数据对齐、缺失掩码、编码流程、CVAE分支、loss公式和预测头定义见 [真机训练模型架构](../architecture/REAL_ROBOT_MODEL_ARCHITECTURE.md)。本文侧重运行方法和检查结果。

## 数据与时间语义

经用户纠正：action 是 `joint_cmd.position` 的七关节绝对目标位置，单位 rad；接触信号来自 `/tianji_marvin_teleop_bridge/estimated_joint_disturbance.effort`，为七维外部关节力矩估计，单位 N·m。它保留独立 `timestamps/disturbance`，历史、未来高频及端点监督均按该流的原生 `log_ns` 采样。`joint_states.effort` 保存为内部关节力矩，仅供审查，当前不输入网络、不作监督、不参与force统计。补偿细节未核实，不额外执行补偿。

原v1数据来源有误，需要新v2转换；读取器及checkpoint合同拒绝旧来源。保留旧文件供追溯，旧Contact/Motion短训练结果已作废，不能继续训练或部署为外力感知模型。

原始文件不重写、不补帧、不插值、不添加末尾保持。`joint_vel` 留在原始文件供诊断，模型不输入速度，也没有新增任务完成分类头。

当前无法确认所有 ROS header 时钟均同步，因此采样规则 v2 仍使用 MCAP `log_ns` 接收时间。`header_ns` 保留用于采集时间、图像年龄和 `log−header` 延迟诊断，不作为“消息已经可用”的依据。接收时间是录制端可用性的代理，不能独立证明真实控制器的可用时间或物理传感延迟；若日后验证了统一时钟，应另行版本化双时间轴规则并重新生成训练索引。

默认配置见 [right_arm.json](../../configs/real_robot/right_arm.json)：

| 项目 | 规则与原因 |
| --- | --- |
| 动作 | 固定100 Hz时间槽，每步10 ms；真实消息在±5 ms内一对一匹配；未匹配位置独立mask，后续动作不前移。v2 用开头2秒action的 `log_ns` 相对10 ms周期的循环均值估计网格相位，不由首条消息单独定相。 |
| 两相机 | 各自选取anchor之前最新可用图像，640×480；允许使用尚未过旧的上一帧，不生成新帧。默认最大接收年龄100 ms。 |
| 当前位置 | 选取anchor之前最新状态，默认最大接收年龄20 ms。 |
| 历史力矩 | `(anchor−200 ms, anchor]`内的真实样本，分20个10 ms区间；保留实际时间和缺口。每区间容量8，溢出明确报错，不丢弃样本。 |
| 未来力矩 | 每个动作区间有3个固定300 Hz监督槽，时间为区间左端后3.333/6.667/10 ms；容差约±1.667 ms，一对一匹配，缺失不前移、不插值。 |
| 力矩端点 | 只监督第3槽，独立mask；前两个槽有效不代表端点有效。 |
| 标签边界 | 超出实际记录覆盖范围的时间槽无效；没有复制最后位置或命令。 |

这些年龄和匹配容差是首版可审查的参数，不是对所有任务都最优的物理常数。匹配会近似小幅抖动，诊断记录时间残差。当前anchor必须有可匹配的当步action、有效状态、两路图像及至少一条历史力矩；ACT也使用同一组anchor以便公平比较。

相位估计只使用开始2秒的接收时间戳，不读取动作数值、接触力矩或任务结果。`data_quality.json` 的逐段 `action_grid` 记录所用时间轴、算法、窗口、样本数、相位纳秒、循环集中度和首尾anchor；`run_manifest.json` 与 checkpoint 保存数据配置及 v2 采样合同。配置缺少相位字段时恢复 v1 的 `first_action` 算法；这不允许读取原始schema v1或复用错误力来源的checkpoint。若相位窗口样本不足或相位分布不稳定，读取器显式报错，要求审查该段。

`chunk_len=100`对应100个10 ms动作步，即约1秒。网络查询频率与动作步长独立：无需每10 ms重新推理。推理缓冲按绝对时间tick取动作，允许低频或异步查询；预测迟到时跳过已经过期的元素，chunk用完即报无可用动作，不自动保持。

查询调度器仍需把预测的起始时刻放在100 Hz动作网格上；例如30 Hz触发可选最近已到达的控制tick作为观测anchor，并记录真实计算完成时间。当前回放验证输入一致性，不接管实际机器人调度。

## 使用方法

以下命令在仓库根目录运行。使用已安装torch、torchvision、h5py的Python；本机对应 `/home/stw/miniconda3/envs/forceact/bin/python`。

只检查数据，不构建网络：

```bash
python scripts/train_real_robot.py \
  --data-root data/real_robot --data-config configs/real_robot/right_arm.json \
  --output-dir outputs/right_arm_inspect --mode inspect
```

单条数据的训练链路检查：

```bash
python scripts/train_real_robot.py \
  --data-root data/real_robot --data-config configs/real_robot/right_arm.json \
  --output-dir outputs/right_arm_contact_smoke \
  --mode smoke --model contact --steps 2 --chunk-len 100 --device cuda
```

`--model`支持 `contact`、`motion`、`dual-zero`、`act`。`smoke`缩小网络宽度和深度并关闭预训练下载，仍使用640×480输入；同一条记录上的误差明确标记为 `same_recording_smoke_check`，不能作为泛化效果指标。

完整网络短测试使用 `--mode test --steps 10`（可显式设置至多300次更新），保留独立验证、原网络规模及明确测试角色；不支持epoch预算或早停。正式队列保持未批准启动状态。

收集多条独立记录后，正式训练：

```bash
python scripts/train_real_robot.py \
  --data-root data/real_robot --data-config configs/real_robot/right_arm.json \
  --output-dir outputs/right_arm_contact_train \
  --mode train --model contact --steps 10000 --batch-size 2 --device cuda \
  --train-sampling-mode episodic_uniform
```

正式训练默认使用原网络规模和ImageNet预训练；需要显式指定总优化步数。显存不够时先降低batch，不应为了通过检查静默缩小原始图像。

### 训练采样方式

通过 `--train-sampling-mode` 选择，四种实机模型共用实现：

| 参数值 | 样本选择规则 |
| --- | --- |
| `episodic_uniform` | 每个采样epoch先随机打乱训练episode顺序，每段在自己的有效anchor中均匀随机选一个起点；每段每轮恰好贡献一个窗口，下一轮重新选点。 |
| `uniform_valid_anchor_with_replacement_v1` | 原默认规则：每步从所有训练段的有效anchor总集合中均匀、有放回抽样；各段出现概率与其有效anchor数成正比。 |

新实验未传此参数时保留原默认方式；上面的正式训练示例显式启用 `episodic_uniform`。恢复时未传参数则继承checkpoint的采样方式，显式指定不同方式会拒绝续训，避免混合两个实验。

episodic模式的训练索引仍复用经过100 Hz对齐和有效性筛选的anchor，不在原生状态计数或含无效槽的网格上选点。某个训练episode没有有效anchor时明确报错，不自动忽略该段。采样选择不使用action或未来力矩数值。

设训练段数为 `E`，batch size为 `B`，则每个完整采样epoch含 `E` 个窗口和 `ceil(E/B)` 次优化更新。轮末不足 `B` 个样本时使用小batch，不跨轮补齐；`B>E` 时每轮一个包含 `E` 个样本的batch。例如16段、batch=2时，每轮16个窗口、8次更新。优化步数预算可能在一轮中间结束，只有完整轮次保证各段出现次数一致。

相同数据、split、采样方式、seed和batch设置下，四种模型取得相同采样序列。episode等权指完整轮次的样本次数；loss仍按既有有效标签规则归约，轮末小batch也正常执行一次更新，未新增逐episode loss加权。归一化仍只统计训练段的原生真实消息。验证继续完整遍历有效窗口，不改为随机episode取点。

100 Hz网格相位属于时间对齐规则；`episodic_uniform`属于训练窗口选择规则，两者分别记录。新规则的采样合同为 `episodic_uniform_valid_anchor_per_epoch_v1`。

自动划分以原始MCAP的`source_sha256`分组，同源派生片段不会分到训练和验证两侧。也可通过`--split split.json`明确提供全部episode：

```json
{"train_episode_ids": ["recording_a", "recording_b"], "validation_episode_ids": ["recording_c"]}
```

相邻窗口不会随机拆成训练/验证。一条记录会拒绝正式训练划分。跨采集会话的独立性仍需在采集和显式split中保证，文件名不同不代表任务条件独立。

恢复训练时重复原来的模型、数据、chunk及batch设置，增加总步数：

```bash
python scripts/train_real_robot.py \
  --data-root data/real_robot --data-config configs/real_robot/right_arm.json \
  --output-dir outputs/right_arm_contact_train \
  --mode train --model contact --steps 20000 --batch-size 2 --device cuda \
  --resume outputs/right_arm_contact_train/last.pt
```

非默认data/optimization配置同样需要重复传入。恢复检查源HDF5内容SHA-256、关节/相机顺序、数据合同、归一化、模型、优化设置及抽样规则；恢复模型、AdamW状态、随机数和下一批采样状态。episodic模式保存当前采样epoch、打乱后的episode顺序、已消费游标和采样器随机数状态，即使在轮次中间恢复，也会先读取剩余episode。旧anchor-uniform checkpoint保留原Tensor采样器状态格式和抽样序列。旧六维wrench checkpoint不可直接续训。已有输出目录不可混入另一实验，也不可用旧checkpoint覆盖更新的日志。

## 输出和比较规则

### 早停、固定验证与独立测试

- `--max-sampling-epochs` 与 `--steps` 二选一；epoch预算要求 `episodic_uniform`，转换为 `epoch * ceil(训练段数 / batch)` 次更新。
- `--validation-anchor-stride 10` 在每段有效anchor内每10个取1个，并包含首尾；索引和SHA-256写入 `validation_indices.json`，恢复时检查索引合同。
- `--selection-metric episode_macro_action_mae_rad` 先计算每段动作MAE，再等权平均。默认旧指标 `action_mae_rad` 仍按全体有效标量汇总。
- `--early-stop-patience 8 --minimum-sampling-epochs 500 --early-stop-min-relative-improvement 0.005` 启用早停。500轮内不计耐心；之后显著改善重置计数，连续8次未达到0.5%改善时停止。`best.pt`仍保存所有验证中最低指标，独立于显著改善阈值。
- checkpoint保存早停参考值、耐心计数及最后验证步数；中途续训保持计数。早停配置和验证规则不可在同一输出目录中切换。
- 显式split允许 `test_episode_ids`。三侧合计必须覆盖每条发现的记录一次，同源片段不能跨侧。测试段不参与训练、归一化、验证或模型选择。
- 训练结束后，`scripts/evaluate_real_robot.py` 使用checkpoint里的统计、源文件指纹和划分执行冻结模型的全量验证/测试，记录逐段、逐关节、预测距离误差以及重复当前qpos的基线。输出不覆盖已有报告。

2026-09-28这20条数据的正确来源版本为 `/home/stw/Downloads/1mm_gap_hdf5_raw_v2`。更正记录见 [外力来源更正](../experiments/REAL_ROBOT_1MM_GAP_FORCE_SOURCE_CORRECTION.md)。旧v1结果仅保留作历史记录。正式实验队列仍未启动；配置在 `configs/experiments/1mm_gap_20_v1/queue.json`，排除Dual-Zero，20,000采样epoch上限及原12/4/4划分保留。

### 输出文件

- `data_quality.json`：每流计数、时间间隔、header与log差值、逐段相位估计、匹配残差、有效anchor及排除原因。
- `sample_traces.json`：开头、中间、末尾样本的来源索引、纳秒时间、观测年龄、标签mask和力矩槽映射。
- `run_manifest.json`：模型、采样、优化配置，训练/验证划分、源文件指纹及归一化；记录采样合同版本、每段有效anchor数，episodic模式另记录每轮窗口数、更新数和轮末batch规则。
- `metrics.jsonl`：训练损失、梯度范数、采样索引、episode ID、anchor网格索引、实际batch大小、部署路径的验证误差和有效标量数；episodic模式另记录当前采样epoch及是否已完成本轮。
- `last.pt`：可恢复checkpoint；正式训练另按验证action MAE保存`best.pt`。
- `summary.json`：步数、样本数、训练段数、采样方式、最终结果、checkpoint保存/恢复检查；episodic模式另记录已完成的完整采样轮次。

归一化只统计训练记录中的原生真实位置、命令和力矩流，每条消息计算一次；不把重叠窗口和mask占位值算进去。各维标准差下限默认0.01，记录在checkpoint中。ACT网络只接收图像与位置，不接收力矩；其位置/action统计与高频模型相同。

实机ACT动作L1按有效标量平均，避免缺失越多损失越小；旧Official ACT入口的原始padding分母保持不变。ACT模型内部执行ImageNet归一化，高频模型由共享读取层执行，均只做一次。

所有高频变体分别屏蔽action、未来力矩样本和端点目标。无力矩标签时训练项为可微零，质量指标为未定义并报告有效计数，不宣称零误差。后验仅在训练使用未来标签；默认正式验证和在线推理使用可部署的zero latent路径，不使用未来标签。

## 离线部署回放

```bash
python scripts/replay_real_robot.py \
  --checkpoint outputs/right_arm_contact_smoke/last.pt \
  --episode data/real_robot/RECORDING/episode.hdf5 \
  --max-samples 10 --device cuda --output outputs/right_arm_replay.json
```

回放检查在线缓存与离线读取的图像、位置、力矩历史和mask一致，随后执行模型推理。它不发布机器人命令。延迟包含输入准备和模型预测、包含首次调用、排除HDF5读取；compact模型的结果不代表完整网络或真实控制链路达到实时要求。实机ROS通信、执行器限位和真实控制频率验收需要后续与机器人接口集成。

## 2026-09-28：episodic采样变更记录

- 新增 `real_robot/training_sampling.py` 的 `RealRobotTrainingSampler` 和模式解析函数；实机训练CLI接入 `--train-sampling-mode`。原默认采样保持兼容，正式训练示例显式采用 `episodic_uniform`。
- checkpoint增加嵌套采样状态比较和恢复前校验，支持轮次中途续训；日志记录采样episode、anchor、实际batch大小和轮次。
- 数据张量布局保持不变：action `[B,K,7]`、历史力矩 `[B,20,8,7]`、未来力矩 `[B,K,8,7]`（默认配置）。轮末只有 `B` 可能减小；不改变100 Hz对齐、历史范围、标签mask或模型loss。
- 93项相关测试通过（26.18秒）。四种模型均在有效anchor数不同的独立合成记录上检查了每轮段数平衡、轮末小batch，以及中途恢复和连续训练的采样序列、完整模型参数和优化器状态一致性；原默认采样也保留精确的generator序列。此结果验证实现，未启动20条新MCAP数据的正式训练。

复现命令（仓库根目录）：

```bash
PYTHONPATH=src /home/stw/miniconda3/envs/forceact/bin/python -m pytest \
  tests/test_real_robot_data.py tests/test_real_robot_inference.py \
  tests/test_real_robot_model_contract.py tests/test_real_robot_training.py \
  tests/test_real_robot_cli.py tests/test_real_robot_training_sampling.py -q
```

## 训练进度监测

轻量只读脚本 `scripts/monitor_real_robot_training.py` 默认监测本次
`outputs/1mm_gap_20_v2/runs/contact_seed0`。在另一个终端执行：

```bash
/home/stw/miniconda3/envs/forceact/bin/python \
  /home/stw/ForceAwareACT_workspace/ForceAwareACT/scripts/monitor_real_robot_training.py \
  --watch --interval 5
```

- 每5秒刷新：进程PID、更新步数、完整采样epoch、最近100步损失均值、最新/最佳验证值、早停计数、checkpoint更新时间，以及GPU利用率与本进程显存。
- epoch表示一个完整的episode采样轮次。本次12段、batch=2，每6次更新完成一轮；百分比以20,000 epoch上限计算。
- ETA显示“跑到设定上限预计还需”。持续监测时采用最近墙钟进度，包括验证/保存停顿；单次快照采用进程累计速度。续训时先积累刷新数据再估算，避免把续训前的步数计入当前进程速度。早停可能提前结束，ETA不预测早停时间。
- `Ctrl+C`退出监测窗口。训练完成或早停后自动退出watch；进程未检测到时保留窗口，便于等待续训。
- 指定其他记录用 `--run-dir /absolute/path/to/run`；单次查看去掉 `--watch`；`--json`输出快照，配合watch输出逐行JSON；`--no-gpu`关闭nvidia-smi查询。
- 只增量读取日志和小型JSON元数据，以文件属性查看checkpoint；不加载权重或数据集，不分配GPU张量。监测不修改训练参数、数据、文件或进程。

新增 `IncrementalMetrics` 和 `WallRate`；动作/力矩训练张量布局保持原样。测试覆盖半行写入、日志替换/截断、历史最优值、续训旧summary、早停完成状态、多进程匹配、非有限指标以及GPU工具缺失。运行：

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/stw/miniconda3/envs/forceact/bin/python -m pytest \
  tests/test_monitor_real_robot_training.py -q
```

## 历史真实记录检查

以下是2026-09-20采用 v1 首条指令定相规则留下的历史验证结果，不代表 v2 新规则的计数：`Tianji_Marvin_M6CCS_260917_190046_0`有4,739个候选anchor，4,525个有效；210个因当步action无匹配标签排除，4个因相机尚不可用或过旧排除。匹配到的action时间残差中位数约0.64 ms，95分位约1.64 ms。记录末尾的未知action/力矩槽保留mask，没有人工延长。

四种模型均完成640×480真实输入的小网络短训练和checkpoint往返检查；Contact-CVAE与ACT还完成完整512维、4层encoder／7层decoder、100步chunk的一次真实前后向（随机初始化、不下载预训练权重）。这验证接口、损失和梯度，不代表模型已经学会任务。离线回放分别检查10个跨整段轨迹的时刻。

2026-09-20验证：142项相关测试通过，包含旧高频/ACT回归、异步缺失、在线离线一致性与正式CLI恢复。两条独立合成记录上，中断恢复和连续训练的模型参数、优化器状态及采样序列完全一致。原始实机HDF5在检查前后内容SHA-256一致。

详细本地结果位于 `outputs/real_robot_adaptation/`，包含 `verification.json`、`full_architecture_checks.json`、各模型的诊断、采样追踪、短训练与回放报告；这些数据和checkpoint不纳入Git。
