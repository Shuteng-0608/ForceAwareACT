# 多指标早停与分别保存最佳 checkpoint

开发分支：`codex/multi-metric-early-stopping`。基于 `2529a07`。

## 目的和范围

原规则只用动作验证误差决定停止和 `best.pt`。本次使带力模型能够在动作
暂时停滞、力预测仍改善时继续学习，并保存不同目标各自最佳的模型。

新增 `--early-stop-metrics multi`。通用 Python 入口的新运行默认仍为 `action`，
保留历史使用方式；两个 x14y08 启动脚本显式选择 `multi`。

没有改动网络结构、输入、目标、归一化、数据集、采样、损失函数、优化器、
学习率或验证锚点。没有修改实机部署代码。此次交付不启动真实数据训练。

## 指标定义

所有指标来自固定验证集的部署前向计算，按 episode 等权汇总。
未来标签只用于评估误差，不参与验证时的隐变量编码。

| 模型/表示 | 独立监控指标 |
| --- | --- |
| ACT-baseline | 动作 MAE（rad） |
| 传感器三维力、历史法兰三维力 Contact/Motion | 动作 MAE；三维力综合 MAE（N） |
| 传感器六维力 Contact/Motion | 动作 MAE；三维力综合 MAE（N）；三维力矩综合 MAE（N·m） |
| 历史七维关节力矩模型 | 动作 MAE；关节外力矩综合 MAE（N·m） |

每种力学量内部独立计算：

```
综合 MAE = (2 × endpoint episode-macro MAE + interval episode-macro MAE) / 3
```

2:1 沿用现有 endpoint/interval 损失的系数比例；这是验证指标的固定组合，
不改变训练损失，也不把 rad、N 与 N·m 混合求和。三维模式不读取力矩指标。
多指标模式要求 `--selection-metric episode_macro_action_mae_rad`，并使用留出验证集。

## 早停规则

各指标拥有独立的 `reference_metric`、`bad_checks` 和 `stopped`。

1. 完成 epoch ≤ 3000 时更新历史最佳参考值，计数保持 0。
2. 之后，当前值严格小于 `reference × (1 − 0.001)` 时，更新该指标参考值、计数归零。
3. 否则仅该指标计数加一。
4. 所有有效指标的计数同时达到 20，才停止训练。
5. 已停滞的单个指标如果后来改善，可以重新归零；单指标停滞不是永久状态。
6. 最大训练预算仍为 20000 epoch。`patience=0` 仍关闭早停，但多目标最佳模型照常保存。

每次验证仍间隔 250 次更新。40 条训练数据、batch=8、episodic_uniform
对应每 epoch 5 次更新，即每 50 epoch 验证。

缺失、NaN、Inf、负数指标会明确报错，不能被当作“无改善”而悄悄触发早停。

## checkpoint 和记录

| 文件 | 选择标准 |
| --- | --- |
| `best.pt` | 动作验证 MAE 最低；名称与历史部署入口兼容 |
| `best_force.pt` | 三维力综合 MAE 最低；仅三维/六维带力模型 |
| `best_moment.pt` | 三维力矩综合 MAE 最低；仅六维带力模型 |
| `best_torque.pt` | 七维关节外力矩综合 MAE 最低；仅历史七维表示 |
| `last.pt` | 最近一次保存或停止时的状态 |

每个最佳文件包含该时间点的完整模型、优化器、采样器和随机数状态。
保存严格数值最优，不使用 0.1% 的早停阈值筛掉微小但真实的最佳值。
`best.pt` 始终代表动作最佳，不自动选择“综合更好”的部署策略。

新增记录：

- `run_manifest.json/run_config/validation_objectives`：版本、指标名称、单位、组合系数、文件名和停止规则。
- `metrics.jsonl/validation_objective_values`：每次验证的各目标值。
- `metrics.jsonl/best_checkpoints`：每个目标的最佳值、step、epoch、原始分项和对应文件。
- `metrics.jsonl/early_stopping`：分别记录各指标参考值、计数和全局停止状态。
- checkpoint 的 `multi_metric_state`：即使禁用早停也保留完整的最佳选择状态。
- 最佳文件的 `checkpoint_selection`：该文件为何被选中及当时对应分项。
- `summary.json`：所有最佳记录和最终控制状态。

监控脚本显示每个目标的当前值、最佳值、单位、参考值、计数和 checkpoint，
并明确提示“全部指标停滞后才停止”。旧格式日志仍可直接读取。

## 配置保持不变

| 项目 | 值 |
| --- | --- |
| 数据集与划分 | 原 raw_v3；原 40/10/5 split |
| batch / seed / 采样 | 8 / 0 / episodic_uniform |
| 动作、力历史、未来力 | 各 0.2 s（ACT 无力输入/预测） |
| 学习率、骨干学习率 | 1e-5、1e-5 |
| weight decay / 梯度裁剪 | 1e-4 / 1.0 |
| 预算 / 最低 epoch / patience / 相对改善 | 20000 / 3000 / 20 / 0.001 |
| 验证间隔 / stride | 250 updates / 10 |
| 六维 endpoint / interval 损失权重 | 1.0 / 0.5 |
| 三维 endpoint / interval 损失权重 | 0.5 / 0.25 |
| 动作 / KL / Contact prior-match 权重 | 1 / 10 / 1 |

脚本仍引用原配置文件，仅增加早停模式，并使用新的输出目录：

```
outputs/1mm_x14y08_55_wrench6_multimetric_v1/
outputs/1mm_x14y08_55_sensor_force3_multimetric_v1/
```

## 兼容和恢复

- 旧 checkpoint 省略新参数时继续使用旧单指标规则。
- 新 checkpoint 省略 `--early-stop-metrics` 时继承 multi；其余原有恢复参数仍须匹配。
- 同一运行不能在 action 和 multi 之间静默切换；指标定义/权重变化也会拒绝恢复。
- 恢复模型、优化器、RNG 前，校验多指标版本、计数、最佳值和文件选择依据。
- 已满足全局早停的 checkpoint 仍拒绝直接续训；本次不添加延长训练或迁移旧实验的入口。
- 使用 `last.pt` 在同一输出目录继续中断的运行，可保留之前各目标的最佳文件。

## 验证方式

在 `forceact` 环境、此工作树根目录执行：

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/test_real_robot_multi_metric_control.py \
  tests/test_real_robot_multi_metric_cli.py \
  tests/test_monitor_real_robot_training.py \
  tests/test_real_robot_training_control.py \
  tests/test_real_robot_training.py \
  tests/test_real_robot_cli.py -q
```

验证结果：上述 84 项测试全部通过（31.00 秒）；两个启动脚本通过 `bash -n`，
`git diff --check` 通过。对照原启动脚本，除新增模式和输出目录外，其余参数一致。

多指标集成测试使用临时合成 HDF5、一个标记权重和脚本化验证值，验证每个
最佳文件对应不同正确时间点，以及恢复后的状态、日志和不中断执行一致。
不使用真实数据集、GPU 或预训练网络。

真实日志只作离线重放。测试 fixture 来源为此次 4090 Contact/Motion 的验证
序列，附来源 SHA256；不包含图像或 checkpoint。重放旧停止点的计数：

| 模型 | 原停止 epoch | 动作 / 力 / 力矩计数 | 新规则停止 |
| --- | --- | --- | --- |
| Contact | 4050 | 20 / 10 / 21 | 否 |
| Motion | 5500 | 20 / 7 / 13 | 否 |

这只能说明新规则会在旧结束点继续运行，不能推断未记录的后续训练质量。
