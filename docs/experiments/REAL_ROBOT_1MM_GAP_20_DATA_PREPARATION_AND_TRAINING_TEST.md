# 1 mm gap：20条数据准备与训练测试记录

> **信号来源更正（2026-09-28）：本页是历史v1测试记录。** 用户明确 `joint_states.effort` 是内部关节力矩，外力估计来自 `estimated_joint_disturbance.effort`。本页Contact/Motion测试使用了错误来源，外力感知结论已作废；旧文件和模型仅供追溯。正确v2数据及重测见 [外力来源更正](REAL_ROBOT_1MM_GAP_FORCE_SOURCE_CORRECTION.md)。正式训练尚未启动。

日期：2026-09-28。按用户最后指令，完成准备和短训练测试后先汇报，正式训练未启动。

## 数据准备

- 原始MCAP：`/home/stw/Downloads/1mm_gap_mcap`；20个源文件SHA-256再次核对均未变化。
- 转换目录：`/home/stw/Downloads/1mm_gap_hdf5_raw_v1`；20条HDF5，总计26,678,893,003 bytes（26.68 GB / 24.85 GiB）。
- 每条均通过第二次完整源读取：选定数值float64字节、原始时间戳、JPEG字节和解码RGB逐项一致。每条转换后保存HDF5 SHA-256。
- 对齐采用log_ns、开头2秒循环均值相位、100 Hz动作、300 Hz力矩、200 ms历史及独立mask；转换后相位和有效anchor与原审核记录一致。
- 标签保留四个条件组与全部成功插入的用户确认来源。

| 划分 | episode | 有效anchor |
| --- | ---: | ---: |
| 训练 | 12 | 40,719 |
| 验证 | 4 | 12,690 |
| 独立测试 | 4 | 13,784 |

每组3/1/1，固定划分seed=20260928；仅训练段的原生流参与归一化。split和优化配置位于 `configs/experiments/1mm_gap_20_v1/`。

## 完整网络10步测试

三个模型使用512维、4层encoder/7层decoder、ImageNet预训练ResNet18、双640×480图像、100步chunk、batch=2、FP32、episodic_uniform。测试使用12条训练数据，验证仅稀疏抽取（stride=1000），不用于正式模型效果结论。

| 模型 | 每次更新含读取 / s | 峰值张量显存 / GiB | checkpoint往返 |
| --- | ---: | ---: | --- |
| Contact-CVAE | 0.206 | 2.42 | passed |
| ACT | 0.142 | 1.97 | passed |
| Motion-CVAE | 0.161 | 2.39 | passed |

时间含首次训练调用和样本读取，不含模型初始化、数据指纹计算或checkpoint I/O；短测不是完整训练时长承诺。

## 双轨迹300步拟合检查

选择训练集中的x16y08与x15y08各一条，重新只用这两条计算统计；其余记录隔离。仍为完整网络和batch=2，各模型300次更新。

| 模型 | 前30步平均动作L1 | 后30步平均动作L1 | 降幅 |
| --- | ---: | ---: | ---: |
| Contact-CVAE | 0.7475 | 0.2051 | 72.6% |
| ACT | 0.7724 | 0.1713 | 77.8% |

上述L1为归一化训练损失；每步随机起点不同，采用前后各30步均值。说明短测试中优化有效，不能据此判断泛化、收敛或1 mm插入成功率。

![短拟合动作损失](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_20_v1/short_fit_action_loss.png)

## 离线回放与回归

- 三种模型各回放10个跨整段真实轨迹的时刻。图像、位置、力矩历史、时间和mask的在线/离线输入一致，均未发布机器人命令。
- 100项相关回归通过（48.21秒）；新增早停中断恢复测试通过（5.63秒）；包含正式队列门控的6项控制测试通过（0.72秒）。去重共102个测试案例。
- 早停恢复与连续训练的采样、模型参数、早停计数和终止步数一致；三侧同源泄漏检查和独立评估入口通过。

复现：

```bash
PYTHONPATH=src /home/stw/miniconda3/envs/forceact/bin/python -m pytest tests/test_real_robot*.py -q
```

## 已准备的正式配置，尚未执行

- 模型：Contact-CVAE、ACT、Motion-CVAE；排除Dual-Zero。首轮各seed0，Contact/ACT再补seed1、2，共7个实验。
- 12条训练数据、batch=2、episodic_uniform；20,000采样epoch上限对应120,000次优化更新。
- 至少500epoch；每50epoch（300步）固定stride=10验证；逐段等权动作MAE选择best。
- 早停：连续8次验证未达到0.5%相对改善；保存参考值、计数和采样状态以支持精确续训。
- 恒定主干/主网络学习率1e-5，AdamW，weight decay=1e-4，clip=1.0；原损失权重保持。
- 训练结束后冻结best.pt，再执行全量验证和独立测试。
- `queue.json`标记为prepared_not_started_per_user_request；队列入口拒绝此状态启动。正式启动需要用户新的指令，不能由测试结束自动触发。

## 文件与实现

- 新增 `RealRobotTrainingSampler`（前次）、`EarlyStoppingConfig`、`EarlyStoppingController`、固定验证索引/逐段聚合工具。
- 更新训练入口、checkpoint校验和部署路径指标；新增准备、独立评估和顺序实验队列脚本及对应测试。
- 张量布局保持action `[B,K,7]`、历史 `[B,20,8,7]`、未来力矩 `[B,K,8,7]`；轮末允许小batch。
- 原始数据、ROS部署接口和未来标签隔离规则保持原合同。

机器可读报告：`/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_20_v1/training_test_report.json`；全部日志、模型和回放结果：`/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_20_v1/profiles`。
