# Paired50 Q=1 Temporal Rollout Pilot 计划

## 1. 目的

本阶段只回答一个问题：两个已经训练完成的策略在固定原点孔位失败，是否对
Q=1 temporal aggregation 的新旧 prediction 偏好敏感。

它不是最终模型性能对比，也不用于判断模型泛化能力。单点、单 seed、每配置一次
rollout 只能用于：

1. 验证新增在线 executor 与离线 Q=1 sweep 的语义一致；
2. 排除某一执行方式立即导致异常动作、安全停机或严重计算超时；
3. 为后续多次重复和多点实验筛选值得保留的 executor。

## 2. 实验矩阵

两个 checkpoint 都运行完全相同的三种执行器，共 6 次 rollout：

| model | executor | Q | 权重语义 |
| --- | --- | ---: | --- |
| Official ACT | signed `k=-0.01` | 1 | 官方 temporal `k=0.01` 的严格等价表示，轻微偏旧 |
| Contact-CVAE | signed `k=-0.01` | 1 | 同上 |
| Official ACT | signed `k=+0.3` | 1 | 明显偏向新 prediction 的离线候选 |
| Contact-CVAE | signed `k=+0.3` | 1 | 同上 |
| Official ACT | `latest_only` | 1 | 只执行当前新 chunk 的 `chunk[0]` |
| Contact-CVAE | `latest_only` | 1 | 同上 |

`k=+0.3`来自验证集离线 action-execution sweep 的候选结果，但离线误差优势不代表
闭环成功率优势，因此仍需真实 rollout。`latest_only`是精确端点，不用一个任意大的
正 `k`近似。

## 3. 固定公平性协议

以下参数不得在两个模型之间变化：

| 项目 | 固定值 |
| --- | --- |
| hole offset | `(0,0,0)` m |
| rollout seed | `0` |
| policy rate | `30 Hz` |
| maximum policy steps | `600`，名义 20 s |
| action mode | absolute `joint_pos` |
| latent | deterministic `zero` |
| EMA alpha | `1.0` |
| per-joint max delta | `0.02 rad/step` |
| hard stop | measured force norm `>100 N` |
| safe-success | max measured force norm `<=40 N` |
| geometric success | site distance `<=3 mm`，连续 `0.1 s` |
| videos | enabled |

模型输入契约仍由各自 checkpoint 决定：Official ACT 不接收力；Contact-CVAE 使用
连续 500 Hz 在线力 ring buffer 和 zero contact latent。所谓公平不是强行让输入模态
相同，而是保持除模型固有输入以外的环境、执行器和控制后处理一致。

## 4. Checkpoint

```text
Official ACT:
runs/paired50_official_act_formal_e2000_b8_seed0/best_policy.pt

High-rate Contact-CVAE:
runs/paired50_highrate_contact_v3_formal_s10000_b8_seed0/best.pt
```

此 pilot 不重新选择 checkpoint，也不使用 rollout 结果回头修改 checkpoint。

## 5. 安全执行入口

先只生成计划，不施加动作：

```bash
cd /home/stw/ForceAwareACT_workspace/ForceAwareACT
conda activate forceact

PYTHONPATH=src python scripts/run_paired50_q1_temporal_rollout_pilot.py
python -m json.tool \
  runs/paired50_q1_temporal_rollout_pilot_b1/pilot_plan.json | less
```

runner 默认是 `PLAN_ONLY`。只有显式加入 `--execute-rollouts` 才会把
`--execute-actions`传给单次 rollout。

建议先逐条运行，以便每次检查视频、力和动作后再进入下一条：

```bash
PYTHONPATH=src python scripts/run_paired50_q1_temporal_rollout_pilot.py \
  --configuration official_act__signed_km0p01 \
  --execute-rollouts
```

审核该结果后，可用同一命令依次替换为：

```text
highrate_contact_v3__signed_km0p01
official_act__signed_kp0p3
highrate_contact_v3__signed_kp0p3
official_act__latest_only
highrate_contact_v3__latest_only
```

也可以一次顺序运行全部 6 条：

```bash
PYTHONPATH=src python scripts/run_paired50_q1_temporal_rollout_pilot.py \
  --execute-rollouts \
  --skip-existing
```

`--skip-existing`只接受通过完整协议审计的既有 `summary.json`，不会把任意同名目录
当作已完成结果。runner 每完成一条都会重建累计 `aggregate.csv`。

## 6. 每条 rollout 的必审计内容

不能只看 `success`。至少检查：

1. `policy_query_interval == 1`；
2. executor、signed decay 或 endpoint 与计划一致；
3. Contact-CVAE 的 `deployment_latent_source == zero` 且 latent max abs 为 0；
4. Contact-CVAE 的高频力有效样本、interval 数和输入峰值正常；
5. `max_force_norm`、stop reason、是否触发 100 N hard stop；
6. 最小/最终 peg-to-hole distance；
7. action、qcmd 和 qpos 轨迹是否出现振荡、滞后或突跳；
8. policy inference p95、完整 step compute p95 和 deadline-miss fraction；
9. 两路视频中的接触路径和最终姿态。

## 7. 阶段通过标准

只有满足以下条件，才进入重复 rollout：

- 6 个 summary 均通过 runner 的协议一致性检查；
- 没有非有限值、输入契约错误或实现异常；
- force hard stop 如发生，可由真实接触行为解释；
- 至少一个 executor 没有明显控制异常，值得增加重复次数；
- 推理延迟统计完整，能判断 30 Hz 计算预算是否满足。

若所有执行器都在同一阶段、以相似轨迹失败，不能继续无限扫描 `k`；下一步应优先
检查目标/控制语义、训练分布覆盖和闭环状态偏移。若结果随 executor 显著改变，再对
候选设置做多 seed、多次重复，之后才扩展孔位。
