# ACT-aligned 模型充分训练与收敛协议

本文定义 Official ACT 与高频 Contact-CVAE 的训练终止、验证选模、续训和产物
契约，不包含实验结果。

## 1. 为什么相同步数不等于都已充分训练

相同 optimizer-step 预算适合严格控制参数更新次数，但两个模型的采样和监督不同：

- Official ACT 每个 epoch 对每条 episode 均匀抽一个 timestep；
- Contact-CVAE 的 data epoch 遍历全部 `(episode,timestep)` windows；
- Contact-CVAE 还优化 endpoint force、原生 500 Hz force 和 prior matching。

因此充分训练实验允许实际 update 数不同，但必须预先固定相同 split、部署路径、
物理 action 指标和停止规则。

## 2. 三层训练边界

1. `target_optimizer_steps`：绝对 global-step 硬上限，只作资源和故障保护；
2. `minimum_optimizer_steps`：最低训练量，此前不允许平台停止；
3. validation patience：达到 minimum 后，连续若干次无有效改善才停止。

所有目标均为绝对值。checkpoint 已在 25,000，target 100,000 表示最多继续到
100,000，而不是追加 100,000。Official ACT target 必须位于完整 episodic epoch
边界；Contact-CVAE 支持 data epoch 中间精确停止和恢复。

## 3. 统一选模指标

两模型都使用：

```text
deployment_zero_action_l1_physical
```

契约为：validation episodes 与训练集不相交；枚举每个 window；部署 latent 为零；
action 反标准化到物理关节单位；对全部有效 action scalar 全局加权平均。Official
sampled loss 与 Contact 的 force/latent 指标继续记录，但只作诊断。

## 4. Patience 的严格语义

设平台参考值为 `r`，新指标为 `m`，relative threshold 为 `delta`。只有：

```text
m < r - abs(r) * delta
```

才清空 patience。任何 `m < best` 都更新 best，即使改善不足 delta。minimum 前的
无改善不累计 patience。停止原因必须是 `maximum_optimizer_steps_reached` 或
`early_stopping_plateau`。

## 5. Validation cadence

Official ACT 每个 epoch 保留 sampled validation，但只有 run start、每个 full
validation interval 和最终时刻运行完整物理验证。Contact-CVAE 在 run start、每个
完整 data epoch和部分终止 epoch 运行完整物理验证。patience 以完整物理验证次数
计数，不以 epoch 或 step 计数。

## 6. 续训与旧 checkpoint 迁移

resume 恢复 model、optimizer、progress、RNG、split 和 normalization。runtime
horizon 独立于不可变训练 config，因而已完成 checkpoint 可指定更大的绝对 target。

续训记录直接父 checkpoint 的解析路径、文件字节数、SHA-256、format version 和
global step。旧 checkpoint 若没有当前物理指标：Contact-CVAE 在 run start 复评
当前恢复权重；Official ACT 同时复评当前权重与内嵌历史 best。迁移前后指标与原因
写入 checkpoint 和摘要。

## 7. Best artifact 契约

best descriptor 包含路径、metric name、metric value、global step 和存储类型。
Contact 的 `best.pt` 是完整可恢复 checkpoint；Official 的 `best_policy.pt` 是轻量
部署权重，resume 所需 optimizer/RNG 位于 `final.pt` 或 `latest.pt`。

新目录续训 Contact 时，父 best 未被超越会创建符号链接；出现新 best 后，原子
保存替换链接本身，不覆盖父文件。

## 8. 监控与结束审计

```bash
python scripts/monitor_act_aligned_training.py RUN_DIR --watch --interval 10
```

监控器自动读取 start、target 和 checkpoint interval，并显示 best step、patience、
minimum 与 relative threshold。`EARLY STOPPED (CONVERGED)` 和跑满硬上限的
`COMPLETED` 是不同语义。

结束后至少检查 summary 的 `passed`/`stop_reason`/reload audit、run control、best
artifact、父 SHA-256、可能的 selection migration，以及 minimum 后是否覆盖了完整
patience 区间。
