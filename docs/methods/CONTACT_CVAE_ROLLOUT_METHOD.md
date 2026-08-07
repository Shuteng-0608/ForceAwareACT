# Contact-CVAE Rollout 方法与执行逻辑

本文说明当前 MuJoCo rollout 中从 checkpoint 加载、在线输入构造、模型推理、
action chunk 执行、控制后处理、安全监测到成功判定的完整方法。本文不记录任何
具体 rollout 实验结果。

适用的主入口为：

```text
scripts/run_mujoco_policy_rollout.py
```

当前协议版本：

```text
rollout protocol: paired_action_executor_rollout_v3
high-rate force:  causal_raw_500hz_last_100_grouped_state_intervals_v2
```

## 1. Rollout 的总体循环

标准 policy rate 为 30 Hz。每个 policy step 的逻辑顺序是：

1. 读取当前 7D qpos、qvel 和 6D 实测 wrench；
2. 为高频模型登记当前 policy-state timestamp；
3. 从连续 500 Hz ring buffer 构造过去 100 点在线力输入；
4. 从两台相机渲染原生 480 x 640 RGB；
5. 用 checkpoint 内置统计量准备图像、qpos 和 force；
6. 在需要 query 的 step 调用 policy，得到 100 步 action/force chunk；
7. 根据 action executor 从一个或多个 chunk 中选择当前动作；
8. 把 normalized action prediction 反标准化为绝对关节目标；
9. 依次做可选 axial bias、单步 delta clip、EMA 和 actuator ctrlrange clip；
10. 若为 execute 模式，把最终命令写入 `data.ctrl`；
11. 推进若干 MuJoCo physics steps；
12. 每个 physics step 都读取实测力、更新 500 Hz ring，并执行 100 N hard stop；
13. 写逐步 CSV；结束时写 summary JSON。

默认最多 600 个 policy steps，在 30 Hz 下对应名义 20 秒。实际每个 policy
interval 的 physics step 数由累计时钟调度器决定，而不是简单截断固定整数。

## 2. Checkpoint 是部署数据契约的唯一来源

新的高频 Contact-CVAE checkpoint 内嵌：

- architecture 与 training version；
- 完整 model config；
- normalization stats；
- model state；
- chunk length、图像尺寸、force rate/window 等输入契约。

rollout adapter 根据 architecture version 重建完全相同的模型并 strict load。
用户传入的 `--chunk-len`、`--force-window-len` 或 force duration 如果与 checkpoint
冲突，程序会报错，不会静默覆盖 checkpoint。

高频模型还要求：

```text
policy_rate_hz == checkpoint.policy_sample_rate_hz == 30
force_window_len == 100
force_window_duration == (100-1)/500 == 0.198 s
MuJoCo physics rate >= 500 Hz
```

这样做保证 rollout 中“同名输入”的维度、采样率和时间范围与训练一致。

## 3. 环境初始化与观测

rollout 按名称解析 7 个 joints、7 个 actuators、2 个 cameras、force/torque
sensors、peg-tip site 和 hole site。环境 reset 后把 qpos、qvel 与初始 control
设到统一初始值，再执行 `mj_forward`。hole offset 如被请求，会在 rollout 前按
明确的 world/body frame 修改并记录。

每个 policy step 使用当前 MuJoCo state：

- `qpos`：7D 当前关节位置；
- `qvel`：仅用于日志，不进入当前 policy；
- `wrench`：3D force + 3D torque；
- 两路 RGB：当前 simulator state 下渲染；
- task geometry：peg tip、hole center 及其距离。

## 4. 在线图像输入

renderer 按 checkpoint 图像高度和宽度创建。两路相机帧按固定 camera order
堆叠：

```text
uint8 [2,480,640,3]
-> float / 255
-> CHW [2,3,480,640]
-> adapter.prepare_images
-> ImageNet normalization
-> [1,2,3,480,640]
```

原生尺寸符合 checkpoint 时不 resize；如果外部输入尺寸不一致，adapter 使用与
训练相同的 bilinear、`align_corners=False`、`antialias=True` resize。由同一个
adapter 承担预处理，防止 rollout 继续使用旧的 224 x 224 legacy 路径。

## 5. 在线 qpos 输入

当前 qpos 使用 checkpoint 内嵌训练统计量逐维标准化：

```text
qpos_normalized = (qpos - qpos_mean) / qpos_std
[7] -> [1,7]
```

当前高频 checkpoint 预测 normalized absolute action。因此新的 checkpoint 不允许
用 delta-action mode 解释输出，避免训练标签与控制语义不一致。

## 6. 连续 500 Hz force ring buffer

### 6.1 采样机制

reset 时当前 wrench 作为第 0 个样本。ring buffer 容量为 100，采样周期为
`1/500 = 2 ms`，并独立保存最近 8 个 policy-state timestamps，以定义最多 7 个
历史区间。

MuJoCo 每推进一个 physics step，buffer 都会观察新的 timestamp/wrench。达到
下一个 2 ms deadline 时记录一个样本；一次 physics observation 最多记录一个点。
如果 physics timestep 太大导致跨过一个应采样点，程序累加 skipped 诊断并立即
报错，而不是伪造或插值样本。

ring 在 policy step 边界不会清空。因此 30 Hz policy 看到的是连续 500 Hz 历史，
不是每帧重新从一个 30 Hz wrench 构造窗口。

### 6.2 policy query 时的打包

在 policy step `t` 先登记当前 state timestamp，然后 snapshot：

```text
force_timestamps : last <=100 native timestamps
force_values     : matching [N,6]
state_timestamps : last <=8 policy timestamps
```

adapter 调用与训练完全相同的 `build_online_force_intervals`：

1. 只保留不晚于当前 state timestamp 的最后 100 点；
2. 按真实 state timestamp 分进最多 7 个 `(left,right]` 区间；
3. 每区间最多 20 点，有效点左对齐；
4. episode/rollout 开头用 mask 表示空位，不复制实测值；
5. 用 checkpoint 的 native-force mean/std 标准化有效点；
6. padding 值标准化后严格置零。

最终模型输入为：

| 张量 | 形状 |
| --- | --- |
| force intervals | `[1,7,20,6]` |
| relative time | `[1,7,20]` |
| sample padding mask | `[1,7,20]` |
| interval padding mask | `[1,7]` |

模型内部先用共享四层 TCN 编码每个区间，再用四层 online-force Transformer
得到 `z_F_online`；随后它作为 query 进入视觉 cross-attention 得到 `z_VF`。

## 7. 部署 latent

默认命令行为：

```text
--contact-latent-mode zero
```

模型会创建严格全零的 `[1,32] z_contact`。rollout adapter 检查输出中的
`contact_latent_source == "zero"`，并要求 `max(abs(z_contact)) == 0`，否则报错。

也可以显式使用：

```text
--contact-latent-mode prior
```

该模式使用 conditional-prior mean，是确定性的；不会在 rollout 中随机 sample。
posterior 不可用于在线部署，因为部署 forward 根本不接受未来 action/force。

zero 与 prior 都只改变 latent 来源，不改变在线图像、qpos、force encoding、
decoder 或 action executor。

## 8. 推理输出与反标准化

每次 policy query 输出：

```text
pred_action         [1,100,7]
pred_force          [1,100,6]
pred_force_highrate [1,100,20,6]
```

adapter 使用 checkpoint 的 action/force mean/std 对前两项反标准化。当前 rollout
执行器只使用反标准化后的 `pred_action`。`pred_force` 用于日志与诊断；
`pred_force_highrate` 当前不参与 action selection、安全停机或反馈控制。

尤其需要区分：

- 模型预测力是辅助输出；
- hard stop 与 safe-success 一律使用传感器实测力；
- 当前代码没有根据预测力在线修正动作。

## 9. Action chunk 的时间对齐

模型在 query step `tau` 预测一个 chunk：

```text
A_tau = [a_tau,0, a_tau,1, ..., a_tau,99]
```

其中 `a_tau,j` 代表绝对 rollout step `tau+j` 的动作预测。temporal executor 在
当前绝对 step `t` 只允许使用满足 `j=t-tau` 的对齐项；不能把不同绝对时间的
chunk index 直接平均。

例如 step 3 时，可对齐的候选是：

```text
query 0 的 chunk[3]
query 1 的 chunk[2]
query 2 的 chunk[1]
query 3 的 chunk[0]
```

超过 100 步的旧 prediction 会从 executor 中移除。

## 10. 默认 Official temporal aggregation

默认：

```text
--action-select-mode temporal
--temporal-agg-decay 0.01
```

该模式每个 policy step 都重新 query，即 `Q=1`。对当前 step 的候选按
“最旧 prediction 到最新 prediction”排列，令候选序号为 `i=0...n-1`：

```text
w_i = exp(-k*i) / sum_j exp(-k*j)
a_t = sum_i w_i * aligned_action_i
```

这严格复现官方代码的 candidate-row order 与权重公式。必须注意：在这个定义中
`k>0` 会让 `i=0` 的最旧 prediction 权重最大，而不是偏向最新 prediction。
`k=0.01` 是官方等价默认值。

这种执行的目的，是让同一绝对时间来自不同历史 query 的预测形成时间集成，降低
单次预测抖动。但它同时引入“旧规划惯性”，因此比较模型时必须固定相同 executor
与 `k`；若研究新旧 prediction 偏好，应作为独立消融变量。

## 11. 其他 action executor

### 11.1 Recency temporal

```text
--action-select-mode recency_temporal
```

同样每步 query、同样做绝对时间对齐，但权重显式按 prediction age：

```text
age_i = current_step - prediction_step_i
w_i proportional to exp(-k*age_i)
```

因此这里 `k>0` 才表示偏向更新的 prediction。CLI 要求 decay 非负。

### 11.2 Receding chunk

```text
--action-select-mode receding_chunk
--receding-query-interval Q
```

只在 `step % Q == 0` 时 query 一次，在接下来 `Q` 个 step 顺序执行 active chunk
的 `chunk[0:Q]`。约束为 `1 <= Q <= K`：

- `Q=1`：每步重规划，只执行最新 chunk 的第 0 项；
- `Q=K=100`：预测一次后完整开环执行 100 项，再重新 query；这也是默认 Q；
- 中间 Q：固定长度的 receding-horizon 执行。

非 query step 复用缓存 chunk，不再次运行模型，也不把新的在线观测送入 policy。

### 11.3 固定 chunk index

`first`、`mid`、`last` 或 1-based 数字索引会每个 step 重新 query，但始终选择新
chunk 的固定 index。它们不是严格的绝对时间 receding-horizon 控制：例如每帧
选择 `last`，实际总是在执行约 99 步以后所预测的目标。因此主要用于诊断，不应
与默认 temporal 结果混为同一控制方法。

### 11.4 Signed-age temporal 与 latest-only

单次 MuJoCo rollout 支持统一的有符号年龄参数：

```text
--action-select-mode signed_temporal
--temporal-agg-decay signed_k
```

该模式固定每个 policy step query，即 `Q=1`，并使用：

```text
w_i proportional to exp(-signed_k * age_i)
```

- `signed_k < 0`：偏向旧 prediction；
- `signed_k = 0`：所有有效 prediction 等权；
- `signed_k > 0`：偏向新 prediction。

官方 `temporal --temporal-agg-decay 0.01` 在 `Q=1` 下与
`signed_temporal --temporal-agg-decay -0.01` 完全等价。summary 同时记录原始
CLI 参数与 `temporal_signed_decay_equivalent`，避免两套符号约定混淆。

精确的最新预测端点为：

```text
--action-select-mode latest_only
```

它仍然 `Q=1`，但只执行本帧新 chunk 的 `chunk[0]`；其行为不依赖
`--temporal-agg-decay`。这是有限大正 `signed_k` 的明确端点，不用一个任意大数
近似。每次 summary 都记录 `policy_query_interval=1` 和
`temporal_endpoint=newest`。

## 12. 动作语义与控制后处理

新 checkpoint 的 action prediction 是绝对 joint-position target。action executor
得到当前 `selected_raw_action [7]` 后，控制链为：

```text
target = selected_raw_action
target_with_bias = target + optional_axial_push_dq
delta = clip(target_with_bias - current_qpos, -0.02, +0.02)  # per joint
delta_clipped = current_qpos + delta
ema = alpha * delta_clipped + (1-alpha) * previous_command
ctrl = clip(ema, actuator_ctrlrange)
```

默认 `max_delta_q=0.02`、`ema_alpha=1.0`，所以 EMA 默认不改变命令，但 per-joint
delta clip 和 actuator range clip 仍生效。只有 `--execute-actions` 时才把 `ctrl`
写入 simulator；否则是 dry-run，仍执行完整推理与日志路径但不应用策略命令。

optional axial push 默认关闭。开启后会根据 peg-tip Jacobian 将沿 hole axis 的小
笛卡尔推进转换为 joint bias，并在后处理前加入目标。模型公平比较必须为所有
策略保持同样的 axial-push、delta clip、EMA 与 ctrlrange 设置。

## 13. Policy 时钟与 physics 时钟

policy rate 默认 30 Hz，physics timestep 由 XML 决定。累计调度器计算：

```text
ideal_physics_steps = 1 / (policy_rate_hz * physics_timestep)
cumulative_target_n = round(n * ideal_physics_steps)
steps_this_interval = cumulative_target_n - cumulative_target_(n-1)
```

当 physics 为 1 kHz 时，30 Hz 无法对应整数 physics steps，调度器会按累计误差
交替发出 33/34 步，而不是永远使用 33 或 34。这保证长时间平均 policy rate
接近请求值。

模型“推理一次的 wall-clock 延迟”和“语义上的 policy rate”是两件事。当前模型
训练目标按 30 Hz state/action interval 定义，rollout 也强制高频 checkpoint 以
30 Hz query；更高的闭环 policy rate 需要重新定义数据标签、时间区间和训练契约，
不能仅靠缩短 simulator wait 直接宣称等价。

协议 v3 逐步记录两种 wall-clock 指标：

- `policy_inference_time_ms`：在 forward 前后执行 CUDA synchronize，只测所选部署
  latent 的 policy forward；
- `policy_step_compute_time_ms`：从读取 observation 到控制命令准备/写入，包含渲染、
  输入构造、policy forward、temporal aggregation 和控制后处理，但不包含 MuJoCo
  physics stepping 与 CSV 构造。

summary 给出两者的 mean、p50、p95 和 max，并用 `1000/policy_rate_hz` 作为 deadline
统计 `policy_deadline_miss_count/fraction`。若使用 `contact-latent-mode=prior`，代码
还会额外运行 zero-latent 诊断 forward；它不计入所选 policy forward 延迟，但计入
完整 step compute 延迟。上述统计用于判断计算预算，不改变 policy 的 30 Hz 时间
语义，也不等价于真实机器人端到端控制延迟。协议 v3 的 summary 还必须保存
rollout `seed`，使配对实验在脱离原始命令后仍可审计随机性设置。

## 14. 实测力安全逻辑

实测 linear-force norm 定义为 wrench 前三维的二范数：

```text
force_norm = ||wrench[:3]||_2
```

它在每个 physics step 检查，而不是只在 30 Hz policy query 检查。

### 14.1 Hard stop

默认阈值为 100 N：

```text
if measured_force_norm > 100 N: stop
```

execute 模式下触发后，把当前 qpos 经过 ctrlrange clip 作为 hold command，停止
当前 physics interval 并结束 rollout。该阈值不使用预测力，也不使用 torque norm。

### 14.2 Safe success

任务成功首先由几何条件定义；rollout 结束后再计算：

```text
safe_success = task_success and max_measured_force_norm <= 40 N
```

40 N 是结果分层阈值，不参与任务是否几何成功的状态机，也不会像 100 N hard
stop 一样立即停止控制。

## 15. 任务成功语义

成功条件与数据采集器统一为 peg-tip site 与 hole-center site 的总欧氏距离：

```text
peg_to_hole_distance <= 0.003 m
```

该条件需要连续维持 0.10 秒。tracker 使用 simulator timestamp 累计真实 dwell
time；条件中断后计时归零。默认首次满足 dwell 后结束 rollout，可用
`--disable-success-stop` 继续运行。

几何成功条件在每个 policy step（默认 30 Hz）采样，而不是在每个 physics step
采样；这里的“连续”严格指连续 policy observations 均满足条件，并用相邻观测的
simulator timestamp 差累计时间。与之不同，100 N hard stop 才是在每个 physics
step 检查。

lateral threshold 只保留为 deprecated CLI compatibility，不参与 task success。
axial/lateral error 仍写入日志用于诊断，但不能替代上述统一成功定义。

## 16. 停止条件与优先级

rollout 可能因以下原因结束：

1. non-finite model/state/sensor value；
2. 当前 policy observation 已超过 100 N；
3. physics interval 内任一步超过 100 N；
4. 达到并启用 success stop；
5. 达到 `max_rollout_steps`。

无论停止原因如何，最终逐步 CSV 和 summary JSON 都会写出。summary 中同时记录
checkpoint/input contract、executor、latent source、policy/physics timing、后处理、
实测力与成功语义，便于后续审计。

## 17. 公平模型对比的固定项

对 Official ACT 与高频 Contact-CVAE 做 rollout 对比时，除模型必需输入不同外，
至少必须固定：

- 相同 MuJoCo XML、初始状态、hole offset 和 seed；
- 相同 policy rate 与 max rollout steps；
- 相同 action executor、Q 和 temporal decay；
- 相同 absolute-action 语义；
- 相同 `max_delta_q`、EMA、ctrlrange clip；
- axial push 同开或同关；
- 相同成功距离、dwell time、40 N safe threshold 与 100 N hard stop；
- 两类 CVAE 均使用其预先声明的部署 latent；当前 Contact-CVAE 默认 zero；
- 安全判定均使用相同的实测传感器通道。

Official ACT 不输入 force；Contact-CVAE 多出的在线 force 是模型定义差异，不能
为了输入“完全相同”而删除。公平性要求的是各自 rollout 输入与各自训练契约一致，
同时动作执行、环境、后处理和判定协议一致。

## 18. 当前方法边界

1. policy 每 30 Hz query 一次，不代表 force safety 只有 30 Hz；安全监测在每个
   physics step，force history 以 500 Hz 连续采样。
2. `pred_force_highrate` 是辅助预测，不是 500 Hz actuator command。
3. temporal aggregation 会显著改变动作执行含义；离线 action L1 不能代替真实
   rollout 对 executor 的验证。
4. 默认 official temporal 的正 `k` 偏向旧 prediction，这是复现语义，不应按
   “正值通常代表偏新”来误读。
5. fixed-index、receding chunk 和 temporal aggregation 是不同控制协议，比较时
   必须明确命名，不能只记录 checkpoint。
6. 当前 single-rollout 标准化完成后，批量 rollout 应复用同一 adapter、executor、
   后处理与成功状态机，而不是另写一套相似逻辑。

## 19. 源码索引

- rollout 主循环：`scripts/run_mujoco_policy_rollout.py`
- checkpoint/input adapter：`src/force_aware_act/inference/rollout_policy_adapter.py`
- action executors：`src/force_aware_act/inference/action_chunk_executor.py`
- timing/postprocess/success：`src/force_aware_act/inference/rollout_protocol.py`
- 500 Hz ring buffer：`src/force_aware_act/inference/high_rate_force_buffer.py`
- 训练/rollout 共用 packer：`src/force_aware_act/high_rate_force.py`
