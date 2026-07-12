# ForceAwareACT MuJoCo Rollout 标准实验手册

本文档规定从训练完成的 checkpoint 到单次 MuJoCo rollout、批量点位实验、多模型 suite、多 seed 稳健性评估、监控、汇总和可视化的标准流程。

命令均从项目根目录执行：

```bash
cd ~/ForceAwareACT_workspace/ForceAwareACT
conda activate forceact
```

Rollout 会执行模型输出并改变 MuJoCo 仿真状态。开始大规模实验前，必须先完成 checkpoint/stats 一致性检查、模型离线推理 smoke、XML 孔装配检查和短程单次 rollout。

标准流程：

```text
checkpoint + normalization stats + MuJoCo XML
  -> 离线 deployable inference smoke
  -> 孔装配和 offset 检查
  -> 单次无动作 rollout 检查
  -> 单次短程 execute smoke
  -> 单次完整 rollout
  -> 批量 grid/random/LHS rollout
  -> [可选] 多模型、多推理模式 suite
  -> [可选] 多 point-set seed / rollout seed suite
  -> 运行监控
  -> 汇总、绘图和传感器分析
  -> 实验归档
```

## 1. Rollout 脚本总览

### 1.1 执行和编排脚本

| 脚本 | 层级 | 主要作用 | 典型使用场景 |
|---|---|---|---|
| `run_mujoco_policy_rollout.py` | 单次 rollout 内核 | 加载一个 checkpoint 和 stats，在一个孔偏移位置执行一次受保护的 MuJoCo rollout | 新模型 smoke、单点调试、单次正式运行 |
| `run_mujoco_hole_grid.py` | 批量点位 | 生成 grid、random 或 Latin hypercube 点位，逐点调用单次内核 | 一个模型/latent/action-select 配置的空间鲁棒性实验 |
| `run_xz_rollout_suite.py` | 多配置 suite | 顺序运行预注册模型和 `mid/temporal` 组合，每个组合调用批量点位脚本 | 当前 peg-hole 五类模型的统一对照实验 |
| `run_xz_multiseed_rollout_suite.py` | 多 seed suite | 对 point-set seeds 和 rollout-seed bases 的组合反复运行 x/z suite，并聚合置信区间 | 多 seed 稳健性与 safe-success 统计 |
| `monitor_xz_rollout_suite.py` | 只读监控 | 显示多 seed suite 当前模型、seed、点位、完成数和队列 | 长时间多 seed 任务监控 |
| `run_peg_fixed_insert_100_experiment.sh` | 历史实验 wrapper | 将特定 100-episode 实验的训练、评估和若干单次 rollout 串起来 | 复现 fixed-insert 历史流程，不作为通用 rollout 入口 |

实际调用关系：

```text
run_xz_multiseed_rollout_suite.py
  -> run_xz_rollout_suite.py
       -> run_mujoco_hole_grid.py
            -> run_mujoco_policy_rollout.py
```

只有 `run_mujoco_policy_rollout.py` 实现模型推理、动作选择、安全裁剪、MuJoCo stepping 和单次日志；其他三个 runner 都是上层编排器。

`run_peg_fixed_insert_100_experiment.sh rollout` 使用固定 checkpoint、较低的历史 hard-force threshold、固定模式组合并默认保存视频。它保留用于历史实验复现；新模型应使用本手册后续的单次/grid 标准命令，不应直接继承其中的阈值。

### 1.2 检查、汇总和可视化脚本

| 脚本 | 作用 |
|---|---|
| `run_policy_inference_smoke.py` | 从一条 HDF5 episode 构造 deployable inference 输入，在进入 MuJoCo 前检查 checkpoint、stats 和模型推理。 |
| `inspect_hole_assembly.py` | 检查 XML 中 hole site/body/geoms 的归属，并验证指定 offset 是否正确移动整套孔装配。 |
| `summarize_rollouts.py` | 汇总一组单次 rollout 目录；优先读取 `summary.json`，缺失时回退到 `rollout_log.csv`。 |
| `plot_hole_grid_results.py` | 从 `grid_summary.csv` 生成规则网格热图和散点图。 |
| `plot_hole_target_map.py` | 从 `grid_summary.csv` 生成靶心式空间成功/安全成功图。 |
| `plot_rollout_sensor_analysis.py` | 绘制单次 rollout 的力、距离、误差等时序曲线，也支持两个 rollout 对比。 |

## 2. Rollout 前置输入与一致性要求

定义本次实验变量：

```bash
CHECKPOINT="outputs/new_dataset/forceaware_contact_cvae_100k/checkpoint.pt"
STATS="outputs/new_dataset/normalization_stats_action_all.pt"
MODEL_XML="../arm_teleop/model/pangu_all_right.xml"
ROLLOUT_ROOT="outputs/new_dataset/rollouts"

ACTION_MODE="action"
CONTACT_MODE="zero"
ACTION_SELECT_MODE="mid"
CHUNK_LEN=10
FORCE_WINDOW_LEN=20
FORCE_WINDOW_DURATION=0.25
POLICY_RATE_HZ=30

mkdir -p "$ROLLOUT_ROOT"
```

检查输入存在：

```bash
test -f "$CHECKPOINT" || echo "missing checkpoint: $CHECKPOINT"
test -f "$STATS" || echo "missing stats: $STATS"
test -f "$MODEL_XML" || echo "missing MuJoCo XML: $MODEL_XML"
```

必须保持一致的训练/rollout 参数：

| 参数 | 一致性要求 |
|---|---|
| checkpoint | 必须是包含 `model_state_dict` 和 `config` 的训练 checkpoint；rollout 根据 `config.policy_variant` 重建模型。 |
| normalization stats | 必须与该模型训练使用的 qpos/action/force 统计语义一致。 |
| `action_mode` | 必须与 stats 中的 `action_mode` 一致；command mode 不接受缺少 action-mode metadata 的旧 stats。 |
| `chunk_len` | 必须与训练模型的输出 chunk 长度一致。 |
| `force_window_len/duration` | 必须与训练时的历史力输入定义一致。 |
| 图像预处理 | 模型输入尺寸、相机和 normalization 语义必须与训练配置兼容。 |
| MuJoCo XML | 必须含训练/rollout 所需的关节、执行器、相机、力/力矩传感器、peg tip site 和 hole site/body。 |

当前 rollout 从 MuJoCo `force_ee` 与 `torque_ee` 传感器构造 6D wrench。代码没有额外实现 bias removal、重力补偿、滤波、符号或坐标系转换。正式比较前应确认它与 HDF5 `observations/ft_wrench` 的物理约定一致。

## 3. 离线 deployable inference smoke（必须）

这个步骤不运行 MuJoCo，也不执行动作。它使用一条真实 HDF5 episode 检查 checkpoint 加载、stats、输入 shape 和 deployable latent 分支。

```bash
EPISODE="$(sed -n '1p' outputs/new_dataset/all.txt)"
INFERENCE_SMOKE_OUT="$ROLLOUT_ROOT/inference_smoke"

PYTHONPATH=src python scripts/run_policy_inference_smoke.py \
  --episode "$EPISODE" \
  --state-index 100 \
  --checkpoint "$CHECKPOINT" \
  --normalization-stats "$STATS" \
  --chunk-len "$CHUNK_LEN" \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --force-window-duration "$FORCE_WINDOW_DURATION" \
  --contact-latent-mode "$CONTACT_MODE" \
  --output-dir "$INFERENCE_SMOKE_OUT"
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--episode` | 用于构造离线在线观测的一条 HDF5 episode。 |
| `--state-index` | 抽查的状态索引，应保证前面有足够的 force history。 |
| `--checkpoint` | 待 rollout 的模型 checkpoint。 |
| `--normalization-stats` | 与 checkpoint 配套的 stats。 |
| `--chunk-len` | 模型预测动作块长度。 |
| `--force-window-len/duration` | deployable inference 使用的历史力窗口。 |
| `--contact-latent-mode` | Contact-CVAE/双 latent 模型的部署分支：`zero` 或 `prior`。 |
| `--output-dir` | smoke 输出目录。 |

Motion-CVAE 和 ACT baseline 不使用 contact latent；共享 CLI 中的 contact mode 对这些分支没有实际作用。

验收条件：checkpoint 严格加载成功，预测动作/力 shape 正确且无 `NaN/Inf`，没有 stats 或模型配置不匹配。

## 4. 检查 MuJoCo 孔装配和 offset（必须）

在批量移动孔位置前，确认 `hole_goal_site` 和孔碰撞几何属于期望的 `wall_task` body，并测试一个小 offset：

```bash
PYTHONPATH=src python scripts/inspect_hole_assembly.py \
  --model-xml "$MODEL_XML" \
  --hole-site-name hole_goal_site \
  --hole-body-name wall_task \
  --test-offset-x 0.002 \
  --test-offset-y 0.0 \
  --test-offset-z 0.002 \
  --offset-frame world
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--hole-site-name` | 成功判据使用的孔目标 site。 |
| `--hole-body-name` | 应被整体平移的孔/墙装配 body。 |
| `--test-offset-x/y/z` | 测试平移量，单位为米；`0.002` 等于 2 mm。 |
| `--offset-frame` | `world` 表示世界坐标偏移，`body` 表示父 body 局部坐标偏移。 |

验收条件：site 位于指定 body 子树内，相关孔 geoms 会随 body 一起移动，实际 offset 与请求 offset 一致。

## 5. 单次 rollout：`run_mujoco_policy_rollout.py`

### 5.1 脚本作用

单次 rollout 内核会：

1. 从 checkpoint metadata 重建模型并严格加载权重；
2. 加载 normalization stats 并验证 action mode；
3. 加载 MuJoCo XML、相机、关节、执行器和传感器；
4. 应用可选 hole offset；
5. 渲染 `ee_cam` 和 `base_top_cam`，构造 qpos 与历史 force 输入；
6. 执行 deployable inference；
7. 反归一化动作，并按 `first/mid/last/temporal` 选择 chunk 动作；
8. 按 action mode 转成绝对 control target；
9. 依次执行 `max_delta_q`、EMA 和 actuator ctrlrange 裁剪；
10. 可选写入 `data.ctrl`，执行 MuJoCo stepping；
11. 记录成功、硬力停止、几何误差、力和动作链路。

### 5.2 先运行无动作检查

不传 `--execute-actions` 时，模型仍会推理并写日志，但不会把预测动作作为执行控制命令：

```bash
NOEXEC_OUT="$ROLLOUT_ROOT/single_noexecute"

PYTHONPATH=src python scripts/run_mujoco_policy_rollout.py \
  --checkpoint "$CHECKPOINT" \
  --normalization-stats "$STATS" \
  --model-xml "$MODEL_XML" \
  --contact-latent-mode "$CONTACT_MODE" \
  --action-mode "$ACTION_MODE" \
  --action-select-mode "$ACTION_SELECT_MODE" \
  --chunk-len "$CHUNK_LEN" \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --force-window-duration "$FORCE_WINDOW_DURATION" \
  --policy-rate-hz "$POLICY_RATE_HZ" \
  --max-rollout-steps 10 \
  --max-delta-q 0.02 \
  --force-stop-threshold 1000 \
  --hole-axis-world 0 -1 0 \
  --output-dir "$NOEXEC_OUT" \
  2>&1 | tee "$NOEXEC_OUT.console.log"
```

检查生成的 `rollout_log.csv` 和 `summary.json`，确认模型、相机、力输入、动作输出、孔几何和日志字段正常。

### 5.3 短程 execute smoke

无动作检查通过后，再明确加入 `--execute-actions`，先运行 30～50 policy steps：

```bash
SMOKE_OUT="$ROLLOUT_ROOT/single_execute_smoke"

PYTHONPATH=src python scripts/run_mujoco_policy_rollout.py \
  --checkpoint "$CHECKPOINT" \
  --normalization-stats "$STATS" \
  --model-xml "$MODEL_XML" \
  --contact-latent-mode "$CONTACT_MODE" \
  --action-mode "$ACTION_MODE" \
  --action-select-mode mid \
  --chunk-len "$CHUNK_LEN" \
  --force-window-len "$FORCE_WINDOW_LEN" \
  --force-window-duration "$FORCE_WINDOW_DURATION" \
  --policy-rate-hz "$POLICY_RATE_HZ" \
  --max-rollout-steps 50 \
  --ema-alpha 0.3 \
  --max-delta-q 0.02 \
  --force-stop-threshold 1000 \
  --success-distance-threshold 0.005 \
  --success-lateral-threshold 0.006 \
  --success-force-threshold 40 \
  --success-hold-steps 15 \
  --hole-axis-world 0 -1 0 \
  --seed 0 \
  --output-dir "$SMOKE_OUT" \
  --execute-actions \
  2>&1 | tee "$SMOKE_OUT.console.log"
```

验收条件：控制量有限，`max_delta_q` 和 ctrlrange 裁剪生效，力值与位置误差合理，没有渲染、传感器或模型异常。

### 5.4 单次正式 rollout 示例

```bash
SINGLE_OUT="$ROLLOUT_ROOT/contact_cvae_zero_mid_nominal_seed0"

PYTHONPATH=src python scripts/run_mujoco_policy_rollout.py \
  --checkpoint "$CHECKPOINT" \
  --normalization-stats "$STATS" \
  --model-xml "$MODEL_XML" \
  --contact-latent-mode zero \
  --action-mode action \
  --action-select-mode mid \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --policy-rate-hz 30 \
  --max-rollout-steps 900 \
  --ema-alpha 0.3 \
  --max-delta-q 0.02 \
  --force-stop-threshold 1000 \
  --success-distance-threshold 0.005 \
  --success-lateral-threshold 0.006 \
  --success-force-threshold 40 \
  --success-hold-steps 15 \
  --hole-site-name hole_goal_site \
  --hole-body-name wall_task \
  --hole-offset-x 0 \
  --hole-offset-y 0 \
  --hole-offset-z 0 \
  --hole-offset-frame world \
  --hole-axis-world 0 -1 0 \
  --seed 0 \
  --output-dir "$SINGLE_OUT" \
  --execute-actions \
  2>&1 | tee "$SINGLE_OUT.console.log"
```

### 5.5 单次 rollout 核心参数

| 参数 | 作用 |
|---|---|
| `--checkpoint` | 模型 checkpoint。 |
| `--normalization-stats` | qpos/action/force stats。 |
| `--model-xml` | MuJoCo 模型 XML；默认路径也是 `../arm_teleop/model/pangu_all_right.xml`。 |
| `--contact-latent-mode` | `zero` 或 deployable `prior`；Motion-CVAE/ACT 分支忽略它。 |
| `--action-mode` | 输出解释方式；必须与训练和 stats 一致。 |
| `--action-select-mode` | `first`、`mid`、`last` 或 `temporal`。 |
| `--temporal-agg-decay` | temporal aggregation 衰减，单次脚本默认 `0.3`。 |
| `--policy-rate-hz` | 策略调用频率；物理仿真在策略步之间继续 stepping。 |
| `--max-rollout-steps` | 最大策略步数。 |
| `--image-width`、`--image-height` | MuJoCo renderer 分辨率，默认 `640×480`。 |
| `--image-size` | 送入模型前的方形 resize，默认 `224`。 |
| `--execute-actions` | 明确允许模型控制量写入执行器；不传时为无动作检查模式。 |
| `--ema-alpha` | 动作 EMA 平滑权重，默认 `0.3`。 |
| `--max-delta-q` | 每个策略步相对当前 qpos 的逐关节最大变化量。 |
| `--force-stop-threshold` | 瞬时 `force_norm` 超过该值立即停止，是硬安全停止。 |
| `--success-*` | 成功条件阈值和连续保持步数。 |
| `--disable-success-stop` | 检测到成功后仍继续运行，但仍记录首次成功。 |
| `--hole-offset-x/y/z` | 孔位置偏移，单位为米。 |
| `--hole-offset-frame` | offset 使用世界或 body 局部坐标。 |
| `--hole-axis-world` | 计算 axial/lateral error 的世界坐标孔轴方向。 |
| `--seed` | 单次 rollout 随机 seed。 |

可选诊断参数：

- `--save-camera-snapshots --snapshot-every 10`：周期保存相机帧；
- `--save-videos --video-fps 30 --video-every 1`：保存视频，批量实验会显著增加磁盘和耗时；
- `--enable-axial-push` 及 `--axial-push-*`：额外诊断偏置，不属于纯策略 rollout，除非实验协议明确要求，否则不要启用。

### 5.6 动作选择模式

| 模式 | 含义 |
|---|---|
| `first` | 使用当前预测 chunk 的第一个动作。 |
| `mid` | 使用 chunk 中间动作。当前批量实验常用。 |
| `last` | 使用 chunk 最后动作。 |
| `temporal` | 聚合当前及历史预测 chunk 中对齐到当前时刻的动作。 |

不同 action-select mode 属于 rollout 控制协议差异。比较模型时必须保持一致，或将它作为显式实验变量。

### 5.7 成功、安全成功和硬力停止

单次 task success 要求以下条件连续满足 `success_hold_steps` 个 policy steps：

```text
peg_to_hole_dist < success_distance_threshold
peg_to_hole_lateral_error < success_lateral_threshold
force_norm < success_force_threshold
```

三个概念不能混用：

- **task success**：上述瞬时条件连续保持规定步数；
- **safe success**：批量汇总中，task success 且整条 rollout 的 `max_force < success_force_threshold`；
- **hard force stop**：任一步 `force_norm > force_stop_threshold` 时立即中止。

`success_force_threshold` 是成功质量阈值，`force_stop_threshold` 是硬停止阈值，二者用途不同。

### 5.8 单次输出

每次至少生成：

```text
output_dir/
  rollout_log.csv
  summary.json
```

`rollout_log.csv` 保存逐 policy-step 的 qpos、wrench、预测动作、动作解释、安全裁剪、实际控制、几何误差、成功 hold counter 和停止原因。

`summary.json` 保存配置、成功状态、停止原因、步数、最大/平均力、最终及最小距离/误差、孔 offset、日志路径和视频信息。

## 6. 批量点位 rollout：`run_mujoco_hole_grid.py`

### 6.1 适用范围

该脚本针对一个固定的：

```text
checkpoint + contact latent mode + action mode + action-select mode
```

生成多个孔位置并逐点调用单次 rollout。它适合任意受单次内核支持的 checkpoint，也是新训练模型最通用的批量入口。

重要：批量脚本对子命令始终加入 `--execute-actions`。先使用自身的 `--dry-run` 检查命令，不要把批量脚本当作无动作模式。

### 6.2 先预览批量命令

```bash
GRID_OUT="$ROLLOUT_ROOT/contact_cvae_zero_mid_lhs50_xz4mm"

PYTHONPATH=src python scripts/run_mujoco_hole_grid.py \
  --sampling-mode latin_hypercube \
  --num-points 50 \
  --x-min -0.004 --x-max 0.004 \
  --z-min -0.004 --z-max 0.004 \
  --point-set-seed 20260702 \
  --rollout-seed-base 31000 \
  --checkpoint "$CHECKPOINT" \
  --normalization-stats "$STATS" \
  --model-xml "$MODEL_XML" \
  --contact-latent-mode zero \
  --action-mode action \
  --action-select-mode mid \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --policy-rate-hz 30 \
  --max-rollout-steps 900 \
  --max-delta-q 0.02 \
  --force-stop-threshold 1000 \
  --success-distance-threshold 0.005 \
  --success-lateral-threshold 0.006 \
  --success-force-threshold 40 \
  --success-hold-steps 15 \
  --hole-axis-world 0 -1 0 \
  --output-root "$GRID_OUT" \
  --dry-run
```

确认打印的所有子命令、点位、路径和参数正确后，再删除 `--dry-run` 正式执行：

```bash
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py \
  --sampling-mode latin_hypercube \
  --num-points 50 \
  --x-min -0.004 --x-max 0.004 \
  --z-min -0.004 --z-max 0.004 \
  --point-set-seed 20260702 \
  --rollout-seed-base 31000 \
  --checkpoint "$CHECKPOINT" \
  --normalization-stats "$STATS" \
  --model-xml "$MODEL_XML" \
  --contact-latent-mode zero \
  --action-mode action \
  --action-select-mode mid \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --policy-rate-hz 30 \
  --max-rollout-steps 900 \
  --max-delta-q 0.02 \
  --force-stop-threshold 1000 \
  --success-distance-threshold 0.005 \
  --success-lateral-threshold 0.006 \
  --success-force-threshold 40 \
  --success-hold-steps 15 \
  --hole-axis-world 0 -1 0 \
  --output-root "$GRID_OUT" \
  --skip-existing \
  --continue-on-error \
  --no-plot-results \
  2>&1 | tee "$GRID_OUT.console.log"
```

### 6.3 采样模式

| 模式 | 控制参数 | 作用 |
|---|---|---|
| `grid` | `--x-offsets`、`--z-offsets`、`--y-offset`、`--repeats` | 运行显式笛卡尔网格；`num-points` 和随机范围不决定网格点数。 |
| `random` | `--num-points`、x/z min/max、point-set seed | 在矩形范围内独立均匀随机采样。 |
| `latin_hypercube` | 同 random | 用 LHS 更均匀覆盖矩形范围，空间鲁棒性实验首选。 |

规则 3×3 grid 示例：

```bash
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py \
  --sampling-mode grid \
  --x-offsets=-0.004,0,0.004 \
  --z-offsets=-0.004,0,0.004 \
  --y-offset 0 \
  --repeats 1 \
  --checkpoint "$CHECKPOINT" \
  --normalization-stats "$STATS" \
  --model-xml "$MODEL_XML" \
  --contact-latent-mode zero \
  --action-mode action \
  --action-select-mode mid \
  --max-rollout-steps 900 \
  --max-delta-q 0.02 \
  --output-root "$ROLLOUT_ROOT/contact_cvae_zero_mid_grid3x3" \
  --continue-on-error
```

### 6.4 Point-set seed 与 rollout seed

推荐使用分离的 seed 参数：

| 参数 | 作用 |
|---|---|
| `--point-set-seed` | 只控制 random/LHS 点位生成。相同值和范围产生相同点集。 |
| `--rollout-seed-base` | 控制每条 rollout 的 seed；第 `i` 个点使用 `base + i - 1`。 |
| `--base-seed` | 旧版耦合参数；同时控制点集和首个 rollout seed。新实验优先使用两个分离参数。 |

分离两个随机来源可以分别研究：

- 空间采样点变化造成的结果差异；
- 同一空间点集上 rollout 随机性造成的差异。

### 6.5 续跑和错误处理

| 参数 | 作用 |
|---|---|
| `--skip-existing` | 如果某点位目录已有可读 `summary.json`，跳过该点。 |
| `--continue-on-error` | 某个子进程错误后继续后续点位。 |
| `--dry-run` | 生成点位和 manifest、打印命令，但不执行子 rollout。 |
| `--mujoco-gl` | 给子进程设置 MuJoCo GL backend，例如无头环境可能使用 `egl`。 |
| `--save-videos` | 所有子 rollout 保存视频，耗时和磁盘开销较大。 |

续跑必须使用完全相同的点位生成参数、seed、模型和输出目录。`--skip-existing` 只根据 `summary.json` 判断完成，不会验证旧结果的协议是否与新命令一致，因此实验者必须自行保证配置一致。

当前 grid runner 不能直接加载外部 `task_points.csv` 或旧 manifest；它依靠 sampling mode、bounds 和 seed 重新生成点位。精确复现实验时必须保存并记录这些参数。

### 6.6 批量输出结构

```text
GRID_OUT/
  task_points.csv
  grid_manifest.json
  grid_summary.csv
  random_position_summary.json
  x_..._z_..._repeat_.../
    rollout_log.csv
    summary.json
  plots/                       # 启用 plot-results 时
```

- `task_points.csv`：计划点位和对应 rollout seeds；
- `grid_manifest.json`：协议、点位、子命令、状态、返回码和时间；运行中持续更新；
- `grid_summary.csv`：完成点位的逐点结果；
- `random_position_summary.json`：完成率、task/safe-success、象限和半径分组统计。

## 7. 多模型/多模式 suite：`run_xz_rollout_suite.py`

### 7.1 作用和适用边界

该脚本把以下预注册配置依次交给 grid runner：

| key | checkpoint/模式 |
|---|---|
| `contact_cvae` | 历史 Contact-CVAE 100k，contact latent=`zero` |
| `contact_cvae_prior` | 同一 Contact-CVAE，contact latent=`prior` |
| `motion_cvae` | 历史 Motion-CVAE 100k |
| `dualzero` | 历史 DualZero 100k |
| `act_baseline` | 历史 ACT baseline 100k |

checkpoint 路径硬编码在 `run_xz_rollout_suite.py` 的 `MODEL_SPECS` 中，并指向 `outputs/peg_hole_100/...`。该脚本适合仓库当前已注册模型的对照实验，不提供任意 `--checkpoint` 参数。

对于刚训练的新 checkpoint：

- 单模型实验使用 `run_mujoco_hole_grid.py`；
- 只有明确要把新模型加入固定 suite 时，才修改 `MODEL_SPECS` 并同步实验协议和测试。

### 7.2 先 dry-run

```bash
PYTHONPATH=src python scripts/run_xz_rollout_suite.py \
  --models contact_cvae contact_cvae_prior motion_cvae act_baseline \
  --action-select-modes mid temporal \
  --num-points 50 \
  --offset-mm 4 \
  --point-set-seed 20260702 \
  --rollout-seed-base 31000 \
  --max-rollout-steps 900 \
  --max-delta-q 0.02 \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_all100.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-base outputs/peg_hole_100/xz_suite_pm4mm \
  --dry-run
```

### 7.3 正式运行

```bash
PYTHONPATH=src python scripts/run_xz_rollout_suite.py \
  --models contact_cvae contact_cvae_prior motion_cvae act_baseline \
  --action-select-modes mid temporal \
  --sampling-mode latin_hypercube \
  --num-points 50 \
  --offset-mm 4 \
  --point-set-seed 20260702 \
  --rollout-seed-base 31000 \
  --action-mode action \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --policy-rate-hz 30 \
  --max-rollout-steps 900 \
  --max-delta-q 0.02 \
  --force-stop-threshold 1000 \
  --success-distance-threshold 0.005 \
  --success-lateral-threshold 0.006 \
  --success-force-threshold 40 \
  --success-hold-steps 15 \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_all100.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-base outputs/peg_hole_100/xz_suite_pm4mm \
  --skip-existing \
  --continue-on-error \
  --keep-going \
  --target-maps \
  --no-plot-results \
  2>&1 | tee outputs/peg_hole_100/xz_suite_pm4mm.log
```

suite 关键参数：

| 参数 | 说明 |
|---|---|
| `--models` | 选择预注册模型/latent 配置；默认运行全部五项。 |
| `--action-select-modes` | `mid`、`temporal`；默认两者都运行。 |
| `--offset-mm` | x/z 对称范围，单位为毫米；suite 会换算为米传给 grid。 |
| `--keep-going` | 一个模型×action-select 实验失败后继续后续组合。 |
| `--target-maps` | 每个 grid 完成后自动运行 `plot_hole_target_map.py`。 |
| `--plot-results` | 是否让 grid runner 同时生成其内置 heatmap/scatter。默认关闭。 |

默认值需要显式审阅：50 点、±6 mm、900 steps、LHS、`mid+temporal`、`max_delta_q=0.02`、hard stop 1000 N、safe-success force threshold 40 N。

## 8. 多 seed suite：`run_xz_multiseed_rollout_suite.py`

### 8.1 作用

该脚本重复调用 x/z suite，并产生：

- 每个 seed 配置下各模型的 task/safe-success rate；
- pooled success rate；
- Wilson 95% 区间；
- 不同 seed 配置成功率均值和标准差；
- point set 在同一 seed 下是否一致；
- process error 计数。

多 seed 脚本默认只运行 `mid`，因为它面向 safe-success 稳健性统计；需要 temporal 时必须显式指定。

### 8.2 推荐：分离 point-set 与 rollout seeds

```bash
MULTISEED_OUT="outputs/peg_hole_100/xz_multiseed_pm4mm"

PYTHONPATH=src python scripts/run_xz_multiseed_rollout_suite.py \
  --point-set-seeds 20260702 20260703 20260704 \
  --rollout-seed-bases 31000 32000 \
  --models contact_cvae contact_cvae_prior motion_cvae act_baseline \
  --action-select-modes mid \
  --num-points 50 \
  --offset-mm 4 \
  --max-rollout-steps 900 \
  --max-delta-q 0.02 \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_all100.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-base "$MULTISEED_OUT" \
  --target-maps \
  --keep-going \
  2>&1 | tee "$MULTISEED_OUT.console.log"
```

这里会运行 point-set seeds 与 rollout-seed bases 的笛卡尔积：

```text
3 point sets × 2 rollout seed bases = 6 seed configurations
```

目录按随机来源隔离：

```text
MULTISEED_OUT/
  suite_plan.json
  pointset_20260702/rollout_31000/...
  pointset_20260702/rollout_32000/...
  pointset_20260703/rollout_31000/...
  ...
  summary/per_seed_summary.csv
  summary/aggregate_summary.csv
```

### 8.3 旧版耦合 seed

```bash
PYTHONPATH=src python scripts/run_xz_multiseed_rollout_suite.py \
  --seeds 20260702 20260703 20260704 20260705 20260706 \
  --models contact_cvae motion_cvae act_baseline \
  --action-select-modes mid \
  --num-points 50 \
  --offset-mm 4 \
  --output-base outputs/peg_hole_100/xz_multiseed_legacy
```

`--seeds` 让每个值同时控制点位和第一个 rollout seed，保留用于复现旧实验。新实验优先分离两个 seed 维度。

### 8.4 只重新聚合现有结果

```bash
PYTHONPATH=src python scripts/run_xz_multiseed_rollout_suite.py \
  --point-set-seeds 20260702 20260703 20260704 \
  --rollout-seed-bases 31000 32000 \
  --models contact_cvae contact_cvae_prior motion_cvae act_baseline \
  --action-select-modes mid \
  --num-points 50 \
  --offset-mm 4 \
  --output-base "$MULTISEED_OUT" \
  --aggregate-only
```

`--aggregate-only` 不启动 rollout，只读取现有 seed 目录并重写汇总 CSV。传入的模型、action-select、点数、范围和 seed 协议必须与原实验一致。

### 8.5 多 seed 续跑规则

- 启动正式任务时写入 `suite_plan.json`；同一输出目录下协议不匹配会报错；
- 完整的 seed 配置会整体跳过；
- 部分完成的配置由下层 `--skip-existing` 逐点续跑；
- `--keep-going` 控制某个 seed-level suite 出错后是否继续后续 seed；
- 每次续跑必须复用同一个 `output-base` 和同一组协议参数。

## 9. Rollout 过程监控

### 9.1 查找活动进程

```bash
pgrep -af "python.*scripts/run_.*rollout"
```

也可以分别查找单次和批量子进程：

```bash
pgrep -af "run_mujoco_policy_rollout.py"
pgrep -af "run_mujoco_hole_grid.py"
pgrep -af "run_xz_multiseed_rollout_suite.py"
```

### 9.2 多 seed suite 专用监控

如果输出目录含 `suite_plan.json`，只需提供输出根目录：

```bash
PYTHONPATH=src python scripts/monitor_xz_rollout_suite.py \
  --output-base "$MULTISEED_OUT" \
  --watch \
  --interval 10
```

它会显示：

- 活动模型和 action-select mode；
- 当前 point-set seed 与 rollout-seed base；
- 当前点位和已完成点数；
- 完整、部分完成和排队中的 seed 配置；
- 活动 rollout/grid 进程。

`--watch` 只刷新只读状态；按 `Ctrl+C` 不会停止 rollout。

对于没有 `suite_plan.json` 的旧任务，需要同时传入 `--point-set-seeds`、`--rollout-seed-bases`、`--models`、`--action-select-modes`、`--num-points` 和 `--offset-mm` 等协议参数。

### 9.3 查看日志和 manifest

```bash
tail -f "$GRID_OUT.console.log"
```

批量执行时，`grid_manifest.json` 会在每个点位状态变化后更新。可用以下命令查看最近状态：

```bash
python -m json.tool "$GRID_OUT/grid_manifest.json" | tail -80
```

## 10. 结果汇总与可视化

### 10.1 汇总一组单次 rollout

```bash
PYTHONPATH=src python scripts/summarize_rollouts.py \
  --root "$ROLLOUT_ROOT" \
  --pattern "contact_cvae_zero_mid_*" \
  --output "$ROLLOUT_ROOT/contact_cvae_zero_mid_summary.csv"
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--root` | 包含多个单次 rollout 子目录的根目录。 |
| `--pattern` | 匹配 rollout 子目录名的 glob。 |
| `--output` | 汇总 CSV。 |

脚本优先读取每个目录的 `summary.json`；只有缺失时才从 `rollout_log.csv` 推断。不要用过宽 pattern 混入不同实验协议。

### 10.2 Grid heatmap/scatter

```bash
PYTHONPATH=src python scripts/plot_hole_grid_results.py \
  --summary-csv "$GRID_OUT/grid_summary.csv" \
  --output-dir "$GRID_OUT/plots" \
  --formats png,pdf \
  --dpi 300 \
  --annotate
```

`--annotate` 在图中标注数值/点位信息；大量 LHS 点时可能较拥挤。`--show` 会打开交互窗口，不适合无头服务器。

### 10.3 靶心式空间图

```bash
PYTHONPATH=src python scripts/plot_hole_target_map.py \
  --grid-summary-csv "$GRID_OUT/grid_summary.csv" \
  --output-dir "$GRID_OUT/plots" \
  --output-stem contact_cvae_zero_mid_target \
  --title "Contact-CVAE zero + mid, +/-4 mm" \
  --ring-step-mm 1 \
  --max-radius-mm 6 \
  --marker-size 44 \
  --safe-force-threshold 40 \
  --show-point-index \
  --show-sampling-boundary \
  --formats png pdf \
  --dpi 300
```

`--safe-force-threshold` 只在绘图时重新计算 safe success，不修改输入 CSV。正式报告应明确它是否与实验原始 `success_force_threshold` 相同。

### 10.4 单次传感器曲线

```bash
PYTHONPATH=src python scripts/plot_rollout_sensor_analysis.py \
  --rollout-dir "$SINGLE_OUT" \
  --output-dir "$SINGLE_OUT/sensor_analysis" \
  --contact-force-threshold 5 \
  --high-force-threshold 20 \
  --very-high-force-threshold 40 \
  --success-distance-threshold 0.005 \
  --success-lateral-threshold 0.006 \
  --success-force-threshold 40 \
  --success-hold-steps 15 \
  --smooth-window 5 \
  --formats png,pdf \
  --dpi 300
```

也可以比较两条 rollout：

```bash
PYTHONPATH=src python scripts/plot_rollout_sensor_analysis.py \
  --compare-rollout-dir-a outputs/.../contact_cvae_zero \
  --compare-rollout-dir-b outputs/.../contact_cvae_prior \
  --label-a "Contact zero" \
  --label-b "Contact prior" \
  --output-dir outputs/.../sensor_comparison
```

## 11. 推荐的标准实验路线

### 11.1 新 checkpoint 首次 rollout

```text
1. run_policy_inference_smoke.py
2. inspect_hole_assembly.py
3. run_mujoco_policy_rollout.py，不传 --execute-actions，10 steps
4. run_mujoco_policy_rollout.py，传 --execute-actions，50 steps
5. 检查 rollout_log.csv、summary.json 和传感器图
6. nominal offset 单次完整 rollout
7. 9 点小 grid 或 10 点 LHS
8. 通过后扩展到 50/100 点正式 LHS
```

### 11.2 单模型空间鲁棒性

```text
单次 smoke
  -> run_mujoco_hole_grid.py --dry-run
  -> 10 点 pilot LHS
  -> 检查失败、力和停止原因
  -> 冻结协议
  -> 50/100 点正式 LHS
  -> grid_summary + target map + sensor case analysis
```

### 11.3 当前仓库模型对照

```text
确认 MODEL_SPECS 路径
  -> run_xz_rollout_suite.py --dry-run
  -> 小点数 pilot
  -> 固定 point-set seed，使模型共享完全相同点位
  -> 正式 suite
  -> 比较 task success、safe success、force 和空间分布
```

### 11.4 多 seed 统计

```text
先完成单 seed suite
  -> 决定 point-set seeds 与 rollout-seed bases
  -> run_xz_multiseed_rollout_suite.py
  -> monitor_xz_rollout_suite.py
  -> 检查 per_seed_summary.csv
  -> 检查 aggregate_summary.csv、Wilson CI 和 seed 方差
```

## 12. 实验设计注意事项

### 12.1 配对比较

比较两个模型时应保持以下条件相同：

- 同一 `task_points.csv` 对应的 point-set seed、sampling mode 和 bounds；
- 同一 rollout-seed base；
- 相同 action mode、action-select mode、policy rate 和 max steps；
- 相同 `max_delta_q`、EMA、成功阈值和 hard force stop；
- 相同 XML、hole body/site、offset frame 和 hole axis；
- 相同视频/渲染设置，避免额外性能差异。

### 12.2 不要只报告 task success

至少同时报告：

- completion/process error；
- task success rate；
- safe-success rate；
- max/mean force；
- final/min distance 和 lateral error；
- stop reason；
- 多 seed 时的 pooled rate、95% CI 和 seed-level variation。

### 12.3 阈值必须冻结

成功判据或 safe-force threshold 改变后，实验含义也改变。不要在看到结果后无记录地修改阈值。每次报告必须列出：

```text
distance threshold
lateral threshold
success force threshold
success hold steps
hard force stop threshold
```

### 12.4 当前实现限制

- Grid runner 不能直接读取保存的 task-points CSV/manifest，只能由参数和 seed 重生成；
- Grid runner 没有 subprocess timeout 和自动 retry；
- Grid runner 没有显式转发 `temporal-agg-decay`，使用单次脚本默认值 `0.3`；
- x/z suite 只支持 `MODEL_SPECS` 中的固定 checkpoint；
- 多 seed wrapper 只暴露部分 x/z suite 参数，自定义成功阈值等协议时应检查它是否能转发；
- `--skip-existing` 依据 `summary.json`，不会自动验证旧目录与当前协议一致；
- posterior oracle 使用未来标签，不属于 MuJoCo deployable rollout；rollout 仅使用 zero/prior 等在线分支。

## 13. 实验归档清单

每次正式 rollout 至少归档：

```text
[ ] checkpoint 路径、文件 hash 和训练 step
[ ] normalization stats 路径和 hash
[ ] Git commit
[ ] MuJoCo XML 路径和版本/hash
[ ] 完整命令和 console log
[ ] action/contact latent/action-select mode
[ ] chunk、force window、policy rate、max steps
[ ] max_delta_q、EMA 和 hard force stop
[ ] 全部 success thresholds
[ ] hole body/site/axis/offset frame
[ ] point-set seed、rollout-seed base 和 sampling bounds
[ ] task_points.csv 和 grid_manifest.json
[ ] 每次 summary.json 和 rollout_log.csv
[ ] grid/per-seed/aggregate summaries
[ ] 关键图表和典型成功/失败 case
[ ] Python、PyTorch、MuJoCo、CUDA 和 GPU/renderer 信息
```

## 14. Rollout 启动前最终检查

```text
[ ] checkpoint、stats 和 XML 都存在
[ ] checkpoint policy_variant 可被 rollout dispatch
[ ] action mode 与 stats metadata 一致
[ ] chunk/force-window/image preprocessing 与训练一致
[ ] HDF5 force 与 MuJoCo sensor wrench 约定已核对
[ ] deployable inference smoke 通过
[ ] hole assembly 和 offset 检查通过
[ ] 单次无动作检查通过
[ ] 单次 execute smoke 通过
[ ] 成功和安全阈值已冻结并记录
[ ] batch/suite 先执行过 --dry-run
[ ] 输出目录不会混入不同协议结果
[ ] seed、点位范围、模型组合和 action-select mode 已记录
[ ] 磁盘、预计耗时和视频策略满足长任务要求
```
