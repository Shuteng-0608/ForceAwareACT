# 60 mm 视觉泛化 + 2 mm 接触精修：分阶段训练协议

## 当前状态与使用边界

这套协议面向尚未完成采集的两组新数据：60 mm 孔位随机化 50 条，以及约 2 mm 孔位抖动、接触恢复模式更丰富的 50 条。示例配置只是可审计的起点，不能在新数据到齐、质检、划分和冻结前直接宣称正式实验成立。

现有 `mujoco_data/hole_random_60mm_hmj` 与 `mujoco_data/peg_hole_hmj_60N_limit` 只能用于只读的加载兼容性、数据契约和验证流程冒烟。它们不得替代新采数据，不得被重新包装成正式协议的 train/val/test，也不得把冒烟结果写成新协议的性能结论。历史文件缺少 UUID 时，只有历史兼容清单才允许显式使用 `--derive-uuid-from-sha256`；新采集数据不得使用这个退路。

当前工作区实际扫描到的历史 HDF5 数量分别是 100 和 177，并不是各 50。本轮只读冒烟按规范化路径字典序临时取各自前 50 条：R60 的 50 条都成功建立索引，共 19,745 个可用 decision state；R2 的 50 条也全部建立索引，共 22,172 个。必需数值字段、严格时间戳/长度、因果图像对齐与抽样张量形状均通过。这个选择没有落成正式 list，只是可重复的兼容性 smoke；正式实验仍必须人工冻结明确的 list 并记录哈希。该结果不代表图像内容、phase 标签、专家质量或模型效果已经验收。

示例协议位于 `configs/experiments/staged_visual_force_protocol.example.json`。其中所有路径都相对该 JSON 文件；两个全零 SHA-256 是故意设置的阻断占位符。创建正式 manifest 和归一化统计后，必须替换它们，未替换时 `train_staged.py` 应当失败。

## 协议要解决的两件事

Stage 1 `spatial_r60` 只使用大范围孔位训练集，目标是建立孔位、末端与接近动作之间的视觉空间表征。训练批次为 16 个 R60 样本；视觉 backbone 使用基础学习率的 0.5 倍，其他模块正常更新。R2 验证集仍单独记录，但不参与 Stage 1 checkpoint 选择。

Stage 2 `contact_r2` 从 Stage 1 最佳 checkpoint 初始化，目标是学习接触建立、恢复与插入。每个 16 样本批次固定为 12 个 R2 样本和 4 个 R60 rehearsal 样本，即 75%/25%；视觉 backbone 降到基础学习率的 0.1 倍，避免接触精修抹掉视觉泛化。R2 的 12 个样本进一步按 `approach=3`、`contact_onset=3`、`recovery=4`、`insertion=2` 采样。该配比是首轮假设，不是已经由数据证明的最优值。

Stage 2 的主指标是 `r2_contact_val/deploy_loss`，但只有 `r60_spatial_val/deploy_loss` 相对 Stage 2 初始化基线退化不超过 5% 时，候选 checkpoint 才能通过 retention gate。两个验证域始终独立统计，不能把它们拼成一个平均数掩盖遗忘。

## 先冻结数据，再训练

推荐把每个 50 条域固定划为 40 train / 5 val / 5 test。若更看重 checkpoint 选择稳定性，也可以在训练前一次性预注册 35/10/5；不能看到结果后再换比例。划分应在 episode 级按条件分层：R60 至少兼顾孔位半径/方向，R2 至少兼顾接触方向、模式与恢复类型，再在层内使用固定 seed。不要按帧随机切分；同一 episode 的任何副本都只能属于一个 split。划分种子、分层表、生成命令和 list 文件都应版本化。

无论 40/5/5 还是 35/10/5，5 条 test 的 episode 级样本量都很小。最终报告应给出逐 episode 结果和不确定性，避免把大量相关时间片误当成独立样本；任务层结论还应依赖预注册的成对 MuJoCo 点集，而不是只报告一个小 test 的均值。

正式新 episode 应在 HDF5 根属性 `episode_uuid` 或同目录 `metadata.json` 中携带有效且稳定的 UUID。manifest 在三条彼此独立的轴上检查泄漏：UUID、规范化绝对路径和文件内容 SHA-256。因此，改名或复制相同 HDF5 也不能绕过检查。manifest 还把每个文件绑定到唯一的 `domain` 与 `split`：

- `r60_visual`: 60 mm 大范围孔位数据；
- `r2_contact`: 约 2 mm 小范围抖动、密集接触恢复数据；
- `train`: 只允许进入训练和归一化；
- `val`: 只允许用于选择与阶段门禁；
- `test`: 在模型、阈值和候选选择全部冻结前保持封存。

R2 数据不能只增加碰撞数量。每条专家示范应尽量形成“接触建立 → 明确修正 → 力/接触改善 → 成功插入”的闭环，并对接触方向、初始位姿和恢复动作做对称覆盖。无接触正常插入也必须保留，防止模型学成持续搜索或回撤。

示例把监督目标固定为 HDF5 的 `action`。这只是需要在采集前确认的语义假设：如果专家命令实际记录在 `actions/joint_pos_command`，就必须在任何统计或训练开始前统一修改 protocol 与归一化命令的 action mode。两域还必须使用同一关节顺序、力/力矩单位与坐标系、相机名称和图像预处理，并记录传感器调零规则；不能依赖 normalization 掩盖数据契约差异。

### R2 phase catalog

示例配置启用了 phase quota，因此 `r2_train_phase_catalog.json` 是正式协议资产。它采用 `src/force_aware_act/training/catalog.py` 的 schema version 2：每个训练 episode 同时固定 `path`、`domain`、原生 `episode_uuid`、`file_sha256`，并用不重叠的 `[start, stop)` state-index 段标为 `approach`、`contact_onset`、`recovery` 或 `insertion`。catalog 必须覆盖数据集可产生的每个 state index，并在训练前由人工结合力曲线、动作与视频抽查边界；不要仅凭一个未经验证的力阈值自动生成“真值”。

先制作人工标注 CSV。四列都是必需的，`start` 包含、`stop` 不包含；本协议的 `phase` 只能使用示例配置中的四个名称：

```csv
episode_path,start,stop,phase
path/to/r2_episode_0001.hdf5,0,120,approach
path/to/r2_episode_0001.hdf5,120,155,contact_onset
path/to/r2_episode_0001.hdf5,155,260,recovery
path/to/r2_episode_0001.hdf5,260,340,insertion
```

`build_phase_catalog.py` 不读取力阈值推断 phase，只把人工/采集元数据标注与当前 dataset semantics 对齐。必须先按下文生成 dataset manifest，并把脚本打印的 canonical manifest 内容哈希原样传入。构建器会重新校验 manifest 哈希和所有 episode 文件哈希，要求 episode list 精确等于 manifest 中 `r2_contact/train` 的全集，且只接受原生 UUID。它还拒绝重叠、空洞、重复路径、未精确覆盖所有可用 dataset state index 的标注以及覆盖已有输出：

```bash
PYTHONPATH=src python scripts/build_phase_catalog.py \
  --annotation configs/splits/staged_visual_force/r2_train_phase_annotations.csv \
  --episode-list configs/splits/staged_visual_force/r2_train.txt \
  --dataset-manifest outputs/staged_visual_force/contracts/dataset_manifest.json \
  --dataset-manifest-sha256 <manifest_content_sha256> \
  --source-domain r2_contact \
  --output configs/splits/staged_visual_force/r2_train_phase_catalog.json \
  --action-mode action \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --camera-names ee_cam base_top_cam \
  --image-size 224 224 \
  --image-alignment latest_past \
  --max-image-lag-seconds 0.1 \
  --strict-lengths
```

输出的 `labeler` 固定记录 dataset manifest SHA-256、annotation/list SHA-256、dataset semantics、builder 版本与 phase 名称。脚本打印的 `phase_catalog_sha256` 是 canonical catalog 内容哈希，必须写入对应 source 的 `sample_catalog_sha256`；训练准备会再次核验 catalog 哈希、manifest 哈希以及每条 episode 的 path/domain/UUID/file SHA。如果首轮数据尚无可信 phase 标签，就从协议中同时移除 `phase_quotas`、`min_episodes_per_phase`、`sample_catalog` 和 `sample_catalog_sha256`，让 R2 域先按 episode 均衡抽样，而不是伪造标签。

## 从原始数据到正式运行

以下命令从仓库根目录执行。先创建并人工审查六个占位 list：

```text
configs/splits/staged_visual_force/r60_train.txt
configs/splits/staged_visual_force/r60_val.txt
configs/splits/staged_visual_force/r60_test.txt
configs/splits/staged_visual_force/r2_train.txt
configs/splits/staged_visual_force/r2_val.txt
configs/splits/staged_visual_force/r2_test.txt
```

每个文件一行一个 episode 路径。正式建议计数分别为 40/5/5 和 40/5/5。创建 manifest 时同时提交六个分组，不要使用历史兼容 UUID 开关：

```bash
PYTHONPATH=src python scripts/build_dataset_manifest.py \
  --group r60_visual:train=configs/splits/staged_visual_force/r60_train.txt \
  --group r60_visual:val=configs/splits/staged_visual_force/r60_val.txt \
  --group r60_visual:test=configs/splits/staged_visual_force/r60_test.txt \
  --group r2_contact:train=configs/splits/staged_visual_force/r2_train.txt \
  --group r2_contact:val=configs/splits/staged_visual_force/r2_val.txt \
  --group r2_contact:test=configs/splits/staged_visual_force/r2_test.txt \
  --output outputs/staged_visual_force/contracts/dataset_manifest.json
```

脚本拒绝覆盖已有输出，并打印 `manifest_content_sha256`。将该值写入协议的 `dataset_manifest.sha256`；这里需要的是 canonical manifest 内容哈希，不是 JSON 文件的普通 `sha256sum`。

协议中的 `test_episode_lists` 不能只填写裸路径；每个命名测试集必须同时声明 manifest 域，避免 R60 与 R2 的 test list 被对调后仍通过数据契约。例如：

```json
"test_episode_lists": {
  "r60_spatial_test": {
    "domain": "r60_visual",
    "episode_list": "../splits/staged_visual_force/r60_test.txt",
    "expected_episode_count": 5
  },
  "r2_contact_test": {
    "domain": "r2_contact",
    "episode_list": "../splits/staged_visual_force/r2_test.txt",
    "expected_episode_count": 5
  }
}
```

训练准备阶段会逐文件校验 `split=test` 和这里声明的 `domain`，并把名称、域、list 路径及解析后路径哈希写入 checkpoint/run manifest 的 data provenance。

归一化只从两个完整 train list 的并集计算，严禁包含 val/test，也不要为两个阶段各算一套统计。balanced raw estimator 先在 episode 内按时间点统计，再在域内等权 episode，最后等权两个域，避免长 episode 或重叠 ACT chunk 支配统计量：

```bash
PYTHONPATH=src python scripts/compute_normalization_stats.py \
  --domain r60_visual=configs/splits/staged_visual_force/r60_train.txt \
  --domain r2_contact=configs/splits/staged_visual_force/r2_train.txt \
  --estimator balanced_raw \
  --action-mode action \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --image-size 224 224 \
  --camera-names ee_cam base_top_cam \
  --strict-lengths \
  --output outputs/staged_visual_force/contracts/normalization_balanced_train_union.pt
```

命令打印的 `normalization_content_sha256` 是语义哈希；将它写入协议的 `normalization.sha256`。不要填统计文件的普通文件哈希。若修改任何 train list、原始 HDF5、action mode、窗口或相机设置，应创建新的 manifest、stats 和 run name，不能覆盖旧证据链。

balanced stats 默认拒绝覆盖已有输出；只有明确执行一次有记录的重建时才可使用 `--overwrite-existing`。训练和评估会分别重算 `normalization_config` 与 `population_identities` 的子哈希，再验证总语义哈希，手工改元数据不会被当成有效统计。

先对 Stage 1 做完整 dry-run。正式 dry-run 不应使用 `--skip-dataset-file-verification`：

```bash
PYTHONPATH=src python scripts/train_staged.py \
  --protocol configs/experiments/staged_visual_force_protocol.example.json \
  --stage spatial_r60 \
  --device cuda \
  --num-workers 0 \
  --dry-run
```

确认报告中的 manifest/stats hash、两个验证域、optimizer groups、R60 batch quota 和样本量都正确后启动 Stage 1：

```bash
PYTHONPATH=src python scripts/train_staged.py \
  --protocol configs/experiments/staged_visual_force_protocol.example.json \
  --stage spatial_r60 \
  --device cuda \
  --num-workers 0 \
  --output-dir outputs/staged/staged_visual_force_50plus50_v1/spatial_r60
```

Stage 1 通过下述升级门槛后，先 dry-run Stage 2，再正式初始化。`--init-from` 只载入模型权重，并建立新的 Stage 2 optimizer、sampler、monitor 与 retention 基线。输入 checkpoint 还必须由上一阶段 monitor 选中；普通 final 或选择点之后的 periodic checkpoint 会被拒绝：

```bash
PYTHONPATH=src python scripts/train_staged.py \
  --protocol configs/experiments/staged_visual_force_protocol.example.json \
  --stage contact_r2 \
  --init-from outputs/staged/staged_visual_force_50plus50_v1/spatial_r60/checkpoint_best.pt \
  --device cuda \
  --num-workers 0 \
  --dry-run

PYTHONPATH=src python scripts/train_staged.py \
  --protocol configs/experiments/staged_visual_force_protocol.example.json \
  --stage contact_r2 \
  --init-from outputs/staged/staged_visual_force_50plus50_v1/spatial_r60/checkpoint_best.pt \
  --device cuda \
  --num-workers 0 \
  --output-dir outputs/staged/staged_visual_force_50plus50_v1/contact_r2
```

`--resume-from` 的含义完全不同：它只用于同一阶段的中断续训，严格恢复模型、optimizer、sampler、monitor 和 RNG。必须使用原 output directory、未改动的协议/manifest/stats/list，并保持 `--num-workers 0`：

```bash
PYTHONPATH=src python scripts/train_staged.py \
  --protocol configs/experiments/staged_visual_force_protocol.example.json \
  --stage contact_r2 \
  --resume-from outputs/staged/staged_visual_force_50plus50_v1/contact_r2/checkpoint_latest.pt \
  --device cuda \
  --num-workers 0 \
  --output-dir outputs/staged/staged_visual_force_50plus50_v1/contact_r2
```

不得用 `--resume-from` 做跨阶段迁移，也不得用 `--init-from` 冒充断点续训。

协议强制 `checkpoint_every_steps == validation_every_steps` 且 `max_steps` 能被该 cadence 整除，因此每个正式验证点（包括最终点）都有一个不可变 periodic candidate，不会出现验证到的最佳点无法进入候选集合；验证聚合固定为 episode-uniform，不能让较长 episode 以更多帧数支配选择。每次被 monitor 选中的权重还会同时保存为不可变的 `checkpoint_best_step_XXXXXXXX.pt` 和稳定别名 `checkpoint_best.pt`。通常应从 `checkpoint_latest.pt` 续训。若确实从更旧的 periodic checkpoint 回退，默认会因日志或未来 artifact 不一致而失败；只有显式加入 `--trim-resume-logs-to-checkpoint` 才会原子裁剪 CSV，并把未来的 periodic/latest/final/best 移入 `resume_quarantine/`，写入文件 SHA-256 清单，再恢复与 checkpoint monitor 状态一致的 best。隔离是可恢复移动，不是静默删除。

## 验证、候选选择和最终 test

Stage 2 完成后，不要只看训练过程中的一个 `checkpoint_best.pt`。将 `stage_completion.json` 中 `candidate_checkpoints` 的每一行按原顺序写入候选 CSV；不得删掉表现不好的中间点、增加别的 checkpoint、改变顺序或跨 run 混合。评估器会根据 run manifest、completion 和训练 cadence 复验候选集合必须完整且唯一。路径相对 CSV 文件，SHA-256 是 checkpoint 文件的普通文件哈希，`epoch`、`step` 两列可省略。下面两行只是短运行的格式示意；正式示例配置应包含每个 500-step 周期点：

```csv
candidate_id,checkpoint_path,checkpoint_sha256
stage2_00500,contact_r2/checkpoint_step_00000500.pt,<sha256sum 输出>
stage2_01000,contact_r2/checkpoint_step_00001000.pt,<sha256sum 输出>
```

然后只在两个 val list 上做离线 retention gate 与 shortlist：

```bash
PYTHONPATH=src python scripts/evaluate_staged_checkpoints.py \
  --candidates-csv outputs/staged/staged_visual_force_50plus50_v1/stage2_candidates.csv \
  --stage1-reference outputs/staged/staged_visual_force_50plus50_v1/spatial_r60/checkpoint_best.pt \
  --stage1-reference-sha256 "REPLACE_WITH_STAGE1_CHECKPOINT_FILE_SHA256" \
  --protocol configs/experiments/staged_visual_force_protocol.example.json \
  --normalization-stats outputs/staged_visual_force/contracts/normalization_balanced_train_union.pt \
  --val-domain r2_contact_val=configs/splits/staged_visual_force/r2_val.txt \
  --val-domain r60_spatial_val=configs/splits/staged_visual_force/r60_val.txt \
  --objective-domain r2_contact_val \
  --retention-domain r60_spatial_val \
  --metric deploy_loss \
  --max-relative-degradation 0.05 \
  --max-absolute-degradation 0.0 \
  --min-relative-improvement 0.005 \
  --shortlist-size 3 \
  --batch-size 16 \
  --num-workers 0 \
  --device cuda \
  --output-dir outputs/staged/staged_visual_force_50plus50_v1/validation_gate_v1
```

该命令拒绝覆盖已有报告，并输出长表指标、逐候选决策和 shortlist，最后写入 `evaluation_completion.json`。只有该证明为 `status=complete`，且其中四个报告文件的路径和 SHA-256 都能复验，shortlist 才可进入 frozen test。评估强制 `protocol.deterministic=true` 和 PyTorch deterministic algorithms，并在报告中固定 seed、batch size、worker 数、解析后的 device 与 Python/PyTorch/NumPy/HDF5/CUDA/cuDNN 版本；候选循环结束后还会重新核验 CSV、checkpoint、protocol、stats、list、manifest、run evidence 和每个 validation HDF5 的哈希，任何中途变更都使本次运行失败。Stage 1 reference 必须是已完成上一阶段的最终 selected-best 别名；Stage 2 候选必须来自同一 run，`global_step` 严格递增并精确覆盖 completion 声明的完整周期集合，而且都要把该 reference 的文件 SHA-256 记录为父 lineage。objective/retention 域、metric、相对退化和最小改善量必须与 Stage 2 `monitor` 完全一致；CLI 写错会直接失败，不能另造一套门禁。候选集合、顺序、metric 和阈值必须在打开 test 之前冻结。若无 Stage 2 候选通过 retention gate，应回退 Stage 1 reference，而不是放宽门槛直到某个候选通过。

选定唯一 checkpoint 后，先把 `shortlist.json` 的普通文件哈希写入冻结实验记录，再用一个命令打开协议内全部 test 域。正式入口不接受 checkpoint 或 test-list 覆盖参数；它只读取经过哈希固定的 shortlist 中的唯一选择，以及协议中预注册的 5+5 test：

```bash
sha256sum outputs/staged/staged_visual_force_50plus50_v1/validation_gate_v1/shortlist.json

PYTHONPATH=src python scripts/evaluate_staged_frozen_test.py \
  --protocol configs/experiments/staged_visual_force_protocol.example.json \
  --selection-report outputs/staged/staged_visual_force_50plus50_v1/validation_gate_v1/shortlist.json \
  --selection-report-sha256 "REPLACE_WITH_FROZEN_SHORTLIST_JSON_SHA256" \
  --normalization-stats outputs/staged_visual_force/contracts/normalization_balanced_train_union.pt \
  --batch-size 16 \
  --device cuda \
  --bootstrap-seed 20260722 \
  --bootstrap-replicates 10000 \
  --output-dir outputs/staged/staged_visual_force_50plus50_v1/final_test_v1
```

该入口强制 deterministic prior 与 episode-uniform 聚合，并重新校验 v2 checkpoint、protocol/normalization/manifest 哈希、原生 episode UUID、每个文件的 SHA-256、`split=test`、域分配、list 哈希和精确条数。输出包含逐 episode 指标、两个域各自的 episode bootstrap 95% CI、所有输入/输出哈希；`completion.json` 最后原子写入，只有 `status=complete` 且其中的 artifact 哈希复验通过，才算一次完整测试。输出目录必须事先不存在，命令拒绝 symlink 和覆盖。

`evaluate_inference_modes.py` 保留为 zero/prior/posterior 模式对照和 oracle 诊断工具，不是本协议的正式 frozen-test 入口。posterior 使用未来标签，不能作为部署结果，也不能用它重新选择 checkpoint。

不要在这 100 条正式新数据上反复看 test、改超参数、重新选择 checkpoint 再看 test。尤其是保留的 5+5 test，一旦用于决策就失去无偏测试资格；需要继续迭代时，应从 train/val 设计新实验，或另采一批全新的 final test。

## 阶段升级与验收门槛

以下门槛应在第一轮正式训练前写入实验记录，不能看到结果后再修改：

1. 数据门槛：两域 episode 数与预注册 split 计数（示例为 40/5/5）、UUID/path/content 三轴无泄漏；全部新 HDF5 通过 schema、有限值、时间戳、图像和长度检查；R2 四个 phase 均有多个 episode 覆盖，不能靠单条轨迹填满一个 bucket。
2. Stage 1 升级门槛：无 NaN/Inf 或梯度异常；实际采样计数与 R60=16 完全一致；至少完成 `min_validations=5`；`r60_spatial_val` 的离线指标和预注册的空间 rollout 指标都优于预先定义的基线。仅凭 train loss 下降不得升级。
3. Stage 2 接受门槛：实际批次严格为 R2=12、R60=4，R2 phase 为 3/3/4/2；冻结视觉 backbone 的 BatchNorm running statistics（affine 参数仍按 optimizer group 配置训练），避免 40 条局部数据改写视觉统计；R2 主指标相对 Stage 1 baseline 达到预注册改善量；R60 retention 相对退化不超过 5%；接触恢复 rollout 同时报告成功率、恢复率、峰值力和 force-stop，不能用更大碰撞换取成功率。
4. 最终门槛：唯一 checkpoint 与所有阈值冻结后，只运行一次独立 test；分别报告 R60 与 R2，不合并成一个总分；最终结论同时包含离线 deployment-path 指标和 rollout 任务/安全指标。

`deploy_loss` 是归一化离线误差，不等价于插入成功率。具体 rollout 成功率、恢复率与力阈值应先用 train/val 和任务安全约束预注册；本示例不凭空给出一个看似精确的合格百分比。

## 功能测试

修改训练协议、数据契约或 checkpoint 逻辑后，至少运行以下聚焦测试：

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_data_manifest.py \
  tests/test_build_dataset_manifest.py \
  tests/test_build_phase_catalog.py \
  tests/test_balanced_normalization.py \
  tests/test_training_protocol.py \
  tests/test_training_catalog.py \
  tests/test_training_sampling.py \
  tests/test_training_optimizer.py \
  tests/test_training_checkpointing.py \
  tests/test_training_multisplit_validation.py \
  tests/test_training_engine.py \
  tests/test_train_staged_integration.py \
  tests/test_evaluate_staged_checkpoints.py \
  tests/test_evaluate_staged_frozen_test.py \
  tests/test_contact_recovery_metrics.py \
  tests/test_mujoco_rollout_action_modes.py \
  tests/test_hole_offset_and_grid.py
```

合并或正式训练前再运行完整测试：

```bash
PYTHONPATH=src python -m pytest -q
```

单元测试通过只能证明代码路径与不变量成立；真实数据 dry-run、Stage 1→Stage 2 lineage、精确 resume、两域 validation gate 和冻结 test 仍需分别保留产物与哈希。
