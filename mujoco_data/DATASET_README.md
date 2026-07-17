# MuJoCo 数据集目录

最近盘点：2026-07-16。

本文档是仓库内 `mujoco_data/` 的选型索引，回答“当前有哪些数据、各自适合什么实验、使用前要注意什么”。统计来自本地 episode 文件、`quality_report.csv` 和 `quality_summary.json` 的只读审计；它们是当前快照，不会随目录内容自动更新。

> 数据文件仍被 `.gitignore` 排除，只有本索引进入版本控制。不要把 `bad_data*` 混入训练集，也不要仅凭目录名推断孔位范围或安全保证。

## 快速选型

| 数据集 | 推荐定位 | Episode | 采集日期 | 本地占用 | 离线质量 | Episode 峰值力：中位 / P95 / 最大 | 现成 split |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| `peg_hole_100` | 既有基线复现、固定任务训练、与历史 100k 模型对齐 | 100 | 2026-07-01 | 13G | 100 good；24 个 >60 N 告警 | 36.84 / 90.66 / 101.00 N | 80/10/10 |
| `peg_hole_hmj_60N_limit` | 更大规模、峰值力受控的固定任务训练 | 177 | 2026-07-11 至 07-13 | 29G | 177 good；无告警 | 41.66 / 57.40 / 59.82 N | 无 |
| `hole_random_60mm_hmj` | 随机孔位采集、提高任务分布多样性 | 47 | 2026-07-14 | 7.6G | 47 good；5 个 >60 N 告警 | 44.19 / 73.07 / 81.92 N | 无 |

这里的“P95”是对“每个 episode 的峰值力”再取第 95 百分位；“本地占用”来自 `du -sh`，不是压缩包大小。`good` 是质量脚本的综合判定，不等于“全程低于 60 N”。

建议按目标选择：

- 要复现仓库已有训练、checkpoint 或论文实验，优先使用 `peg_hole_100`，因为现有 episode list、归一化产物路径和大量报告均围绕它建立。
- 要重新训练更大的固定任务模型，并希望训练集内 episode 峰值力保持在 60 N 以下，优先使用 `peg_hole_hmj_60N_limit`。
- 要增加随机孔位视觉/运动分布，使用 `hole_random_60mm_hmj`；但当前文件没有可用的逐 episode 孔位标签，不能据此做按真实孔位监督、分层切分或误差重建。
- 要研究固定到随机的泛化，固定集与随机集应作为不同 domain 管理，先去重，再单独建立 train/val/test 清单和归一化统计。

## 三个主数据集总览

三个主目录合计：

- 324 个可读取的成功 episode，648 个原始 episode 文件（每个目录一个 HDF5 和一个 `metadata.json`）；
- 121,899 帧状态、121,899 组双相机图像、2,029,031 帧力/力矩；
- 约 4,052.36 秒（67.54 分钟）episode 时间；
- 统一 HDF5 schema：`compact_mujoco_hdf5_v1`；
- 全部 HDF5 `status` 为 `auto_stop_task_success`；
- 全部 `action` 与 `actions/joint_pos_command` 逐值相同。

## 1. `peg_hole_100`

这是仓库目前兼容性和复现实验基础最完整的数据集。现有训练说明、模型注册表和大多数 rollout 报告都以它为数据来源。

| 指标 | 当前值 |
| --- | ---: |
| Episode | 100 |
| 状态/图像帧 | 30,977 / 30,977 |
| 力/力矩帧 | 515,524 |
| 总时长 | 1,029.19 s（17.15 min） |
| 每集状态帧：最小 / 中位 / 最大 | 213 / 306.5 / 448 |
| 每集时长：最小 / 中位 / 最大 | 7.07 / 10.18 / 14.90 s |
| 每集峰值力：最小 / 中位 / P95 / 最大 | 8.42 / 36.84 / 90.66 / 101.00 N |
| 质量分数：均值 | 98.8 |
| 质量结果 | 100 good；0 error；24 warning |

特点与注意事项：

- 按现有数据卡和采集配置记录，它属于未启用孔位随机化的历史固定任务集。
- 24 个 episode 的瞬时峰值力超过质量审计的 60 N 告警线，但仍因总分不低于 90 被分为 `good`。若实验要求严格安全阈值，应根据 `quality_report.csv` 另行过滤，而不是直接使用全部 100 集。
- 仓库提供稳定的 episode 级划分：
  - `configs/splits/peg_hole_100_train80.txt`
  - `configs/splits/peg_hole_100_val10.txt`
  - `configs/splits/peg_hole_100_test10.txt`
- `mujoco_data/DATASET_CARD.md` 是这一套 100-episode 数据的详细发布卡，不代表整个 `mujoco_data/`。

适合：历史结果复现、模型/脚本回归、固定任务基线，以及需要直接复用已有 80/10/10 split 的实验。

## 2. `peg_hole_hmj_60N_limit`

这是三个主目录中规模最大、离线峰值力最受控的一套。目录名中的 `60N_limit` 与当前审计结果一致：177 个 episode 的最大峰值为 59.82 N。

| 指标 | 当前值 |
| --- | ---: |
| Episode | 177 |
| 状态/图像帧 | 72,022 / 72,022 |
| 力/力矩帧 | 1,198,922 |
| 总时长 | 2,394.76 s（39.91 min） |
| 每集状态帧：最小 / 中位 / 最大 | 289 / 399 / 701 |
| 每集时长：最小 / 中位 / 最大 | 9.60 / 13.27 / 23.33 s |
| 每集峰值力：最小 / 中位 / P95 / 最大 | 16.83 / 41.66 / 57.40 / 59.82 N |
| 质量分数：均值 | 100.0 |
| 质量结果 | 177 good；0 error；0 warning |

特点与注意事项：

- 按当前目录和原始采集路径命名，它属于 HMJ 固定任务的 60 N 限制保留集。
- “60 N”在本文档中只表示当前离线审计观察到的 episode 峰值上界；它不是模型部署的安全证明，也不能替代 rollout 的 hard force stop。
- 当前没有随仓库维护的 train/val/test episode list。正式训练前应固定随机种子生成 split，并只用 train split 计算 normalization stats。
- 与另外两套数据相同，孔中心字段不可用，因此不能从 HDF5 独立证明每个 episode 的真实孔位完全一致。

适合：新的固定任务主训练集、受控峰值力实验，以及需要比 `peg_hole_100` 更多成功示范的训练。

## 3. `hole_random_60mm_hmj`

这套数据按当前目录名归类为随机孔位 HMJ 数据，主要价值是引入不同孔位下的视觉、接触和动作分布。

| 指标 | 当前值 |
| --- | ---: |
| Episode | 47 |
| 状态/图像帧 | 18,900 / 18,900 |
| 力/力矩帧 | 314,585 |
| 总时长 | 628.41 s（10.47 min） |
| 每集状态帧：最小 / 中位 / 最大 | 271 / 391 / 589 |
| 每集时长：最小 / 中位 / 最大 | 9.00 / 13.00 / 19.60 s |
| 每集峰值力：最小 / 中位 / P95 / 最大 | 10.19 / 44.19 / 73.07 / 81.92 N |
| 质量分数：均值 | 99.47 |
| 质量结果 | 47 good；0 error；5 warning |

特点与注意事项：

- 5 个 episode 的峰值力超过 60 N；若与 `peg_hole_hmj_60N_limit` 做安全约束对比，需要先统一过滤口径。
- 当前没有现成 split，样本量也明显小于两个固定任务集。建议优先把它作为独立 validation/test domain，或与固定集按明确采样权重混合，而不是直接拼接后随机逐帧切分。
- **孔位标签不可验证**：三个主数据集的 `episode_metadata/initial_hole_center_pos`、`final_hole_center_pos` 和 `initial_task_error_xyz` 都是 `NaN`。历史 recorder 使用了未解析成功的旧 site 名称，且没有把 `last_hole_randomization` 写入 HDF5。
- **命名存在冲突**：当前目录叫 `hole_random_60mm_hmj`，但每个 `metadata.json` 中保留的原始路径包含 `hole_random_4mm_hmj`。仓库内没有与这 47 个 episode 绑定的采集 manifest 可以消解该冲突。因此，在补齐原始采集配置或外部 manifest 前，不应在论文或评估中把“60 mm”写成由 HDF5 直接验证的真实随机半径。

适合：不依赖显式孔位标签的随机任务训练、固定到随机的 domain shift 评估。暂不适合：按孔位半径分桶、位置条件策略、逐点覆盖图或从离线文件重算 peg-to-hole error。

## 统一目录与 HDF5 格式

每个主数据集均采用：

```text
mujoco_data/<dataset>/
├── YYYYMMDD_HHMMSS_teleop_NNN/
│   ├── episode.hdf5
│   └── metadata.json
├── quality_report.csv
└── quality_summary.json
```

主要字段：

| 内容 | HDF5 路径 | 每帧形状 / dtype | 典型频率 |
| --- | --- | --- | ---: |
| 关节位置、速度、执行器广义力 | `observations/joint_pos`, `joint_vel`, `joint_torque` | `[7]`, `float64` | 约 30.3 Hz |
| 末端位姿 | `observations/ee_pose` | `[7]`, `float64`，`x y z qw qx qy qz` | 约 30.3 Hz |
| 双相机 RGB | `observations/images/{ee_cam,base_top_cam}` | `[480,640,3]`, `uint8` | 约 30.3 Hz |
| 补偿/原始/重力 wrench | `observations/ft_wrench{,_raw,_gravity}` | `[6]`, `float64`，`Fx Fy Fz Tx Ty Tz` | 500 Hz |
| 关节命令 | `action`, `actions/joint_pos_command` | `[7]`, `float64` | 约 30.3 Hz |
| 时间戳 | `timestamps/{state,image,force}_episode` | 标量序列，`float64` | 对应各数据流 |
| 事件 | `events/{names,t_episode,t_sim,t_wall}` | 事件序列 | 非周期 |

状态、图像和 command action 的第一维在每个 episode 内一致；force stream 以 500 Hz 独立采样。`action` 是 `actions/joint_pos_command` 的 ACT 兼容别名，不要把它与 loader 的 `action_mode=joint_pos` 混淆：后者使用未来观测关节位置作为标签。

训练时应使用 [`src/force_aware_act/data/contact_force_hdf5_dataset.py`](../src/force_aware_act/data/contact_force_hdf5_dataset.py) 中的 `ContactForceHDF5Dataset` 完成因果对齐。尤其是 force window，只允许使用当前 state timestamp 及以前的力数据。

## 质量报告应该怎样解释

三套主数据均使用 `scripts/evaluate_dataset_quality.py` 的同一套阈值生成当前报告：

- 目标成功状态：`auto_stop_task_success`；
- 时长范围：3–30 s；
- 峰值力告警线：60 N；
- 最大关节速度：1 rad/s；
- 最大相邻 command step：0.05 rad；
- `quality_score >= 90` 才标为 `good`。

质量脚本会对问题扣分，但单个峰值力告警不会自动把 episode 判为 bad。因此选安全训练子集时，应显式过滤 `max_force_n`，并在实验记录中写清 `< 60 N` 还是 `<= 60 N`。

`status=auto_stop_task_success` 表示采集器成功停止事件；三个主数据集的 `episode_metadata.attrs["task_success"]` 均为 `unknown`。判断成功应以 `status` 和 `events` 为主，不应使用为 `NaN` 的 hole/task-error 字段。

## 其他目录

| 目录/文件 | 数量与状态 | 用途 |
| --- | --- | --- |
| `bad_data/` | 71 个 HDF5；70 个可读，1 个损坏；可读文件中 57 `auto_stop_task_success`、7 `controller_shutdown`、6 `manual_keep` | HMJ 固定任务的隔离/拒绝区；即使 status 看似成功，也不要未经人工复核加入训练 |
| `bad_data_random_hole/` | 7 个可读 HDF5；6 `manual_keep`、1 `controller_shutdown` | 随机孔位采集的隔离/拒绝区 |
| `hf_episode_verify/` | `raw/` 下 1 个 episode，约 138M | Hugging Face 数据下载/打包验证样本，不应计作独立训练集 |
| `DATASET_CARD.md` | 1 份 | `peg_hole_100` 的详细英文发布卡，不是全目录索引 |

`bad_data/20260711_165048_teleop_010/episode.hdf5` 当前无法由 HDF5 读取，且缺少 `metadata.json`。本文档只报告现状，不对数据做修复或删除。

## 开始训练前的最小流程

1. 选定数据集和过滤规则，把最终 HDF5 路径冻结为 episode list；不要在训练时直接扫描一个仍会变化的目录。
2. 对最终列表重新运行质量检查和 dataset-loader 检查，确认相机、action mode、chunk 长度及时间对齐。
3. 按 episode 划分 train/val/test；混合固定与随机 domain 时保留 domain 标签，并检查跨目录重复 episode。
4. 只用 train list 计算 normalization stats；训练、离线评估和 rollout 必须使用相同的 action mode、相机顺序、图像预处理、force window 与 stats。
5. 保存 Git commit/dirty status、episode lists、过滤阈值、stats 路径和数据审计摘要，确保结果可复现。

现有完整流程见：

- [`docs/data/NEW_DATASET_TRAINING_MANUAL.md`](../docs/data/NEW_DATASET_TRAINING_MANUAL.md)
- [`docs/data/ACTION_SEMANTICS.md`](../docs/data/ACTION_SEMANTICS.md)
- [`docs/reference/COMMAND_RECIPES.md`](../docs/reference/COMMAND_RECIPES.md)
- [`configs/splits/README.md`](../configs/splits/README.md)

## 快速复核命令

从仓库根目录运行：

```bash
# 统计三个主数据集的 episode 数
for dataset in peg_hole_100 peg_hole_hmj_60N_limit hole_random_60mm_hmj; do
  printf '%s: ' "$dataset"
  find "mujoco_data/$dataset" -mindepth 2 -maxdepth 2 -name episode.hdf5 | wc -l
done

# 重新生成质量报告；建议写到 outputs，避免覆盖随数据保存的审计快照
PYTHONPATH=src python scripts/evaluate_dataset_quality.py \
  mujoco_data/peg_hole_hmj_60N_limit \
  --output-csv outputs/peg_hole_hmj_60N_limit/quality_report.csv \
  --output-json outputs/peg_hole_hmj_60N_limit/quality_summary.json
```

当 episode 增删、重分类、修复，或补回随机孔位 manifest 后，应同步更新本文档顶部日期、三张详细统计表、快速选型表和已知限制。
