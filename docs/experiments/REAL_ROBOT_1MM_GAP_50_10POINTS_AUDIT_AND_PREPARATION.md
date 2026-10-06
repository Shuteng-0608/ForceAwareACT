# 1 mm gap：50条、10位置的数据审核报告

日期：2026-10-03。用户确认本批50条全部成功插入；任务标签来源为用户确认。

**审核结论：50/50条通过数据完整性与基础训练可用性审核；12条包含外力固定槽对齐提示，保留明确的缺失掩码。**

原始目录：`/home/stw/Downloads/1mm-gap-10points`。50条总计8,236,812,108 bytes（8.24 GB），10个位置各5条，总动作时长1732.58秒（约28.88分钟）。

| 项目 | 审核结果 |
| --- | --- |
| 当前下载文件SHA-256及大小 | 50/50与下载清单一致 |
| 完整MCAP解析 | 1,591,823条全部消息；所声明chunk CRC及非零DataEnd data CRC通过 |
| 外力来源 | estimated_joint_disturbance.effort，七关节，独立时间轴 |
| 图像 | 103,264张全部解码成功，640×480；未触发纯黑/纯白及连续重复压缩帧>1秒指标 |
| 接收时间 | 5个选定流均无重复或倒退；严格使用log_ns |
| 数值 | 完整右臂字段，无NaN/Inf |
| 有效anchor | 173,134/173,248（99.9342%） |
| 紧随一个区间的外力槽覆盖 | 94.5668% |
| 紧随一个区间的外力端点覆盖 | 96.4415% |

审核硬门槛为：文件身份与大小一致、完整MCAP解析与所声明CRC正确、关键流及关节字段齐全、数值有限、log时间严格递增、全部图像可解码且尺寸一致、动作相位集中度≥0.5、历史区间容量≤8、有效anchor≥95%。间隔变长、缺失监督或固定槽覆盖率低作为显式记录，不自动插值或补值。

外力使用 `/tianji_marvin_teleop_bridge/estimated_joint_disturbance.effort`；`joint_states.effort`保存为内部关节力矩，当前不作输入、标签或归一化。重力/摩擦补偿细节仍未核实。

12条外力固定槽覆盖率低于90%，最低79.6839%。此指标在100 Hz动作网格后每个10 ms区间固定3个300 Hz时间槽、容差±1.666666 ms下计算，包含记录边界。原始外力消息仍约300 Hz；单段覆盖率会受到到达相位及抖动影响，不等于下载缺失率或整段100步chunk覆盖率。后续数据转换保持原生样本并使用mask；不以内部力矩替代缺失外力。

| 记录 | 位置 | 外力槽覆盖 |
| --- | --- |
| Tianji_Marvin_M6CCS_261003_152933_0_x14y08 | x14y08 | 87.57% |
| Tianji_Marvin_M6CCS_261003_182523_0_x15y08 | x15y08 | 79.68% |
| Tianji_Marvin_M6CCS_261003_182819_0_x15y08 | x15y08 | 87.48% |
| Tianji_Marvin_M6CCS_261003_183221_0_x15y08 | x15y08 | 89.89% |
| Tianji_Marvin_M6CCS_261003_184957_0_x17y08 | x17y08 | 83.38% |
| Tianji_Marvin_M6CCS_261003_185633_0_x17y08 | x17y08 | 82.65% |
| Tianji_Marvin_M6CCS_261003_191421_0_x17y07 | x17y07 | 90.00% |
| Tianji_Marvin_M6CCS_261003_193005_0_x16y07 | x16y07 | 84.28% |
| Tianji_Marvin_M6CCS_261003_194452_0_x15y07 | x15y07 | 81.22% |
| Tianji_Marvin_M6CCS_261003_194659_0_x15y07 | x15y07 | 87.48% |
| Tianji_Marvin_M6CCS_261003_195023_0_x15y07 | x15y07 | 82.52% |
| Tianji_Marvin_M6CCS_261003_200318_0_x14y07 | x14y07 | 83.88% |

每条记录两个相机各保留首/中/末3张抽样证据，合计300张缩略图，位于 `audit/episodes/`。全部103,264帧已进行解码、分辨率、亮度/低纹理、压缩字节重复指标检查；图像抽样不用于推断插入成功标签。

审核通过后，已生成独立raw_v2 HDF5，逐条二次读取源消息比较原始数值字节、所有时间戳、JPEG字节及RGB像素。默认每位置3/1/1，固定seed=20261003，共train30/validation10/test10；同一记录不会跨集合。此划分检验同位置新记录表现，不能代表未见位置的泛化。

已完成的训练侧准备包括：逐条保存稳定动作相位与原始行索引、训练集原生流归一化、固定验证提议索引、每条起始/中间/末尾窗口的实际读取验证；不初始化模型、优化器或启动训练。

机器可读审核依据：`audit/audit_manifest.json`，逐条详情：`audit/episodes/<episode>/audit.json`。审核主体先于转换生成；转换与准备完成结果见下文。

补充时间诊断：检查4条最低覆盖及2条最高覆盖记录。最低记录的全网格log固定槽覆盖约79.6903%，header诊断约99.0561%；log网格中1,915个空槽与1,901个双消息槽接近抵消，表明原始样本仍在，接收抖动/相位落在固定槽边界附近使一对一取样舍弃部分标签。其余低覆盖记录header诊断约94.61%～97.72%。这是同步尚未确认条件下的对照，不能作为自动改用header的依据；本批生产处理继续log_ns。完整数据保留两套时钟，后续确认同步后可另行版本化标签对齐。

已人工查看10位置总览和x15y07/x15y08详细拼图；两路视角能看到接近与插入过程，D405完成阶段目标明显靠近镜头，属于运动产生的视野变化。成功标签仍以用户确认为来源。

| 流 | 实际汇总频率 / Hz | 原生消息数 | 最长接收间隔 / ms |
| --- | ---: | ---: | ---: |
| state | 299.713 | 519,376 | 38.167 |
| disturbance | 299.609 | 519,194 | 38.202 |
| action | 100.011 | 173,327 | 41.235 |
| d455 | 29.967 | 51,949 | 67.635 |
| d405 | 29.610 | 51,315 | 101.368 |

采样间隔超过1.5个标称周期仅作为时间质量指标，不代表已证实丢包。位置、动作和外力均有实际变化；外力各通道在全部记录中均非全段常值，无额外滤波、去偏或裁剪。

![位置与记录质量](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_50_10points_20261003/quality_by_position.png)

![时间轴对照](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_50_10points_20261003/force_clock_diagnostic.png)


## 已完成的数据转换与准备

完成时间：2026-10-03 21:21:47（Asia/Shanghai）。**50/50条完成转换及二次源文件比较，数据集状态为ready；本批未启动训练。**

- HDF5目录：[1mm-gap-10points_hdf5_raw_v2](/home/stw/Downloads/1mm-gap-10points_hdf5_raw_v2)，格式`real_robot_right_arm_raw_v2`。
- 配置入口：[dataset.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/configs/datasets/1mm_gap_50_10points_v2/dataset.json)。
- 固定划分：[split.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/configs/datasets/1mm_gap_50_10points_v2/split.json)，seed=20261003，每位置3/1/1，同一记录不跨集合，全部50个源文件SHA-256唯一。
- 归一化：[normalization.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/configs/datasets/1mm_gap_50_10points_v2/normalization.json)，仅训练30条，每条原生qpos、action、external torque各消息使用一次；std下限0.01，内部力矩不参与。
- 固定验证提议：[validation_indices_stride10.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/configs/datasets/1mm_gap_50_10points_v2/validation_indices_stride10.json)，每条验证记录从有效anchor序列每隔10个取一个，并纳入最后一个，共3,487窗口。尚未执行模型验证。
- 数据标签：[dataset_labels.csv](/home/stw/Downloads/1mm-gap-10points_hdf5_raw_v2/dataset_labels.csv)，50条`task_success=True`及用户确认来源。
- 转换与逐文件比较：[conversion_manifest.json](/home/stw/Downloads/1mm-gap-10points_hdf5_raw_v2/conversion_manifest.json)。每条目录还包含`quality_report.json`、`sampling_index.npz`及`conversion.log`。

### 转换的保真检查与处理记录

全部50条均重新读取源MCAP，逐元素核对：所有选定流时间戳完全相等、float64数值字节完全相等、原始JPEG字节完全相等、解码RGB像素完全相等。转换输出的动作相位、候选anchor、有效anchor索引与动作源行索引也和转换前审核完全相等。

原始MCAP保持原样；派生HDF5保存独立state/disturbance/action/相机时间轴、整数纳秒header/log/publish时间戳、原始JPEG和原分辨率RGB。没有插值、重采样、额外滤波、去偏、末尾延长或以内部力矩代替外力。HDF5文件总计68,589,875,423 bytes（68.59 GB，63.88 GiB），约为MCAP体积的8.33倍，主要空间用于保留解码RGB与JPEG。

生产采样规则保存在[data_config.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/configs/datasets/1mm_gap_50_10points_v2/data_config.json)：

| 项目 | 本批使用规则 |
| --- | --- |
| 时间基准 | `log_ns`，作为录包可用时间代理；header/publish保留 |
| 动作网格 | 100 Hz；开头2秒指令的圆均值估计稳定相位；每条相位及来源索引保留 |
| 观测 | anchor之前最近的实际state/相机；state最大20 ms，相机最大100 ms |
| 动作标签 | 最近固定时间槽，容差±5 ms；缺失或越过记录边界保留padding mask |
| 外力历史 | 过去200 ms的实际样本，按20个10 ms区间打包，每区间最多8个；保留实际相对时间 |
| 外力未来标签 | 每10 ms固定3个300 Hz槽，容差±1.666666 ms；不存在的标签保留mask |
| 训练窗口 | chunk_len=100（1秒）；future只用于监督，观测与历史不读取anchor之后的信息 |

共剔除114个不满足有效观测门槛的候选anchor，按顺序计数的原因如下：`{'missing_current_action': 47, 'stale_or_missing_state': 2, 'stale_or_missing_camera': 65, 'empty_torque_history': 0}`。剩余173,134个anchor保留真实时间间隔。记录选定流的log时间范围之间没有重叠，最短相邻录制间隔44.589秒。

### 完整100步窗口标签覆盖率

下面覆盖率按全部有效anchor加权计算，分母为每个窗口100个动作标签、300个外力槽或100个外力端点，**包含记录尾部不足1秒的padding**。它不同于前文只考察紧随anchor的单个10 ms区间，也不表示丢包率。

| 集合 | 记录数 | 可用anchor | 动作标签覆盖 | 外力槽覆盖 | 外力端点覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 30 | 105,673 | 98.5688% | 93.0554% | 94.7991% |
| validation | 10 | 34,733 | 98.5463% | 91.9130% | 94.7605% |
| test | 10 | 32,728 | 98.4604% | 95.0432% | 96.2148% |
| 全部 | 50 | 173,134 | 98.5438% | 93.2020% | 95.0590% |

采用`episodic_uniform`时，训练记录等权、每段内有效anchor均匀随机，训练集期望的记录宏平均覆盖率为：动作98.5192%、外力槽92.8189%、外力端点94.5962%。每个完整采样epoch产生30个窗口。

173,134个窗口中，11个没有未来外力槽，28个没有未来外力端点标签；动作当前槽仍有效。这些窗口保留独立外力mask，缺失位置的数值padding不作为0 N·m标签参与监督。审核通过意味着可以进行带掩码的离线训练测试；外力固定槽对齐提示仍需在结果解释中保留。

完整覆盖率通过生产`future_torque_slot_indices`计算，再滚动统计100区间；每条首/中/末共150个窗口逐一对照直接调用结果，目标时间及原始源行索引完全一致。详情见[full_chunk_label_coverage.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_50_10points_20261003/full_chunk_label_coverage.json)。

### 读取验证、代码与复现记录

每条实际读取首/中/末3个窗口，共150个；batch=2，检查观测因果性、所有数值有限及张量结构：[data_loader_validation.json](/home/stw/Downloads/1mm-gap-10points_hdf5_raw_v2/data_loader_validation.json)。

| 张量 | 验证形状 |
| --- | --- |
| images | `[2,2,3,480,640]`，RGB，相机顺序d455/d405 |
| qpos | `[2,7]` |
| action | `[2,100,7]` |
| history | `[2,20,8,7]` |
| future torque | `[2,100,8,7]`，3个固定时间槽，其余为容量padding |

本次新建以下两个模块及对应单元测试；没有新增模型类：

- [audit_mcap_right_arm_dataset.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/scripts/audit_mcap_right_arm_dataset.py)：`fingerprint`、`clock_quality`、`alignment_metrics`、`audit_recording`、`summarize`，实现完整源审核、逐流指标与转换前索引冻结。
- [prepare_audited_real_robot_dataset.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/scripts/prepare_audited_real_robot_dataset.py)：`balanced_recording_split`、`require_passed_audit`、`validate_data_batches`，实现审核门槛、逐记录划分、转换与训练集统计、数据读取检查。
- [test_audit_mcap_right_arm_dataset.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/tests/test_audit_mcap_right_arm_dataset.py)及[test_prepare_audited_real_robot_dataset.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/tests/test_prepare_audited_real_robot_dataset.py)：覆盖源SHA/CRC损坏、实际MCAP解码、相位和源行一致、划分防泄漏、失败审核拦截、合成MCAP全流程。

数据相关测试**47项全部通过**，记录在[data_tests_fixed.log](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_50_10points_20261003/data_tests_fixed.log)。复测命令：

```bash
cd /home/stw/ForceAwareACT_workspace/ForceAwareACT
PYTHONPATH=src:/tmp/1mm_gap_python_deps PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/stw/miniconda3/envs/forceact/bin/python -m pytest \
  tests/test_real_robot_conversion.py tests/test_real_robot_data.py \
  tests/test_prepare_1mm_gap_dataset.py tests/test_audit_mcap_right_arm_dataset.py \
  tests/test_prepare_audited_real_robot_dataset.py -q
```

代码与配置SHA-256均已记录，完成时再次核对一致；相关处理代码和新测试保存于[provenance/code](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_50_10points_20261003/provenance/code)，运行库版本保存于[runtime.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_50_10points_20261003/provenance/runtime.json)。汇总与文件哈希见[dataset_preparation_report.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_50_10points_20261003/dataset_preparation_report.json)；逐条50条指标见[episode_quality.csv](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_gap_50_10points_20261003/episode_quality.csv)。

`dataset.json`记录推荐训练采样为`episodic_uniform`。现有训练CLI需要显式传入`--train-sampling-mode episodic_uniform`，并用本批`--data-config`、`--split`；匹配固定验证提议需`--validation-anchor-stride 10`。训练CLI会按同一训练集原生消息规则重算归一化，可与已保存统计核对。本次只完成数据准备，没有初始化模型或优化器。
