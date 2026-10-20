# 1 mm x14y08：57条新数据检查报告

日期：2026-10-20。原始目录：`/home/stw/Downloads/1mm-x14y08`。

**57条均通过文件结构、所声明CRC、关键流及基础采样检查；建议将55条作为成功示范候选，另外2条短记录暂列人工复核。** 用户表示两条可能有问题、其余正常插入；结合本次对话中指出的文件和图像证据，将两条映射为17:43:51和19:44:15。没有将其直接标注为失败。位置分组x14y08来自目录名称。

本次完成检查和报告，尚未进行HDF5转换或训练。文件总量7,282,031,362 bytes（7.28 GB），完整解析1,802,039条消息，全部记录动作时长约24.67分钟。目录没有下载清单，因此SHA-256用于记录本地文件身份和检查审核期间文件未变；未声称与下载源校验值核对。57个源文件SHA-256唯一。

## 两条待复核记录

| 文件 | 录制时长 | 证据和处理建议 |
| --- | ---: | --- |
| Tianji_Marvin_M6CCS_261020_174351_0.mcap | 2.82秒 | 首、中、末图像只见靠近，末端力三个分量接近零，未见明显接触过程。暂不纳入成功示范训练集。 |
| Tianji_Marvin_M6CCS_261020_194415_0.mcap | 10.53秒 | 图像出现靠近后回退，力信号约7～8秒出现接触后回落。暂不纳入成功示范训练集。 |

两条文件本身完整，未发现解码损坏；其任务完整性与插入结果仍应和数据完整性分开判断。原文件全部保留。

![两条短记录与正常记录的双相机首中末对照](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_57_20261020/selected_recordings_contact_sheet.png)

## 关键数据质量

87,777张图像全部解码成功，均640×480；未发现纯黑/纯白帧，也未发现连续相同压缩字节的冻结帧。每个相机每条保留首、中、末3张缩略图，共342张；此次人工查看两条短记录、普通成功记录及最低外力槽覆盖记录的拼图。没有声称逐帧人工确认任务结果。

所选七关节state、action、关节外力估计的数值全部有限，字段齐全；所选log时间戳严格递增。全部57条有效anchor为147,871/148,033（99.8906%）。排除两条待复核记录后，55条有146,546/146,699个有效anchor（99.8957%），动作总时长约24.45分钟。

| 流 | 消息数 | 汇总频率 / Hz | 最长log间隔 / ms |
| --- | ---: | ---: | ---: |
| state | 443,666 | 299.612 | 39.064 |
| disturbance | 443,435 | 299.456 | 39.077 |
| action | 148,119 | 100.015 | 39.464 |
| d455 | 44,393 | 29.969 | 69.428 |
| d405 | 43,384 | 29.294 | 198.063 |
| ft_sensor | 443,138 | 299.255 | 39.083 |

D405是本批需要关注的图像时间质量项：平均29.294 Hz，超过1.5个标称周期的间隔有1,116处；最长198.063 ms发生在`Tianji_Marvin_M6CCS_261020_192405_0`。上一批50条D405平均29.610 Hz、最长101.368 ms。本批全部57条有116个anchor因相机不存在或陈旧被排除，46个因当前动作标签缺失被排除；这是门槛按顺序统计的互斥原因。排除过旧观测后整体采样可用率仍较高。时间间隔变长不直接等价于已证实网络丢包。

## 三种力数据的含义

| 数据 | 字段 | 维度 | 当前含义 |
| --- | --- | ---: | --- |
| 内部关节力矩 | `io_teleop/joint_states.effort` | 7 | 机械臂真实内部关节力矩 |
| 关节外力估计 | `/tianji_marvin_teleop_bridge/estimated_joint_disturbance.effort` | 7 | 现有实机训练读取器使用的外力来源 |
| 末端力传测量 | `/io_teleop/right/ft_sensor.effort` | 6 | 用户确认：已经做过重力补偿的末端六维力/力矩 |

新增流使用`sensor_msgs/JointState`容器，`name=[Fx,Fy,Fz,Tx,Ty,Tz]`、`frame_id=right_ft_measurement`，抽查消息的`position`和`velocity`为空。57条均有该流，共443,138条消息，约299.255 Hz；全部六维数值有限、log/header时间无重复或倒退，各通道有变化，无全零向量。

重力补偿状态已根据用户说明保存，本次没有再做重力扣除。该JointState容器没有显式提供混合力/力矩单位及轴符号说明，图和峰值均使用原始数值。末端六维测量与七维关节外力估计应独立保存；不能直接替换同一个七维数组。**当前raw_v2转换器仅抽取原有五类流，尚不保存新增ft_sensor；后续转换需要扩展保存该流和独立时间轴。**

### 末端接触峰值备注

`Tianji_Marvin_M6CCS_261020_174427_0`的Fz原始值最低为**−133.4261**，位于首条末端力消息后约9.855秒。共435个样本低于−100，分布在约7.622～10.098秒范围内，曲线显示一段较大的接触过程。其余记录中第二大的负向Fz峰值约−85.6692。

这一条仍在用户确认成功的55条之中，但应增加“早段较大接触力”质量备注；成功插入本身不能说明接触过程是否符合希望学习的轻接触策略。本次保留原值，没有自动裁剪、滤波或删除。具体数值与时间位置见[ft_peak_review.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_57_20261020/ft_peak_review.json)。

![末端力测量曲线：两条短记录、普通记录、峰值记录](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_57_20261020/ft_recording_examples.png)

## 既有七维关节外力采样规则的检查

沿用`log_ns`、开头2秒稳定相位的100 Hz动作网格；观测使用anchor之前最近状态/图像，最大年龄20/100 ms；外力历史保留过去200 ms实际样本，每10 ms最多8个；未来关节外力每10 ms固定3槽，容差±1.666666 ms。没有更换时钟、插值或补齐数值。

全57条的紧随10 ms区间关节外力槽覆盖率93.3667%，端点覆盖率95.6591%；55条成功候选对应93.3104%与95.6266%。上一批50条槽覆盖率94.5668%。这仍是固定槽匹配指标，包含记录边界，**不是丢包率，也不是完整100步chunk覆盖率，更不是新增末端力传的覆盖率**。

16条低于90%，最低80.4461%。这些记录均通过基础完整性检查，未来使用现有七维外力时仍应保留缺失标签mask。上一批接收相位/抖动分析可作为解释线索，但本次未做header重对齐对照，因此没有将本批所有缺槽归因于单一原因。

| 记录时间（文件名尾段） | 紧随区间关节外力槽覆盖 |
| --- | ---: |
| 172657 | 85.27% |
| 173133 | 89.11% |
| 173236 | 82.85% |
| 174547 | 88.19% |
| 175137 | 81.36% |
| 175356 | 88.89% |
| 190921 | 85.89% |
| 191834 | 83.22% |
| 192245 | 84.12% |
| 192528 | 85.11% |
| 192900 | 81.87% |
| 193116 | 80.45% |
| 193244 | 81.41% |
| 193840 | 80.67% |
| 193953 | 81.63% |
| 194306 | 85.42% |

这批是x14y08单位置重复示范；55条可用于该位置的离线训练测试候选。记录划分应以完整episode为单位，任务标签、末端力峰值备注与时间匹配mask分别保留。

## 结果文件与审核代码

- 最终人工复核与任务标签：[review_manifest.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_57_20261020/review_manifest.json)。其中55条`task_success=true`，2条`task_success=null`且`hold_for_manual_review`；该文件记录了用户在审核过程中补充的信息。
- 完整源审核：[audit_manifest.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_57_20261020/audit/audit_manifest.json)。审核开始时任务结果未确认，原始审核中的任务标签为null；最终任务标签以review_manifest为准。
- 六维末端力全量检查：[ft_sensor_audit.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_57_20261020/ft_sensor_audit.json)，重力补偿和测量来源记录为用户确认。
- 每条时间、数值、图像统计和采样映射：`/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_57_20261020/audit/episodes/<episode>/`。
- 稳定相位与源行索引保存在每条`sampling_index.npz`；原始文件SHA和审核代码SHA保存于审核清单，处理代码快照位于[code_snapshot](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_57_20261020/code_snapshot)。

本次修改审核脚本，使`audit_recording`支持可选下载参考、显式分组和三态任务标签；CLI增加`--local-files`、`--group`、`--success-provenance`，成功确认不再是检查文件的前提。去除了上一批50条成功标签的硬编码，并保留原有下载清单核验路径。未新增模型类、未改变模型张量；本次核查的数据形状为关节数据`[N,7]`、新增末端力`[N,6]`、原始图像`[N,480,640,3]`。

相关修改：[audit_mcap_right_arm_dataset.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/scripts/audit_mcap_right_arm_dataset.py)、[test_audit_mcap_right_arm_dataset.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/tests/test_audit_mcap_right_arm_dataset.py)、[test_prepare_audited_real_robot_dataset.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/tests/test_prepare_audited_real_robot_dataset.py)。**44项数据相关测试通过**，包括本地无清单模式不伪造下载核验、不继承旧任务成功标签，以及真实CLI子进程检查。测试日志：[data_tests.log](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_57_20261020/data_tests.log)。

```bash
cd /home/stw/ForceAwareACT_workspace/ForceAwareACT
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
 /home/stw/miniconda3/envs/forceact/bin/python -m pytest \
 tests/test_audit_mcap_right_arm_dataset.py tests/test_prepare_audited_real_robot_dataset.py \
 tests/test_real_robot_conversion.py tests/test_real_robot_data.py -q
```
