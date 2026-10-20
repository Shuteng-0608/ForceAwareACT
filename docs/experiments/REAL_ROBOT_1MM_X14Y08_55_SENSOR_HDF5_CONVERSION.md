# 1mm-x14y08：55条末端六维力HDF5转换完成报告

完成时间：2026-10-20 21:20:02（Asia/Shanghai）。**55/55条已转换为`real_robot_right_arm_raw_v3`，逐条完整二次源数据比较通过；本次仅进行原始数据转换。**

## 数据位置与数量

- 原始MCAP：`/home/stw/Downloads/1mm-x14y08`，当前55条，合计7,218,578,363 bytes。
- HDF5目录：[1mm-x14y08_hdf5_raw_v3](/home/stw/Downloads/1mm-x14y08_hdf5_raw_v3)，55个`<episode_id>/episode.hdf5`，共58,203,234,463 bytes（58.20 GB / 54.21 GiB）。
- 数据格式说明：[README.md](/home/stw/Downloads/1mm-x14y08_hdf5_raw_v3/README.md)。
- 批量转换清单：[conversion_manifest.json](/home/stw/Downloads/1mm-x14y08_hdf5_raw_v3/conversion_manifest.json)。
- 本次55条源清单：[source_inventory.json](/home/stw/Downloads/1mm-x14y08_hdf5_raw_v3/source_inventory.json)。用户已删除的174351、194415两条未纳入；原57条审核背景保存于provenance目录。

| 数据流 | 原生消息数 |
| --- | ---: |
| state | 439,663 |
| action | 146,783 |
| estimated_joint_disturbance | 439,433 |
| ft_sensor | **439,136** |
| d455 | 43,993 |
| d405 | 42,998 |

图像合计86,991帧。HDF5同时保存原始JPEG和640×480 RGB像素；较大存储量主要来自解码RGB。

## 末端力数据的保存方式

主力数据为`observations/ee_wrench`，float64 `[N_ft,6]`，按消息name固定映射到`[Fx,Fy,Fz,Tx,Ty,Tz]`。来源是`/io_teleop/right/ft_sensor.effort`；独立时间轴为`timestamps/ft_sensor`，完整保存header_ns、log_ns、publish_ns、sequence、source_topic_message_index和逐消息frame_id。

用户确认的“已经经过重力补偿”写入根属性和力数据集属性：`gravity_compensation=applied_upstream`，并保存说明来源和日期。转换没有再次扣除重力、去偏、滤波、裁剪、插值或进行坐标/单位换算。单位元数据为原始`source_native`，因为源JointState没有显式编码混合力/力矩单位。当前坐标系原样保存为`right_ft_measurement`。

原七维`observations/joint_external_torque`继续保存为辅助估计；`observations/joint_internal_torque`保存内部关节力矩。它们拥有各自来源和时间轴，主力字段明确指向末端传感器。

## 保真验证

每条文件先转换到临时文件，再完整重读选定MCAP数据流进行比较，通过后发布HDF5：

- 原始float64数值按对应关节/轴逐字节一致；末端力保留全部六维。
- 所有选定时间戳、sequence、源topic行索引和frame_id完全一致。
- 原始JPEG字节、格式字符串及解码RGB像素完全一致。
- 转换前后源MCAP的SHA-256一致；每个输出HDF5也记录SHA-256。
- 最终55条HDF5的六维轴序、形状、消息数、frame_id计数、最小值、最大值和标准差均与转换前末端力审核一致。

55个输出均通过，没有遗留`.partial`文件。此前174427记录的Fz最小值**−133.42608858212827**原样保留。每条目录都有`quality_report.json`和`conversion.log`；汇总验证见[final_verification.json](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_55_conversion_20261020/final_verification.json)。

原审核中的“16条固定外力槽覆盖较低”针对七维关节外力估计。此次保存原生异步流，尚未对新六维传感器生成训练采样网格，不把旧提示解释为新传感器的缺失率。

## 本次代码与验证范围

修改[convert_mcap_right_arm.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/scripts/convert_mcap_right_arm.py)：新增`stream_spec`、传感器读取与六轴校验、raw_v3元数据，`convert`可选择包含传感器；`verify_against_source`扩展六维数值及来源校验；新增`convert_verified`统一完整源比较和文件发布。原raw_v2路径继续保留。

新增[convert_reviewed_mcap_dataset.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/scripts/convert_reviewed_mcap_dataset.py)：`reviewed_sources`检查当前文件与55条审核身份一致，`convert_one`逐条转换验证，`main`管理批量状态、哈希和代码快照。没有新增模型类或修改训练/部署代码。

原始数组形状为：末端力`[N_ft,6]`；关节位置/内部力矩`[N_state,7]`；关节外力估计`[N_disturbance,7]`；动作`[N_action,7]`；图像`[N_camera,480,640,3]`。各流长度独立保存。

扩展[test_real_robot_conversion.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/tests/test_real_robot_conversion.py)并新增[test_convert_reviewed_mcap_dataset.py](/home/stw/ForceAwareACT_workspace/ForceAwareACT/tests/test_convert_reviewed_mcap_dataset.py)。**57项相关测试通过**，覆盖真实MCAP、轴顺序、独立时钟、补偿状态、缺失/非法传感器、输出损坏检出、文件身份变化、批量CLI及原路径兼容。测试日志：[tests.log](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_55_conversion_20261020/tests.log)。

```bash
cd /home/stw/ForceAwareACT_workspace/ForceAwareACT
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
 /home/stw/miniconda3/envs/forceact/bin/python -m pytest \
 tests/test_real_robot_conversion.py tests/test_convert_reviewed_mcap_dataset.py \
 tests/test_audit_mcap_right_arm_dataset.py tests/test_prepare_audited_real_robot_dataset.py \
 tests/test_real_robot_data.py -q
```

处理代码快照、用户说明来源、原审核背景、运行库版本和测试记录保存在[provenance](/home/stw/Downloads/1mm-x14y08_hdf5_raw_v3/provenance)。本次批量调用记录在[conversion.log](/home/stw/ForceAwareACT_workspace/ForceAwareACT/outputs/1mm_x14y08_55_conversion_20261020/conversion.log)；转换入口为`convert_reviewed_mcap_dataset.py`，使用55条当前源清单、4个worker及明确的上游重力补偿声明。

输出状态为`converted_verified`，`training_ready=false`。本次没有生成训练划分、归一化、对齐采样网格或启动训练；后续训练读取器需适配raw_v3及传感器数据来源。
