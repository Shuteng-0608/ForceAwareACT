传感器三维力训练实验（2026-10-21）

来源与处理
  分支：codex/ee-force3-raw-v3；开发基点：321550b。
  原始数据：~/Downloads/1mm-x14y08_hdf5_raw_v3。
  数据仍为 raw_v3 六维传感器格式；读取 observations/ee_wrench 的列 0、1、2。
  训练表示：end_effector_force3；坐标系：right_ft_measurement；单位：N。
  使用 timestamps/ft_sensor/log_ns，保留原来的相位估计、因果历史与未来标签边界。
  读取时选取 Fx/Fy/Fz，之后才进行归一化、历史/未来窗口编码与损失计算。
  Tx/Ty/Tz 不进入归一化、模型输入、Contact 后验或预测目标。
  上游重力补偿按原元数据继承；没有追加偏置扣除、滤波、削峰、空间变换。
  与旧 flange_force3 的区别：旧模式由关节扰动和关节位置通过雅可比换算，
  使用法兰坐标系；本实验直接使用传感器数据。两种模式在代码和记录中明确区分。

模型与配置
  复用既有 Contact-CVAE / Motion-CVAE 的 force_dim=3 网络。
  保留均值/最大值/最后有效位置池化、时间与位置编码、latent 维度等结构。
  Contact：历史三维力用于在线条件，未来三维力与动作用于训练后验。
  Motion：历史三维力用于在线条件，后验继续使用 qpos 和未来动作。
  动作 [B,20,7]；历史与未来区间力 [B,20,8,3]；endpoint [B,20,3]。
  动作、力历史和未来力窗口都是 0.2 秒；8 是区间容量，保留真实有效性 mask。
  完全复用 configs/datasets/1mm_x14y08_55_wrench6_v1/split.json，40/10/5。
  仅用 40 条训练记录计算均值/标准差，与六维实验前三维相同。
  batch=8，episodic_uniform，seed=0，每 epoch 5 次更新。
  学习率/骨干学习率=1e-5，weight_decay=1e-4，梯度裁剪=1.0。
  每 250 次更新（50 epoch）验证；stride=10；使用同一验证锚点。
  最大 20000 epoch，早停最少 3000 epoch、patience=20 次验证、相对改善=0.001。
  最优 checkpoint 按 episode_macro_action_mae_rad 选择；验证使用部署时的零隐变量。

损失权重
  action=1.0，endpoint force=0.5，interval force=0.25，posterior KL=10。
  Contact 另有 prior match=1.0；Motion 没有 prior match 项。
  六维版本的每路 loss 对六个归一化通道平均，相当于三维力与三维力矩各占一半。
  这里保留原三维力项的系数：1.0/2=0.5、0.5/2=0.25。
  这一选择控制的是损失系数，不表示不同网络的数值梯度或初始权重完全相同。
  实际权重见本目录 optimization.json；日志中的 loss_force 等是加权前的值。
  三维与六维的 total loss 包含不同任务，不直接以 total loss 大小比较模型优劣。
  比较共同验证集上的动作 MAE、三维力 endpoint/interval MAE（N）与逐轴误差。

处理记录
  preparation_summary.json：来源约定、配置、权重策略和准备核验摘要。
  normalization.json：训练集归一化统计。
  verification_summary.json：回归测试、真实数据短测试与 checkpoint 恢复结果。
  outputs/1mm_x14y08_55_sensor_force3_v1/preparation/preparation.json：逐条数据对齐记录。
  各次运行的 run_manifest.json 另存完整数据配置、来源 SHA256、统计量和训练参数。

实现位置
  real_robot/config.py：uses_wrench_sensor 属性与 end_effector_force3 数据约定。
  real_robot/reader.py、normalization.py、data.py：数据来源与使用维度分开处理。
  real_robot/wrench_metrics.py：evaluate_sensor_force / aggregate_sensor_force_metrics，
  复用传感器评估归约，保持六维旧接口兼容，三维只记录力与保持当前力基线。
  train_real_robot.py、real_robot/evaluation.py：显式传递数据表示，避免把三维传感器误标为法兰换算。
  monitor_real_robot_training.py：显示传感器坐标系、三维力 loss 和单位 N。
  prepare_sensor_force3_training.py：可重复的归一化与时间/通道/mask 一致性核验。
  没有新增模型类；实机部署接收器不在本次改动范围。

复核命令（在仓库根目录，forceact 环境）
  PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest tests/test_real_robot*.py tests/test_prepare_sensor_force3_training.py tests/test_prepare_wrench6_training.py tests/test_monitor_real_robot_training.py -q

  python scripts/prepare_sensor_force3_training.py --data-root "$HOME/Downloads/1mm-x14y08_hdf5_raw_v3" --output-dir outputs/1mm_x14y08_55_sensor_force3_v1/preparation_review

  bash scripts/training/train_x14y08_sensor_force3.sh contact test --output-dir outputs/1mm_x14y08_55_sensor_force3_v1/checks/contact_review
  bash scripts/training/train_x14y08_sensor_force3.sh motion test --output-dir outputs/1mm_x14y08_55_sensor_force3_v1/checks/motion_review

  test 为完整网络、batch=8、10 次更新；输出目录已存在运行时会拒绝覆盖。

正式训练入口（待用户审核后执行，本次没有启动）
  bash scripts/training/train_x14y08_sensor_force3.sh contact train
  bash scripts/training/train_x14y08_sensor_force3.sh motion train

  训练输出独立保存到 outputs/1mm_x14y08_55_sensor_force3_v1/runs/。
  从头训练；三维模式不能直接恢复六维模式 checkpoint。
  4090 当前六维队列仍使用原分支，本次开发只在本机进行。

监控示例
  python scripts/monitor_real_robot_training.py --run-dir outputs/1mm_x14y08_55_sensor_force3_v1/runs/contact_sensor_force3_ah20_fh20_seed0_b8_es3000_p20_r001 --watch --interval 5
