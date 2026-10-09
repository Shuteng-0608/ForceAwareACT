# Contact-CVAE：末端三维力输入与监督

日期：2026-10-09。本功能用于已有20条数据上的训练流程验证，完整数据收集完成后再正式训练。

## 表示与共用换算

统一训练入口 `scripts/train_real_robot.py` 新增 `--force-representation flange_force3`，也可通过 `configs/real_robot/right_arm_force3.json` 选择。旧配置默认为 `joint_torque7`，默认七维数据合同的序列化内容保持兼容。

`src/force_aware_act/real_robot/flange_force.py` 是训练和可复用推理缓冲共用的纯数学模块，无 SDK、ROS 或硬件连接。`joint_torque_to_flange_force` 同时接受 `[7]` 单帧和 `[N,7]` 批量输入，输出三维力、有效性、完整六维结果、条件数与重建残差。

- 来源是 `/tianji_marvin_teleop_bridge/estimated_joint_disturbance.effort`，顺序 Joint1_R…Joint7_R；内部关节力矩不参与换算。
- 使用已有 M6 CCS4.0 右臂模型，模型 SHA-256 为 `ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52`。
- 几何平移从毫米转换到米，雅可比参考点是法兰中心。
- 对 `tau ≈ J(q).T @ [Fx,Fy,Fz,Mx,My,Mz]` 求七关节残差等权的六维 SVD 最小二乘解，再选前三个力分量。没有在求解时假定末端力矩为零。
- 以0.1m特征长度缩放线速度行，SVD相对截断值1e-10；满秩时等价于普通无权重最小二乘，没有阻尼。
- 按对应样本的 FK 旋转将力表达为移动法兰坐标系，单位 N。参考点不包含未标定的针尖 TCP 偏移。
- 非满秩或缩放条件数超过200的转换标记无效，模型侧零值只是 mask 占位。相对残差保留作诊断，不按残差过滤接触数据。此阈值是数值条件检查，不是机器人安全阈值。
- 没有平滑、插值、额外去偏置或幅值裁剪；符号沿用源扰动估计，不能视为独立标定的末端测力。

## 时间、按需计算与记录

HDF5保持原样。`FlangeForceStream` 只保留原始关节位置、原始扰动力矩和配对索引。启动时检查同帧配对及姿态数值可逆性；每次读取训练样本时才计算该窗口的三维力，不保存预计算力缓存。

使用完全相同且唯一的 `header_ns` 识别同一反馈帧；它不作为消息已到达的依据。派生力可用时间为 `max(disturbance.log_ns, paired_state.log_ns)`，避免使用尚未到达的姿态。不能配对或不能数值求解的样本保留原槽位置并mask。历史只读锚点之前200ms；未来力只作为训练监督。

归一化在训练前对显式训练段的有效原生三维力遍历一次，训练与部署使用同一组固定mean/std，标准差下限0.01。验证和测试记录不参与统计。

对本20条记录的检查结果：201,463个扰动样本中201,462个能同帧配对，1个无法配对，已mask；79个样本需等待稍晚到达的状态。数值条件检查无额外无效样本，全部20条记录的有效锚点与七维版本相同，共67,193个。详见 `outputs/1mm_gap_20_v2/force3_preparation_v1/data_preparation.json`。

每次训练的 `run_manifest.json` 和 checkpoint 都记录表示、单位、求解器、模型校验值、时间语义、归一化及原始文件指纹；`data_quality.json` 和 `sample_traces.json` 记录配对、延迟、无效槽及源索引。七维checkpoint不能续训为三维版本，必须从头训练。

## 模型与损失

| 内容 | 形状/规则 |
|---|---|
| 两路RGB | `[B,2,3,480,640]` |
| 关节位置/动作 | `[B,7]` / `[B,100,7]` |
| 历史三维力 | `[B,20,8,3]`，保留原生采样与独立mask |
| 未来高频力 | `[B,100,8,3]`，每10ms区间三个固定300Hz槽，其余为mask |
| 未来端点力 | `[B,100,3]`，对应每个区间第三槽，独立mask |
| 力损失 | 标准化后的端点L1和高频L1，权重沿用1.0/0.5 |
| 动作及潜变量 | 动作L1、后验KL和先验匹配保持原定义与权重 |
| 验证 | 动作MAE单位rad；三维力及端点MAE单位N，同时报告Fx/Fy/Fz分轴MAE |

三维力损失只对三个力分量计算；完整换算中的三个力矩分量不进入网络和训练目标。当前版本使用与旧版本相同的CVAE结构与隐藏维度，仅力输入投影和力预测输出维度改为3。

## 开始训练与监控

### Batch配置核对（2026-10-09）

最近的50条数据实验使用 **batch=8**：`outputs/1mm_gap_50_10points_v1/runs/contact_seed0_b8/run_manifest.json` 和 `act_seed0_b8/run_manifest.json` 均记录 `run_config.batch_size=8`。两次实验的 `metrics.jsonl` 前四次更新实际batch均为8、8、8、6，对应30条训练记录每个episodic_uniform epoch各取一个窗口，末批不跨epoch补齐。

此前将新三维力训练推荐为batch4，误用了早期20条数据实验的配置。若延续最近50条数据的实验，应使用batch8，并沿用该实验的数据目录、30/10/10划分及训练控制参数。最近50条实验的早停参数为minimum_epochs=500、patience=8、minimum_relative_improvement=0.005，与下方20条示例的宽松早停配置不同；不能只替换batch就称为完全相同配置。

### 最新50条数据、batch8训练命令

4090电脑的Motion-CVAE启动脚本、环境验收和数据传输说明见[Motion三维力4090准备记录](MOTION_FORCE3_4090_SETUP_2026_10_09.md)。该入口保留三维力输入，采用motion posterior。

延续最近 `contact_seed0_b8` 的数据划分、优化参数、采样方式及早停设置，改用三维力配置并从头训练。验证间隔已核对原日志为200次更新，即50个采样epoch。输出使用新目录，checkpoint每200次更新保存。本命令尚未执行长训练。

```bash
source /home/stw/miniconda3/etc/profile.d/conda.sh
conda activate forceact
cd /home/stw/ForceAwareACT_workspace/ForceAwareACT

python -u scripts/train_real_robot.py \
  --mode train --model contact \
  --data-root /home/stw/Downloads/1mm-gap-10points_hdf5_raw_v2 \
  --data-config configs/real_robot/right_arm_force3.json \
  --optimization-config configs/experiments/1mm_gap_50_10points_v1/optimization.json \
  --split configs/datasets/1mm_gap_50_10points_v2/split.json \
  --output-dir outputs/1mm_gap_50_10points_v1/runs/contact_force3_seed0_b8 \
  --train-sampling-mode episodic_uniform \
  --batch-size 8 --chunk-len 100 --seed 0 \
  --max-sampling-epochs 20000 --minimum-sampling-epochs 500 \
  --early-stop-patience 8 --early-stop-min-relative-improvement 0.005 \
  --validation-interval 200 --validation-anchor-stride 10 \
  --selection-metric episode_macro_action_mae_rad \
  --checkpoint-interval 200 --record-timing --device cuda --threads 4
```

监控命令：

```bash
cd /home/stw/ForceAwareACT_workspace/ForceAwareACT
/home/stw/miniconda3/envs/forceact/bin/python scripts/monitor_real_robot_training.py \
  --run-dir outputs/1mm_gap_50_10points_v1/runs/contact_force3_seed0_b8 \
  --watch --interval 5
```

归一化由当前训练集重新计算，其中力为三维、单位N；不复用七维关节力矩的归一化文件。

### 早期20条数据的训练示例

下方保留20条数据、batch4的示例，作为早期数据实验配置，不代表最近50条实验的配置。实现验收仅执行了20条数据上的完整网络6次更新（2个采样epoch）及四条验证记录的134个固定窗口检查，并未启动以下长训练；这项短测试不构成三维力模型batch8的运行验收。

```bash
source /home/stw/miniconda3/etc/profile.d/conda.sh
conda activate forceact
cd /home/stw/ForceAwareACT_workspace/ForceAwareACT

python -u scripts/train_real_robot.py \
  --mode train --model contact \
  --data-root /home/stw/Downloads/1mm_gap_hdf5_raw_v2 \
  --data-config configs/real_robot/right_arm_force3.json \
  --optimization-config configs/experiments/1mm_gap_20_v1/optimization.json \
  --split configs/experiments/1mm_gap_20_v1/split.json \
  --output-dir outputs/1mm_gap_20_v2/runs/contact_force3_seed0_batch4_es3000_p20_r001 \
  --train-sampling-mode episodic_uniform --batch-size 4 --chunk-len 100 --seed 0 \
  --max-sampling-epochs 20000 --minimum-sampling-epochs 3000 \
  --early-stop-patience 20 --early-stop-min-relative-improvement 0.001 \
  --validation-interval 150 --validation-anchor-stride 10 \
  --selection-metric episode_macro_action_mae_rad \
  --checkpoint-interval 50 --record-timing --device cuda --threads 4
```

固定12/4/4段划分。每个采样epoch12个训练窗口、3次更新；每50epoch验证一次，最低3000epoch后启用早停，patience20次验证，改善阈值0.1%。保留动作验证宏平均MAE最小的best.pt。固定验证stride10与原batch4实验一致。

另一个终端执行：

```bash
cd /home/stw/ForceAwareACT_workspace/ForceAwareACT
/home/stw/miniconda3/envs/forceact/bin/python scripts/monitor_real_robot_training.py \
  --run-dir outputs/1mm_gap_20_v2/runs/contact_force3_seed0_batch4_es3000_p20_r001 \
  --watch --interval 5
```

监控中的三维力L1为标准化损失；`force_mae_n`、`force_endpoint_mae_n`、对应分轴指标为未标准化物理量。不能将新版本的N指标与旧版本的Nm指标直接比较。

## 推理接口与验证范围

`RealRobotObservationBuffer.append_external_torque(..., qpos=same_frame_qpos)` 在收到同帧姿态/扰动时调用同一换算函数，保存三维力及有效性。七维接口兼容旧调用。缺少同帧qpos的三维输入明确报错，不用最近状态猜测姿态。

`RealRobotPolicyAdapter` 支持三维checkpoint和独立三维归一化。它不发布控制指令。本次未修改硬件部署脚本；将此模型用于实机前，接收端仍需按上述接口传递同帧姿态，并在目标部署电脑上验证完整负载延迟。

验收记录：`outputs/1mm_gap_20_v2/force3_preparation_v1/full_model_short_test.json`。6次完整网络更新的loss/梯度均有限，三维力输入投影梯度非零；完整网络batch4峰值CUDA分配约2.77GiB。短测试不足以评价最终模型质量。小网络CLI测试覆盖了保存、加载及续训，完整网络短测试不额外保存大型权重。

共用函数在本机Ultra9 275HX、forceact环境中实测10000次单帧换算，均值0.130ms、P99 0.156ms，未超过3.33ms计算预算。这是CPU换算测试，不包含部署端通信、相机和推理负载。

相关测试：

```bash
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/stw/miniconda3/envs/forceact/bin/python -m pytest -q \
  tests/test_real_robot_flange_force.py tests/test_real_robot_data.py \
  tests/test_real_robot_inference.py tests/test_real_robot_cli.py \
  tests/test_real_robot_training.py tests/test_real_robot_model_contract.py \
  tests/test_monitor_real_robot_training.py
```
