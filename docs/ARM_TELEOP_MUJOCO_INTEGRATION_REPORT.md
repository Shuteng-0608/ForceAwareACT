# Arm Teleop MuJoCo Integration Report

This report documents the read-only integration study of `../arm_teleop` on
branch `dev_peg_in_hole` for a future ForceAwareACT closed-loop rollout. The
selected collection model and runtime behavior were cross-checked against one
recorded episode from the 100-episode collection. No rollout implementation is
included here.

## 1. Entry Point and Runtime Flow

The main entry point is `../arm_teleop/vptele/main_mujoco.py`.

1. `TeleopROSNode` initializes a ROS node named `teleop_system`.
2. It reads ROS parameters, defaulting to
   `config/config_arm_right_peg.yaml`.
3. `run_teleop_system()` resolves and loads the YAML configuration, configures
   logging, constructs `TeleopSystemMujoco`, and calls
   `system.initialize(mode="full")`.
4. `TeleopSystemMujoco` initializes the Vision Pro stream, waits for a valid
   right-wrist transform, constructs the peg-tool MuJoCo controller, and
   constructs `ArmTeleopMujoco`.
5. `system.start()` starts the arm teleoperation thread. The outer ROS loop is
   only a 10 Hz supervisory loop waiting for shutdown.

The actual MuJoCo physics/control loop runs in
`RobotControllerMuJoCoPegTool.visualization_thread()`. With the configured
actuator mode, each iteration:

1. Rate-limits `command_joints` toward `target_joints`.
2. Writes position targets to `data.ctrl`.
3. Calls `mujoco.mj_step()`.
4. Samples active recorders when their simulation-time deadlines are reached.
5. Optionally renders the camera monitor.
6. Synchronizes the passive viewer at 60 Hz.
7. Sleeps to approximately maintain real time.

The older `robot_controller_mujoco.py` is not the active peg collection
controller. It directly writes qpos during its loop and lacks the peg-specific
contact, HDF5, camera, force-sensor, and task-randomization behavior.

The teleoperation command path is separate:

1. `ArmTeleopMujoco` reads the latest Vision Pro right-wrist pose.
2. It maps and filters the pose.
3. It calls the ROS IK service `/arm_teleop/right_arm_ik_srv`.
4. It smooths the returned joint solution.
5. It calls `robot_controller.set_arm_positions()`.

Runtime dependencies include:

- ROS and the generated `arm_teleop` service types.
- The `/arm_teleop/right_arm_ik_srv` IK service.
- A Vision Pro tracking stream and configured `vp_ip`.
- MuJoCo and its passive viewer.
- OpenCV for the live camera monitor.
- The optional keyboard recording client, which calls a ROS service.

A policy-only rollout does not need Vision Pro or IK if the policy directly
produces joint-position targets, but it should reuse the MuJoCo controller's
physics, rendering, sensing, and actuator-target behavior.

## 2. MuJoCo Model and XML

The default peg configuration selects:

```text
../arm_teleop/model/pangu_all_right.xml
```

The YAML contains an absolute path for the original collection machine, but a
recorded 100-episode HDF5 file also identifies `pangu_all_right.xml` as its
model. The important model candidates are:

- `pangu_all_right.xml`: selected collection model; includes the two recorded
  cameras and the peg/wall contact task.
- `right_arm_peg_tool_wall_contact.xml`: closely related fallback model with
  peg/hole task sites, but without the collection cameras.
- `right_arm_stable.xml`: general arm/hand model, not the selected peg
  collection model.

`pangu_all_right.xml` defines:

- A seven-joint right arm with mesh geometry.
- A rigid `peg_tool` body attached to `link_7`.
- A collidable cylindrical peg named `cylindrical_peg`.
- A collidable floor.
- A `wall_task` body containing four wall-frame bars and a 24-segment circular
  hole ring.
- Fixed world cameras plus an end-effector-mounted camera.
- Position actuators for all seven arm joints.
- Force and torque sensors at `ft_sensor_site`.

There is no separate table geom in the selected XML; the task consists of the
floor plus the vertical wall/hole fixture.

The selected XML uses a 0.001 s MuJoCo timestep, equivalent to a nominal
1,000 Hz physics loop. Contact computation is enabled for the peg, floor, and
wall/hole task geometry; robot body collision geometry is effectively disabled
by the XML defaults.

The selected XML has `peg_tip_site` and `hole_center_site` commented out. This
means the recorder's initial/final peg-hole error metadata is written as
non-finite values, and the visual alignment guides cannot use those sites. The
force/torque sensing site remains available.

## 3. Joint and Actuator Interface

The arm joint order is:

1. `joint_1`
2. `joint_2`
3. `joint_3`
4. `joint_4`
5. `joint_5`
6. `joint_6`
7. `joint_7`

The corresponding actuator order is:

1. `motor_joint_1`
2. `motor_joint_2`
3. `motor_joint_3`
4. `motor_joint_4`
5. `motor_joint_5`
6. `motor_joint_6`
7. `motor_joint_7`

All seven actuators are MuJoCo position actuators. The controller's normal
contact-compatible path writes the internal joint-position targets to
`data.ctrl`, then advances dynamics with `mujoco.mj_step()`. It does not
continuously overwrite `data.qpos`. Direct qpos writing exists only for
initialization/reset and a debug `qpos` control mode.

The configured initial teleoperation joints are:

```text
[-0.046, -0.2, 0.0, 1.6, -1.32, 0.005, 0.005]
```

The public teleoperation API applies:

```text
arm_sign = [-1, 1, 1, -1, 1, 1, 1]
```

before setting internal targets. Therefore the initial internal MuJoCo pose has
the signs of joints 1 and 4 flipped. On initialization and simulation-thread
reset, the controller hard-sets this internal pose, zeros joint velocities,
sets matching actuator targets, and calls `mj_forward()`.

This sign conversion is the most important rollout integration issue.
ForceAwareACT's HDF5 `joint_pos` and action targets are read directly from
MuJoCo `data.qpos`, so predictions are in the internal MuJoCo convention.
Passing them directly to `set_arm_positions()` would apply the sign conversion
again. A rollout must use an explicitly internal target interface, or convert
from internal predictions back to the public teleoperation convention before
calling the public method.

The controller limits internal command motion to 0.5 rad/s by advancing each
command toward its target on every physics step. XML actuator control ranges are
`[-3.2, 3.2]`; force limits vary by joint.

## 4. Camera Interface

The HDF5 collection cameras are:

- `ee_cam`: mounted on `peg_tool`, near the flange/peg connection, with
  `fovy="80"`.
- `base_top_cam`: fixed near the robot base/top and aimed at the wall task,
  with `fovy="60"`.

Both are present in `pangu_all_right.xml`. Other available cameras are
`wall_task_cam`, `wall_side_cam`, and `cctv_cam`. The passive viewer is set to
`cctv_cam`; this does not change HDF5 camera rendering.

The HDF5 recorder creates a `mujoco.Renderer` at 640 by 480, selects each
configured camera, and stores the rendered result as raw `uint8` RGB with shape
`[480, 640, 3]`. The live monitor separately converts RGB to BGR and overlays
camera labels for OpenCV display; those labels are not stored in HDF5.

ForceAwareACT preprocessing must match training:

- Use cameras in order `("ee_cam", "base_top_cam")`.
- Resize RGB frames from 480 by 640 to 224 by 224.
- Convert to channel-first float tensors and scale to `[0, 1]`.
- Use the same ImageNet-normalization setting as the trained checkpoint/data
  pipeline. Current training scripts default to no ImageNet normalization.

`enable_visual_guides` is false in the collection config. Additionally, the
custom guide geometry is added to the passive viewer's user scene rather than
the HDF5 renderer. Recorded policy images therefore do not contain the visual
guides.

## 5. Force/Wrench Interface

The selected XML defines:

- Site: `ft_sensor_site`, located at the origin of `peg_tool`.
- Force sensor: `peg_ft_force`.
- Torque sensor: `peg_ft_torque`.

The HDF5 recorder reads both sensors directly from `data.sensordata` and
concatenates them as:

```text
[Fx, Fy, Fz, Tx, Ty, Tz]
```

The controller documents these values as expressed in the local
`ft_sensor_site` frame. Because the site has no explicit local rotation, its
orientation follows the `peg_tool` body. The exact MuJoCo sign convention and
whether deployment needs bias removal or frame transformation should still be
verified with controlled contact tests.

The XML physics timestep is 0.001 s, while the configured force recording rate
is 500 Hz. The recorder runs after every physics step and appends a force sample
whenever simulation time reaches the next 0.002 s force deadline. State and
image streams are recorded at 30 Hz. ForceAwareACT constructs a fixed-length
past-only force window from `timestamps/force_episode`.

Potential online mismatch risks include:

- Sampling a live force window at a different rate or duration than training.
- Using world-frame forces instead of the local sensor frame.
- Reordering force and torque components.
- Changing the XML, sensor site transform, contact parameters, or actuator
  gains.
- Not reproducing the recorder's simulation-time alignment.
- Async stop-time one-frame length mismatches in recorded streams; the dataset
  reader tolerates and safely trims these, but online buffers must remain
  coherent.

## 6. HDF5 Recording Pipeline

The active compact recorder is
`../arm_teleop/vptele/utils/mujoco_hdf5_recorder.py`,
instantiated by `RobotControllerMuJoCoPegTool`.

It records:

| HDF5 field | Source |
|---|---|
| `observations/ee_pose` | `peg_tool` body world position and quaternion `[x, y, z, qw, qx, qy, qz]` |
| `observations/joint_pos` | Internal MuJoCo `data.qpos` for `joint_1` through `joint_7` |
| `observations/joint_vel` | Internal MuJoCo `data.qvel` |
| `observations/joint_torque` | `data.qfrc_actuator` at each joint DOF |
| `observations/ft_wrench` | Concatenated `peg_ft_force` and `peg_ft_torque` sensor values |
| `observations/images/ee_cam` | Raw HDF5 `uint8` RGB frames |
| `observations/images/base_top_cam` | Raw HDF5 `uint8` RGB frames |
| `timestamps/state`, `force`, `image` | Absolute MuJoCo `data.time` |
| `timestamps/state_episode`, `force_episode`, `image_episode` | MuJoCo time relative to recording start |
| `episode_metadata/` | Rates, names, model path, initial/final state, task labels, and nominal task-site values |
| `events/` | Record-start and stop events with simulation and wall-clock times |

Configured rates are:

- Physics: nominal 1,000 Hz.
- Force: 500 Hz.
- State: 30 Hz.
- Images: 30 Hz.
- Live camera monitor: 15 Hz.
- Passive viewer synchronization: 60 Hz.

All stream scheduling uses MuJoCo `data.time`, and the recorder samples from the
physics thread after `mj_step()`. A checked collection episode confirms the
selected model, rates, camera order, image shape, joint order, and actuator
order.

Recording is not automatic in the selected config. The keyboard client
`../arm_teleop/scripts/recording_keyboard_client.py` calls
`/mujoco_hdf5_recording/set_recording`:

- First Enter: starts an episode, after randomizing `wall_task`.
- Second Enter: asks whether to keep the episode, then stops recording.
- A discarded episode is removed by the external collection project.

The configured hole randomization changes `wall_task` before each episode:

- x offset: `[-0.020, 0.020]` m.
- y offset: fixed at `+0.010` m relative to nominal.
- z offset: `[-0.020, 0.020]` m.

The compact recorder does not currently write `last_hole_randomization` into
HDF5 metadata. It also leaves `task_success` as `unknown`.

## 7. Mapping to ForceAwareACT Inputs

| ForceAwareACT input | Source in arm_teleop / MuJoCo | Shape | Rate | Preprocessing needed | Potential issue |
|---|---|---:|---:|---|---|
| `images` | Render `ee_cam`, then `base_top_cam` | Online raw `[2, 480, 640, 3]`; model `[1, 2, 3, 224, 224]` | 30 Hz training | RGB, resize, float32, scale `[0,1]`, optional training-matched ImageNet normalization | Camera order, resolution, color format, lighting, and XML must match |
| `qpos` | Internal `data.qpos` for `joint_1` to `joint_7` | `[1, 7]` | 30 Hz training; available each physics step | Apply saved `qpos_mean/std` | Must not confuse internal qpos with sign-adjusted teleop joints |
| `qvel` | Internal `data.qvel` at joint DOFs | `[1, 7]` | 30 Hz recorded | Log only; normalize if used later | Current policy forward does not consume it |
| `joint_torque` | `data.qfrc_actuator` at joint DOFs | `[1, 7]` | 30 Hz recorded | Log only; normalize if used later | Actuator generalized force is not an external joint torque sensor |
| `ee_pose` | `peg_tool` body `xpos/xquat` | `[1, 7]` | 30 Hz recorded | Log only | Current policy forward does not consume it; not peg-tip pose |
| `force_window` | Ring buffer of `peg_ft_force` + `peg_ft_torque` | Raw buffer, resampled to `[1, L, 6]` | 500 Hz source | Past-only window over configured duration, resample to `L`, apply saved `force_mean/std` | Frame/sign/rate and first-window padding must match dataset behavior |
| Action target | ForceAwareACT `pred_action`, trained from future internal `joint_pos` | Predicted `[1, K, 7]`; execute one or a short horizon | Training targets at 30 Hz | Denormalize with saved `action_mean/std`; clip and smooth | Predictions are internal qpos; public teleop setter applies signs |

The current model consumes only `images`, normalized `qpos`, and normalized
past `force_window` during deployable inference. It predicts normalized future
action and force chunks. `qvel`, `joint_torque`, and `ee_pose` should still be
logged during rollout for diagnostics.

## 8. Proposed Deployment Architecture

A future `run_mujoco_policy_rollout.py` should use the following architecture:

1. **Load artifacts.** Load the ForceAwareACT checkpoint, reconstruct the model
   from checkpoint config, load normalization statistics, select deterministic
   contact-prior inference, and set the model to evaluation mode.
2. **Initialize MuJoCo.** Load the relative local equivalent of
   `pangu_all_right.xml`, construct `RobotControllerMuJoCoPegTool` in actuator
   mode, and reset to a known internal initial pose.
3. **Initialize task state.** Either reproduce collection-time `wall_task`
   randomization or use an explicit fixed validation configuration. Record the
   chosen task pose.
4. **Create online sensors.** Resolve joint IDs, camera IDs, and
   `peg_ft_force`/`peg_ft_torque`; allocate a timestamped past-force ring
   buffer.
5. **Run physics continuously.** Keep MuJoCo stepping at the XML timestep and
   actuator targets updating through the existing rate-limited control path.
6. **Run policy at the training state rate.** At approximately 30 Hz, render
   both cameras, read internal qpos, resample the past-only 0.25 s force window,
   apply training-equivalent preprocessing and normalization, and run
   `contact_latent_mode="prior"` with deterministic prior mean.
7. **Execute safely.** Denormalize `pred_action`; begin with receding-horizon
   execution of only the first target. Send it through an explicitly internal
   target method, with joint-limit clipping, delta clipping, velocity limiting,
   finite checks, and an emergency stop. Do not pass internal predictions
   through `set_arm_positions()` without correcting its sign conversion.
8. **Log rollout data.** Record raw observations, normalized model inputs,
   predicted action/force chunks, selected commands, actual state, wrench,
   contacts, task geometry, inference latency, safety interventions, and
   termination reason.
9. **Terminate conservatively.** Stop on timeout, non-finite predictions,
   excessive force/torque, joint-limit violation, viewer close, or explicit
   operator stop.

Vision Pro and the IK service should be excluded from policy rollout unless
they are intentionally retained as an operator override. The existing ROS
recording service can be reused for observation logging, but rollout-specific
prediction and safety metadata will need a separate read/write design later.

## 9. Risks and Open Questions

1. **Internal versus teleoperation joint convention.** Confirm the exact
   command interface for internal qpos predictions and add a regression test
   proving no double sign conversion occurs.
2. **Force frame and sign.** Verify `ft_sensor_site` axes and wrench signs using
   controlled contacts along known world directions.
3. **Sensor bias.** Determine whether the MuJoCo sensor has a nonzero
   gravity/contact-free baseline and whether training implicitly learned it.
4. **Policy rate.** Decide whether inference runs at exactly 30 Hz, and whether
   one target or several chunk targets are executed per inference.
5. **Simulation scheduling.** Camera rendering and inference must not stall or
   destabilize the 1,000 Hz physics loop; separate threads may be needed.
6. **Action safety.** Define joint limits, maximum per-command delta, maximum
   target velocity, force thresholds, smoothing, and emergency-stop behavior.
7. **Camera match.** Confirm RGB order, camera order, 640 by 480 rendering,
   resize interpolation, lighting, and ImageNet-normalization setting.
8. **Initial-state distribution.** Define resets and task randomization that
   remain within the demonstrated distribution.
9. **Success/failure metric.** `task_success` is currently `unknown`, and the
   selected XML comments out peg-tip and hole-center sites. A robust success
   criterion is required before evaluating closed-loop rollout.
10. **Task metadata.** Hole randomization is not persisted by the compact
    recorder, limiting reproducibility and stratified evaluation.
11. **Contact geometry compatibility.** Model XML, gains, solver settings,
    timestep, friction, and contact parameters must remain consistent with
    collection.
12. **Predicted future force use.** The force head should initially be used for
    diagnostics and safety monitoring only, not as a direct control input.
13. **No-leakage guarantee.** Deployable rollout must use only online images,
    current qpos, and past force; posterior/oracle mode must remain disabled.
14. **Thread safety.** Reading/rendering and setting targets must respect the
    controller lock and MuJoCo renderer constraints.

## 10. Recommended Next Implementation Plan

1. Add a read-only environment probe that loads `pangu_all_right.xml`, resolves
   all required joints, actuators, cameras, sensors, and prints their IDs,
   shapes, timestep, and control ranges.
2. Add a coordinate-convention test that compares recorded internal qpos,
   `get_current_joints()`, public `set_arm_positions()`, and an internal-target
   setter. Do not proceed until the sign behavior is explicit and tested.
3. Add a force-frame calibration probe that applies known contacts and verifies
   force/torque component directions, baseline, and sampling behavior.
4. Add a deterministic reset/task-randomization helper and log the exact
   `wall_task` pose used for every rollout.
5. Implement an online observation adapter that reproduces dataset image
   preprocessing and past-force resampling exactly. Compare its tensors against
   `ContactForceHDF5Dataset` on a recorded episode.
6. Implement a policy runner that loads checkpoint/stats and performs
   deterministic prior inference without sending commands. Validate shapes,
   latency, finite outputs, and denormalization.
7. Implement guarded internal qpos-target execution with first-action-only
   receding horizon, conservative clipping, force limits, timeout, and operator
   stop.
8. Run free-space rollouts before enabling peg contact. Compare commanded and
   measured joint trajectories.
9. Run fixed-task contact rollouts with low force limits, then gradually test
   collection-matched hole randomization.
10. Define success/failure instrumentation and evaluate zero versus prior mode
    under identical resets before making performance claims.
