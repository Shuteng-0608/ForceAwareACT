# Current arm_teleop HDF5 Recording and Command Audit

## Executive Conclusion

The current fixed-start insertion HDF5 dataset records actual MuJoCo state and sensor observations, not executable joint command labels.

The active MuJoCo teleoperation controller uses actuator position control. Public teleoperation joint targets are converted to internal MuJoCo joint convention, stored in `target_joints`, rate-limited into `command_joints`, then written to `data.ctrl`. The compact HDF5 recorder does not currently write `target_joints`, `command_joints`, or `data.ctrl` into the HDF5 file.

Therefore, the current ForceAwareACT action target is future actual joint state:

```text
action_chunk = observations/joint_pos[i + 1 : i + K + 1]
```

It is not the command that was sent to MuJoCo actuators. This explains why static forward replay of `joint_pos` can show successful insertion while dynamic replay of `joint_pos` as commands can fail or produce high force: the file contains states achieved by the closed-loop simulator/controller, but not the exact actuator command trajectory that produced those states.

## Files Inspected

arm_teleop:

- `/Users/wangshuteng/Desktop/arm_teleop/vptele/main_mujoco.py`
- `/Users/wangshuteng/Desktop/arm_teleop/vptele/core/teleop_system_mujoco.py`
- `/Users/wangshuteng/Desktop/arm_teleop/vptele/arm_control/robot_controller_mujoco_peg_tool_contact.py`
- `/Users/wangshuteng/Desktop/arm_teleop/vptele/arm_control/arm_teleop_mujoco.py`
- `/Users/wangshuteng/Desktop/arm_teleop/vptele/utils/mujoco_hdf5_recorder.py`
- `/Users/wangshuteng/Desktop/arm_teleop/vptele/config/config_arm_right_peg.yaml`
- `/Users/wangshuteng/Desktop/arm_teleop/model/pangu_all_right.xml`

ForceAwareACT:

- `/Users/wangshuteng/Desktop/ForceAwareACT/src/force_aware_act/data/contact_force_hdf5_dataset.py`

Representative HDF5 files:

- `/Users/wangshuteng/Desktop/ForceAwareACT/mujoco_data/peg_hole_fixed_insertion/20260617_125307_teleop_015/episode.hdf5`
- `/Users/wangshuteng/Desktop/ForceAwareACT/mujoco_data/peg_hole_fixed_insertion/20260617_131825_teleop_071/episode.hdf5`
- `/Users/wangshuteng/Desktop/ForceAwareACT/mujoco_data/peg_hole_fixed_insertion/20260617_133655_teleop_124/episode.hdf5`

## Active Runtime and Recorder Path

The ROS entry point is `vptele/main_mujoco.py`. `TeleopROSNode` reads `~config_path`, defaulting to `config/config_arm_right_peg.yaml`, then calls `run_teleop_system()` and starts the returned system. `run_teleop_system()` constructs `TeleopSystemMujoco` and calls `system.initialize()` (`main_mujoco.py`, lines 12-63 and 65-125).

`TeleopSystemMujoco._initialize_robot_controller()` constructs the contact MuJoCo controller, explicitly selecting actuator mode:

```text
control_mode = "actuator"
max_joint_velocity = 0.5
arm_sign = [-1, 1, 1, -1, 1, 1, 1]
record_hdf5 = true
hdf5_state_hz = 30
hdf5_force_hz = 500
hdf5_image_hz = 30
```

This is in `core/teleop_system_mujoco.py`, lines 98-224.

The active controller class is `RobotControllerMuJoCoPegTool` in `vptele/arm_control/robot_controller_mujoco_peg_tool_contact.py`. It creates `MujocoHDF5Recorder` when `record_hdf5` is true (`robot_controller_mujoco_peg_tool_contact.py`, lines 253-302).

The recording service is registered at `/mujoco_hdf5_recording/set_recording` (`robot_controller_mujoco_peg_tool_contact.py`, lines 315-333). Starting recording stops old teleop, resets the arm, optionally randomizes the hole, recalibrates teleop, starts HDF5 recording, enables commands, then starts teleoperation (`robot_controller_mujoco_peg_tool_contact.py`, lines 515-618). Stopping recording disables commands, stops teleop, stops the HDF5 recorder, and resets the arm (`robot_controller_mujoco_peg_tool_contact.py`, lines 619-655).

`record_if_needed()` is called after every `mj_step()` inside `_physics_step()` (`robot_controller_mujoco_peg_tool_contact.py`, lines 1054-1075).

## Command Pipeline

Text diagram:

```text
Vision Pro right wrist transform
  -> ArmTeleopMujoco.map_hand_to_robot()
  -> pose filtering
  -> /arm_teleop/right_arm_ik_srv
  -> IK solution joint_angles
  -> smooth_values() -> smooth_joint_angles
  -> robot_controller.set_arm_positions(smooth_joint_angles)
  -> _convert_arm_command_to_internal(..., arm_sign)
  -> target_joints[:7]
  -> _step_command_toward_target_locked(dt)
       command_joints += clipped(target_joints - command_joints,
                                 max_joint_velocity * dt)
  -> _apply_actuator_targets(command_joints)
  -> data.ctrl[actuator_id] = command_joints[i]
  -> mujoco.mj_step()
  -> HDF5 recorder samples actual data.qpos/qvel/qfrc_actuator/sensordata
```

Important code points:

- `ArmTeleopMujoco` gets the IK solution, smooths it, and calls `set_arm_positions(smooth_joint_angles)` (`arm_teleop_mujoco.py`, lines 440-512).
- `set_arm_positions()` rejects commands unless recording/teleop has enabled them, converts public convention to internal convention, and writes `target_joints` (`robot_controller_mujoco_peg_tool_contact.py`, lines 918-930).
- `_convert_arm_command_to_internal()` applies `arm_sign = [-1, 1, 1, -1, 1, 1, 1]` (`robot_controller_mujoco_peg_tool_contact.py`, lines 878-882).
- `_step_command_toward_target_locked()` rate-limits `command_joints` toward `target_joints` using `max_joint_velocity * dt` (`robot_controller_mujoco_peg_tool_contact.py`, lines 1040-1046).
- `_apply_actuator_targets()` writes `command_joints` into `data.ctrl` via the actuator map (`robot_controller_mujoco_peg_tool_contact.py`, lines 1048-1052).

The exact final executable command variable is:

```text
self.command_joints
```

immediately before and after `_apply_actuator_targets(self.command_joints)`. It is after sign conversion and velocity limiting. It is in internal MuJoCo qpos convention. MuJoCo itself then enforces actuator `ctrlrange` from the XML.

## Current HDF5 Schema

The compact recorder creates these groups and datasets in `MujocoHDF5Recorder._create_file_structure()` (`mujoco_hdf5_recorder.py`, lines 487-547).

| HDF5 path | Shape | Rate | Source function | Source variable / MuJoCo array | Convention | Type |
|---|---:|---:|---|---|---|---|
| `timestamps/state` | `[N_state]` | 30 Hz | `_append_state_sample()` | `data.time` | MuJoCo simulation time | timestamp |
| `timestamps/state_episode` | `[N_state]` | 30 Hz | `_append_state_sample()` | `data.time - episode_start_sim_time` | episode-relative MuJoCo time | timestamp |
| `observations/ee_pose` | `[N_state, 7]` | 30 Hz | `_ee_pose()` | `data.xpos[ee_body_id]`, `data.xquat[ee_body_id]` | world pose, quaternion `qw qx qy qz` | actual measured state |
| `observations/joint_pos` | `[N_state, 7]` | 30 Hz | `_joint_pos()` | `data.qpos[jnt_qposadr]` | internal MuJoCo qpos | actual measured state |
| `observations/joint_vel` | `[N_state, 7]` | 30 Hz | `_joint_vel()` | `data.qvel[jnt_dofadr]` | internal MuJoCo dof velocity | actual measured state |
| `observations/joint_torque` | `[N_state, 7]` | 30 Hz | `_joint_torque()` | `data.qfrc_actuator[jnt_dofadr]` | generalized actuator force at joint dof | actual actuator force/diagnostic |
| `timestamps/force` | `[N_force]` | 500 Hz | `_append_force_sample()` | `data.time` | MuJoCo simulation time | timestamp |
| `timestamps/force_episode` | `[N_force]` | 500 Hz | `_append_force_sample()` | `data.time - episode_start_sim_time` | episode-relative MuJoCo time | timestamp |
| `observations/ft_wrench` | `[N_force, 6]` | 500 Hz | `_ft_wrench()` | raw FT sensor minus compensation if enabled | `[Fx,Fy,Fz,Tx,Ty,Tz]`, sensor frame | force observation |
| `observations/ft_wrench_raw` | `[N_force, 6]` if enabled in latest code | 500 Hz | `_ft_wrench_raw()` | `data.sensordata` for `peg_ft_force` and `peg_ft_torque` | sensor frame | diagnostic |
| `observations/ft_wrench_gravity` | `[N_force, 6]` if enabled in latest code | 500 Hz | `_ft_gravity_wrench()` | predicted gravity wrench in sensor frame | sensor frame | diagnostic |
| `timestamps/image` | `[N_image]` | 30 Hz | `_append_image_sample()` | `data.time` | MuJoCo simulation time | timestamp |
| `timestamps/image_episode` | `[N_image]` | 30 Hz | `_append_image_sample()` | `data.time - episode_start_sim_time` | episode-relative MuJoCo time | timestamp |
| `observations/images/ee_cam` | `[N_image,480,640,3]` | 30 Hz | `_append_image_sample()` | `mujoco.Renderer(...).render()` | uint8 RGB | image observation |
| `observations/images/base_top_cam` | `[N_image,480,640,3]` | 30 Hz | `_append_image_sample()` | `mujoco.Renderer(...).render()` | uint8 RGB | image observation |
| `observations/images/camera_names` | `[N_cam]` | once | `_create_file_structure()` | recorder camera list | UTF-8 strings | metadata |
| `episode_metadata/*` | scalar/vector/string | once at start/end | `_write_initial_metadata()`, `_write_final_metadata()` | state getters, model/config names | mixed | metadata |
| `events/*` | `[N_event]` | event-based | `add_event()`, `_write_events()` | recording service events | mixed | metadata |

The current code writes metadata for `joint_names`, `actuator_names`, camera names, initial/final joint state, initial/final `ee_pose`, initial/final force, and initial/final peg/hole positions (`mujoco_hdf5_recorder.py`, lines 700-787 and 789-832).

## Direct Answers

### A. Does the current HDF5 record actual joint states?

Yes.

- `observations/joint_pos` is actual `data.qpos` read from MuJoCo (`mujoco_hdf5_recorder.py`, lines 882-889).
- `observations/joint_vel` is actual `data.qvel` (`mujoco_hdf5_recorder.py`, lines 891-898).
- `observations/joint_torque` is actual `data.qfrc_actuator` at each joint dof (`mujoco_hdf5_recorder.py`, lines 900-911).

### B. Does the current HDF5 record executable joint commands?

No, not in the fixed-start insertion dataset inspected here.

Across all 100 files in:

```text
/Users/wangshuteng/Desktop/ForceAwareACT/mujoco_data/peg_hole_fixed_insertion
```

there are zero datasets whose path contains:

```text
cmd, ctrl, target, action, command
```

The compact recorder has no command dataset in `_create_file_structure()` and `_append_state_sample()` appends only `ee_pose`, `joint_pos`, `joint_vel`, and `joint_torque` (`mujoco_hdf5_recorder.py`, lines 639-647).

The older CSV-style `MujocoDataRecorder` has helper methods for `ctrl`, `q_target`, and `q_cmd`, but that is not the compact HDF5 schema used by these files.

### C. If commands are not recorded, what should be recorded?

Record the final executable internal command:

```text
self.command_joints
```

from `RobotControllerMuJoCoPegTool`, sampled after `_step_command_toward_target_locked()` and immediately before or after `_apply_actuator_targets(self.command_joints)` in `_physics_step()`.

This variable is:

- after public-to-internal sign conversion,
- after the controller's velocity limiting,
- before MuJoCo actuator dynamics,
- in internal MuJoCo qpos convention,
- the value written to `data.ctrl[actuator_id]`.

If possible, also record:

- `self.target_joints`: internal post-sign target before velocity limiting,
- `data.ctrl[actuator_ids]`: actual MuJoCo control array after assignment,
- public IK command before sign conversion,
- command rate-limit/clipping metadata.

### D. Does `observations/joint_pos` equal actual `data.qpos`? Does it ever equal `data.ctrl` command?

`observations/joint_pos` equals actual `data.qpos`, not `data.ctrl`, by code. It may numerically approach `data.ctrl` when tracking is good, but it is sampled from `data.qpos`.

The current HDF5 files do not contain `data.ctrl`, so direct per-sample comparison is impossible from the dataset alone.

### E. How is `ft_wrench` recorded?

The current latest code:

- reads raw force from `peg_ft_force`,
- reads raw torque from `peg_ft_torque`,
- both sensors are attached to `ft_sensor_site` in `pangu_all_right.xml` (`pangu_all_right.xml`, lines 230-232),
- raw values come from `data.sensordata` via `model.sensor_adr` and `model.sensor_dim` (`mujoco_hdf5_recorder.py`, lines 925-935 and 977-986),
- gravity compensation computes a predicted wrench in the sensor frame and `observations/ft_wrench = raw - gravity` when `ft_compensation_mode == "gravity"` (`mujoco_hdf5_recorder.py`, lines 1040-1116).

The fixed-start insertion files inspected here contain only:

```text
observations/ft_wrench
```

They do not contain:

```text
observations/ft_wrench_raw
observations/ft_wrench_gravity
```

That means the files were generated with an older/effective compact schema than the latest code path that can preserve raw and gravity diagnostic streams.

### F. Are per-state `peg_tip_pos` and `hole_center_pos` recorded?

No. The current compact state stream does not include per-frame task site positions.

The files contain metadata datasets:

```text
episode_metadata/initial_peg_tip_pos
episode_metadata/final_peg_tip_pos
episode_metadata/initial_hole_center_pos
episode_metadata/final_hole_center_pos
```

but in all 100 fixed-start insertion HDF5 files inspected, `initial_peg_tip_pos` is non-finite, and the representative first/middle/last files show both initial and final peg/hole metadata position datasets as `[nan, nan, nan]`.

This is why replay through MuJoCo is required to reconstruct task error.

### G. What is `ee_pose`?

`ee_pose` is the pose of body `peg_tool`, not `peg_tip_site`.

The recorder default is `ee_body_name="peg_tool"` (`mujoco_hdf5_recorder.py`, lines 57-58 and 112-116). `_ee_pose()` reads:

```text
data.xpos[ee_body_id]
data.xquat[ee_body_id]
```

and stores `[x,y,z,qw,qx,qy,qz]` (`mujoco_hdf5_recorder.py`, lines 913-918).

In the XML, `peg_tool` contains:

- `ee_cam`
- `cylindrical_peg`
- `peg_tip_site`
- `ft_sensor_site`

with `peg_tip_site` at local `pos="0 0 -0.090000"` and `ft_sensor_site` at local `pos="0 0 0"` (`pangu_all_right.xml`, lines 73-82). Thus `ee_pose` is the peg-tool body frame / FT sensor origin frame, offset from the peg tip by roughly 9 cm along the local peg axis.

## Representative HDF5 Inspection

Three files were inspected directly: first, middle, and last lexicographic episode.

| Episode | `n_state` | `n_force` | `n_image` | State rate | Force rate | Command-like datasets | Force norm min / max / mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `20260617_125307_teleop_015` | 238 | 3962 | 238 | 30.000 Hz | 500.000 Hz | 0 | 0.202 / 801.741 / 109.174 |
| `20260617_131825_teleop_071` | 409 | 6812 | 409 | 30.000 Hz | 500.037 Hz | 0 | 0.319 / 1269.493 / 371.918 |
| `20260617_133655_teleop_124` | 230 | 3829 | 230 | 30.001 Hz | 500.065 Hz | 0 | 0.061 / 421.788 / 92.909 |

All three contain:

```text
episode_metadata
events
observations
timestamps
```

All three contain:

```text
observations/ee_pose
observations/ft_wrench
observations/images/base_top_cam
observations/images/camera_names
observations/images/ee_cam
observations/joint_pos
observations/joint_torque
observations/joint_vel
timestamps/force
timestamps/force_episode
timestamps/image
timestamps/image_episode
timestamps/state
timestamps/state_episode
```

All 100 episodes have the same dataset path set. None contain command-like datasets. None contain `observations/ft_wrench_raw` or `observations/ft_wrench_gravity`.

The only non-finite datasets in the representative files are the peg-tip/hole metadata datasets:

```text
episode_metadata/initial_peg_tip_pos
episode_metadata/final_peg_tip_pos
episode_metadata/initial_hole_center_pos
episode_metadata/final_hole_center_pos
episode_metadata/initial_task_error_xyz
```

## Implication for Dynamic Replay

The dynamic replay script currently sends recorded actual `joint_pos` back into MuJoCo as actuator position commands. That is not the same signal used during data collection.

During collection:

```text
command_joints -> data.ctrl -> actuator dynamics/contact -> data.qpos
```

The HDF5 stores only the result:

```text
data.qpos -> observations/joint_pos
```

Using `data.qpos` as future commands changes the control problem. It can create lag, contact mismatch, or high force because the controller that originally generated those states used a rate-limited command trajectory that is not recorded.

This supports the interpretation that the current dynamic replay failure is not proof that the demonstrations are invalid. It is evidence that the executable command trajectory is missing.

## Implication for ForceAwareACT

ForceAwareACT currently only supports:

```text
action_mode="joint_pos"
```

and constructs:

```text
action_chunk = joint_pos[i + 1 : i + K + 1]
```

from `observations/joint_pos` (`contact_force_hdf5_dataset.py`, lines 198-233 and 285-287).

Therefore the model is trained to predict future actual state, not future executable actuator command. In closed-loop deployment, those predictions are used as commands. That mismatch is likely important for the 3-4 cm stall and for dynamic replay discrepancies.

## Recommended Recorder Modification

Add command recording to `MujocoHDF5Recorder`.

Recommended primary dataset:

```text
observations/joint_cmd
```

or, if you prefer separating controls:

```text
controls/joint_cmd
```

Recommended contents:

```text
self.command_joints[:7]
```

sampled at the same rate as `observations/joint_pos`, immediately after:

```text
_step_command_toward_target_locked(self.sim_timestep)
```

and immediately before or after:

```text
_apply_actuator_targets(self.command_joints)
```

Recommended metadata:

```text
episode_metadata/joint_cmd_convention = "internal_mujoco_qpos"
episode_metadata/joint_cmd_source = "RobotControllerMuJoCoPegTool.command_joints"
episode_metadata/joint_cmd_stage = "after_sign_conversion_after_velocity_limit_before_mj_step"
episode_metadata/arm_sign = [-1, 1, 1, -1, 1, 1, 1]
episode_metadata/max_joint_velocity = 0.5
```

Recommended diagnostics:

```text
controls/joint_target_internal     # self.target_joints[:7]
controls/joint_cmd_internal        # self.command_joints[:7]
controls/actuator_ctrl             # data.ctrl[actuator_ids]
controls/joint_target_public       # optional pre-sign IK/smoothed target
controls/command_tracking_error    # norm(data.qpos - data.ctrl), optional
```

Also record per-state task sites:

```text
observations/peg_tip_pos
observations/hole_center_pos
observations/peg_to_hole_error
```

The current XML has valid sites, but current dataset metadata has NaNs, so direct per-frame task-state recording would remove the need for offline reconstruction.

## Recommended ForceAwareACT Dataset Changes After Command Recording

Add new action modes:

```text
action_mode="joint_cmd"
```

where:

```text
action_chunk = observations/joint_cmd[i + 1 : i + K + 1]
```

and optionally:

```text
action_mode="delta_joint_cmd"
```

where:

```text
action_chunk = observations/joint_cmd[i + 1 : i + K + 1] - observations/joint_pos[i]
```

For deployment, denormalize predicted commands and write them to MuJoCo `data.ctrl[actuator_ids]` using the same internal qpos convention. Keep `observations/joint_pos` as state input, but do not use it as the supervised command label unless the goal is explicitly state prediction.

## Bottom Line

The current fixed-start insertion HDF5 demonstrations record actual achieved state, force, and images, but they do not record the executable actuator command labels. The right command label to record is `RobotControllerMuJoCoPegTool.command_joints`, in internal MuJoCo qpos convention, after sign conversion and velocity limiting. ForceAwareACT is currently trained on future actual `joint_pos`, so its action head is learning a state trajectory that is only being treated as a command at deployment time.
