# arm_teleop HDF5 Recording Field Audit

## Executive Conclusion

The active `arm_teleop` compact HDF5 recorder defines `ee_pose` as the world
pose of the MuJoCo body named `peg_tool`:

```python
p = data.xpos[peg_tool_body_id]
q = data.xquat[peg_tool_body_id]
```

It is **not** the `peg_tip_site` pose and is **not** the `link_7` flange pose.
The `ft_sensor_site` is located at local position `[0, 0, 0]` inside
`peg_tool`, so its world position coincides with the recorded `ee_pose`
translation. The peg tip is at local position `[0, 0, -0.09]` in the same
body, giving a fixed 0.09 m tool-frame offset from `ee_pose`/`ft_sensor_site`
to `peg_tip_site`.

This distinction is important for task metrics and any future pose-conditioned
policy. However, it does **not directly explain the current ForceAwareACT
rollout stopping early**, because the current policy does not consume
`ee_pose`. It consumes images, current joint position, and wrench history, and
predicts future joint positions and wrench. The dataset loader returns
`ee_pose`, but `train_minimal.py` does not pass it to `ForceAwareACTPolicy`.

The active compact recorder also does not save per-state
`observations/peg_tip_pos` or `observations/hole_center_pos`. It saves those
positions only in initial/final episode metadata. The older recorder saved
both as per-state observation datasets.

## Audit Scope

- External repository: `../arm_teleop`
- Branch inspected: `dev_peg_in_hole`
- Active model: `../arm_teleop/model/pangu_all_right.xml`
- Active recorder: `../arm_teleop/vptele/utils/mujoco_hdf5_recorder.py`
- Representative episode inspected read-only:
  `mujoco_data/peg_in_hole_hdf5_100/20260610_151602_teleop_008/episode.hdf5`

The external XML had local visibility-only changes that make
`peg_tip_site` and `hole_center_site` active and invisible. Their positions
and the peg geometry were not changed by that diff.

## Recorder Runtime Path

`vptele/main_mujoco.py` loads `config/config_arm_right_peg.yaml`, which selects
`pangu_all_right.xml`. `TeleopSystemMujoco` constructs
`RobotControllerMuJoCoPegTool` in
`vptele/core/teleop_system_mujoco.py:225-238`.

The controller imports and constructs the active `MujocoHDF5Recorder` in
`vptele/arm_control/robot_controller_mujoco_peg_tool_contact.py:36` and
`:214-238`. Recording is started and stopped through the ROS service callback
at `:388-477`. Each physics step calls `hdf5_recorder.record_if_needed()` at
`:730-742`.

The recorder schedules independent streams from MuJoCo `data.time` in
`vptele/utils/mujoco_hdf5_recorder.py:221-239`:

- force: 500 Hz
- state: 30 Hz
- images: 30 Hz

These rates and camera settings are configured in
`vptele/config/config_arm_right_peg.yaml:9-29`.

## Exact HDF5 Writers

The active recorder creates datasets in
`vptele/utils/mujoco_hdf5_recorder.py:247-342` and appends samples in
`:347-379`.

| HDF5 field | Source in code | MuJoCo object/source | Coordinate frame | Notes |
|---|---|---|---|---|
| `observations/joint_pos` | `_joint_pos()`, lines 531-538 | `data.qpos[model.jnt_qposadr[joint_id]]` for `joint_1` through `joint_7` | Internal MuJoCo joint convention | State stream, 30 Hz |
| `observations/joint_vel` | `_joint_vel()`, lines 540-547 | `data.qvel[model.jnt_dofadr[joint_id]]` | Internal MuJoCo generalized velocity convention | State stream, 30 Hz |
| `observations/joint_torque` | `_joint_torque()`, lines 549-560 | `data.qfrc_actuator[model.jnt_dofadr[joint_id]]` | Generalized joint force/torque | Actuator contribution, not total joint load |
| `observations/ee_pose` | `_ee_pose()`, lines 562-567 | `data.xpos[peg_tool_body_id]`, `data.xquat[peg_tool_body_id]` | World position and world quaternion, MuJoCo `w x y z` order | `ee_body_name` defaults to `peg_tool` at lines 47 and 80 |
| `observations/ft_wrench` | `_ft_wrench()`, lines 574-589 | `peg_ft_force` and `peg_ft_torque` via `sensor_adr`/`sensor_dim` into `data.sensordata` | `ft_sensor_site` frame | `[Fx,Fy,Fz,Tx,Ty,Tz]`, force stream at 500 Hz |
| `observations/images/ee_cam` | `_append_image_sample()`, lines 364-379 | MuJoCo renderer using `ee_cam` | RGB camera frame | HDF5 `uint8`, `[N,480,640,3]`, 30 Hz |
| `observations/images/base_top_cam` | `_append_image_sample()`, lines 364-379 | MuJoCo renderer using `base_top_cam` | RGB camera frame | HDF5 `uint8`, `[N,480,640,3]`, 30 Hz |
| `episode_metadata/initial_peg_tip_pos`, `final_peg_tip_pos` | `_site_pos()`, lines 569-572; metadata writes at 432/441 and 465 | `data.site_xpos[peg_tip_site_id]` | World position | Active compact schema: metadata only |
| `episode_metadata/initial_hole_center_pos`, `final_hole_center_pos` | `_site_pos()`, lines 569-572; metadata writes at 433/442 and 466 | `data.site_xpos[hole_center_site_id]` | World position | Active compact schema: metadata only |

The recorder stores the resolved semantic names as HDF5 metadata attributes:
`ee_body_name`, force/torque sensor names, and peg/hole site names at
`mujoco_hdf5_recorder.py:414-418`.

### Representative Episode Verification

The inspected compact episode declares:

- `schema_version = compact_mujoco_hdf5_v1`
- `ee_body_name = peg_tool`
- `ft_force_sensor_name = peg_ft_force`
- `ft_torque_sensor_name = peg_ft_torque`
- `peg_tip_site_name = peg_tip_site`
- `hole_center_site_name = hole_center_site`

Its observations contain `ee_pose`, joint state, wrench, and both camera
tensors. It does not contain per-state `observations/peg_tip_pos` or
`observations/hole_center_pos`.

### Older Recorder Difference

`vptele/utils/mujoco_hdf5_recorder_old.py` explicitly collected
`peg_tool` body pose plus `peg_tip_site` and `hole_center_site` positions in
`_state_row()` at lines 213-225. It wrote:

- `observations/ee_pose` from `peg_tool_*` fields at line 385
- `observations/peg_tip_pos` from site position fields at line 387
- `observations/hole_center_pos` from site position fields at line 388

This explains why analysis and trimming utilities recognize per-state
peg-tip/hole-center fields even though current compact episodes omit them.

## MuJoCo Object Geometry

Relevant definitions in `model/pangu_all_right.xml`:

| Object | Parent | Local pose/geometry | Meaning |
|---|---|---|---|
| `peg_tool` body | `link_7` | `pos="0 -0.0125 0.0025"`, `quat="0.707107 -0.707107 0 0"` | Recorded `ee_pose` body |
| `ft_sensor_site` | `peg_tool` | `pos="0 0 0"` | Coincident with `peg_tool` body origin |
| `peg_tip_site` | `peg_tool` | `pos="0 0 -0.090000"` | Peg tip task point |
| `cylindrical_peg` geom | `peg_tool` | center `pos="0 0 -0.045"`, cylinder half-length `0.045` | Extends from local `z=0` to `z=-0.09` |
| `ee_cam` | `peg_tool` | `pos="0.06 0 0.018"` | Wrist/tool-mounted camera |
| `base_top_cam` | world | `pos="0 -0.1 1.5"` | Fixed external camera |
| `hole_center_site` | `wall_task` | `pos="0 0 0"` | Hole center task point |

The force and torque sensors are attached to `ft_sensor_site` at XML lines
140-141. The controller documents the wrench as expressed in the
`ft_sensor_site` frame at
`robot_controller_mujoco_peg_tool_contact.py:777-786`.

## Initial-Pose MuJoCo Probe

The model was loaded and set to the ForceAwareACT rollout internal initial
joint pose:

```text
[0.046, -0.2, 0.0, -1.6, -1.32, 0.005, 0.005]
```

World positions after `mj_forward`:

| Object | World position `[x, y, z]` m |
|---|---|
| `ft_sensor_site` | `[-0.283008739, -0.273851518, 1.021132905]` |
| `peg_tool` body | `[-0.283008739, -0.273851518, 1.021132905]` |
| `peg_tip_site` | `[-0.282913175, -0.363828431, 1.019096745]` |
| `hole_center_site` | `[-0.25, -0.5, 1.0]` |

Distances:

| Pair | Distance |
|---|---:|
| `peg_tip_site` to `ft_sensor_site` | `0.090000 m` |
| `peg_tip_site` to `peg_tool` body origin | `0.090000 m` |
| `peg_tip_site` to `hole_center_site` | `0.141388 m` |
| `ft_sensor_site` to `hole_center_site` | `0.229520 m` |

The world-space offset direction changes with tool orientation, but its norm
is fixed at 0.09 m.

## Direct Answers About `ee_pose`

- **Is `ee_pose` the flange pose?** No. It is not `link_7`; it is the child
  `peg_tool` body origin. It may be interpreted as a tool-base pose near the
  flange, but it is a distinct body with its own local transform.
- **Is `ee_pose` the `ft_sensor_site` pose?** Its translation is exactly
  coincident with `ft_sensor_site`. Its orientation is the `peg_tool` body
  orientation; the site has no additional local rotation, so the frames also
  coincide.
- **Is `ee_pose` the `peg_tool` body pose?** Yes, exactly.
- **Is `ee_pose` the `peg_tip_site` pose?** No. The peg tip is offset by
  `[0, 0, -0.09]` in the `peg_tool` frame.

## Mapping to Current ForceAwareACT Training

`ContactForceHDF5Dataset` reads and returns `ee_pose` at
`src/force_aware_act/data/contact_force_hdf5_dataset.py:267` and `:305`.
However:

- Action targets are future `joint_pos` values at lines 285-287.
- `train_minimal.py:214-222` passes only images, qpos, force window, action
  chunk, and future force chunk to the model.
- `ForceAwareACTPolicy.forward()` accepts no `ee_pose` argument and its online
  inputs are images, qpos, and force window.
- Normalization statistics do not include `ee_pose`.

Therefore, current ForceAwareACT training uses neither flange/tool pose nor
peg-tip pose as a direct policy input or target. It uses internal joint
positions as state/action and observes the task indirectly through cameras and
wrench.

## Could This Explain Stopping Before Insertion?

The 9 cm semantic offset is real, but it is unlikely to be the direct cause of
the current rollout stopping 7-8 cm from the hole because `ee_pose` is not
used by the policy or rollout controller.

It can still cause confusion or errors in:

- analyses that label `ee_pose` as the task-space end point;
- success metrics computed from `ee_pose` instead of `peg_tip_site`;
- future model variants that add `ee_pose` without encoding the tool offset;
- comparisons between tool-base-to-hole distance and peg-tip-to-hole distance.

The current rollout's `peg_to_hole_dist` correctly uses `peg_tip_site` and
`hole_center_site`. Likely causes of early stopping should therefore still
include action-chunk execution strategy, closed-loop distribution shift,
camera/task-state observability, force response, and safety filtering.

## Recommended Fixes

1. Add per-state `observations/peg_tip_pose` to future HDF5 episodes. Record
   world position from `site_xpos` and orientation from `site_xmat`, converted
   to a documented quaternion convention.
2. Keep the existing tool-base pose under an explicit name such as
   `observations/ee_pose_tool_base` or `observations/ee_pose_flange`, while
   preserving `ee_pose` compatibility during migration.
3. Add per-state `observations/peg_tip_pos` and
   `observations/hole_center_pos` to the compact recorder, not only
   initial/final metadata.
4. Continue using `peg_tip_site` to `hole_center_site` error for task metrics
   and rollout success/failure decisions.
5. If adding task-space state to ForceAwareACT, prefer peg-tip pose and
   peg-tip-to-hole error over the ambiguous `ee_pose` name.
6. Retrain or fine-tune only if the model architecture is changed to consume
   the new peg-tip/task-space features. Existing joint-position action targets
   remain semantically valid.

## Remaining Questions

- The compact recorder does not store the per-state randomized hole position,
  although the controller can randomize `wall_task` before recording. This
  limits offline reconstruction of peg-tip-to-hole error from compact files.
- The task distinguishes hole center from a visualized hole entrance offset.
  Success criteria should explicitly define whether insertion progress is
  measured to the center, entrance, or a depth threshold.
- Existing episodes should be audited for recorder schema version before
  assuming per-state peg-tip/hole-center fields are available.
