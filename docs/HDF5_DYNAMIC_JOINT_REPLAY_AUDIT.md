# HDF5 Dynamic Joint Replay Audit

## Purpose

The static replay audit sets each recorded `joint_pos` frame directly into
MuJoCo and calls `mj_forward`. That verifies the kinematic relationship between
the recorded joints, `peg_tip_site`, and `hole_center_site`.

Dynamic replay tests a stricter question: if the same recorded `joint_pos`
trajectory is sent back through the current MuJoCo position actuators as
commands, does the simulated robot dynamically reproduce insertion?

This separates two possible failure modes:

- If recorded trajectories dynamically insert, then the current XML and
  actuator interface can reproduce the demonstrations, and a closed-loop policy
  stall is more likely a learned policy or imitation issue.
- If recorded trajectories do not dynamically insert, then there may be an XML,
  actuator, timing, or data-collection mismatch between demonstrations and the
  current rollout environment.

## What Is Replayed

The script reads each HDF5 episode in read-only mode and uses:

- `observations/joint_pos` as the initial actual MuJoCo qpos state.
- `--command-field` as the commanded 7-DoF internal MuJoCo actuator target.
- `observations/ft_wrench` only as a recorded force reference.
- `timestamps/state_episode` or `timestamps/state/state_episode` when
  `--use-recorded-state-timestamps` is enabled.
- `timestamps/force_episode` or `timestamps/force/force_episode` to align the
  recorded force norm when available.

The script does not apply the teleoperation public/internal sign convention.
The selected command field is treated as internal MuJoCo joint coordinates.

Supported command-field examples:

- `observations/joint_pos`: state-as-command baseline. This replays actual
  measured qpos as if it were an actuator command.
- `action`: executable command playback for the new ACT-compatible schema,
  where `/action` equals `data.ctrl[actuator_ids]`.
- `actions/joint_pos_command`: semantic copy of the executable command.

When `--playback-max-joint-velocity` is provided, the script rate-limits the
applied control from the previous control toward the selected target:

```text
max_delta = playback_max_joint_velocity * dt_hold
ctrl_next = ctrl_prev + clip(target - ctrl_prev, -max_delta, max_delta)
```

Using `--command-field action --playback-max-joint-velocity 1.2` is the
controller-faithful playback mode for datasets where `action` stores the
executable command label.

## What Is Measured

For each commanded target, MuJoCo is stepped for the selected hold duration and
the script records:

- commanded qpos `qcmd_0 ... qcmd_6`,
- selected command target `replay_target_0 ... replay_target_6`,
- final applied actuator control `applied_ctrl_0 ... applied_ctrl_6`,
- actual simulated qpos `qpos_0 ... qpos_6`,
- qpos tracking error,
- target-to-actual error,
- applied-control-to-actual error,
- target-to-applied-control error,
- `peg_tip_site` and `hole_center_site` world positions,
- total peg-tip-to-hole distance,
- axial and lateral error relative to `--hole-axis-world`,
- online MuJoCo force norm from `peg_ft_force` when available,
- nearest aligned recorded force norm from HDF5 when available.

## Interpreting Results

Successful dynamic replay should show the same qualitative insertion progress
as static replay:

- low final and minimum peg-tip-to-hole distance,
- threshold reaches below 5 cm, 3 cm, 2 cm, 1 cm, and ideally 5 mm,
- bounded qpos tracking error,
- force values that are plausible for the task.

If static replay reaches below 5 mm but dynamic replay stalls around 3-4 cm,
the recorded trajectory may depend on a controller, timing, or XML condition
that is not represented by direct actuator replay. If dynamic replay succeeds
but the learned policy stalls, the more likely bottleneck is closed-loop
imitation, action selection, temporal aggregation, or distribution shift.

## Example

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_hdf5_joint_trajectory_mujoco.py \
  --data-dir mujoco_data/peg_hole_fixed_insertion \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-dir outputs/peg_fixed_insert_100/dynamic_replay_audit \
  --command-field observations/joint_pos \
  --hole-axis-world 0 -1 0 \
  --control-rate-hz 30 \
  --save-frame-csv
```

Executable command playback on the new schema:

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_hdf5_joint_trajectory_mujoco.py \
  --data-dir mujoco_data/peg_hole_playback_test \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-dir outputs/peg_hole_playback_test/dynamic_replay_action \
  --command-field action \
  --control-rate-hz 30 \
  --save-frame-csv
```

Controller-faithful command playback with a velocity limit:

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_hdf5_joint_trajectory_mujoco.py \
  --data-dir mujoco_data/peg_hole_playback_test \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-dir outputs/peg_hole_playback_test/dynamic_replay_action_vlim_1p2 \
  --command-field action \
  --playback-max-joint-velocity 1.2 \
  --control-rate-hz 30 \
  --save-frame-csv
```

For a single episode with videos:

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_hdf5_joint_trajectory_mujoco.py \
  --episode-path mujoco_data/peg_hole_fixed_insertion/<episode_dir>/episode.hdf5 \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-dir outputs/peg_fixed_insert_100/dynamic_replay_single \
  --command-field observations/joint_pos \
  --use-recorded-state-timestamps \
  --render-videos \
  --video-every 1
```

Primary outputs:

```text
replay_episode_summary.csv
replay_dataset_summary.json
replay_frame_metrics.csv
videos/<episode_name>_ee_cam.mp4
videos/<episode_name>_base_top_cam.mp4
```
