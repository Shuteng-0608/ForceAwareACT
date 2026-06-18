# Command Action Rollout

`run_mujoco_policy_rollout.py` supports the same action modes as the HDF5
dataset and training pipeline. The denormalized policy output must be
interpreted according to the training `action_mode`; rollout must not blindly
treat every output as future measured qpos.

## Rollout Semantics

`action_mode="joint_pos"` is the legacy mode. The model output is treated as an
absolute joint-position target and written through the existing safety pipeline
to MuJoCo actuator controls.

`action_mode="action"` means the model predicts the absolute executable actuator
command:

```text
target_ctrl = pred_action
```

`action_mode="joint_pos_command"` has the same rollout semantics as
`action_mode="action"`, but its training labels came from
`/actions/joint_pos_command`.

`action_mode="delta_joint_cmd"` means the model predicts the executable command
delta relative to the current measured qpos:

```text
target_ctrl = current_qpos + pred_action
```

`action_mode="delta_joint_pos_command"` has the same rollout semantics as
`delta_joint_cmd`, but its training labels came from
`/actions/joint_pos_command`.

After `target_ctrl` is formed, the rollout script applies the existing guarded
execution path:

1. optional diagnostic axial push bias,
2. per-joint `--max-delta-q` clipping relative to current qpos,
3. EMA smoothing,
4. actuator ctrlrange clipping,
5. write to `data.ctrl[actuator_ids]`.

## Normalization Compatibility

Command-based rollouts require normalization stats computed with the same
`action_mode`. If the stats file declares a different `action_mode`, rollout
fails. If the stats file lacks `action_mode` metadata, only legacy
`action_mode="joint_pos"` is allowed.

## Initial Smoke Parameters

Recommended first smoke settings:

- `--policy-rate-hz 30`
- `--max-delta-q 0.017` or `--max-delta-q 0.02`
- high `--force-stop-threshold`, for example `1000`, for contact-rich playback
  smoke tests
- no axial push or task-space bias for the first command-action comparison

`--force-stop-threshold 20` is too low for the new contact-rich playback
dataset and can stop the rollout before useful contact behavior is observed.

## Expected First Smoke Tests

Absolute command model:

```bash
.venv/bin/python scripts/run_mujoco_policy_rollout.py \
  --checkpoint outputs/peg_hole_playback_test/smoke_action_all10/checkpoint.pt \
  --normalization-stats outputs/peg_hole_playback_test/normalization_stats_action.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --contact-latent-mode prior \
  --action-mode action \
  --action-select-mode mid \
  --max-rollout-steps 50 \
  --policy-rate-hz 30 \
  --max-delta-q 0.02 \
  --force-stop-threshold 1000 \
  --output-dir outputs/peg_hole_playback_test/rollout_smoke_action_mid \
  --execute-actions
```

Delta command model:

```bash
.venv/bin/python scripts/run_mujoco_policy_rollout.py \
  --checkpoint outputs/peg_hole_playback_test/smoke_delta_joint_cmd_all10/checkpoint.pt \
  --normalization-stats outputs/peg_hole_playback_test/normalization_stats_delta_joint_cmd.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --contact-latent-mode prior \
  --action-mode delta_joint_cmd \
  --action-select-mode mid \
  --max-rollout-steps 50 \
  --policy-rate-hz 30 \
  --max-delta-q 0.02 \
  --force-stop-threshold 1000 \
  --output-dir outputs/peg_hole_playback_test/rollout_smoke_delta_mid \
  --execute-actions
```

For both modes, inspect `rollout_log.csv` columns:

- `selected_action_raw_*`: denormalized model output before action-mode
  interpretation.
- `target_ctrl_*`: absolute actuator target after action-mode interpretation
  and before safety clipping.
- `applied_ctrl_*`: command actually present in `data.ctrl` after guarded
  filtering.
- `target_ctrl_delta_from_qpos_norm`
- `applied_ctrl_delta_from_qpos_norm`
