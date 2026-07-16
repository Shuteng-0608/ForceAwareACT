# Action Semantics: Dataset, Training, and Rollout

This is the canonical action-label contract for the repository. It consolidates
the former dataset-mode, command-training, and command-rollout notes so that an
action mode is defined in one place from HDF5 label through deployed MuJoCo
control.

`ContactForceHDF5Dataset` supports both the legacy state-as-action target and
new executable command labels for MuJoCo peg-in-hole demonstrations.

## Key Distinction

`observations/joint_pos` is the measured MuJoCo joint state, `data.qpos`. It is
not necessarily the command written to the actuator controls.

The new command-labeled HDF5 schema records:

- `/action`: ACT-compatible executable actuator position command, equal to
  `data.ctrl[actuator_ids]`.
- `/actions/joint_pos_command`: semantic copy of `/action`.
- `/observations/joint_pos`: actual measured MuJoCo qpos state.

Dynamic replay showed that state-as-command playback can fail even when static
forward replay reaches insertion. Command labels should therefore be preferred
for training deployable policies.

## Supported Modes

- `action_mode="joint_pos"`: legacy baseline. The target is
  `observations/joint_pos[i + 1 : i + K + 1]`.
- `action_mode="action"`: absolute executable command prediction. The target is
  `/action[i : i + K]`.
- `action_mode="joint_pos_command"`: absolute executable command prediction
  using `/actions/joint_pos_command[i : i + K]`.
- `action_mode="delta_joint_cmd"`: command delta prediction. The target is
  `/action[i : i + K] - observations/joint_pos[i]`.
- `action_mode="delta_joint_pos_command"`: command delta prediction using
  `/actions/joint_pos_command[i : i + K] - observations/joint_pos[i]`.

All modes return `action_chunk` with shape `[chunk_len, 7]` and dtype
`torch.float32`.

## Selection Guidance

First smoke test:

```bash
action_mode="action"
```

Delta-mode candidate:

```bash
action_mode="delta_joint_cmd"
```

The absolute command mode is easier to inspect and is the current canonical
five-configuration comparison mode. Delta command may be better conditioned
because it represents future executable command relative to current qpos, but
that is an experimental hypothesis rather than a repository default. Compute
separate stats and keep the rollout `action_mode` identical for either choice.

## Alignment Contract

| `action_mode` | HDF5 source | Chunk at state index `i` | Rollout interpretation |
| --- | --- | --- | --- |
| `joint_pos` | `observations/joint_pos` | `[i + 1 : i + K + 1]` | absolute joint-position target |
| `action` | root `action` | `[i : i + K]` | absolute actuator command |
| `joint_pos_command` | `actions/joint_pos_command` | `[i : i + K]` | absolute actuator command |
| `delta_joint_cmd` | root `action` and current qpos | `action[i : i + K] - qpos[i]` | add predicted delta to current qpos |
| `delta_joint_pos_command` | `actions/joint_pos_command` and current qpos | `command[i : i + K] - qpos[i]` | add predicted delta to current qpos |

Command modes start at the current decision index because the stored command is
the control applied at that state time. The legacy `joint_pos` mode starts one
step in the future because it treats the next measured state as the first
target. Samples that cannot provide a complete chunk are excluded rather than
padded.

For the current three main MuJoCo datasets, root `action` and
`actions/joint_pos_command` are value-identical. Both names remain supported so
ACT-compatible and semantically named recordings can use the same loader.

## Force Signal Is Independent of Action Mode

`observations/ft_wrench` remains the default force signal for force-aware
policies. Changing action mode does not change force-window construction or
future-force labels.

The repository does not infer or validate the wrench frame, sign convention,
bias, filtering, or gravity compensation. Those properties belong to the
recorder/data contract and must be audited separately for each dataset.

## Normalization and Training

Compute a separate normalization file for each action mode. The action
distribution for absolute state, absolute command, and delta command can differ
substantially. Training and evaluation reject stats whose recorded
`action_mode` conflicts with the requested mode.

Absolute-command example:

```bash
PYTHONPATH=src python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --action-mode action \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output outputs/peg_hole_100/normalization_stats_action_train80.pt

PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --val-episode-list configs/splits/peg_hole_100_val10.txt \
  --policy-variant force_aware_contact_cvae \
  --action-mode action \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --max-steps 20 \
  --batch-size 2 \
  --output-dir outputs/peg_hole_100/smoke_action \
  --log-csv outputs/peg_hole_100/smoke_action/train_log.csv
```

To test delta commands, change all three of the mode, stats filename, and output
directory together:

```text
--action-mode delta_joint_cmd
normalization_stats_delta_joint_cmd_train80.pt
smoke_delta_joint_cmd/
```

## Rollout Interpretation

`run_mujoco_policy_rollout.py` denormalizes the selected policy action and then
interprets it according to the checkpoint/training mode:

```text
joint_pos/action/joint_pos_command:
    target_ctrl = predicted_action

delta_joint_cmd/delta_joint_pos_command:
    target_ctrl = current_qpos + predicted_action
```

After forming the absolute `target_ctrl`, rollout applies the shared guarded
execution path:

1. optional diagnostic axial-push bias;
2. per-joint `--max-delta-q` clipping relative to current qpos;
3. optional EMA smoothing;
4. actuator `ctrlrange` clipping;
5. write to `data.ctrl[actuator_ids]`.

The rollout log separates each stage:

- `selected_action_raw_*`: denormalized model output before mode interpretation;
- `target_ctrl_*`: absolute target after mode interpretation and before safety filtering;
- `applied_ctrl_*`: command present in `data.ctrl` after guarded filtering;
- `target_ctrl_delta_from_qpos_norm` and `applied_ctrl_delta_from_qpos_norm`:
  target/applied command distance from measured qpos.

## Compatibility Rules

- Dataset inspection, normalization, training, offline evaluation, and rollout
  must use one action mode.
- A stats file declaring a different `action_mode` is rejected.
- A legacy stats file with no `action_mode` metadata is accepted only for
  `joint_pos` compatibility.
- A checkpoint's recorded action mode should be treated as authoritative; do
  not override it merely to make a rollout command start.
- `max_delta_q` is a rollout safety/execution parameter, not the definition of
  delta action labels.

## Smoke-Test Order

1. Inspect `action`, `actions/joint_pos_command`, and
   `observations/joint_pos` in representative HDF5 episodes.
2. Compute mode-specific stats and run a short training smoke.
3. Run deployable offline inference using the same stats and action mode.
4. Run a short closed-loop rollout with action execution enabled.
5. Inspect raw action, target control, applied control, force, and stop reason
   before increasing duration or starting a grid/suite.

Historical playback tests started near `--max-delta-q 0.017` to `0.02` at
30 Hz and sometimes used a high force-stop threshold to separate contact from
other failures. Those values are experiment history, not universal safety
defaults. Derive and freeze thresholds for the selected dataset and XML.

Current end-to-end rollout commands and safety semantics are maintained in
[`ROLLOUT_EXPERIMENT_MANUAL.md`](ROLLOUT_EXPERIMENT_MANUAL.md); compact commands
are in [`COMMAND_RECIPES.md`](COMMAND_RECIPES.md).
