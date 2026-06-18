# Command Action Training Pipeline

ForceAwareACT originally used future `observations/joint_pos` as the action
target. That field is the measured MuJoCo state, `data.qpos`, not necessarily
the executable actuator command. Dynamic replay showed that treating measured
state as actuator control can fail even when the recorded state trajectory
statically reaches insertion.

The command-labeled MuJoCo HDF5 schema adds executable action labels:

- `/action`: ACT-compatible actuator position command, equal to
  `data.ctrl[actuator_ids]`.
- `/actions/joint_pos_command`: semantic copy of `/action`.
- `/observations/joint_pos`: actual measured qpos state.

`observations/ft_wrench` remains the default force signal and contains the
gravity-compensated wrench.

## Action Modes

`action_mode="action"` trains the policy to predict absolute executable joint
commands from `/action[i : i + K]`.

`action_mode="delta_joint_cmd"` trains the policy to predict executable command
deltas relative to the current measured qpos:

```text
action_chunk[j] = action[state_index + j] - observations/joint_pos[state_index]
```

The legacy `action_mode="joint_pos"` is still available for comparison, but it
is a state-as-action baseline rather than an executable command-label mode.

## Normalization

Action normalization must be recomputed for each action mode. The action
distribution for `joint_pos`, `action`, and `delta_joint_cmd` can differ
substantially. New normalization stats files record `action_mode` metadata, and
training/evaluation will reject a stats file whose `action_mode` does not match
the requested CLI mode.

## Recommended Smoke Order

1. Train a short smoke model with `action_mode="action"`.
2. Train a short smoke model with `action_mode="delta_joint_cmd"`.
3. Compare offline zero/prior/posterior metrics using stats computed for the
   same mode.
4. Update rollout later to interpret delta-command outputs before closed-loop
   deployment.

For later rollout experiments, start with `max_delta_q` around `0.017` to
`0.02` at 30 Hz based on the command distribution. A force stop threshold of
`20` is likely too low for the contact-rich playback dataset and should be
revisited before judging insertion behavior.

## Example Commands

Compute absolute-command stats:

```bash
PYTHONPATH=src .venv/bin/python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/your_playback_train.txt \
  --action-mode action \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output outputs/playback/normalization_stats_action.pt
```

Run a short absolute-command training smoke:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py \
  --episode-list configs/splits/your_playback_train.txt \
  --action-mode action \
  --normalization-stats outputs/playback/normalization_stats_action.pt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --max-steps 20 \
  --batch-size 2 \
  --output-dir outputs/playback/smoke_action \
  --log-csv outputs/playback/smoke_action/train_log.csv
```

Compute delta-command stats:

```bash
PYTHONPATH=src .venv/bin/python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/your_playback_train.txt \
  --action-mode delta_joint_cmd \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output outputs/playback/normalization_stats_delta_joint_cmd.pt
```
