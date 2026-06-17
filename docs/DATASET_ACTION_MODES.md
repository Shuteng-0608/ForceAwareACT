# Dataset Action Modes

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

## Recommended Next Training Modes

First smoke test:

```bash
action_mode="action"
```

Main candidate:

```bash
action_mode="delta_joint_cmd"
```

The absolute command mode is easier to inspect. The delta command mode may be
better conditioned for closed-loop policy learning because it represents the
future executable command relative to the current measured qpos.
