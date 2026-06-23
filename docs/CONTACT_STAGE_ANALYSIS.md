# Contact Stage Analysis

`scripts/analyze_contact_stage.py` summarizes peg-in-hole contact behavior from
rollout logs and, in a lightweight first pass, from command-action HDF5 demos.

The goal is to separate three failure modes:

- the peg is still hovering in free space,
- the peg has reached the hole entrance plane but is laterally misaligned,
- the policy reaches contact but stops issuing useful insertion/correction
  commands.

## Rollout Log Analysis

For `rollout_log.csv`, the script uses existing rollout columns:

- `peg_to_hole_dist`
- `peg_to_hole_axial_error`
- `peg_to_hole_lateral_error`
- `force_norm`
- `target_ctrl_*`
- `applied_ctrl_*`
- `current_qpos_*`

It computes:

- `entrance_axial_error = peg_to_hole_axial_error - hole_entrance_offset`
- first contact onset step/time for force thresholds such as 5, 10, 20, 50 N
- distance, axial error, and lateral error at force onset
- minimum distance, maximum force, and minimum lateral-error rows
- target/applied command delta norms relative to current qpos
- whether force rises while the peg is near the entrance plane
- whether lateral error decreases after contact onset

The default `--hole-entrance-offset 0.024` matches the observed stall region in
recent command-action rollouts.

Example:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_contact_stage.py \
  outputs/peg_hole_playback_test/rollout_smoke_action_mid/rollout_log.csv \
  --hole-entrance-offset 0.024 \
  --force-thresholds 5 10 20 50 \
  --output-csv outputs/peg_hole_playback_test/contact_stage_summary.csv \
  --plot outputs/peg_hole_playback_test/contact_stage_plots
```

## HDF5 Demo Analysis

The first HDF5 mode is read-only and does not require MuJoCo. It reads:

- `observations/ft_wrench`
- `observations/joint_pos`
- `/action` if available
- state/force timestamps when available

It reports force-contact timing and teacher command behavior:

- nearest-aligned force norm per state sample
- `action_delta_from_qpos = ||action - observations/joint_pos||`
- `action_step_delta = ||action[t+1] - action[t]||`
- command-change means after contact onset

It does not reconstruct peg-tip geometry in this first version. For axial,
lateral, and distance metrics on demos, use the static or dynamic MuJoCo replay
audits, or extend this script with `--model-xml` replay later.

Example:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_contact_stage.py \
  --hdf5 mujoco_data/peg_hole_playback_test/20260617_200306_teleop_002/episode.hdf5 \
  --force-thresholds 5 10 20 50 \
  --output-csv outputs/peg_hole_playback_test/demo_contact_stage_summary.csv
```

## Interpreting Results

If force rises while `entrance_axial_error` is close to zero, the peg is likely
at the hole entrance plane rather than hovering.

If `mean_lateral_after_contact` is not lower than
`mean_lateral_before_contact`, the policy is not improving lateral alignment
after contact onset.

If `mean_target_delta_after_contact` or `mean_applied_delta_after_contact` is
small while lateral error remains nonzero, the rollout may be command-limited or
the policy may not be producing enough post-contact correction.
