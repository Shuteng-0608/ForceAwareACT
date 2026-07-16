# HDF5 Replay Task-Error Audit

## Why Replay Is Needed

The compact MuJoCo HDF5 demonstrations do not store per-state
`peg_tip_pos` or `hole_center_pos`. They store joint state, force/wrench,
images, timestamps, and initial/final metadata. To determine whether a
demonstration actually moves through the 3-4 cm pre-contact region, we need
to reconstruct task-space state by replaying the recorded internal MuJoCo
joint positions in the same XML and reading `peg_tip_site` and
`hole_center_site`.

The replay audit is read-only:

- it opens HDF5 files in read mode,
- sets MuJoCo `qpos` in memory,
- calls `mj_forward`,
- reads site positions,
- never steps simulation,
- never modifies HDF5 files or training code.

## Reconstructed Fields

For each recorded state frame, the audit reconstructs:

- `peg_tip_site` world position
- `hole_center_site` world position
- peg-tip-to-hole vector
- total peg-to-hole distance
- signed axial error along `--hole-axis-world`
- lateral error orthogonal to `--hole-axis-world`
- joint motion magnitude to the next/previous frame
- nearest-aligned translational force norm when `ft_wrench` is available

The default hole axis is:

```text
0 -1 0
```

This should match the wall insertion axis used by the MuJoCo task.

## Interpreting Final and Minimum Distance

`final_peg_to_hole_dist` answers where the demonstration ended.

`min_peg_to_hole_dist` answers whether the demonstration ever got closer than
the final frame. If minimum distance is much smaller than final distance, the
episode likely inserted or contacted and then retreated. If final and minimum
distance are both around 3-4 cm, the demonstration itself may not contain the
insertion behavior needed by closed-loop rollout.

Threshold fractions in `dataset_summary.json` show how often demonstrations
reach:

- `< 5 cm`
- `< 3 cm`
- `< 2 cm`
- `< 1 cm`
- `< 5 mm`

The distance-band motion metrics are especially useful for diagnosing stalls:

- mean joint delta in the `3-5 cm` band
- mean joint delta in the `2-3 cm` band
- mean joint delta in the `1-2 cm` band

If demonstrations keep moving in these bands but the policy stalls, the issue
is likely closed-loop execution or model behavior. If demonstrations also stop
there, the data may not contain enough insertion progress.

## Comparing Against Policy Stalls

For a policy that stalls at 3-4 cm, inspect:

- whether demonstrations spend time in the `0.03-0.05 m` band,
- whether `qpos_delta_next` remains nonzero in that band,
- whether trajectories cross below `0.03 m`, `0.02 m`, and `0.01 m`,
- whether the remaining error is axial or lateral.

If lateral error is already low but axial error remains positive, the
demonstration indicates insertion should continue along the hole axis. If
lateral error remains high, the policy may be aligned poorly even if total
distance looks small.

## XML Matching Requirement

Replay must use the same MuJoCo XML geometry and site definitions used during
data collection. A different XML can change:

- `peg_tip_site` position,
- `hole_center_site` position,
- peg length/radius,
- wall/task body pose,
- joint kinematics.

Using a mismatched XML can produce incorrect reconstructed task errors even
when the HDF5 joint positions are correct.

## Example

```bash
PYTHONPATH=src python scripts/audit_hdf5_replay_task_error.py \
  --data-dir mujoco_data/peg_hole_fixed_insertion \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-dir outputs/peg_fixed_insert_100/replay_audit \
  --hole-axis-world 0 -1 0 \
  --save-frame-csv \
  --plot
```

Primary outputs:

```text
episode_summary.csv
dataset_summary.json
frame_metrics.csv
final_distance_hist.png
min_distance_hist.png
distance_trajectories.png
axial_lateral_scatter.png
band_time_bar.png
```
