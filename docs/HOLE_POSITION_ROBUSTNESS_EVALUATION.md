# Hole Position Robustness Evaluation

## Purpose

Evaluate small hole-position perturbations without changing hole orientation or
the robot initial pose. The rollout script applies a runtime MuJoCo model-body
translation so the physical hole assembly and `hole_goal_site` move together.

## Why Perturb X and Z First

The current hole axis is `[0, -1, 0]`, so the world x-z plane is the lateral
alignment plane for peg insertion. The first robustness experiment keeps y and
orientation fixed and tests small lateral offsets only.

## Important Warning

The complete physical hole body must move. Moving only `hole_goal_site`
invalidates both the collision geometry and success metrics, because the policy
would be evaluated against a target that no longer matches the actual hole.

## Recommended First Grid

Use a deterministic 3x3 grid:

- x offsets: -2 mm, 0 mm, +2 mm
- z offsets: -2 mm, 0 mm, +2 mm
- y offset: 0 mm

## Recommended Linux Command

```bash
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py \
  --checkpoint outputs/peg_hole_100/action_trainzero_all100_20k_bs16/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_all100.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --contact-latent-mode zero \
  --action-mode action \
  --action-select-mode mid \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --policy-rate-hz 30 \
  --max-rollout-steps 900 \
  --max-delta-q 0.02 \
  --force-stop-threshold 1000 \
  --hole-axis-world 0 -1 0 \
  --hole-site-name hole_goal_site \
  --x-offsets=-0.002,0,0.002 \
  --y-offset 0 \
  --z-offsets=-0.002,0,0.002 \
  --success-distance-threshold 0.005 \
  --success-lateral-threshold 0.006 \
  --success-force-threshold 80 \
  --success-hold-steps 15 \
  --output-root outputs/peg_hole_100/hole_grid_xz_2mm_mid_dq002 \
  --save-videos
```

Use the equals-sign form for negative comma-separated arguments:

```bash
--x-offsets=-0.002,0,0.002
--z-offsets=-0.002,0,0.002
```

## Resume Command

Add `--skip-existing` to resume a partially completed grid. A run is skipped
only when its `summary.json` exists and is readable.

## Dry Run

Run this before the actual execution to verify all child commands and output
directories:

```bash
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py \
  --checkpoint outputs/peg_hole_100/action_trainzero_all100_20k_bs16/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_all100.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-root outputs/peg_hole_100/hole_grid_xz_2mm_mid_dq002 \
  --x-offsets=-0.002,0,0.002 \
  --z-offsets=-0.002,0,0.002 \
  --dry-run \
  --no-plot-results
```

## Heatmaps

```bash
PYTHONPATH=src python scripts/plot_hole_grid_results.py \
  --summary-csv outputs/peg_hole_100/hole_grid_xz_2mm_mid_dq002/grid_summary.csv \
  --output-dir outputs/peg_hole_100/hole_grid_xz_2mm_mid_dq002/plots \
  --formats png,pdf \
  --annotate
```

## Interpretation

- Start with +/-2 mm.
- If most cells succeed, expand to +/-4 mm.
- Keep y and orientation fixed during the first evaluation.
- Compare success rate, success time, lateral error, and contact force.
- Do not change success thresholds between cells.

Suggested second grid only after the 3x3 test succeeds:

```text
x,z in {-0.004, -0.002, 0, 0.002, 0.004} m
```
