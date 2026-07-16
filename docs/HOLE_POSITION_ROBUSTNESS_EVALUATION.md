# Hole Position Robustness Evaluation

> Current workflow note (2026-07-16): the runner now supports exact point CSV
> input and independent point-set/rollout seeds. This document preserves the
> original grid/LHS protocol examples; use
> [`ROLLOUT_EXPERIMENT_MANUAL.md`](ROLLOUT_EXPERIMENT_MANUAL.md) for the current
> fixed-point and multi-seed workflow.

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

In `pangu_all_right.xml`, `hole_goal_site` is owned directly by body
`wall_task`. The physical hole collision geoms are also directly on `wall_task`:

- `wall_hole_ring_00` through `wall_hole_ring_23`
- `hole_back_stop`

`wall_task` is therefore the exact common body whose translation moves the
target site and the complete physical hole collision assembly together. It also
contains the rectangular wall fixture geoms `wall_top`, `wall_bottom`,
`wall_left`, and `wall_right`; the XML does not define a smaller hole-only child
body. Robot bodies, cameras, lights, and room/background bodies are outside this
body and are not moved by the runtime offset.

The project now uses explicit body validation rather than heuristic ancestor
guessing: the rollout defaults to `--hole-body-name wall_task`, verifies that
`hole_goal_site` is inside that body subtree, and verifies that all expected hole
geoms above are inside the same subtree. A different XML can still override
`--hole-body-name`, but an inconsistent body fails clearly.

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
  --hole-body-name wall_task \
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
  --hole-site-name hole_goal_site \
  --hole-body-name wall_task \
  --x-offsets=-0.002,0,0.002 \
  --z-offsets=-0.002,0,0.002 \
  --dry-run \
  --no-plot-results
```

## Geometry-Only Inspection

Use this command on macOS or Linux to validate the XML structure and runtime
offset without loading a policy checkpoint:

```bash
PYTHONPATH=src python scripts/inspect_hole_assembly.py \
  --model-xml /Users/wangshuteng/Desktop/arm_teleop/model/pangu_all_right.xml \
  --hole-site-name hole_goal_site \
  --hole-body-name wall_task \
  --test-offset-x 0.002 \
  --test-offset-y 0 \
  --test-offset-z 0.002 \
  --offset-frame world
```

## Heatmaps

```bash
PYTHONPATH=src python scripts/plot_hole_grid_results.py \
  --summary-csv outputs/peg_hole_100/hole_grid_xz_2mm_mid_dq002/grid_summary.csv \
  --output-dir outputs/peg_hole_100/hole_grid_xz_2mm_mid_dq002/plots \
  --formats png,pdf \
  --annotate
```

## 50-Point Latin-Hypercube Evaluation

After the deterministic 3x3 grid, use 50 paired x-z task points to estimate
spatial robustness under a uniform position distribution over the same small
offset box. Latin hypercube sampling gives more even one-dimensional coverage
than naive random sampling while remaining deterministic from `--base-seed`.

Recommended first 50-point Linux run, without videos:

```bash
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py \
  --sampling-mode latin_hypercube \
  --num-points 50 \
  --x-min -0.002 \
  --x-max 0.002 \
  --z-min -0.002 \
  --z-max 0.002 \
  --base-seed 20260702 \
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
  --hole-body-name wall_task \
  --hole-offset-frame world \
  --y-offset 0 \
  --success-distance-threshold 0.005 \
  --success-lateral-threshold 0.006 \
  --success-force-threshold 80 \
  --success-hold-steps 15 \
  --output-root outputs/peg_hole_100/hole_lhs_50_xz_2mm_mid_dq002 \
  --continue-on-error
```

Dry-run first:

```bash
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py \
  --sampling-mode latin_hypercube \
  --num-points 50 \
  --x-min -0.002 \
  --x-max 0.002 \
  --z-min -0.002 \
  --z-max 0.002 \
  --base-seed 20260702 \
  --checkpoint outputs/peg_hole_100/action_trainzero_all100_20k_bs16/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_all100.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-root outputs/peg_hole_100/hole_lhs_50_xz_2mm_mid_dq002 \
  --dry-run \
  --no-plot-results
```

Resume a partially completed 50-point run with `--skip-existing`. The generated
`task_points.csv` and `grid_manifest.json` record the exact sampled offsets, and
`grid_summary.csv` records success, safe success, final errors, force metrics,
quadrant, and radial offset for completed task runs. `random_position_summary.json`
contains the Wilson 95% confidence interval for the success rate.

Plot irregular sampled results:

```bash
PYTHONPATH=src python scripts/plot_hole_grid_results.py \
  --summary-csv outputs/peg_hole_100/hole_lhs_50_xz_2mm_mid_dq002/grid_summary.csv \
  --output-dir outputs/peg_hole_100/hole_lhs_50_xz_2mm_mid_dq002/plots \
  --formats png,pdf
```

One rollout per point estimates spatial robustness for this deterministic
evaluation protocol. It does not measure repeated-trial stochastic reliability
at a fixed point; use repeated samples per point later if that question matters.

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
