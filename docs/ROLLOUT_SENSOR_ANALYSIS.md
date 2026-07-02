# Rollout Sensor Analysis

## Purpose

Use `scripts/plot_rollout_sensor_analysis.py` to analyze whether contact and
force events are temporally coupled with policy corrections during closed-loop
MuJoCo rollouts. The plots are meant to show when contact begins, whether force
rises before or during corrective behavior, whether lateral error improves after
contact, and whether predicted force traces move with retry behavior.

## Single-Run Example

```bash
PYTHONPATH=src python scripts/plot_rollout_sensor_analysis.py \
  --rollout-dir outputs/peg_hole_100/rollout_success_mid_dq002_001 \
  --output-dir outputs/peg_hole_100/rollout_success_mid_dq002_001/analysis_plots \
  --formats png,pdf
```

This reads:

- `outputs/peg_hole_100/rollout_success_mid_dq002_001/rollout_log.csv`
- `outputs/peg_hole_100/rollout_success_mid_dq002_001/summary.json`, if present

It writes plots and marker metadata to the output directory.

## Success vs Failure Comparison

```bash
PYTHONPATH=src python scripts/plot_rollout_sensor_analysis.py \
  --compare-rollout-dir-a outputs/peg_hole_100/rollout_success_mid_dq002_001 \
  --compare-rollout-dir-b outputs/peg_hole_100/new_success_mid_dq001_001 \
  --label-a success_mid_dq002 \
  --label-b failed_mid_dq001 \
  --formats png,pdf
```

The default comparison output directory is:

```text
outputs/peg_hole_100/comparison_analysis_success_mid_dq002_vs_failed_mid_dq001
```

## Outputs

Single-rollout mode writes:

- `task_error_vs_time.<format>`
- `force_vs_time.<format>`
- `action_adjustment_vs_time.<format>`
- `predicted_force_vs_time.<format>`
- `joint_command_vs_time.<format>`
- `joint_position_vs_time.<format>`
- `combined_analysis.<format>`
- `summary_markers.json`

Comparison mode writes:

- `compare_task_error.<format>`
- `compare_force.<format>`
- `compare_action_adjustment.<format>`
- `compare_predicted_force.<format>`
- `compare_combined_analysis.<format>`
- `compare_summary.json`

## What To Look For

- A `force_norm` rise at contact onset.
- Subsequent reduction in `peg_to_hole_lateral_error`.
- A temporary pause or reversal in axial error before re-approach.
- Changes in action-update magnitude after contact.
- Predicted-force changes around contact, retry, or insertion phases.
- In comparison mode, whether the successful run shows lower final distance and
  lateral error without excessive peak force.

## Interpretation Caution

These plots show temporal correlation, not definitive causality. To prove
force-conditioned behavior more strongly, compare with force ablations,
zero-force rollouts, or controlled rollouts where force inputs are delayed or
masked.

Recommended paper wording:

> The successful rollout shows a temporal coupling between contact-force rise,
> policy action updates, and lateral-error reduction, suggesting that the policy
> uses force-conditioned cues for contact-stage refinement.
