# Historical Rollout Action-Selection Notes

> Historical experiment note: this file preserves early single-checkpoint
> commands and observations comparing action-selection modes and
> `max_delta_q`. It is not the current rollout contract or a general policy
> recommendation. Use
> [`ROLLOUT_EXPERIMENT_MANUAL.md`](ROLLOUT_EXPERIMENT_MANUAL.md) for current
> success, safety, execution, and aggregation behavior.

## Success Criteria

A rollout policy step is a success candidate when all three conditions hold:

- `peg_to_hole_dist < --success-distance-threshold`
- `peg_to_hole_lateral_error < --success-lateral-threshold`
- `force_norm < --success-force-threshold`

The defaults are:

- `--success-distance-threshold 0.005`
- `--success-lateral-threshold 0.006`
- `--success-force-threshold 40.0`
- `--success-hold-steps 15`

Success is recorded only after the candidate condition holds for `success_hold_steps`
consecutive policy steps. The hold requirement filters out transient passes through
the goal region and makes the stop reason better reflect stable insertion rather
than one lucky sampled state.

Pass `--disable-success-stop` to keep running after success is detected. The rollout
still records `success`, `success_step`, `success_time`, and hold statistics in
`summary.json` and `rollout_log.csv`.

## Example Rollout

Mid action selection with `max_delta_q=0.020`:

```bash
PYTHONPATH=src python scripts/run_mujoco_policy_rollout.py \
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
  --max-rollout-steps 120 \
  --max-delta-q 0.02 \
  --force-stop-threshold 1000 \
  --hole-axis-world 0 -1 0 \
  --output-dir outputs/peg_hole_100/rollout_action_trainzero_20k_bs16_mid_dq002 \
  --execute-actions
```

Each run writes:

- `rollout_log.csv`
- `summary.json`

The terminal output also prints `summary_json=...`.

## Repeated Rollouts

Compare first, mid, last, and temporal action selection at `max_delta_q=0.010`:

```bash
for mode in first mid last temporal; do
  PYTHONPATH=src python scripts/run_mujoco_policy_rollout.py \
    --checkpoint outputs/peg_hole_100/action_trainzero_all100_20k_bs16/checkpoint.pt \
    --normalization-stats outputs/peg_hole_100/normalization_stats_action_all100.pt \
    --model-xml ../arm_teleop/model/pangu_all_right.xml \
    --contact-latent-mode zero \
    --action-mode action \
    --action-select-mode "$mode" \
    --chunk-len 10 \
    --force-window-len 20 \
    --force-window-duration 0.25 \
    --policy-rate-hz 30 \
    --max-rollout-steps 120 \
    --max-delta-q 0.010 \
    --force-stop-threshold 1000 \
    --hole-axis-world 0 -1 0 \
    --output-dir "outputs/peg_hole_100/rollout_action_trainzero_20k_bs16_${mode}_dq001" \
    --execute-actions
done
```

Compare mid action selection at `max_delta_q=0.010`, `0.015`, and `0.020`:

```bash
for dq in 0.010 0.015 0.020; do
  tag=$(printf "%s" "$dq" | tr -d .)
  PYTHONPATH=src python scripts/run_mujoco_policy_rollout.py \
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
    --max-rollout-steps 120 \
    --max-delta-q "$dq" \
    --force-stop-threshold 1000 \
    --hole-axis-world 0 -1 0 \
    --output-dir "outputs/peg_hole_100/rollout_action_trainzero_20k_bs16_mid_dq${tag}" \
    --execute-actions
done
```

## Aggregate Results

```bash
PYTHONPATH=src python scripts/summarize_rollouts.py \
  --root outputs/peg_hole_100 \
  --pattern "rollout_action_trainzero_20k_bs16*" \
  --output outputs/peg_hole_100/rollout_action_trainzero_20k_bs16_summary.csv
```

The aggregator prefers each run's `summary.json`. If it is missing, it falls back
to `rollout_log.csv`.

## Recommended Interpretation

- `mid + dq=0.015` is safer if prioritizing low force.
- `mid + dq=0.020` gives the best current distance in the observed run.
- `first` is not suitable for this policy/checkpoint.
- `temporal` may increase contact force and needs careful evaluation.
