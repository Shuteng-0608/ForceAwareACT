# Command Recipes

This is the compact copy/paste command reference. It intentionally omits the
decision rationale and troubleshooting found in the full training and rollout
manuals.

Commands use repository-relative placeholder paths. Replace `outputs/...`, split files, and XML paths with local artifacts. All commands were checked against current parser help on 2026-07-16. Run them from the repository root in an activated environment.

## 1. Dataset Inspection

```bash
PYTHONPATH=src python scripts/inspect_episode_collection.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output-csv outputs/peg_hole_100/dataset_inspection.csv
```

Use `inspect_real_hdf5.py` for one episode and `inspect_action_modes.py` when comparing action label sources. For current command-labelled recordings, also run the stricter collection-quality gate:

```bash
PYTHONPATH=src python scripts/evaluate_dataset_quality.py \
  mujoco_data/peg_hole_100 \
  --output-csv outputs/peg_hole_100/quality_report.csv \
  --output-json outputs/peg_hole_100/quality_summary.json
```

`evaluate_dataset_quality.py` checks collection status, command tracking, timing, force, motion, and sampled image quality. Its thresholds produce review signals rather than physical proof, and its schema is stricter than the general dataset reader. Both audits are needed for a new dataset.

## 2. Normalization-Stat Computation

```bash
PYTHONPATH=src python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --action-mode action \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --camera-names ee_cam base_top_cam \
  --output outputs/peg_hole_100/normalization_stats_action_train80.pt
```

Stats must match downstream `--action-mode`. Recompute stats when changing action label semantics.

## 3. 5k Pilot Training

Dual-latent:

```bash
PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --val-episode-list configs/splits/peg_hole_100_val10.txt \
  --policy-variant force_aware_act \
  --action-mode action \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --lambda-force 0.1 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --max-steps 5000 \
  --output-dir outputs/peg_hole_100/pilot_force_aware_act_5k \
  --log-csv outputs/peg_hole_100/pilot_force_aware_act_5k/train_log.csv
```

Contact-only and motion-only pilots change only `--policy-variant` and prior settings as described in `README.md`.

## 4. Periodic Checkpoint Training

```bash
PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --val-episode-list configs/splits/peg_hole_100_val10.txt \
  --policy-variant force_aware_contact_cvae \
  --action-mode action \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --max-steps 20000 \
  --save-every 5000 \
  --save-steps 3000 10000 \
  --seed 0 \
  --deterministic \
  --torch-num-threads 4 \
  --torch-num-interop-threads 1 \
  --output-dir outputs/peg_hole_100/contact_cvae_20k \
  --log-csv outputs/peg_hole_100/contact_cvae_20k/train_log.csv
```

Intermediate checkpoints are named `checkpoint_step_XXXXXXXX.pt`; `checkpoint.pt` is always the final state. With a validation list, `checkpoint_best.pt` is updated on monitored-metric improvement and should be used for candidate selection. There is no resume CLI.

The seed/deterministic/thread example applies to `train_minimal.py`. `train_act_baseline.py` and `train_contact_prior_stage2.py` do not currently expose those flags; record that asymmetry in comparisons.

## 5. Offline Zero/Prior/Posterior Comparison

Definitions:

- Training-time posterior: future labels are used during training to sample a posterior latent.
- Deployment zero: online-only zero latent inference.
- Deployment prior: online-only deterministic conditional prior inference.
- Offline posterior oracle: future labels are encoded during evaluation; not deployable.

Contact-only:

```bash
PYTHONPATH=src python scripts/evaluate_contact_cvae_modes.py \
  --episode-list configs/splits/peg_hole_100_val10.txt \
  --checkpoint outputs/peg_hole_100/contact_cvae_20k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --action-mode action \
  --posterior-mode mean \
  --output-csv outputs/peg_hole_100/contact_cvae_20k/offline_modes.csv
```

Motion-only:

```bash
PYTHONPATH=src python scripts/evaluate_motion_cvae_modes.py \
  --episode-list configs/splits/peg_hole_100_val10.txt \
  --checkpoint outputs/peg_hole_100/motion_cvae_20k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --action-mode action \
  --posterior-mode mean \
  --output-csv outputs/peg_hole_100/motion_cvae_20k/offline_modes.csv
```

Dual-latent:

```bash
PYTHONPATH=src python scripts/evaluate_inference_modes.py \
  --episode-list configs/splits/peg_hole_100_val10.txt \
  --checkpoint outputs/peg_hole_100/force_aware_act_20k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --action-mode action \
  --output-csv outputs/peg_hole_100/force_aware_act_20k/offline_modes.csv
```

## 6. Fixed Nominal Rollout

```bash
PYTHONPATH=src python scripts/run_mujoco_policy_rollout.py \
  --checkpoint outputs/peg_hole_100/contact_cvae_20k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --contact-latent-mode prior \
  --action-mode action \
  --action-select-mode mid \
  --max-rollout-steps 900 \
  --max-delta-q 0.02 \
  --output-dir outputs/peg_hole_100/rollouts/contact_cvae_prior_nominal \
  --execute-actions
```

## 7. 50-Point LHS Rollout

```bash
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py \
  --sampling-mode latin_hypercube \
  --num-points 50 \
  --x-min -0.002 --x-max 0.002 \
  --z-min -0.002 --z-max 0.002 \
  --base-seed 20260702 \
  --checkpoint outputs/peg_hole_100/contact_cvae_20k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --contact-latent-mode prior \
  --action-mode action \
  --action-select-mode mid \
  --output-root outputs/peg_hole_100/lhs_contact_cvae_prior \
  --continue-on-error
```

## 8. Resume After Process Errors

There is no training resume CLI. For grid rollout process errors:

1. Rerun the same command with identical bounds, seed, checkpoint, stats, and output root.
2. Add `--skip-existing` to avoid rerunning directories that already have `summary.json`.
3. Keep `--continue-on-error` to collect later points if one subprocess fails.

```bash
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py ... \
  --output-root outputs/peg_hole_100/lhs_contact_cvae_prior \
  --skip-existing \
  --continue-on-error
```

## 9. Paired Model Comparison

For paired comparison, freeze one point CSV and pass it to every model. A grid-generated `task_points.csv` contains extra protocol/output columns but still includes the required `point_index`, `hole_offset_x`, `hole_offset_y`, and `hole_offset_z` fields.

```bash
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py \
  --sampling-mode file \
  --task-points-csv configs/experiments/fibonacci_disk_100_r4mm.csv \
  --checkpoint outputs/model/checkpoint_best.pt \
  --normalization-stats outputs/stats.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --output-root outputs/fixed_protocol/model_a \
  --skip-existing --continue-on-error
```

For generated random/LHS protocols, set `--point-set-seed` and `--rollout-seed-base` separately. Point `i` uses `rollout_seed_base + i - 1`; identical coordinates alone do not imply identical rollout randomness.

Compare:

- `force_aware_act` zero vs prior.
- `force_aware_contact_cvae` zero vs prior.
- `force_aware_motion_cvae` zero-latent rollout.
- `act_baseline` zero-latent rollout.

## 10. Video Saving

Single rollout:

```bash
PYTHONPATH=src python scripts/run_mujoco_policy_rollout.py ... \
  --save-videos \
  --video-fps 30 \
  --video-every 1
```

Grid rollout:

```bash
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py ... --save-videos
```

Video writing requires `imageio` and `imageio-ffmpeg`.

## 11. Parameter Audit

```bash
PYTHONPATH=src python scripts/audit_model_components.py \
  --policy-variant force_aware_act \
  --device cpu

PYTHONPATH=src python scripts/audit_model_components.py \
  --policy-variant force_aware_motion_cvae \
  --device cpu

PYTHONPATH=src python scripts/audit_model_components.py \
  --policy-variant force_aware_contact_cvae \
  --device cpu

PYTHONPATH=src python scripts/audit_model_components.py \
  --policy-variant act_baseline \
  --device cpu
```

Use `--config-from-checkpoint` to audit with checkpoint model dimensions. `--policy-variant both` compares only `force_aware_act` and `act_baseline`; it does not cover the two single-latent force-aware variants.

## 12. Experiment Output Naming

Recommended structure:

```text
outputs/<dataset_or_protocol>/
  normalization_stats_<action_mode>_<split>.pt
  <policy>_<steps>/
    train_log.csv
    checkpoint.pt
    checkpoint_step_XXXXXXXX.pt
    offline_modes.csv
  rollouts/<policy>_<mode>_<selection>/
  lhs_<policy>_<mode>_<seed>/
```

## 13. Recording Commands and Results

For every experiment record:

- Git status and commit hash if available.
- Exact command lines.
- Split files and episode counts.
- Normalization stats path and metadata.
- Checkpoint path and `config`.
- Evaluation CSV path.
- Rollout/grid output directory.
- `task_points.csv` and `grid_manifest.json` for paired comparisons.
- Point-set seed and rollout-seed base as distinct fields.
- Test result from `PYTHONPATH=src python -m pytest -q`.

## 14. Multi-Seed Rollout and Safety Reclassification

Use the multi-seed suite when estimating sensitivity to sampled task points and rollout randomness:

```bash
PYTHONPATH=src python scripts/run_xz_multiseed_rollout_suite.py \
  --point-set-seeds 20260702 20260703 \
  --rollout-seed-bases 31000 32000 \
  --action-select-modes mid \
  --offset-mm 4 \
  --output-base outputs/peg_hole_100/separated_seed_rollouts
```

The command writes a protocol `suite_plan.json` before running and aggregates only complete model/mode configurations. Monitor it with `monitor_xz_rollout_suite.py`.

`plot_hole_target_map.py --safe-force-threshold N` can reclassify a generic grid CSV without altering it. `analyze_rollout_safety_threshold.py` is not a generic alternative: it assumes the dataset-scaling `mix50/mix100/mix150/mix203` directory names and exactly 100 summaries per model.
