# Experiment Workflows

Commands use repository-relative placeholder paths. Replace `outputs/...`, split files, and XML paths with local artifacts. All commands were checked against current parser help where applicable.

## 1. Dataset Inspection

```bash
PYTHONPATH=src .venv/bin/python scripts/inspect_episode_collection.py \
  --episode-list configs/splits/peg_in_hole_100_train80.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output-csv outputs/peg_hole_100/dataset_inspection.csv
```

Use `inspect_real_hdf5.py` for one episode and `inspect_action_modes.py` when comparing action label sources.

## 2. Normalization-Stat Computation

```bash
PYTHONPATH=src .venv/bin/python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/peg_in_hole_100_train80.txt \
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
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py \
  --episode-list configs/splits/peg_in_hole_100_train80.txt \
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
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py \
  --episode-list configs/splits/peg_in_hole_100_train80.txt \
  --policy-variant force_aware_contact_cvae \
  --action-mode action \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --max-steps 20000 \
  --save-every 5000 \
  --save-steps 3000 10000 \
  --output-dir outputs/peg_hole_100/contact_cvae_20k \
  --log-csv outputs/peg_hole_100/contact_cvae_20k/train_log.csv
```

Intermediate checkpoints are named `checkpoint_step_XXXXXXXX.pt`; `checkpoint.pt` is always the final checkpoint. There is no resume CLI.

## 5. Offline Zero/Prior/Posterior Comparison

Definitions:

- Training-time posterior: future labels are used during training to sample a posterior latent.
- Deployment zero: online-only zero latent inference.
- Deployment prior: online-only deterministic conditional prior inference.
- Offline posterior oracle: future labels are encoded during evaluation; not deployable.

Contact-only:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_contact_cvae_modes.py \
  --episode-list configs/splits/peg_in_hole_100_val10.txt \
  --checkpoint outputs/peg_hole_100/contact_cvae_20k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --action-mode action \
  --posterior-mode mean \
  --output-csv outputs/peg_hole_100/contact_cvae_20k/offline_modes.csv
```

Motion-only:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_motion_cvae_modes.py \
  --episode-list configs/splits/peg_in_hole_100_val10.txt \
  --checkpoint outputs/peg_hole_100/motion_cvae_20k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --action-mode action \
  --posterior-mode mean \
  --output-csv outputs/peg_hole_100/motion_cvae_20k/offline_modes.csv
```

Dual-latent:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_inference_modes.py \
  --episode-list configs/splits/peg_in_hole_100_val10.txt \
  --checkpoint outputs/peg_hole_100/force_aware_act_20k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --action-mode action \
  --output-csv outputs/peg_hole_100/force_aware_act_20k/offline_modes.csv
```

## 6. Fixed Nominal Rollout

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_policy_rollout.py \
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
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_hole_grid.py \
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
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_hole_grid.py ... \
  --output-root outputs/peg_hole_100/lhs_contact_cvae_prior \
  --skip-existing \
  --continue-on-error
```

## 9. Paired Model Comparison

For paired comparison, preserve one `task_points.csv`/`grid_manifest.json` per protocol. The current runner regenerates task points from args and seed; it cannot load a task-point CSV directly. Use identical `--sampling-mode`, bounds, `--num-points`, and `--base-seed` across model runs.

Compare:

- `force_aware_act` zero vs prior.
- `force_aware_contact_cvae` zero vs prior.
- `force_aware_motion_cvae` zero-latent rollout.
- `act_baseline` zero-latent rollout.

## 10. Video Saving

Single rollout:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_policy_rollout.py ... \
  --save-videos \
  --video-fps 30 \
  --video-every 1
```

Grid rollout:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_hole_grid.py ... --save-videos
```

Video writing requires `imageio` and `imageio-ffmpeg`.

## 11. Parameter Audit

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_model_components.py \
  --policy-variant force_aware_act \
  --device cpu

PYTHONPATH=src .venv/bin/python scripts/audit_model_components.py \
  --policy-variant force_aware_motion_cvae \
  --device cpu

PYTHONPATH=src .venv/bin/python scripts/audit_model_components.py \
  --policy-variant force_aware_contact_cvae \
  --device cpu

PYTHONPATH=src .venv/bin/python scripts/audit_model_components.py \
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
- Test result from `PYTHONPATH=src .venv/bin/python -m pytest -q`.
