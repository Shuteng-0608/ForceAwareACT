# ForceAwareACT

ForceAwareACT is a research codebase for ACT-style visuomotor policies that study how online wrist force and learned latent variables affect contact-rich peg-in-hole insertion. The repository contains HDF5 dataset readers, normalization utilities, four implemented policy families, offline latent-mode evaluators, guarded MuJoCo rollout tooling, batch hole-position experiments, plotting utilities, and architecture/test documentation.

This repository intentionally does not include HDF5 datasets, checkpoints, generated outputs, CSV logs, videos, or plots. Those artifacts are ignored by git and should remain local experiment outputs.

## Research Motivation

The project separates several questions that are often mixed together in contact-rich imitation learning:

- Does online force history improve action prediction and rollout behavior?
- Does a motion CVAE latent help model multi-modal future actions?
- Does a contact CVAE latent help model future force/contact outcomes?
- Can a conditional contact prior replace oracle future-contact labels at deployment?
- How do offline posterior-oracle metrics relate to deployable zero/prior rollout modes?

The implementation treats future action chunks and future force chunks as training/evaluation labels only. Deployment and MuJoCo rollout use online images, current qpos, and historical force only, except for the force-free ACT baseline which does not read force at all.

## Implemented Policy Families

| policy_variant | Python class | Source | Research role |
| --- | --- | --- | --- |
| `force_aware_act` | `ForceAwareACTPolicy` | `src/force_aware_act/models/policy.py` | Full dual-latent model with motion and contact latents. |
| `force_aware_motion_cvae` | `ForceAwareACTMotionCVAEPolicy` | `src/force_aware_act/models/force_aware_motion_cvae_policy.py` | Force-aware, structurally motion-only CVAE. No contact latent or contact prior. |
| `force_aware_contact_cvae` | `ForceAwareACTContactCVAEPolicy` | `src/force_aware_act/models/force_aware_contact_cvae_policy.py` | Force-aware, structurally contact-only CVAE. No motion latent. |
| `act_baseline` | `ACTPolicyBaseline` | `src/force_aware_act/models/act_policy.py` | Force-free ACT-style Motion-CVAE baseline. No learned contact latent, online force, force head, or contact prior. |

## Model Comparison Table

| policy | vision | qpos | online force | z_motion | z_contact | contact prior | action head | force head | training latent | deployment latent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `force_aware_act` | yes | yes | yes | yes | yes | yes | decoder hidden | decoder hidden + `z_contact` | posterior or zero via `--train-latent-mode` | zero or deterministic prior via `--contact-latent-mode` |
| `force_aware_motion_cvae` | yes | yes | yes | yes | no | no | decoder hidden | decoder hidden + zero aux | motion posterior | zero motion latent in rollout |
| `force_aware_contact_cvae` | yes | yes | yes | no | yes | yes | decoder hidden | decoder hidden + `z_contact` | contact posterior | zero or deterministic contact prior |
| `act_baseline` | yes | yes | no | yes | no | no | decoder hidden | none | motion posterior | zero motion latent |

Token order is policy-specific and documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). In short: the force-aware policies use visual spatial tokens plus state/force/fusion tokens and the relevant latent token(s); the ACT baseline uses visual tokens, qpos token, and motion latent token only.

## Repository Structure

```text
src/force_aware_act/
  data/        HDF5 episode dataset, timestamp alignment, normalization
  models/      encoders, posterior/prior modules, heads, policy classes
  training/    losses, epoch accounting, deployment validation, and early stopping
  utils/       episode-list path resolution helpers
scripts/       training, normalization, evaluation, rollout, plotting, inspection
tests/         unit, integration, CLI, evaluator, rollout, and audit tests
docs/          architecture, workflow, script, test, and historical reports
configs/       split lists and experiment notes
```

See [docs/SCRIPTS_REFERENCE.md](docs/SCRIPTS_REFERENCE.md) and [docs/TESTING.md](docs/TESTING.md) for full inventories.

## Installation

Use an existing local virtual environment or create one outside version control. The package requires Python 3.9+ and the base dependencies in `pyproject.toml`: `h5py`, `numpy`, `torch`, and `torchvision`. Tests require `pytest`. MuJoCo rollout scripts additionally require `mujoco`; video output requires `imageio` and `imageio-ffmpeg`.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Dataset Format

`ContactForceHDF5Dataset` expects synchronized HDF5 episodes with:

- `timestamps/state_episode` or `timestamps/state`
- `timestamps/image_episode` or `timestamps/image`
- `timestamps/force_episode` or `timestamps/force` for force-aware policies
- `observations/images/<camera>`, default `ee_cam` and `base_top_cam`
- `observations/joint_pos`, `observations/joint_vel`, `observations/joint_torque`, `observations/ee_pose`
- `observations/ft_wrench` for force-aware policies
- action labels from `observations/joint_pos`, root `action`, or `actions/joint_pos_command` depending on `--action-mode`

Supported action modes are `joint_pos`, `action`, `joint_pos_command`, `delta_joint_cmd`, and `delta_joint_pos_command`. `joint_pos` uses an offset of one state step; command modes are aligned at the current decision index. Delta modes subtract current qpos from the selected action source.

Images are read as HWC RGB, scaled to `[0, 1]` by default, resized to `--image-size`, converted to CHW tensors, and optionally ImageNet-normalized. State, image, and force groups tolerate a one-frame mismatch by default in the dataset implementation.

Force-frame assumptions are not encoded in code. The reader uses recorded `observations/ft_wrench` directly; rollout uses MuJoCo force/torque sensors directly. There is no implemented bias removal, filtering, gravity compensation, sign conversion, or frame conversion.

## Normalization Workflow

Compute stats on the training split and keep the same `--action-mode`, chunk/window lengths, cameras, image size, and ImageNet setting for training, evaluation, and rollout.

```bash
PYTHONPATH=src .venv/bin/python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --action-mode action \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output outputs/peg_hole_100/normalization_stats_action_train80.pt
```

The `.pt` file is a `torch.save` dictionary with tensor fields `qpos_mean`, `qpos_std`, `action_mean`, `action_std`, `force_mean`, `force_std`, plus metadata such as `action_mode`, `chunk_len`, `force_window_len`, `force_window_duration`, `camera_names`, `image_size`, `imagenet_normalize`, `episode_paths`, and `episode_list`.

## Training Workflows

All commands below use placeholder local paths; replace `outputs/...` and split files with paths present in your workspace.

ACT baseline:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_act_baseline.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --val-episode-list configs/splits/peg_hole_100_val10.txt \
  --action-mode action \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --max-steps 200000 \
  --max-epochs 100 \
  --save-every 1000 \
  --output-dir outputs/peg_hole_100/act_baseline_5k \
  --log-csv outputs/peg_hole_100/act_baseline_5k/train_log.csv
```

Dual-latent ForceAwareACT:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --policy-variant force_aware_act \
  --action-mode action \
  --train-latent-mode posterior \
  --lambda-force 0.1 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --max-steps 5000 \
  --output-dir outputs/peg_hole_100/force_aware_act_5k \
  --log-csv outputs/peg_hole_100/force_aware_act_5k/train_log.csv
```

Motion-only force-aware CVAE:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --policy-variant force_aware_motion_cvae \
  --action-mode action \
  --train-latent-mode posterior \
  --lambda-force 0.1 \
  --lambda-prior 0 \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --max-steps 5000 \
  --output-dir outputs/peg_hole_100/motion_cvae_5k \
  --log-csv outputs/peg_hole_100/motion_cvae_5k/train_log.csv
```

Contact-only force-aware CVAE:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py \
  --episode-list configs/splits/peg_hole_100_train80.txt \
  --policy-variant force_aware_contact_cvae \
  --action-mode action \
  --train-latent-mode posterior \
  --train-contact-latent-mode posterior \
  --lambda-force 0.1 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --max-steps 5000 \
  --output-dir outputs/peg_hole_100/contact_cvae_5k \
  --log-csv outputs/peg_hole_100/contact_cvae_5k/train_log.csv
```

Add `--val-episode-list` to enable epoch-level deterministic deployment validation and early stopping. The defaults validate every epoch, require at least 10 epochs, and stop after 8 validation checks without a 0.5% relative improvement. `--max-steps` and optional `--max-epochs` are simultaneous safety bounds; the first reached bound wins.

Training writes `checkpoint.pt`, `train_log.csv`, and optional `checkpoint_step_XXXXXXXX.pt` files. Validation additionally writes `checkpoint_best.pt` and `validation_log.csv`. The final checkpoint records epoch position, best metric metadata, patience state, and `stop_reason`. Use `checkpoint_best.pt` for test/rollout selection. There is no resume CLI.

## Offline Latent-Mode Evaluation

Deployment zero means the deployable zero latent branch. Deployment prior means deterministic conditional-prior inference using only online inputs. Offline posterior oracle means future action/force labels are encoded for analysis and must not be treated as deployable inference.

Dual-latent evaluator:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_inference_modes.py \
  --episode-list configs/splits/peg_hole_100_val10.txt \
  --checkpoint outputs/peg_hole_100/force_aware_act_5k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --action-mode action \
  --output-csv outputs/peg_hole_100/force_aware_act_5k/inference_modes.csv
```

Motion-CVAE evaluator:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_motion_cvae_modes.py \
  --episode-list configs/splits/peg_hole_100_val10.txt \
  --checkpoint outputs/peg_hole_100/motion_cvae_5k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --action-mode action \
  --posterior-mode mean \
  --output-csv outputs/peg_hole_100/motion_cvae_5k/zero_vs_posterior.csv
```

Contact-CVAE evaluator:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_contact_cvae_modes.py \
  --episode-list configs/splits/peg_hole_100_val10.txt \
  --checkpoint outputs/peg_hole_100/contact_cvae_5k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --action-mode action \
  --posterior-mode mean \
  --output-csv outputs/peg_hole_100/contact_cvae_5k/zero_prior_posterior.csv
```

## MuJoCo Rollout and Deployment

`scripts/run_mujoco_policy_rollout.py` reconstructs the policy from checkpoint metadata, loads normalization stats, renders `ee_cam` and `base_top_cam`, constructs a historical force window from MuJoCo sensors, runs deployable inference, denormalizes actions, selects an action from the chunk, clips to `--max-delta-q`, applies EMA, clips to actuator control range, and logs rollout diagnostics.

Dual-latent zero mode:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_policy_rollout.py \
  --checkpoint outputs/peg_hole_100/force_aware_act_5k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --contact-latent-mode zero \
  --action-mode action \
  --action-select-mode mid \
  --output-dir outputs/peg_hole_100/rollouts/force_aware_act_zero \
  --execute-actions
```

Dual-latent prior mode uses `--contact-latent-mode prior`. Contact-only zero/prior rollout uses the contact-only checkpoint with the same two modes. Motion-CVAE and ACT-baseline checkpoints are dispatched by `config.policy_variant`; `--contact-latent-mode` is still present in the shared CLI, but it is ignored by the motion-CVAE and ACT branches.

Motion-CVAE rollout:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_policy_rollout.py \
  --checkpoint outputs/peg_hole_100/motion_cvae_5k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --action-mode action \
  --action-select-mode mid \
  --output-dir outputs/peg_hole_100/rollouts/motion_cvae \
  --execute-actions
```

ACT baseline rollout:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_policy_rollout.py \
  --checkpoint outputs/peg_hole_100/act_baseline_5k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --action-mode action \
  --action-select-mode mid \
  --output-dir outputs/peg_hole_100/rollouts/act_baseline \
  --execute-actions
```

Task success, safe success, and hard force stop are different concepts. Task success is a held rollout condition using distance, lateral error, and instantaneous force thresholds. Safe success is computed in grid summaries as task success with max force below the success force threshold. Hard force stop is an immediate stop when `force_norm > --force-stop-threshold`.

## Batch Hole-Position Experiments

Run a 50-point Latin-hypercube perturbation experiment:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_hole_grid.py \
  --sampling-mode latin_hypercube \
  --num-points 50 \
  --x-min -0.002 --x-max 0.002 \
  --z-min -0.002 --z-max 0.002 \
  --base-seed 20260702 \
  --checkpoint outputs/peg_hole_100/contact_cvae_5k/checkpoint.pt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_train80.pt \
  --model-xml ../arm_teleop/model/pangu_all_right.xml \
  --contact-latent-mode prior \
  --action-mode action \
  --action-select-mode mid \
  --output-root outputs/peg_hole_100/hole_lhs_contact_cvae_prior \
  --continue-on-error
```

The runner writes `task_points.csv`, `grid_manifest.json`, per-run rollout directories, `grid_summary.csv`, `random_position_summary.json`, and optional plots.

## Checkpoint Format

Current training checkpoints are dictionaries with:

- `model_state_dict`: PyTorch model parameters.
- `optimizer_state_dict`: AdamW state.
- `config`: resolved training/model metadata, including `policy_variant`.
- `step`: final or intermediate one-based training step.

Compatibility behavior varies by consumer. Rollout defaults missing `config.policy_variant` to `force_aware_act` and loads strictly. Motion/contact evaluators accept raw state dictionaries but require the matching policy variant for envelope checkpoints and use strict loading. ACT baseline rollout supports legacy zero-latent ACT checkpoints through `LegacyZeroLatentACTPolicyBaseline`; the dedicated ACT evaluator requires the corrected `act_baseline_version=motion_cvae_v1`.

## Testing

Full suite:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Focused model tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_force_aware_act_policy.py \
  tests/test_force_aware_motion_cvae_policy.py \
  tests/test_force_aware_contact_cvae_policy.py \
  tests/test_act_policy_baseline.py
```

Model audit for all four variants:

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

`--policy-variant both` is a narrow comparison mode for `force_aware_act` versus `act_baseline`; it does not audit the two single-latent force-aware variants.

See [docs/TESTING.md](docs/TESTING.md) for the complete test-file inventory.

## Reproducibility and Experiment Hygiene

- Compute normalization stats from the training split only.
- Keep `action_mode`, `chunk_len`, `force_window_len`, `force_window_duration`, cameras, and image preprocessing consistent across normalization, training, evaluation, and rollout.
- Record every command, checkpoint path, stats path, split file, seed, and output directory.
- Keep generated outputs under ignored directories such as `outputs/`.
- Treat offline posterior-oracle numbers as analysis, not deployment performance.
- Preserve grid `task_points.csv` and `grid_manifest.json` when comparing model families pairwise.

## Documentation Index

- [Architecture](docs/ARCHITECTURE.md)
- [Scripts Reference](docs/SCRIPTS_REFERENCE.md)
- [Testing](docs/TESTING.md)
- [Experiment Workflows](docs/EXPERIMENT_WORKFLOWS.md)
- [Five-Model Training and Early-Stopping Manual](docs/MODEL_TRAINING_AND_EARLY_STOPPING_MANUAL.md)
- [Repository Architecture Audit](docs/REPOSITORY_ARCHITECTURE_AUDIT.md)
- Historical reports remain in `docs/` and are useful as experiment records, but the files above are the current canonical entry points.

## Known Limitations

- No training resume CLI is implemented.
- Model construction and checkpoint dispatch logic are duplicated across scripts.
- The shared rollout CLI still exposes `--contact-latent-mode` for policies that ignore it.
- Force coordinate frame, sign convention, bias removal, filtering, gravity compensation, and dataset-vs-MuJoCo wrench equivalence are not encoded or validated.
- The motion-CVAE evaluator uses `Path(episode_path).stem` for `episode_identifier`, so distinct files named `episode.hdf5` can collapse to `episode`. The contact-CVAE evaluator uses the parent episode directory name, with a stem fallback.
- Grid/LHS runner regenerates points from arguments and seed; it cannot load an existing task-point CSV as the source of truth.
- Training scripts use small hard-coded model defaults in `train_minimal.py`; `train_act_baseline.py` exposes more architecture flags.

## Development Status

Current code implements all four policy families listed above, including `force_aware_contact_cvae`. Documentation verification snapshot on 2026-07-07: this README reflects source code and CLI help inspected during the documentation audit. No source behavior is changed by the documentation audit.
