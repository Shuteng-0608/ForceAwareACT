# Scripts Reference

This inventory covers every file directly under `scripts/`. Status terms are descriptive: `current`, `specialized`, `diagnostic`, `compatibility`, `experiment-specific`, or `historical`.

## Training

| script | status | purpose | policies | inputs | outputs | modifies data | MuJoCo | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `train_minimal.py` | current | Main trainer for force-aware policies. | `force_aware_act`, `force_aware_motion_cvae`, `force_aware_contact_cvae` | HDF5 episodes or `--episode-list`, optional stats | `checkpoint.pt`, optional step checkpoints, CSV log | no | no | optional |
| `train_act_baseline.py` | current | Train force-free ACT Motion-CVAE baseline. | `act_baseline` | HDF5 episodes/list, optional stats | `checkpoint.pt`, optional step checkpoints, CSV log | no | no | optional |
| `train_contact_prior_stage2.py` | specialized | Stage-2 contact-prior distillation for dual-latent policy. | `force_aware_act` | Stage-1 checkpoint, HDF5/list, stats | checkpoint and CSV log | no | no | optional |

Typical commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py --episode-list configs/splits/peg_in_hole_100_train80.txt --policy-variant force_aware_contact_cvae --normalization-stats outputs/stats.pt --output-dir outputs/contact_cvae
PYTHONPATH=src .venv/bin/python scripts/train_act_baseline.py --episode-list configs/splits/peg_in_hole_100_train80.txt --normalization-stats outputs/stats.pt --output-dir outputs/act_baseline
```

Key flags: `--policy-variant`, `--action-mode`, `--train-latent-mode`, `--train-contact-latent-mode`, `--lambda-force`, `--lambda-prior`, `--prior-loss-mode`, `--beta-motion-max`, `--beta-contact-max`, `--save-every`, `--save-steps`, `--normalization-stats`.

Limitations: no resume CLI; `train_minimal.py` uses hard-coded small model settings; `train_contact_prior_stage2.py` is dual-latent-specific.

## Normalization

| script | status | purpose | policies | inputs | outputs | modifies data | MuJoCo | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `compute_normalization_stats.py` | current | Compute qpos/action/force mean/std and metadata. | all dataset-backed policies | HDF5/list | `.pt` stats | writes stats only | no | no |

Typical command:

```bash
PYTHONPATH=src .venv/bin/python scripts/compute_normalization_stats.py --episode-list configs/splits/peg_in_hole_100_train80.txt --action-mode action --chunk-len 10 --force-window-len 20 --output outputs/stats.pt
```

## Offline Evaluation

| script | status | purpose | policies | inputs | outputs | modifies data | MuJoCo | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `evaluate_inference_modes.py` | current | Compare dual-latent zero/prior/posterior contact modes. | `force_aware_act`; limited ACT fallback | checkpoint, stats, HDF5/list | CSV and optional ranked cases | no | no | optional |
| `evaluate_motion_cvae_modes.py` | current | Compare motion-CVAE zero vs posterior oracle. | `force_aware_motion_cvae` | checkpoint, stats, HDF5/list | per-sample CSV and printed summary | no | no | optional |
| `evaluate_contact_cvae_modes.py` | current | Compare contact-CVAE zero/prior/posterior oracle. | `force_aware_contact_cvae` | checkpoint, stats, HDF5/list | per-sample CSV and printed summary | no | no | optional |
| `evaluate_act_baseline_modes.py` | current | Compare ACT zero vs posterior motion oracle. | `act_baseline` | checkpoint, stats, HDF5/list | per-sample CSV and printed summary | no | no | optional |

Typical commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_contact_cvae_modes.py --episode-list configs/splits/peg_in_hole_100_val10.txt --checkpoint outputs/contact_cvae/checkpoint.pt --normalization-stats outputs/stats.pt --action-mode action --output-csv outputs/contact_cvae/eval.csv
PYTHONPATH=src .venv/bin/python scripts/evaluate_motion_cvae_modes.py --episode-list configs/splits/peg_in_hole_100_val10.txt --checkpoint outputs/motion_cvae/checkpoint.pt --normalization-stats outputs/stats.pt --posterior-mode mean
```

Limitations: posterior modes are oracle-only and non-deployable. `evaluate_motion_cvae_modes.py` uses `Path(episode_path).stem` for `episode_identifier`, so distinct files named `episode.hdf5` can collapse to `episode`. `evaluate_contact_cvae_modes.py` uses the parent episode directory name, with a stem fallback.

## Rollout and Simulation

| script | status | purpose | policies | inputs | outputs | modifies data | MuJoCo | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `run_mujoco_policy_rollout.py` | current | Single guarded MuJoCo rollout. | all checkpoint-dispatched policies | checkpoint, stats, XML | rollout CSV, summary JSON, snapshots/videos | no source/data change | yes | no |
| `run_mujoco_hole_grid.py` | current | Batch grid/random/LHS hole-offset rollout wrapper. | all rollout-supported policies | checkpoint, stats, XML | manifest, task CSV, grid summary, random summary | writes outputs | yes | no |
| `summarize_rollouts.py` | current | Aggregate rollout directories. | policy-agnostic | rollout dirs | summary CSV/table | writes summary | no | no |
| `plot_hole_grid_results.py` | current | Plot grid/LHS result heatmaps and scatters. | policy-agnostic | `grid_summary.csv` | PNG/PDF/etc and result tables | writes plots | no | no |

Typical commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_policy_rollout.py --checkpoint outputs/model/checkpoint.pt --normalization-stats outputs/stats.pt --model-xml ../arm_teleop/model/pangu_all_right.xml --action-mode action --action-select-mode mid --output-dir outputs/rollout --execute-actions
PYTHONPATH=src .venv/bin/python scripts/run_mujoco_hole_grid.py --sampling-mode latin_hypercube --num-points 50 --checkpoint outputs/model/checkpoint.pt --normalization-stats outputs/stats.pt --model-xml ../arm_teleop/model/pangu_all_right.xml --output-root outputs/lhs --continue-on-error
```

Key flags: `--contact-latent-mode`, `--action-select-mode`, `--temporal-agg-decay`, `--max-delta-q`, `--ema-alpha`, `--force-stop-threshold`, success thresholds, hole offset flags, `--save-videos`, `--skip-existing`, `--dry-run`, `--continue-on-error`.

## Visualization and Analysis

| script | status | purpose | inputs | outputs | notes |
| --- | --- | --- | --- | --- | --- |
| `analyze_train_log.py` | diagnostic | Summarize/plot training CSV metrics. | train log CSV | printed summary, optional plot | handles available metric columns. |
| `plot_rollout_sensor_analysis.py` | diagnostic | Analyze force/success/action markers from rollout logs. | rollout dir(s) | plots and summary files | supports compare mode. |
| `analyze_contact_stage.py` | diagnostic | Analyze contact-stage behavior from rollout CSVs and demos. | rollout logs, optional HDF5 | CSV/plot | can use model XML for geometry context. |
| `analyze_contact_latent.py` | specialized | Analyze dual-latent posterior contact latents and optional prior overlay. | HDF5/list, checkpoint, stats | CSV/plots | dual-latent oriented. |
| `inspect_inference_case_predictions.py` | diagnostic | Inspect one saved inference case/prediction. | episode, state index, checkpoint, stats | output directory artifacts | contact-mode debugging. |
| `inspect_worst_case_episode.py` | diagnostic | Inspect signals around one HDF5 state index. | episode, state index | CSV/frames | read-only on HDF5. |
| `plot_hole_target_map.py` | current | Plot measured hole-position rollout outcomes as a target-style spatial map. | `grid_summary.csv` with `point_index`, `hole_offset_x`, `hole_offset_z`, and `success` | PNG/PDF/SVG target maps | success is green circles, failure is red circles; target rings are concentric mm offsets only. |

`plot_hole_target_map.py` converts hole offsets from metres to millimetres, centers the nominal hole at `(0, 0)`, draws light grey concentric rings at `--ring-step-mm` intervals, and uses identical circular markers with black edges for measured success/failure outcomes. It plots measured rollout samples only; it does not estimate or interpolate a continuous success region.

Typical command:

```bash
PYTHONPATH=src python scripts/plot_hole_target_map.py --grid-summary-csv outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/grid_summary.csv --output-dir outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/plots --output-stem contact_cvae_zero_mid_10mm_target --title "Contact-CVAE zero + mid, ±10 mm LHS" --ring-step-mm 2 --formats png pdf --dpi 300
```

Labeled diagnostic command:

```bash
PYTHONPATH=src python scripts/plot_hole_target_map.py --grid-summary-csv outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/grid_summary.csv --output-dir outputs/peg_hole_100/hole_lhs_50_xz_10mm_contact_cvae100k_zero_mid_dq002/plots --output-stem contact_cvae_zero_mid_10mm_target_labeled --title "Contact-CVAE zero + mid, ±10 mm LHS" --ring-step-mm 2 --show-point-index --show-sampling-boundary --formats png --dpi 300
```

Important flags: `--max-radius-mm` for an explicit symmetric plot extent, `--marker-size` for point size, `--show-point-index` for per-point labels, `--show-sampling-boundary` for the sampled x/z bounding rectangle, `--formats` for `png`, `pdf`, and/or `svg`, and `--dpi` for raster output.

## Dataset Inspection

| script | status | purpose | inputs | outputs | notes |
| --- | --- | --- | --- | --- | --- |
| `inspect_real_hdf5.py` | diagnostic | Inspect one HDF5 episode fields and lengths. | episode path | printed report | dataset audit. |
| `inspect_episode_collection.py` | diagnostic | Inspect many HDF5 episodes for safe lengths/sample counts. | HDF5/list | optional CSV | checks action/window settings. |
| `inspect_action_modes.py` | diagnostic | Compare action-mode labels in local data. | data dir | printed samples | hard-coded default data dir. |

## Model Inspection and Smoke Tests

| script | status | purpose | policies | inputs | outputs |
| --- | --- | --- | --- | --- | --- |
| `audit_model_components.py` | current | Parameter-count/component audit. | all four variants when selected individually; `both` compares only `force_aware_act` and `act_baseline` | optional checkpoint | printed/JSON audit |
| `debug_one_batch.py` | diagnostic | Run one HDF5 batch through dual-latent policy and backprop. | `force_aware_act` | one episode | printed shapes/loss |
| `debug_inference_modes.py` | diagnostic | Compare one-batch zero/prior/posterior dual-latent outputs. | `force_aware_act` | checkpoint, stats, HDF5/list | printed metrics |
| `run_policy_inference_smoke.py` | diagnostic | Run deployable inference on one recorded sample. | force-aware variants | episode, checkpoint, stats | printed/chunk outputs |
| `forceact_eta.py` | diagnostic | Estimate ETA from a training log and PID. | n/a | log, max steps, PID | printed ETA |

## Replay and Geometry Validation

| script | status | purpose | requires MuJoCo | outputs |
| --- | --- | --- | --- | --- |
| `replay_hdf5_joint_trajectory_mujoco.py` | specialized | Replay HDF5 joint trajectories in MuJoCo. | yes | CSV/video/summary outputs |
| `audit_hdf5_replay_task_error.py` | specialized | Replay recorded qpos with `mj_forward` to audit task-space error. | yes | CSV/JSON/plots |
| `inspect_hole_assembly.py` | diagnostic | Validate selected hole body/site/geoms and test offsets. | yes | printed/JSON |
| `probe_arm_teleop_mujoco_env.py` | diagnostic | Probe arm_teleop XML names/sensors/cameras. | yes | printed/JSON |
| `probe_joint_command_convention.py` | diagnostic | Probe MuJoCo joint command convention. | yes | printed/JSON |

## Compatibility and Experiment Wrappers

| script | status | purpose | notes |
| --- | --- | --- | --- |
| `script_utils.py` | compatibility | Re-export path helpers for older script imports. | no CLI. |
| `run_peg_fixed_insert_100_experiment.sh` | experiment-specific | Fixed 100-episode prepare/train/eval/rollout wrapper for dual-latent workflow. | Preserves historical workflow; default stage is `smoke`. |

## General Limitations

- Several scripts assume repository root or insert `src` into `sys.path`.
- Some scripts use hard-coded default paths such as `../arm_teleop/model/pangu_all_right.xml` or `mujoco_data/...`.
- Generated outputs should remain under ignored output directories.
- Many diagnostic scripts are dual-latent-specific and should not be assumed to support every policy variant.
