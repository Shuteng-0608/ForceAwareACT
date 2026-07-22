# Scripts Reference

This inventory covers every file directly under `scripts/` as audited on 2026-07-16. Status terms are descriptive: `current`, `specialized`, `diagnostic`, `compatibility`, or `experiment-specific`. Run commands from the repository root in an activated environment.

## Training

| script | status | purpose | policies | inputs | outputs | modifies data | MuJoCo | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `train_minimal.py` | current | Main trainer with epoch accounting, deployment-path validation, and early stopping. | `force_aware_act`, `force_aware_motion_cvae`, `force_aware_contact_cvae` | train HDF5/list, optional validation list and stats | final/best/step checkpoints, train and validation CSV logs | no | no | optional |
| `train_staged.py` | current | Strict protocol-driven multi-stage training with exact domain/phase quotas, selected-parent lineage, hashed auxiliary resume state, rollback quarantine, and independent retention validation. See [the staged visual-force protocol](../training/STAGED_VISUAL_FORCE_TRAINING.md). | all force-aware variants | protocol JSON, manifest, balanced stats, split lists, optional phase catalog and selected parent checkpoint | versioned best/periodic checkpoints, run manifest, train/validation CSV logs, optional resume quarantine | no | no | optional |
| `train_act_baseline.py` | current | Train force-free ACT Motion-CVAE baseline with validation and early stopping. | `act_baseline` | train HDF5/list, optional validation list and stats | final/best/step checkpoints, train and validation CSV logs | no | no | optional |
| `train_contact_prior_stage2.py` | specialized | Stage-2 contact-prior distillation with optional deployment-path validation and early stopping. | `force_aware_act` | Stage-1 checkpoint, train HDF5/list, optional validation list, stats | final/best checkpoints and train/validation CSV logs | no | no | optional |

Typical commands:

```bash
PYTHONPATH=src python scripts/train_minimal.py --episode-list configs/splits/peg_hole_100_train80.txt --val-episode-list configs/splits/peg_hole_100_val10.txt --policy-variant force_aware_contact_cvae --lambda-prior 0.1 --normalization-stats outputs/stats.pt --max-steps 200000 --max-epochs 100 --output-dir outputs/contact_cvae
PYTHONPATH=src python scripts/train_act_baseline.py --episode-list configs/splits/peg_hole_100_train80.txt --val-episode-list configs/splits/peg_hole_100_val10.txt --normalization-stats outputs/stats.pt --max-steps 200000 --max-epochs 100 --output-dir outputs/act_baseline
```

Key stopping flags: `--val-episode-list`, `--max-epochs`, `--val-every-epochs`, `--early-stop-patience`, `--early-stop-min-epochs`, `--early-stop-min-delta`, `--early-stop-metric`, and `--validation-deployment-mode`. `--max-steps` remains a safety cap and legacy step-only runs remain supported.

With validation enabled, `checkpoint_best.pt` contains the lowest monitored deployment metric, while `checkpoint.pt` contains the final state and records `stop_reason`. Force-aware defaults monitor normalized `action_l1 + lambda_force * force_l1`; ACT baseline monitors normalized action L1. Conditional-prior models use deterministic prior validation only when prior training is enabled.

Limitations: no resume CLI; `train_minimal.py` uses hard-coded small model settings; `train_contact_prior_stage2.py` is dual-latent-specific. Reproducible seed/deterministic/thread flags are implemented only by `train_minimal.py`; the ACT baseline and stage-2 trainers do not currently expose them.

## Dataset Splitting

| script | status | purpose | inputs | outputs |
| --- | --- | --- | --- | --- |
| `split_episode_list.py` | current | Deterministically split a source list at episode granularity. | source episode list, counts, seed | train/validation/test lists with provenance headers |
| `build_dataset_manifest.py` | current | Build a canonical domain/split manifest and reject UUID, canonical-path, or content-SHA leakage; SHA-derived UUIDs are explicit historical compatibility only. | repeated domain/split episode-list groups | immutable manifest JSON and content SHA-256 |
| `build_phase_catalog.py` | current | Validate manual phase-segment CSV annotations against the exact usable dataset indices and bind every entry to a pinned dataset manifest; it never infers labels from force thresholds. | annotation CSV, exact domain train episode list, pinned manifest path/content SHA-256, exact dataset semantics | immutable schema-v2 phase-catalog JSON and content SHA-256 |

## Normalization

| script | status | purpose | policies | inputs | outputs | modifies data | MuJoCo | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `compute_normalization_stats.py` | current | Compute qpos/action/force mean/std and provenance; repeated `--domain` inputs select domain→episode→time balanced raw-stream statistics. | all dataset-backed policies | HDF5/list | `.pt` stats | writes stats only | no | no |

Typical command:

```bash
PYTHONPATH=src python scripts/compute_normalization_stats.py --episode-list configs/splits/peg_hole_100_train80.txt --action-mode action --chunk-len 10 --force-window-len 20 --output outputs/stats.pt
```

## Offline Evaluation

| script | status | purpose | policies | inputs | outputs | modifies data | MuJoCo | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `evaluate_inference_modes.py` | diagnostic | Compare dual-latent zero/prior/posterior contact modes; posterior is an oracle diagnostic and this is not the staged protocol's formal test entrypoint. | `force_aware_act`; limited ACT fallback | checkpoint, stats, HDF5/list | CSV and optional ranked cases | no | no | optional |
| `evaluate_staged_checkpoints.py` | current | Hash-verify the complete Stage-2 checkpoint cadence, evaluate independent validation domains, enforce Stage-1 retention, and emit a non-overwriting shortlist. | all staged force-aware variants | candidate CSV, Stage-1 reference, protocol, stats, validation lists | long-form metrics, decisions, shortlist CSV/JSON, completion attestation | no | no | optional |
| `evaluate_staged_frozen_test.py` | current/formal | Evaluate the one SHA-pinned shortlist selection once on every protocol test domain using deterministic prior inference, episode-uniform metrics, and fixed episode bootstrap CIs. | contact-prior staged variants | protocol, completed shortlist report, stats | per-episode metrics, per-domain CIs, input/output hashes, completion attestation | no | no | optional |
| `evaluate_motion_cvae_modes.py` | current | Compare motion-CVAE zero vs posterior oracle. | `force_aware_motion_cvae` | checkpoint, stats, HDF5/list | per-sample CSV and printed summary | no | no | optional |
| `evaluate_contact_cvae_modes.py` | current | Compare contact-CVAE zero/prior/posterior oracle. | `force_aware_contact_cvae` | checkpoint, stats, HDF5/list | per-sample CSV and printed summary | no | no | optional |
| `evaluate_act_baseline_modes.py` | current | Compare ACT zero vs posterior motion oracle. | `act_baseline` | checkpoint, stats, HDF5/list | per-sample CSV and printed summary | no | no | optional |

Typical commands:

```bash
PYTHONPATH=src python scripts/evaluate_contact_cvae_modes.py --episode-list configs/splits/peg_hole_100_val10.txt --checkpoint outputs/contact_cvae/checkpoint_best.pt --normalization-stats outputs/stats.pt --action-mode action --output-csv outputs/contact_cvae/eval.csv
PYTHONPATH=src python scripts/evaluate_motion_cvae_modes.py --episode-list configs/splits/peg_hole_100_val10.txt --checkpoint outputs/motion_cvae/checkpoint_best.pt --normalization-stats outputs/stats.pt --posterior-mode mean
```

Limitations: posterior modes are oracle-only and non-deployable. `evaluate_motion_cvae_modes.py` uses `Path(episode_path).stem` for `episode_identifier`, so distinct files named `episode.hdf5` can collapse to `episode`. `evaluate_contact_cvae_modes.py` uses the parent episode directory name, with a stem fallback.

## Rollout and Simulation

| script | status | purpose | policies | inputs | outputs | modifies data | MuJoCo | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `run_mujoco_policy_rollout.py` | current | Single guarded MuJoCo rollout. | all checkpoint-dispatched policies | checkpoint, stats, XML | rollout CSV, summary JSON, snapshots/videos | no source/data change | yes | optional |
| `run_mujoco_hole_grid.py` | current | Batch grid/random/LHS/fixed-file hole-offset rollout wrapper. `--point-set-seed` controls generated points and `--rollout-seed-base` independently controls per-point rollout seeds; `--task-points-csv` is the exact point source when supplied. | all rollout-supported policies | checkpoint, stats, XML, optional point CSV | manifest, task CSV, grid summary, random summary | writes outputs | yes | optional |
| `run_xz_rollout_suite.py` | experiment-specific | Sequential x/z suite for Contact-CVAE zero/prior, Motion-CVAE, DualZero, and ACT baseline with mid/temporal selection. Defaults to 50 LHS points, ±6 mm, and 900 steps; it can instead forward a fixed point CSV. | five configured experiment variants | local checkpoints, stats, XML, optional point CSV | per-experiment grid outputs and safe-success target maps | writes outputs | yes | optional |
| `run_xz_multiseed_rollout_suite.py` | experiment-specific | Run `run_xz_rollout_suite.py` across independent point-set and rollout-seed dimensions, then aggregate task/safe-success rates, Wilson intervals, and run-level variation. `--point-set-seeds` and `--rollout-seed-bases` form a Cartesian product under isolated `pointset_<seed>/rollout_<seed>/` directories. Legacy `--seeds` remains supported. Defaults to `mid`; complete configurations are skipped and partial configurations resume through `--skip-existing`. | configured suite variants | seed lists, local checkpoints, stats, XML | seed directories, plan, per-seed and aggregate CSVs | writes outputs | yes | optional |
| `run_hole_random_5model_fibonacci_r60_rollouts.py` | experiment-specific/current | Sequentially evaluate the five `hole_random_60mm_hmj` early-stopping best checkpoints on the exact fixed 100-point 60 mm Fibonacci CSV with paired rollout seeds and deployment-correct latent modes. Supports preflight, dry-run, protocol locking, and point-level resume. | five best checkpoints | checkpoints, train90 stats, fixed points, XML | five grid directories, manifests, summaries, logs, suite plan | writes outputs | yes | optional |
| `monitor_hole_random_5model_fibonacci_r60_rollouts.py` | current diagnostic | Read-only watch monitor for the fixed 60 mm five-model suite: attempted/valid points, task and 40 N safe successes, process/force-stop errors, peak force, GPU state, current grid PID, and measured ETA. | fixed five-model suite | suite output root | terminal progress report | no | no | no |
| `run_hole_random_contact_cvae_action_sweep_r60_rollouts.py` | experiment-specific/current | Sequentially evaluates the two early-stopping Contact-CVAE best checkpoints with 1-based action chunk positions 1–10 and temporal aggregation on the paired fixed 100-point 60 mm Fibonacci set. Includes protocol locking and point-level resume. | Contact-CVAE action-selection sweep | two best checkpoints, train90 stats, fixed points, XML | 22 grid result directories, suite plan and logs | writes outputs | yes | optional |
| `monitor_hole_random_contact_cvae_action_sweep_r60_rollouts.py` | current diagnostic | Read-only monitor for the 22-experiment Contact-CVAE action sweep, including dynamic grid PID, per-mode success/safe-success, errors, force stops, peak force and measured ETA. | Contact-CVAE action-selection sweep | action-sweep output root | terminal progress report | no | no | no |
| `monitor_xz_rollout_suite.py` | diagnostic | Read-only progress monitor for an x/z multi-seed suite. Reports the active model, point-set seed, rollout-seed base, point index, completed configurations, partial work, and queued configurations. Uses `suite_plan.json` automatically or accepts the protocol flags manually for already-running legacy jobs. | n/a | suite output directory and optional protocol flags | terminal progress report | no | no | no |
| `monitor_dataset_scaling_rollout.py` | experiment-specific | Read-only monitor for the hard-coded dataset-scaling `mix50/mix100/mix150/mix203` rollout layout; can watch and report GPU/process state. | fixed scaling variants | scaling experiment root and expected point count | terminal progress report | no | no | no |
| `summarize_rollouts.py` | current | Aggregate rollout directories. | policy-agnostic | rollout dirs | summary CSV/table | writes summary | no | no |
| `plot_hole_grid_results.py` | current | Plot grid/LHS result heatmaps and scatters. | policy-agnostic | `grid_summary.csv` | PNG/PDF/etc and result tables | writes plots | no | no |
| `generate_fibonacci_disk_points.py` | current | Generate deterministic equal-area Fibonacci disk points in metres, with an optional diagnostic plot. | n/a | point count, radius, rotation | fixed point CSV, optional PNG | writes config/plot | no | no |
| `generate_random_disk_points.py` | current | Generate seeded area-uniform random disk points in metres with atomic CSV writing. | n/a | point count, radius, seed | fixed point CSV | writes config | no | no |

Typical commands:

```bash
PYTHONPATH=src python scripts/run_mujoco_policy_rollout.py --checkpoint outputs/model/checkpoint.pt --normalization-stats outputs/stats.pt --model-xml ../arm_teleop/model/pangu_all_right.xml --action-mode action --action-select-mode mid --output-dir outputs/rollout --execute-actions
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py --sampling-mode latin_hypercube --num-points 50 --checkpoint outputs/model/checkpoint.pt --normalization-stats outputs/stats.pt --model-xml ../arm_teleop/model/pangu_all_right.xml --output-root outputs/lhs --continue-on-error
PYTHONPATH=src python scripts/run_mujoco_hole_grid.py --sampling-mode file --task-points-csv configs/experiments/fibonacci_disk_100_r4mm.csv --checkpoint outputs/model/checkpoint.pt --normalization-stats outputs/stats.pt --model-xml ../arm_teleop/model/pangu_all_right.xml --output-root outputs/fixed_points --continue-on-error
python scripts/run_xz_rollout_suite.py --num-points 50 --offset-mm 6 --max-rollout-steps 900
python scripts/run_xz_multiseed_rollout_suite.py --seeds 20260702 20260703 20260704 20260705 20260706 --offset-mm 4 --output-base outputs/peg_hole_100/new_goal_multiseed
python scripts/run_xz_multiseed_rollout_suite.py --point-set-seeds 20260702 20260703 --rollout-seed-bases 31000 32000 --action-select-modes mid --offset-mm 4 --output-base outputs/peg_hole_100/separated_seed_rollouts
python scripts/monitor_xz_rollout_suite.py --output-base outputs/peg_hole_100/separated_seed_rollouts --watch --interval 10
```

For separated seeds, point `i` uses `rollout_seed_base + i - 1`. The same
point-set seed therefore produces identical task points across models and
rollout-seed bases, while each output directory records both seed dimensions.
New multi-seed runs write `suite_plan.json` before starting MuJoCo, allowing the
monitor to reconstruct the full queue from `--output-base` alone.

Key flags: `--contact-latent-mode`, `--action-select-mode`, `--temporal-agg-decay`, `--max-delta-q`, `--ema-alpha`, `--force-stop-threshold`, success thresholds, contact enter/exit/min-step thresholds, independent safe/hard-force thresholds, hole offset flags, `--save-videos`, `--skip-existing`, `--dry-run`, `--continue-on-error`. New rollout summaries and grid CSV/JSON reports preserve legacy task success while separately reporting recovery success, metric validity, contact duration/events, force-excess integral, and hard-force violations.

## Visualization and Analysis

| script | status | purpose | inputs | outputs | notes |
| --- | --- | --- | --- | --- | --- |
| `analyze_train_log.py` | diagnostic | Summarize/plot training CSV metrics. | train log CSV | printed summary, optional plot | handles available metric columns. |
| `plot_rollout_sensor_analysis.py` | diagnostic | Analyze force/success/action markers from rollout logs. | rollout dir(s) | plots and summary files | supports compare mode. |
| `analyze_contact_stage.py` | diagnostic | Analyze contact-stage behavior from rollout CSVs and demos. | rollout logs, optional HDF5 | CSV/plot | can use model XML for geometry context. |
| `analyze_contact_latent.py` | specialized | Analyze dual-latent posterior contact latents and optional prior overlay. | HDF5/list, checkpoint, stats | CSV/plots | dual-latent oriented. |
| `inspect_inference_case_predictions.py` | diagnostic | Inspect one saved inference case/prediction. | episode, state index, checkpoint, stats | output directory artifacts | contact-mode debugging. |
| `inspect_worst_case_episode.py` | diagnostic | Inspect signals around one HDF5 state index. | episode, state index | CSV/frames | read-only on HDF5. |
| `plot_hole_target_map.py` | current | Plot measured hole-position rollout outcomes as a target-style spatial map. | `grid_summary.csv` with `point_index`, `hole_offset_x`, `hole_offset_z`, `success`, optional `safe_success`, and `max_force` when overriding the threshold | PNG/PDF/SVG target maps | safe success is green, task success that is not safe is amber, and failure is red; `--safe-force-threshold N` recomputes safe success as `success AND max_force < N` without modifying the CSV. |
| `analyze_rollout_safety_threshold.py` | experiment-specific | Recompute safety labels and paired statistics for the fixed `mix50/mix100/mix150/mix203`, 100-point scaling layout. | completed scaling output root, force threshold | four CSV summaries/case tables | assumes exactly 100 summaries per model; optional SciPy supplies exact McNemar p-values. |

`plot_hole_target_map.py` converts hole offsets from metres to millimetres, centers the nominal hole at `(0, 0)`, draws light grey concentric rings at `--ring-step-mm` intervals, and uses identical circular markers with black edges for measured outcomes. When `safe_success` is available, safe successes are green, task successes that are not safe are amber, failures are red, and the title reports the safe-success count and rate. `--safe-force-threshold N` overrides the stored classification for plotting by applying the strict rule `success AND max_force < N`; the title and legend display the selected threshold. Historical CSVs without `safe_success` retain the original green-success/red-failure display and task-success title. It plots measured rollout samples only; it does not estimate or interpolate a continuous success region.

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
| `evaluate_dataset_quality.py` | current/specialized | Read-only batch quality gate for current command-labelled recordings: collection status, required schema, timing, force, motion, command tracking, and sampled image quality. | data directory or one HDF5 | per-episode CSV and aggregate JSON | stricter schema than `ContactForceHDF5Dataset`; tune thresholds and manually review warnings. |
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
| `update_hf_model_registry.py` | experiment-specific | Query configured Hugging Face repositories and regenerate pinned model-registry JSON/CSV/Markdown. | Requires network/Hugging Face access and hard-coded release definitions; mutates `docs/model_registry/*`, so run only for an intentional registry update. |

## General Limitations

- Several scripts assume repository root or insert `src` into `sys.path`.
- Some scripts use hard-coded default paths such as `../arm_teleop/model/pangu_all_right.xml` or `mujoco_data/...`.
- Generated outputs should remain under ignored output directories.
- Many diagnostic scripts are dual-latent-specific and should not be assumed to support every policy variant.
