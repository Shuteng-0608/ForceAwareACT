# Testing

Verification snapshot on 2026-07-07: the audited `tests/` tree contained 27 test files, and the verification run collected 209 passing tests.

## Commands

Full suite:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Focused policy tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_force_aware_act_policy.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_force_aware_motion_cvae_policy.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_force_aware_contact_cvae_policy.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_act_policy_baseline.py
```

Optional dependencies: most tests use `torch`, `numpy`, `h5py`, and `pytest`. Plot tests use `pandas`/matplotlib paths. MuJoCo helper tests use mocked or minimal geometry paths where possible, but real rollout execution requires `mujoco`.

## Test File Inventory

| file | subsystem | category | policies covered | main coverage | gaps |
| --- | --- | --- | --- | --- | --- |
| `test_act_baseline_checkpointing.py` | ACT training checkpoints | training pipeline | `act_baseline` | parser flags, checkpoint schedule, periodic saves | no long-run resume. |
| `test_act_policy_baseline.py` | ACT baseline model | unit/integration | `act_baseline` | no force/contact modules, posterior training, zero deploy, loss, checkpoint roundtrip, rollout dispatch | no real MuJoCo rollout. |
| `test_action_mode_pipeline.py` | action modes/checkpoints | integration | force-aware variants | dataset action modes, stats metadata, mismatch validation, schedule, checkpoint envelope | no real dataset scale test. |
| `test_analyze_contact_stage.py` | contact-stage analysis | unit | policy-agnostic | force/contact marker analysis and command deltas | synthetic data only. |
| `test_audit_model_components.py` | model audit | parameter audit | all major variants | component sums, no double count, motion/contact/ACT boundaries | `--policy-variant both` compares only `force_aware_act` and `act_baseline`; audit all four variants individually for complete coverage. |
| `test_contact_force_hdf5_dataset.py` | dataset reader | unit/integration | all dataset policies | shapes, timestamp alignment, action modes, mismatch tolerance, ImageNet normalization | no physical force convention validation. |
| `test_contact_prior_distillation_loss.py` | prior loss | unit | contact-capable | MSE/ KL modes, detach behavior, validation | no full training convergence. |
| `test_contact_prior_encoder.py` | contact prior | unit | contact-capable | shapes, visual summary, gradients, deterministic mu/logvar | no calibration metric. |
| `test_episode_paths.py` | path helpers | unit | all scripts | absolute, project-relative, list-relative resolution | no symlink policy tests. |
| `test_evaluate_act_baseline_modes.py` | ACT evaluator | evaluator/CLI | `act_baseline` | zero/posterior modes, metrics, checkpoint marker, CSV smoke | no legacy rollout behavior. |
| `test_evaluate_inference_modes.py` | dual-latent evaluator | evaluator | `force_aware_act` | ranked best/worst case sorting | limited end-to-end coverage here. |
| `test_evaluate_motion_cvae_modes.py` | motion evaluator | evaluator/CLI | `force_aware_motion_cvae` | zero/posterior modes, metrics, strict checkpoint dispatch, CSV smoke | no rollout metric correlation. |
| `test_force_aware_act_policy.py` | dual-latent model | unit/integration | `force_aware_act` | training/inference shapes, zero/prior/posterior validation, force head latent use, gradients | no real image backbone training. |
| `test_force_aware_contact_cvae_policy.py` | contact-only model | unit/integration/evaluator | `force_aware_contact_cvae` | no motion outputs, zero/prior deploy, posterior override, loss, gradients, dispatch, exports | no real MuJoCo rollout. |
| `test_force_aware_motion_cvae_policy.py` | motion-only model | unit | `force_aware_motion_cvae` | no contact modules, zero deploy, validation, force head zero aux, loss | no conditional prior by design. |
| `test_force_vision_cross_attention.py` | force-vision fusion | unit | force-aware | shapes, attention weights, projection, validation, gradients | no visual quality tests. |
| `test_hole_offset_and_grid.py` | hole offsets/grid/plots | rollout utility | rollout-supported | offset transforms, schema keys, dry-run manifest, LHS determinism, plots | no real subprocess rollout. |
| `test_mujoco_rollout_action_modes.py` | rollout action/safety helpers | rollout utility | rollout-supported | absolute/delta target interpretation, stats action-mode validation, success condition/schema | no full physics stepping. |
| `test_normalization.py` | normalization | unit | all dataset policies | stats shapes, normalize/denormalize, std floor | no serialized metadata check here. |
| `test_plot_rollout_sensor_analysis.py` | rollout sensor plots | visualization | policy-agnostic | marker extraction, success reconstruction, optional predicted-force columns, compare output | synthetic logs only. |
| `test_posterior_encoders.py` | posterior modules | unit | motion/contact latent policies | shapes, long-chunk rejection, KL, gradients | no latent disentanglement tests. |
| `test_prediction_heads.py` | heads | unit | force-aware and ACT | action/force head shapes, latent sensitivity, validation, gradients | no loss integration beyond unit. |
| `test_resnet18_vision_encoder.py` | vision encoder | unit | all visual policies | default/projection shapes, freeze behavior | no pretrained download test. |
| `test_summarize_rollouts.py` | rollout summarizer | unit | policy-agnostic | summary preference, sorting, CSV columns | no huge directory performance test. |
| `test_temporal_force_encoder.py` | force encoder | unit | force-aware | shapes, projected dim, long-window rejection, gradients | no filtering/frame tests. |
| `test_training_losses.py` | dual-latent loss | unit/integration | `force_aware_act` | weighted equation, zero latent KL disabling, prior loss, warmup, one-batch backward | variant losses covered in policy-specific files. |

## Verification Snapshot

The full suite was run after documentation edits with:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Verification snapshot on 2026-07-07: `209 passed, 14 warnings in 29.82s`. The warnings came from matplotlib/pyparsing deprecations in `tests/test_hole_offset_and_grid.py::test_heatmap_outputs_are_created`.

## Testing Conventions

- Prefer small synthetic tensors and temporary HDF5 files.
- Validate shape, error-path, dispatch, and gradient invariants.
- Keep policy-specific behavior in dedicated files.
- For a future policy variant, add tests for model forward modes, loss equation, checkpoint config/dispatch, offline evaluator modes, rollout dispatch, parameter audit grouping, and README command coverage.
