# Repository Architecture Audit

Audit snapshot date: 2026-07-16. This report distinguishes confirmed current behavior at that snapshot, known limitations, and future recommendations. Source and parser help were re-inspected after the recent dataset-quality, reproducibility, early-stopping, fixed-point, multi-seed, monitoring, and safety-analysis changes.

## 1. Executive Summary

Confirmed current behavior: the repository implements four policy families: dual-latent `force_aware_act`, motion-only `force_aware_motion_cvae`, contact-only `force_aware_contact_cvae`, and force-free `act_baseline`. Data ingestion, normalization, training, offline evaluation, MuJoCo rollout, grid/LHS batch experiments, summaries, plotting, and tests are present.

Known limitation: model construction and checkpoint dispatch are duplicated across scripts, training has no resume CLI, force physical conventions are not validated, metadata checks are incomplete, and controlled training seeds/threads are exposed by only one trainer.

Future recommendation: centralize policy registry/config dispatch, centralize normalization compatibility checks, add resumable training, and formalize force-frame metadata.

## 2. Repository Tree

```text
src/force_aware_act/
  data/contact_force_hdf5_dataset.py
  data/normalization.py
  models/act_policy.py
  models/policy.py
  models/force_aware_motion_cvae_policy.py
  models/force_aware_contact_cvae_policy.py
  models/{vision,force,cross_attention,posterior,contact_prior,heads}.py
  training/losses.py
  utils/episode_paths.py
scripts/
  training, normalization, evaluation, rollout, plotting, inspection, replay, wrappers
tests/
  36 test files (verification result recorded in TESTING.md)
docs/
  canonical docs plus historical reports
configs/
  split lists and experiment command notes
```

## 3. Model Inventory

Confirmed current behavior:

- `force_aware_act`: `ForceAwareACTPolicy`; dual-latent; online force; future force prediction; motion/contact posteriors; contact prior.
- `force_aware_motion_cvae`: `ForceAwareACTMotionCVAEPolicy`; structurally motion-only; online force; future force prediction; no contact latent/prior.
- `force_aware_contact_cvae`: `ForceAwareACTContactCVAEPolicy`; structurally contact-only; online force; future force prediction; contact posterior/prior; no motion latent.
- `act_baseline`: `ACTPolicyBaseline`; force-free; action-only head; motion posterior; zero motion latent at deployment.

Known limitation: rollout CLI uses shared `--contact-latent-mode` even for policies that ignore it.

## 4. Data Pipeline

Confirmed current behavior: `ContactForceHDF5Dataset` validates synchronized groups, aligns images by nearest timestamp, samples historical force windows using non-future force samples, constructs future action chunks, constructs future force chunks by nearest force timestamp to future state timestamps, and returns qpos/qvel/torque/ee_pose even though policies use only a subset.

Two complementary collection checks now exist. `inspect_episode_collection.py` exercises the general dataset reader and its one-frame tolerance. `evaluate_dataset_quality.py` is a stricter, current-recording quality gate that checks collection status, command tracking, timing, force/motion thresholds, and sampled image quality, but requires command-labelled `*_episode` fields not required by every legacy dataset.

Known limitation: force coordinate frame, sign convention, bias removal, filtering, gravity compensation, and dataset-vs-MuJoCo wrench equivalence are undocumented in data artifacts and not checked by code.

## 5. Normalization Pipeline

Confirmed current behavior: stats are feature-wise mean/std for qpos, actions, and force. Force stats combine historical and future force tensors. Stats are serialized by `torch.save` with metadata. Consumers validate `action_mode` when metadata is present.

Known limitation: compatibility checks are repeated in multiple scripts and do not validate every metadata field equally.

## 6. Training Pipeline

Confirmed current behavior:

- `train_minimal.py` supports `force_aware_act`, `force_aware_motion_cvae`, and `force_aware_contact_cvae`.
- `train_act_baseline.py` supports `act_baseline`.
- `train_contact_prior_stage2.py` supports dual-latent contact-prior distillation.
- Optimizer is AdamW.
- Beta values use `linear_warmup`.
- CSV logs and checkpoints are written.
- Optional periodic checkpoints are saved after optimizer steps.
- With a validation list, all three trainers support epoch accounting, deterministic deployment-path validation, best checkpoints, and relative-improvement early stopping.
- `train_minimal.py` supports explicit RNG seed, strict deterministic execution, CPU thread controls, DataLoader worker seeding, and initial-model fingerprinting.

Known limitation: no resume CLI; `train_minimal.py` hard-codes compact model settings; duplicated checkpoint helpers are imported by ACT trainer; ACT baseline and stage-2 do not expose the seed/deterministic/thread controls.

## 7. Offline Evaluation

Confirmed current behavior:

- Dual-latent evaluator compares zero, deterministic prior, and posterior oracle.
- Motion-CVAE evaluator compares zero and posterior mean/sample oracle.
- Contact-CVAE evaluator compares zero, deterministic prior, and posterior mean/sample oracle.
- ACT evaluator compares zero and posterior mean/sample oracle.

Known limitation: posterior oracle uses future labels and is not deployable. `evaluate_motion_cvae_modes.py` uses `Path(episode_path).stem` for `episode_identifier`, so distinct files named `episode.hdf5` can collapse to `episode`. `evaluate_contact_cvae_modes.py` uses the parent episode directory name, with a stem fallback.

## 8. Rollout and Deployment

Confirmed current behavior: rollout dispatches from `config.policy_variant`, loads stats, renders cameras, resamples force history, normalizes inputs, runs deployable inference, denormalizes action, chooses `first`/`mid`/`last`/`temporal`, applies delta interpretation, optional axial push, clipping, EMA, actuator clipping, success/force-stop checks, CSV logging, and summary JSON.

Known limitation: MuJoCo force sensor convention is not reconciled with dataset `ft_wrench`; no real-robot deployment scripts were modified or audited beyond repository-local rollout scripts.

## 9. Batch Experiment System

Confirmed current behavior: `run_mujoco_hole_grid.py` supports grid, random, Latin hypercube, and fixed CSV sampling. It writes `task_points.csv`, `grid_manifest.json`, per-run outputs, `grid_summary.csv`, and `random_position_summary.json`; it supports `--dry-run`, `--skip-existing`, and `--continue-on-error`. Point generation and rollout randomness use separate seeds. Fixed Fibonacci/random disk generators, a five-configuration suite, Cartesian-product multi-seed orchestration, read-only monitors, target plots, and threshold re-analysis are present.

Known limitation: a saved manifest cannot be supplied as input; `--skip-existing` trusts a readable `summary.json` and does not compare old protocol metadata with the new command. Several suite/monitor/analysis scripts are deliberately tied to local checkpoint names or the `mix50/mix100/mix150/mix203` layout.

## 10. Script Inventory Summary

Confirmed current behavior: 43 Python files exist directly under `scripts/`, covering training, normalization, offline evaluation, rollout, batch experiments, point generation, visualization, dataset inspection/quality, model inspection, debugging, replay, geometry validation, monitoring, release-registry maintenance, compatibility, and experiment wrappers. Full details are in [SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md).

## 11. Test Inventory Summary

The current tree contains 36 `test_*.py` files, including new coverage for dataset quality, split control, reproducible/thread-controlled training, early stopping, fixed point files, rollout suites/monitoring, target maps, and sensor analysis. The actual 2026-07-16 verification result is recorded in [TESTING.md](TESTING.md).

## 12. Checkpoint and Dispatch System

Confirmed current behavior: current checkpoints contain `model_state_dict`, `optimizer_state_dict`, `config`, and `step`; current trainers also record epoch/early-stop/stop-reason fields. `train_minimal.py` adds seed, deterministic, thread, and initial-model fingerprint metadata. `config.policy_variant` controls construction. Strict loading is used for rollout and dedicated evaluators. Motion/contact evaluators also support raw state dicts. ACT legacy checkpoints without `act_baseline_version=motion_cvae_v1` are routed to a legacy class in rollout.

Known limitation: fallback to `force_aware_act` for missing policy variant can mask old metadata.

## 13. Documentation Inconsistencies Found

Confirmed current behavior: older docs predate `force_aware_contact_cvae`, fixed point CSV replay, separated seeds, or early stopping. Historical docs also contain experiment-specific checkpoint paths that may not exist locally. The repository does not contain a `.venv/bin/python`; current docs therefore use the interpreter from an explicitly activated environment.

Action taken: a documentation index now defines precedence; canonical architecture/data/training/rollout/script/test docs were updated. Historical experiment measurements and model cards were not changed.

## 14. Reproducibility Risks

Known limitations:

- No training resume.
- No central experiment manifest writer for all scripts.
- Hard-coded default paths to external MuJoCo XML/data locations.
- Only `train_minimal.py` has explicit training seed/thread controls.
- Stats metadata beyond action mode and conditional episode provenance is not uniformly enforced.
- Generated outputs are ignored but broad `*.csv`/`*.png` ignore patterns can hide useful small artifacts unless intentionally documented.
- Checkpoint configs may omit older metadata such as `train_latent_mode` or dropout.

## 15. Architecture Risks

Known limitations:

- Duplicated policy construction and checkpoint parsing.
- Duplicated normalization validation.
- Latent mode terminology differs between training, evaluation, and rollout.
- Force conventions are implicit.
- `qvel`, torque, and ee pose are loaded but not model inputs.
- `train_contact_prior_stage2.py` remains dual-latent-specific while newer contact-only policy has integrated prior loss support.

## 16. Deferred Refactors

Future recommendations:

- Add a central policy registry with constructors, required inputs, valid modes, and evaluator names.
- Add a shared checkpoint loader with explicit compatibility policy.
- Add a shared normalization metadata validator.
- Add optional training resume.
- Generalize experiment-specific suite/monitor/safety-analysis layouts through explicit manifests.
- Add force-frame metadata to stats/checkpoints and datasets.
- Add a shared experiment manifest writer.

## 17. Recommended Future Repository Structure

Future recommendation:

```text
src/force_aware_act/
  policies/registry.py
  checkpoints.py
  normalization_contract.py
  rollout/
  evaluation/
scripts/
  thin CLI wrappers only
docs/
  canonical/
  historical/
```

## 18. Recommended Future Development Priorities

1. Centralize policy/checkpoint dispatch.
2. Add training resume and manifest logging.
3. Formalize force convention metadata and validation.
4. Make all trainers expose the same seed/deterministic/thread contract.
5. Reduce duplicated evaluator metric code.
6. Keep historical experiment docs immutable except for clearly marked current-status notes.
