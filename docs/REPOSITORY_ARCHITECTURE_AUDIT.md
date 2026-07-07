# Repository Architecture Audit

Audit snapshot date: 2026-07-07. This report distinguishes confirmed current behavior at that snapshot, known limitations, and future recommendations. It does not implement any recommendation.

## 1. Executive Summary

Confirmed current behavior: the repository implements four policy families: dual-latent `force_aware_act`, motion-only `force_aware_motion_cvae`, contact-only `force_aware_contact_cvae`, and force-free `act_baseline`. Data ingestion, normalization, training, offline evaluation, MuJoCo rollout, grid/LHS batch experiments, summaries, plotting, and tests are present.

Known limitation: model construction and checkpoint dispatch are duplicated across scripts, training has no resume CLI, force physical conventions are not validated, and grid runs cannot load an explicit saved task-point CSV.

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
  verification snapshot: 27 test files, 209 collected passing tests
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

Known limitation: no resume CLI; `train_minimal.py` hard-codes compact model settings; duplicated checkpoint helpers are imported by ACT trainer.

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

Confirmed current behavior: `run_mujoco_hole_grid.py` supports grid, random, and Latin hypercube sampling. It writes `task_points.csv`, `grid_manifest.json`, per-run outputs, `grid_summary.csv`, and `random_position_summary.json`; it supports `--dry-run`, `--skip-existing`, and `--continue-on-error`.

Known limitation: existing task-point CSV cannot be supplied as input.

## 10. Script Inventory Summary

Confirmed current behavior: 31 files exist under `scripts/`, covering training, normalization, offline evaluation, rollout, batch experiments, visualization, dataset inspection, model inspection, debugging, replay, geometry validation, compatibility, and experiment wrappers. Full details are in [SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md).

## 11. Test Inventory Summary

Verification snapshot on 2026-07-07: 27 test files collected 209 passing tests, covering model forward paths, losses, dataset behavior, normalization, checkpointing, evaluators, rollout helpers, grid/LHS utilities, plotting helpers, and parameter audits. Full details are in [TESTING.md](TESTING.md).

## 12. Checkpoint and Dispatch System

Confirmed current behavior: current checkpoints contain `model_state_dict`, `optimizer_state_dict`, `config`, and `step`. `config.policy_variant` controls construction. Strict loading is used for rollout and dedicated evaluators. Motion/contact evaluators also support raw state dicts. ACT legacy checkpoints without `act_baseline_version=motion_cvae_v1` are routed to a legacy class in rollout.

Known limitation: fallback to `force_aware_act` for missing policy variant can mask old metadata.

## 13. Documentation Inconsistencies Found

Confirmed current behavior: older docs predate `force_aware_contact_cvae` and sometimes describe `train_minimal.py` as supporting only dual-latent and motion-CVAE. Some command examples use `PYTHONPATH=src python` instead of the repository's `.venv/bin/python` convention. Historical docs also contain experiment-specific checkpoint paths that may not exist locally.

Action taken: canonical docs were rewritten; a current-status note was added to the older architecture audit. Historical experiment measurements were not changed.

## 14. Reproducibility Risks

Known limitations:

- No training resume.
- No central experiment manifest writer for all scripts.
- Hard-coded default paths to external MuJoCo XML/data locations.
- Grid point regeneration relies on unchanged code and seed.
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
- Add a task-point CSV input mode for grid/LHS experiments.
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
4. Add task-point CSV replay for paired grid comparisons.
5. Reduce duplicated evaluator metric code.
6. Keep historical experiment docs immutable except for clearly marked current-status notes.
