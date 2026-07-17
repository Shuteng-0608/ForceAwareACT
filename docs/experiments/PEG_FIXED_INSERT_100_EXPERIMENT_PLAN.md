# Peg Fixed Insert 100 Experiment Plan

## Goal

Train and evaluate a fresh ForceAwareACT model on the fixed-start
insertion-only dataset:

```text
mujoco_data/peg_hole_fixed_insertion
```

All outputs should go under:

```text
outputs/peg_fixed_insert_100
```

This experiment must not overwrite or reuse `outputs/peg100`.

## Dataset Status

The dataset has been verified externally:

- 100 `episode.hdf5` files
- `ContactForceHDF5Dataset` loads successfully
- dataset length with `chunk_len=10` and `force_window_len=20`: `35117`
- sample keys include `images`, `qpos`, `qvel`, `joint_torque`, `ee_pose`,
  `force_window`, `action_chunk`, and `future_force_chunk`
- expected sample shapes:
  - `images`: `[2, 3, 224, 224]`
  - `qpos`: `[7]`
  - `force_window`: `[20, 6]`
  - `action_chunk`: `[10, 7]`
  - `future_force_chunk`: `[10, 6]`

The shell driver includes a smoke test that rechecks the episode count,
dataset length, sample keys, and tensor shapes before training.

## Existing Peg100 Baseline Configuration

The previous peg100 experiment used:

- split files: `configs/splits/peg_in_hole_100_train80.txt`,
  `peg_in_hole_100_val10.txt`, `peg_in_hole_100_test10.txt`
- normalization stats:
  `outputs/peg100/normalization_stats_train80.pt`
- Stage 1 checkpoint:
  `outputs/peg100/stage1/checkpoint.pt`
- Stage 2 checkpoint:
  `outputs/peg100/stage2/checkpoint.pt`

From the saved checkpoint configs:

| Setting | Peg100 value |
|---|---:|
| train episodes | 80 |
| `chunk_len` | 10 |
| `force_window_len` | 20 |
| `force_window_duration` | 0.25 |
| image size | 224 x 224 |
| cameras | `ee_cam`, `base_top_cam` |
| batch size | 4 |
| Stage 1 steps | 10000 |
| Stage 2 steps | 10000 |
| learning rate | `1e-4` |
| `lambda_force` | 0.1 |
| Stage 1 `lambda_prior` | 0.1 |
| prior loss mode | `mse_mu` |
| model `d_model` | 128 |
| model `z_dim` | 16 |
| model heads | 4 |
| encoder/decoder layers | 1 / 1 |
| feedforward size | 256 |

The fixed-insert experiment mirrors these settings unless explicitly
overridden by environment variables in the shell script.

## Proposed Split

The script creates deterministic split files inside the new output directory:

```text
outputs/peg_fixed_insert_100/splits/train80.txt
outputs/peg_fixed_insert_100/splits/val10.txt
outputs/peg_fixed_insert_100/splits/test10.txt
outputs/peg_fixed_insert_100/splits/all100.txt
```

The split rule is simple and reproducible:

- sort all `episode.hdf5` paths lexicographically
- first 80 episodes: train
- next 10 episodes: validation
- last 10 episodes: test

Because this dataset is fixed-start and controlled, lexicographic splitting is
acceptable for the first pass. If collection order contains hidden temporal
or operator drift, create a shuffled split file later and record the seed.

## Pipeline

Use the driver:

```bash
scripts/run_peg_fixed_insert_100_experiment.sh smoke
scripts/run_peg_fixed_insert_100_experiment.sh prepare
scripts/run_peg_fixed_insert_100_experiment.sh train
scripts/run_peg_fixed_insert_100_experiment.sh eval
scripts/run_peg_fixed_insert_100_experiment.sh rollout
```

For the complete sequence:

```bash
scripts/run_peg_fixed_insert_100_experiment.sh all
```

The default stage is `smoke` so accidentally running the script does not start
long training.

## Stage Details

### Smoke

The smoke stage:

1. finds all local `episode.hdf5` files,
2. creates deterministic split files under `outputs/peg_fixed_insert_100`,
3. runs `ContactForceHDF5Dataset`,
4. checks expected sample shapes,
5. runs `scripts/inspect_episode_collection.py` in tolerant read-only mode.

### Prepare

The prepare stage computes train80 normalization statistics:

```text
outputs/peg_fixed_insert_100/normalization_stats_train80.pt
```

### Stage 1

Stage 1 uses `scripts/train_minimal.py` on train80 with:

- `chunk_len=10`
- `force_window_len=20`
- `force_window_duration=0.25`
- `batch_size=4`
- `max_steps=10000`
- `lambda_prior=0.1`
- `prior_loss_mode=mse_mu`

Outputs:

```text
outputs/peg_fixed_insert_100/stage1/checkpoint.pt
outputs/peg_fixed_insert_100/stage1/train_log.csv
```

### Stage 2

Stage 2 uses `scripts/train_contact_prior_stage2.py` and freezes everything
except `contact_prior`.

Outputs:

```text
outputs/peg_fixed_insert_100/stage2/checkpoint.pt
outputs/peg_fixed_insert_100/stage2/train_log.csv
```

### Offline Evaluation

The evaluation stage runs `scripts/evaluate_inference_modes.py` on:

- train80
- val10
- test10

It writes one aggregate CSV per split under `outputs/peg_fixed_insert_100/stage2`.

### MuJoCo Rollouts

The rollout stage runs deployable closed-loop MuJoCo policies with no axial
push and no task-space bias:

- prior + first
- prior + mid
- prior + last
- zero + mid

Each rollout uses:

- checkpoint: `outputs/peg_fixed_insert_100/stage2/checkpoint.pt`
- normalization stats: `outputs/peg_fixed_insert_100/normalization_stats_train80.pt`
- model XML: `../arm_teleop/model/pangu_all_right.xml`
- policy rate: 30 Hz
- EMA alpha: 0.2
- max joint delta: 0.02
- force stop threshold: 20
- hole axis: `0 -1 0`
- `--execute-actions`
- `--save-videos`

No rollout command includes `--enable-axial-push`.

## Primary Questions

1. Does fixed-start insertion-only data make prior inference more decisive
   near the hole?
2. Is `prior + mid` still the best closed-loop action selection mode?
3. Does `prior + last` remain too aggressive on the controlled dataset?
4. Does the prior improve force prediction over zero baseline offline?
5. Do videos show actual insertion/contact rather than alignment-only stopping?

## Expected Artifacts

Generated locally, ignored by git:

```text
outputs/peg_fixed_insert_100/splits/*.txt
outputs/peg_fixed_insert_100/normalization_stats_train80.pt
outputs/peg_fixed_insert_100/stage1/checkpoint.pt
outputs/peg_fixed_insert_100/stage1/train_log.csv
outputs/peg_fixed_insert_100/stage2/checkpoint.pt
outputs/peg_fixed_insert_100/stage2/train_log.csv
outputs/peg_fixed_insert_100/stage2/inference_eval_*.csv
outputs/peg_fixed_insert_100/rollouts/*/rollout_log.csv
outputs/peg_fixed_insert_100/rollouts/*/videos/*.mp4
```

Do not commit HDF5 files, checkpoints, CSV logs, videos, plots, or any files
under `outputs/`.

## Notes

- The current policy still predicts internal MuJoCo joint-position targets.
- Rollout action commands are written directly to MuJoCo actuator targets.
- `ee_pose` is not consumed by the current model.
- Evaluation is offline imitation-style evaluation plus MuJoCo smoke rollout,
  not a final success-rate benchmark.
