# ACT / Contact-CVAE training-sampling ablation

This protocol isolates model family and training-window sampling as a 2 x 2
experiment.  It does not change validation sampling: model selection always
uses deterministic dense-window physical-action validation.

## Frozen data contract

- data root: `/home/stw/ForceAwareACT_workspace/arm_teleop/data/moving_hole_fixed_peg`
- manifest: `configs/experiments/moving_hole_fixed_peg_scripted200_seed0.json`
- train / validation / untouched holdout: 80 / 20 / 100 episodes
- clean / collision balance: 40 / 40 in train, 10 / 10 in validation, 50 / 50 in holdout
- train windows: 50,536
- validation windows: 12,641
- batch size: 8
- random-sampling epoch: 80 samples = 10 optimizer steps
- dense data epoch: 50,536 samples = 6,317 optimizer steps

The common fixed budget is 63,170 optimizer steps.  This is the least common
multiple of 10 and 6,317, so it ends on an epoch boundary for all four arms:

- episodic-uniform arms: 6,317 sampling epochs;
- dense-window arms: 10 data epochs;
- each arm consumes 505,360 training samples, counting repeated random draws.

Do not compare epoch counts across sampling modes.

## Validation and checkpoint cadence

Dense arms validate once per data epoch: every 6,317 steps.  Episodic arms use
632 sampling epochs between validations: every 6,320 steps.  The three-step
difference is the smallest whole-random-epoch approximation.

The Contact-CVAE trainer overwrites `last.pt` only at validation or termination
and can additionally write periodic step checkpoints.  The commands below set
the periodic interval to the final horizon to avoid retaining ten large
optimizer checkpoints; `last.pt` still provides recovery at each validation.

All four arms select `best` using:

```text
deployment_zero_action_l1_physical
```

The 100-episode holdout remains untouched until all training and checkpoint
selection decisions are frozen.

## Commands

Run from the repository root in the `forceact` environment.

```bash
DATA_ROOT=/home/stw/ForceAwareACT_workspace/arm_teleop/data/moving_hole_fixed_peg
MANIFEST=configs/experiments/moving_hole_fixed_peg_scripted200_seed0.json
STEP_TARGET=63170
```

### 1. Official ACT, canonical episodic-uniform sampling

```bash
PYTHONPATH=src python scripts/train_official_act.py "$DATA_ROOT" \
  --experiment-manifest "$MANIFEST" \
  --output-dir outputs/moving_hole_fixed_peg_scripted200/sampling_ablation/official_act_episodic_seed0 \
  --device cuda \
  --batch-size 8 \
  --num-workers 0 \
  --seed 0 \
  --train-sampling-mode episodic_uniform \
  --target-optimizer-steps "$STEP_TARGET" \
  --checkpoint-interval-epochs 632 \
  --full-validation-interval-epochs 632 \
  --log-interval 100
```

### 2. Official ACT, dense-window sampling

```bash
PYTHONPATH=src python scripts/train_official_act.py "$DATA_ROOT" \
  --experiment-manifest "$MANIFEST" \
  --output-dir outputs/moving_hole_fixed_peg_scripted200/sampling_ablation/official_act_dense_seed0 \
  --device cuda \
  --batch-size 8 \
  --num-workers 0 \
  --seed 0 \
  --train-sampling-mode dense_windows \
  --target-optimizer-steps "$STEP_TARGET" \
  --checkpoint-interval-epochs 1 \
  --full-validation-interval-epochs 1 \
  --log-interval 100
```

### 3. Native-500-Hz Contact-CVAE, canonical dense-window sampling

```bash
PYTHONPATH=src python scripts/train_act_aligned_high_rate_contact_cvae.py "$DATA_ROOT" \
  --experiment-manifest "$MANIFEST" \
  --output-dir outputs/moving_hole_fixed_peg_scripted200/sampling_ablation/contact_cvae_dense_seed0 \
  --device cuda \
  --batch-size 8 \
  --num-workers 0 \
  --seed 0 \
  --train-sampling-mode dense_windows \
  --run-mode formal \
  --target-optimizer-steps "$STEP_TARGET" \
  --official-reference-epochs 2000 \
  --checkpoint-interval-steps "$STEP_TARGET" \
  --log-interval 100
```

### 4. Native-500-Hz Contact-CVAE, episodic-uniform sampling

```bash
PYTHONPATH=src python scripts/train_act_aligned_high_rate_contact_cvae.py "$DATA_ROOT" \
  --experiment-manifest "$MANIFEST" \
  --output-dir outputs/moving_hole_fixed_peg_scripted200/sampling_ablation/contact_cvae_episodic_seed0 \
  --device cuda \
  --batch-size 8 \
  --num-workers 0 \
  --seed 0 \
  --train-sampling-mode episodic_uniform \
  --run-mode formal \
  --target-optimizer-steps "$STEP_TARGET" \
  --official-reference-epochs 2000 \
  --checkpoint-interval-steps "$STEP_TARGET" \
  --log-interval 100
```

For the scripted-200 manifest, the episodic Contact-CVAE run records
`validation_interval_sampling_epochs=632` and
`validation_interval_optimizer_steps=6320`.  A different dataset or batch size
may produce a different automatically derived interval.

## Acceptance checks before holdout evaluation

For every arm:

1. the training summary reports `passed=true` and the requested global step;
2. checkpoint reload audit passes;
3. run metadata records the intended `train_sampling_mode`;
4. all losses and gradient norms remain finite;
5. validation contains exactly 12,641 full windows;
6. no holdout episode is read during normalization, training, validation, or
   checkpoint selection.

Report optimizer steps, consumed samples, sampling/data epochs, wall-clock
time, best validation physical L1, and final paired holdout metrics.  If seed 0
shows a meaningful interaction, repeat the full 2 x 2 matrix with additional
seeds rather than adding seeds to only the favored arm.
