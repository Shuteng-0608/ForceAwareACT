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
- batch size: 16
- random-sampling epoch: 80 samples = 5 optimizer steps
- dense data epoch: 50,536 samples = 3,159 optimizer steps

The shared maximum budget is 157,950 optimizer steps.  This is 50 dense data
epochs and 31,590 episodic sampling epochs:

- episodic-uniform arms: at most 31,590 sampling epochs;
- dense-window arms: at most 50 data epochs;
- episodic arms consume at most 2,527,200 sampled windows; dense arms consume
  2,526,800 windows because each data epoch ends with an eight-sample partial
  batch.  The optimizer-step budget differs by only 400 examples (0.016%).

This is a ceiling, not a target that every run must reach.  Validation-driven
early stopping is the primary termination mechanism.  Do not compare epoch
counts across sampling modes.

## Validation and checkpoint cadence

Dense arms validate once per data epoch: every 3,159 steps.  Episodic arms use
632 sampling epochs between validations: every 3,160 steps.  The one-step
difference is the smallest whole-random-epoch approximation.

All arms use the same convergence rule:

- do not stop before 31,590 optimizer steps (about ten validation intervals);
- require at least 0.5% relative improvement to reset patience;
- stop after five consecutive full validations without such an improvement;
- keep the checkpoint with the best
  `deployment_zero_action_l1_physical`, not the final checkpoint.

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
STEP_TARGET=157950
MINIMUM_STEPS=31590
PATIENCE=5
MIN_RELATIVE_IMPROVEMENT=0.005
```

### 1. Official ACT, canonical episodic-uniform sampling

```bash
PYTHONPATH=src python scripts/train_official_act.py "$DATA_ROOT" \
  --experiment-manifest "$MANIFEST" \
  --output-dir outputs/moving_hole_fixed_peg_scripted200/sampling_ablation_b16_earlystop/01_official_act_episodic_seed0 \
  --device cuda \
  --batch-size 16 \
  --num-workers 0 \
  --seed 0 \
  --train-sampling-mode episodic_uniform \
  --target-optimizer-steps "$STEP_TARGET" \
  --minimum-optimizer-steps "$MINIMUM_STEPS" \
  --early-stop-patience-validations "$PATIENCE" \
  --early-stop-min-relative-improvement "$MIN_RELATIVE_IMPROVEMENT" \
  --checkpoint-interval-epochs 632 \
  --full-validation-interval-epochs 632 \
  --log-interval 100
```

### 2. Official ACT, dense-window sampling

```bash
PYTHONPATH=src python scripts/train_official_act.py "$DATA_ROOT" \
  --experiment-manifest "$MANIFEST" \
  --output-dir outputs/moving_hole_fixed_peg_scripted200/sampling_ablation_b16_earlystop/02_official_act_dense_seed0 \
  --device cuda \
  --batch-size 16 \
  --num-workers 0 \
  --seed 0 \
  --train-sampling-mode dense_windows \
  --target-optimizer-steps "$STEP_TARGET" \
  --minimum-optimizer-steps "$MINIMUM_STEPS" \
  --early-stop-patience-validations "$PATIENCE" \
  --early-stop-min-relative-improvement "$MIN_RELATIVE_IMPROVEMENT" \
  --checkpoint-interval-epochs 1 \
  --full-validation-interval-epochs 1 \
  --log-interval 100
```

### 3. Native-500-Hz Contact-CVAE, canonical dense-window sampling

```bash
PYTHONPATH=src python scripts/train_act_aligned_high_rate_contact_cvae.py "$DATA_ROOT" \
  --experiment-manifest "$MANIFEST" \
  --output-dir outputs/moving_hole_fixed_peg_scripted200/sampling_ablation_b16_earlystop/03_contact_cvae_dense_seed0 \
  --device cuda \
  --batch-size 16 \
  --num-workers 0 \
  --seed 0 \
  --train-sampling-mode dense_windows \
  --run-mode formal \
  --target-optimizer-steps "$STEP_TARGET" \
  --minimum-optimizer-steps "$MINIMUM_STEPS" \
  --early-stop-patience-validations "$PATIENCE" \
  --early-stop-min-relative-improvement "$MIN_RELATIVE_IMPROVEMENT" \
  --official-reference-epochs 2000 \
  --checkpoint-interval-steps "$STEP_TARGET" \
  --log-interval 100
```

### 4. Native-500-Hz Contact-CVAE, episodic-uniform sampling

```bash
PYTHONPATH=src python scripts/train_act_aligned_high_rate_contact_cvae.py "$DATA_ROOT" \
  --experiment-manifest "$MANIFEST" \
  --output-dir outputs/moving_hole_fixed_peg_scripted200/sampling_ablation_b16_earlystop/04_contact_cvae_episodic_seed0 \
  --device cuda \
  --batch-size 16 \
  --num-workers 0 \
  --seed 0 \
  --train-sampling-mode episodic_uniform \
  --run-mode formal \
  --target-optimizer-steps "$STEP_TARGET" \
  --minimum-optimizer-steps "$MINIMUM_STEPS" \
  --early-stop-patience-validations "$PATIENCE" \
  --early-stop-min-relative-improvement "$MIN_RELATIVE_IMPROVEMENT" \
  --official-reference-epochs 2000 \
  --checkpoint-interval-steps "$STEP_TARGET" \
  --log-interval 100
```

For the scripted-200 manifest, the episodic Contact-CVAE run records
`validation_interval_sampling_epochs=632` and
`validation_interval_optimizer_steps=3160`.  A different dataset or batch size
may produce a different automatically derived interval.

## Completed seed-0 training audit

The four seed-0 runs above completed successfully and stopped on the common
plateau rule before the maximum budget.  These are the frozen checkpoints for
offline and rollout evaluation:

| arm | stop step | best step | best validation physical action L1 | checkpoint |
| --- | ---: | ---: | ---: | --- |
| Official ACT / episodic | 113,760 | 97,960 | 0.0023619740 | `01_official_act_episodic_seed0/best_policy.pt` |
| Official ACT / dense | 91,611 | 75,816 | 0.0025896759 | `02_official_act_dense_seed0/best_policy.pt` |
| Contact-CVAE / dense | 88,452 | 72,657 | 0.0026402107 | `03_contact_cvae_dense_seed0/best.pt` |
| Contact-CVAE / episodic | 60,040 | 44,240 | 0.0026680049 | `04_contact_cvae_episodic_seed0/best.pt` |

All four training summaries report `passed=true`; all best checkpoints pass a
strict reload audit and embed the expected 100-step chunk, 480 x 640 image,
split, normalization, and sampling contracts.  Run 4 was resumed after an
interruption, and its checkpoint lineage audit also passes.  These validation
numbers are useful for checkpoint selection only: they are not the final
holdout comparison.

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
