# ACT-aligned `z_motion` latent-collapse control

## Question

The contact-CVAE run reached 24,000 optimizer steps but showed almost no
posterior/deployment prediction gap and near-zero across-sample posterior-mean
variance. This control asks whether the same training/data pipeline also
collapses when the contact posterior is replaced by the official ACT
action-only motion posterior.

This is a diagnostic control, not a replacement of the contact model.
Checkpoints and logs use separate architecture/training versions and an
independent output directory.

## Controlled architecture

The implementation is
`ACTAlignedMotionCVAEControlPolicy` in
`src/force_aware_act/models/act_aligned/motion_policy.py`.

The following remain identical to the ACT-aligned contact run:

- two normalized `224 x 224` camera images;
- normalized current `qpos [B, 7]`;
- normalized causal force history `[B, 20, 6]`;
- ResNet-18 visual backbone;
- 512-wide, 8-head, 4-layer online-force encoder;
- force/vision cross-attention;
- 4-layer policy encoder and 7-layer query decoder;
- 100 action queries;
- parallel action `[B, 100, 7]` and force `[B, 100, 6]` heads;
- dataset split, normalization statistics, batch size, seed, AdamW
  hyperparameters, and validation cadence.

The isolated latent path follows official ACT:

1. `qpos [B, 7] -> [B, 1, 512]`.
2. `action_chunk [B, 100, 7] -> [B, 100, 512]`.
3. Assemble `[CLS, qpos, action_0 ... action_99]`, shape
   `[B, 102, 512]`.
4. Apply the 4-layer ACT-style posterior Transformer encoder.
5. Read the CLS output `[B, 512]`.
6. Project to `mu_motion [B, 32]` and `logvar_motion [B, 32]`.
7. During training, sample
   `z_motion = mu + exp(0.5 * logvar) * epsilon`.
8. Project `z_motion [B, 32] -> [B, 1, 512]` and place it in the same
   policy-memory slot previously occupied by `z_contact`.
9. During deployment, set `z_motion` to an exact zero tensor `[B, 32]`.

There is no future-force argument in the motion posterior or in the policy
training API. There is no learned prior module or prior-matching loss.

The control still predicts force and still consumes causal force history.
Therefore it isolates the latent mechanism while retaining the
ForceAwareACT task; it is intentionally not a strict force-free reproduction
of the official ACT policy.

The canonical contact model has 103,388,877 parameters and the motion control
has 102,303,373. After renaming the contact/motion latent module prefixes and
excluding the contact-only prior and future-force posterior adapter, all 405
shared state-dictionary entries have identical shapes. The parameter-count
difference is therefore confined to the intended latent-path removal.

## Objective and schedule

The objective is

```text
loss =
    1.0 * masked_action_L1
  + 1.0 * masked_force_L1
  + 10.0 * KL(q_motion || N(0, I))
```

The KL is summed over 32 latent dimensions and averaged over the batch, as in
the existing ACT-aligned standard-normal KL implementation. There is no KL
warm-up, free-bits threshold, conditional prior, or prior distillation.

The canonical full schedule remains 24,000 optimizer steps. The first
diagnostic run is intentionally capped at 6,920 steps: two complete epochs on
the current 90-episode training split (`3,460` batches per epoch at batch size
8). This gives comparable validation points at step 3,460 and step 6,920
without immediately spending the full training budget.

## Server command

Use a new output directory:

```bash
mkdir -p runs/act_aligned_wst_motion_control_b8_seed0_6920

PYTHONPATH=src python scripts/train_act_aligned_motion_cvae_control.py \
  mujoco_data/peg_hole_wst \
  --output-dir runs/act_aligned_wst_motion_control_b8_seed0_6920 \
  --device cuda \
  --batch-size 8 \
  --num-workers 2 \
  --seed 0 \
  --run-mode burn_in \
  --max-train-steps 6920 \
  --official-reference-epochs 2000 \
  --checkpoint-interval-steps 2000 \
  --log-interval 10 \
  2>&1 | tee runs/act_aligned_wst_motion_control_b8_seed0_6920/console.log
```

Monitor it with the control-specific target:

```bash
python scripts/monitor_act_aligned_training.py \
  runs/act_aligned_wst_motion_control_b8_seed0_6920 \
  --target-steps 6920 \
  --watch \
  --interval 10
```

For a local end-to-end functional check:

```bash
PYTHONPATH=src python scripts/train_act_aligned_motion_cvae_control.py \
  mujoco_data/peg_hole_100 \
  --output-dir /tmp/forceact_motion_control_smoke \
  --device cpu \
  --num-workers 0 \
  --smoke \
  --run-mode burn_in \
  --max-train-steps 1 \
  --checkpoint-interval-steps 1 \
  --log-interval 1
```

## Decision signals

Inspect at least:

- `posterior_kl_standard`;
- `posterior_mean_abs`;
- `posterior_std_mean`;
- `posterior_mean_across_sample_variance`;
- `posterior_zero_action_delta`;
- `posterior_zero_force_delta`;
- posterior action/force L1 versus deployment-zero action/force L1.

If `z_motion` also converges to `mu ~= 0`, `std ~= 1`, negligible
across-sample mean variance, and negligible posterior/zero output deltas, the
evidence points toward a shared optimization or decoder-bypass issue rather
than a contact-specific posterior design problem.

If `z_motion` remains sample-dependent and materially improves posterior
reconstruction over zero deployment while `z_contact` collapses under the
matched pipeline, the evidence points toward the contact posterior/objective
or contact-label signal.

This comparison is diagnostic rather than causal proof: motion and contact
labels have different information content even when all surrounding training
conditions are controlled.
