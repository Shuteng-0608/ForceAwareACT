# ACT-CVAE Baseline Implementation

## Summary

`ACTPolicyBaseline` in `src/force_aware_act/models/act_policy.py` is now a structurally force-free ACT-style Motion-CVAE baseline. The earlier baseline was useful as a zero-latent ablation, but it was not a standard ACT-CVAE comparison because it used an exact zero motion latent during both training and deployment and did not instantiate a motion posterior.

The corrected baseline trains with `q(z_motion | qpos_t, action_{t:t+K})`, deploys with exact `z_motion = 0`, and consumes only RGB images plus normalized qpos in the deployment path.

## Retained Components

- multi-camera RGB input `[B, N_cam, 3, H, W]`;
- `ResNet18VisionEncoder`;
- `JointMLP` qpos encoder;
- `MotionPosteriorEncoder` for training-only motion posterior inference;
- `motion_latent_proj`;
- policy Transformer encoder and decoder;
- learned `future_queries`;
- `ActionHead`;
- action chunk prediction `[B, chunk_len, action_dim]`.

## Removed Components

`ACTPolicyBaseline` does not instantiate or consume:

- `force_window`;
- `TemporalForceEncoder`;
- `ForceVisionCrossAttention`;
- `ForceHead` or `pred_force`;
- `future_force_chunk`;
- force-derived policy tokens;
- contact posterior, contact prior, or contact latent projection;
- contact KL or contact-prior distillation losses.

## Token Sequence

The force-free policy encoder receives:

```text
visual_tokens, z_q, z_motion
```

For the matched `224 x 224`, two-camera, `d_model=128`, `chunk_len=10` setting:

```text
images -> ResNet18VisionEncoder -> visual_tokens [B, 98, 128]
qpos   -> JointMLP             -> z_q           [B, 128]
z      -> motion_latent_proj   -> z_motion_tok  [B, 128]
policy tokens                                 -> [B, 100, 128]
future_queries                               -> [B, 10, 128]
decoder_hidden                               -> [B, 10, 128]
pred_action                                  -> [B, 10, 7]
```

## Forward API

```python
outputs = model(
    images,
    qpos,
    action_chunk=None,
    is_training=True,
    motion_latent_override=None,
)
```

Training requires `action_chunk` and returns `pred_action`, `z_motion`, `mu_motion`, `logvar_motion`, and `decoder_hidden`. Deployment requires `action_chunk=None`, uses exact zeros when no override is provided, and never reads future action labels. `motion_latent_override` is inference-only and enables zero-vs-posterior offline evaluation.

## Loss

The ACT baseline loss is:

```text
loss_total = loss_action + beta_motion * kl_motion
loss_action = L1(pred_action, action_chunk)
kl_motion = mean_batch(sum_latent KL[N(mu, sigma), N(0, I)])
```

`beta_motion` uses linear warmup from `0` to `beta_motion_max` during `warmup_steps`.

## Training

`scripts/train_act_baseline.py` uses `ContactForceHDF5Dataset(..., include_force=False)`, normalizes qpos/action chunks, calls the model with `is_training=True`, and optimizes action L1 plus motion KL. It logs `loss_total`, `loss_action`, `kl_motion`, `beta_motion`, `train_latent_mode=posterior`, `uses_posterior_latent=True`, `uses_zero_latent=False`, and `normalization_enabled`.

Example smoke command:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_act_baseline.py \
  test_data/episode.hdf5 \
  --device cpu \
  --max-steps 2 \
  --batch-size 1 \
  --output-dir /tmp/act_baseline_smoke \
  --log-csv /tmp/act_baseline_smoke/train_log.csv
```

Long-run trajectory checkpointing matches `scripts/train_minimal.py`: `--save-every 0`
disables periodic checkpoints, positive values save after the optimizer step at each
one-based interval, and filenames use `checkpoint_step_XXXXXXXX.pt` with eight
digits. The final `<output-dir>/checkpoint.pt` is always written, so a run whose
last step is divisible by `--save-every` writes both the final periodic checkpoint
and `checkpoint.pt`.

Example 100k command:

```bash
cd ~/ForceAwareACT_workspace/ForceAwareACT
conda activate forceact

OUT="outputs/peg_hole_100/act_baseline_motion_cvae_betam5e4_trajectory100k"
mkdir -p "$OUT"

PYTHONPATH=src python scripts/train_act_baseline.py \
  --episode-list outputs/peg_hole_100/all100.txt \
  --normalization-stats outputs/peg_hole_100/normalization_stats_action_all100.pt \
  --action-mode action \
  --chunk-len 10 \
  --image-size 224 224 \
  --camera-names ee_cam base_top_cam \
  --beta-motion-max 5e-4 \
  --warmup-steps 2000 \
  --max-steps 100000 \
  --save-every 10000 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --d-model 128 \
  --z-dim 16 \
  --nhead 4 \
  --num-encoder-layers 1 \
  --num-decoder-layers 1 \
  --dim-feedforward 256 \
  --dropout 0.0 \
  --device cuda \
  --output-dir "$OUT" \
  --log-csv "$OUT/train_log.csv" \
  2>&1 | tee "$OUT/console.log"
```

## Checkpoints and Legacy Behavior

New checkpoints include:

```text
policy_variant = act_baseline
act_baseline_version = motion_cvae_v1
uses_force = false
uses_contact_latent = false
motion_latent_mode = posterior_train_zero_deploy
train_latent_mode = posterior
```

The config also records model dimensions, vision settings, preprocessing settings, optimizer settings, `beta_motion_max`, `warmup_steps`, `save_every`, `save_steps`, and the resolved `intermediate_checkpoint_steps`.

Legacy `act_baseline` checkpoints without `act_baseline_version=motion_cvae_v1` are not loaded into the corrected CVAE class with `strict=False`. Rollout and offline inference instantiate `LegacyZeroLatentACTPolicyBaseline` for those checkpoints and load strictly, with a warning that the checkpoint is the legacy zero-latent baseline. The dedicated zero/posterior evaluator rejects legacy checkpoints because they have no motion posterior.

## Evaluation and Rollout

Use the zero/posterior evaluator:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_act_baseline_modes.py \
  --episode-list episodes.txt \
  --checkpoint outputs/act_baseline/checkpoints/latest.pt \
  --normalization-stats outputs/act_baseline/normalization_stats.pt \
  --action-mode action \
  --chunk-len 10 \
  --posterior-mode mean \
  --device cpu \
  --output-csv outputs/act_baseline/zero_vs_posterior.csv
```

MuJoCo rollout still reads force for safety stopping and logging, but does not pass force tensors to ACT. Action denormalization, action selection, `max_delta_q`, EMA, clipping, success logic, hole randomization, and LHS/grid command paths are shared with the force-aware policies.

## Matched Settings

The code supports matching the documented successful ForceAwareACT setting through config:

| Setting | Matched value |
| --- | --- |
| cameras | `ee_cam`, `base_top_cam` |
| image size | `224 x 224` |
| ImageNet normalization | disabled |
| ResNet18 pretrained | disabled |
| ResNet18 trainable | yes |
| qpos/action dim | `7 / 7` |
| chunk length | `10` |
| `d_model` | `128` |
| encoder/decoder layers | `1 / 1` |
| attention heads | `4` |
| dropout | `0.0` |
| optimizer/LR | AdamW, `1e-4` |
| loss | action L1 plus motion KL |

## Parameter Audit

With the matched synthetic config:

| Policy | Total params | Trainable params |
| --- | ---: | ---: |
| ForceAwareACTPolicy | 12,202,509 | 12,202,509 |
| ACTPolicyBaseline | 11,735,655 | 11,735,655 |
| Difference | 466,854 | 466,854 |

Corrected ACT-CVAE component groups:

| Component | Parameters |
| --- | ---: |
| vision backbone | 11,242,176 |
| state projection | 17,536 |
| Transformer encoder | 132,480 |
| Transformer decoder | 198,784 |
| action queries/head | 2,183 |
| motion latent modules | 142,496 |
| force temporal encoder | 0 |
| force-vision fusion | 0 |
| force head | 0 |
| contact latent/prior/posterior | 0 |
| other/unclassified | 0 |

## Why Not Zero Force Input

Zero-valued force input is not a true ACT baseline. In ForceAwareACT, force modules remain instantiated, learned tokens and biases can still produce nonzero activations, force-conditioned tokens still enter the policy Transformer, force-loss wiring may still exist depending on configuration, and parameter capacity remains larger. The corrected ACT-CVAE removes the force/contact structure entirely while retaining the ACT motion posterior comparison.

## Remaining Ambiguity

The exact documented 20k successful checkpoint is not present in this workspace. The implementation supports the reported matching settings, but final experiment commands should verify checkpoint metadata, dataset split, normalization stats, and LHS manifest on the Linux training/evaluation machine.
