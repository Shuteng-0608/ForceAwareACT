# ACT Baseline Implementation

## Summary

This repository now includes a structurally force-free ACT-style zero-latent baseline, implemented as `ACTPolicyBaseline` in `src/force_aware_act/models/act_policy.py`. It is a separate policy class and does not modify `ForceAwareACTPolicy`, existing checkpoints, or existing ForceAwareACT training/rollout defaults.

## Retained Components

The ACT baseline retains:

- multi-camera RGB input with shape `[B, N_cam, 3, H, W]`;
- `ResNet18VisionEncoder`;
- `JointMLP` qpos encoder;
- a zero motion latent tensor `[B, z_dim]`;
- `motion_latent_proj`;
- policy Transformer encoder;
- policy Transformer decoder;
- learned `future_queries`;
- `ActionHead`;
- action chunk prediction `[B, chunk_len, action_dim]`;
- L1 action reconstruction loss only.

Matched successful-experiment settings are supported through config rather than hidden model constants:

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

## Removed Components

The ACT baseline structurally excludes:

- `force_window` model input;
- `TemporalForceEncoder`;
- `ForceVisionCrossAttention`;
- `z_F_online`;
- `z_VF`;
- force-derived policy tokens;
- `ForceHead`;
- `pred_force`;
- `future_force_chunk` supervision;
- `ContactPosteriorEncoder`;
- `ContactPriorEncoder`;
- `contact_latent_proj`;
- `z_contact`;
- contact KL;
- contact-prior distillation;
- future-force loss.

No ACT parameter name or module name contains force/contact prefixes, and the parameter audit reports zero parameters in force/contact groups.

## Token Sequence

The ACT policy encoder receives exactly:

```text
[ visual_tokens, z_q token, zero-motion-latent token ]
```

For two 224x224 cameras and `d_model=128`:

```text
images -> ResNet18VisionEncoder -> visual_tokens [B, 98, 128]
qpos   -> JointMLP             -> z_q           [B, 128]
zero z -> motion_latent_proj   -> z_motion_tok  [B, 128]
policy tokens                                 -> [B, 100, 128]
```

The decoder receives learned future queries:

```text
future_queries [1, 10, 128] -> decoder_hidden [B, 10, 128]
decoder_hidden -> ActionHead -> pred_action [B, 10, 7]
```

## Training and Inference Shapes

Training inputs consumed by ACT:

| Tensor | Shape | Use |
| --- | --- | --- |
| `images` | `[B, N_cam, 3, H, W]` | visual tokens |
| `qpos` | `[B, 7]` | qpos token |
| `action_chunk` | `[B, K, 7]` | L1 target |

Inference inputs consumed by ACT:

| Tensor | Shape | Use |
| --- | --- | --- |
| `images` | `[B, N_cam, 3, H, W]` | visual tokens |
| `qpos` | `[B, 7]` | qpos token |

`force_window` and `future_force_chunk` are not accepted by `ACTPolicyBaseline.forward`.

## Loss

The ACT-only loss is:

```python
loss_action = L1(pred_action, action_chunk)
loss_total = loss_action
```

It is implemented as `compute_act_baseline_loss` and returns only ACT metrics:

- `loss_total`;
- `loss_action`;
- `policy_variant = "act_baseline"`.

It does not call the ForceAwareACT loss with zero force weights.

## Training Entry Point

Use:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_act_baseline.py \
  test_data/episode.hdf5 \
  --device cpu \
  --max-steps 2 \
  --batch-size 1 \
  --output-dir outputs/act_baseline_smoke \
  --log-csv outputs/act_baseline_smoke/train_log.csv
```

The script uses `ContactForceHDF5Dataset(..., include_force=False)`, so ACT training batches do not need or consume force fields. The dataset default remains `include_force=True`, preserving ForceAwareACT behavior.

## Checkpoint Metadata

ACT checkpoints record:

```text
policy_variant = act_baseline
uses_force = false
uses_contact_latent = false
motion_latent_mode = zero
```

They also save architecture and preprocessing fields needed to reconstruct the model:

- camera names;
- image size;
- ImageNet normalization flag;
- action mode;
- chunk length;
- optimizer and learning rate;
- model config including `pretrained_resnet18`, `freeze_resnet18`, `d_model`, `z_dim`, `q_dim`, `action_dim`, layers, heads, feedforward width, and dropout.

## Evaluation and Rollout

`scripts/evaluate_inference_modes.py` and `scripts/run_mujoco_policy_rollout.py` inspect checkpoint metadata. If `policy_variant="act_baseline"`, they instantiate `ACTPolicyBaseline` and call it with only:

```python
model(images=images, qpos=qpos)
```

MuJoCo rollout still reads force/torque for safety stopping and logging, but does not provide force tensors to the ACT policy. Rollout controls remain shared:

- action denormalization;
- action interpretation;
- action selection mode;
- `max_delta_q`;
- EMA;
- actuator clipping;
- success criterion;
- force safety stop;
- initial pose;
- hole randomization;
- LHS/grid positions and seed.

`scripts/run_mujoco_hole_grid.py` delegates to `run_mujoco_policy_rollout.py`, so ACT-baseline checkpoints use the same grid/LHS command path.

## Parameter Counts

Using the matched synthetic config:

| Policy | Total params | Trainable params |
| --- | ---: | ---: |
| ForceAwareACTPolicy | 12,202,509 | 12,202,509 |
| ACTPolicyBaseline | 11,595,335 | 11,595,335 |
| Difference | 607,174 | 607,174 |

ACT force/contact groups are all zero:

- force temporal encoder: `0`;
- force-vision fusion: `0`;
- force head: `0`;
- contact latent/prior/posterior: `0`;
- other/unclassified: `0`.

## Why Not Zero Force Input

Zero-valued force input is not a true ACT baseline. In the ForceAwareACT architecture, the force encoder remains instantiated, its learned CLS token and positional embeddings remain active, linear/attention biases can produce nonzero activations, and force-derived tokens still enter the policy Transformer. It also leaves the model with larger capacity and different gradient paths. The ACT baseline therefore removes the force and contact structure entirely.

## Remaining Ambiguity

The exact documented 20k successful checkpoint is not present in this workspace. The implementation supports the documented matching settings, but final paper comparisons should verify the exact checkpoint metadata and rollout LHS manifest when those artifacts are available.
