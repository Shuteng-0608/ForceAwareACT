---
language:
  - en
tags:
  - robotics
  - mujoco
  - imitation-learning
  - act
  - force-aware
  - peg-in-hole
  - ablation
library_name: pytorch
datasets:
  - shuteng0608/forceawareact-peg-hole-mujoco
model-index:
  - name: ForceAwareACT DualZero for MuJoCo Peg-in-Hole
    results: []
---

# ForceAwareACT DualZero for MuJoCo Peg-in-Hole

## Model Summary

This repository stores the 100k-step `forceaware_dualzero_trajectory100k` ablation model for the ForceAwareACT MuJoCo peg-in-hole project. The source-supported policy variant is `force_aware_act`, implemented by `ForceAwareACTPolicy` in `src/force_aware_act/models/policy.py`.

Intended private Hugging Face model repository:

```text
shuteng0608/forceawareact-dualzero-peg-hole-100k
```

DualZero is a training and deployment latent-mode ablation, not a separate no-latent architecture. The model still instantiates the full dual-latent ForceAwareACT architecture, including motion posterior, contact posterior, contact prior, motion latent projection, and contact latent projection modules. With `--train-latent-mode zero`, the training forward pass uses exact zero tensors for both `z_motion` and `z_contact`, bypasses posterior/prior computation, disables posterior KL losses, and disables contact-prior matching.

The recommended checkpoint is:

```text
checkpoints/checkpoint_step_00100000.pt
```

The associated dataset is `shuteng0608/forceawareact-peg-hole-mujoco` at revision `e6f60d7351d4992f0083028bee0efaceba64f5f2`.

## Architecture

Source-proven architecture:

- Policy variant: `force_aware_act`.
- Python class: `ForceAwareACTPolicy`.
- Training script: `scripts/train_minimal.py`.
- Vision encoder: `ResNet18VisionEncoder`.
- State input: current 7-DoF joint position `qpos`.
- Force input: causal historical 6-axis wrench window.
- Force encoder: `TemporalForceEncoder`.
- Force-vision fusion: `ForceVisionCrossAttention`.
- Latent modules present: motion posterior, contact posterior, contact conditional prior.
- Action head: `ActionHead`.
- Auxiliary force head: `ForceHead`.

Policy encoder token order:

```text
visual_tokens, z_VF, z_q, z_F_online, z_motion, z_contact
```

## Exact DualZero Definition

Source-proven training behavior:

- `--train-latent-mode zero` is passed to the full `force_aware_act` branch as `contact_latent_mode="zero"`.
- The model sets `z_motion = 0` and `z_contact = 0` during training.
- Motion posterior, contact posterior, and contact prior outputs are not produced in this forward path.
- `compute_force_aware_act_loss(..., use_posterior_kl=False)` sets `kl_motion = 0` and `kl_contact = 0`.
- `lambda_prior` is effectively disabled because the training loop sets `loss_lambda_prior = 0` when posterior latents are not used.
- Enabled losses are action L1 and `lambda_force * force L1`.
- Disabled losses are motion KL, contact KL, and contact-prior distillation.

Source-proven inference behavior:

- `ForceAwareACTPolicy` always sets `z_motion = 0` at inference.
- With `contact_latent_mode="zero"`, it also sets `z_contact = 0`.
- With `contact_latent_mode="prior"`, the same checkpoint can evaluate the conditional contact prior from online inputs. This is a different inference mode from the DualZero ablation setting.

The latent-related parameters remain in the checkpoint. They are not removed from the architecture.

## Intended Inputs and Outputs

Training inputs:

- `images`: `[B, N_cam, 3, 224, 224]`, cameras `ee_cam` and `base_top_cam`.
- `qpos`: `[B, 7]`.
- `force_window`: `[B, 20, 6]`, reconstructed from current defaults as 20 samples over 0.25 s.
- `action_chunk`: `[B, 10, 7]`.
- `future_force_chunk`: `[B, 10, 6]`.

Deployment inputs:

- RGB images, qpos, and force window only.
- No future action or future force labels are used during deployment.

Outputs:

- `pred_action`: future action chunk `[B, 10, 7]`.
- `pred_force`: auxiliary future wrench chunk `[B, 10, 6]`.

## Training Dataset

Dataset repository: `shuteng0608/forceawareact-peg-hole-mujoco`.

Dataset revision:

```text
e6f60d7351d4992f0083028bee0efaceba64f5f2
```

The local dataset audit found 100 MuJoCo HDF5 episode directories. The reconstructed command uses `outputs/peg_hole_100/all100.txt` and `normalization_stats_action_all100.pt`, but the original Linux episode-list contents are not available in this macOS workspace.

## Training Configuration

| Field | Value | Evidence status |
| --- | --- | --- |
| Experiment directory | `forceaware_dualzero_trajectory100k` | User-provided |
| Policy variant | `force_aware_act` | Source-supported reconstruction |
| Policy class | `ForceAwareACTPolicy` | Source-proven |
| Training entry point | `scripts/train_minimal.py` | Source-proven |
| Action mode | `action` | Reconstructed from paired 100k records |
| Max steps | `100000` | User-provided checkpoint series |
| Checkpoint interval | `10000` | User-provided checkpoint series |
| Batch size | `16` | Reconstructed from paired 100k records |
| Chunk length | `10` | Current source default and paired records |
| Force window | `20` samples, `0.25` s | Current source default |
| Cameras | `ee_cam`, `base_top_cam` | Current source default and dataset |
| Image size | `224 224` | Current source default |
| Normalization stats | `normalization_stats_action_all100.pt` | Reconstructed from paired records |
| Optimizer | `torch.optim.AdamW` | Source-proven |
| Learning rate | `1e-4` | Current source default |
| Weight decay | PyTorch AdamW default, not CLI-configurable | Source behavior |
| `lambda_force` | `0.1` | Current source default |
| `lambda_prior` | effectively `0.0` in zero-latent training | Source-proven |
| `beta_motion_max` | configured value irrelevant to loss | Source-proven |
| `beta_contact_max` | configured value irrelevant to loss | Source-proven |
| Warmup steps | `2000` | Reconstructed from paired 100k records |
| Random seed | Not configured by current trainer | Unresolved |

## Checkpoint Organization

Intended remote layout:

```text
README.md
checkpoints/
  checkpoint_step_00010000.pt
  checkpoint_step_00020000.pt
  checkpoint_step_00030000.pt
  checkpoint_step_00040000.pt
  checkpoint_step_00050000.pt
  checkpoint_step_00060000.pt
  checkpoint_step_00070000.pt
  checkpoint_step_00080000.pt
  checkpoint_step_00090000.pt
  checkpoint_step_00100000.pt
config/
  normalization_stats_action_all100.pt
  training_command.txt
training/
  train_log.csv
  console.log
```

The repository retains the explicitly numbered 100k checkpoint as the recommended model artifact. The unnumbered local `checkpoint.pt` is not used as the canonical remote filename. Logical equality between `checkpoint.pt` and `checkpoint_step_00100000.pt` for this specific experiment requires Linux-side verification.

## Checkpoint Payload Structure

Current source writes:

```text
model_state_dict
optimizer_state_dict
config
step
```

The checkpoint does not store scheduler state, AMP scaler state, RNG state, epoch, metric-best marker, or dataset revision.

## Normalization Requirements

Use `config/normalization_stats_action_all100.pt`. The evaluator validates `action_mode` metadata when present and expects qpos, action, and force normalization tensors.

## Evaluation Example

The current full ForceAware evaluator compares zero, prior, and offline posterior contact modes:

```bash
PYTHONPATH=src python scripts/evaluate_inference_modes.py \
  --episode-list outputs/peg_hole_100/all100.txt \
  --checkpoint checkpoints/checkpoint_step_00100000.pt \
  --normalization-stats config/normalization_stats_action_all100.pt \
  --action-mode action \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --image-size 224 224 \
  --camera-names ee_cam base_top_cam \
  --batch-size 16 \
  --max-batches 50 \
  --device cuda \
  --output-csv outputs/eval_dualzero_100k.csv
```

For the DualZero ablation setting, deployable zero mode is the relevant mode. Prior and posterior outputs are diagnostic comparisons and should not be described as the trained DualZero behavior.

## Training Command

The reconstructed portable command is stored in `config/training_command.txt` for the Hugging Face repository and mirrored in this source repository at `docs/huggingface_models/forceaware_dualzero_100k/training_command.txt`.

## Known Limitations

- Historical Linux checkpoint contents and exact console command are unavailable in this workspace.
- DualZero bypasses latent posterior/prior learning but does not remove latent modules from the architecture.
- Configured beta coefficients may appear in logs/config but do not affect the zero-latent loss path.
- This is a MuJoCo simulation model and is not evidence of physical-robot safety.

## Out-of-Scope Uses

Do not use this model card to claim physical-robot safety, real-world robustness, certification, or performance outside the documented MuJoCo peg-in-hole setup.

## Reproducibility Notes

The current training loop saves numbered checkpoints immediately after optimizer updates at scheduled one-based step numbers. For `checkpoint_step_00100000.pt`, the source convention is after the 100000th optimizer update.

## Access, License, and Citation

The intended Hugging Face repository is private. No public license or paper citation is declared in this repository snapshot.
