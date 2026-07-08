---
language:
  - en
tags:
  - robotics
  - mujoco
  - imitation-learning
  - act
  - cvae
  - force-aware
  - contact-latent
  - peg-in-hole
  - ablation
library_name: pytorch
datasets:
  - shuteng0608/forceawareact-peg-hole-mujoco
model-index:
  - name: ForceAwareACT Contact CVAE for MuJoCo Peg-in-Hole
    results: []
---

# ForceAwareACT Contact CVAE for MuJoCo Peg-in-Hole

## Model Summary

This repository stores the 100k-step `forceaware_contact_cvae_betac5e4_lp01_trajectory100k` ablation model for the ForceAwareACT MuJoCo peg-in-hole project. The source-supported policy variant is `force_aware_contact_cvae`, implemented by `ForceAwareACTContactCVAEPolicy` in `src/force_aware_act/models/force_aware_contact_cvae_policy.py`.

Intended private Hugging Face model repository:

```text
shuteng0608/forceawareact-contact-cvae-betac5e4-lp01-peg-hole-100k
```

The model is structurally contact-only: it contains a contact posterior and conditional contact prior, but no motion latent module. It consumes online RGB, qpos, and force history, predicts future action chunks, and predicts future force chunks.

The recommended checkpoint is:

```text
checkpoints/checkpoint_step_00100000.pt
```

The associated dataset is `shuteng0608/forceawareact-peg-hole-mujoco` at revision `e6f60d7351d4992f0083028bee0efaceba64f5f2`.

## Architecture

Source-proven architecture:

- Policy variant: `force_aware_contact_cvae`.
- Python class: `ForceAwareACTContactCVAEPolicy`.
- Training script: `scripts/train_minimal.py`.
- Vision encoder: `ResNet18VisionEncoder`.
- State input: current 7-DoF joint position `qpos`.
- Force input: causal historical 6-axis wrench window.
- Force encoder: `TemporalForceEncoder`.
- Force-vision fusion: `ForceVisionCrossAttention`.
- Contact posterior: `ContactPosteriorEncoder`.
- Contact prior: `ContactPriorEncoder`.
- Motion latent: absent from this class.
- Action head: `ActionHead`.
- Auxiliary force head: `ForceHead`.

Policy encoder token order:

```text
visual_tokens, z_VF, z_q, z_F_online, z_contact
```

## Contact-Latent Design

Source-proven contact posterior inputs:

```text
qpos, future action chunk, future force chunk
```

Source-proven contact prior inputs:

```text
z_q, z_F_online, z_VF, visual token summary
```

The latent dimensionality in the current training construction is `z_dim=16`. During training, `contact_latent_mode="posterior"` is required; the model samples `z_contact` with the reparameterization trick. During deployment, supported modes are:

- `zero`: exact zero contact latent, deployable.
- `prior`: deterministic prior mean by default, deployable.

Offline posterior-oracle evaluation encodes the posterior separately and passes it through `contact_latent_override`; it is not deployable.

## Coefficient Meanings

The directory abbreviation `betac5e4` is reconstructed as `--beta-contact-max 5e-4`, which matches the current contact-CVAE training CLI and loss.

The abbreviation `lp01` is reconstructed as `--lambda-prior 0.1`, not `lambda_force`. In current source, `lambda_prior` weights the contact-prior distillation loss, while `lambda_force` separately weights future-force prediction loss and defaults to `0.1`.

## Intended Inputs and Outputs

Training inputs:

- `images`: `[B, N_cam, 3, 224, 224]`, cameras `ee_cam` and `base_top_cam`.
- `qpos`: `[B, 7]`.
- `force_window`: `[B, 20, 6]`, reconstructed from current defaults as 20 samples over 0.25 s.
- `action_chunk`: `[B, 10, 7]`.
- `future_force_chunk`: `[B, 10, 6]`.

Deployment inputs:

- RGB images, qpos, and force window only.
- `prior` mode uses only online features to estimate the contact prior.

Outputs:

- `pred_action`: future action chunk `[B, 10, 7]`.
- `pred_force`: auxiliary future wrench chunk `[B, 10, 6]`.
- Training diagnostics include `mu_contact`, `logvar_contact`, `mu_contact_prior`, and `logvar_contact_prior`.

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
| Experiment directory | `forceaware_contact_cvae_betac5e4_lp01_trajectory100k` | User-provided |
| Policy variant | `force_aware_contact_cvae` | Source-supported reconstruction |
| Policy class | `ForceAwareACTContactCVAEPolicy` | Source-proven |
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
| `lambda_prior` | `0.1` | Reconstructed from `lp01` and source loss |
| `prior_loss_mode` | `mse_mu` | Current source default |
| `beta_contact_max` | `5e-4` | Reconstructed from `betac5e4` and source CLI |
| `beta_motion_max` | inactive for contact-only policy | Source-proven |
| Warmup steps | `2000` | Reconstructed from paired 100k records |
| Random seed | Not configured by current trainer | Unresolved |

## Training Objective

Source-proven loss:

```text
loss_total = L1(pred_action, action_chunk)
           + lambda_force * L1(pred_force, future_force_chunk)
           + beta_contact * KL(q_contact || N(0, I))
           + lambda_prior * L_prior
```

With `prior_loss_mode=mse_mu`, `L_prior` is MSE between contact-prior mean and detached contact-posterior mean.

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
evaluation/
  contact_mode_eval_100k.csv
  contact_mode_eval_100k.log
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

The current dedicated Contact-CVAE evaluator compares zero, deterministic prior, and offline posterior-oracle modes:

```bash
PYTHONPATH=src python scripts/evaluate_contact_cvae_modes.py \
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
  --device cuda \
  --posterior-mode mean \
  --seed 0 \
  --output-csv evaluation/contact_mode_eval_100k.csv
```

The remote repository may include contact-mode evaluation artifacts under `evaluation/`. Numerical results are not summarized in this Model Card unless their complete evaluation protocol is available.

## Training Command

The reconstructed portable command is stored in `config/training_command.txt` for the Hugging Face repository and mirrored in this source repository at `docs/huggingface_models/forceaware_contact_cvae_100k/training_command.txt`.

## Known Limitations

- Historical Linux checkpoint contents and exact console command are unavailable in this workspace.
- The meanings of `betac5e4` and `lp01` are reconstructed from the current CLI/loss naming and should be verified against the original Linux command.
- Numerical evaluation artifacts are mentioned but not summarized because the CSV/log files are not present locally.
- This is a MuJoCo simulation model and is not evidence of physical-robot safety.

## Out-of-Scope Uses

Do not use this model card to claim physical-robot safety, real-world robustness, certification, or performance outside the documented MuJoCo peg-in-hole setup.

## Reproducibility Notes

The current training loop saves numbered checkpoints immediately after optimizer updates at scheduled one-based step numbers. For `checkpoint_step_00100000.pt`, the source convention is after the 100000th optimizer update.

## Access, License, and Citation

The intended Hugging Face repository is private. No public license or paper citation is declared in this repository snapshot.
