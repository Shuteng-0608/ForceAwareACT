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
  - peg-in-hole
library_name: pytorch
datasets:
  - shuteng0608/forceawareact-peg-hole-mujoco
license: null
model-index:
  - name: ForceAwareACT Motion CVAE for MuJoCo Peg-in-Hole
    results: []
---

# ForceAwareACT Motion CVAE for MuJoCo Peg-in-Hole

## Model Summary

This repository stores the 100k-step `force_aware_motion_cvae` checkpoint series for the ForceAwareACT MuJoCo peg-in-hole project. The model is a structurally motion-only force-aware CVAE policy: it consumes RGB observations, robot joint position, and a causal online force/torque history, predicts future joint-position command chunks, and also predicts future wrench chunks as an auxiliary supervised output.

The recommended checkpoint for evaluation or rollout is:

```text
checkpoints/checkpoint_step_00100000.pt
```

The associated dataset is `shuteng0608/forceawareact-peg-hole-mujoco` at revision `e6f60d7351d4992f0083028bee0efaceba64f5f2`.

## Model Architecture

Current implementation class: `ForceAwareACTMotionCVAEPolicy` in `src/force_aware_act/models/force_aware_motion_cvae_policy.py`.

The policy contains:

- ResNet18 visual encoder for multi-camera RGB images.
- Joint MLP for current 7-DoF joint position.
- Temporal force encoder for a causal historical 6-axis wrench window.
- Force-vision cross-attention fusion.
- Motion posterior encoder `q(z_motion | qpos, future_action_chunk)`.
- Transformer policy encoder and decoder with learned future queries.
- Action head for future action chunks.
- Force head for future wrench chunks.

It does not contain a contact latent, contact posterior, or contact conditional prior.

Policy encoder token order:

```text
visual_tokens, z_VF, z_q, z_F_online, z_motion
```

## ForceAware-Specific Components

The ForceAware path is active in the action policy. The online force window is encoded as `z_F_online`, fused with visual tokens into `z_VF`, and both tokens are included before the motion latent token. The force head predicts a future force chunk from the decoder hidden states with a zero auxiliary latent placeholder because this policy has no contact latent.

## Intended Inputs

Training inputs:

- `images`: `[B, N_cam, 3, 224, 224]`, cameras `ee_cam` and `base_top_cam`.
- `qpos`: `[B, 7]`.
- `force_window`: `[B, 20, 6]`, sampled causally over 0.25 seconds.
- `action_chunk`: `[B, 10, 7]`.
- `future_force_chunk`: `[B, 10, 6]`.

Deployment inputs:

- `images`, `qpos`, and `force_window` only.
- Deployment does not use future actions or future force labels.

## Predicted Outputs

- `pred_action`: `[B, 10, 7]`, normalized action chunk during training/evaluation.
- `pred_force`: `[B, 10, 6]`, normalized future force/wrench chunk.
- Training diagnostics include `mu_motion`, `logvar_motion`, and `z_motion`.

## Training Dataset

Dataset repository: `shuteng0608/forceawareact-peg-hole-mujoco`.

Dataset revision documented for long-term use:

```text
e6f60d7351d4992f0083028bee0efaceba64f5f2
```

The local dataset audit found a raw MuJoCo HDF5 archive with 100 episode directories, RGB camera streams, joint state, force/torque wrench streams, action command labels, and sidecar `metadata.json` files. The preserved training command refers to `outputs/peg_hole_100/all100.txt`; that exact episode-list file is not present in this macOS workspace, so the precise path contents should be verified from the Linux training record before claiming bitwise reproducibility.

## Training Configuration

| Field | Value | Evidence status |
| --- | --- | --- |
| Policy variant | `force_aware_motion_cvae` | Source-defined and experiment name |
| Training entry point | `scripts/train_minimal.py` | Source-defined |
| Model class | `ForceAwareACTMotionCVAEPolicy` | Source-defined |
| Action mode | `action` | Reconstructed from experiment naming and normalization filename |
| Max steps | `100000` | Experiment name and checkpoint series |
| Checkpoint interval | `10000` | Checkpoint series and source save schedule |
| Batch size | `16` | Reconstructed from paired preserved baseline command |
| Chunk length | `10` | Source default and preserved experiment pattern |
| Force window | `20` samples over `0.25` s | Current source default; verify original log |
| Image size | `224 224` | Current source default and preserved experiment pattern |
| Cameras | `ee_cam`, `base_top_cam` | Current source default and dataset |
| Normalization stats | `normalization_stats_action_all100.pt` | Preserved filename |
| Optimizer | `torch.optim.AdamW` | Source-defined |
| Learning rate | `1e-4` | Source default and preserved experiment pattern |
| Weight decay | PyTorch AdamW default, not CLI-configurable | Source relies on optimizer default |
| `beta_motion_max` | `5e-4` | Experiment name and preserved baseline command |
| `warmup_steps` | `2000` | Reconstructed from paired preserved baseline command |
| `lambda_force` | `0.1` | Current source default; verify original log |
| `lambda_prior` | `0.0` | Required by source for motion-only CVAE |
| `prior_loss_mode` | `mse_mu` | Source default; inactive for this policy |
| Random seed | Not configured by current trainer | Unverified |
| Device | `cuda` | Preserved experiment pattern |
| Current audited source commit | `26d4b63` | Local repository state |
| Historical Linux training commit | Not verified | Requires original run metadata |

## Training Objective

The source loss is:

```text
loss_total = L1(pred_action, action_chunk)
           + lambda_force * L1(pred_force, future_force_chunk)
           + beta_motion * KL(q(z_motion | qpos, action_chunk) || N(0, I))
```

For the documented 100k run, `beta_motion` warms up linearly to `5e-4` over 2000 steps. `lambda_force` is documented as `0.1` from current source defaults unless the original Linux console log proves a different override.

## Checkpoint Organization

The Hugging Face repository intentionally stores only the numbered checkpoint series:

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
training/
  train_log.csv
  console.log
  checkpoint_training_metrics.csv
config/
  normalization_stats_action_all100.pt
  training_command.txt
```

The Linux-side checkpoint audit established that `checkpoint.pt` and `checkpoint_step_00100000.pt` both store `step=100000`; their model parameters, optimizer states, config dictionaries, and all other checkpoint fields are logically equal. Their different SHA-256 hashes and file sizes are caused only by PyTorch serialization-level differences. The unnumbered `checkpoint.pt` is therefore omitted from this Hugging Face repository as a logically redundant duplicate, not as a best checkpoint.

## Checkpoint Payload Structure

Current source writes the same envelope for numbered and final checkpoints:

```text
model_state_dict
optimizer_state_dict
config
step
```

No scheduler state, AMP scaler state, epoch counter, metric-best marker, random-number-generator state, or save-reason field is written by the current trainer.

## Normalization Requirements

Use `config/normalization_stats_action_all100.pt` with checkpoints from this repository. The training and evaluation code validates the stored normalization `action_mode` when present. ForceAware motion evaluation expects qpos, action, and force statistics.

## Force-Window Construction

The current dataset loader samples a fixed-length causal force window ending at the current state timestamp. With the documented configuration, it samples 20 wrench values over the preceding 0.25 seconds. The force window is normalized with `force_mean` and `force_std` from the normalization file before being passed to the model.

## Action Representation

The raw dataset contains action command arrays. With `--action-mode action`, the loader reads the root HDF5 `action` dataset and constructs a length-10 action chunk aligned to the current decision index. These labels are normalized before training.

## Inference and Evaluation Example

From a local clone of the ForceAwareACT repository:

```bash
PYTHONPATH=src python scripts/evaluate_motion_cvae_modes.py \
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
  --output-csv outputs/eval_motion_cvae_100k.csv
```

The deployable mode is zero motion latent. The evaluator also reports an offline posterior-oracle mode using future action labels; that posterior mode is not deployable.

## Training Command

The portable reconstructed command is stored in `config/training_command.txt` for the Hugging Face repository and mirrored in this source repository at `docs/huggingface_models/forceaware_motion_cvae_100k/training_command.txt`.

## Available Training Artifacts

- Numbered checkpoints every 10k steps.
- `training/train_log.csv`.
- `training/console.log`.
- `training/checkpoint_training_metrics.csv`.
- `config/normalization_stats_action_all100.pt`.
- `config/training_command.txt`.

## Evaluation Status

No rollout result is included in this card because this macOS workspace does not contain complete evaluation artifacts tied to all required conditions: exact checkpoint, policy variant, action-selection mode, temporal execution mode, test-hole set, number of rollout trials, task-success definition, and safe-success definition.

## Known Limitations

- This is a MuJoCo simulation policy, not evidence of physical-robot safety.
- Deployment uses zero `z_motion`; posterior oracle metrics are diagnostic only.
- The historical Linux source commit and exact console command require verification from the original run record.
- The checkpoint payload does not store RNG state, scheduler state, AMP scaler state, or a dataset revision field.
- Force coordinate-frame and preprocessing details should be checked against the dataset card before public claims.

## Out-of-Scope Uses

This model card does not support claims about real-world robot deployment, human safety, medical/industrial certification, or robustness outside the documented MuJoCo peg-in-hole setup.

## Reproducibility Notes

The current source code uses a one-based loop `for step in range(1, max_steps + 1)`. `checkpoint_step_00100000.pt` is saved immediately after the 100000th optimizer update. In a normal uninterrupted 100k run with `--save-every 10000`, the final unnumbered `checkpoint.pt` stores the same step and logically identical training state.

## License and Access

No public license is declared in this repository snapshot. Do not assume open redistribution rights without a separate license statement from the repository owner.

## Citation

No paper citation is provided in this repository snapshot.
