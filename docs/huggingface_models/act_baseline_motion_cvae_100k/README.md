---
language:
  - en
tags:
  - robotics
  - mujoco
  - imitation-learning
  - act
  - cvae
  - peg-in-hole
  - baseline
library_name: pytorch
datasets:
  - shuteng0608/forceawareact-peg-hole-mujoco
# license: null
model-index:
  - name: ACT Baseline Motion CVAE for MuJoCo Peg-in-Hole
    results: []
---

# ACT Baseline Motion CVAE for MuJoCo Peg-in-Hole

## Model Summary

This repository stores the 100k-step `act_baseline` checkpoint series for the ForceAwareACT MuJoCo peg-in-hole project. The model is a structurally force-free ACT-style Motion-CVAE baseline: it consumes RGB observations and robot joint position, trains with a motion posterior over future action chunks, and deploys with an exact zero motion latent.

The recommended checkpoint for evaluation or rollout is:

```text
checkpoints/checkpoint_step_00100000.pt
```

The associated dataset is `shuteng0608/forceawareact-peg-hole-mujoco` at revision `e6f60d7351d4992f0083028bee0efaceba64f5f2`.

## Model Architecture

Current implementation class: `ACTPolicyBaseline` in `src/force_aware_act/models/act_policy.py`.

The policy contains:

- ResNet18 visual encoder for multi-camera RGB images.
- Joint MLP for current 7-DoF joint position.
- Motion posterior encoder `q(z_motion | qpos, future_action_chunk)`.
- Transformer policy encoder and decoder with learned future queries.
- Action head for future action chunks.

It does not instantiate or consume:

- force windows;
- temporal force encoder;
- force-vision fusion;
- force-derived policy tokens;
- force head;
- future force supervision;
- contact latent modules;
- contact conditional prior.

Policy encoder token order:

```text
visual_tokens, z_q, z_motion
```

## Intended Inputs

Training inputs:

- `images`: `[B, N_cam, 3, 224, 224]`, cameras `ee_cam` and `base_top_cam`.
- `qpos`: `[B, 7]`.
- `action_chunk`: `[B, 10, 7]`.

Deployment inputs:

- `images` and `qpos` only.
- Deployment does not use future actions, force history, or future force labels.

## Predicted Outputs

- `pred_action`: `[B, 10, 7]`, normalized action chunk during training/evaluation.
- Training diagnostics include `mu_motion`, `logvar_motion`, and `z_motion`.

The ACT baseline does not predict force.

## Training Dataset

Dataset repository: `shuteng0608/forceawareact-peg-hole-mujoco`.

Dataset revision documented for long-term use:

```text
e6f60d7351d4992f0083028bee0efaceba64f5f2
```

The local dataset audit found a raw MuJoCo HDF5 archive with 100 episode directories, RGB camera streams, joint state, force/torque streams, action command labels, and sidecar `metadata.json` files. The baseline trainer builds `ContactForceHDF5Dataset(..., include_force=False)`, so force data may exist in the dataset but is not consumed by the model.

## Training Configuration

| Field | Value | Evidence status |
| --- | --- | --- |
| Policy variant | `act_baseline` | Source-defined |
| Baseline version | `motion_cvae_v1` | Source-defined |
| Training entry point | `scripts/train_act_baseline.py` | Source-defined |
| Model class | `ACTPolicyBaseline` | Source-defined |
| Action mode | `action` | Preserved experiment command |
| Max steps | `100000` | Preserved experiment command and checkpoint series |
| Checkpoint interval | `10000` | Preserved experiment command and checkpoint series |
| Batch size | `16` | Preserved experiment command |
| Chunk length | `10` | Preserved experiment command |
| Force window | Not used by model or trainer | Source-defined |
| Image size | `224 224` | Preserved experiment command |
| Cameras | `ee_cam`, `base_top_cam` | Preserved experiment command |
| Normalization stats | `normalization_stats_action_all100.pt` | Preserved experiment command |
| Optimizer | `torch.optim.AdamW` | Source-defined |
| Learning rate | `1e-4` | Preserved experiment command |
| Weight decay | PyTorch AdamW default, not CLI-configurable | Source relies on optimizer default |
| `beta_motion_max` | `5e-4` | Preserved experiment command |
| `warmup_steps` | `2000` | Preserved experiment command |
| Force-loss coefficient | Not applicable | Source-defined |
| Random seed | Not configured by current trainer | Unverified |
| Device | `cuda` | Preserved experiment command |
| Current audited source commit | `26d4b63` | Local repository state |
| Historical Linux training commit | Not verified | Requires original run metadata |

## Training Objective

The source loss is:

```text
loss_total = L1(pred_action, action_chunk)
           + beta_motion * KL(q(z_motion | qpos, action_chunk) || N(0, I))
```

For the documented 100k run, `beta_motion` warms up linearly to `5e-4` over 2000 steps.

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

Use `config/normalization_stats_action_all100.pt` with checkpoints from this repository. The ACT baseline evaluator expects qpos and action statistics. It does not require force statistics for model inputs.

## Action Representation

With `--action-mode action`, the loader reads the root HDF5 `action` dataset and constructs a length-10 action chunk aligned to the current decision index. These labels are normalized before training.

## Inference and Evaluation Example

From a local clone of the ForceAwareACT repository:

```bash
PYTHONPATH=src python scripts/evaluate_act_baseline_modes.py \
  --episode-list outputs/peg_hole_100/all100.txt \
  --checkpoint checkpoints/checkpoint_step_00100000.pt \
  --normalization-stats config/normalization_stats_action_all100.pt \
  --action-mode action \
  --chunk-len 10 \
  --image-size 224 224 \
  --camera-names ee_cam base_top_cam \
  --batch-size 16 \
  --device cuda \
  --output-csv outputs/eval_act_baseline_motion_cvae_100k.csv
```

The deployable mode is zero motion latent. The evaluator also reports an offline posterior-oracle mode using future action labels; that posterior mode is not deployable.

## Training Command

The portable reconstructed command is stored in `config/training_command.txt` for the Hugging Face repository and mirrored in this source repository at `docs/huggingface_models/act_baseline_motion_cvae_100k/training_command.txt`.

## Available Training Artifacts

- Numbered checkpoints every 10k steps.
- `training/train_log.csv`.
- `training/console.log`.
- `config/normalization_stats_action_all100.pt`.
- `config/training_command.txt`.

## Evaluation Status

No rollout result is included in this card because this macOS workspace does not contain complete evaluation artifacts tied to all required conditions: exact checkpoint, policy variant, action-selection mode, temporal execution mode, test-hole set, number of rollout trials, task-success definition, and safe-success definition.

## Known Limitations

- This is a MuJoCo simulation policy, not evidence of physical-robot safety.
- Deployment uses zero `z_motion`; posterior oracle metrics are diagnostic only.
- The historical Linux source commit requires verification from the original run record.
- The checkpoint payload does not store RNG state, scheduler state, AMP scaler state, or a dataset revision field.
- This baseline is force-free by design; it is not expected to model contact force history.

## Out-of-Scope Uses

This model card does not support claims about real-world robot deployment, human safety, medical/industrial certification, or robustness outside the documented MuJoCo peg-in-hole setup.

## Reproducibility Notes

The current source code uses a one-based loop `for step in range(1, max_steps + 1)`. `checkpoint_step_00100000.pt` is saved immediately after the 100000th optimizer update. In a normal uninterrupted 100k run with `--save-every 10000`, the final unnumbered `checkpoint.pt` stores the same step and logically identical training state.

## License and Access

No public license is declared in this repository snapshot. Do not assume open redistribution rights without a separate license statement from the repository owner.

## Citation

No paper citation is provided in this repository snapshot.
