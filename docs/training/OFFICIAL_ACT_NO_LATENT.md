# Official ACT NoLatent baseline

This document defines the force-free, structurally latent-free control paired
with the existing Official ACT baseline. Source code and CLI `--help` output
take precedence.

## Architecture contract

The model retains the Official ACT ResNet-18 camera layout, qpos projection,
policy encoder, query decoder, action head, pad head, image normalization and
action chunk length. It removes the complete CVAE mechanism:

| model | policy memory prefix | posterior | deployment latent | force input |
| --- | --- | --- | --- | --- |
| Official ACT | `z_motion, qpos` | action-conditioned Gaussian | exact zero | none |
| Official ACT NoLatent | `qpos` | none | none exists | none |

At the canonical resolution the memory length is 601 tokens: one qpos token
and 600 visual tokens. Passing a zero vector through the old latent projection
is deliberately not used, because its affine bias and position embedding would
remain learned constant tokens.

The architecture version is `official_act_single_arm_no_latent_v1`. Its config
does not contain `latent_dim`, and its policy has no registered module or
parameter containing `latent`, `posterior`, or `prior`.

## Data and loss

The trainer reuses the Official ACT episode split, per-episode timestep
sampling, qpos/action normalization, image loading and action padding mask. It
therefore forms a controlled comparison with Official ACT.

The only objective is the existing Official ACT masked action L1:

```text
L = lambda_action * mean(abs(pred_action - target) * non_padding)
```

There is no KL term and no sampled/posterior validation path. Checkpoint
selection uses `no_latent_validation_action_l1`.

This is a force-free baseline. It does not read low-rate or high-rate force and
does not require force normalization. Models that use force remain subject to
the native 500 Hz contract documented in
[`NATIVE_RATE_CONTROL_MODELS.md`](NATIVE_RATE_CONTROL_MODELS.md).

## Preflight and training

Run a real-episode CPU preflight before a long job:

```bash
PYTHONPATH=src python scripts/preflight_official_act_no_latent.py \
  mujoco_data/peg_hole_100 --device cpu --smoke \
  --output outputs/official_act_no_latent_preflight.json
```

Run a compact end-to-end smoke training:

```bash
PYTHONPATH=src python scripts/train_official_act_no_latent.py \
  mujoco_data/peg_hole_100 \
  --output-dir outputs/official_act_no_latent_smoke \
  --device cpu --num-workers 0 --smoke
```

Canonical training uses the same interface without `--smoke`:

```bash
PYTHONPATH=src python scripts/train_official_act_no_latent.py \
  mujoco_data/peg_hole_100 \
  --output-dir outputs/official_act_no_latent \
  --batch-size 8 --num-epochs 2000
```

Use `--experiment-manifest` for a paired comparison. Resume requires the same
model and training configuration recorded in the checkpoint:

```bash
PYTHONPATH=src python scripts/train_official_act_no_latent.py \
  mujoco_data/peg_hole_100 \
  --output-dir outputs/official_act_no_latent \
  --resume outputs/official_act_no_latent/latest.pt
```

The output directory contains JSONL metrics, run metadata, strict latest/final
checkpoints, a model-only best policy checkpoint and a reload-audited summary.

## Rollout contract

`run_mujoco_policy_rollout.py` dispatches this architecture as
`official_act_no_latent`. The shared rollout option remains
`--contact-latent-mode zero` for CLI compatibility, but diagnostics report:

```text
latent_name=none
latent_source=none
latent_max_abs=0
force_history_valid_samples=0
force_history_padding_samples=0
```

Inference accepts only current images and qpos. Future actions, future force,
force histories and latent overrides are absent from the policy API.
