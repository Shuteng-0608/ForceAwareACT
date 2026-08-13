# Native-rate Motion-CVAE and Dual-Zero controls

This document is the source of truth for the ACT-aligned high-rate control
models added alongside the existing Contact-CVAE and Official ACT baseline.
Source code and CLI `--help` output take precedence.

## Controlled comparison

Both controls retain the same online observation and prediction components as
the native-rate Contact-CVAE:

- two RGB cameras and current normalized qpos;
- the causal last 100 raw wrench samples at 500 Hz, grouped by policy-state
  intervals and packed as `[B, I, S, 6]`;
- the shared local temporal force encoder, online interval encoder and
  force-vision fusion;
- action `[B, K, 7]`, interval endpoint force `[B, K, 6]`, and native-rate
  force `[B, K, S, 6]` heads.

The experimental variable is only the latent mechanism:

| model | training latent | future-force posterior input | deployment | memory prefix |
| --- | --- | --- | --- | --- |
| Contact-CVAE | contact posterior and conditional prior | yes, encoded native-rate intervals | zero or prior | `z_contact, qpos, z_F_online, z_VF` |
| Motion-CVAE | ACT action-only motion posterior | no | exact-zero `z_motion` | `z_motion, qpos, z_F_online, z_VF` |
| Dual-Zero | none | no | no latent exists | `qpos, z_F_online, z_VF` |

Dual-Zero deliberately removes the latent token and adapter. Passing a zero
vector through a learned affine adapter would create a trainable constant token
through its bias and is therefore not equivalent to a structurally latent-free
control.

The older `act_aligned_motion_cvae_control_v1` state-rate implementation remains
loadable and unchanged for checkpoint compatibility. New experiments should use
`act_aligned_motion_cvae_highrate_force_v2` when force is part of the policy.

## Losses

Motion-CVAE uses:

```text
L_motion = lambda_action * masked_action_L1
         + lambda_force * masked_endpoint_force_L1
         + lambda_highrate * interval_balanced_500Hz_force_L1
         + beta_motion * KL(q_motion || N(0, I))
```

Dual-Zero uses only the first three reconstruction terms. Its training config
does not contain posterior KL or prior-matching weights.

The high-rate loss first averages wrench dimensions, then valid samples within
each interval, and finally valid intervals. Thus 16-sample and 17-sample policy
intervals receive equal weight. Padded samples and intervals do not contribute.

## Preflight

Run preflight before a long training job:

```bash
PYTHONPATH=src python scripts/preflight_act_aligned_high_rate_controls.py \
  motion_cvae mujoco_data/peg_hole_100 --device cpu --smoke

PYTHONPATH=src python scripts/preflight_act_aligned_high_rate_controls.py \
  dual_zero mujoco_data/peg_hole_100 --device cpu --smoke
```

Preflight verifies optimizer coverage, nonzero gradients through every major
module, output shapes, the latent contract, and an intervention on one valid
2 ms force sample. The intervention must change both `z_F_online` and the action
prediction.

## Training

One-step burn-in examples:

```bash
PYTHONPATH=src python scripts/train_act_aligned_high_rate_motion_cvae.py \
  mujoco_data/peg_hole_100 \
  --output-dir outputs/native_rate_motion_burnin \
  --smoke --run-mode burn_in --max-train-steps 1

PYTHONPATH=src python scripts/train_act_aligned_high_rate_dual_zero.py \
  mujoco_data/peg_hole_100 \
  --output-dir outputs/native_rate_dual_zero_burnin \
  --smoke --run-mode burn_in --max-train-steps 1
```

Remove `--smoke`, select a formal output directory, and omit burn-in flags for
the canonical run length. Both trainers reuse the ACT-aligned split,
normalization, logging, periodic/best/final checkpoint, exact mid-epoch resume,
RNG restoration and checkpoint reload audit infrastructure.

Resume example:

```bash
PYTHONPATH=src python scripts/train_act_aligned_high_rate_dual_zero.py \
  mujoco_data/peg_hole_100 \
  --output-dir outputs/native_rate_dual_zero_burnin \
  --resume outputs/native_rate_dual_zero_burnin/burn_in.pt \
  --smoke --run-mode burn_in --max-train-steps 2
```

## Rollout and force audit

The checkpoint embeds model dimensions and normalization statistics. The
rollout adapter dispatches on `architecture_version` and requires the native
high-rate tensors for both models. The shared CLI option must remain
`--contact-latent-mode zero`; prior mode is rejected because neither checkpoint
has a contact prior.

Before closed-loop evaluation, confirm empirical force use on a held-out
episode:

```bash
PYTHONPATH=src python scripts/audit_rollout_force_usage.py \
  outputs/native_rate_dual_zero/best.pt \
  path/to/episode.hdf5 \
  --device cpu \
  --output outputs/native_rate_dual_zero/force_usage_audit.json
```

This audit holds image and qpos fixed while comparing actual, physical-zero,
time-reversed and delayed native-rate force. Nonzero changes in online force
features and predictions prove that the checkpoint uses force, but do not by
themselves establish useful closed-loop behavior.

## Compatibility invariants

- Existing Contact-CVAE and state-rate Motion-CVAE architecture versions,
  state-dict keys and rollout behavior are unchanged.
- Deployment policy APIs never accept future actions or future force.
- Motion posterior APIs never accept future force.
- Dual-Zero contains no module whose registered name includes latent,
  posterior or prior.
- Checkpoint loading remains strict; an architecture version selects exactly
  one policy class.
