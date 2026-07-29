# Official ACT single-arm baseline

## Purpose

This baseline reproduces the original ACT repository at `/home/stw/act`
without force-aware extensions. It is independent of the repository's legacy
`ACTPolicyBaseline`, ACT-aligned contact model, and motion-control model.

The scientific question is whether the official ACT architecture and
episodic sampling retain an action-conditioned latent on the peg-hole data.

## Faithful official structure

Formal defaults reproduce the official README command:

```text
hidden_dim=512
dim_feedforward=3200
encoder_layers=4
decoder_layers=7
nheads=8
chunk_size=100
kl_weight=10
batch_size=8
num_epochs=2000
lr=1e-5
backbone=resnet18
```

The model has 83,908,744 parameters, consistent with the paper's “around
80M” description.

Inputs and outputs:

```text
images:       [B, 2, 3, 480, 640], float in [0, 1]
qpos:         [B, 7], normalized
action label: [B, 100, 7], normalized
padding:      [B, 100], True means padded
prediction:   [B, 100, 7]
```

Posterior:

```text
[CLS, qpos, action_0 ... action_99] -> [B, 102, 512]
4-layer post-norm encoder
CLS -> mu/logvar [B, 32]
z = mu + exp(0.5 * logvar) * epsilon
```

Policy:

```text
shared ImageNet-pretrained ResNet-18 with FrozenBatchNorm
camera feature maps concatenated along feature-map width
[projected z, projected qpos, visual tokens]
4-layer post-norm Transformer encoder
100 learned decoder queries
7-layer post-norm Transformer decoder
Linear(512, 7) action head
Linear(512, 1) is-pad head
```

No camera-identity embedding, force input, force target, force head,
conditional prior, token-type embedding, KL warm-up, free-bits threshold, or
learning-rate scheduler is present. Deployment sets `z` to an exact zero
tensor.

## Necessary dataset adaptations

The official ALOHA code hard-codes 14-D bimanual state/action. This task uses
one 7-DoF arm, so `q_dim=action_dim=7`.

The compact schema maps:

```text
official /observations/qpos -> observations/joint_pos
official /action            -> action
official camera list        -> episode_metadata/camera_names
```

Formal training keeps the native `480 x 640` frames. The compact smoke preset
uses `64 x 64` only for functional tests.

The official code calls `set_seed(1)` before its NumPy 80/20 episode split and
then initializes training with seed 0. The implementation records
`split_seed=1` and `seed=0` separately.

Sampling remains one uniformly selected timestep per episode per epoch.
Sampling is deterministically keyed by epoch and episode so an epoch-boundary
resume is reproducible; this changes the random-number bookkeeping but not
the sampling distribution or number of samples.

Normalization follows the official behavior:

- compute qpos/action statistics over all 100 episodes, including validation;
- concatenate variable-length episode timesteps before computing moments
  (the official task files are fixed-length and use an equivalent stack);
- use PyTorch's unbiased standard deviation;
- clamp standard deviations to at least `1e-2`;
- divide images by 255 in the dataset;
- apply ImageNet normalization inside the policy.

## Loss and schedule

The L1 implementation exactly retains the official denominator:

```text
all_l1 = abs(action - prediction)
l1 = (all_l1 * not_padding[..., None]).mean()
loss = l1 + 10 * KL(q_motion || N(0, I))
```

Masked positions contribute zero to the numerator but remain in the complete
`B x 100 x 7` mean denominator.

With 100 episodes and the official 80/20 split:

```text
80 train episodes / batch 8 = 10 optimizer steps per epoch
10 steps * 2000 epochs = 20,000 optimizer steps
```

Validation occurs before training in every epoch, matching the official loop.
The official sampled-posterior validation loss is computed every epoch.
Deterministic posterior-mean versus zero-latent diagnostics run at epoch 0,
every 100 epochs, and after training; they do not update model state.

As in the official loop, the best validation weights are retained in CPU
memory during training. Every 100 epochs, overwrite-style `latest.pt` stores
the current model, optimizer, RNG, and best state for resume. Training writes
one lightweight `best_policy.pt` at completion instead of retaining 20 large
milestone files. This does not change model optimization or best selection.

## Local functional smoke

```bash
PYTHONPATH=src python scripts/train_official_act.py \
  mujoco_data/peg_hole_100 \
  --output-dir /tmp/forceact_official_act_smoke \
  --device cpu \
  --num-workers 0 \
  --smoke \
  --log-interval 1
```

## Server formal training

First use a new output directory:

```bash
mkdir -p runs/official_act_wst_b8_seed0
```

Then train on one selected GPU:

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train_official_act.py \
  mujoco_data/peg_hole_wst \
  --output-dir runs/official_act_wst_b8_seed0 \
  --device cuda \
  --batch-size 8 \
  --num-epochs 2000 \
  --num-workers 2 \
  --seed 0 \
  --checkpoint-interval-epochs 100 \
  --log-interval 10 \
  2>&1 | tee runs/official_act_wst_b8_seed0/console.log
```

Monitor the official 20,000-step target:

```bash
python scripts/monitor_act_aligned_training.py \
  runs/official_act_wst_b8_seed0 \
  --target-steps 20000 \
  --watch \
  --interval 10
```

## Interpretation

Compare:

- posterior KL;
- posterior mean absolute value and standard deviation;
- posterior mean across-sample variance;
- posterior-zero action delta;
- posterior action L1 and deployment-zero action L1.

If official ACT retains a meaningful posterior-zero gap while the
force-aware motion control collapses, investigate the added force/context
tokens, policy memory structure, resizing, and all-window sampler.

If official ACT also collapses, the primary explanation is more likely the
dataset's conditional predictability or lack of demonstration multimodality
under `KL=10`. Latent collapse alone does not imply a failed deployable
policy; rollout performance must still be evaluated.
