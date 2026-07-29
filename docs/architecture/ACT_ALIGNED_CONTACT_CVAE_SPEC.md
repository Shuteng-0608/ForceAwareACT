# ACT-Aligned Contact-CVAE Architecture Specification

Status: stage-5B complete independent training path. The
`act_aligned` package provides the complete architecture, while
`act_aligned_training` provides its own batch contract, conditional-CVAE
objective, AdamW parameter groups, causal HDF5 windows, training-only
normalization, epoch loops, zero/prior deployment comparison, and versioned
checkpoint persistence.

Architecture version:

```text
act_aligned_contact_cvae_v1
```

This document is the source of truth for the new ACT-aligned contact-CVAE
family. Existing policies and checkpoints remain legacy implementations and
must not be silently reinterpreted as this architecture.

## 1. Objective

The new family keeps the official ACT encoder/decoder structure wherever the
same kind of information is encoded, while adding causal force observations,
contact posterior learning, a deployable contact prior, and future-force
prediction.

Alignment means:

- the same Transformer depth and block conventions for encoder-like modules;
- a common batch-first token layout;
- ACT-style posterior and decoder sequence semantics;
- explicit spatial, temporal, modality, and camera identity;
- checkpoint metadata that records the architecture actually trained.

Alignment does not mean copying known defects or weakening the repository's
current data-isolation and deployment-validation rules.

## 2. Canonical Configuration

The canonical preset is defined by `ACTAlignedConfig.canonical_act()`:

| Field | Value |
| --- | ---: |
| `d_model` | 512 |
| attention heads | 8 |
| attention head dimension | 64 |
| feed-forward dimension | 3200 |
| encoder depth | 4 |
| decoder depth | 7 |
| dropout | 0.1 |
| activation | ReLU |
| normalization order | post-norm |
| contact latent dimension | 32 |
| action chunk length | 100 |
| backbone | ResNet18 |
| pretrained backbone | yes |
| backbone batch normalization | frozen |
| image normalization | ImageNet |

The canonical chunk length is 100, matching the official ALOHA ACT example.
Raw qpos/action dimensions, force dimensions, number of cameras, and input
resolution remain task-level settings. An experiment may override the chunk
length—for example, to preserve a fixed physical-time horizon at another
control frequency—but the override must be explicit and recorded in its
checkpoint. The default input uses two 224 by 224 cameras and a 20-sample force
window.

`ACTAlignedConfig.compact_smoke()` reduces width and feed-forward size for
tests. It deliberately keeps four encoder layers and seven decoder layers.
Smoke tests must not validate a structurally different one-layer model.

## 3. Shared Tensor Layout

All new Transformer-facing modules use batch-first layout:

```text
token tensor: [B, S, D]
position:     [1 or B, S, D]
padding mask: [B, S], bool, True means padding
```

Raw modalities are converted through explicit adapters:

```text
qpos   [B, q_dim]       -> [B, 1, D]
action [B, K, a_dim]    -> [B, K, D]
force  [B, L, f_dim]    -> [B, L, D]
image  [B, C, 3, H, W]  -> [B, C*H'*W', D]
latent [B, z_dim]       -> [B, 1, D]
```

This permits the official 14-dimensional bimanual setting and the current
7-dimensional single-arm setting to share the same Transformer representation
without pretending that their raw state spaces are identical.

## 4. Vision Contract

The shared ResNet18 processes every camera with the same weights:

```text
[B, C, 3, H, W]
-> [B*C, 512, H', W']
-> projection to D
-> [B, C*H'*W', D]
```

Each visual token receives:

- a two-dimensional sine position encoding;
- a learned camera embedding.

For two 224 by 224 cameras and output stride 32:

```text
H' = W' = 7
visual token count = 2 * 7 * 7 = 98
visual tokens = [B, 98, D]
```

Pretrained vision and ImageNet normalization are a coupled setting. V1 rejects
configurations that enable only one of them.

ImageNet normalization is performed once by the data pipeline; the backbone
records and validates that contract but does not normalize tensors internally.
The ResNet18 is shared across cameras, uses frozen batch-normalization
statistics/affine buffers, and its trainable convolution weights remain
available to the backbone optimizer parameter group.

## 5. Contact Posterior Contract

Future action and force values with the same future timestep are embedded into
one time-aligned contact-step token:

```text
contact_step_content_t =
    action_adapter(action_t)
  + force_adapter(force_t)

contact_step_attention_position_t =
    fixed_sinusoidal_temporal_position_t
  + contact-step type embedding
```

The posterior sequence is:

```text
[CLS, qpos, contact_step_0, ..., contact_step_K-1]
```

Shape:

```text
[B, K+2, D]
```

The sequence receives the same fixed one-dimensional sine/cosine positions as
the official ACT posterior, plus distinct learned CLS, qpos, and contact-step
type embeddings. Given an action padding mask `[B, K]`, two
unmasked columns are prepended for CLS and qpos before it reaches the encoder.
Padded action-force values therefore cannot affect the encoded CLS result.

The sequence passes through a four-layer post-norm Transformer encoder. The
encoded CLS token produces:

```text
mu_contact     [B, z_dim]
logvar_contact [B, z_dim]
z_contact      [B, z_dim]
```

The posterior is training-only. Deployment forward methods must reject future
action and future force labels.

The conditional prior accepts only four deployment-available features:

```text
qpos feature, z_F_online, z_VF, mean visual feature
-> MLP(4D -> D), ReLU, dropout
-> prior mu/logvar [B, z_dim]
```

Default deployment follows official ACT and uses an all-zero contact latent.
The conditional-prior mean remains an explicit deterministic deployment mode,
and stochastic rollout may use a reparameterized prior sample. Future action
and force tensors are deliberately absent from every deployment interface.

## 6. Online Force Contract

Only historical and current force samples are legal:

```text
[CLS_F, force_0, ..., force_L-1]
-> fixed one-dimensional sine/cosine position
-> four-layer encoder
-> z_F_online [B, D]
```

CLS and force-sample type embeddings make their roles explicit. A force
padding mask `[B, L]` is extended with one unmasked CLS column.

The dataset and rollout implementations remain responsible for ensuring that
no selected force timestamp is later than the current decision timestamp.

## 7. Force-Vision Fusion

Online force attends into spatial visual tokens:

```text
query = z_F_online       [B, 1, D]
key/value = visual      [B, N_visual, D]
output = z_VF           [B, D]
```

Visual spatial/camera positions are added to cross-attention keys only; values
remain the unmodified visual tokens. Optional attention diagnostics have shape
`[B, 1, N_visual]`.

This cross-attention is a distinct fusion block, not an encoder layer and not a
replacement for the four-layer policy encoder.

## 8. Policy Memory Contract

Special-token order is frozen as:

```text
z_contact, qpos, z_F_online, z_VF
```

The complete policy input is:

```text
[z_contact, qpos, z_F_online, z_VF, visual tokens]
```

For the default two-camera input:

```text
4 special tokens + 98 visual tokens = 102 tokens
policy input = [B, 102, D]
```

Every special token receives a learned type embedding. The sequence passes
through a four-layer post-norm policy encoder.

The executable policy freezes the order above through
`POLICY_SPECIAL_TOKEN_NAMES`. Its position tensor is assembled in the same
order: four learned special-token positions followed by the spatial/camera
positions of all visual tokens. Both the policy encoder and decoder
cross-attention receive this position tensor.

## 9. Decoder Contract

The decoder follows ACT/DETR query semantics:

```text
initial target: zeros([B, K, D])
query position: K learned embeddings
memory:         encoded policy tokens
```

The decoder has seven layers. Action and force heads consume the final decoder
layer:

```text
decoder hidden [B, K, D]
pred_action    [B, K, action_dim]
pred_force     [B, K, force_dim]
```

Returning intermediate decoder activations for diagnostics is allowed, but
selecting decoder layer zero as the policy output is not.

Future gradient tests must prove that every decoder layer participates in the
training graph.

Both prediction heads are parallel linear projections of the same final
decoder tensor:

```text
pred_action = Linear(D, action_dim)(decoder_hidden)
pred_force  = Linear(D, force_dim)(decoder_hidden)
```

The force head does not concatenate `z_contact` a second time. Contact
information already enters the decoder through policy memory.

The executable Transformer blocks preserve the official ACT/DETR attention
semantics: positional embeddings are added to queries and keys only; attention
values remain the unmodified token content. The encoder has no extra
post-stack LayerNorm in the configured post-norm mode. The decoder has one
shared final LayerNorm and exposes either its final result or all seven
normalized intermediate results.

## 10. Behaviors Inherited from Official ACT

- ResNet18 spatial features.
- Two-dimensional sine visual positions.
- Fixed one-dimensional sine/cosine positions for temporal encoder sequences.
- Four-layer posterior encoder.
- Four-layer policy encoder.
- Seven-layer action-query decoder.
- ReLU feed-forward blocks, dropout, residual connections, and post-norm.
- Learned action-query positions with zero initial decoder targets.
- A latent projected into policy memory.
- Separate backbone and non-backbone optimizer parameter groups.

## 11. Behaviors Explicitly Not Inherited

- Selecting decoder output index zero when seven layers are configured.
- Calling a deterministic zero latent a sampled prior.
- Training an unused padding-prediction head.
- Dividing a masked reconstruction loss by padded elements.
- Computing normalization statistics from validation episodes.
- Resampling a random validation timestep on every validation pass.
- Relying on optimizer defaults for weight decay or other recorded settings.
- Allowing a training script to silently override architecture depth.

## 12. Preserved Repository Improvements

- Episode-disjoint train/validation splits.
- Normalization statistics computed from training episodes only.
- Complete non-padded action chunks where available.
- Strictly causal online force history.
- Future-force auxiliary supervision.
- Official-style zero-latent deployment and conditional-prior evaluation.
- Full fixed-window validation and relative-improvement early stopping.
- Explicit training seed, DataLoader seed, and checkpoint metadata.

## 13. Checkpoint Compatibility

V1 checkpoints must store:

```text
policy_variant
architecture_version
model configuration
explicit depth of every encoder and decoder
decoder_output_layer = "last"
token_layout = "batch_first"
vision preprocessing settings
```

Legacy checkpoints without
`architecture_version="act_aligned_contact_cvae_v1"` must continue to use their
existing policy classes. New code must not infer V1 solely from similar
state-dictionary keys.

The legacy files below remain unchanged during the initial implementation:

```text
src/force_aware_act/models/force_aware_contact_cvae_policy.py
src/force_aware_act/models/posterior.py
src/force_aware_act/models/vision.py
scripts/train_minimal.py
```

## 14. Stage-1 Acceptance Criteria

Stage 1 is complete when:

- immutable canonical and compact presets exist;
- invalid non-aligned configurations are rejected;
- sequence lengths and token order are executable assertions;
- the default shape contract resolves to:

  ```text
  visual tokens:            [B, 98, 512]
  contact posterior tokens: [B, 102, 512]
  force encoder tokens:     [B, 21, 512]
  policy memory:            [B, 102, 512]
  decoder hidden:           [B, 100, 512]
  action output:            [B, 100, 7]
  force output:             [B, 100, 6]
  ```

- checkpoint metadata exposes all encoder and decoder depths;
- no legacy policy or checkpoint dispatch behavior changes.

## 15. Stage-2 Implementation Map and Acceptance Criteria

The shared components are isolated under:

```text
src/force_aware_act/models/act_aligned/
  token_adapters.py
  position_encoding.py
  backbone.py
  transformer.py
```

Stage 2 is complete when:

- qpos, action, force, and latent adapters enforce their raw input layout and
  map every modality to `[B, S, D]`;
- temporal, spatial, camera, and token-type encodings preserve the same token
  layout;
- one shared ResNet18 maps `[B, 2, 3, 224, 224]` to `[B, 98, D]`, with a
  matching positional tensor;
- the encoder has four independently parameterized layers and no final stack
  normalization;
- the query decoder has seven independently parameterized layers, 100 learned
  query positions, zero initial targets, and returns `[B, 100, D]`;
- a backward test proves that all four encoder layers and all seven decoder
  layers receive gradients;
- no legacy model, trainer, data loader, or checkpoint dispatch is modified.

## 16. Stage-3 Implementation Map and Acceptance Criteria

The latent and force-conditioning components are isolated under:

```text
src/force_aware_act/models/act_aligned/
  contact_latent.py
  online_force.py
  fusion.py
```

Stage 3 is complete when:

- the contact posterior maps `[B, 7]`, `[B, 100, 7]`, and `[B, 100, 6]`
  through a 102-token, four-layer encoder to `mu/logvar/z_contact [B, 32]`;
- future action and force values are paired by timestep rather than appended as
  two disjoint sequence segments;
- posterior and force-history masks make altered padded values irrelevant to
  the returned CLS feature;
- the conditional prior only accepts online qpos, force, force-vision, and
  visual-summary features and uses its mean for deterministic deployment;
- the online force encoder maps `[B, 20, 6]` through 21 tokens and four layers
  to `z_F_online [B, 512]`;
- force-vision cross-attention maps `[B, 512]` and `[B, 98, 512]` to
  `z_VF [B, 512]`;
- reparameterization, every posterior/force encoder layer, and fusion all
  receive gradients;
- the stage-3 modules compose end to end without constructing the full policy;
- no legacy model, trainer, data loader, or checkpoint dispatch is modified.

## 17. Stage-4 Complete Policy and Acceptance Criteria

The complete architecture is assembled in:

```text
src/force_aware_act/models/act_aligned/policy.py
```

Its two policy-level execution paths are deliberately separate:

```text
forward_train(images, qpos, force_history, action_chunk, future_force_chunk)
forward(images, qpos, force_history)
```

`forward_train` samples `z_contact` from the future-conditioned posterior and
also computes the online conditional prior for its training objective.
`forward` has no future-label arguments and uses zero latent by default.
Conditional-prior mean, stochastic prior rollout, and an explicit offline
latent override remain available without widening the deployment input
contract.

Stage 4 is complete when:

- policy tokens and positions have shape `[B, 102, 512]` in canonical mode;
- special-token order is exactly
  `[z_contact, qpos, z_F_online, z_VF]`, followed by visual tokens;
- the four-layer policy encoder produces memory `[B, 102, 512]`;
- the seven-layer query decoder produces final hidden `[B, 100, 512]`;
- parallel linear heads produce action `[B, 100, 7]` and force `[B, 100, 6]`;
- both heads consume the final normalized decoder layer with no latent
  concatenation in the force head;
- deployment cannot accept future action or force labels;
- all four policy-encoder layers, all seven decoder layers, and both prediction
  heads receive gradients;
- intermediate decoder diagnostics have shape `[7, B, 100, 512]` while policy
  predictions continue to use only the final slice;
- no legacy policy, trainer, data loader, or checkpoint dispatch is modified.

## 18. Stage-5A Independent Training Core

The new trainer is isolated from the legacy training package:

```text
src/force_aware_act/act_aligned_training/
  config.py
  batch.py
  losses.py
  optimizer.py
  trainer.py
```

Its canonical objective is:

```text
loss =
    masked_action_l1
  + force_loss_weight * masked_force_l1
  + posterior_kl_weight * KL(q_contact || N(0, I))
  + prior_match_weight
      * KL(stop_gradient(q_contact) || p_contact_conditional)
```

The posterior KL retains official ACT's standard-normal regularization. The
prior-matching KL treats the posterior mean and log-variance as a detached
teacher distribution. It therefore trains both conditional-prior mean and
variance without allowing prior quality to drag the future-informed posterior.
Reconstruction and standard-normal KL train the posterior; detached matching
trains the conditional prior.

Both reconstruction terms divide by the exact number of valid scalar targets:

```text
valid future steps * output feature dimension
```

An all-padding target batch is rejected. The optimizer is AdamW with an
identity-based parameter partition:

```text
ResNet18 body                              -> backbone learning rate
input projection and every other module   -> main learning rate
```

Every trainable parameter must occur exactly once. Canonical defaults are main
learning rate `1e-5`, backbone learning rate `1e-5`, weight decay `1e-4`, and
posterior KL weight `10`. Prior matching uses weight `1`.

Validation contains one posterior call and two deployment calls:

```text
posterior validation:
  forward_train(..., sample_posterior=False)

zero deployment:
  forward(..., contact_latent_mode="zero")

prior deployment:
  forward(..., contact_latent_mode="prior", deterministic_prior=True)
```

Future targets remain outside the deployment call and are used only afterward
to compute metrics. The canonical model-selection metric is zero-deployment
action L1; prior-deployment metrics are recorded independently.

Stage 5A is complete when:

- a strict batch validates canonical image, qpos, history-force, action,
  future-force, and both padding-mask shapes;
- masked losses are invariant to padded values and reject an empty target;
- posterior standard KL sends gradients to the posterior;
- detached Gaussian matching sends gradients to the prior but never to the
  posterior module;
- ResNet body parameters use the backbone group, while projection,
  posterior/prior, Transformer, query, and head parameters use the main group;
- the two optimizer groups are complete, disjoint, and duplicate-free;
- one synthetic training step updates both the backbone and an output head;
- validation uses the posterior mean for reconstruction and reports both zero
  and conditional-prior-mean deployment;
- the independent training package imports no legacy model or training module.

## 19. Stage-5B Data, Epoch, and Checkpoint Path

The real-data path is implemented independently in:

```text
src/force_aware_act/act_aligned_training/
  schema.py
  split.py
  normalization.py
  data.py
  loop.py
  checkpoint.py

scripts/train_act_aligned_contact_cvae.py
```

The audited `peg_hole_100` schema contains state/action/image at approximately
30 Hz and compensated force at 500 Hz. State timestamps are the decision
clock. For every state time, force and image alignment select the last source
sample whose timestamp is not later than the decision time. On the audited 100
episodes, causal force lag is at most 2 ms.

Every state timestep becomes one fixed dataset index:

```text
images[t]                    -> [2, 3, 224, 224]
qpos[t]                      -> [7]
aligned_force[t-19:t+1]      -> [20, 6]
action[t:t+100]              -> [100, 7]
aligned_force[t:t+100]       -> [100, 6]
```

History is left-padded and future chunks are right-padded. Padding values are
zero in normalized space. Action and future force share the exact future
timestamp range and mask.

Raw images remain `uint8 RGB [H, W, 3]` in HDF5. Dataset preprocessing converts
them to float CHW, resizes them, and applies ImageNet normalization only when
the model configuration requires it. Collection-time resizing or
normalization is not required.

Episode splitting is deterministic and episode-disjoint. Qpos, action, and
state-rate force statistics are computed from train episodes only and stored
with the split manifest.

Epoch aggregation weights reconstruction metrics by valid scalar targets and
latent metrics by batch samples. Validation reports posterior, zero-deployment,
and prior-mean-deployment metrics, including posterior/prior mean distance and
both average standard deviations.

Checkpoint format V1 stores:

```text
model/training versions and complete configs
model and AdamW states
epoch, global step, and best metric
normalization and episode split manifest
Python, NumPy, PyTorch, CUDA, and DataLoader RNG states
```

Saves use a temporary file followed by atomic replacement. Loads reject config
or format mismatches rather than dispatching to legacy implementations.

The standalone CLI supports canonical training, exact resume, and a small
real-data `--smoke` mode. It imports only the new ACT-aligned model and training
packages.

## 20. Stage-6A Canonical Training Preflight

Long training is gated by:

```text
src/force_aware_act/act_aligned_training/diagnostics.py
scripts/preflight_act_aligned_contact_cvae.py
```

The preflight deliberately performs no optimizer step. It reports:

- exact total/trainable parameter counts by non-overlapping top-level module;
- complete, duplicate-free optimizer group coverage, LR, and weight decay;
- training and zero/prior deployment output shapes;
- one total-loss backward gradient norm for every core module;
- an isolated prior-match gradient check proving posterior norm is exactly
  zero while conditional-prior norm is nonzero;
- forward/backward time and, on CUDA, peak allocated and reserved memory.

CPU/real-data functional smoke:

```text
python scripts/preflight_act_aligned_contact_cvae.py \
  --smoke --device cpu --batch-size 1 \
  --data-root mujoco_data/peg_hole_100
```

Canonical GPU gate:

```text
python scripts/preflight_act_aligned_contact_cvae.py \
  --device cuda --batch-size 1 \
  --data-root mujoco_data/peg_hole_100 \
  --output runs/act_aligned_preflight_batch1.json
```

The canonical command retains pretrained ResNet18 and ImageNet normalization.
Batch size must be raised only after the batch-1 report passes and its peak
memory is known.
