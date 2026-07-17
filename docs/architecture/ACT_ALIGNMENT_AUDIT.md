# ACT Alignment Audit

This document audits how closely the current ForceAwareACT implementation aligns with the original ACT architecture at a conceptual level, and identifies which pieces should remain ACT-faithful for fair baseline comparisons.

This is documentation only. It does not propose changing current algorithm behavior inside this audit.

## Summary Table

| Component | Original ACT behavior | Current ForceAwareACT behavior | Alignment status | Notes |
| --- | --- | --- | --- | --- |
| Vision backbone | ResNet18-style visual backbone for RGB observations. | `ResNet18VisionEncoder` uses torchvision ResNet18 without avgpool/classifier. | Aligned | Code path: `src/force_aware_act/models/vision.py`. |
| Vision pretraining | ACT-style task training commonly uses from-scratch visual training for fair task-specific comparison. | `train_minimal.py` hard-codes `pretrained_resnet18=False`; checkpoints inspected in `VISION_BACKBONE_AUDIT.md` also record `False`. | Aligned | Pretrained support exists in the module but is not exposed by the training CLI. |
| Image resolution | ACT commonly uses fixed-size resized camera images. | Dataset resizes to `image_size`, default `(224, 224)`. ResNet tokens are 7x7 per camera at 224x224. | Aligned | Exact resolution can be varied by CLI; current model tests assume 224x224. |
| Multi-camera tokenization | Multi-camera RGB images are encoded and provided as visual context. | Images have shape `[B, N_cam, 3, H, W]`; each camera is flattened through one shared ResNet18 and tokens are concatenated across cameras. | Aligned | For two 224x224 cameras, output is `[B, 98, d_model]`. |
| Joint/proprioceptive input | Current joint positions are used as proprioceptive state. | `ForceAwareACTPolicy.forward` consumes only `qpos` as online proprioception. | Aligned | Dataset loads more state fields, but model input remains qpos-only. |
| qpos encoder | qpos projected to transformer hidden dimension. | `JointMLP`: Linear qpos-to-`d_model`, ReLU, Linear to `d_model`, producing one `z_q` token. | Aligned | Slight implementation detail differs from a single projection, but concept is faithful. |
| qvel/joint_torque/ee_pose usage | Not part of the core ACT proprioceptive input unless explicitly added. | Dataset returns `qvel`, `joint_torque`, and `ee_pose`; policy does not consume them. | Aligned | Keep these as ablations, not the main ACT baseline. |
| CVAE motion latent z | Posterior latent trained from future action sequence; inference uses zero latent. | `MotionPosteriorEncoder(qpos, action_chunk)` exists in posterior mode; inference always sets `z_motion=0`. | Partially aligned | Posterior mode is ACT-like. Current strongest baseline uses zero-latent training, which is engineering-motivated rather than canonical CVAE training. |
| Contact latent `z_contact` | Not present in original ACT. | `ContactPosteriorEncoder` and `ContactPriorEncoder` add a second contact latent; `z_contact` is inserted into policy tokens and force head. | Extension | Should not be silently included in an ACT baseline. |
| Inference latent | Original ACT sets latent z to zero at inference. | Default inference uses `z_motion=0`, `z_contact=0`; rollout can also use `contact_latent_mode="prior"`. | Partially aligned | `zero` is ACT-faithful; `prior` is an extension. |
| Transformer encoder-decoder | Transformer encoder-decoder predicts a future action sequence from observation context and action queries. | `policy_encoder` encodes context tokens; `policy_decoder` decodes learned `future_queries`. | Aligned | Context token set is extended by force/contact tokens. |
| Future action queries | ACT decodes a chunk with learned/fixed future queries or positional/action query embeddings. | `future_queries` is a learned `[1, chunk_len, d_model]` parameter. | Aligned | Query parameter is initialized with truncated normal. |
| Action target semantics | Future absolute joint/action targets, not deltas for the main baseline. | `joint_pos` predicts future measured qpos; `action` predicts absolute executable command; delta modes also exist. | Partially aligned | For command-labeled demos, `action_mode="action"` is the closest ACT-faithful executable target. |
| Delta action usage | Delta actions are not the main ACT target and have been reported as weaker in ACT-style comparisons. | `delta_joint_cmd` and `delta_joint_pos_command` subtract current qpos from command labels. | Different | Treat delta modes as ablations. |
| L1 loss | L1 action reconstruction loss. | `loss_action = L1(pred_action, action_chunk)`. | Aligned | Code path: `src/force_aware_act/training/losses.py`. |
| Force auxiliary loss | Not part of original ACT. | `loss_force = L1(pred_force, future_force_chunk)` with `lambda_force`. | Extension | Should be off or separately ablated for pure ACT baseline. |
| Temporal ensembling / chunk aggregation | ACT uses temporal aggregation/ensembling over overlapping action chunks at inference. | Rollout supports `first`, `mid`, `last`, and `temporal`; default CLI is `first`, current recommendation has used `mid`. | Partially aligned | The `temporal` option is closest conceptually; verify details before claiming exact ACT temporal ensembling. |
| Policy frequency | ACT executes policy chunks at a fixed control/policy rate. | Rollout uses `--policy-rate-hz`, default 30 Hz. | Aligned | Low-level MuJoCo steps can be faster than policy decisions. |
| Rollout command interpretation | ACT action output is applied as the target action/joint command. | Absolute modes use `target_ctrl = pred_action`; delta modes use `target_ctrl = current_qpos + pred_action`; then safety filters run. | Partially aligned | Absolute `action` mode is most faithful for executable command labels; safety filters are deployment-specific. |

## Joint-State Encoding Audit

The current model consumes only `qpos` as online proprioceptive input:

```text
ForceAwareACTPolicy.forward(images, qpos, force_window, ...)
```

`ContactForceHDF5Dataset` loads and returns:

- `qpos` from `observations/joint_pos`
- `qvel` from `observations/joint_vel`
- `joint_torque` from `observations/joint_torque`
- `ee_pose` from `observations/ee_pose`

Only `qpos` is passed into `ForceAwareACTPolicy`. `qvel`, `joint_torque`, and `ee_pose` are currently dataset outputs but not model inputs.

`qpos` is encoded by `JointMLP`:

```text
qpos [B, 7] -> Linear(7, d_model) -> ReLU -> Linear(d_model, d_model) -> z_q [B, d_model]
```

This is ACT-faithful at the comparison level: the baseline should use current joint positions only. Adding qvel, torque, or end-effector pose would add extra proprioceptive information and should be treated as a separate ablation or improved robotics variant, not as the main ACT baseline.

Recommendation:

- Main ACT comparison: images + qpos only.
- Ablations: add qvel, joint torque, and/or ee pose one at a time with explicit checkpoint metadata.

## Vision Audit Summary

`docs/architecture/VISION_BACKBONE_AUDIT.md` found:

- `ResNet18VisionEncoder` supports ImageNet weights through `ResNet18_Weights.DEFAULT`.
- The active `scripts/train_minimal.py` path hard-codes `pretrained_resnet18=False`.
- Inspected checkpoints record `model.pretrained_resnet18: False`.
- No training CLI flag exposes pretrained/freeze behavior.
- The backbone is fully trainable because `freeze_resnet18` defaults to `False`.

For the main ACT comparison, keep ResNet18 from scratch. ImageNet-pretrained ResNet18 should be an optional ablation, especially for small 10-100 demonstration datasets.

## Action Target Audit

Original ACT predicts a future chunk of target actions or joint positions, not a delta from current qpos as the main target.

Current dataset modes:

- `joint_pos`: legacy future measured state target, `observations/joint_pos[i + 1 : i + K + 1]`.
- `action`: absolute executable actuator command, `/action[i : i + K]`.
- `joint_pos_command`: absolute executable command copy, `/actions/joint_pos_command[i : i + K]`.
- `delta_joint_cmd`: `/action[i : i + K] - observations/joint_pos[i]`.
- `delta_joint_pos_command`: `/actions/joint_pos_command[i : i + K] - observations/joint_pos[i]`.

For command-labeled demonstrations, `action_mode="action"` is the best ACT-faithful target because it is an absolute executable command. `joint_pos` is ACT-like in shape but can be a weaker physical target because it is measured qpos rather than the actuator command actually sent to MuJoCo.

`delta_joint_cmd` and `delta_joint_pos_command` should be treated as ablations. They are useful engineering experiments, but they are not the main ACT-faithful target.

## Latent Audit

Original ACT uses a CVAE-style latent `z`:

- During training, the posterior sees future action labels and samples a latent.
- During inference, `z` is set to zero.

Current ForceAwareACT has three latent-related mechanisms:

1. `z_motion`
   - ACT-like motion latent from `MotionPosteriorEncoder(qpos, action_chunk)`.
   - Inference sets `z_motion=0`.
2. `z_contact`
   - Force-aware extension from `ContactPosteriorEncoder(qpos, action_chunk, future_force_chunk)`.
   - Inserted into policy tokens and passed directly to `ForceHead`.
3. Conditional contact prior
   - Online prior from `z_q`, `z_F_online`, `z_VF`, and visual summary.
   - Used only in inference when `contact_latent_mode="prior"` and in posterior training for optional prior distillation.

For an ACT-faithful baseline:

- Use one motion latent if matching the original CVAE setup.
- At inference, set that latent to zero.
- Do not include `z_contact` or a contact prior in the reported ACT baseline.

For the current robust engineering baseline:

- `train_latent_mode="zero"` trains with `z_motion=0` and `z_contact=0`.
- This avoids the posterior-vs-zero deployment mismatch observed in recent experiments.
- It is a practical deterministic baseline, but it is not the canonical ACT CVAE training recipe.

`z_contact` and the conditional prior should be reported as ForceAwareACT extensions or ablations.

## Force-Aware Extensions

These components are not ACT-faithful base components. They are the ForceAwareACT contribution space and should be added incrementally:

- `force_window` online input.
- `TemporalForceEncoder` Transformer with a force CLS token.
- `ForceVisionCrossAttention`, where force is the query and vision tokens are keys/values.
- `z_F_online` token in the policy encoder context.
- `z_VF` fused force-vision token in the policy encoder context.
- `ForceHead`, which predicts future wrench chunks.
- `loss_force` with coefficient `lambda_force`.
- `z_contact` posterior latent.
- Conditional contact prior and prior distillation loss.

The current implementation always constructs and uses the force encoder, force-vision fusion, contact latent token, and force head. There are no CLI flags in `train_minimal.py` to disable these pieces for a pure ACT baseline.

## Rollout Alignment

Rollout action interpretation is action-mode aware:

- Absolute modes: `target_ctrl = pred_action`.
- Delta modes: `target_ctrl = current_qpos + pred_action`.

Then the rollout applies deployment guards:

- optional axial push bias,
- `max_delta_q` clipping,
- EMA smoothing,
- actuator ctrlrange clipping,
- force stop threshold.

For fair ACT comparison in MuJoCo:

- Use an absolute action target (`action_mode="action"` when command labels are available).
- Keep rollout interpretation consistent across ACT and ForceAwareACT variants.
- Treat safety clipping/smoothing as deployment infrastructure and report it clearly.
- Prefer `action_select_mode="temporal"` only if its implementation is verified to match the ACT temporal ensembling recipe closely enough; otherwise report `first`, `mid`, or `last` as chunk-selection baselines.

## Recommended Experiment Matrix

| Experiment | Inputs | Architecture | Losses | Action target | Purpose |
| --- | --- | --- | --- | --- | --- |
| A. ACT baseline | images + qpos | ResNet18 from scratch, qpos token, transformer encoder-decoder, one motion latent if CVAE mode is desired | L1 action + optional motion KL | absolute command `action` if available | Fair ACT-style baseline. |
| B. ForceAwareACT-minimal | images + qpos + force_window | ACT baseline plus force token; no `z_contact` | L1 action; force loss optional and separately reported | absolute command `action` | Isolate benefit of online force history. |
| C. ForceAwareACT-fusion | images + qpos + force_window | Add force-vision cross-attention token `z_VF` | L1 action; optional force loss | absolute command `action` | Test whether force-guided visual fusion helps. |
| D. ForceAwareACT-full | images + qpos + force_window | Add `z_contact`, conditional prior, and force head after stable baseline | action + force + KL/prior terms as configured | absolute command `action` | Full proposed extension. |
| E. Optional ablations | vary one factor | qvel/torque/ee_pose, pretrained ResNet18, delta targets, temporal selection, force window settings | matched except tested factor | matched except tested factor | Attribute gains or failures. |

Concrete ablations to keep separate from the main ACT baseline:

- qvel input.
- joint torque input.
- ee pose input.
- ImageNet-pretrained ResNet18.
- frozen or partially frozen vision backbone.
- delta action targets.
- `action_select_mode` variants: `first`, `mid`, `last`, `temporal`.
- `force_window_len` and `force_window_duration`.
- `lambda_force` on/off.
- `z_contact` and conditional prior on/off.

## Future Code/Config Recommendations

Do not implement these in this audit, but future work should add explicit profiles and flags so comparisons are reproducible:

```text
--profile act_faithful
--profile force_aware_minimal
--profile force_aware_full
```

Suggested explicit flags:

```text
--use-force-window
--use-force-vision-fusion
--use-force-head
--use-contact-latent
--use-qvel
--use-joint-torque
--use-ee-pose
--vision-pretrained none|imagenet
--freeze-vision-backbone
--unfreeze-vision-last-block
```

Checkpoint config should record every comparison-relevant flag, including:

- force input/fusion/head enabled,
- contact latent/prior enabled,
- qpos-only vs extra proprioception,
- vision initialization and freeze policy,
- action target mode,
- latent training mode,
- rollout chunk selection mode.

If temporal ensembling is intended to match ACT, verify or implement the exact ACT-style weighting and overlapping-chunk aggregation, then expose it as an explicit named rollout mode.

## Main Conclusions

Already ACT-faithful or close:

- Multi-camera RGB input.
- ResNet18 visual encoder trained from scratch in the current training path.
- qpos-only online proprioception.
- qpos projection to transformer dimension.
- Transformer encoder-decoder with learned future queries.
- Future action chunk prediction.
- L1 action reconstruction loss.
- Zero latent at inference when `contact_latent_mode="zero"`.
- Absolute command execution when `action_mode="action"`.

Different or ForceAwareACT-specific:

- Mandatory force-window input in the current policy forward path.
- Temporal force encoder.
- Force-vision cross-attention.
- Force auxiliary prediction head and force loss.
- Contact latent and conditional contact prior.
- Zero-latent training as the current strongest engineering baseline rather than canonical posterior CVAE training.
- Delta action modes.
- Deployment safety filters and non-ACT chunk selection defaults.

For the main fair baseline, preserve ACT-faithful components: images + qpos, from-scratch ResNet18, absolute action target, L1 action loss, transformer chunk decoder, and zero latent at inference. Treat force sensing, contact latent/prior, extra proprioception, pretrained vision, delta targets, and rollout selection variants as explicit extensions or ablations.
