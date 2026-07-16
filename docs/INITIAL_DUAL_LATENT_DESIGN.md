# Initial Dual-Latent ForceAwareACT Design

> Historical design document: this file records the original implementation
> specification and staged contact-prior plan. It is useful for design intent,
> but some details predate the current four-policy implementation and CLI.
> Use [`ARCHITECTURE.md`](ARCHITECTURE.md) and
> [`DUAL_LATENT_ALGORITHM.md`](DUAL_LATENT_ALGORITHM.md) for current behavior.

## 0. Project Goal

Implement a contact-dynamics-aware vision-force ACT policy for contact-rich manipulation.

The model extends an ACT-style policy with:

1. Multi-view image encoding using **ResNet18**.
2. Robot state token encoding.
3. Online past-force-window encoding.
4. Force-as-query visual-force cross-attention.
5. Motion latent `z_motion`.
6. Contact-dynamics latent `z_contact`.
7. Dual output heads:
   - action chunk prediction
   - future force chunk prediction

The policy should learn from HDF5 demonstration episodes collected from MuJoCo / robot teleoperation.

The implementation must strictly separate:

- **Online inference inputs**
  - current multi-view RGB image `I_t`
  - current robot state `q_t`
  - past force window `F_{t-L:t}`

- **Training-only supervision**
  - future action chunk `a_{t:t+K}`
  - future force chunk `F_{t:t+K}`

During inference, the model must never access future action chunks or future force chunks.

---

## 1. Current HDF5 Episode Structure

Each episode is stored as:

```text
episode.hdf5
├── episode_metadata/
│   ├── joint_names
│   ├── actuator_names
│   ├── camera_names
│   ├── initial_joint_pos
│   ├── initial_joint_vel
│   ├── initial_joint_torque
│   ├── initial_ee_pose
│   ├── initial_ft_wrench
│   ├── initial_peg_tip_pos
│   ├── initial_hole_center_pos
│   ├── initial_task_error_xyz
│   ├── final_joint_pos
│   ├── final_joint_vel
│   ├── final_joint_torque
│   ├── final_ee_pose
│   ├── final_ft_wrench
│   ├── final_peg_tip_pos
│   └── final_hole_center_pos
│
├── timestamps/
│   ├── state
│   ├── state_episode
│   ├── force
│   ├── force_episode
│   ├── image
│   └── image_episode
│
├── observations/
│   ├── ee_pose          [N_state, 7]
│   ├── joint_pos        [N_state, 7]
│   ├── joint_vel        [N_state, 7]
│   ├── joint_torque     [N_state, 7]
│   ├── ft_wrench        [N_force, 6]
│   └── images/
│       ├── camera_names
│       ├── ee_cam        [N_image, H, W, 3]
│       └── base_top_cam  [N_image, H, W, 3]
│
└── events/
    ├── names
    ├── t_sim
    ├── t_episode
    └── t_wall
```

Field meanings:

```text
ee_pose      = [x, y, z, qw, qx, qy, qz]
joint_pos    = 7 joint angles
joint_vel    = 7 joint velocities
joint_torque = 7 generalized actuator torques from data.qfrc_actuator
ft_wrench    = [Fx, Fy, Fz, Tx, Ty, Tz]
images/cam   = uint8 RGB image sequence, shape = [N, H, W, 3]
```

---

## 2. Important Design Choice: What Is the Action?

The current HDF5 structure does not explicitly contain an `/actions` dataset.

Therefore, for the first implementation, define the action target from the recorded robot trajectory.

Recommended first baseline:

```text
action_t = joint_pos[t+1]
```

or action chunk:

```text
a_{t:t+K} = joint_pos[t+1:t+K+1]
```

This trains the policy to predict future joint position targets.

Alternative later:

```text
action_t = ee_pose[t+1]
```

or:

```text
action_t = delta_ee_pose[t -> t+1]
```

The first implementation should use joint position action chunks because they are simple, stable, and directly compatible with ACT-style behavior cloning.

Implementation requirement:

The dataset class should support configurable action mode:

```yaml
action_mode: joint_pos
# later possible:
# action_mode: ee_pose
# action_mode: delta_ee_pose
```

For `action_mode: joint_pos`:

```python
action_dim = 7
action_chunk = joint_pos_future
```

For `action_mode: ee_pose`:

```python
action_dim = 7
action_chunk = ee_pose_future
```

---

## 3. Training Sample Definition

For each sampled policy time index `i`, the dataset should return:

```python
sample = {
    "images": images_i,
    "qpos": q_i,
    "qvel": qd_i,
    "joint_torque": tau_i,
    "ee_pose": ee_i,
    "force_window": F_past,
    "action_chunk": A_future,
    "future_force_chunk": F_future,
}
```

Mathematically:

```text
D_i = { I_i, q_i, F_{i-L:i}, a_{i:i+K}, F_{i:i+K} }
```

Where:

```text
I_i              = multi-view RGB images at policy index i
q_i              = current joint position
F_{i-L:i}        = past force window
a_{i:i+K}        = future action chunk
F_{i:i+K}        = future force chunk
```

Online inference inputs:

```text
I_i
q_i
F_{i-L:i}
```

Training-only labels:

```text
a_{i:i+K}
F_{i:i+K}
```

The future action chunk and future force chunk must never be used during inference.

---

## 4. Canonical Tensor Shapes

Raw HDF5 shapes:

```python
ee_pose:       [N_state, 7]
joint_pos:     [N_state, 7]
joint_vel:     [N_state, 7]
joint_torque:  [N_state, 7]
ft_wrench:     [N_force, 6]
ee_cam:        [N_image, H, W, 3]
base_top_cam:  [N_image, H, W, 3]
```

Dataset output shapes:

```python
images:             [N_cam, C, H, W]
qpos:               [7]
qvel:               [7]
joint_torque:       [7]
ee_pose:            [7]
force_window:       [L, 6]
action_chunk:       [K, action_dim]
future_force_chunk: [K, 6]
```

Batch shapes:

```python
images:             [B, N_cam, C, H, W]
qpos:               [B, 7]
qvel:               [B, 7]
joint_torque:       [B, 7]
ee_pose:            [B, 7]
force_window:       [B, L, 6]
action_chunk:       [B, K, action_dim]
future_force_chunk: [B, K, 6]
```

Internal model shapes:

```python
visual_tokens:      [B, N_v, d_model]
z_q:                [B, d_model]
z_F_online:         [B, d_model]
z_VF:               [B, d_model]
z_motion:           [B, z_dim]
z_contact:          [B, z_dim]
policy_tokens:      [B, N_tokens, d_model]
decoder_hidden:     [B, K, d_model]
pred_action:        [B, K, action_dim]
pred_force:         [B, K, 6]
```

---

## 5. Timestamp Alignment

The HDF5 file contains separate timestamp streams:

```text
timestamps/state
timestamps/state_episode
timestamps/force
timestamps/force_episode
timestamps/image
timestamps/image_episode
```

Prefer `*_episode` timestamps for within-episode alignment.

Use:

```text
timestamps/state_episode  -> state time base
timestamps/force_episode  -> force time base
timestamps/image_episode  -> image time base
```

The dataset should sample on the state timeline by default because actions are derived from state trajectories.

For each state index `i`:

```text
t = timestamps/state_episode[i]
```

### Current state

Directly use:

```python
qpos_i = joint_pos[i]
qvel_i = joint_vel[i]
tau_i  = joint_torque[i]
ee_i   = ee_pose[i]
```

### Current images

Find nearest image index:

```python
j = nearest_index(timestamps/image_episode, t)
```

Then load:

```python
ee_cam[j]
base_top_cam[j]
```

Return them as:

```python
images = stack([ee_cam[j], base_top_cam[j]])
```

Then convert:

```python
[N_cam, H, W, 3] uint8
```

to:

```python
[N_cam, 3, H, W] float32
```

and normalize to `[0, 1]`, followed by ImageNet normalization if using pretrained ResNet18.

### Past force window

Use force timestamps:

```python
force_ts = timestamps/force_episode
```

For current time `t`, construct:

```text
F_{t - T_window : t}
```

Then resample to fixed length `L`.

Example:

```python
force_window_duration = 0.25  # seconds
force_window_len = 50
```

Select:

```python
mask = (force_ts >= t - force_window_duration) & (force_ts <= t)
```

Then resample selected `ft_wrench` to:

```python
[L, 6]
```

If the window is too short at episode start, pad using the earliest available force sample.

### Future force chunk

For each future state step:

```python
t_m = timestamps/state_episode[i + m]
```

Find nearest force sample:

```python
force_idx_m = nearest_index(force_ts, t_m)
```

Return:

```python
future_force_chunk[m] = ft_wrench[force_idx_m]
```

Shape:

```python
[K, 6]
```

### Future action chunk

For `action_mode: joint_pos`:

```python
action_chunk[m] = joint_pos[i + m + action_offset]
```

Recommended first version:

```python
action_offset = 1
```

So:

```text
action_chunk = joint_pos[i+1 : i+K+1]
```

If insufficient future horizon exists at the end of an episode, drop invalid tail indices.

---

## 6. Valid Sampling Range

For each episode, valid state index `i` must satisfy:

```python
i >= 0
i + K + action_offset < N_state
```

Force window start can be before episode start because it can be padded.

However, future chunks must be valid.

Recommended valid range:

```python
valid_indices = range(0, N_state - K - action_offset)
```

Optionally avoid first few frames if force window padding should be avoided:

```python
min_time = force_window_duration
valid_indices = [i for i in range(...) if state_ts[i] >= min_time]
```

---

## 7. Dataset Class Requirements

Implement a dataset class such as:

```python
class ContactForceHDF5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_paths,
        camera_names=("ee_cam", "base_top_cam"),
        action_mode="joint_pos",
        chunk_len=50,
        force_window_len=50,
        force_window_duration=0.25,
        image_size=(224, 224),
        normalize=True,
        load_in_memory=False,
    ):
        ...
```

`__getitem__` should return:

```python
{
    "images": torch.FloatTensor,             # [N_cam, 3, H, W]
    "qpos": torch.FloatTensor,               # [7]
    "qvel": torch.FloatTensor,               # [7]
    "joint_torque": torch.FloatTensor,       # [7]
    "ee_pose": torch.FloatTensor,            # [7]
    "force_window": torch.FloatTensor,       # [L, 6]
    "action_chunk": torch.FloatTensor,       # [K, action_dim]
    "future_force_chunk": torch.FloatTensor, # [K, 6]
    "episode_path": str,
    "state_index": int,
}
```

Optional debug fields:

```python
{
    "t_state": float,
    "image_index": int,
    "force_indices": np.ndarray,
}
```

---

## 8. Data Normalization

Compute normalization statistics from the training set only.

Normalize:

```text
qpos
qvel
joint_torque
ee_pose
ft_wrench
action_chunk
future_force_chunk
force_window
```

Recommended first version:

```python
qpos_mean, qpos_std
qvel_mean, qvel_std
ee_pose_mean, ee_pose_std
force_mean, force_std
action_mean, action_std
```

Important:

Use the same `force_mean` and `force_std` for:

```text
force_window
future_force_chunk
pred_force
```

Store normalization statistics in:

```text
normalization_stats.pkl
```

or inside checkpoint.

For images:

```python
images = images.float() / 255.0
```

Then apply standard ImageNet normalization if using pretrained ResNet18:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## 9. Model Inputs

Training forward:

```python
outputs = model(
    images=images,
    qpos=qpos,
    force_window=force_window,
    action_chunk=action_chunk,
    future_force_chunk=future_force_chunk,
    is_training=True,
)
```

Inference forward:

```python
outputs = model(
    images=images,
    qpos=qpos,
    force_window=force_window,
    action_chunk=None,
    future_force_chunk=None,
    is_training=False,
)
```

Inference must not call:

```text
motion posterior encoder
contact posterior encoder
```

---

## 10. Model Architecture

### 10.1 Vision Encoder: ResNet18

Use **ResNet18** for image processing.

Input:

```python
images: [B, N_cam, 3, H, W]
```

Output:

```python
visual_tokens: [B, N_v, d_model]
```

Recommended first implementation:

1. Flatten camera dimension into batch:

```python
x = images.reshape(B * N_cam, 3, H, W)
```

2. Pass each image through ResNet18 up to the final convolutional feature map, not the classification head:

```text
ResNet18 stem + layer1 + layer2 + layer3 + layer4
```

3. For input size `224 x 224`, ResNet18 typically outputs:

```python
feat: [B * N_cam, 512, 7, 7]
```

4. Flatten spatial dimensions into visual tokens:

```python
feat = feat.flatten(2).transpose(1, 2)
# [B * N_cam, 49, 512]
```

5. Restore camera dimension:

```python
feat = feat.reshape(B, N_cam * 49, 512)
# [B, N_v, 512], where N_v = N_cam * 49
```

6. Project to `d_model` if needed:

```python
visual_tokens = visual_proj(feat)
# [B, N_v, d_model]
```

Recommended class:

```python
class ResNet18VisionEncoder(nn.Module):
    def __init__(self, d_model=512, pretrained=True, freeze_backbone=False):
        ...
```

Implementation notes:

- Remove `avgpool` and `fc`.
- Use the convolutional feature map as spatial visual tokens.
- If `d_model == 512`, `visual_proj` can be identity.
- If using pretrained ImageNet weights, apply ImageNet normalization.
- For two cameras and 224x224 images, `N_v = 2 * 7 * 7 = 98`.
- If memory becomes an issue, optionally reduce visual tokens using a 1x1 conv, spatial pooling, or camera-wise pooling later.

---

### 10.2 Joint Token Encoder

Input:

```python
qpos: [B, 7]
```

Output:

```python
z_q: [B, d_model]
```

Implementation:

```python
z_q = JointMLP(qpos)
```

Optional later:

include qvel, joint_torque, or ee_pose:

```python
robot_state = concat(qpos, qvel, joint_torque, ee_pose)
```

First baseline:

```text
use qpos only
```

---

### 10.3 Temporal Force Encoder

Input:

```python
force_window: [B, L, 6]
```

Output:

```python
z_F_online: [B, d_model]
```

Recommended implementation:

```text
force projection -> CLS token -> positional embedding -> TransformerEncoder -> CLS output
```

Pseudo-code:

```python
class TemporalForceEncoder(nn.Module):
    def forward(self, force_window):
        x = self.force_proj(force_window)      # [B, L, d_model]
        cls = self.cls_token.expand(B, 1, d_model)
        x = torch.cat([cls, x], dim=1)         # [B, L+1, d_model]
        x = x + self.pos_embed[:, :L+1]
        h = self.encoder(x)
        return h[:, 0]                         # [B, d_model]
```

---

### 10.4 Force-Vision Cross-Attention

Purpose:

Use force as query and visual tokens as key/value.

Inputs:

```python
z_F_online:    [B, d_model]
visual_tokens: [B, N_v, d_model]
```

Computation:

```text
Q = z_F_online
K = visual_tokens
V = visual_tokens
```

Output:

```python
z_VF: [B, d_model]
```

Implementation:

```python
q = z_F_online[:, None, :]   # [B, 1, d_model]
z_VF, attn = cross_attn(q, visual_tokens, visual_tokens)
z_VF = z_VF[:, 0, :]
```

Use `nn.MultiheadAttention(batch_first=True)`.

---

### 10.5 Motion Posterior Encoder

Training only.

Inputs:

```python
qpos: [B, 7]
action_chunk: [B, K, action_dim]
```

Outputs:

```python
mu_motion:     [B, z_dim]
logvar_motion: [B, z_dim]
z_motion:      [B, z_dim]
```

This models motion-style variation in action chunks.

Inference:

```python
z_motion = zeros([B, z_dim])
```

---

### 10.6 Contact Posterior Encoder

Training only.

Inputs:

```python
qpos: [B, 7]
action_chunk: [B, K, action_dim]
future_force_chunk: [B, K, 6]
```

Outputs:

```python
mu_contact:     [B, z_dim]
logvar_contact: [B, z_dim]
z_contact:      [B, z_dim]
```

This models latent contact-dynamics mode from future action-force chunks.

Inference baseline:

```python
z_contact = zeros([B, z_dim])
```

Later optional conditional prior:

```python
z_contact = contact_prior(images, qpos, force_window)
```

---

### 10.7 Policy Input Tokens

Assemble:

```python
policy_tokens = concat(
    visual_tokens,
    z_VF,
    z_q,
    z_F_online,
    z_motion,
    z_contact
)
```

More explicitly:

```python
tokens = [
    visual_tokens,                         # [B, N_v, d_model]
    z_VF[:, None, :],                      # [B, 1, d_model]
    z_q[:, None, :],                       # [B, 1, d_model]
    z_F_online[:, None, :],                # [B, 1, d_model]
    motion_latent_proj(z_motion)[:, None, :],
    contact_latent_proj(z_contact)[:, None, :],
]

policy_tokens = torch.cat(tokens, dim=1)
```

---

### 10.8 ACT-Style Transformer Policy

Input:

```python
policy_tokens: [B, N_tokens, d_model]
```

Output:

```python
decoder_hidden: [B, K, d_model]
```

The decoder uses future query embeddings:

```python
future_queries: [B, K, d_model]
```

---

### 10.9 Action Head

Input:

```python
decoder_hidden: [B, K, d_model]
```

Output:

```python
pred_action: [B, K, action_dim]
```

---

### 10.10 Force Head

Input:

```python
decoder_hidden: [B, K, d_model]
z_contact: [B, z_dim]
```

Output:

```python
pred_force: [B, K, 6]
```

The force head must explicitly receive `z_contact`.

Implementation:

```python
z_contact_rep = z_contact[:, None, :].expand(-1, K, -1)
force_input = torch.cat([decoder_hidden, z_contact_rep], dim=-1)
pred_force = force_head(force_input)
```

This ensures:

```text
z_contact -> pred_force -> L_force
```

so the force loss directly supervises the contact-dynamics latent.

---

## 11. Reparameterization and KL

Use standard VAE reparameterization:

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

KL to standard normal:

```python
def kl_normal(mu, logvar):
    return -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=-1
    ).mean()
```

---

## 12. Training Loss

Base loss:

```python
L_action = F.l1_loss(pred_action, action_chunk)
L_force = F.l1_loss(pred_force, future_force_chunk)

KL_motion = kl_normal(mu_motion, logvar_motion)
KL_contact = kl_normal(mu_contact, logvar_contact)

loss = (
    L_action
    + lambda_force * L_force
    + beta_motion * KL_motion
    + beta_contact * KL_contact
)
```

Use KL warm-up:

```text
beta_motion:  0 -> beta_motion_max
beta_contact: 0 -> beta_contact_max
```

Log:

```python
{
    "loss_total": loss.item(),
    "loss_action": L_action.item(),
    "loss_force": L_force.item(),
    "kl_motion": KL_motion.item(),
    "kl_contact": KL_contact.item(),
    "beta_motion": beta_motion,
    "beta_contact": beta_contact,
}
```

---

## 13. Optional Conditional Prior for z_contact

This is optional and should not be implemented before the baseline works.

Goal:

Predict `z_contact` from online inputs:

```python
p_psi(z_contact | I_t, q_t, F_{t-L:t})
```

Inputs:

```python
z_VF
z_q
z_F_online
```

Outputs:

```python
mu_contact_prior:     [B, z_dim]
logvar_contact_prior: [B, z_dim]
z_contact_prior:      [B, z_dim]
```

Recommended staged training:

### Stage A

Train posterior and decoder with weak standard normal KL:

```text
KL(q_contact || N(0, I))
```

### Stage B

Train prior by posterior distillation with stop-gradient:

```python
L_prior = KL(stop_gradient(q_contact) || p_prior)
```

or simpler:

```python
L_prior = mse(mu_contact_prior, mu_contact.detach())
```

### Stage C

Optional joint fine-tuning:

```text
KL(q_contact || p_prior)
+ weak KL(p_prior || N(0, I))
```

Inference:

```python
z_contact = mu_contact_prior
```

Baseline inference:

```python
z_contact = zeros([B, z_dim])
```

---

## 14. Training vs Inference Rules

Training may use:

```python
images
qpos
force_window
action_chunk
future_force_chunk
```

Inference may use only:

```python
images
qpos
force_window
```

Inference must not use:

```python
action_chunk
future_force_chunk
motion posterior encoder
contact posterior encoder
```

Add explicit assertions to enforce this.

---

## 15. Safety Rules for Deployment

Before deploying on a robot, pass predicted actions through a safety wrapper.

Required checks:

```text
1. joint position limits
2. joint velocity limits
3. maximum action delta
4. maximum action norm
5. force magnitude threshold
6. emergency stop condition
7. workspace boundary condition
```

If force exceeds threshold:

```python
return safe_stop_action
```

or hold current pose.

---

## 16. Implementation Order for Codex

Implement in this order:

```text
1. HDF5 dataset reader for the current episode structure
2. Timestamp alignment and sample slicing
3. Dataset unit tests with fake HDF5
4. Normalization statistics
5. ResNet18VisionEncoder
6. TemporalForceEncoder
7. ForceVisionCrossAttention
8. ContactPosteriorEncoder
9. ForceHead
10. Policy token integration
11. Training loss and logging
12. One-batch forward/backward test
13. Tiny overfit test
14. Full training
15. Inference wrapper
16. Optional conditional prior
17. Simulation test
18. Real-robot deployment with safety wrapper
```

---

## 17. Required Unit Tests

### HDF5 dataset test

Create fake HDF5 with:

```python
N_state = 100
N_force = 500
N_image = 50
H, W = 64, 64
```

Check:

```python
sample["images"].shape == [2, 3, H, W]
sample["qpos"].shape == [7]
sample["force_window"].shape == [L, 6]
sample["action_chunk"].shape == [K, action_dim]
sample["future_force_chunk"].shape == [K, 6]
```

### Timestamp alignment test

Verify:

```text
force_window uses only force timestamps <= t
future_force_chunk uses timestamps >= t
image index is nearest to t
```

### Model module tests

```python
ResNet18VisionEncoder:
[B, N_cam, 3, H, W] -> [B, N_cam * 49, d_model] for 224x224 input

TemporalForceEncoder:
[B, L, 6] -> [B, d_model]

ForceVisionCrossAttention:
[B, d_model], [B, N_v, d_model] -> [B, d_model]

ContactPosteriorEncoder:
[B, 7], [B, K, action_dim], [B, K, 6]
-> mu, logvar, z each [B, z_dim]

Policy forward:
pred_action [B, K, action_dim]
pred_force [B, K, 6]
```

### Inference no-leakage test

In inference mode, assert:

```python
action_chunk is None
future_force_chunk is None
motion posterior encoder is not called
contact posterior encoder is not called
```

---

## 18. Debugging Protocol

Before full training:

```text
1. Print one dataset sample.
2. Check all shapes.
3. Run one batch forward pass.
4. Run one batch backward pass.
5. Overfit one episode.
6. Overfit ten episodes.
7. Check L_action decreases.
8. Check L_force decreases.
9. Check KL_contact is nonzero but not exploding.
10. Visualize mu_contact across contact phases.
```

---

## 19. Failure Modes to Watch

### Future leakage

Invalid:

```text
future_force_chunk enters online inference path
action_chunk enters online inference path
```

### Wrong force window

Invalid:

```text
force_window = F_{t:t+L}
```

Correct:

```text
force_window = F_{t-L:t}
```

### Bad action definition

If action is defined as current joint position:

```text
action_t = joint_pos[t]
```

then the model may learn identity mapping.

Prefer:

```text
action_t = joint_pos[t+1]
```

or future target positions.

### Posterior collapse

If:

```text
KL_contact ≈ 0
```

then `z_contact` may carry no information.

Use lower `beta_contact` and KL warm-up.

### Force scale mismatch

If force values are too large or too small relative to action loss, tune:

```text
lambda_force
```

and normalize force.

### Force contains non-contact artifacts

If `ft_wrench` includes gravity/bias artifacts, the model may learn pose-dependent fake force.

If possible, train on compensated wrench. If only raw wrench is available, document this limitation.

---

## 20. First Baseline Config

```yaml
dataset:
  camera_names: ["ee_cam", "base_top_cam"]
  action_mode: "joint_pos"
  chunk_len: 50
  force_window_len: 50
  force_window_duration: 0.25
  image_size: [224, 224]

model:
  vision_encoder: "resnet18"
  pretrained_resnet18: true
  freeze_resnet18: false
  d_model: 512
  z_dim: 32
  force_dim: 6
  action_dim: 7
  use_force_encoder: true
  use_cross_attention: true
  use_contact_latent: true
  use_contact_prior: false

loss:
  lambda_force: 0.1
  beta_motion_max: 1.0e-4
  beta_contact_max: 1.0e-4
  kl_warmup_steps: 5000

inference:
  z_motion: "zero"
  z_contact: "zero"
```

After baseline works:

```yaml
model:
  use_contact_prior: true

contact_prior:
  training: "posterior_distillation"
  inference: "mean"
```

---

## 21. Codex Operating Rules

When modifying code:

1. Do not rewrite unrelated modules.
2. Preserve backward compatibility with the existing ACT baseline.
3. Add new modules in separate files where possible.
4. Add tests before large integrations.
5. Print tensor shapes in debug mode.
6. Do not modify real-robot deployment code until offline training passes.
7. After every change, summarize:
   - modified files
   - new classes/functions
   - tensor shapes
   - how to run tests
   - known limitations
