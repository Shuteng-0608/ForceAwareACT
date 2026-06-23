# ForceAwareACT

ForceAwareACT is a research prototype for contact-rich visuomotor imitation learning. It implements a Force-Aware ACT-style Transformer policy for peg-in-hole manipulation using multi-camera images, robot joint state, and wrist 6-axis force/torque wrench input.

The policy combines visual tokens, temporal force-window encoding, force-vision cross-attention, posterior motion/contact latents, and a conditional contact prior for deployable inference.

## Artifact Warning

This repository does not include real HDF5 datasets, checkpoints, generated outputs, CSV logs, or plots. Large artifacts are intentionally ignored by git. Use this repository with local data and generated outputs placed outside version control.

## Repository Structure

```text
src/force_aware_act/data/       HDF5 dataset reader, image preprocessing, normalization utilities
src/force_aware_act/models/     Vision, force, latent, attention, policy, and prediction modules
src/force_aware_act/training/   Training loss utilities and KL/prior-distillation helpers
scripts/                       Offline inspection, training, evaluation, and analysis scripts
tests/                         Unit tests for datasets, models, losses, and policy behavior
configs/                       Lightweight split files and experiment command recipes
docs/                          Design notes and experiment documentation
```

For a code-grounded walkthrough of the algorithm implementation, data flow,
training losses, inference modes, and MuJoCo rollout semantics, see
`docs/ALGORITHM_FRAMEWORK_README.md`.

## Data Format

The HDF5 episode reader expects contact-rich manipulation episodes with this schema at a high level:

```text
observations/ee_pose                 [N_state, 7]
observations/joint_pos               [N_state, 7]
observations/joint_vel               [N_state, 7]
observations/joint_torque            [N_state, 7]
observations/ft_wrench               [N_force, 6]
observations/images/ee_cam           [N_image, H, W, 3]
observations/images/base_top_cam     [N_image, H, W, 3]
timestamps/state_episode             [N_state]
timestamps/force_episode             [N_force]
timestamps/image_episode             [N_image]
episode_metadata/
```

The dataset uses past force samples for the online force window and future state-aligned chunks for action and future-force supervision during training.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
PYTHONPATH=src python -m pytest -q
```

## Quick-Start Workflow

1. Prepare an episode list, such as `configs/splits/peg_in_hole_all10.txt`, or create a local episode-list file with one HDF5 path per line.
2. Compute normalization statistics.
3. Run Stage 1 policy training.
4. Run Stage 2 contact-prior distillation.
5. Evaluate zero/prior/posterior inference modes.
6. Analyze posterior and prior contact latents.

### Compute Normalization Stats

```bash
PYTHONPATH=src python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/peg_in_hole_all10.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output outputs/normalization_stats.pt
```

### Stage 1 Training

```bash
PYTHONPATH=src python scripts/train_minimal.py \
  --episode-list configs/splits/peg_in_hole_all10.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --normalization-stats outputs/normalization_stats.pt \
  --max-steps 1000 \
  --batch-size 4 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --output-dir outputs/stage1 \
  --log-csv outputs/stage1/train_log.csv
```

### Stage 2 Contact-Prior Distillation

```bash
PYTHONPATH=src python scripts/train_contact_prior_stage2.py \
  --episode-list configs/splits/peg_in_hole_all10.txt \
  --checkpoint outputs/stage1/checkpoint.pt \
  --normalization-stats outputs/normalization_stats.pt \
  --max-steps 3000 \
  --batch-size 4 \
  --output-dir outputs/stage2 \
  --log-csv outputs/stage2/train_log.csv
```

### Evaluate Inference Modes

```bash
PYTHONPATH=src python scripts/evaluate_inference_modes.py \
  --episode-list configs/splits/peg_in_hole_all10.txt \
  --checkpoint outputs/stage2/checkpoint.pt \
  --normalization-stats outputs/normalization_stats.pt \
  --batch-size 8 \
  --max-batches 500 \
  --output-csv outputs/stage2/inference_eval.csv
```

### Analyze Contact Latents

See `configs/experiments/` and `docs/EXPERIMENTS.md` for force-balanced latent analysis and prior-vs-posterior overlay commands.

## Model Architecture

- Vision encoder: ResNet18 converts `[B, N_cam, 3, H, W]` images into spatial visual tokens.
- Joint encoder: current `qpos` is embedded as an online joint token.
- Force encoder: a temporal force-window Transformer encodes past wrist wrench samples into an online force token.
- Force-vision cross-attention: the online force token is the query; visual tokens are keys and values.
- Motion posterior encoder: predicts `z_motion` from `qpos` and the future action chunk.
- Contact posterior encoder: predicts `z_contact` from `qpos`, future action chunk, and future force chunk.
- Conditional contact prior: predicts a deployable contact latent from online features without future labels.
- ACT-style Transformer: policy encoder-decoder maps context tokens and future queries to decoder hidden states.
- Prediction heads: `ActionHead` predicts future actions; `ForceHead` predicts future wrench chunks.

## Training Losses

The Stage 1 training objective is:

```text
L = L_action
  + lambda_force L_force
  + beta_motion KL_motion
  + beta_contact KL_contact
  + lambda_prior L_prior
```

`L_action` and `L_force` are supervised prediction losses. `KL_motion` and `KL_contact` regularize posterior latents. `L_prior` is optional contact-prior distillation against the contact posterior.

Stage 2 freezes the policy and posterior teacher, then trains only `ContactPriorEncoder`. Posterior targets are detached, so gradients flow only through the conditional contact prior.

## Inference Modes

- `zero`: deployable baseline with `z_contact = 0`.
- `prior`: deployable inference using deterministic `z_contact = mu_contact_prior`.
- `posterior`: oracle/debug mode using future labels. This is useful for training and analysis, but not deployment.

## Current Experimental Status

On small local peg-in-hole data, prior inference improved future-force prediction over the zero baseline in offline evaluation. These are preliminary local results, not final performance claims.

The current 8/2 split held-out evaluation is useful for debugging, but it is not a publishable generalization result. Stronger conclusions require larger and more diverse train/validation splits, likely 50-100 episodes or more.

## Development Notes

- Run tests before committing:

```bash
PYTHONPATH=src python -m pytest -q
```

- Keep HDF5 data, checkpoints, generated outputs, CSV logs, and plots out of git.
- Prefer a private repository while research is ongoing.
- The implementation design is documented in `docs/contact_dynamics_force_act_design.md`.

## Citation

TBD

## License

TBD
