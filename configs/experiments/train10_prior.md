# Train10 Prior Experiment

Commands below assume the repository root as the working directory.

## 1. Compute Normalization Stats

```bash
PYTHONPATH=src .venv/bin/python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/peg_in_hole_all10.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output outputs/normalization_stats_10eps.pt
```

## 2. Stage 1 Joint Policy Training

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py \
  --episode-list configs/splits/peg_in_hole_all10.txt \
  --normalization-stats outputs/normalization_stats_10eps.pt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --max-steps 1000 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --output-dir outputs/minimal_train \
  --log-csv outputs/minimal_train/train_log_10eps_prior_1000.csv
```

## 3. Stage 2 Contact-Prior Distillation

```bash
PYTHONPATH=src .venv/bin/python scripts/train_contact_prior_stage2.py \
  --episode-list configs/splits/peg_in_hole_all10.txt \
  --checkpoint outputs/minimal_train/checkpoint.pt \
  --normalization-stats outputs/normalization_stats_10eps.pt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --max-steps 3000 \
  --output-dir outputs/contact_prior_stage2 \
  --log-csv outputs/contact_prior_stage2/train_log_3000.csv
```

## 4. Evaluate Inference Modes

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_inference_modes.py \
  --episode-list configs/splits/peg_in_hole_all10.txt \
  --checkpoint outputs/contact_prior_stage2/checkpoint.pt \
  --normalization-stats outputs/normalization_stats_10eps.pt \
  --batch-size 8 \
  --max-batches 50 \
  --output-csv outputs/contact_prior_stage2/inference_eval_50_batches.csv
```

## 5. Analyze Contact Latents: Force-Balanced Overlay

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_contact_latent.py \
  --episode-list configs/splits/peg_in_hole_all10.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --sampling-mode force_balanced \
  --max-samples 300 \
  outputs/contact_prior_stage2/checkpoint.pt \
  outputs/normalization_stats_10eps.pt \
  outputs/contact_prior_stage2/contact_latents_force_balanced.csv \
  --plot outputs/contact_prior_stage2/contact_latent_force_balanced.png \
  --color-by future_force_mean_raw
```

## 6. Analyze Prior-Posterior Overlay

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_contact_latent.py \
  --episode-list configs/splits/peg_in_hole_all10.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --sampling-mode force_balanced \
  --max-samples 300 \
  --include-prior \
  outputs/contact_prior_stage2/checkpoint.pt \
  outputs/normalization_stats_10eps.pt \
  outputs/contact_prior_stage2/contact_latents_prior_overlay.csv \
  --plot-prior-overlay outputs/contact_prior_stage2/contact_prior_overlay.png \
  --color-by future_force_mean_raw
```
