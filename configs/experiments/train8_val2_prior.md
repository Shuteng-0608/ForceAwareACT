# Train8 / Val2 Prior Experiment

Commands below assume the repository root as the working directory.

## 1. Compute Train8 Normalization Stats

```bash
PYTHONPATH=src .venv/bin/python scripts/compute_normalization_stats.py \
  --episode-list configs/splits/peg_in_hole_train8.txt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --output outputs/normalization_stats_train8.pt
```

## 2. Stage 1 Train on Train8

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minimal.py \
  --episode-list configs/splits/peg_in_hole_train8.txt \
  --normalization-stats outputs/normalization_stats_train8.pt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --max-steps 1000 \
  --lambda-prior 0.1 \
  --prior-loss-mode mse_mu \
  --output-dir outputs/train8_minimal_prior \
  --log-csv outputs/train8_minimal_prior/train_log.csv
```

## 3. Stage 2 Contact-Prior Distillation on Train8

```bash
PYTHONPATH=src .venv/bin/python scripts/train_contact_prior_stage2.py \
  --episode-list configs/splits/peg_in_hole_train8.txt \
  --checkpoint outputs/train8_minimal_prior/checkpoint.pt \
  --normalization-stats outputs/normalization_stats_train8.pt \
  --chunk-len 10 \
  --force-window-len 20 \
  --force-window-duration 0.25 \
  --max-steps 3000 \
  --output-dir outputs/train8_contact_prior_stage2 \
  --log-csv outputs/train8_contact_prior_stage2/train_log.csv
```

## 4. Evaluate Inference Modes on Val2

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_inference_modes.py \
  --episode-list configs/splits/peg_in_hole_val2.txt \
  --checkpoint outputs/train8_contact_prior_stage2/checkpoint.pt \
  --normalization-stats outputs/normalization_stats_train8.pt \
  --batch-size 8 \
  --max-batches 50 \
  --output-csv outputs/train8_contact_prior_stage2/inference_eval_val2.csv
```

## 5. Optional Train8 Evaluation

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_inference_modes.py \
  --episode-list configs/splits/peg_in_hole_train8.txt \
  --checkpoint outputs/train8_contact_prior_stage2/checkpoint.pt \
  --normalization-stats outputs/normalization_stats_train8.pt \
  --batch-size 8 \
  --max-batches 50 \
  --output-csv outputs/train8_contact_prior_stage2/inference_eval_train8.csv
```
