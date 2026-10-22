#!/usr/bin/env bash
# Usage: bash scripts/training/train_x14y08_wrench6.sh [act|contact|motion] [test|train] [extra CLI args...]
# Multi-metric early stopping; the architecture, loss weights and training budget are unchanged.
# Default is a bounded 10-update full-network check. Formal training requires explicit 'train'.
set -euo pipefail
model_kind="${1:-contact}"
run_mode="${2:-test}"
case "$model_kind" in
  act) force_args=(); run_name="act_baseline_ah20_seed0_b8_es3000_p20_r001" ;;
  contact|motion) force_args=(--force-prediction-steps 20 --strict-force-horizon)
                  run_name="${model_kind}_wrench6_ah20_fh20_seed0_b8_es3000_p20_r001" ;;
  *) echo 'Model must be act, contact or motion' >&2; exit 2 ;;
esac
case "$run_mode" in
  test) budget=(--steps 10); output="outputs/1mm_x14y08_55_wrench6_multimetric_v1/checks/${model_kind}_fullnet" ;;
  train) budget=(--max-sampling-epochs 20000 --minimum-sampling-epochs 3000 --early-stop-patience 20 --early-stop-min-relative-improvement 0.001)
         output="outputs/1mm_x14y08_55_wrench6_multimetric_v1/runs/${run_name}" ;;
  *) echo 'Mode must be test or train' >&2; exit 2 ;;
esac
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${FORCEACT_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate forceact
cd "$task_repo_root"
exec python -u scripts/train_real_robot.py \
  --mode "$run_mode" --model "$model_kind" \
  --data-root "${FORCEACT_DATA_ROOT:-${HOME}/Downloads/1mm-x14y08_hdf5_raw_v3}" \
  --data-config configs/real_robot/right_arm_wrench6.json \
  --optimization-config configs/experiments/1mm_x14y08_55_wrench6_v1/optimization.json \
  --split configs/datasets/1mm_x14y08_55_wrench6_v1/split.json \
  --output-dir "$output" \
  --train-sampling-mode episodic_uniform --batch-size 8 \
  --chunk-len 20 "${force_args[@]}" --seed 0 \
  "${budget[@]}" --validation-interval 250 --validation-anchor-stride 10 \
  --selection-metric episode_macro_action_mae_rad --checkpoint-interval 250 \
  --early-stop-metrics multi \
  --record-timing --device cuda --threads 4 "${@:3}"
