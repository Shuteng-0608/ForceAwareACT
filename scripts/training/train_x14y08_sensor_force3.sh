#!/usr/bin/env bash
# Usage: bash scripts/training/train_x14y08_sensor_force3.sh [contact|motion] [test|train] [extra CLI args...]
# Source: raw_v3 ee_wrench[:, :3], sensor frame. No joint-torque inversion.
# Keep the original six-axis experiment's force-only coefficients: endpoint 0.5, interval 0.25.
# Multi-metric early stopping; the architecture, loss weights and training budget are unchanged.
# Default: bounded 10-update full-network check; formal training requires explicit 'train'.
set -euo pipefail
model_kind="${1:-contact}"
run_mode="${2:-test}"
case "$model_kind" in contact|motion) ;; *) echo 'Model must be contact or motion' >&2; exit 2 ;; esac
case "$run_mode" in
  test) budget=(--steps 10); output="outputs/1mm_x14y08_55_sensor_force3_multimetric_v1/checks/${model_kind}_fullnet" ;;
  train) budget=(--max-sampling-epochs 20000 --minimum-sampling-epochs 3000 --early-stop-patience 20 --early-stop-min-relative-improvement 0.001)
         output="outputs/1mm_x14y08_55_sensor_force3_multimetric_v1/runs/${model_kind}_sensor_force3_ah20_fh20_seed0_b8_es3000_p20_r001" ;;
  *) echo 'Mode must be test or train' >&2; exit 2 ;;
esac
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${FORCEACT_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate forceact
cd "$task_repo_root"
exec python -u scripts/train_real_robot.py \
  --mode "$run_mode" --model "$model_kind" \
  --data-root "${FORCEACT_DATA_ROOT:-${HOME}/Downloads/1mm-x14y08_hdf5_raw_v3}" \
  --data-config configs/real_robot/right_arm_sensor_force3.json \
  --optimization-config configs/experiments/1mm_x14y08_55_sensor_force3_v1/optimization.json \
  --split configs/datasets/1mm_x14y08_55_wrench6_v1/split.json \
  --output-dir "$output" \
  --train-sampling-mode episodic_uniform --batch-size 8 \
  --chunk-len 20 --force-prediction-steps 20 --strict-force-horizon --seed 0 \
  "${budget[@]}" --validation-interval 250 --validation-anchor-stride 10 \
  --selection-metric episode_macro_action_mae_rad --checkpoint-interval 250 \
  --early-stop-metrics multi \
  --record-timing --device cuda --threads 4 "${@:3}"
