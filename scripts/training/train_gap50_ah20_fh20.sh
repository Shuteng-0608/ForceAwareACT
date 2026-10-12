#!/usr/bin/env bash
# Usage: bash scripts/training/train_gap50_ah20_fh20.sh [contact|motion] [force3|torque7] [extra CLI args...]
# 0.2 s actions, 0.2 s future force and 0.2 s force history.
set -euo pipefail
model_kind="${1:-contact}"
force_kind="${2:-force3}"
case "$model_kind" in contact|motion) ;; *) echo 'Model must be contact or motion' >&2; exit 2 ;; esac
case "$force_kind" in
  force3) data_config=configs/real_robot/right_arm_force3.json ;;
  torque7) data_config=configs/real_robot/right_arm.json ;;
  *) echo 'Force representation must be force3 or torque7' >&2; exit 2 ;;
esac
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate forceact
cd "$task_repo_root"
exec python -u scripts/train_real_robot.py \
  --mode train --model "$model_kind" \
  --data-root "${FORCEACT_DATA_ROOT:-${HOME}/Downloads/1mm-gap-10points_hdf5_raw_v2}" \
  --data-config "$data_config" \
  --optimization-config configs/experiments/1mm_gap_50_10points_v1/optimization.json \
  --split configs/datasets/1mm_gap_50_10points_v2/split.json \
  --output-dir "outputs/1mm_gap_50_10points_v1/runs/${model_kind}_${force_kind}_ah20_fh20_seed0_b8_es3000_p20_r001" \
  --train-sampling-mode episodic_uniform \
  --batch-size 8 --chunk-len 20 --force-prediction-steps 20 --strict-force-horizon --seed 0 \
  --max-sampling-epochs 20000 --minimum-sampling-epochs 3000 \
  --early-stop-patience 20 --early-stop-min-relative-improvement 0.001 \
  --validation-interval 200 --validation-anchor-stride 10 \
  --selection-metric episode_macro_action_mae_rad \
  --checkpoint-interval 200 --record-timing --device cuda --threads 4 \
  "${@:3}"
