#!/usr/bin/env bash
# 1 s actions, 0.2 s future flange force, 0.2 s force history.
set -euo pipefail
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate forceact
cd "$task_repo_root"
exec python -u scripts/train_real_robot.py \
  --mode train --model contact \
  --data-root "${FORCEACT_DATA_ROOT:-${HOME}/Downloads/1mm-gap-10points_hdf5_raw_v2}" \
  --data-config configs/real_robot/right_arm_force3.json \
  --optimization-config configs/experiments/1mm_gap_50_10points_v1/optimization.json \
  --split configs/datasets/1mm_gap_50_10points_v2/split.json \
  --output-dir outputs/1mm_gap_50_10points_v1/runs/contact_force3_fh20_seed0_b8_es3000_p20_r001 \
  --train-sampling-mode episodic_uniform \
  --batch-size 8 --chunk-len 100 --force-prediction-steps 20 --seed 0 \
  --max-sampling-epochs 20000 --minimum-sampling-epochs 3000 \
  --early-stop-patience 20 --early-stop-min-relative-improvement 0.001 \
  --validation-interval 200 --validation-anchor-stride 10 \
  --selection-metric episode_macro_action_mae_rad \
  --checkpoint-interval 200 --record-timing --device cuda --threads 4 \
  "$@"
