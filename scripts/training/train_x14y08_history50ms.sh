#!/usr/bin/env bash
# Usage: bash scripts/training/train_x14y08_history50ms.sh [wrench6|sensor_force3] [contact|motion] [test|train] [extra CLI args...]
# Change only causal history packing: 4 x 50ms, capacity 20. Future stays 20 x 10ms, capacity 8.
# Delegate every optimization, sampling, horizon and multi-metric stopping option to the existing launcher.
set -euo pipefail
representation="${1:-wrench6}"
model_kind="${2:-contact}"
run_mode="${3:-test}"
case "$representation" in wrench6|sensor_force3) ;; *) echo 'Representation must be wrench6 or sensor_force3' >&2; exit 2 ;; esac
case "$model_kind" in contact|motion) ;; *) echo 'Model must be contact or motion' >&2; exit 2 ;; esac
case "$run_mode" in test|train) ;; *) echo 'Mode must be test or train' >&2; exit 2 ;; esac
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
output_root="$task_repo_root/outputs/1mm_x14y08_55_${representation}_history50ms_multimetric_v1"
if [[ "$run_mode" == test ]]; then
  output="$output_root/checks/${model_kind}_fullnet"
else
  output="$output_root/runs/${model_kind}_${representation}_h4x50ms_ah20_fh20_seed0_b8_es3000_p20_r001"
fi
exec bash "$task_repo_root/scripts/training/train_x14y08_${representation}.sh" "$model_kind" "$run_mode" \
  --data-config "$task_repo_root/configs/real_robot/right_arm_${representation}_history50ms.json" \
  --output-dir "$output" "${@:4}"
