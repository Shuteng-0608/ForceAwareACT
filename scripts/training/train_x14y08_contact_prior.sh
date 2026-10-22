#!/usr/bin/env bash
# Usage: bash scripts/training/train_x14y08_contact_prior.sh [wrench6|sensor_force3] [10|50] [test|train] [extra CLI args...]
# Posterior training is unchanged; prior mean drives validation/selection, zero is reference-only.
set -euo pipefail
representation="${1:-wrench6}"
history_ms="${2:-10}"
run_mode="${3:-test}"
case "$representation" in wrench6|sensor_force3) ;; *) echo 'Representation must be wrench6 or sensor_force3' >&2; exit 2 ;; esac
case "$history_ms" in 10|50) ;; *) echo 'History interval must be 10 or 50 ms' >&2; exit 2 ;; esac
case "$run_mode" in test|train) ;; *) echo 'Mode must be test or train' >&2; exit 2 ;; esac
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
output_root="${FORCEACT_OUTPUT_ROOT:-$task_repo_root/outputs/1mm_x14y08_55_${representation}_prior_mean_h${history_ms}ms_v1}"
if [[ "$run_mode" == test ]]; then
  output="$output_root/checks/contact_fullnet"
else
  output="$output_root/runs/contact_${representation}_prior_mean_h${history_ms}ms_ah20_fh20_seed0_b8_es3000_p20_r001"
fi
if [[ "$history_ms" == 50 ]]; then
  launcher=("$task_repo_root/scripts/training/train_x14y08_history50ms.sh" "$representation" contact "$run_mode")
else
  launcher=("$task_repo_root/scripts/training/train_x14y08_${representation}.sh" contact "$run_mode")
fi
exec bash "${launcher[@]}" --output-dir "$output" "${@:4}" --contact-latent-mode prior_mean
