#!/usr/bin/env bash
# Usage: bash scripts/training/train_x14y08_contact_prior_kl.sh [beta1|warmup1000] [test|train] [extra CLI args...]
# Wrench6, history20x10ms, action/force20x10ms, prior_mean; default is 10-update TEST.
set -euo pipefail
profile="${1:-beta1}"
run_mode="${2:-test}"
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$profile" in
  beta1) schedule=(--posterior-kl-schedule constant --posterior-kl-warmup-epochs 0)
         optimization="$task_repo_root/configs/experiments/1mm_x14y08_contact_prior_kl_v1/beta1_optimization.json" ;;
  warmup1000) schedule=(--posterior-kl-schedule linear_warmup --posterior-kl-warmup-epochs 1000)
              optimization="$task_repo_root/configs/experiments/1mm_x14y08_55_wrench6_v1/optimization.json" ;;
  *) echo 'Profile must be beta1 or warmup1000' >&2; exit 2 ;;
esac
case "$run_mode" in test|train) ;; *) echo 'Mode must be test or train' >&2; exit 2 ;; esac
export FORCEACT_OUTPUT_ROOT="${FORCEACT_OUTPUT_ROOT:-$task_repo_root/outputs/1mm_x14y08_55_wrench6_prior_kl_v1}/$profile"
exec bash "$task_repo_root/scripts/training/train_x14y08_contact_prior.sh" wrench6 10 "$run_mode" \
  "${@:3}" --optimization-config "$optimization" "${schedule[@]}"
