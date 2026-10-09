#!/usr/bin/env bash
set -euo pipefail
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate forceact
cd "$task_repo_root"
exec python scripts/monitor_real_robot_training.py \
  --run-dir outputs/1mm_gap_50_10points_v1/runs/motion_force3_seed0_b8 \
  --watch --interval 5 "$@"
