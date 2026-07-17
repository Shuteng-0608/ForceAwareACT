#!/usr/bin/env bash
set -euo pipefail

# Sequentially train the five early-stopping configurations on one GPU.
#
# The default paths match the hole_random_60mm_hmj train90/val10 experiment.
# Override any uppercase setting through the environment when starting a new
# run. The script deliberately refuses to overwrite a partial model run because
# the current trainers do not support resume.
#
# Usage:
#   bash scripts/run_hole_random_5model_earlystop.sh preflight
#   bash scripts/run_hole_random_5model_earlystop.sh train

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PREP_ROOT="${PREP_ROOT:-outputs/hole_random_60mm_hmj/train90_val10_seed20260716}"
TRAIN_LIST="${TRAIN_LIST:-$PREP_ROOT/train90.txt}"
VAL_LIST="${VAL_LIST:-$PREP_ROOT/val10.txt}"
STATS="${STATS:-$PREP_ROOT/normalization_stats_action_train90.pt}"
RUN_ROOT="${RUN_ROOT:-outputs/hole_random_60mm_hmj/earlystop_train90_splitseed20260716_run1}"
SMOKE_ROOT="${SMOKE_ROOT:-$RUN_ROOT/smoke}"
FORMAL_ROOT="${FORMAL_ROOT:-$RUN_ROOT/formal}"
PIPELINE_DIR="$FORMAL_ROOT/.pipeline"

CHUNK_LEN="${CHUNK_LEN:-10}"
FORCE_WINDOW_LEN="${FORCE_WINDOW_LEN:-20}"
FORCE_WINDOW_DURATION="${FORCE_WINDOW_DURATION:-0.25}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-224}"
IMAGE_WIDTH="${IMAGE_WIDTH:-224}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_STEPS="${MAX_STEPS:-200000}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
VAL_EVERY_EPOCHS="${VAL_EVERY_EPOCHS:-1}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-20}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.005}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
MODEL_SEED="${MODEL_SEED:-0}"
DEVICE="${DEVICE:-cuda}"
REQUIRE_SMOKE="${REQUIRE_SMOKE:-1}"
SKIP_CUDA_CHECK="${SKIP_CUDA_CHECK:-0}"

COMMON_DATA=(
  --episode-list "$TRAIN_LIST"
  --val-episode-list "$VAL_LIST"
  --normalization-stats "$STATS"
  --action-mode action
  --chunk-len "$CHUNK_LEN"
  --image-size "$IMAGE_HEIGHT" "$IMAGE_WIDTH"
  --camera-names ee_cam base_top_cam
)

FORCE_DATA=(
  "${COMMON_DATA[@]}"
  --force-window-len "$FORCE_WINDOW_LEN"
  --force-window-duration "$FORCE_WINDOW_DURATION"
)

FORMAL_CONTROL=(
  --max-steps "$MAX_STEPS"
  --max-epochs "$MAX_EPOCHS"
  --val-every-epochs "$VAL_EVERY_EPOCHS"
  --early-stop-min-epochs "$EARLY_STOP_MIN_EPOCHS"
  --early-stop-patience "$EARLY_STOP_PATIENCE"
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA"
  --early-stop-metric deploy_loss
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --learning-rate "$LEARNING_RATE"
  --save-every "$SAVE_EVERY"
  --device "$DEVICE"
)

timestamp() {
  date --iso-8601=seconds
}

write_status() {
  local path="$1"
  local status="$2"
  local temporary="${path}.tmp"
  mkdir -p "$(dirname "$path")"
  printf '%s\t%s\n' "$status" "$(timestamp)" > "$temporary"
  mv "$temporary" "$path"
}

count_entries() {
  awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$1"
}

validate_inputs() {
  "$PYTHON_BIN" -c 'import torch; print(f"torch={torch.__version__}")'

  for path in "$TRAIN_LIST" "$VAL_LIST" "$STATS"; do
    if [[ ! -f "$path" ]]; then
      echo "error: required file does not exist: $path" >&2
      return 1
    fi
  done

  local train_count
  local val_count
  train_count="$(count_entries "$TRAIN_LIST")"
  val_count="$(count_entries "$VAL_LIST")"
  if [[ "$train_count" != "90" || "$val_count" != "10" ]]; then
    echo "error: expected train90/val10, found train=$train_count val=$val_count" >&2
    return 1
  fi

  if [[ "$MODEL_SEED" != "0" ]]; then
    echo "error: this five-model queue uses seed0 output names; MODEL_SEED must be 0" >&2
    return 1
  fi

  TRAIN_LIST="$TRAIN_LIST" VAL_LIST="$VAL_LIST" STATS="$STATS" \
    EXPECTED_CHUNK_LEN="$CHUNK_LEN" \
    EXPECTED_FORCE_WINDOW_LEN="$FORCE_WINDOW_LEN" \
    EXPECTED_FORCE_WINDOW_DURATION="$FORCE_WINDOW_DURATION" \
    EXPECTED_IMAGE_HEIGHT="$IMAGE_HEIGHT" \
    EXPECTED_IMAGE_WIDTH="$IMAGE_WIDTH" \
    PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import torch


def entries(name: str) -> list[Path]:
    path = Path(os.environ[name])
    return [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


train = entries("TRAIN_LIST")
validation = entries("VAL_LIST")
if len(train) != len(set(train)):
    raise SystemExit("training list contains duplicate paths")
if len(validation) != len(set(validation)):
    raise SystemExit("validation list contains duplicate paths")
overlap = set(train) & set(validation)
if overlap:
    raise SystemExit(f"training/validation overlap: {sorted(overlap)[:3]}")
missing = [path for path in train + validation if not path.is_file()]
if missing:
    raise SystemExit(f"missing episode files: {missing[:3]}")

stats = torch.load(os.environ["STATS"], map_location="cpu", weights_only=False)
stats_paths = {
    Path(path).expanduser().resolve()
    for path in stats.get("episode_paths", ())
}
if stats_paths != set(train):
    raise SystemExit(
        "normalization stats provenance does not match train90: "
        f"stats={len(stats_paths)} train={len(train)}"
    )
expected = {
    "action_mode": "action",
    "chunk_len": int(os.environ["EXPECTED_CHUNK_LEN"]),
    "force_window_len": int(os.environ["EXPECTED_FORCE_WINDOW_LEN"]),
    "force_window_duration": float(os.environ["EXPECTED_FORCE_WINDOW_DURATION"]),
    "image_size": (
        int(os.environ["EXPECTED_IMAGE_HEIGHT"]),
        int(os.environ["EXPECTED_IMAGE_WIDTH"]),
    ),
    "camera_names": ("ee_cam", "base_top_cam"),
    "imagenet_normalize": False,
}
for key, value in expected.items():
    actual = stats.get(key)
    if key in {"image_size", "camera_names"} and actual is not None:
        actual = tuple(actual)
    if actual != value:
        raise SystemExit(f"stats {key} mismatch: expected {value!r}, got {actual!r}")

print("data_and_stats_preflight=passed")
print(f"train_episodes={len(train)} validation_episodes={len(validation)}")
PY

  if [[ "$REQUIRE_SMOKE" == "1" ]]; then
    local smoke_names=(
      contact_cvae_zero
      contact_cvae_prior
      motion_cvae
      dualzero
      act_baseline
    )
    local smoke_name
    for smoke_name in "${smoke_names[@]}"; do
      if [[ ! -f "$SMOKE_ROOT/$smoke_name/checkpoint_best.pt" ]]; then
        echo "error: required smoke checkpoint missing: $SMOKE_ROOT/$smoke_name/checkpoint_best.pt" >&2
        return 1
      fi
    done
    echo "smoke_preflight=passed"
  else
    echo "warning: smoke checkpoint requirement disabled"
  fi

  if [[ "$DEVICE" == cuda* && "$SKIP_CUDA_CHECK" != "1" ]]; then
    "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA preflight failed: no CUDA device is available")
print(f"cuda_preflight=passed device={torch.cuda.get_device_name(0)}")
PY
  fi

  echo "formal_root=$FORMAL_ROOT"
  echo "preflight=passed"
}

model_completed() {
  local model_name="$1"
  local output_dir="$FORMAL_ROOT/$model_name"
  local status_file="$PIPELINE_DIR/$model_name.status"
  [[ -f "$status_file" ]] \
    && [[ "$(cut -f1 "$status_file")" == "completed" ]] \
    && [[ -f "$output_dir/checkpoint.pt" ]] \
    && [[ -f "$output_dir/checkpoint_best.pt" ]]
}

run_model() {
  local model_name="$1"
  shift
  local output_dir="$FORMAL_ROOT/$model_name"
  local status_file="$PIPELINE_DIR/$model_name.status"
  local command=("$@")

  if model_completed "$model_name"; then
    echo "[$(timestamp)] skip completed model=$model_name"
    return 0
  fi

  if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    write_status "$status_file" partial
    echo "error: refusing to overwrite partial model directory: $output_dir" >&2
    echo "error: current trainers cannot resume; use a new RUN_ROOT for a clean rerun" >&2
    return 1
  fi

  mkdir -p "$output_dir"
  printf '%s\n' "$model_name" > "$PIPELINE_DIR/current_model"
  printf '%s\n' "$(timestamp)" > "$PIPELINE_DIR/$model_name.started_at"
  rm -f "$PIPELINE_DIR/$model_name.finished_at"
  write_status "$status_file" running

  {
    printf 'cd %q\n' "$REPO_ROOT"
    printf 'PYTHONPATH=src '
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "$output_dir/training_command.sh"

  echo "[$(timestamp)] start model=$model_name"
  local return_code=0
  PYTHONPATH=src "${command[@]}" \
    2>&1 | tee "$output_dir/console.log" || return_code=$?

  if [[ "$return_code" != "0" ]]; then
    printf '%s\n' "$(timestamp)" > "$PIPELINE_DIR/$model_name.finished_at"
    write_status "$status_file" failed
    write_status "$PIPELINE_DIR/pipeline.status" failed
    printf '%s\n' "$return_code" > "$output_dir/exit_code.txt"
    echo "[$(timestamp)] failed model=$model_name exit_code=$return_code" >&2
    return "$return_code"
  fi

  if [[ ! -f "$output_dir/checkpoint.pt" || ! -f "$output_dir/checkpoint_best.pt" ]]; then
    write_status "$status_file" failed
    write_status "$PIPELINE_DIR/pipeline.status" failed
    echo "error: model exited successfully but required checkpoints are missing: $model_name" >&2
    return 1
  fi

  printf '0\n' > "$output_dir/exit_code.txt"
  printf '%s\n' "$(timestamp)" > "$PIPELINE_DIR/$model_name.finished_at"
  write_status "$status_file" completed
  rm -f "$PIPELINE_DIR/current_model"
  echo "[$(timestamp)] complete model=$model_name"
}

train_all() {
  validate_inputs
  mkdir -p "$PIPELINE_DIR"
  printf '%s\n' "$$" > "$PIPELINE_DIR/runner.pid"
  printf '%s\n' "$(timestamp)" > "$PIPELINE_DIR/started_at"
  write_status "$PIPELINE_DIR/pipeline.status" running

  exec > >(tee -a "$PIPELINE_DIR/runner.log") 2>&1

  run_model contact_cvae_zero_seed0 \
    "$PYTHON_BIN" scripts/train_minimal.py \
    "${FORCE_DATA[@]}" \
    --policy-variant force_aware_contact_cvae \
    --train-latent-mode posterior \
    --train-contact-latent-mode posterior \
    --lambda-force 0.1 \
    --lambda-prior 0 \
    --prior-loss-mode mse_mu \
    --beta-contact-max 5e-4 \
    --warmup-steps 2000 \
    --validation-deployment-mode zero \
    --seed "$MODEL_SEED" \
    "${FORMAL_CONTROL[@]}" \
    --output-dir "$FORMAL_ROOT/contact_cvae_zero_seed0" \
    --log-csv "$FORMAL_ROOT/contact_cvae_zero_seed0/train_log.csv" \
    --validation-log "$FORMAL_ROOT/contact_cvae_zero_seed0/validation_log.csv"

  run_model contact_cvae_prior_seed0 \
    "$PYTHON_BIN" scripts/train_minimal.py \
    "${FORCE_DATA[@]}" \
    --policy-variant force_aware_contact_cvae \
    --train-latent-mode posterior \
    --train-contact-latent-mode posterior \
    --lambda-force 0.1 \
    --lambda-prior 0.1 \
    --prior-loss-mode mse_mu \
    --beta-contact-max 5e-4 \
    --warmup-steps 2000 \
    --validation-deployment-mode prior \
    --seed "$MODEL_SEED" \
    "${FORMAL_CONTROL[@]}" \
    --output-dir "$FORMAL_ROOT/contact_cvae_prior_seed0" \
    --log-csv "$FORMAL_ROOT/contact_cvae_prior_seed0/train_log.csv" \
    --validation-log "$FORMAL_ROOT/contact_cvae_prior_seed0/validation_log.csv"

  run_model motion_cvae_seed0 \
    "$PYTHON_BIN" scripts/train_minimal.py \
    "${FORCE_DATA[@]}" \
    --policy-variant force_aware_motion_cvae \
    --train-latent-mode posterior \
    --lambda-force 0.1 \
    --lambda-prior 0 \
    --beta-motion-max 5e-4 \
    --warmup-steps 2000 \
    --validation-deployment-mode zero \
    --seed "$MODEL_SEED" \
    "${FORMAL_CONTROL[@]}" \
    --output-dir "$FORMAL_ROOT/motion_cvae_seed0" \
    --log-csv "$FORMAL_ROOT/motion_cvae_seed0/train_log.csv" \
    --validation-log "$FORMAL_ROOT/motion_cvae_seed0/validation_log.csv"

  run_model dualzero_seed0 \
    "$PYTHON_BIN" scripts/train_minimal.py \
    "${FORCE_DATA[@]}" \
    --policy-variant force_aware_act \
    --train-latent-mode zero \
    --lambda-force 0.1 \
    --lambda-prior 0 \
    --beta-motion-max 1e-4 \
    --beta-contact-max 1e-4 \
    --warmup-steps 2000 \
    --validation-deployment-mode zero \
    --seed "$MODEL_SEED" \
    "${FORMAL_CONTROL[@]}" \
    --output-dir "$FORMAL_ROOT/dualzero_seed0" \
    --log-csv "$FORMAL_ROOT/dualzero_seed0/train_log.csv" \
    --validation-log "$FORMAL_ROOT/dualzero_seed0/validation_log.csv"

  run_model act_baseline_run0 \
    "$PYTHON_BIN" scripts/train_act_baseline.py \
    "${COMMON_DATA[@]}" \
    --beta-motion-max 5e-4 \
    --warmup-steps 2000 \
    --d-model 128 \
    --z-dim 16 \
    --nhead 4 \
    --num-encoder-layers 1 \
    --num-decoder-layers 1 \
    --dim-feedforward 256 \
    --dropout 0.0 \
    "${FORMAL_CONTROL[@]}" \
    --output-dir "$FORMAL_ROOT/act_baseline_run0" \
    --log-csv "$FORMAL_ROOT/act_baseline_run0/train_log.csv" \
    --validation-log "$FORMAL_ROOT/act_baseline_run0/validation_log.csv"

  rm -f "$PIPELINE_DIR/current_model"
  printf '%s\n' "$(timestamp)" > "$PIPELINE_DIR/finished_at"
  write_status "$PIPELINE_DIR/pipeline.status" completed
  echo "[$(timestamp)] all five models completed"
}

handle_signal() {
  local signal="$1"
  write_status "$PIPELINE_DIR/pipeline.status" interrupted
  echo "[$(timestamp)] pipeline interrupted by $signal" >&2
  exit 130
}

handle_exit() {
  local return_code=$?
  rm -f "$PIPELINE_DIR/runner.pid"
  if [[ "$return_code" != "0" ]] \
    && [[ -f "$PIPELINE_DIR/pipeline.status" ]] \
    && [[ "$(cut -f1 "$PIPELINE_DIR/pipeline.status")" == "running" ]]; then
    write_status "$PIPELINE_DIR/pipeline.status" failed
  fi
}

stage="${1:-preflight}"
case "$stage" in
  preflight)
    validate_inputs
    ;;
  train)
    mkdir -p "$PIPELINE_DIR"
    trap 'handle_signal INT' INT
    trap 'handle_signal TERM' TERM
    trap handle_exit EXIT
    train_all
    ;;
  *)
    echo "usage: $0 {preflight|train}" >&2
    exit 2
    ;;
esac
