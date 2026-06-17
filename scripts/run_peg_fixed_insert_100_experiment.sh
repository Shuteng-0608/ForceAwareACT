#!/usr/bin/env bash
set -euo pipefail

# Fixed-start insertion-only ForceAwareACT experiment.
#
# Default stage is "smoke" so accidental invocation does not start long training.
#
# Usage:
#   scripts/run_peg_fixed_insert_100_experiment.sh smoke
#   scripts/run_peg_fixed_insert_100_experiment.sh prepare
#   scripts/run_peg_fixed_insert_100_experiment.sh train
#   scripts/run_peg_fixed_insert_100_experiment.sh eval
#   scripts/run_peg_fixed_insert_100_experiment.sh rollout
#   scripts/run_peg_fixed_insert_100_experiment.sh all

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_DIR="${DATA_DIR:-mujoco_data/peg_hole_fixed_insertion}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/peg_fixed_insert_100}"
MODEL_XML="${MODEL_XML:-../arm_teleop/model/pangu_all_right.xml}"

CHUNK_LEN="${CHUNK_LEN:-10}"
FORCE_WINDOW_LEN="${FORCE_WINDOW_LEN:-20}"
FORCE_WINDOW_DURATION="${FORCE_WINDOW_DURATION:-0.25}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STAGE1_STEPS="${STAGE1_STEPS:-10000}"
STAGE2_STEPS="${STAGE2_STEPS:-10000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LAMBDA_FORCE="${LAMBDA_FORCE:-0.1}"
LAMBDA_PRIOR="${LAMBDA_PRIOR:-0.1}"
PRIOR_LOSS_MODE="${PRIOR_LOSS_MODE:-mse_mu}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-500}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"

SPLIT_DIR="$OUTPUT_DIR/splits"
ALL_LIST="$SPLIT_DIR/all100.txt"
TRAIN_LIST="$SPLIT_DIR/train80.txt"
VAL_LIST="$SPLIT_DIR/val10.txt"
TEST_LIST="$SPLIT_DIR/test10.txt"
NORMALIZATION_STATS="$OUTPUT_DIR/normalization_stats_train80.pt"
STAGE1_DIR="$OUTPUT_DIR/stage1"
STAGE2_DIR="$OUTPUT_DIR/stage2"
STAGE1_CHECKPOINT="$STAGE1_DIR/checkpoint.pt"
STAGE2_CHECKPOINT="$STAGE2_DIR/checkpoint.pt"

stage="${1:-smoke}"

run_python() {
  PYTHONPATH=src "$PYTHON_BIN" "$@"
}

make_splits() {
  mkdir -p "$SPLIT_DIR"
  find "$DATA_DIR" -name episode.hdf5 | sort > "$ALL_LIST"
  local episode_count
  episode_count="$(wc -l < "$ALL_LIST" | tr -d ' ')"
  if [[ "$episode_count" != "100" ]]; then
    echo "error: expected 100 episodes under $DATA_DIR, found $episode_count" >&2
    return 1
  fi
  sed -n '1,80p' "$ALL_LIST" > "$TRAIN_LIST"
  sed -n '81,90p' "$ALL_LIST" > "$VAL_LIST"
  sed -n '91,100p' "$ALL_LIST" > "$TEST_LIST"
  echo "created_splits=$SPLIT_DIR"
  echo "train80=$(wc -l < "$TRAIN_LIST" | tr -d ' ')"
  echo "val10=$(wc -l < "$VAL_LIST" | tr -d ' ')"
  echo "test10=$(wc -l < "$TEST_LIST" | tr -d ' ')"
}

smoke_test_dataset() {
  make_splits
  DATA_DIR="$DATA_DIR" \
  ALL_LIST="$ALL_LIST" \
  CHUNK_LEN="$CHUNK_LEN" \
  FORCE_WINDOW_LEN="$FORCE_WINDOW_LEN" \
  FORCE_WINDOW_DURATION="$FORCE_WINDOW_DURATION" \
  PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from force_aware_act.data import ContactForceHDF5Dataset

all_list = Path(os.environ["ALL_LIST"])
episode_paths = [Path(line.strip()) for line in all_list.read_text().splitlines() if line.strip()]
dataset = ContactForceHDF5Dataset(
    episode_paths,
    camera_names=("ee_cam", "base_top_cam"),
    action_mode="joint_pos",
    chunk_len=int(os.environ["CHUNK_LEN"]),
    force_window_len=int(os.environ["FORCE_WINDOW_LEN"]),
    force_window_duration=float(os.environ["FORCE_WINDOW_DURATION"]),
    image_size=(224, 224),
    imagenet_normalize=False,
)
print(f"episode_count={len(episode_paths)}")
print(f"dataset_length={len(dataset)}")
sample = dataset[0]
print(f"sample_keys={sorted(sample.keys())}")
for key in ("images", "qpos", "force_window", "action_chunk", "future_force_chunk"):
    print(f"{key}_shape={tuple(sample[key].shape)}")

expected = {
    "images": (2, 3, 224, 224),
    "qpos": (7,),
    "force_window": (20, 6),
    "action_chunk": (10, 7),
    "future_force_chunk": (10, 6),
}
for key, shape in expected.items():
    if tuple(sample[key].shape) != shape:
        raise SystemExit(f"unexpected {key} shape: {tuple(sample[key].shape)} != {shape}")
if len(dataset) != 35117:
    raise SystemExit(f"unexpected dataset length: {len(dataset)} != 35117")
PY
  run_python scripts/inspect_episode_collection.py \
    --episode-list "$ALL_LIST" \
    --chunk-len "$CHUNK_LEN" \
    --force-window-len "$FORCE_WINDOW_LEN" \
    --force-window-duration "$FORCE_WINDOW_DURATION" \
    --tolerate-length-mismatch \
    --max-length-mismatch 1
}

prepare_stats() {
  make_splits
  run_python scripts/compute_normalization_stats.py \
    --episode-list "$TRAIN_LIST" \
    --chunk-len "$CHUNK_LEN" \
    --force-window-len "$FORCE_WINDOW_LEN" \
    --force-window-duration "$FORCE_WINDOW_DURATION" \
    --batch-size 64 \
    --num-workers "$NUM_WORKERS" \
    --output "$NORMALIZATION_STATS"
}

train_stage1() {
  make_splits
  run_python scripts/train_minimal.py \
    --episode-list "$TRAIN_LIST" \
    --chunk-len "$CHUNK_LEN" \
    --force-window-len "$FORCE_WINDOW_LEN" \
    --force-window-duration "$FORCE_WINDOW_DURATION" \
    --normalization-stats "$NORMALIZATION_STATS" \
    --max-steps "$STAGE1_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --learning-rate "$LEARNING_RATE" \
    --lambda-force "$LAMBDA_FORCE" \
    --lambda-prior "$LAMBDA_PRIOR" \
    --prior-loss-mode "$PRIOR_LOSS_MODE" \
    --warmup-steps "$WARMUP_STEPS" \
    --output-dir "$STAGE1_DIR" \
    --log-csv "$STAGE1_DIR/train_log.csv"
}

train_stage2() {
  make_splits
  run_python scripts/train_contact_prior_stage2.py \
    --episode-list "$TRAIN_LIST" \
    --checkpoint "$STAGE1_CHECKPOINT" \
    --normalization-stats "$NORMALIZATION_STATS" \
    --chunk-len "$CHUNK_LEN" \
    --force-window-len "$FORCE_WINDOW_LEN" \
    --force-window-duration "$FORCE_WINDOW_DURATION" \
    --max-steps "$STAGE2_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --prior-loss-mode "$PRIOR_LOSS_MODE" \
    --output-dir "$STAGE2_DIR" \
    --log-csv "$STAGE2_DIR/train_log.csv"
}

evaluate_split() {
  local name="$1"
  local list_path="$2"
  run_python scripts/evaluate_inference_modes.py \
    --episode-list "$list_path" \
    --checkpoint "$STAGE2_CHECKPOINT" \
    --normalization-stats "$NORMALIZATION_STATS" \
    --batch-size 8 \
    --max-batches "$EVAL_MAX_BATCHES" \
    --chunk-len "$CHUNK_LEN" \
    --force-window-len "$FORCE_WINDOW_LEN" \
    --force-window-duration "$FORCE_WINDOW_DURATION" \
    --output-csv "$STAGE2_DIR/inference_eval_${name}.csv"
}

evaluate_offline() {
  make_splits
  evaluate_split train80 "$TRAIN_LIST"
  evaluate_split val10 "$VAL_LIST"
  evaluate_split test10 "$TEST_LIST"
}

run_rollout() {
  local mode="$1"
  local action_select="$2"
  local name="${mode}_${action_select}"
  run_python scripts/run_mujoco_policy_rollout.py \
    --checkpoint "$STAGE2_CHECKPOINT" \
    --normalization-stats "$NORMALIZATION_STATS" \
    --model-xml "$MODEL_XML" \
    --contact-latent-mode "$mode" \
    --action-select-mode "$action_select" \
    --chunk-len "$CHUNK_LEN" \
    --force-window-len "$FORCE_WINDOW_LEN" \
    --force-window-duration "$FORCE_WINDOW_DURATION" \
    --policy-rate-hz 30 \
    --max-rollout-steps "$ROLLOUT_STEPS" \
    --ema-alpha 0.2 \
    --max-delta-q 0.02 \
    --force-stop-threshold 20 \
    --hole-axis-world 0 -1 0 \
    --output-dir "$OUTPUT_DIR/rollouts/$name" \
    --execute-actions \
    --save-videos
}

rollout_modes() {
  run_rollout prior first
  run_rollout prior mid
  run_rollout prior last
  run_rollout zero mid
}

case "$stage" in
  smoke)
    smoke_test_dataset
    ;;
  prepare)
    smoke_test_dataset
    prepare_stats
    ;;
  train)
    prepare_stats
    train_stage1
    train_stage2
    ;;
  eval)
    evaluate_offline
    ;;
  rollout)
    rollout_modes
    ;;
  all)
    smoke_test_dataset
    prepare_stats
    train_stage1
    train_stage2
    evaluate_offline
    rollout_modes
    ;;
  *)
    echo "usage: $0 {smoke|prepare|train|eval|rollout|all}" >&2
    exit 2
    ;;
esac
