#!/usr/bin/env python3
"""Evaluate a frozen real-robot checkpoint on its validation or test episodes."""

import argparse
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from force_aware_act.act_aligned_training.normalization import NormalizationStats
from force_aware_act.real_robot.checkpoint import read_checkpoint, source_fingerprint
from force_aware_act.real_robot.config import RealRobotDataConfig
from force_aware_act.real_robot.data import RealRobotHighRateDataset
from force_aware_act.real_robot.evaluation import evaluate_episodes, fixed_indices, index_manifest
from force_aware_act.real_robot.reader import discover_real_robot_episodes
from force_aware_act.real_robot.training import build_model, make_model_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-role", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("evaluation output already exists")
    if args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    payload = read_checkpoint(args.checkpoint)
    config = RealRobotDataConfig.from_dict(payload["data_config"])
    by_id = {r.episode_id: r for r in discover_real_robot_episodes(args.data_root, config)}
    ids = payload["split_manifest"].get(f"{args.split_role}_episode_ids", [])
    if not ids:
        raise ValueError(f"checkpoint has no {args.split_role} episodes")
    records = tuple(by_id[key] for key in ids)
    for record in records:
        if source_fingerprint(record.resolve(args.data_root)) != payload["source_fingerprints"][record.episode_id]:
            raise ValueError("evaluation source differs from the frozen checkpoint")
    normalization = NormalizationStats.from_dict(payload["normalization"])
    kind = payload["model_kind"]
    model = build_model(kind, payload["model_config"], initialize_pretrained=False).to(args.device)
    model.load_state_dict(payload["model_state"], strict=True)
    model_config = model.config if kind != "act" else make_model_config(
        "contact", config, chunk_len=model.config.chunk_len, pretrained=False)
    dataset = RealRobotHighRateDataset(args.data_root, records, normalization, model_config, config)
    indices = fixed_indices(dataset, args.stride)
    start = time.monotonic()
    metrics = evaluate_episodes(kind, model, dataset, indices, normalization, args.device, args.batch_size, detailed=True)
    groups = payload["split_manifest"].get("condition_groups", {})
    report = {"version": "real_robot_frozen_evaluation_v1", "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_fingerprint": source_fingerprint(args.checkpoint), "checkpoint_step": payload["global_step"],
              "model_kind": kind, "role": args.split_role, "stride": args.stride, "metrics": metrics,
              "indices": index_manifest(dataset, indices, args.stride), "condition_groups": {key: groups.get(key) for key in ids},
              "normalization_source": "checkpoint_training_episodes_only", "latent_mode": "zero",
              "evaluation_seconds": time.monotonic() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "metrics": metrics}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()
