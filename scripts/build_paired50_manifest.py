#!/usr/bin/env python3
"""Build the fixed paired50 train/validation/holdout experiment manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.experiments import (  # noqa: E402
    build_paired_episode_subset_manifest,
    write_paired_episode_subset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = build_paired_episode_subset_manifest(
        args.data_root,
        seed=args.seed,
    )
    write_paired_episode_subset_manifest(manifest, args.output)
    audit = manifest["selection"]["audit"]
    summary = {
        "manifest": str(args.output.resolve()),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "all_episodes": manifest["all_episode_count"],
        "selected_episodes": manifest["selected_episode_count"],
        "train_episodes": manifest["train_episode_count"],
        "validation_episodes": manifest["validation_episode_count"],
        "holdout_episodes": manifest["holdout_episode_count"],
        "train_timesteps": audit["summaries"]["train"]["total_timesteps"],
        "validation_timesteps": audit["summaries"]["validation"][
            "total_timesteps"
        ],
        "balance": audit["standardized_mean_rms"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
