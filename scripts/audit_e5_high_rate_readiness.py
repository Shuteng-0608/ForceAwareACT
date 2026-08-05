#!/usr/bin/env python3
"""Emit a reproducible E5 data/config/schedule readiness audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_high_rate_force_contract import audit_collection  # noqa: E402
from force_aware_act.act_aligned_training import (  # noqa: E402
    ACTAlignedHighRateTrainingConfig,
)
from force_aware_act.experiments import (  # noqa: E402
    load_paired_episode_subset_manifest,
)
from force_aware_act.inference import (  # noqa: E402
    DEFAULT_FORCE_STOP_THRESHOLD,
    DEFAULT_POLICY_RATE_HZ,
    DEFAULT_SAFE_FORCE_THRESHOLD,
    OFFICIAL_TEMPORAL_AGGREGATION_DECAY,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateContactCVAEPolicy,
)
from force_aware_act.models.official_act import (  # noqa: E402
    OfficialACTConfig,
    OfficialACTPolicy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reference-epochs", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    subset = load_paired_episode_subset_manifest(
        args.manifest, data_root=args.data_root
    )
    force_audit = audit_collection(args.data_root)
    training = ACTAlignedHighRateTrainingConfig(
        batch_size=args.batch_size,
        reference_train_episodes=len(subset.train_episodes),
        official_reference_epochs=args.reference_epochs,
    )
    official_config = OfficialACTConfig(
        pretrained_backbone=False, imagenet_normalize=False
    )
    high_rate_config = ACTAlignedHighRateConfig.canonical_act(
        pretrained_backbone=False, imagenet_normalize=False
    )
    official_model = OfficialACTPolicy(official_config)
    high_rate_model = ACTAlignedHighRateContactCVAEPolicy(high_rate_config)
    result = {
        "passed": bool(force_audit["passed"]),
        "dataset": {
            "fingerprint": subset.dataset_fingerprint,
            "selected_episodes": len(subset.train_episodes) + len(subset.validation_episodes),
            "train_episodes": len(subset.train_episodes),
            "validation_episodes": len(subset.validation_episodes),
            "train_windows": sum(record.num_steps for record in subset.train_episodes),
            "validation_windows": sum(record.num_steps for record in subset.validation_episodes),
        },
        "force_contract": {
            name: force_audit[name]
            for name in (
                "contract_version",
                "episode_count",
                "force_samples_per_action_values",
                "force_samples_per_action_histogram",
                "force_samples_per_action_max",
                "online_nonempty_intervals_max",
                "online_seven_interval_anchor_count",
                "online_full_window_span_min",
                "online_full_window_span_max",
                "episodes_failing_contract",
            )
        },
        "models": {
            "official_act": {
                "architecture_version": official_config.architecture_version,
                "parameters": sum(parameter.numel() for parameter in official_model.parameters()),
            },
            "high_rate_contact_cvae": {
                "architecture_version": high_rate_config.architecture_version,
                "parameters": sum(parameter.numel() for parameter in high_rate_model.parameters()),
                "chunk_len": high_rate_config.chunk_len,
                "force_sample_rate_hz": high_rate_config.force_sample_rate_hz,
                "policy_sample_rate_hz": high_rate_config.policy_sample_rate_hz,
                "online_force_window_len": high_rate_config.online_force_window_len,
            },
        },
        "training": training.checkpoint_metadata(),
        "rollout_defaults": {
            "policy_rate_hz": DEFAULT_POLICY_RATE_HZ,
            "temporal_aggregation_decay": OFFICIAL_TEMPORAL_AGGREGATION_DECAY,
            "safe_force_threshold": DEFAULT_SAFE_FORCE_THRESHOLD,
            "force_stop_threshold": DEFAULT_FORCE_STOP_THRESHOLD,
        },
        "cuda_gate_required": True,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
