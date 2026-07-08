#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
# from huggingface_hub.errors import RevisionNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "peg_hole_100"
REGISTRY_DIR = PROJECT_ROOT / "docs" / "model_registry"

RELEASE_TAG = "v1.0.0"

DATASET_REPO = "shuteng0608/forceawareact-peg-hole-mujoco"
DATASET_REVISION = "e6f60d7351d4992f0083028bee0efaceba64f5f2"

NORMALIZATION_STATS = (
    OUTPUT_ROOT / "normalization_stats_action_all100.pt"
)

MODELS: list[dict[str, Any]] = [
    {
        "experiment_id": "forceaware_motion_cvae_100k",
        "display_name": "ForceAwareACT Motion CVAE",
        "hf_repo_id": (
            "shuteng0608/"
            "forceawareact-motion-cvae-peg-hole-100k"
        ),
        "policy_variant": "force_aware_motion_cvae",
        "policy_class": "ForceAwareACTMotionCVAEPolicy",
        "train_latent_mode": "posterior",
        "deployment_latent_mode": "zero",
        "local_run_directory": (
            "outputs/peg_hole_100/"
            "forceaware_motion_cvae_betam5e4_trajectory100k"
        ),
    },
    {
        "experiment_id": "act_baseline_motion_cvae_100k",
        "display_name": "ACT Baseline Motion CVAE",
        "hf_repo_id": (
            "shuteng0608/"
            "act-baseline-motion-cvae-peg-hole-100k"
        ),
        "policy_variant": "act_baseline",
        "policy_class": "ACTPolicyBaseline",
        "train_latent_mode": "posterior",
        "deployment_latent_mode": "zero",
        "local_run_directory": (
            "outputs/peg_hole_100/"
            "act_baseline_motion_cvae_betam5e4_trajectory100k"
        ),
    },
    {
        "experiment_id": "forceaware_dualzero_100k",
        "display_name": "ForceAwareACT DualZero",
        "hf_repo_id": (
            "shuteng0608/"
            "forceawareact-dualzero-peg-hole-100k"
        ),
        "policy_variant": "force_aware_act",
        "policy_class": "ForceAwareACTPolicy",
        "train_latent_mode": "zero",
        "deployment_latent_mode": "zero",
        "local_run_directory": (
            "outputs/peg_hole_100/"
            "forceaware_dualzero_trajectory100k"
        ),
    },
    {
        "experiment_id": "forceaware_contact_cvae_100k",
        "display_name": "ForceAwareACT Contact CVAE",
        "hf_repo_id": (
            "shuteng0608/"
            "forceawareact-contact-cvae-betac5e4-lp01-"
            "peg-hole-100k"
        ),
        "policy_variant": "force_aware_contact_cvae",
        "policy_class": "ForceAwareACTContactCVAEPolicy",
        "train_latent_mode": "posterior",
        "deployment_latent_mode": "prior_or_zero",
        "local_run_directory": (
            "outputs/peg_hole_100/"
            "forceaware_contact_cvae_betac5e4_lp01_trajectory100k"
        ),
    },
]

CANONICAL_CHECKPOINT = (
    "checkpoints/checkpoint_step_00100000.pt"
)
LOCAL_CANONICAL_CHECKPOINT = "checkpoint_step_00100000.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

def get_ref_target(
    api: HfApi,
    repo_id: str,
    ref_name: str,
    ref_kind: str,
) -> str | None:
    refs = api.list_repo_refs(
        repo_id=repo_id,
        repo_type="model",
    )

    if ref_kind == "branch":
        candidates = refs.branches
    elif ref_kind == "tag":
        candidates = refs.tags
    else:
        raise ValueError(f"Unsupported ref kind: {ref_kind}")

    for ref in candidates:
        if ref.name == ref_name:
            return ref.target_commit

    return None


def resolve_or_create_tag(
    api: HfApi,
    repo_id: str,
    main_sha: str,
) -> str:
    tag_sha = get_ref_target(
        api=api,
        repo_id=repo_id,
        ref_name=RELEASE_TAG,
        ref_kind="tag",
    )

    if tag_sha is None:
        api.create_tag(
            repo_id=repo_id,
            tag=RELEASE_TAG,
            revision=main_sha,
            tag_message=(
                "Complete 100k checkpoint archive with Model Card, "
                "training logs, normalization statistics, and "
                "reproducibility metadata."
            ),
            repo_type="model",
            exist_ok=True,
        )

        tag_sha = get_ref_target(
            api=api,
            repo_id=repo_id,
            ref_name=RELEASE_TAG,
            ref_kind="tag",
        )

        if tag_sha is None:
            raise RuntimeError(
                f"{repo_id}: tag {RELEASE_TAG} was created "
                "but its target commit could not be resolved"
            )

    elif tag_sha != main_sha:
        print(
            f"WARNING: {repo_id}: existing tag {RELEASE_TAG} "
            f"points to {tag_sha}, while current main is {main_sha}. "
            "The existing tag is preserved.",
            file=sys.stderr,
        )

    return tag_sha


def collect_record(
    api: HfApi,
    model: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    repo_id = model["hf_repo_id"]

    main_sha = get_ref_target(
        api=api,
        repo_id=repo_id,
        ref_name="main",
        ref_kind="branch",
    )

    if main_sha is None:
        raise RuntimeError(
            f"{repo_id}: main branch target commit could not be resolved"
        )
    release_sha = resolve_or_create_tag(
        api=api,
        repo_id=repo_id,
        main_sha=main_sha,
    )

    remote_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="model",
        revision=release_sha,
    )

    numbered_checkpoints = sorted(
        path
        for path in remote_files
        if path.startswith("checkpoints/checkpoint_step_")
        and path.endswith(".pt")
    )

    if len(numbered_checkpoints) != 10:
        raise RuntimeError(
            f"{repo_id}: expected 10 numbered checkpoints at "
            f"{release_sha}, found {len(numbered_checkpoints)}"
        )

    if CANONICAL_CHECKPOINT not in remote_files:
        raise RuntimeError(
            f"{repo_id}: canonical checkpoint is missing: "
            f"{CANONICAL_CHECKPOINT}"
        )

    local_run_dir = PROJECT_ROOT / model["local_run_directory"]
    local_checkpoint = local_run_dir / LOCAL_CANONICAL_CHECKPOINT

    if not local_checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing local checkpoint: {local_checkpoint}"
        )

    record = dict(model)
    record.update(
        {
            "repo_type": "model",
            "visibility": "private",
            "release_tag": RELEASE_TAG,
            "main_sha_at_registry_time": main_sha,
            "release_revision": release_sha,
            "recorded_at_utc": generated_at,
            "canonical_checkpoint": CANONICAL_CHECKPOINT,
            "canonical_checkpoint_step": 100000,
            "canonical_checkpoint_sha256": sha256_file(
                local_checkpoint
            ),
            "canonical_checkpoint_size_bytes": (
                local_checkpoint.stat().st_size
            ),
            "numbered_checkpoint_count": len(
                numbered_checkpoints
            ),
            "dataset_repo_id": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "normalization_stats_path": (
                "config/normalization_stats_action_all100.pt"
            ),
            "normalization_stats_sha256": (
                sha256_file(NORMALIZATION_STATS)
                if NORMALIZATION_STATS.is_file()
                else None
            ),
            "historical_training_source_commit": None,
            "historical_training_source_commit_status": (
                "unresolved"
            ),
            "archive_status": "complete",
        }
    )

    return record


def write_json(
    records: list[dict[str, Any]],
    generated_at: str,
) -> None:
    payload = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "release_tag": RELEASE_TAG,
        "dataset": {
            "repo_id": DATASET_REPO,
            "revision": DATASET_REVISION,
        },
        "models": records,
    }

    path = REGISTRY_DIR / "forceawareact_100k_models.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(records: list[dict[str, Any]]) -> None:
    fields = [
        "experiment_id",
        "display_name",
        "hf_repo_id",
        "policy_variant",
        "policy_class",
        "train_latent_mode",
        "deployment_latent_mode",
        "release_tag",
        "release_revision",
        "canonical_checkpoint",
        "canonical_checkpoint_step",
        "canonical_checkpoint_sha256",
        "canonical_checkpoint_size_bytes",
        "numbered_checkpoint_count",
        "dataset_repo_id",
        "dataset_revision",
        "normalization_stats_sha256",
        "local_run_directory",
        "archive_status",
        "recorded_at_utc",
    ]

    path = REGISTRY_DIR / "forceawareact_100k_models.csv"

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


def write_markdown(
    records: list[dict[str, Any]],
    generated_at: str,
) -> None:
    lines = [
        "# ForceAwareACT 100k Model Registry",
        "",
        "This registry freezes the first complete Hugging Face "
        "archive of the four 100k-step models.",
        "",
        f"- Registry generated at: `{generated_at}`",
        f"- Release tag: `{RELEASE_TAG}`",
        f"- Dataset repository: `{DATASET_REPO}`",
        f"- Dataset revision: `{DATASET_REVISION}`",
        "- Canonical checkpoint: "
        f"`{CANONICAL_CHECKPOINT}`",
        "- Historical training source commit: unresolved",
        "",
        "## Registered Models",
        "",
        "| Experiment | Policy variant | Hugging Face repository | "
        "Release revision | Canonical checkpoint SHA-256 | Status |",
        "|---|---|---|---|---|---|",
    ]

    for record in records:
        lines.append(
            f"| {record['display_name']} "
            f"| `{record['policy_variant']}` "
            f"| `{record['hf_repo_id']}` "
            f"| `{record['release_revision']}` "
            f"| `{record['canonical_checkpoint_sha256']}` "
            f"| {record['archive_status']} |"
        )

    lines.extend(
        [
            "",
            "## Reproducible Download",
            "",
            "Always use the full release revision rather than "
            "the mutable `main` branch.",
            "",
        ]
    )

    for record in records:
        lines.extend(
            [
                f"### {record['display_name']}",
                "",
                "```bash",
                f"hf download {record['hf_repo_id']} \\",
                f"  {record['canonical_checkpoint']} \\",
                f"  --revision {record['release_revision']}",
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Verification",
            "",
            "After downloading, verify the checkpoint with:",
            "",
            "```bash",
            "sha256sum checkpoint_step_00100000.pt",
            "```",
            "",
            "The result must match the SHA-256 value in this registry.",
            "",
            "## Maintenance Rule",
            "",
            "- Do not move the `v1.0.0` tag.",
            "- Future repository changes may advance `main` but must not "
            "change this registry entry.",
            "- Create a new tag and a new registry revision for any "
            "material model, configuration, or artifact change.",
            "- Do not replace an existing checkpoint under the same "
            "tag.",
            "",
        ]
    )

    path = REGISTRY_DIR / "forceawareact_100k_models.md"
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    api = HfApi()

    records = [
        collect_record(
            api=api,
            model=model,
            generated_at=generated_at,
        )
        for model in MODELS
    ]

    write_json(records, generated_at)
    write_csv(records)
    write_markdown(records, generated_at)

    print("Registry created:")
    print(
        REGISTRY_DIR / "forceawareact_100k_models.md"
    )
    print(
        REGISTRY_DIR / "forceawareact_100k_models.json"
    )
    print(
        REGISTRY_DIR / "forceawareact_100k_models.csv"
    )

    print("\nPinned revisions:")
    for record in records:
        print(
            f"{record['experiment_id']}: "
            f"{record['release_revision']}"
        )


if __name__ == "__main__":
    main()
