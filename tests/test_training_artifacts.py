from pathlib import Path

from force_aware_act.training import (
    best_policy_artifact,
    checkpoint_identity,
    materialize_best_artifact_reference,
    selection_metric_migration,
)


def test_checkpoint_identity_is_content_addressed(tmp_path):
    path = tmp_path / "parent.pt"
    path.write_bytes(b"checkpoint payload")

    identity = checkpoint_identity(
        path,
        payload={
            "format_version": "test_v1",
            "progress": {"global_step": 25},
        },
        chunk_bytes=3,
    )

    assert identity["path"] == str(path.resolve())
    assert identity["size_bytes"] == len(b"checkpoint payload")
    assert identity["sha256"] == (
        "e95244785d8f11623398cf32958aefef6064cd0e34eff42a40597d337a8c74fe"
    )
    assert identity["global_step"] == 25


def test_prior_best_can_be_materialized_without_copying(tmp_path):
    source = tmp_path / "old" / "best.pt"
    source.parent.mkdir()
    source.write_bytes(b"best")
    artifact = best_policy_artifact(
        source,
        metric_name="action_l1_physical",
        metric=0.1,
        global_step=10,
        storage="full_resume_checkpoint",
    )
    destination = tmp_path / "new" / "best.pt"

    assert materialize_best_artifact_reference(destination, artifact)
    assert destination.is_symlink()
    assert destination.read_bytes() == b"best"


def test_selection_migration_preserves_both_metric_protocols():
    migration = selection_metric_migration(
        source_metric_name="normalized_action_l1",
        source_best_metric=0.2,
        target_metric_name="action_l1_physical",
        target_metric=0.03,
        global_step=25_000,
        reason="legacy_metric_not_comparable",
    )

    assert migration["source_best_metric"] == 0.2
    assert migration["target_metric"] == 0.03
    assert migration["global_step"] == 25_000
