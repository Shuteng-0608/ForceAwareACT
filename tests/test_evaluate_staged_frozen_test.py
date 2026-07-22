from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

import scripts.evaluate_staged_frozen_test as frozen
from force_aware_act.data import (
    DatasetManifest,
    EpisodeIdentity,
    EpisodeManifestEntry,
    canonical_json_sha256,
)


def _write_shortlist_csv(path: Path, row: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=frozen.selection_evaluator.SHORTLIST_FIELDS
        )
        writer.writeheader()
        writer.writerow(row)


def test_load_selection_artifact_requires_pinned_consistent_report(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    normalization = tmp_path / "normalization.pt"
    normalization.write_bytes(b"normalization")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("candidate_id\nselected\n", encoding="utf-8")
    metrics = tmp_path / "metrics_long.csv"
    decisions = tmp_path / "decisions.csv"
    metrics.write_text("metric\n", encoding="utf-8")
    decisions.write_text("decision\n", encoding="utf-8")
    selection_completion = tmp_path / "evaluation_completion.json"
    shortlist_csv = tmp_path / "shortlist.csv"
    selected = {
        "rank": 1,
        "candidate_id": "selected",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": frozen.file_sha256(checkpoint),
        "checkpoint_epoch": 3,
        "checkpoint_step": 20,
        "metric": "deploy_loss",
        "objective_value": 0.5,
        "retention_value": 0.6,
        "retention_limit": 0.7,
        "final_selected": True,
        "fallback_reference": False,
    }
    _write_shortlist_csv(shortlist_csv, selected)
    reference_run_manifest = tmp_path / "reference_run_manifest.json"
    reference_stage_completion = tmp_path / "reference_stage_completion.json"
    candidate_run_manifest = tmp_path / "candidate_run_manifest.json"
    candidate_stage_completion = tmp_path / "candidate_stage_completion.json"
    reference_run_manifest.write_text('{"run_id":"' + "a" * 32 + '"}\n', encoding="utf-8")
    reference_stage_completion.write_text("{}\n", encoding="utf-8")
    candidate_run_document = {"run_id": "b" * 32}
    candidate_run_manifest.write_text(
        json.dumps(candidate_run_document) + "\n", encoding="utf-8"
    )
    candidate_stage_completion.write_text("{}\n", encoding="utf-8")
    run_evidence = {
        "reference_run_id": "a" * 32,
        "reference_run_manifest": str(reference_run_manifest),
        "reference_run_manifest_file_sha256": frozen.file_sha256(reference_run_manifest),
        "reference_stage_completion": str(reference_stage_completion),
        "reference_stage_completion_sha256": frozen.file_sha256(reference_stage_completion),
        "candidate_run_id": "b" * 32,
        "candidate_run_manifest": str(candidate_run_manifest),
        "candidate_run_manifest_sha256": canonical_json_sha256(candidate_run_document),
        "candidate_run_manifest_file_sha256": frozen.file_sha256(candidate_run_manifest),
        "candidate_stage_completion": str(candidate_stage_completion),
        "candidate_stage_completion_sha256": frozen.file_sha256(candidate_stage_completion),
        "candidate_stage_steps": [10, 20],
    }
    report = tmp_path / "shortlist.json"
    evaluation_contract = {
        "deployment_mode": "prior",
        "aggregation": "episode_uniform",
        "seed": 17,
        "protocol_deterministic": True,
        "deterministic_algorithms": True,
        "batch_size": 16,
        "num_workers": 0,
        "device": "cpu",
        "runtime_versions": {"torch": str(torch.__version__)},
    }
    document = {
        "schema_version": 2,
        "candidates_csv": str(candidates),
        "candidates_csv_sha256": frozen.file_sha256(candidates),
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": frozen.file_sha256(protocol_path),
        "protocol_sha256": "1" * 64,
        "normalization_stats_path": str(normalization),
        "normalization_stats_file_sha256": frozen.file_sha256(normalization),
        "normalization_sha256": "2" * 64,
        "deployment_mode": "prior",
        "fallback_to_stage1_reference": False,
        "run_evidence": run_evidence,
        "evaluation_contract": evaluation_contract,
        "selected": selected,
        "shortlist": [selected],
        "report_files": {
            "metrics": str(metrics),
            "decisions": str(decisions),
            "shortlist_csv": str(shortlist_csv),
            "shortlist_json": str(report),
            "completion": str(selection_completion),
        },
    }
    report.write_text(json.dumps(document), encoding="utf-8")
    selection_completion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "selection_report": str(report),
                "selection_report_sha256": frozen.file_sha256(report),
                "protocol_sha256": "1" * 64,
                "normalization_sha256": "2" * 64,
                "selected_checkpoint_path": str(checkpoint),
                "selected_checkpoint_sha256": frozen.file_sha256(checkpoint),
                "fallback_to_stage1_reference": False,
                "run_evidence": run_evidence,
                "evaluation_contract": evaluation_contract,
                "report_files": {
                    name: {
                        "path": str(path),
                        "file_sha256": frozen.file_sha256(path),
                    }
                    for name, path in {
                        "metrics": metrics,
                        "decisions": decisions,
                        "shortlist_csv": shortlist_csv,
                        "shortlist_json": report,
                    }.items()
                },
            }
        ),
        encoding="utf-8",
    )
    protocol = SimpleNamespace(
        source_path=protocol_path.resolve(),
        content_sha256="1" * 64,
        seed=17,
        deterministic=True,
    )

    artifact = frozen.load_selection_artifact(
        report,
        expected_sha256=frozen.file_sha256(report),
        protocol=protocol,
        normalization_path=normalization,
    )
    assert artifact.selected_id == "selected"
    assert artifact.checkpoint_path == checkpoint.resolve()
    assert set(artifact.companion_sha256) == {
        "metrics",
        "decisions",
        "shortlist_csv",
        "shortlist_json",
        "completion",
    }

    metrics.write_text("tampered\n", encoding="utf-8")
    # Companion hashes are captured as frozen-test inputs; central selection
    # tampering itself is rejected by the mandatory external report digest.
    with pytest.raises(ValueError, match="selection report SHA256 mismatch"):
        frozen.load_selection_artifact(
            report,
            expected_sha256="f" * 64,
            protocol=protocol,
            normalization_path=normalization,
        )


def test_resolve_frozen_test_domains_requires_exact_registered_five_plus_five(
    tmp_path: Path,
) -> None:
    specs = {}
    for domain_index, (name, manifest_domain) in enumerate(
        (("r60_test", "r60_visual"), ("r2_test", "r2_contact"))
    ):
        episodes = []
        for episode_index in range(5):
            path = tmp_path / f"{domain_index}_{episode_index}.hdf5"
            path.write_bytes(b"episode")
            episodes.append(path)
        episode_list = tmp_path / f"{name}.txt"
        episode_list.write_text(
            "".join(f"{path.resolve()}\n" for path in episodes), encoding="utf-8"
        )
        specs[name] = SimpleNamespace(
            domain=manifest_domain,
            episode_list=episode_list,
            expected_episode_count=5,
        )
    protocol = SimpleNamespace(test_episode_lists=specs)

    domains = frozen.resolve_frozen_test_domains(protocol)
    assert [domain.name for domain in domains] == ["r60_test", "r2_test"]
    assert [len(domain.episode_paths) for domain in domains] == [5, 5]

    specs["r2_test"].expected_episode_count = 4
    with pytest.raises(ValueError, match="exactly 5"):
        frozen.resolve_frozen_test_domains(protocol)


def test_manifest_validation_requires_native_uuid_and_current_episode_bytes(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode.hdf5"
    episode.write_bytes(b"immutable episode")
    manifest = DatasetManifest(
        episodes=(
            EpisodeManifestEntry(
                identity=EpisodeIdentity.from_path(
                    episode, "12345678-1234-5678-9234-567812345678"
                ),
                domain="r60_visual",
                split="test",
                metadata={"episode_uuid_source": "hdf5_root_attr:episode_uuid"},
            ),
        ),
        metadata={"derive_uuid_from_sha256": False},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8"
    )
    protocol = SimpleNamespace(
        dataset_manifest=SimpleNamespace(
            path=manifest_path, expected_sha256=manifest.content_sha256
        )
    )
    domain = frozen.FrozenTestDomain(
        name="r60_test",
        manifest_domain="r60_visual",
        episode_list=tmp_path / "r60_test.txt",
        episode_list_sha256="1" * 64,
        episode_paths=(episode.resolve(),),
    )

    loaded, digest, entries = frozen.load_and_validate_manifest(protocol, (domain,))
    assert digest == manifest.content_sha256
    assert entries[episode.resolve()].identity.episode_uuid == (
        "12345678-1234-5678-9234-567812345678"
    )
    assert loaded.metadata["derive_uuid_from_sha256"] is False

    wrong_assignment = DatasetManifest(
        episodes=(
            EpisodeManifestEntry(
                identity=manifest.episodes[0].identity,
                domain="r2_contact",
                split="val",
                metadata={"episode_uuid_source": "hdf5_root_attr:episode_uuid"},
            ),
        ),
        metadata={"derive_uuid_from_sha256": False},
    )
    manifest_path.write_text(
        json.dumps(wrong_assignment.to_dict()), encoding="utf-8"
    )
    protocol.dataset_manifest.expected_sha256 = wrong_assignment.content_sha256
    with pytest.raises(ValueError, match="manifest assignment mismatch"):
        frozen.load_and_validate_manifest(protocol, (domain,))

    derived = DatasetManifest(
        episodes=(
            EpisodeManifestEntry(
                identity=manifest.episodes[0].identity,
                domain="r60_visual",
                split="test",
                metadata={"episode_uuid_source": "derived:uuid5(file_sha256)"},
            ),
        ),
        metadata={"derive_uuid_from_sha256": True},
    )
    manifest_path.write_text(json.dumps(derived.to_dict()), encoding="utf-8")
    protocol.dataset_manifest.expected_sha256 = derived.content_sha256
    with pytest.raises(ValueError, match="forbids SHA-derived"):
        frozen.load_and_validate_manifest(protocol, (domain,))

    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    protocol.dataset_manifest.expected_sha256 = manifest.content_sha256
    episode.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="episode file SHA-256 mismatch"):
        frozen.load_and_validate_manifest(protocol, (domain,))


class _BatchAwarePriorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(
            (kwargs["contact_latent_mode"], kwargs["deterministic_prior"])
        )
        batch_size = int(kwargs["qpos"].shape[0])
        return {
            "pred_action": torch.zeros(batch_size, 2, 7),
            "pred_force": torch.zeros(batch_size, 2, 6),
        }


def _sample(path: Path, state_index: int, action: float, force: float):
    return {
        "images": torch.zeros(1, 3, 4, 4),
        "qpos": torch.zeros(7),
        "force_window": torch.zeros(3, 6),
        "action_chunk": torch.full((2, 7), action),
        "future_force_chunk": torch.full((2, 6), force),
        "episode_path": str(path),
        "state_index": state_index,
    }


def test_episode_metrics_force_prior_and_keep_episodes_independent(tmp_path: Path) -> None:
    first = (tmp_path / "first.hdf5").resolve()
    second = (tmp_path / "second.hdf5").resolve()
    third = (tmp_path / "third.hdf5").resolve()
    domain = frozen.FrozenTestDomain(
        name="r2_test",
        manifest_domain="r2_contact",
        episode_list=tmp_path / "test.txt",
        episode_list_sha256="1" * 64,
        episode_paths=(first, second),
    )
    wide_domain = frozen.FrozenTestDomain(
        name="r60_test",
        manifest_domain="r60_visual",
        episode_list=tmp_path / "wide_test.txt",
        episode_list_sha256="2" * 64,
        episode_paths=(third,),
    )
    loader = DataLoader(
        [
            _sample(first, 0, 1.0, 2.0),
            _sample(first, 1, 3.0, 4.0),
            _sample(second, 0, 5.0, 6.0),
        ],
        batch_size=2,
        shuffle=False,
    )
    wide_loader = DataLoader(
        [_sample(third, 0, 7.0, 8.0)], batch_size=1, shuffle=False
    )
    entries = {
        first: SimpleNamespace(
            identity=SimpleNamespace(episode_uuid="uuid-first", file_sha256="a" * 64)
        ),
        second: SimpleNamespace(
            identity=SimpleNamespace(episode_uuid="uuid-second", file_sha256="b" * 64)
        ),
        third: SimpleNamespace(
            identity=SimpleNamespace(episode_uuid="uuid-third", file_sha256="c" * 64)
        ),
    }
    stats = {
        "qpos_mean": torch.zeros(7),
        "qpos_std": torch.ones(7),
        "action_mean": torch.zeros(7),
        "action_std": torch.ones(7),
        "force_mean": torch.zeros(6),
        "force_std": torch.ones(6),
    }
    model = _BatchAwarePriorModel()
    rows = frozen.evaluate_episode_metrics(
        model=model,
        dataloaders={"r2_test": loader, "r60_test": wide_loader},
        domains=(domain, wide_domain),
        manifest_entries=entries,
        device=torch.device("cpu"),
        normalization_stats=stats,
        lambda_force=0.5,
    )

    assert len(rows) == 3
    assert rows[0]["num_samples"] == 2
    assert rows[0]["action_l1"] == pytest.approx(2.0)
    assert rows[0]["force_l1"] == pytest.approx(3.0)
    assert rows[0]["deploy_loss"] == pytest.approx(3.5)
    assert rows[1]["action_l1"] == pytest.approx(5.0)
    assert rows[2]["test_name"] == "r60_test"
    assert rows[2]["action_l1"] == pytest.approx(7.0)
    assert model.calls and set(model.calls) == {("prior", True)}


def test_bootstrap_and_domain_aggregation_are_reproducible(tmp_path: Path) -> None:
    values = [1.0, 2.0, 7.0, 8.0, 9.0]
    first = frozen.bootstrap_episode_mean(values, seed=17, replicates=1000)
    second = frozen.bootstrap_episode_mean(values, seed=17, replicates=1000)
    assert first == second
    assert first["mean"] == pytest.approx(sum(values) / len(values))
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]

    domain = frozen.FrozenTestDomain(
        name="r60_test",
        manifest_domain="r60_visual",
        episode_list=tmp_path / "r60.txt",
        episode_list_sha256="1" * 64,
        episode_paths=tuple(tmp_path / f"{index}.hdf5" for index in range(5)),
    )
    rows = [
        {
            "test_name": "r60_test",
            "num_samples": index + 1,
            "action_l1": value,
            "force_l1": value + 1,
            "deploy_loss": value + 2,
        }
        for index, value in enumerate(values)
    ]
    summary = frozen.aggregate_domain_metrics(
        rows,
        (domain,),
        bootstrap_seed=17,
        bootstrap_replicates=1000,
    )
    assert summary["aggregation"] == "episode_uniform"
    assert summary["domains"]["r60_test"]["num_episodes"] == 5
    assert summary["domains"]["r60_test"]["num_samples"] == 15


def test_outputs_are_exclusive_atomic_and_completion_is_last(tmp_path: Path) -> None:
    output_dir, paths = frozen.create_output_directory(tmp_path / "frozen")
    episode_rows = [
        {
            "test_name": "r60_test",
            "manifest_domain": "r60_visual",
            "episode_order": 0,
            "episode_uuid": "uuid",
            "episode_path": "/episode.hdf5",
            "episode_file_sha256": "a" * 64,
            "num_samples": 2,
            "action_l1": 1.0,
            "force_l1": 2.0,
            "deploy_loss": 3.0,
        }
    ]
    completion = frozen.write_frozen_test_outputs(
        output_paths=paths,
        episode_rows=episode_rows,
        domain_metrics={"schema_version": 1, "domains": {}},
        inputs={"protocol_sha256": "1" * 64},
        evaluation_contract={"deployment_mode": "prior"},
    )
    assert completion["status"] == "complete"
    assert paths["completion"].is_file()
    recorded = json.loads(paths["completion"].read_text(encoding="utf-8"))
    assert recorded["artifacts"]["report"]["sha256"] == frozen.file_sha256(
        paths["report"]
    )
    assert not list(output_dir.glob("*.tmp-*"))

    with pytest.raises(FileExistsError, match="refuses to reuse"):
        frozen.create_output_directory(output_dir)

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        frozen.create_output_directory(link / "result")


def test_checkpoint_test_provenance_is_exact(monkeypatch, tmp_path: Path) -> None:
    paths = tuple((tmp_path / f"episode_{index}.hdf5").resolve() for index in range(5))
    domain = frozen.FrozenTestDomain(
        name="r2_test",
        manifest_domain="r2_contact",
        episode_list=(tmp_path / "r2_test.txt").resolve(),
        episode_list_sha256="3" * 64,
        episode_paths=paths,
    )
    assignment = {
        "domain": "r2_contact",
        "episode_list": str(domain.episode_list),
        "resolved_paths_sha256": canonical_json_sha256([str(path) for path in paths]),
    }
    stage1 = SimpleNamespace(
        name="spatial_r60",
        objective=SimpleNamespace(
            validation_deployment_mode="prior",
            train_latent_mode="posterior",
            lambda_prior=1.0,
        ),
        monitor=SimpleNamespace(aggregation="episode_uniform"),
    )
    stage2 = SimpleNamespace(
        name="contact_r2",
        objective=SimpleNamespace(
            validation_deployment_mode="prior",
            train_latent_mode="posterior",
            lambda_prior=1.0,
        ),
        monitor=SimpleNamespace(aggregation="episode_uniform"),
    )
    protocol = SimpleNamespace(
        content_sha256="1" * 64,
        stages=(stage1, stage2),
        model=SimpleNamespace(policy_variant="force_aware_contact_cvae"),
    )
    candidate_run_manifest = tmp_path / "candidate_run_manifest.json"
    candidate_run_document = {"run_id": "b" * 32, "schema_version": 3}
    candidate_run_manifest.write_text(
        json.dumps(candidate_run_document), encoding="utf-8"
    )
    candidate_run_semantic_sha256 = canonical_json_sha256(candidate_run_document)
    config = {
        "protocol_sha256": "1" * 64,
        "normalization_sha256": "2" * 64,
        "validation_deployment_mode": "prior",
        "validation_aggregation": "episode_uniform",
        "run_id": "b" * 32,
        "run_manifest_sha256": candidate_run_semantic_sha256,
        "data_provenance": {
            "dataset_manifest_sha256": "4" * 64,
            "episode_list_sha256": {str(domain.episode_list): "3" * 64},
            "episode_counts": {"tests": {"r2_test": 5}},
            "test_domain_assignments": {"r2_test": assignment},
        },
    }
    checkpoint = SimpleNamespace(
        config=config,
        payload={
            "integrity": {
                "protocol_sha256": "1" * 64,
                "normalization_sha256": "2" * 64,
            }
        },
        stage_index=1,
        stage_name="contact_r2",
    )
    selection = SimpleNamespace(
        report_path=tmp_path / "selection.json",
        selected_id="selected",
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_sha256="5" * 64,
        checkpoint_epoch=1,
        checkpoint_step=1,
        document={
            "normalization_sha256": "2" * 64,
            "fallback_to_stage1_reference": False,
            "run_evidence": {
                "candidate_run_id": "b" * 32,
                "candidate_run_manifest": str(candidate_run_manifest),
                "candidate_run_manifest_sha256": candidate_run_semantic_sha256,
            },
        },
    )
    monkeypatch.setattr(
        frozen.selection_evaluator, "load_checkpoint_strict", lambda **kwargs: checkpoint
    )
    monkeypatch.setattr(
        frozen.selection_evaluator, "_expected_config_from_protocol", lambda protocol: {}
    )
    monkeypatch.setattr(frozen, "validate_checkpoint_compatibility", lambda *args, **kwargs: None)

    loaded, loaded_stage = frozen.validate_selected_checkpoint(
        selection,
        protocol=protocol,
        normalization_sha256="2" * 64,
        manifest_sha256="4" * 64,
        domains=(domain,),
    )
    assert loaded is checkpoint
    assert loaded_stage is stage2

    config["data_provenance"]["episode_counts"]["tests"]["r2_test"] = 4
    with pytest.raises(ValueError, match="test count mismatch"):
        frozen.validate_selected_checkpoint(
            selection,
            protocol=protocol,
            normalization_sha256="2" * 64,
            manifest_sha256="4" * 64,
            domains=(domain,),
        )
