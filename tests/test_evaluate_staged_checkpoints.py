from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.evaluate_staged_checkpoints as evaluator
from force_aware_act.data import (
    DatasetManifest,
    EpisodeIdentity,
    EpisodeManifestEntry,
    canonical_json_sha256,
)
from force_aware_act.training.checkpointing import (
    build_checkpoint_v2,
    save_checkpoint_atomic,
)
from force_aware_act.training.policies import resolved_model_config
from force_aware_act.training.protocol import DatasetSpec, ModelSpec


PROTOCOL_SHA = "1" * 64


class _ScalarModel(torch.nn.Module):
    def __init__(self, value: float = -1.0) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(float(value)))


def _model_spec() -> ModelSpec:
    return ModelSpec(
        policy_variant="force_aware_motion_cvae",
        pretrained_resnet18=False,
        d_model=8,
        z_dim=2,
        action_dim=7,
        force_dim=6,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )


def _dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        action_mode="joint_pos",
        chunk_len=2,
        force_window_len=3,
        force_window_duration=0.2,
        image_size=(8, 8),
        camera_names=("cam",),
        imagenet_normalize=False,
        strict_lengths=True,
        image_alignment="latest_past",
        max_image_lag_seconds=0.1,
    )


def _checkpoint_config(
    *,
    stage_name: str,
    stage_index: int,
    normalization_sha256: str,
    protocol_path: Path,
) -> dict[str, object]:
    model_spec = _model_spec()
    dataset_spec = _dataset_spec()
    return {
        "policy_variant": model_spec.policy_variant,
        "action_mode": dataset_spec.action_mode,
        "chunk_len": dataset_spec.chunk_len,
        "force_window_len": dataset_spec.force_window_len,
        "force_window_duration": dataset_spec.force_window_duration,
        "image_size": dataset_spec.image_size,
        "camera_names": dataset_spec.camera_names,
        "imagenet_normalize": dataset_spec.imagenet_normalize,
        "image_alignment": dataset_spec.image_alignment,
        "max_image_lag_seconds": dataset_spec.max_image_lag_seconds,
        "model": resolved_model_config(model_spec, dataset_spec),
        "optimizer_groups": [
            {
                "name": "all",
                "param_names": ["anchor"],
                "lr_multiplier": 1.0,
                "lr": 1.0e-3,
                "weight_decay": 0.0,
            }
        ],
        "data_provenance": {},
        "training_stage": stage_name,
        "stage_index": stage_index,
        "normalization_stats_path": "unused-in-test.pt",
        "normalization_sha256": normalization_sha256,
        "validation_deployment_mode": "zero",
        "validation_aggregation": "episode_uniform",
        "training_device": "cpu",
        "freeze_vision_batch_norm": stage_index > 0,
        "run_id": ("a" if stage_index == 0 else "b") * 32,
        "run_manifest_sha256": ("c" if stage_index == 0 else "d") * 64,
        "stage_initial_global_step": 0 if stage_index == 0 else 10,
        "checkpoint_every_steps": 10,
        "validation_every_steps": 10,
        "minimum_validations": 1,
        "training_code_sha256": "e" * 64,
        "runtime_versions": {"python": "test", "torch": "test"},
        "protocol_path": str(protocol_path),
        "protocol_sha256": PROTOCOL_SHA,
    }


def _write_checkpoint(
    path: Path,
    *,
    value: float,
    stage_name: str,
    stage_index: int,
    normalization_sha256: str,
    protocol_path: Path,
    epoch: int,
    global_step: int,
    parent_checkpoint_sha256: str | None = None,
) -> str:
    model = _ScalarModel(value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    payload = build_checkpoint_v2(
        model=model,
        optimizer=optimizer,
        config=_checkpoint_config(
            stage_name=stage_name,
            stage_index=stage_index,
            normalization_sha256=normalization_sha256,
            protocol_path=protocol_path,
        ),
        global_step=global_step,
        stage_step=global_step if stage_index == 0 else global_step - 10,
        epoch=epoch,
        step_in_epoch=1,
        stage_name=stage_name,
        stage_index=stage_index,
        protocol_sha256=PROTOCOL_SHA,
        normalization_sha256=normalization_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    return save_checkpoint_atomic(payload, path)


def _normalization_stats(population_path: Path) -> dict[str, object]:
    stats: dict[str, object] = {
        "qpos_mean": torch.zeros(7),
        "qpos_std": torch.ones(7),
        "action_mean": torch.zeros(7),
        "action_std": torch.ones(7),
        "force_mean": torch.zeros(6),
        "force_std": torch.ones(6),
        "action_mode": "joint_pos",
        "chunk_len": 2,
        "force_window_len": 3,
        "force_window_duration": 0.2,
        "camera_names": ("cam",),
        "image_size": (8, 8),
        "imagenet_normalize": False,
        "normalization_estimator": "balanced_raw",
        "normalization_method": "domain_episode_time_equal_raw_hdf5_v1",
        "normalization_implementation_version": 1,
        "domain_weights": {"train": 1.0},
        "domain_episode_paths": {
            "train": [str(population_path.resolve())]
        },
        "domain_episode_counts": {"train": 1},
        "episode_timepoint_counts": [
            {
                "domain": "train",
                "path": str(population_path.resolve()),
                "qpos": 2,
                "action": 2,
                "force": 2,
            }
        ],
        "normalization_config": {
            "implementation_version": 1,
            "method": "domain_episode_time_equal_raw_hdf5_v1",
            "weighting_hierarchy": ["domain", "episode", "time_point"],
            "action_mode": "joint_pos",
            "action_dataset": "observations/joint_pos",
            "action_alignment": (
                "observations/joint_pos[1:N]; legacy next-state offset=1"
            ),
            "action_offset": 1,
            "accumulation_dtype": "float64",
            "output_dtype": "torch.float32",
            "eps": 1.0e-6,
            "tolerate_length_mismatch": False,
            "max_length_mismatch": 0,
            "read_chunk_size": 65536,
            "domain_weights": {"train": 1.0},
        },
        "population_identities": [
            {
                "domain": "train",
                "path": str(population_path.resolve()),
                "file_sha256": evaluator.file_sha256(population_path),
            }
        ],
        "population_paths": [str(population_path.resolve())],
    }
    stats["normalization_config_sha256"] = canonical_json_sha256(
        stats["normalization_config"]
    )
    stats["population_sha256"] = canonical_json_sha256(
        stats["population_identities"]
    )
    descriptor = {
        "normalization_config_sha256": stats["normalization_config_sha256"],
        "population_sha256": stats["population_sha256"],
        "statistics": {
            key: {
                "dtype": str(stats[key].dtype),
                "shape": list(stats[key].shape),
                "values": stats[key].tolist(),
            }
            for key in (
                "qpos_mean",
                "qpos_std",
                "action_mean",
                "action_std",
                "force_mean",
                "force_std",
            )
        },
    }
    stats["normalization_content_sha256"] = canonical_json_sha256(descriptor)
    return stats


def _write_episode_list(path: Path, episode: Path) -> None:
    path.write_text(str(episode.resolve()) + "\n", encoding="utf-8")


def test_candidate_csv_is_strict_ordered_and_resolves_relative_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text(
        "candidate_id,checkpoint_path,checkpoint_sha256,epoch,step\n"
        f"one,{first.name},{evaluator.file_sha256(first)},1,10\n"
        f"two,{second.name},{evaluator.file_sha256(second)},,\n",
        encoding="utf-8",
    )

    specs = evaluator.load_candidate_specs(csv_path)
    assert [spec.candidate_id for spec in specs] == ["one", "two"]
    assert specs[0].checkpoint_path == first.resolve()
    assert (specs[0].expected_epoch, specs[0].expected_step) == (1, 10)
    assert specs[1].expected_epoch is None

    csv_path.write_text(
        "candidate_id,checkpoint_path,checkpoint_sha256,unknown\n"
        f"one,{first.name},{evaluator.file_sha256(first)},x\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown columns"):
        evaluator.load_candidate_specs(csv_path)

    csv_path.write_text(
        "candidate_id,checkpoint_path,checkpoint_sha256\n"
        f"stage1_reference,{first.name},{evaluator.file_sha256(first)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reserved for the stage-1 reference"):
        evaluator.load_candidate_specs(csv_path)

    symlink = tmp_path / "candidate_link.pt"
    symlink.symlink_to(first)
    csv_path.write_text(
        "candidate_id,checkpoint_path,checkpoint_sha256\n"
        f"linked,{symlink.name},{evaluator.file_sha256(first)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        evaluator.load_candidate_specs(csv_path)


def test_validation_domain_resolution_rejects_duplicates_and_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator, "REPO_ROOT", tmp_path)
    first = tmp_path / "first.hdf5"
    second = tmp_path / "second.hdf5"
    first.touch()
    second.touch()
    r60 = tmp_path / "r60.txt"
    r2 = tmp_path / "r2.txt"
    r60.write_text(f"{first}\n{first}\n", encoding="utf-8")
    r2.write_text(f"{second}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains duplicates"):
        evaluator.resolve_validation_domains(["r60=r60.txt", "r2=r2.txt"])

    _write_episode_list(r60, first)
    _write_episode_list(r2, first)
    with pytest.raises(ValueError, match="overlap"):
        evaluator.resolve_validation_domains(["r60=r60.txt", "r2=r2.txt"])


def test_strict_normalization_checks_action_hash_and_validation_leakage(
    tmp_path: Path,
) -> None:
    train_episode = tmp_path / "train.hdf5"
    val_episode = tmp_path / "val.hdf5"
    train_episode.touch()
    val_episode.touch()
    val_list = tmp_path / "val.txt"
    _write_episode_list(val_list, val_episode)
    domain = evaluator.ValidationDomain(
        name="r60",
        episode_list=val_list,
        episode_list_sha256=evaluator.file_sha256(val_list),
        episode_paths=(val_episode.resolve(),),
    )
    stats = _normalization_stats(train_episode)
    stats_path = tmp_path / "stats.pt"
    torch.save(stats, stats_path)
    expected_hash = str(stats["normalization_content_sha256"])

    loaded, actual_hash = evaluator.load_normalization_strict(
        stats_path,
        expected_action_mode="joint_pos",
        expected_domain_weights={"train": 1.0},
        strict_lengths=True,
        expected_sha256=expected_hash,
        validation_domains=[domain],
    )
    assert loaded["action_mode"] == "joint_pos"
    assert actual_hash == expected_hash

    with pytest.raises(ValueError, match="action_mode mismatch"):
        evaluator.load_normalization_strict(
            stats_path,
            expected_action_mode="action",
            expected_domain_weights={"train": 1.0},
            strict_lengths=True,
            expected_sha256=expected_hash,
            validation_domains=[domain],
        )

    leaked = _normalization_stats(val_episode)
    leaked_path = tmp_path / "leaked.pt"
    torch.save(leaked, leaked_path)
    with pytest.raises(ValueError, match="overlaps validation"):
        evaluator.load_normalization_strict(
            leaked_path,
            expected_action_mode="joint_pos",
            expected_domain_weights={"train": 1.0},
            strict_lengths=True,
            expected_sha256=str(leaked["normalization_content_sha256"]),
            validation_domains=[domain],
        )

    tampered = dict(stats)
    tampered["qpos_mean"] = torch.ones(7)
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered, tampered_path)
    with pytest.raises(ValueError, match="semantic SHA256 mismatch"):
        evaluator.load_normalization_strict(
            tampered_path,
            expected_action_mode="joint_pos",
            expected_domain_weights={"train": 1.0},
            strict_lengths=True,
            expected_sha256=expected_hash,
            validation_domains=[domain],
        )

    provenance_tampered = dict(stats)
    provenance_tampered["population_identities"] = [
        {**stats["population_identities"][0], "domain": "wrong-domain"}
    ]
    provenance_tampered_path = tmp_path / "provenance_tampered.pt"
    torch.save(provenance_tampered, provenance_tampered_path)
    with pytest.raises(ValueError, match="population_identities SHA256 mismatch"):
        evaluator.load_normalization_strict(
            provenance_tampered_path,
            expected_action_mode="joint_pos",
            expected_domain_weights={"train": 1.0},
            strict_lengths=True,
            expected_sha256=expected_hash,
            validation_domains=[domain],
        )


def test_checkpoint_loader_verifies_external_hash_and_csv_metadata(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    path = tmp_path / "candidate.pt"
    digest = _write_checkpoint(
        path,
        value=1.0,
        stage_name="stage2",
        stage_index=1,
        normalization_sha256="2" * 64,
        protocol_path=protocol_path,
        epoch=2,
        global_step=20,
    )
    loaded = evaluator.load_checkpoint_strict(
        candidate_id="candidate",
        role="candidate",
        path=path,
        expected_sha256=digest,
        expected_epoch=2,
        expected_step=20,
    )
    assert loaded.stage_name == "stage2"
    assert loaded.file_sha256 == digest

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        evaluator.load_checkpoint_strict(
            candidate_id="candidate",
            role="candidate",
            path=path,
            expected_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="step mismatch"):
        evaluator.load_checkpoint_strict(
            candidate_id="candidate",
            role="candidate",
            path=path,
            expected_sha256=digest,
            expected_step=21,
        )
    symlink = tmp_path / "candidate_link.pt"
    symlink.symlink_to(path)
    with pytest.raises(ValueError, match="must not be a symlink"):
        evaluator.load_checkpoint_strict(
            candidate_id="candidate",
            role="candidate",
            path=symlink,
            expected_sha256=digest,
        )


def test_checkpoint_family_rejects_skipping_a_protocol_stage(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    normalization_sha256 = "2" * 64
    reference_config = _checkpoint_config(
        stage_name="stage1",
        stage_index=0,
        normalization_sha256=normalization_sha256,
        protocol_path=protocol_path,
    )
    candidate_config = _checkpoint_config(
        stage_name="stage3",
        stage_index=2,
        normalization_sha256=normalization_sha256,
        protocol_path=protocol_path,
    )
    integrity = {
        "protocol_sha256": PROTOCOL_SHA,
        "normalization_sha256": normalization_sha256,
    }
    reference = evaluator.LoadedCheckpoint(
        candidate_id="reference",
        role="stage1_reference",
        path=tmp_path / "reference.pt",
        file_sha256="a" * 64,
        payload={"integrity": integrity, "lineage": {}},
        config=reference_config,
        stage_name="stage1",
        stage_index=0,
        epoch=1,
        step=10,
    )
    candidate = evaluator.LoadedCheckpoint(
        candidate_id="candidate",
        role="candidate",
        path=tmp_path / "candidate.pt",
        file_sha256="b" * 64,
        payload={
            "integrity": integrity,
            "lineage": {"parent_checkpoint_sha256": reference.file_sha256},
        },
        config=candidate_config,
        stage_name="stage3",
        stage_index=2,
        epoch=1,
        step=20,
    )
    protocol = SimpleNamespace(
        content_sha256=PROTOCOL_SHA,
        deterministic=True,
        seed=17,
        model=_model_spec(),
        dataset=_dataset_spec(),
        stages=(
            SimpleNamespace(name="stage1"),
            SimpleNamespace(name="stage2"),
            SimpleNamespace(name="stage3"),
        ),
    )

    with pytest.raises(ValueError, match="immediately follow"):
        evaluator.validate_checkpoint_family(reference, [candidate], protocol)

    candidate = evaluator.LoadedCheckpoint(
        candidate_id="candidate",
        role="candidate",
        path=tmp_path / "candidate.pt",
        file_sha256="b" * 64,
        payload={
            "integrity": integrity,
            "lineage": {"parent_checkpoint_sha256": reference.file_sha256},
        },
        config=_checkpoint_config(
            stage_name="stage2",
            stage_index=1,
            normalization_sha256=normalization_sha256,
            protocol_path=protocol_path,
        ),
        stage_name="stage2",
        stage_index=1,
        epoch=1,
        step=reference.step,
    )
    with pytest.raises(ValueError, match="strictly greater than.*stage-1 reference"):
        evaluator.validate_checkpoint_family(reference, [candidate], protocol)


def test_selector_contract_is_pinned_to_protocol_monitor() -> None:
    stage = SimpleNamespace(
        monitor=SimpleNamespace(
            primary_domain="r2",
            retention_domain="r60",
            metric="deploy_loss",
            max_retention_regression=0.05,
            min_delta=0.005,
            aggregation="episode_uniform",
        )
    )
    args = argparse.Namespace(
        objective_domain="r2",
        retention_domain="r60",
        metric="deploy_loss",
        max_relative_degradation=0.05,
        max_absolute_degradation=0.0,
        min_relative_improvement=0.005,
    )
    evaluator.validate_selector_contract(args, stage)

    args.metric = "action_l1"
    with pytest.raises(ValueError, match="--metric disagrees"):
        evaluator.validate_selector_contract(args, stage)

    args.metric = "deploy_loss"
    args.max_absolute_degradation = 0.01
    with pytest.raises(ValueError, match="must be 0"):
        evaluator.validate_selector_contract(args, stage)


def test_candidate_chronology_rejects_reordered_or_duplicate_steps(tmp_path: Path) -> None:
    def candidate(candidate_id: str, step: int) -> evaluator.LoadedCheckpoint:
        return evaluator.LoadedCheckpoint(
            candidate_id=candidate_id,
            role="candidate",
            path=tmp_path / f"{candidate_id}.pt",
            file_sha256=candidate_id[0] * 64,
            payload={},
            config={},
            stage_name="stage2",
            stage_index=1,
            epoch=1,
            step=step,
        )

    evaluator.validate_candidate_chronology(
        [candidate("alpha", 20), candidate("beta", 30)]
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluator.validate_candidate_chronology(
            [candidate("alpha", 20), candidate("beta", 10)]
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluator.validate_candidate_chronology(
            [candidate("alpha", 20), candidate("beta", 20)]
        )


def _evaluation_data_contract_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    tuple[evaluator.ValidationDomain, ...],
    evaluator.LoadedCheckpoint,
    evaluator.LoadedCheckpoint,
    Path,
]:
    monkeypatch.setattr(evaluator, "REPO_ROOT", tmp_path)
    r60_episode = tmp_path / "r60.hdf5"
    r2_episode = tmp_path / "r2.hdf5"
    r60_episode.write_bytes(b"r60-validation")
    r2_episode.write_bytes(b"r2-validation")
    r60_list = tmp_path / "r60.txt"
    r2_list = tmp_path / "r2.txt"
    _write_episode_list(r60_list, r60_episode)
    _write_episode_list(r2_list, r2_episode)
    domains = evaluator.resolve_validation_domains(
        ["r60=r60.txt", "r2=r2.txt"]
    )

    manifest = DatasetManifest(
        (
            EpisodeManifestEntry(
                identity=EpisodeIdentity.from_path(
                    r60_episode, "00000000-0000-4000-8000-000000000001"
                ),
                domain="r60_visual",
                split="val",
                metadata={
                    "episode_uuid_source": "hdf5_root_attr:episode_uuid"
                },
            ),
            EpisodeManifestEntry(
                identity=EpisodeIdentity.from_path(
                    r2_episode, "00000000-0000-4000-8000-000000000002"
                ),
                domain="r2_contact",
                split="val",
                metadata={
                    "episode_uuid_source": "hdf5_root_attr:episode_uuid"
                },
            ),
        ),
        metadata={"derive_uuid_from_sha256": False},
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    validation_specs = (
        SimpleNamespace(
            name="r60",
            domain="r60_visual",
            episode_list=r60_list.resolve(),
            expected_episode_count=1,
        ),
        SimpleNamespace(
            name="r2",
            domain="r2_contact",
            episode_list=r2_list.resolve(),
            expected_episode_count=1,
        ),
    )
    stage = SimpleNamespace(validation_domains=validation_specs)
    protocol = SimpleNamespace(
        dataset_manifest=SimpleNamespace(
            path=manifest_path.resolve(), expected_sha256=manifest.content_sha256
        )
    )
    validation_paths_sha256 = canonical_json_sha256(
        {
            domain.name: [str(path) for path in domain.episode_paths]
            for domain in domains
        }
    )
    provenance = {
        "validation_paths_sha256": validation_paths_sha256,
        "episode_list_sha256": {
            str(domain.episode_list): domain.episode_list_sha256
            for domain in domains
        },
        "dataset_manifest_sha256": manifest.content_sha256,
        "episode_counts": {"validation": {"r60": 1, "r2": 1}},
    }

    def checkpoint(
        candidate_id: str, role: str, file_sha256: str
    ) -> evaluator.LoadedCheckpoint:
        return evaluator.LoadedCheckpoint(
            candidate_id=candidate_id,
            role=role,
            path=tmp_path / f"{candidate_id}.pt",
            file_sha256=file_sha256,
            payload={},
            config={"data_provenance": dict(provenance)},
            stage_name="stage1" if role == "stage1_reference" else "stage2",
            stage_index=0 if role == "stage1_reference" else 1,
            epoch=1,
            step=10 if role == "stage1_reference" else 20,
        )

    return (
        protocol,
        stage,
        domains,
        checkpoint("stage1_reference", "stage1_reference", "a" * 64),
        checkpoint("candidate", "candidate", "b" * 64),
        r60_episode,
    )


def test_evaluation_data_contract_checks_reference_and_candidates_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, stage, domains, reference, candidate, _episode = (
        _evaluation_data_contract_fixture(tmp_path, monkeypatch)
    )

    evaluator.validate_evaluation_data_contract(
        protocol=protocol,
        stage=stage,
        domains=domains,
        reference=reference,
        candidates=[candidate],
    )

    reference.config["data_provenance"]["validation_paths_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stage1_reference.*validation path"):
        evaluator.validate_evaluation_data_contract(
            protocol=protocol,
            stage=stage,
            domains=domains,
            reference=reference,
            candidates=[candidate],
        )


@pytest.mark.parametrize(
    ("role", "field", "message"),
    (
        ("stage1_reference", "episode_list_sha256", "episode-list hash mismatch"),
        ("stage1_reference", "dataset_manifest_sha256", "dataset manifest mismatch"),
        ("candidate", "validation_paths_sha256", "validation path provenance mismatch"),
    ),
)
def test_evaluation_data_contract_rejects_provenance_mismatch_for_either_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    field: str,
    message: str,
) -> None:
    protocol, stage, domains, reference, candidate, _episode = (
        _evaluation_data_contract_fixture(tmp_path, monkeypatch)
    )
    checkpoint = reference if role == "stage1_reference" else candidate
    provenance = checkpoint.config["data_provenance"]
    if field == "episode_list_sha256":
        provenance[field] = {}
    else:
        provenance[field] = "0" * 64

    with pytest.raises(ValueError, match=message):
        evaluator.validate_evaluation_data_contract(
            protocol=protocol,
            stage=stage,
            domains=domains,
            reference=reference,
            candidates=[candidate],
        )


def test_evaluation_data_contract_verifies_validation_episode_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, stage, domains, reference, candidate, episode = (
        _evaluation_data_contract_fixture(tmp_path, monkeypatch)
    )
    episode.write_bytes(b"tampered-after-manifest")

    with pytest.raises(ValueError, match="validation episode file SHA256 mismatch"):
        evaluator.validate_evaluation_data_contract(
            protocol=protocol,
            stage=stage,
            domains=domains,
            reference=reference,
            candidates=[candidate],
        )


def test_end_to_end_reports_retention_decisions_and_shortlist_without_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_episode = tmp_path / "train.hdf5"
    r60_episode = tmp_path / "r60.hdf5"
    r2_episode = tmp_path / "r2.hdf5"
    for path in (train_episode, r60_episode, r2_episode):
        path.touch()
    r60_list = tmp_path / "r60.txt"
    r2_list = tmp_path / "r2.txt"
    _write_episode_list(r60_list, r60_episode)
    _write_episode_list(r2_list, r2_episode)

    stats = _normalization_stats(train_episode)
    normalization_hash = str(stats["normalization_content_sha256"])
    stats_path = tmp_path / "stats.pt"
    torch.save(stats, stats_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")

    reference_path = tmp_path / "reference.pt"
    reference_hash = _write_checkpoint(
        reference_path,
        value=0.0,
        stage_name="stage1",
        stage_index=0,
        normalization_sha256=normalization_hash,
        protocol_path=protocol_path,
        epoch=1,
        global_step=10,
    )
    candidates = []
    for candidate_id, value, epoch, step in (
        ("candidate_one", 1.0, 1, 20),
        ("candidate_two", 2.0, 2, 30),
        ("candidate_three", 3.0, 3, 40),
    ):
        path = tmp_path / f"{candidate_id}.pt"
        digest = _write_checkpoint(
            path,
            value=value,
            stage_name="stage2",
            stage_index=1,
            normalization_sha256=normalization_hash,
            protocol_path=protocol_path,
            epoch=epoch,
            global_step=step,
            parent_checkpoint_sha256=reference_hash,
        )
        candidates.append((candidate_id, path, digest, epoch, step))
    candidates_csv = tmp_path / "candidates.csv"
    with candidates_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["candidate_id", "checkpoint_path", "checkpoint_sha256", "epoch", "step"]
        )
        for row in candidates:
            writer.writerow([row[0], row[1].name, row[2], row[3], row[4]])

    stage2 = SimpleNamespace(
        name="stage2",
        objective=SimpleNamespace(
            validation_deployment_mode="zero",
            train_latent_mode="posterior",
            lambda_prior=0.0,
            lambda_force=0.1,
        ),
        monitor=SimpleNamespace(
            primary_domain="r2",
            retention_domain="r60",
            metric="deploy_loss",
            max_retention_regression=0.05,
            min_delta=0.0,
            aggregation="episode_uniform",
        ),
    )
    stage1 = SimpleNamespace(name="stage1")
    protocol = SimpleNamespace(
        content_sha256=PROTOCOL_SHA,
        deterministic=True,
        seed=17,
        model=_model_spec(),
        dataset=_dataset_spec(),
        normalization=SimpleNamespace(
            stats_path=stats_path,
            expected_sha256=normalization_hash,
            domain_weights={"train": 1.0},
        ),
        stages=(stage1, stage2),
        stage=lambda name: stage2 if name == "stage2" else None,
    )
    monkeypatch.setattr(evaluator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        evaluator,
        "validate_evaluation_data_contract",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluator,
        "validate_run_artifacts",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        evaluator,
        "build_validation_loaders",
        lambda _protocol, domains, **_kwargs: {
            domain.name: object() for domain in domains
        },
    )
    monkeypatch.setattr(evaluator, "_build_model_from_config", lambda _config: _ScalarModel())

    def fake_metrics(*, model, **_kwargs):
        value = int(model.anchor.detach().cpu().item())
        table = {
            0: (1.00, 1.40),
            1: (1.04, 0.80),
            2: (1.06, 0.60),
            3: (1.03, 0.70),
        }
        retention, objective = table[value]
        return {
            "r60": {
                "deploy_loss": retention,
                "action_l1": retention,
                "force_l1": 0.0,
                "num_samples": 2.0,
                "num_episodes": 1.0,
            },
            "r2": {
                "deploy_loss": objective,
                "action_l1": objective,
                "force_l1": 0.0,
                "num_samples": 2.0,
                "num_episodes": 1.0,
            },
        }

    monkeypatch.setattr(evaluator, "evaluate_named_deployment_metrics", fake_metrics)
    output_dir = tmp_path / "reports"
    args = argparse.Namespace(
        candidates_csv=candidates_csv,
        stage1_reference=reference_path,
        stage1_reference_sha256=reference_hash,
        protocol=protocol_path,
        normalization_stats=stats_path,
        val_domain=["r60=r60.txt", "r2=r2.txt"],
        objective_domain="r2",
        retention_domain="r60",
        metric="deploy_loss",
        max_relative_degradation=0.05,
        max_absolute_degradation=0.0,
        min_relative_improvement=0.0,
        shortlist_size=2,
        batch_size=2,
        num_workers=0,
        device="cpu",
        output_dir=output_dir,
    )
    document = evaluator.run(args)

    assert document["fallback_to_stage1_reference"] is False
    assert document["selected"]["candidate_id"] == "candidate_three"
    assert [row["candidate_id"] for row in document["shortlist"]] == [
        "candidate_three",
        "candidate_one",
    ]
    assert all(
        (output_dir / filename).is_file()
        for filename in evaluator.REPORT_FILENAMES.values()
    )

    with (output_dir / "decisions.csv").open(newline="", encoding="utf-8") as handle:
        decisions = list(csv.DictReader(handle))
    assert [row["reason"] for row in decisions] == [
        "selected",
        "retention_gate_failed",
        "selected",
    ]
    assert [row["final_best"] for row in decisions] == ["False", "False", "True"]
    with (output_dir / "metrics_long.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 4 * 2 * 5
    assert {row["role"] for row in metrics} == {"stage1_reference", "candidate"}
    persisted = json.loads((output_dir / "shortlist.json").read_text(encoding="utf-8"))
    assert persisted["selector"]["retention_baseline"] == pytest.approx(1.0)
    assert persisted["evaluation_contract"]["seed"] == 17
    assert persisted["evaluation_contract"]["deterministic_algorithms"] is True
    assert persisted["evaluation_contract"]["device"] == "cpu"

    with pytest.raises(FileExistsError, match="requires a new output directory"):
        evaluator.run(args)


def test_shortlist_falls_back_to_stage1_when_every_candidate_fails_gate(
    tmp_path: Path,
) -> None:
    reference_checkpoint = evaluator.LoadedCheckpoint(
        candidate_id="stage1_reference",
        role="stage1_reference",
        path=tmp_path / "reference.pt",
        file_sha256="1" * 64,
        payload={},
        config={},
        stage_name="stage1",
        stage_index=0,
        epoch=1,
        step=10,
    )
    candidate_checkpoint = evaluator.LoadedCheckpoint(
        candidate_id="candidate",
        role="candidate",
        path=tmp_path / "candidate.pt",
        file_sha256="2" * 64,
        payload={},
        config={},
        stage_name="stage2",
        stage_index=1,
        epoch=1,
        step=20,
    )
    metrics_reference = {
        "r60": {"deploy_loss": 1.0},
        "r2": {"deploy_loss": 1.5},
    }
    metrics_candidate = {
        "r60": {"deploy_loss": 1.2},
        "r2": {"deploy_loss": 0.5},
    }
    selector = evaluator.RetentionGatedCheckpointSelector(
        objective_domain="r2",
        retention_domain="r60",
        retention_baseline=1.0,
        max_relative_degradation=0.05,
        best_objective_value=1.5,
        best_retention_value=1.0,
    )
    candidate_evaluation = evaluator.EvaluatedCandidate(
        checkpoint=candidate_checkpoint,
        evaluation_order=1,
        metrics=metrics_candidate,
    )
    candidate_evaluation.decision = selector.update(
        metrics_candidate, epoch=1, step=20
    )
    reference_evaluation = evaluator.EvaluatedCandidate(
        checkpoint=reference_checkpoint,
        evaluation_order=0,
        metrics=metrics_reference,
    )
    domain_file = tmp_path / "domain.txt"
    domain_file.write_text("placeholder\n", encoding="utf-8")
    domain = evaluator.ValidationDomain(
        name="r60",
        episode_list=domain_file,
        episode_list_sha256=evaluator.file_sha256(domain_file),
        episode_paths=(tmp_path / "episode.hdf5",),
    )
    domain2 = evaluator.ValidationDomain(
        name="r2",
        episode_list=domain_file,
        episode_list_sha256=evaluator.file_sha256(domain_file),
        episode_paths=(tmp_path / "episode2.hdf5",),
    )
    candidates_csv = tmp_path / "candidates.csv"
    candidates_csv.write_text("header\n", encoding="utf-8")
    output_paths = evaluator.validate_new_outputs(tmp_path / "reports")
    document = evaluator.write_reports(
        output_paths=output_paths,
        reference=reference_evaluation,
        candidates=[candidate_evaluation],
        domains=[domain, domain2],
        selector=selector,
        shortlist_size=3,
        deployment_mode="zero",
        protocol_sha256="3" * 64,
        normalization_sha256="4" * 64,
        candidates_csv=candidates_csv,
        protocol_path=candidates_csv,
        normalization_stats_path=candidates_csv,
    )
    assert document["fallback_to_stage1_reference"] is True
    assert document["selected"]["candidate_id"] == "stage1_reference"
