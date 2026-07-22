import csv
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from force_aware_act.data.manifest import (
    DatasetManifest,
    EpisodeIdentity,
    EpisodeManifestEntry,
)
from scripts import train_staged
from scripts import evaluate_staged_checkpoints as staged_evaluator


class _Index:
    def __init__(self, episode_path, state_index):
        self.episode_path = Path(episode_path)
        self.state_index = state_index


class _TinyDataset(Dataset):
    def __init__(self, episode_paths, **kwargs):
        self.episode_paths = [Path(path).resolve() for path in episode_paths]
        self.indices = [
            _Index(path, state_index)
            for path in self.episode_paths
            for state_index in range(2)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        value = float(index % 2 + 1)
        return {
            "images": torch.zeros(1, 3, 4, 4),
            "qpos": torch.zeros(7),
            "force_window": torch.zeros(2, 6),
            "action_chunk": torch.full((2, 7), value),
            "future_force_chunk": torch.full((2, 6), value),
            "episode_path": str(self.indices[index].episode_path),
            "state_index": self.indices[index].state_index,
        }


class _TinyMotionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.vision_encoder = torch.nn.Module()
        self.vision_encoder.backbone = torch.nn.Sequential(torch.nn.BatchNorm2d(3))

    def forward(self, **kwargs):
        self.vision_encoder.backbone(kwargs["images"][:, 0])
        batch = kwargs["qpos"].shape[0]
        chunk = kwargs["action_chunk"].shape[1]
        latent = self.weight + torch.zeros(batch, 2, device=self.weight.device)
        return {
            "pred_action": self.weight
            + torch.zeros(batch, chunk, 7, device=self.weight.device),
            "pred_force": self.weight
            + torch.zeros(batch, chunk, 6, device=self.weight.device),
            "mu_motion": latent,
            "logvar_motion": torch.zeros_like(latent),
        }


class _StochasticTinyMotionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, **kwargs):
        batch = kwargs["qpos"].shape[0]
        chunk = kwargs["action_chunk"].shape[1]
        action_noise = self.dropout(
            torch.ones(batch, chunk, 7, device=self.weight.device)
        )
        force_noise = self.dropout(
            torch.ones(batch, chunk, 6, device=self.weight.device)
        )
        latent = self.weight + torch.zeros(batch, 2, device=self.weight.device)
        return {
            "pred_action": self.weight * action_noise,
            "pred_force": self.weight * force_noise,
            "mu_motion": latent,
            "logvar_motion": torch.zeros_like(latent),
        }


def _write_list(path, entries):
    path.write_text("\n".join(str(entry) for entry in entries) + "\n", encoding="utf-8")


def _make_files_and_protocol(tmp_path, *, stage1_steps=2):
    files = {}
    for name in (
        "r60_train",
        "r2_train",
        "r60_val",
        "r2_val",
        "r60_test",
        "r2_test",
    ):
        path = tmp_path / f"{name}.hdf5"
        path.write_bytes(name.encode("ascii"))
        files[name] = path
        _write_list(tmp_path / f"{name}.txt", [path])

    entries = []
    assignments = {
        "r60_train": ("r60_visual", "train"),
        "r2_train": ("r2_contact", "train"),
        "r60_val": ("r60_visual", "val"),
        "r2_val": ("r2_contact", "val"),
        "r60_test": ("r60_visual", "test"),
        "r2_test": ("r2_contact", "test"),
    }
    for name, (domain, split) in assignments.items():
        entries.append(
            EpisodeManifestEntry(
                identity=EpisodeIdentity.from_path(files[name], str(uuid.uuid4())),
                domain=domain,
                split=split,
                metadata={
                    "episode_uuid_source": "hdf5_root_attr:episode_uuid"
                },
            )
        )
    manifest = DatasetManifest(
        entries,
        metadata={"derive_uuid_from_sha256": False},
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )

    stats = {
        "qpos_mean": torch.zeros(7),
        "qpos_std": torch.ones(7),
        "action_mean": torch.zeros(7),
        "action_std": torch.ones(7),
        "force_mean": torch.zeros(6),
        "force_std": torch.ones(6),
        "action_mode": "action",
        "episode_paths": [str(files["r60_train"]), str(files["r2_train"])],
        "population_paths": [str(files["r60_train"]), str(files["r2_train"])],
        "population_identities": [
            {
                "domain": entry.domain,
                "path": str(entry.identity.path),
                "file_sha256": entry.identity.file_sha256,
            }
            for entry in entries
            if entry.split == "train"
        ],
        "normalization_estimator": "balanced_raw",
        "normalization_method": "domain_episode_time_equal_raw_hdf5_v1",
        "normalization_implementation_version": 1,
        "domain_weights": {"r60_visual": 0.5, "r2_contact": 0.5},
        "domain_episode_paths": {
            "r60_visual": [str(files["r60_train"])],
            "r2_contact": [str(files["r2_train"])],
        },
        "domain_episode_counts": {"r60_visual": 1, "r2_contact": 1},
        "episode_timepoint_counts": [
            {
                "domain": "r60_visual",
                "path": str(files["r60_train"]),
                "qpos": 2,
                "action": 2,
                "force": 2,
            },
            {
                "domain": "r2_contact",
                "path": str(files["r2_train"]),
                "qpos": 2,
                "action": 2,
                "force": 2,
            },
        ],
        "chunk_len": 2,
        "force_window_len": 2,
        "force_window_duration": 0.1,
        "camera_names": ("camera",),
        "image_size": (4, 4),
        "imagenet_normalize": False,
        "normalization_config": {
            "implementation_version": 1,
            "method": "domain_episode_time_equal_raw_hdf5_v1",
            "weighting_hierarchy": ["domain", "episode", "time_point"],
            "action_mode": "action",
            "action_dataset": "action",
            "action_alignment": "action[0:N]; command aligned to current state timestamp",
            "action_offset": 0,
            "accumulation_dtype": "float64",
            "output_dtype": "torch.float32",
            "eps": 1.0e-6,
            "tolerate_length_mismatch": False,
            "max_length_mismatch": 0,
            "read_chunk_size": 65536,
            "domain_weights": {"r60_visual": 0.5, "r2_contact": 0.5},
        },
    }
    stats["normalization_config_sha256"] = train_staged.canonical_json_sha256(
        stats["normalization_config"]
    )
    stats["population_sha256"] = train_staged.canonical_json_sha256(
        stats["population_identities"]
    )
    stats["normalization_content_sha256"] = train_staged.canonical_json_sha256(
        {
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
    )
    stats_path = tmp_path / "stats.pt"
    torch.save(stats, stats_path)

    protocol = {
        "schema_version": 2,
        "run_name": "integration",
        "seed": 13,
        "deterministic": True,
        "model": {
            "policy_variant": "force_aware_motion_cvae",
            "d_model": 16,
            "z_dim": 4,
            "nhead": 4,
            "dim_feedforward": 32,
        },
        "dataset": {
            "action_mode": "action",
            "chunk_len": 2,
            "force_window_len": 2,
            "force_window_duration": 0.1,
            "image_size": [4, 4],
            "camera_names": ["camera"],
            "strict_lengths": True,
            "image_alignment": "latest_past",
            "max_image_lag_seconds": 0.1,
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": manifest.content_sha256,
        },
        "normalization": {
            "stats_path": str(stats_path),
            "sha256": stats["normalization_content_sha256"],
            "population_episode_lists": [
                str(tmp_path / "r60_train.txt"),
                str(tmp_path / "r2_train.txt"),
            ],
            "domain_weights": {"r60_visual": 0.5, "r2_contact": 0.5},
        },
        "stages": [
            {
                "name": "spatial_r60",
                "sources": [
                    {
                        "name": "r60",
                        "domain": "r60_visual",
                        "episode_list": str(tmp_path / "r60_train.txt"),
                        "expected_episode_count": 1,
                        "batch_quota": 1,
                    }
                ],
                "validation_domains": [
                    {
                        "name": "visual_wide",
                        "domain": "r60_visual",
                        "episode_list": str(tmp_path / "r60_val.txt"),
                        "expected_episode_count": 1,
                    }
                ],
                "freeze_vision_batch_norm": False,
                "batch_size": 1,
                "samples_per_epoch": 2,
                "max_steps": stage1_steps,
                "validation_every_steps": 1,
                "checkpoint_every_steps": 1,
                "optimizer": {
                    "base_lr": 0.05,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                },
                "objective": {"warmup_steps": 0},
                "monitor": {
                    "primary_domain": "visual_wide",
                    "aggregation": "episode_uniform",
                    "patience": 5,
                    "min_validations": 1,
                },
            },
            {
                "name": "contact_r2",
                "sources": [
                    {
                        "name": "r2",
                        "domain": "r2_contact",
                        "episode_list": str(tmp_path / "r2_train.txt"),
                        "expected_episode_count": 1,
                        "batch_quota": 1,
                    },
                    {
                        "name": "r60_replay",
                        "domain": "r60_visual",
                        "episode_list": str(tmp_path / "r60_train.txt"),
                        "expected_episode_count": 1,
                        "batch_quota": 1,
                    },
                ],
                "validation_domains": [
                    {
                        "name": "contact_rich",
                        "domain": "r2_contact",
                        "episode_list": str(tmp_path / "r2_val.txt"),
                        "expected_episode_count": 1,
                    },
                    {
                        "name": "visual_wide",
                        "domain": "r60_visual",
                        "episode_list": str(tmp_path / "r60_val.txt"),
                        "expected_episode_count": 1,
                    },
                ],
                "freeze_vision_batch_norm": True,
                "batch_size": 2,
                "samples_per_epoch": 4,
                "max_steps": 2,
                "validation_every_steps": 1,
                "checkpoint_every_steps": 1,
                "optimizer": {
                    "base_lr": 0.01,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                },
                "objective": {"warmup_steps": 0},
                "monitor": {
                    "primary_domain": "contact_rich",
                    "aggregation": "episode_uniform",
                    "retention_domain": "visual_wide",
                    "max_retention_regression": 0.05,
                    "patience": 5,
                    "min_validations": 1,
                },
            },
        ],
        "test_episode_lists": {
            "visual_wide": {
                "domain": "r60_visual",
                "episode_list": str(tmp_path / "r60_test.txt"),
                "expected_episode_count": 1,
            },
            "contact_rich": {
                "domain": "r2_contact",
                "episode_list": str(tmp_path / "r2_test.txt"),
                "expected_episode_count": 1,
            },
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return protocol_path


@pytest.fixture
def tiny_runtime(monkeypatch):
    monkeypatch.setattr(train_staged, "ContactForceHDF5Dataset", _TinyDataset)
    monkeypatch.setattr(
        train_staged,
        "build_policy",
        lambda *_args, **_kwargs: _TinyMotionPolicy(),
    )

    def evaluate(*, model, dataloaders, **_kwargs):
        value = max(0.01, 2.0 - float(model.weight.detach().cpu().item()))
        return {
            name: {
                "deploy_loss": value,
                "action_l1": value,
                "force_l1": value,
                "num_samples": float(len(loader.dataset)),
                "num_episodes": float(len(loader.dataset.episode_paths)),
            }
            for name, loader in dataloaders.items()
        }

    monkeypatch.setattr(train_staged, "evaluate_named_deployment_metrics", evaluate)


def _args(protocol, stage, output, *, init_from=None, resume_from=None, dry_run=False):
    return SimpleNamespace(
        protocol=protocol,
        stage=stage,
        init_from=init_from,
        resume_from=resume_from,
        output_dir=output,
        device="cpu",
        num_workers=0,
        dry_run=dry_run,
        trim_resume_logs_to_checkpoint=False,
        allow_legacy_data_contract=False,
        allow_legacy_normalization=False,
        skip_dataset_file_verification=False,
    )


def test_staged_training_writes_lineage_and_retention_fallback(tmp_path, tiny_runtime):
    protocol = _make_files_and_protocol(tmp_path)
    stage1_dir = tmp_path / "stage1"
    assert train_staged.train(_args(protocol, "spatial_r60", stage1_dir)) == 0
    stage1_best = torch.load(stage1_dir / "checkpoint_best.pt", map_location="cpu")
    best_step = stage1_best["training_state"]["stage_step"]
    assert train_staged.file_sha256(stage1_dir / "checkpoint_best.pt") == (
        train_staged.file_sha256(
            stage1_dir / f"checkpoint_best_step_{best_step:08d}.pt"
        )
    )

    stage2_dir = tmp_path / "stage2"
    assert (
        train_staged.train(
            _args(
                protocol,
                "contact_r2",
                stage2_dir,
                init_from=stage1_dir / "checkpoint_best.pt",
            )
        )
        == 0
    )
    stage2 = torch.load(stage2_dir / "checkpoint.pt", map_location="cpu")

    assert stage2["schema_version"] == 2
    assert stage2["stage"] == {"name": "contact_r2", "index": 1}
    assert stage2["training_state"]["stage_step"] == 2
    assert stage2["lineage"]["parent_checkpoint_sha256"] is not None
    assert stage2["lineage"]["parent_checkpoint_sha256"] != stage1_best["integrity"]["model_state_sha256"]
    assert (stage2_dir / "checkpoint_best.pt").is_file()
    assert (stage2_dir / "validation_log.csv").is_file()
    with (stage2_dir / "validation_log.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    baseline_rows = [row for row in rows if row["validation_index"] == "0"]
    assert {row["domain"] for row in baseline_rows} == {
        "contact_rich",
        "visual_wide",
    }
    assert all(
        row["decision_reason"] == "stage_initialization_baseline"
        for row in baseline_rows
    )
    completion = json.loads(
        (stage2_dir / "stage_completion.json").read_text(encoding="utf-8")
    )
    assert completion["schema_version"] == 2
    assert [
        row["stage_step"] for row in completion["candidate_checkpoints"]
    ] == [1, 2]
    reference = staged_evaluator.load_checkpoint_strict(
        candidate_id=staged_evaluator.REFERENCE_CANDIDATE_ID,
        role="stage1_reference",
        path=stage1_dir / "checkpoint_best.pt",
        expected_sha256=train_staged.file_sha256(
            stage1_dir / "checkpoint_best.pt"
        ),
    )
    candidates = tuple(
        staged_evaluator.load_checkpoint_strict(
            candidate_id=f"candidate_{row['stage_step']}",
            role="candidate",
            path=Path(row["checkpoint_path"]),
            expected_sha256=row["checkpoint_sha256"],
            expected_epoch=row["epoch"],
            expected_step=row["global_step"],
        )
        for row in completion["candidate_checkpoints"]
    )
    staged_evaluator.validate_candidate_chronology(
        candidates, reference=reference
    )
    evidence = staged_evaluator.validate_run_artifacts(reference, candidates)
    assert evidence["candidate_run_id"] == stage2["config"]["run_id"]
    with pytest.raises(ValueError, match="complete periodic checkpoint universe"):
        staged_evaluator.validate_run_artifacts(reference, candidates[:-1])


def test_stage2_resume_recovers_logs_after_baseline_checkpoint_crash(
    tmp_path, tiny_runtime, monkeypatch
):
    protocol = _make_files_and_protocol(tmp_path)
    stage1_dir = tmp_path / "stage1"
    assert train_staged.train(_args(protocol, "spatial_r60", stage1_dir)) == 0

    stage2_dir = tmp_path / "stage2"
    original_open_csv = train_staged._open_csv

    def interrupt_before_log_creation(*_args, **_kwargs):
        raise RuntimeError("simulated baseline/log crash window")

    monkeypatch.setattr(train_staged, "_open_csv", interrupt_before_log_creation)
    with pytest.raises(RuntimeError, match="baseline/log crash window"):
        train_staged.train(
            _args(
                protocol,
                "contact_r2",
                stage2_dir,
                init_from=stage1_dir / "checkpoint_best.pt",
            )
        )
    monkeypatch.setattr(train_staged, "_open_csv", original_open_csv)

    stage_zero = stage2_dir / "checkpoint_best_step_00000000.pt"
    assert stage_zero.is_file()
    assert not (stage2_dir / "train_log.csv").exists()
    assert not (stage2_dir / "validation_log.csv").exists()

    assert (
        train_staged.train(
            _args(
                protocol,
                "contact_r2",
                stage2_dir,
                resume_from=stage_zero,
            )
        )
        == 0
    )
    with (stage2_dir / "train_log.csv").open(newline="") as handle:
        train_rows = list(csv.DictReader(handle))
    assert [int(row["stage_step"]) for row in train_rows] == [1, 2]
    with (stage2_dir / "validation_log.csv").open(newline="") as handle:
        validation_rows = list(csv.DictReader(handle))
    baseline_rows = [
        row for row in validation_rows if int(row["validation_index"]) == 0
    ]
    assert {row["domain"] for row in baseline_rows} == {
        "contact_rich",
        "visual_wide",
    }


def test_retention_failure_keeps_stage_initialization_as_best(
    tmp_path, tiny_runtime, monkeypatch
):
    protocol = _make_files_and_protocol(tmp_path)
    stage1_dir = tmp_path / "stage1"
    assert train_staged.train(_args(protocol, "spatial_r60", stage1_dir)) == 0

    calls = {"count": 0}

    def retention_failure(*, dataloaders, **_kwargs):
        calls["count"] += 1
        baseline = calls["count"] == 1
        values = {
            "contact_rich": 1.0 if baseline else 0.5,
            "visual_wide": 1.0 if baseline else 1.2,
        }
        return {
            name: {
                "deploy_loss": values[name],
                "action_l1": values[name],
                "force_l1": values[name],
                "num_samples": float(len(loader.dataset)),
                "num_episodes": float(len(loader.dataset.episode_paths)),
            }
            for name, loader in dataloaders.items()
        }

    monkeypatch.setattr(
        train_staged,
        "evaluate_named_deployment_metrics",
        retention_failure,
    )
    stage2_dir = tmp_path / "stage2"
    assert train_staged.train(
        _args(
            protocol,
            "contact_r2",
            stage2_dir,
            init_from=stage1_dir / "checkpoint_best.pt",
        )
    ) == 0

    best = torch.load(stage2_dir / "checkpoint_best.pt", map_location="cpu")
    assert best["training_state"]["stage_step"] == 0
    assert best["stop_reason"] == "stage_initialization_fallback"
    with (stage2_dir / "validation_log.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [row for row in rows if row["validation_index"] != "0"]
    assert candidates
    assert all(row["selected"] == "False" for row in candidates)
    assert all(
        row["decision_reason"] == "retention_gate_failed"
        for row in candidates
    )


def test_stage_transition_rejects_changed_normalization_identity(
    tmp_path, tiny_runtime
):
    protocol_path = _make_files_and_protocol(tmp_path)
    stage1_dir = tmp_path / "stage1"
    assert train_staged.train(
        _args(protocol_path, "spatial_r60", stage1_dir)
    ) == 0
    protocol = train_staged.load_protocol(protocol_path)
    prepared = train_staged.prepare_stage(
        protocol,
        "contact_r2",
        allow_legacy_data_contract=False,
        allow_legacy_normalization=True,
    )
    payload = torch.load(stage1_dir / "checkpoint_best.pt", map_location="cpu")
    payload["integrity"]["normalization_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="normalization_sha256 mismatch"):
        train_staged._validate_stage_transition_checkpoint(payload, prepared)


def test_prepare_stage_binds_test_lists_to_manifest_domains(tmp_path):
    protocol_path = _make_files_and_protocol(tmp_path)
    protocol = train_staged.load_protocol(protocol_path)
    prepared = train_staged.prepare_stage(
        protocol,
        "spatial_r60",
        allow_legacy_data_contract=False,
        allow_legacy_normalization=False,
    )
    assignments = prepared.data_provenance["test_domain_assignments"]
    assert assignments["visual_wide"]["domain"] == "r60_visual"
    assert assignments["visual_wide"]["episode_list"] == str(
        tmp_path / "r60_test.txt"
    )
    assert len(assignments["visual_wide"]["resolved_paths_sha256"]) == 64

    document = json.loads(protocol_path.read_text(encoding="utf-8"))
    r60_spec = document["test_episode_lists"]["visual_wide"]
    r2_spec = document["test_episode_lists"]["contact_rich"]
    r60_spec["episode_list"], r2_spec["episode_list"] = (
        r2_spec["episode_list"],
        r60_spec["episode_list"],
    )
    protocol_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="test=contact_rich domain mismatch"):
        train_staged.prepare_stage(
            train_staged.load_protocol(protocol_path),
            "spatial_r60",
            allow_legacy_data_contract=False,
            allow_legacy_normalization=False,
        )


def test_prepare_stage_binds_phase_catalog_to_manifest_episode_identity(tmp_path):
    protocol_path = _make_files_and_protocol(tmp_path)
    document = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.load(tmp_path / "dataset_manifest.json")
    entry = next(
        item
        for item in manifest.episodes
        if item.domain == "r2_contact" and item.split == "train"
    )
    catalog_document = {
        "schema_version": 2,
        "episodes": [
            {
                "path": str(entry.identity.path),
                "domain": entry.domain,
                "episode_uuid": entry.identity.episode_uuid,
                "file_sha256": entry.identity.file_sha256,
                "segments": [
                    {"start": 0, "stop": 2, "phase": "contact"}
                ],
            }
        ],
        "labeler": {"dataset_manifest_sha256": manifest.content_sha256},
    }
    catalog_path = tmp_path / "r2_catalog.json"
    catalog_path.write_text(json.dumps(catalog_document), encoding="utf-8")
    r2_source = document["stages"][1]["sources"][0]
    r2_source.update(
        {
            "phase_quotas": {"contact": 1},
            "min_episodes_per_phase": 1,
            "sample_catalog": str(catalog_path),
            "sample_catalog_sha256": train_staged.canonical_json_sha256(
                catalog_document
            ),
        }
    )
    protocol_path.write_text(json.dumps(document), encoding="utf-8")

    prepared = train_staged.prepare_stage(
        train_staged.load_protocol(protocol_path),
        "contact_r2",
        allow_legacy_data_contract=False,
        allow_legacy_normalization=False,
    )
    assert prepared.phase_catalogs["r2"].episode_for(
        entry.identity.path
    ).episode_uuid == entry.identity.episode_uuid

    catalog_document["episodes"][0]["episode_uuid"] = str(uuid.uuid4())
    catalog_path.write_text(json.dumps(catalog_document), encoding="utf-8")
    r2_source["sample_catalog_sha256"] = train_staged.canonical_json_sha256(
        catalog_document
    )
    protocol_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="phase catalog identity mismatch"):
        train_staged.prepare_stage(
            train_staged.load_protocol(protocol_path),
            "contact_r2",
            allow_legacy_data_contract=False,
            allow_legacy_normalization=False,
        )


def test_stage_transition_rejects_nonadjacent_or_changed_episode_lists(
    tmp_path, tiny_runtime
):
    protocol_path = _make_files_and_protocol(tmp_path)
    stage1_dir = tmp_path / "stage1"
    assert train_staged.train(
        _args(protocol_path, "spatial_r60", stage1_dir)
    ) == 0
    protocol = train_staged.load_protocol(protocol_path)
    prepared = train_staged.prepare_stage(
        protocol,
        "contact_r2",
        allow_legacy_data_contract=False,
        allow_legacy_normalization=True,
    )
    payload = torch.load(stage1_dir / "checkpoint_best.pt", map_location="cpu")
    payload["config"]["data_provenance"]["episode_list_sha256"] = {}
    # Refresh the config hash so the payload remains internally self-consistent;
    # transition provenance must still reject it.
    payload["integrity"]["config_sha256"] = (
        train_staged.canonical_json_sha256(payload["config"])
    )

    with pytest.raises(ValueError, match="episode_list_sha256"):
        train_staged._validate_stage_transition_checkpoint(payload, prepared)


def test_stage_transition_rejects_checkpoint_after_selected_best(
    tmp_path, tiny_runtime, monkeypatch
):
    protocol_path = _make_files_and_protocol(tmp_path, stage1_steps=2)
    validation_calls = {"count": 0}

    def worsening_metrics(*, dataloaders, **_kwargs):
        validation_calls["count"] += 1
        value = float(validation_calls["count"])
        return {
            name: {
                "deploy_loss": value,
                "action_l1": value,
                "force_l1": value,
                "num_samples": float(len(loader.dataset)),
                "num_episodes": float(len(loader.dataset.episode_paths)),
            }
            for name, loader in dataloaders.items()
        }

    monkeypatch.setattr(
        train_staged,
        "evaluate_named_deployment_metrics",
        worsening_metrics,
    )
    stage1_dir = tmp_path / "stage1"
    assert train_staged.train(
        _args(protocol_path, "spatial_r60", stage1_dir)
    ) == 0

    with pytest.raises(ValueError, match="not the monitor-selected best"):
        train_staged.train(
            _args(
                protocol_path,
                "contact_r2",
                tmp_path / "stage2",
                init_from=stage1_dir / "checkpoint_latest.pt",
            )
        )


def test_stage_transition_rejects_superseded_best_history(tmp_path, tiny_runtime):
    protocol_path = _make_files_and_protocol(tmp_path, stage1_steps=2)
    stage1_dir = tmp_path / "stage1"
    assert train_staged.train(
        _args(protocol_path, "spatial_r60", stage1_dir)
    ) == 0
    final_best = torch.load(stage1_dir / "checkpoint_best.pt", map_location="cpu")
    assert final_best["training_state"]["stage_step"] == 2

    with pytest.raises(ValueError, match="final selected best"):
        train_staged.train(
            _args(
                protocol_path,
                "contact_r2",
                tmp_path / "stage2",
                init_from=stage1_dir / "checkpoint_best_step_00000001.pt",
            )
        )


def test_exact_resume_matches_continuous_training(tmp_path, tiny_runtime, monkeypatch):
    monkeypatch.setattr(
        train_staged,
        "build_policy",
        lambda *_args, **_kwargs: _StochasticTinyMotionPolicy(),
    )
    protocol = _make_files_and_protocol(tmp_path, stage1_steps=2)
    continuous_dir = tmp_path / "continuous"
    assert train_staged.train(_args(protocol, "spatial_r60", continuous_dir)) == 0

    interrupted_dir = tmp_path / "interrupted"
    original_update = train_staged.train_one_update
    calls = {"count": 0}

    def interrupt_second_update(**kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt("simulated interruption")
        return original_update(**kwargs)

    monkeypatch.setattr(train_staged, "train_one_update", interrupt_second_update)
    with pytest.raises(KeyboardInterrupt, match="simulated"):
        train_staged.train(_args(protocol, "spatial_r60", interrupted_dir))
    monkeypatch.setattr(train_staged, "train_one_update", original_update)

    assert (
        train_staged.train(
            _args(
                protocol,
                "spatial_r60",
                interrupted_dir,
                resume_from=interrupted_dir / "checkpoint_step_00000001.pt",
            )
        )
        == 0
    )
    continuous = torch.load(continuous_dir / "checkpoint.pt", map_location="cpu")
    resumed = torch.load(interrupted_dir / "checkpoint.pt", map_location="cpu")
    assert continuous["training_state"] == resumed["training_state"]
    for name, value in continuous["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][name]), name


def test_resume_rejects_or_explicitly_trims_logs_ahead_of_checkpoint(
    tmp_path, tiny_runtime
):
    protocol = _make_files_and_protocol(tmp_path, stage1_steps=2)
    output = tmp_path / "run"
    assert train_staged.train(_args(protocol, "spatial_r60", output)) == 0

    stale_args = _args(
        protocol,
        "spatial_r60",
        output,
        resume_from=output / "checkpoint_step_00000001.pt",
    )
    with pytest.raises(ValueError, match="logs are ahead"):
        train_staged.train(stale_args)

    stale_args.trim_resume_logs_to_checkpoint = True
    assert train_staged.train(stale_args) == 0
    with (output / "train_log.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["stage_step"]) for row in rows] == [1, 2]
    quarantine_dirs = list((output / "resume_quarantine").iterdir())
    assert len(quarantine_dirs) == 1
    quarantine_manifest = json.loads(
        (quarantine_dirs[0] / "quarantine_manifest.json").read_text()
    )
    quarantined_names = {
        Path(row["quarantined_to"]).name
        for row in quarantine_manifest["moved_artifacts"]
    }
    assert {
        "checkpoint.pt",
        "checkpoint_best.pt",
        "checkpoint_best_step_00000002.pt",
        "checkpoint_latest.pt",
        "checkpoint_step_00000002.pt",
    }.issubset(quarantined_names)
    assert Path(quarantine_manifest["restored_best_from"]).name == (
        "checkpoint_best_step_00000001.pt"
    )


def test_training_output_lock_is_exclusive_and_released(tmp_path):
    output = tmp_path / "locked_run"
    with train_staged._exclusive_training_lock(output):
        with pytest.raises(RuntimeError, match="another staged training writer"):
            with train_staged._exclusive_training_lock(output):
                pass
    with train_staged._exclusive_training_lock(output):
        assert (output / train_staged.TRAINING_LOCK_FILENAME).is_file()


def test_resume_rejects_symlinked_managed_checkpoint(
    tmp_path, tiny_runtime
):
    protocol = _make_files_and_protocol(tmp_path, stage1_steps=2)
    output = tmp_path / "run"
    assert train_staged.train(_args(protocol, "spatial_r60", output)) == 0
    managed = output / "checkpoint_step_00000002.pt"
    external = tmp_path / "external_checkpoint.pt"
    external.write_bytes(managed.read_bytes())
    managed.unlink()
    managed.symlink_to(external)

    args = _args(
        protocol,
        "spatial_r60",
        output,
        resume_from=output / "checkpoint_step_00000001.pt",
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        train_staged.train(args)


def test_resume_rejects_broken_best_alias_without_writing_outside(
    tmp_path, tiny_runtime
):
    protocol = _make_files_and_protocol(tmp_path, stage1_steps=2)
    output = tmp_path / "run"
    assert train_staged.train(_args(protocol, "spatial_r60", output)) == 0
    best_alias = output / "checkpoint_best.pt"
    outside = tmp_path / "outside_created.pt"
    best_alias.unlink()
    best_alias.symlink_to(outside)

    args = _args(
        protocol,
        "spatial_r60",
        output,
        resume_from=output / "checkpoint.pt",
    )
    args.trim_resume_logs_to_checkpoint = True
    with pytest.raises(ValueError, match="must not be a symlink"):
        train_staged.train(args)
    assert not outside.exists()


def test_resume_cross_checks_validation_decisions_with_monitor(
    tmp_path, tiny_runtime
):
    protocol = _make_files_and_protocol(tmp_path, stage1_steps=2)
    output = tmp_path / "run"
    assert train_staged.train(_args(protocol, "spatial_r60", output)) == 0
    validation_log = output / "validation_log.csv"
    with validation_log.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    rows[-1]["selected"] = "False" if rows[-1]["selected"] == "True" else "True"
    with validation_log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=train_staged.VALIDATION_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    args = _args(
        protocol,
        "spatial_r60",
        output,
        resume_from=output / "checkpoint.pt",
    )
    with pytest.raises(ValueError, match="monitor best_step"):
        train_staged.train(args)


def test_resume_of_early_stopped_checkpoint_is_idempotent(
    tmp_path, tiny_runtime, monkeypatch
):
    protocol_path = _make_files_and_protocol(tmp_path, stage1_steps=4)
    document = json.loads(protocol_path.read_text(encoding="utf-8"))
    document["stages"][0]["monitor"]["patience"] = 1
    protocol_path.write_text(json.dumps(document), encoding="utf-8")
    calls = {"count": 0}

    def worsening_metrics(*, dataloaders, **_kwargs):
        calls["count"] += 1
        value = float(calls["count"])
        return {
            name: {
                "deploy_loss": value,
                "action_l1": value,
                "force_l1": value,
                "num_samples": float(len(loader.dataset)),
                "num_episodes": float(len(loader.dataset.episode_paths)),
            }
            for name, loader in dataloaders.items()
        }

    monkeypatch.setattr(
        train_staged, "evaluate_named_deployment_metrics", worsening_metrics
    )
    output = tmp_path / "run"
    assert train_staged.train(_args(protocol_path, "spatial_r60", output)) == 0
    stopped = torch.load(output / "checkpoint.pt", map_location="cpu")
    assert stopped["training_state"]["stage_step"] == 2
    assert stopped["stop_reason"] == "early_stopping"

    def unexpected_validation(**_kwargs):
        raise AssertionError("terminal resume must not perform another validation")

    monkeypatch.setattr(
        train_staged, "evaluate_named_deployment_metrics", unexpected_validation
    )
    resume_args = _args(
        protocol_path,
        "spatial_r60",
        output,
        resume_from=output / "checkpoint.pt",
    )
    assert train_staged.train(resume_args) == 0
    resumed = torch.load(output / "checkpoint.pt", map_location="cpu")
    assert resumed["training_state"] == stopped["training_state"]
    assert resumed["stop_reason"] == "early_stopping"


def test_dry_run_validates_without_creating_output(tmp_path, tiny_runtime):
    protocol = _make_files_and_protocol(tmp_path)
    output = tmp_path / "dry"
    assert train_staged.train(_args(protocol, "spatial_r60", output, dry_run=True)) == 0
    assert not output.exists()


def test_formal_training_rejects_tampered_normalization_provenance(
    tmp_path, tiny_runtime
):
    protocol_path = _make_files_and_protocol(tmp_path)
    protocol = train_staged.load_protocol(protocol_path)
    stats = torch.load(protocol.normalization.stats_path, map_location="cpu")
    stats["population_identities"][0]["domain"] = "tampered_domain"
    torch.save(stats, protocol.normalization.stats_path)

    with pytest.raises(ValueError, match="population_identities SHA256 mismatch"):
        train_staged.train(
            _args(protocol_path, "spatial_r60", tmp_path / "tampered", dry_run=True)
        )


def test_legacy_and_skip_verification_flags_are_dry_run_only(
    tmp_path, tiny_runtime
):
    protocol = _make_files_and_protocol(tmp_path)
    args = _args(protocol, "spatial_r60", tmp_path / "output")
    args.skip_dataset_file_verification = True
    with pytest.raises(ValueError, match="restricted to --dry-run"):
        train_staged.train(args)

    args.dry_run = True
    assert train_staged.train(args) == 0


def test_normalization_manifest_identity_binding_rejects_wrong_domain_or_sha(tmp_path):
    train_path = tmp_path / "train.hdf5"
    train_path.write_bytes(b"train")
    entry = EpisodeManifestEntry(
        identity=EpisodeIdentity.from_path(train_path, str(uuid.uuid4())),
        domain="r60_visual",
        split="train",
    )
    manifest = DatasetManifest([entry])
    correct = {
        "population_identities": [
            {
                "path": str(train_path),
                "domain": "r60_visual",
                "file_sha256": entry.identity.file_sha256,
            }
        ]
    }

    train_staged._validate_normalization_manifest_identities(
        correct,
        manifest,
        allow_legacy=False,
    )
    wrong_domain = {
        "population_identities": [
            {**correct["population_identities"][0], "domain": "r2_contact"}
        ]
    }
    with pytest.raises(ValueError, match="domain_or_sha_mismatch=1"):
        train_staged._validate_normalization_manifest_identities(
            wrong_domain,
            manifest,
            allow_legacy=False,
        )
    wrong_sha = {
        "population_identities": [
            {**correct["population_identities"][0], "file_sha256": "0" * 64}
        ]
    }
    with pytest.raises(ValueError, match="domain_or_sha_mismatch=1"):
        train_staged._validate_normalization_manifest_identities(
            wrong_sha,
            manifest,
            allow_legacy=False,
        )


def test_formal_normalization_requires_balanced_estimator_and_dataset_semantics(
    tmp_path,
):
    protocol_path = _make_files_and_protocol(tmp_path)
    protocol = train_staged.load_protocol(protocol_path)
    semantics = {
        "chunk_len": 2,
        "force_window_len": 2,
        "force_window_duration": 0.1,
        "camera_names": ("camera",),
        "image_size": (4, 4),
        "imagenet_normalize": False,
        "normalization_estimator": "balanced_raw",
    }
    train_staged._validate_normalization_dataset_semantics(
        semantics,
        protocol,
        allow_legacy=False,
    )
    with pytest.raises(ValueError, match="requires normalization_estimator"):
        train_staged._validate_normalization_dataset_semantics(
            {**semantics, "normalization_estimator": "legacy_chunked"},
            protocol,
            allow_legacy=False,
        )
    with pytest.raises(ValueError, match="chunk_len"):
        train_staged._validate_normalization_dataset_semantics(
            {**semantics, "chunk_len": 9},
            protocol,
            allow_legacy=False,
        )


def test_phase_catalog_semantics_and_quota_labels_must_match_exactly(tmp_path):
    protocol_path = _make_files_and_protocol(tmp_path)
    protocol = train_staged.load_protocol(protocol_path)
    semantics = {
        "action_mode": "action",
        "chunk_len": 2,
        "force_window_len": 2,
        "force_window_duration": 0.1,
        "camera_names": ["camera"],
        "image_size": [4, 4],
        "normalize_images": True,
            "imagenet_normalize": False,
            "image_alignment": "latest_past",
            "max_image_lag_seconds": 0.1,
            "include_force": True,
        "tolerate_length_mismatch": False,
        "max_length_mismatch": 0,
    }
    catalog = SimpleNamespace(labeler={"dataset_semantics": semantics})
    train_staged._validate_phase_catalog_semantics(
        catalog,
        protocol,
        source_name="r2_train",
    )
    with pytest.raises(ValueError, match="chunk_len"):
        train_staged._validate_phase_catalog_semantics(
            SimpleNamespace(
                labeler={"dataset_semantics": {**semantics, "chunk_len": 5}}
            ),
            protocol,
            source_name="r2_train",
        )

    train_staged._validate_phase_label_set(
        source_name="r2_train",
        requested_phases={"approach", "recovery"},
        catalog_phases={"recovery", "approach"},
    )
    with pytest.raises(ValueError, match="unrequested_in_catalog=.*insertion"):
        train_staged._validate_phase_label_set(
            source_name="r2_train",
            requested_phases={"approach", "recovery"},
            catalog_phases={"approach", "recovery", "insertion"},
        )
