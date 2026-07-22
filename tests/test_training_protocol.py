import json

import pytest

from force_aware_act.training.protocol import load_protocol, protocol_sha256


def _protocol_document():
    return {
        "schema_version": 2,
        "run_name": "staged_test",
        "seed": 7,
        "deterministic": True,
        "model": {"policy_variant": "force_aware_motion_cvae", "d_model": 32},
        "dataset": {
            "action_mode": "action",
            "chunk_len": 4,
            "force_window_len": 6,
            "force_window_duration": 0.2,
            "image_size": [32, 48],
            "camera_names": ["ee_cam", "base_top_cam"],
            "image_alignment": "latest_past",
            "max_image_lag_seconds": 0.1,
        },
        "dataset_manifest": {"path": "dataset_manifest.json"},
        "normalization": {
            "stats_path": "stats.pt",
            "population_episode_lists": ["r60_train.txt", "r2_train.txt"],
            "domain_weights": {"r60_visual": 0.5, "r2_contact": 0.5},
        },
        "stages": [
            {
                "name": "spatial_r60",
                "sources": [
                    {
                        "name": "r60",
                        "domain": "r60_visual",
                        "episode_list": "r60_train.txt",
                        "expected_episode_count": 10,
                        "batch_quota": 4,
                    }
                ],
                "validation_domains": [
                    {
                        "name": "visual_wide",
                        "domain": "r60_visual",
                        "episode_list": "r60_val.txt",
                        "expected_episode_count": 2,
                    }
                ],
                "freeze_vision_batch_norm": False,
                "batch_size": 4,
                "samples_per_epoch": 16,
                "max_steps": 8,
                "validation_every_steps": 4,
                "checkpoint_every_steps": 4,
                "optimizer": {"base_lr": 0.0001},
                "monitor": {
                    "primary_domain": "visual_wide",
                    "aggregation": "episode_uniform",
                },
            },
            {
                "name": "contact_r2",
                "sources": [
                    {
                        "name": "r2",
                        "domain": "r2_contact",
                        "episode_list": "r2_train.txt",
                        "expected_episode_count": 10,
                        "batch_quota": 3,
                        "phase_quotas": {"pre_contact": 1, "contact": 2},
                        "min_episodes_per_phase": 2,
                        "sample_catalog": "r2_catalog.json",
                        "sample_catalog_sha256": "1" * 64,
                    },
                    {
                        "name": "r60_replay",
                        "domain": "r60_visual",
                        "episode_list": "r60_train.txt",
                        "expected_episode_count": 10,
                        "batch_quota": 1,
                    },
                ],
                "validation_domains": [
                    {
                        "name": "contact_rich",
                        "domain": "r2_contact",
                        "episode_list": "r2_val.txt",
                        "expected_episode_count": 2,
                    },
                    {
                        "name": "visual_wide",
                        "domain": "r60_visual",
                        "episode_list": "r60_val.txt",
                        "expected_episode_count": 2,
                    },
                ],
                "freeze_vision_batch_norm": True,
                "batch_size": 4,
                "samples_per_epoch": 16,
                "max_steps": 6,
                "validation_every_steps": 2,
                "checkpoint_every_steps": 2,
                "optimizer": {
                    "base_lr": 0.00003,
                    "weight_decay": 0.0001,
                    "max_grad_norm": 1.0,
                    "parameter_groups": [
                        {
                            "name": "vision",
                            "prefixes": ["vision_encoder.backbone."],
                            "lr_multiplier": 0.1,
                        },
                        {"name": "rest", "prefixes": [""], "lr_multiplier": 1.0},
                    ],
                },
                "monitor": {
                    "primary_domain": "contact_rich",
                    "aggregation": "episode_uniform",
                    "retention_domain": "visual_wide",
                    "max_retention_regression": 0.05,
                },
            },
        ],
        "test_episode_lists": {
            "visual_wide": {
                "domain": "r60_visual",
                "episode_list": "r60_test.txt",
                "expected_episode_count": 1,
            },
            "contact_rich": {
                "domain": "r2_contact",
                "episode_list": "r2_test.txt",
                "expected_episode_count": 1,
            },
        },
    }


def _write_protocol(tmp_path, document):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_protocol_resolves_paths_and_stage_specs(tmp_path):
    document = _protocol_document()
    protocol = load_protocol(_write_protocol(tmp_path, document))

    assert protocol.content_sha256 == protocol_sha256(document)
    assert protocol.model.policy_variant == "force_aware_motion_cvae"
    assert protocol.dataset.image_size == (32, 48)
    assert protocol.dataset_manifest.path == (tmp_path / "dataset_manifest.json").resolve()
    assert protocol.normalization.stats_path == (tmp_path / "stats.pt").resolve()
    stage = protocol.stage("contact_r2")
    assert stage.freeze_vision_batch_norm is True
    assert [source.batch_quota for source in stage.sources] == [3, 1]
    assert [source.expected_episode_count for source in stage.sources] == [10, 10]
    assert dict(stage.sources[0].phase_quotas) == {"contact": 2, "pre_contact": 1}
    assert stage.sources[0].sample_catalog_sha256 == "1" * 64
    assert stage.monitor.retention_domain == "visual_wide"
    assert stage.optimizer.parameter_groups[0].lr_multiplier == pytest.approx(0.1)
    assert stage.objective.lambda_force == pytest.approx(0.1)
    assert stage.objective.validation_deployment_mode == "auto"
    assert protocol.test_episode_lists["visual_wide"].domain == "r60_visual"
    assert protocol.test_episode_lists["visual_wide"].episode_list == (
        tmp_path / "r60_test.txt"
    ).resolve()


def test_protocol_hash_is_key_order_independent():
    document = _protocol_document()
    reordered = {key: document[key] for key in reversed(list(document))}
    assert protocol_sha256(document) == protocol_sha256(reordered)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.update({"unknown": 1}), "unknown keys"),
        (
            lambda d: d["stages"][0].update({"batch_size": 5}),
            "batch_quotas must sum",
        ),
        (
            lambda d: d["stages"][1]["sources"][0].update(
                {"phase_quotas": {"contact": 1}}
            ),
            "phase_quotas must sum",
        ),
        (
            lambda d: d["stages"][1]["monitor"].update(
                {"retention_domain": "missing"}
            ),
            "not a validation domain",
        ),
        (
            lambda d: d["model"].update({"d_model": 30, "nhead": 4}),
            "divisible",
        ),
        (
            lambda d: d["stages"][0]["optimizer"].update({"base_lr": 0.0}),
            "base_lr must be positive",
        ),
        (
            lambda d: d["model"].update({"action_dim": 8}),
            "action_dim must be 7",
        ),
        (
            lambda d: d["test_episode_lists"].update(
                {"visual_wide": "r60_test.txt"}
            ),
            "must be a JSON object",
        ),
        (
            lambda d: d["test_episode_lists"]["visual_wide"].pop("domain"),
            "missing required key 'domain'",
        ),
        (
            lambda d: d["stages"][0]["sources"][0].pop(
                "expected_episode_count"
            ),
            "missing required key 'expected_episode_count'",
        ),
        (
            lambda d: d["model"].update({"force_dim": 3}),
            "force_dim must be 6",
        ),
        (
            lambda d: d["stages"][0]["monitor"].update({"metric": "deploy_los"}),
            "metric must be one of",
        ),
        (
            lambda d: d["stages"][0]["monitor"].update({"min_validations": 3}),
            "schedules only 2 validations",
        ),
        (
            lambda d: d["stages"][0].pop("freeze_vision_batch_norm"),
            "missing required key 'freeze_vision_batch_norm'",
        ),
        (
            lambda d: d["stages"][1].update({"checkpoint_every_steps": 1}),
            "must equal validation_every_steps",
        ),
        (
            lambda d: d["stages"][1].update({"max_steps": 5}),
            "max_steps must be divisible",
        ),
        (
            lambda d: d["stages"][0]["monitor"].update({"aggregation": "sample"}),
            "must be one of: episode_uniform",
        ),
    ],
)
def test_protocol_rejects_invalid_configuration(tmp_path, mutate, match):
    document = _protocol_document()
    mutate(document)
    with pytest.raises(ValueError, match=match):
        load_protocol(_write_protocol(tmp_path, document))


def test_protocol_rejects_duplicate_stage_names(tmp_path):
    document = _protocol_document()
    document["stages"][1]["name"] = document["stages"][0]["name"]
    with pytest.raises(ValueError, match="duplicate names"):
        load_protocol(_write_protocol(tmp_path, document))


def test_protocol_rejects_noncanonical_digest(tmp_path):
    document = _protocol_document()
    document["normalization"]["sha256"] = "ABC"
    with pytest.raises(ValueError, match="64-character lowercase"):
        load_protocol(_write_protocol(tmp_path, document))


def test_phase_quotas_require_pinned_catalog_hash(tmp_path):
    document = _protocol_document()
    source = document["stages"][1]["sources"][0]
    source.pop("sample_catalog_sha256")

    with pytest.raises(ValueError, match="sample_catalog_sha256"):
        load_protocol(_write_protocol(tmp_path, document))


def test_non_catalog_source_rejects_catalog_hash(tmp_path):
    document = _protocol_document()
    source = document["stages"][0]["sources"][0]
    source["sample_catalog_sha256"] = "1" * 64

    with pytest.raises(ValueError, match="requires phase_quotas and sample_catalog"):
        load_protocol(_write_protocol(tmp_path, document))


def test_protocol_rejects_duplicate_keys_and_nonfinite_json(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_protocol(path)

    path.write_text('{"schema_version":1,"seed":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_protocol(path)
