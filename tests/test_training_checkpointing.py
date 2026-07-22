from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

from force_aware_act.training.checkpointing import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    INIT_COMPATIBILITY_KEYS,
    RESUME_COMPATIBILITY_KEYS,
    build_checkpoint_v2,
    capture_rng_state,
    file_sha256,
    initialize_model_from_checkpoint,
    mapping_sha256,
    restore_rng_state,
    resume_training_from_checkpoint,
    save_checkpoint_atomic,
    state_dict_sha256,
    validate_checkpoint_compatibility,
    validate_checkpoint_v2_payload,
)


PROTOCOL_SHA = "1" * 64
NORMALIZATION_SHA = "2" * 64


def test_init_and_resume_compatibility_keys_are_intentionally_distinct() -> None:
    assert "optimizer_groups" not in INIT_COMPATIBILITY_KEYS
    assert RESUME_COMPATIBILITY_KEYS == INIT_COMPATIBILITY_KEYS + (
        "optimizer_groups",
        "training_device",
        "freeze_vision_batch_norm",
        "run_id",
        "run_manifest_sha256",
        "stage_initial_global_step",
        "checkpoint_every_steps",
        "validation_every_steps",
        "minimum_validations",
        "training_code_sha256",
        "runtime_versions",
    )


def _config(*, chunk_len: int = 4) -> dict[str, object]:
    return {
        "policy_variant": "force_aware_motion_cvae",
        "action_mode": "joint_pos",
        "chunk_len": chunk_len,
        "force_window_len": 5,
        "force_window_duration": 0.2,
        "image_size": (64, 64),
        "camera_names": ("cam_high", "cam_wrist"),
        "imagenet_normalize": False,
        "image_alignment": "latest_past",
        "max_image_lag_seconds": 0.1,
        "model": {
            "d_model": 8,
            "action_dim": 2,
            "chunk_len": chunk_len,
        },
        "optimizer_groups": [
            {
                "name": "all",
                "param_names": [
                    "linear.weight",
                    "linear.bias",
                    "norm.weight",
                    "norm.bias",
                ],
                "lr_multiplier": 1.0,
                "weight_decay": 0.01,
            }
        ],
        "training_device": "cpu",
        "freeze_vision_batch_norm": False,
        "run_id": "a" * 32,
        "run_manifest_sha256": "b" * 64,
        "stage_initial_global_step": 0,
        "checkpoint_every_steps": 2,
        "validation_every_steps": 1,
        "minimum_validations": 1,
        "training_code_sha256": "c" * 64,
        "runtime_versions": {"python": "test", "torch": "test"},
        # Runtime fields intentionally do not participate in compatibility.
        "learning_rate": 1.0e-3,
        "output_dir": "first",
    }


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        # BatchNorm contributes a scalar num_batches_tracked state entry.
        self.norm = torch.nn.BatchNorm1d(2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(value))


class _StatefulSampler:
    def __init__(self, cursor: int = 0) -> None:
        self.cursor = cursor

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.cursor = state["cursor"]


def _populate_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    model.train()
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _build_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Optional[object] = None,
    sampler: Optional[object] = None,
    generator: Optional[torch.Generator] = None,
) -> dict[str, object]:
    return build_checkpoint_v2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        named_generators=None if generator is None else {"loader": generator},
        config=_config(),
        global_step=17,
        stage_step=7,
        epoch=3,
        step_in_epoch=2,
        stage_name="contact_specialization",
        stage_index=2,
        protocol_sha256=PROTOCOL_SHA,
        normalization_sha256=NORMALIZATION_SHA,
        parent_checkpoint_sha256="3" * 64,
        monitor_state={"best_metric": 0.25, "epochs_without_improvement": 1},
    )


def test_hash_helpers_are_deterministic_and_sensitive(tmp_path: Path) -> None:
    first = {"b": 2, "a": (1, Path("some/path"))}
    second = {"a": [1, "some/path"], "b": 2}
    assert mapping_sha256(first) == mapping_sha256(second)

    model = _TinyModel()
    original_hash = state_dict_sha256(model.state_dict())
    assert original_hash == state_dict_sha256(dict(reversed(model.state_dict().items())))
    with torch.no_grad():
        model.linear.weight[0, 0] += 1.0
    assert state_dict_sha256(model.state_dict()) != original_hash

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"staged-training")
    assert file_sha256(artifact) == hashlib.sha256(b"staged-training").hexdigest()


def test_capture_and_restore_rng_state_including_named_generator() -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    generator = torch.Generator().manual_seed(14)
    state = capture_rng_state({"loader": generator})

    expected = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    generator.manual_seed(99)

    restore_rng_state(state, {"loader": generator})
    actual = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])

    with pytest.raises(CheckpointCompatibilityError, match="named generator mismatch"):
        restore_rng_state(state, {})


def test_atomic_v2_save_round_trip_and_replace(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "checkpoint.pt"
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    first_payload = _build_payload(model, optimizer)
    first_hash = save_checkpoint_atomic(first_payload, path)
    assert first_hash == file_sha256(path)
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))

    with torch.no_grad():
        model.linear.bias.add_(2.0)
    second_payload = _build_payload(model, optimizer)
    second_hash = save_checkpoint_atomic(second_payload, path)
    assert second_hash == file_sha256(path)
    assert second_hash != first_hash
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_v2_payload(loaded)
    # Version-2 payloads intentionally contain only restricted-unpickler-safe
    # values so existing inference readers using torch.load defaults still work.
    weights_only_loaded = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_v2_payload(weights_only_loaded)
    assert loaded["step"] == 17
    assert loaded["stage"] == {"name": "contact_specialization", "index": 2}
    assert loaded["lineage"]["parent_checkpoint_sha256"] == "3" * 64
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_weights_only_initialization_supports_legacy_and_leaves_optimizer_fresh(
    tmp_path: Path,
) -> None:
    source_model = _TinyModel()
    with torch.no_grad():
        source_model.linear.weight.fill_(4.0)
    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": source_model.state_dict(),
            "config": _config(),
            "step": 123,
            "optimizer_state_dict": {"must_not": "load"},
        },
        legacy_path,
    )

    target_model = _TinyModel()
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=5.0e-4)
    init_config = {**_config(), "learning_rate": 5.0e-4, "output_dir": "new"}
    init_config["optimizer_groups"] = [{"different": "stage optimizer is reset"}]
    result = initialize_model_from_checkpoint(
        target_model,
        legacy_path,
        expected_config=init_config,
    )
    assert result.schema_version is None
    assert result.source_step == 123
    assert result.compatibility_validated
    assert result.checkpoint_sha256 == file_sha256(legacy_path)
    assert state_dict_sha256(target_model.state_dict()) == state_dict_sha256(
        source_model.state_dict()
    )
    assert target_optimizer.state_dict()["state"] == {}

    raw_target = _TinyModel()
    initialize_model_from_checkpoint(raw_target, source_model.state_dict())
    assert state_dict_sha256(raw_target.state_dict()) == state_dict_sha256(
        source_model.state_dict()
    )


def test_strict_resume_restores_all_training_and_rng_state(tmp_path: Path) -> None:
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    source_model = _TinyModel()
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1.0e-3)
    source_scheduler = torch.optim.lr_scheduler.StepLR(
        source_optimizer, step_size=1, gamma=0.5
    )
    _populate_optimizer(source_model, source_optimizer)
    source_scheduler.step()
    source_sampler = _StatefulSampler(cursor=31)
    source_generator = torch.Generator().manual_seed(24)

    payload = _build_payload(
        source_model,
        source_optimizer,
        scheduler=source_scheduler,
        sampler=source_sampler,
        generator=source_generator,
    )
    path = tmp_path / "resume.pt"
    checkpoint_hash = save_checkpoint_atomic(payload, path)
    expected_rng_values = (
        random.random(),
        float(np.random.rand()),
        torch.rand(2),
        torch.rand(2, generator=source_generator),
    )

    resumed_model = _TinyModel()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=9.0e-3)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer, step_size=1, gamma=0.1
    )
    resumed_sampler = _StatefulSampler(cursor=0)
    resumed_generator = torch.Generator().manual_seed(999)
    result = resume_training_from_checkpoint(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        sampler=resumed_sampler,
        named_generators={"loader": resumed_generator},
        source=path,
        expected_config={**_config(), "learning_rate": 9.0e-3, "output_dir": "resume"},
        expected_stage_name="contact_specialization",
        expected_stage_index=2,
        expected_protocol_sha256=PROTOCOL_SHA,
        expected_normalization_sha256=NORMALIZATION_SHA,
    )

    assert result.checkpoint_sha256 == checkpoint_hash
    assert (result.global_step, result.stage_step, result.epoch, result.step_in_epoch) == (
        17,
        7,
        3,
        2,
    )
    assert result.monitor_state == {
        "best_metric": 0.25,
        "epochs_without_improvement": 1,
    }
    assert state_dict_sha256(resumed_model.state_dict()) == state_dict_sha256(
        source_model.state_dict()
    )
    assert resumed_optimizer.param_groups[0]["lr"] == source_optimizer.param_groups[0]["lr"]
    assert resumed_scheduler.last_epoch == source_scheduler.last_epoch
    assert resumed_sampler.cursor == 31

    actual_rng_values = (
        random.random(),
        float(np.random.rand()),
        torch.rand(2),
        torch.rand(2, generator=resumed_generator),
    )
    assert actual_rng_values[0] == expected_rng_values[0]
    assert actual_rng_values[1] == expected_rng_values[1]
    assert torch.equal(actual_rng_values[2], expected_rng_values[2])
    assert torch.equal(actual_rng_values[3], expected_rng_values[3])


def test_resume_rejects_legacy_checkpoint(tmp_path: Path) -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": _config()}, path)
    with pytest.raises(CheckpointCompatibilityError, match="schema version 2"):
        resume_training_from_checkpoint(
            model=model,
            optimizer=optimizer,
            source=path,
            expected_config=_config(),
            expected_stage_name="contact_specialization",
            expected_stage_index=2,
            expected_protocol_sha256=PROTOCOL_SHA,
            expected_normalization_sha256=NORMALIZATION_SHA,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("chunk_len", 8, "chunk_len"),
        ("camera_names", ("different",), "camera_names"),
        ("model", {"d_model": 16}, "model"),
    ],
)
def test_compatibility_validation_rejects_semantic_changes(
    field: str, value: object, match: str
) -> None:
    current = _config()
    current[field] = value
    with pytest.raises(CheckpointCompatibilityError, match=match):
        validate_checkpoint_compatibility(_config(), current)


def test_strict_resume_rejects_optimizer_group_manifest_change() -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    payload = _build_payload(model, optimizer)
    current = _config()
    current["optimizer_groups"] = [{"name": "different-order"}]
    with pytest.raises(CheckpointCompatibilityError, match="optimizer_groups"):
        resume_training_from_checkpoint(
            model=model,
            optimizer=optimizer,
            source=payload,
            expected_config=current,
            expected_stage_name="contact_specialization",
            expected_stage_index=2,
            expected_protocol_sha256=PROTOCOL_SHA,
            expected_normalization_sha256=NORMALIZATION_SHA,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_device", "cuda:0"),
        ("freeze_vision_batch_norm", True),
        ("training_code_sha256", "f" * 64),
        ("runtime_versions", {"python": "changed", "torch": "changed"}),
        ("run_id", "f" * 32),
        ("run_manifest_sha256", "f" * 64),
    ],
)
def test_strict_resume_rejects_runtime_or_run_identity_change(
    field: str, value: object
) -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    payload = _build_payload(model, optimizer)
    current = _config()
    current[field] = value
    with pytest.raises(CheckpointCompatibilityError, match=field):
        resume_training_from_checkpoint(
            model=model,
            optimizer=optimizer,
            source=payload,
            expected_config=current,
            expected_stage_name="contact_specialization",
            expected_stage_index=2,
            expected_protocol_sha256=PROTOCOL_SHA,
            expected_normalization_sha256=NORMALIZATION_SHA,
        )


def test_resume_rejects_stage_and_provenance_mismatch() -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    payload = _build_payload(model, optimizer)
    with pytest.raises(CheckpointCompatibilityError, match="different stage"):
        resume_training_from_checkpoint(
            model=model,
            optimizer=optimizer,
            source=payload,
            expected_config=_config(),
            expected_stage_name="wrong",
            expected_stage_index=2,
            expected_protocol_sha256=PROTOCOL_SHA,
            expected_normalization_sha256=NORMALIZATION_SHA,
        )
    with pytest.raises(CheckpointCompatibilityError, match="protocol SHA256"):
        resume_training_from_checkpoint(
            model=model,
            optimizer=optimizer,
            source=payload,
            expected_config=_config(),
            expected_stage_name="contact_specialization",
            expected_stage_index=2,
            expected_protocol_sha256="9" * 64,
            expected_normalization_sha256=NORMALIZATION_SHA,
        )


def test_integrity_tampering_and_partial_resume_are_rejected() -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    payload = _build_payload(model, optimizer)
    payload["model_state_dict"]["linear.weight"][0, 0] += 1.0
    with pytest.raises(CheckpointIntegrityError, match="model_state_dict"):
        validate_checkpoint_v2_payload(payload)

    auxiliary_tampered = _build_payload(model, optimizer)
    auxiliary_tampered["monitor_state"]["best_metric"] = 999.0
    with pytest.raises(CheckpointIntegrityError, match="auxiliary state"):
        validate_checkpoint_v2_payload(auxiliary_tampered)

    lineage_tampered = _build_payload(model, optimizer)
    lineage_tampered["lineage"]["parent_checkpoint_sha256"] = "9" * 64
    with pytest.raises(CheckpointIntegrityError, match="auxiliary state"):
        validate_checkpoint_v2_payload(lineage_tampered)

    clean_payload = _build_payload(model, optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    with pytest.raises(CheckpointCompatibilityError, match="scheduler_state_dict"):
        resume_training_from_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            source=clean_payload,
            expected_config=_config(),
            expected_stage_name="contact_specialization",
            expected_stage_index=2,
            expected_protocol_sha256=PROTOCOL_SHA,
            expected_normalization_sha256=NORMALIZATION_SHA,
        )
