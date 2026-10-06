import copy
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from force_aware_act.real_robot.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from force_aware_act.real_robot.config import RealRobotDataConfig
from force_aware_act.real_robot.reader import RealRobotEpisodeRecord
from force_aware_act.real_robot.training_sampling import (
    EPISODIC_UNIFORM_SAMPLING, EPISODIC_UNIFORM_VERSION, RealRobotTrainingSampler,
)
from force_aware_act.real_robot.training import (
    MODEL_KINDS, RealRobotACTCriterion, RealRobotOptimizationConfig, build_model,
    make_model_config,
)


def _runner():
    spec = importlib.util.spec_from_file_location("real_robot_runner", Path(__file__).parents[1] / "scripts/train_real_robot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recording_split_rejects_fragment_leakage_and_single_episode():
    split = _runner().choose_split
    records = tuple(RealRobotEpisodeRecord(name, f"{name}/episode.hdf5", 20, ("d455", "d405"))
                    for name in ("a", "b", "c"))
    groups = {"a": "source1", "b": "source1", "c": "source2"}
    train, val, manifest = split(records, groups, mode="train", seed=2, fraction=.2)
    assert {groups[r.episode_id] for r in train}.isdisjoint(groups[r.episode_id] for r in val)
    with pytest.raises(ValueError, match="leaks"):
        split(records, groups, mode="train", seed=2, fraction=.2,
              supplied={"train_episode_ids": ["a", "c"], "validation_episode_ids": ["b"]})
    with pytest.raises(ValueError, match="independent recordings"):
        split(records[:1], {"a": "source1"}, mode="train", seed=2, fraction=.2)
    train, val, _ = split(records[:1], {"a": "source1"}, mode="smoke", seed=2, fraction=.2)
    assert len(train) == 1 and not val


def test_resume_cannot_mix_another_run_or_rewind_existing_metrics(tmp_path):
    runner = _runner()
    metadata = _metadata()
    (tmp_path / "run_manifest.json").write_text(json.dumps(metadata))
    (tmp_path / "metrics.jsonl").write_text('{"step": 3}\n')
    runner.validate_output_identity(tmp_path, metadata, 3)
    with pytest.raises(ValueError, match="newer"):
        runner.validate_output_identity(tmp_path, metadata, 2)
    changed = copy.deepcopy(metadata)
    changed["model_kind"] = "act"
    with pytest.raises(ValueError, match="another run"):
        runner.validate_output_identity(tmp_path, changed, 3)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_real_robot_model_profile_preserves_resolution_and_time_units(kind):
    config = make_model_config(kind, RealRobotDataConfig(), compact=True)
    assert (config.image_width, config.image_height) == (640, 480)
    assert config.chunk_len == 100
    assert config.q_dim == config.action_dim == 7
    if kind != "act":
        assert config.force_dim == 7
        assert config.max_online_force_intervals == 20
        assert config.online_force_window_len == 60
        assert config.force_sample_rate_hz == 300
        assert config.policy_sample_rate_hz == 100


@pytest.mark.parametrize("field,value", [("action_rate_hz", 100.0), ("torque_rate_hz", 300.0)])
def test_rate_config_cannot_promote_epoch_nanoseconds_to_float(field, value):
    with pytest.raises(ValueError, match="integer"):
        RealRobotDataConfig(**{field: value})


def test_real_robot_act_loss_does_not_shrink_when_more_labels_are_missing():
    criterion = RealRobotACTCriterion(RealRobotOptimizationConfig())
    predicted = torch.ones(1, 3, 7, requires_grad=True)
    target = torch.zeros_like(predicted)
    target[:, 1:] = float("nan")
    outputs = {"pred_action": predicted, "mu_motion": torch.zeros(1, 4), "logvar_motion": torch.zeros(1, 4)}
    losses = criterion(outputs, target, torch.tensor([[False, True, True]]))
    assert losses["loss_action"].item() == 1
    losses["loss_total"].backward()
    assert predicted.grad[:, 1:].eq(0).all()


def _metadata():
    data = RealRobotDataConfig()
    return {"model_kind": "contact", "model_config": {"force_dim": 7},
            "data_config": data.to_dict(), "data_contract": data.contract_dict(),
            "normalization": {"force_mean": [0.] * 7},
            "split_manifest": {"train_episode_ids": ["a"], "validation_episode_ids": []},
            "source_fingerprints": {"a": {"sha256": "original"}},
            "training_config": asdict(RealRobotOptimizationConfig()),
            "run_config": {"mode": "smoke", "steps": 3, "batch_size": 2, "seed": 0,
                           "sampling": "uniform_valid_anchor_with_replacement_v1"}}


def test_checkpoint_restores_optimizer_rng_and_sampling_and_rejects_contract_changes(tmp_path):
    model = torch.nn.Linear(7, 7)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 7)).sum().backward()
    optimizer.step()
    sampler = torch.Generator().manual_seed(11)
    torch.randint(40, (2,), generator=sampler)
    path = tmp_path / "last.pt"
    metadata = _metadata()
    save_checkpoint(path, model=model, optimizer=optimizer, sampler=sampler, global_step=3, metadata=metadata)
    next_indices = torch.randint(40, (2,), generator=sampler)
    next_noise = torch.randn(3)
    payload = read_checkpoint(path)
    with torch.no_grad():
        model.weight.fill_(0)
    assert restore_checkpoint(payload, model=model, optimizer=optimizer, sampler=sampler, expected=metadata) == 3
    assert torch.equal(next_indices, torch.randint(40, (2,), generator=sampler))
    assert torch.equal(next_noise, torch.randn(3))
    assert model.weight.abs().sum() > 0
    for key, replacement in (("data_contract", {"contact_representation": "ft_wrench"}),
                             ("source_fingerprints", {"a": {"sha256": "changed"}}),
                             ("normalization", {"force_mean": [1.] * 7})):
        altered = copy.deepcopy(metadata)
        altered[key] = replacement
        before = model.weight.detach().clone()
        with pytest.raises(ValueError, match=key):
            restore_checkpoint(payload, model=model, optimizer=optimizer, sampler=sampler, expected=altered)
        torch.testing.assert_close(before, model.weight)


def test_checkpoint_rejects_episodic_state_damage_and_mode_switch_before_model_mutation(tmp_path):
    model = torch.nn.Linear(7, 7)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = RealRobotTrainingSampler([2, 5, 1], mode=EPISODIC_UNIFORM_SAMPLING, seed=11)
    sampler.sample_batch(2)
    metadata = _metadata()
    metadata["run_config"]["sampling"] = EPISODIC_UNIFORM_VERSION
    path = tmp_path / "episodic.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, sampler=sampler, global_step=1, metadata=metadata)
    payload = read_checkpoint(path)
    with torch.no_grad():
        model.weight.fill_(0)
    before = model.weight.detach().clone()

    damaged = copy.deepcopy(payload)
    damaged["sampler_state"]["cursor"] = 4
    with pytest.raises(ValueError, match="sampler"):
        restore_checkpoint(damaged, model=model, optimizer=optimizer, sampler=sampler, expected=metadata)
    torch.testing.assert_close(before, model.weight, rtol=0, atol=0)
    with pytest.raises(ValueError, match="run_config.sampling"):
        restore_checkpoint(payload, model=model, optimizer=optimizer, sampler=sampler, expected=_metadata())
    torch.testing.assert_close(before, model.weight, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["contact", "act"])
def test_loading_pretrained_config_uses_saved_weights_without_download(kind, monkeypatch):
    import force_aware_act.models.act_aligned.backbone as backbone
    original = backbone._make_resnet18
    def guarded(*, pretrained, frozen_batch_norm):
        assert not pretrained
        return original(pretrained=False, frozen_batch_norm=frozen_batch_norm)
    monkeypatch.setattr(backbone, "_make_resnet18", guarded)
    import force_aware_act.models.official_act.backbone as official_backbone
    monkeypatch.setattr(official_backbone, "_make_resnet18", guarded)
    config = make_model_config(kind, RealRobotDataConfig(), compact=True)
    values = asdict(config)
    values.update(pretrained_backbone=True, imagenet_normalize=True)
    model = build_model(kind, values, initialize_pretrained=False)
    assert model.config.imagenet_normalize
    actual_backbone = model.backbone if kind == "act" else model.vision_backbone
    assert actual_backbone.config.imagenet_normalize
