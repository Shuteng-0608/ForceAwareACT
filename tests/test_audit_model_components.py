import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("torchvision")


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_model_components.py"

spec = importlib.util.spec_from_file_location("audit_model_components", SCRIPT_PATH)
audit_model_components = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit_model_components)


def _small_audit():
    config = dict(audit_model_components.DEFAULT_SYNTHETIC_CONFIG)
    config.update(
        {
            "d_model": 32,
            "z_dim": 8,
            "chunk_len": 4,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "max_force_window_len": 8,
        }
    )
    model = audit_model_components.build_model(config, device="cpu")
    return model, audit_model_components.parameter_report(model)


def test_parameter_groups_sum_to_total_parameter_count():
    model, report = _small_audit()

    expected_total = sum(parameter.numel() for parameter in model.parameters())
    grouped_total = sum(
        component["total_parameters"] for component in report["components"].values()
    )

    assert report["total_parameters"] == expected_total
    assert grouped_total == expected_total


def test_no_parameter_is_counted_twice():
    model, report = _small_audit()

    grouped_names = [
        name
        for component in report["components"].values()
        for name in component["parameter_names"]
    ]

    assert len(grouped_names) == len(set(grouped_names))
    assert sorted(grouped_names) == sorted(name for name, _ in model.named_parameters())
    assert report["duplicates"] == []


def test_unclassified_parameters_are_surfaced():
    class UnknownParameterModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.unexpected = nn.Parameter(torch.ones(3))

    report = audit_model_components.parameter_report(UnknownParameterModule())

    assert report["components"]["other_unclassified"]["total_parameters"] == 3
    assert report["unclassified_parameter_names"] == ["unexpected"]


def test_force_modules_have_nonzero_parameter_count():
    _model, report = _small_audit()

    assert report["components"]["force_temporal_encoder"]["total_parameters"] > 0
    assert report["components"]["force_vision_fusion"]["total_parameters"] > 0
    assert report["components"]["force_head"]["total_parameters"] > 0


def test_motion_cvae_audit_has_zero_contact_parameter_count():
    config = dict(audit_model_components.DEFAULT_SYNTHETIC_CONFIG)
    config.update(
        {
            "d_model": 32,
            "z_dim": 8,
            "chunk_len": 4,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "max_force_window_len": 8,
        }
    )
    model = audit_model_components.build_model(
        config,
        device="cpu",
        policy_variant="force_aware_motion_cvae",
    )
    report = audit_model_components.parameter_report(model)

    assert report["components"]["force_temporal_encoder"]["total_parameters"] > 0
    assert report["components"]["force_vision_fusion"]["total_parameters"] > 0
    assert report["components"]["force_head"]["total_parameters"] > 0
    assert report["components"]["contact_latent_prior_posterior"]["total_parameters"] == 0


def test_act_boundary_simulation_excludes_force_and_contact_modules_only():
    _model, report = _small_audit()

    act_boundary = report["act_boundary_simulation"]
    excluded_components = set(act_boundary["excluded_components"])

    assert {
        "force_temporal_encoder",
        "force_vision_fusion",
        "force_head",
        "contact_latent_prior_posterior",
        "other_unclassified",
    }.issubset(excluded_components)
    assert "vision_backbone" in act_boundary["included_components"]
    assert act_boundary["included_total_parameters"] == sum(
        report["components"][name]["total_parameters"]
        for name in act_boundary["included_components"]
    )


def test_script_audit_runs_on_cpu_without_data_or_checkpoints():
    audit = audit_model_components.create_audit(
        config=dict(audit_model_components.DEFAULT_SYNTHETIC_CONFIG),
        device="cpu",
    )

    assert audit["total_parameters"] > 0
    assert audit["checkpoint_path"] is None
    assert audit["pretrained_weights_downloaded"] is False
