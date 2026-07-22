from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from force_aware_act.training.optim import (
    build_named_parameter_groups,
    build_parameter_groups_from_specs,
    gradients_are_finite,
    nonfinite_gradient_names,
    set_batch_norm_eval,
    set_frozen_batch_norm_eval,
    validate_and_clip_gradients,
    validate_parameter_groups,
)


class _VisionEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.visual_proj = torch.nn.Linear(4, 4)


class _GroupedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = _VisionEncoder()
        self.force_encoder = torch.nn.Linear(4, 4)
        self.force_vision_cross_attention = torch.nn.Linear(4, 4)
        self.policy_encoder = torch.nn.Linear(4, 4)
        self.action_head = torch.nn.Linear(4, 2)
        self.force_head = torch.nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.vision_encoder.visual_proj(self.vision_encoder.backbone(value))
        hidden = self.force_encoder(hidden) + self.force_vision_cross_attention(hidden)
        hidden = self.policy_encoder(hidden)
        return self.action_head(hidden).sum() + self.force_head(hidden).sum()


def _by_name(groups: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(group["name"]): group for group in groups}


def test_named_groups_are_exhaustive_disjoint_and_apply_lr_multipliers() -> None:
    model = _GroupedModel()
    model.force_head.requires_grad_(False)
    groups = build_named_parameter_groups(
        model,
        base_lr=2.0e-4,
        weight_decay=1.0e-2,
        lr_multipliers={
            "vision_backbone": 0.05,
            "visual_projection": 0.25,
            "force_fusion": 1.0,
            "prediction_heads": 2.0,
            "policy_core": 0.5,
        },
    )
    named = _by_name(groups)
    assert list(named) == [
        "vision_backbone",
        "visual_projection",
        "force_fusion",
        "prediction_heads",
        "policy_core",
    ]
    assert named["vision_backbone"]["lr"] == pytest.approx(1.0e-5)
    assert named["visual_projection"]["lr"] == pytest.approx(5.0e-5)
    assert named["force_fusion"]["lr"] == pytest.approx(2.0e-4)
    assert named["prediction_heads"]["lr"] == pytest.approx(4.0e-4)
    assert named["policy_core"]["lr"] == pytest.approx(1.0e-4)
    assert all(group["weight_decay"] == 1.0e-2 for group in groups)

    all_names = [name for group in groups for name in group["param_names"]]
    expected_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert sorted(all_names) == sorted(expected_names)
    assert not any(name.startswith("force_head.") for name in all_names)
    assert len({id(parameter) for group in groups for parameter in group["params"]}) == len(
        expected_names
    )
    optimizer = torch.optim.AdamW(groups)
    assert [group["name"] for group in optimizer.param_groups] == list(named)


def test_explicit_specs_freeze_groups_and_use_first_match_order() -> None:
    model = _GroupedModel()
    specs = [
        {
            "name": "frozen_backbone",
            "prefixes": ("vision_encoder.backbone.",),
            "lr_multiplier": 0.0,
            "weight_decay": None,
            "trainable": False,
        },
        {
            "name": "vision_projection",
            "prefixes": ("vision_encoder.",),
            "lr_multiplier": 0.25,
            "weight_decay": 0.0,
            "trainable": True,
        },
        {
            "name": "force",
            "prefixes": ("force_encoder.", "force_vision_cross_attention."),
            "lr_multiplier": 1.0,
            "weight_decay": None,
            "trainable": True,
        },
        {
            "name": "rest",
            "prefixes": ("",),
            "lr_multiplier": 2.0,
            "weight_decay": None,
            "trainable": True,
        },
    ]
    groups = build_parameter_groups_from_specs(
        model,
        specs=specs,
        base_lr=1.0e-4,
        default_weight_decay=1.0e-2,
    )
    named = _by_name(groups)
    assert list(named) == ["vision_projection", "force", "rest"]
    assert all(
        not parameter.requires_grad
        for parameter in model.vision_encoder.backbone.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.vision_encoder.visual_proj.parameters()
    )
    assert not any(
        name.startswith("vision_encoder.backbone.")
        for group in groups
        for name in group["param_names"]
    )
    assert named["vision_projection"]["lr"] == pytest.approx(2.5e-5)
    assert named["vision_projection"]["weight_decay"] == 0.0
    assert named["force"]["weight_decay"] == 1.0e-2
    assert named["rest"]["lr"] == pytest.approx(2.0e-4)

    # A subsequent stage can explicitly unfreeze all parameters. Object specs
    # from the parsed protocol are accepted as well as mappings.
    next_stage = [
        SimpleNamespace(
            name="all",
            prefixes=("",),
            lr_multiplier=1.0,
            weight_decay=None,
            trainable=True,
        )
    ]
    next_groups = build_parameter_groups_from_specs(
        model,
        specs=next_stage,
        base_lr=5.0e-5,
        default_weight_decay=0.0,
    )
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert next_groups[0]["param_names"] == tuple(
        name for name, _ in model.named_parameters()
    )


def test_explicit_specs_resolve_overlaps_by_first_match() -> None:
    model = _GroupedModel()
    groups = build_parameter_groups_from_specs(
        model,
        base_lr=1.0e-3,
        default_weight_decay=0.0,
        specs=[
            {
                "name": "vision_first",
                "prefixes": ("vision_encoder.",),
                "lr_multiplier": 1.0,
                "weight_decay": None,
                "trainable": True,
            },
            {
                "name": "specific_later",
                "prefixes": ("vision_encoder.backbone.", "force_head."),
                "lr_multiplier": 1.0,
                "weight_decay": None,
                "trainable": True,
            },
            {
                "name": "rest",
                "prefixes": ("",),
                "lr_multiplier": 1.0,
                "weight_decay": None,
                "trainable": True,
            },
        ],
    )
    named = _by_name(groups)
    assert "vision_encoder.backbone.weight" in named["vision_first"]["param_names"]
    assert "vision_encoder.backbone.weight" not in named["specific_later"]["param_names"]
    assert "force_head.weight" in named["specific_later"]["param_names"]


def test_explicit_specs_reject_unmatched_and_misplaced_catch_all_transactionally() -> None:
    model = _GroupedModel()
    before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="unmatched"):
        build_parameter_groups_from_specs(
            model,
            base_lr=1.0e-3,
            default_weight_decay=0.0,
            specs=[
                {
                    "name": "only_action",
                    "prefixes": ("action_head.",),
                    "lr_multiplier": 1.0,
                    "weight_decay": None,
                    "trainable": False,
                }
            ],
        )
    assert before == {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }

    with pytest.raises(ValueError, match="catch-all"):
        build_parameter_groups_from_specs(
            model,
            base_lr=1.0e-3,
            default_weight_decay=0.0,
            specs=[
                {
                    "name": "all",
                    "prefixes": ("",),
                    "lr_multiplier": 1.0,
                    "weight_decay": None,
                    "trainable": True,
                },
                {
                    "name": "later",
                    "prefixes": ("action_head.",),
                    "lr_multiplier": 1.0,
                    "weight_decay": None,
                    "trainable": True,
                },
            ],
        )


def test_explicit_specs_reject_empty_frozen_group_before_catch_all() -> None:
    model = _GroupedModel()
    before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="frozen_typo.*matched no parameters"):
        build_parameter_groups_from_specs(
            model,
            base_lr=1.0e-3,
            default_weight_decay=0.0,
            specs=[
                {
                    "name": "frozen_typo",
                    "prefixes": ("vision_encoder.typo.",),
                    "lr_multiplier": 0.0,
                    "weight_decay": None,
                    "trainable": False,
                },
                {
                    "name": "rest",
                    "prefixes": ("",),
                    "lr_multiplier": 1.0,
                    "weight_decay": None,
                    "trainable": True,
                },
            ],
        )
    assert before == {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }


def test_overlapping_prefixes_and_unknown_multiplier_are_rejected() -> None:
    model = _GroupedModel()
    with pytest.raises(ValueError, match="matches multiple groups"):
        build_named_parameter_groups(
            model,
            base_lr=1.0e-3,
            weight_decay=0.0,
            group_prefixes={
                "all_vision": ("vision_encoder.",),
                "backbone": ("vision_encoder.backbone.",),
            },
            fallback_group="other",
            lr_multipliers={"all_vision": 1.0, "backbone": 1.0, "other": 1.0},
        )
    with pytest.raises(ValueError, match="unknown LR multiplier"):
        build_named_parameter_groups(
            model,
            base_lr=1.0e-3,
            weight_decay=0.0,
            lr_multipliers={"typo": 1.0},
        )


@pytest.mark.parametrize(
    ("base_lr", "weight_decay", "match"),
    [
        (0.0, 0.0, "base_lr"),
        (float("nan"), 0.0, "base_lr"),
        (1.0e-3, -0.1, "weight_decay"),
    ],
)
def test_group_hyperparameters_are_validated(
    base_lr: float, weight_decay: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_named_parameter_groups(
            _GroupedModel(), base_lr=base_lr, weight_decay=weight_decay
        )


def test_group_validator_rejects_duplicate_omitted_frozen_and_foreign_parameters() -> None:
    model = _GroupedModel()
    parameters = dict(model.named_parameters())

    duplicate = [
        {
            "name": "first",
            "params": list(model.parameters()),
        },
        {
            "name": "second",
            "params": [parameters["action_head.weight"]],
        },
    ]
    with pytest.raises(ValueError, match="appears in both"):
        validate_parameter_groups(model, duplicate)

    omitted = [{"name": "partial", "params": [parameters["action_head.weight"]]}]
    with pytest.raises(ValueError, match="omit"):
        validate_parameter_groups(model, omitted)

    model.force_head.requires_grad_(False)
    groups = build_named_parameter_groups(model, base_lr=1.0e-3, weight_decay=0.0)
    groups[0]["params"].append(parameters["force_head.weight"])
    groups[0]["param_names"] = tuple(groups[0]["param_names"]) + ("force_head.weight",)
    with pytest.raises(ValueError, match="frozen parameter"):
        validate_parameter_groups(model, groups)

    model = _GroupedModel()
    groups = build_named_parameter_groups(model, base_lr=1.0e-3, weight_decay=0.0)
    foreign = torch.nn.Parameter(torch.ones(1))
    groups[0]["params"].append(foreign)
    groups[0]["param_names"] = tuple(groups[0]["param_names"]) + ("foreign",)
    with pytest.raises(ValueError, match="outside the model"):
        validate_parameter_groups(model, groups)


def test_finite_gradient_check_and_global_norm_clipping() -> None:
    model = torch.nn.Linear(3, 2)
    model(torch.ones(4, 3)).sum().backward()
    assert gradients_are_finite(model)
    expected_norm = math.sqrt(
        sum(float(parameter.grad.square().sum()) for parameter in model.parameters())
    )

    result = validate_and_clip_gradients(model, max_norm=0.25)
    assert result.total_norm == pytest.approx(expected_norm)
    assert result.max_norm == 0.25
    assert result.was_clipped
    assert result.parameter_count == 2
    clipped_norm = math.sqrt(
        sum(float(parameter.grad.square().sum()) for parameter in model.parameters())
    )
    assert clipped_norm <= 0.25 + 1.0e-6


def test_nonfinite_gradient_reports_names_and_prevents_clipping() -> None:
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    model.weight.grad[0, 0] = float("nan")
    assert not gradients_are_finite(model)
    assert nonfinite_gradient_names(model) == ("weight",)
    with pytest.raises(FloatingPointError, match="weight"):
        validate_and_clip_gradients(model, max_norm=1.0)


def test_frozen_batch_norm_helper_prevents_running_stat_updates() -> None:
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(3),
        torch.nn.BatchNorm1d(3),
    )
    model[0].requires_grad_(False)
    model.train()
    frozen = set_frozen_batch_norm_eval(model)
    assert frozen == ("0",)
    assert not model[0].training
    assert model[1].training

    before = model[0].running_mean.clone()
    model(torch.full((4, 3), 5.0))
    assert torch.equal(model[0].running_mean, before)
    assert not torch.equal(model[1].running_mean, torch.zeros(3))


def test_explicit_batch_norm_helper_freezes_running_stats_but_not_affine() -> None:
    module = torch.nn.Sequential(torch.nn.BatchNorm1d(3))
    module.train()
    frozen = set_batch_norm_eval(module, name_prefix="vision_encoder.backbone")
    assert frozen == ("vision_encoder.backbone.0",)
    assert not module[0].training
    assert all(parameter.requires_grad for parameter in module[0].parameters())
    before = module[0].running_mean.clone()
    module(torch.full((4, 3), 5.0)).sum().backward()
    assert torch.equal(module[0].running_mean, before)
    assert all(parameter.grad is not None for parameter in module[0].parameters())


def test_gradient_clip_handles_no_gradients_and_validates_limit() -> None:
    model = torch.nn.Linear(2, 1)
    result = validate_and_clip_gradients(model, max_norm=1.0)
    assert result.total_norm == 0.0
    assert not result.was_clipped
    assert result.parameter_count == 0
    with pytest.raises(ValueError, match="max_norm"):
        validate_and_clip_gradients(model, max_norm=0.0)
