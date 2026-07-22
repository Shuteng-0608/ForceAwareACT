"""Optimizer parameter grouping and gradient-safety helpers.

The staged protocol intentionally updates the vision backbone more slowly than
the force/fusion and prediction modules.  This module creates auditable named
parameter groups and verifies that every *trainable* parameter appears exactly
once while frozen parameters never enter the optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


DEFAULT_GROUP_PREFIXES: Mapping[str, Tuple[str, ...]] = {
    "vision_backbone": ("vision_encoder.backbone.",),
    "visual_projection": ("vision_encoder.visual_proj.",),
    "force_fusion": ("force_encoder.", "force_vision_cross_attention."),
    "prediction_heads": ("action_head.", "force_head."),
}
DEFAULT_FALLBACK_GROUP = "policy_core"
DEFAULT_LR_MULTIPLIERS: Mapping[str, float] = {
    "vision_backbone": 0.1,
    "visual_projection": 0.5,
    "force_fusion": 1.0,
    "prediction_heads": 1.0,
    "policy_core": 1.0,
}


@dataclass(frozen=True)
class GradientClipResult:
    """Result of a finite-gradient check followed by global norm clipping."""

    total_norm: float
    max_norm: float
    was_clipped: bool
    parameter_count: int


def _spec_field(spec: Any, field: str, index: int) -> Any:
    if isinstance(spec, Mapping):
        if field not in spec:
            raise ValueError(f"parameter group spec {index} is missing {field!r}")
        return spec[field]
    if not hasattr(spec, field):
        raise ValueError(f"parameter group spec {index} is missing {field!r}")
    return getattr(spec, field)


def build_parameter_groups_from_specs(
    model: torch.nn.Module,
    *,
    specs: Sequence[Any],
    base_lr: float,
    default_weight_decay: float,
) -> list[Dict[str, object]]:
    """Build ordered parameter groups from explicit protocol specifications.

    Each spec may be a mapping or an object exposing ``name``, ``prefixes``,
    ``lr_multiplier``, ``weight_decay``, and ``trainable``. Prefix resolution
    is first-match wins. An empty prefix is a catch-all and is valid only in
    the last spec. Every model parameter must match; this makes protocol typos
    fail before training. Specs with ``trainable=False`` set matching
    parameters' ``requires_grad`` flag to false and exclude them from the
    returned optimizer groups; a later stage can explicitly unfreeze them with
    a trainable spec.
    """

    base_lr = _validate_nonnegative_finite(base_lr, "base_lr", positive=True)
    default_weight_decay = _validate_nonnegative_finite(
        default_weight_decay, "default_weight_decay", positive=False
    )
    if not specs:
        raise ValueError("parameter group specs must not be empty")

    resolved_specs = []
    seen_names = set()
    for index, spec in enumerate(specs):
        name = _spec_field(spec, "name", index)
        prefixes = _spec_field(spec, "prefixes", index)
        multiplier = _spec_field(spec, "lr_multiplier", index)
        weight_decay = _spec_field(spec, "weight_decay", index)
        trainable = _spec_field(spec, "trainable", index)
        if not isinstance(name, str) or not name:
            raise ValueError(f"parameter group spec {index} name must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate parameter group spec name: {name}")
        seen_names.add(name)
        if not isinstance(prefixes, (list, tuple)) or not prefixes:
            raise ValueError(f"parameter group spec {name!r} prefixes must be non-empty")
        if any(not isinstance(prefix, str) for prefix in prefixes):
            raise ValueError(f"parameter group spec {name!r} prefixes must be strings")
        if "" in prefixes and index != len(specs) - 1:
            raise ValueError("the empty catch-all prefix is only valid in the final spec")
        if not isinstance(trainable, bool):
            raise ValueError(f"parameter group spec {name!r} trainable must be boolean")
        multiplier = _validate_nonnegative_finite(
            multiplier,
            f"parameter group spec {name!r} lr_multiplier",
            positive=False,
        )
        if weight_decay is None:
            resolved_weight_decay = default_weight_decay
        else:
            resolved_weight_decay = _validate_nonnegative_finite(
                weight_decay,
                f"parameter group spec {name!r} weight_decay",
                positive=False,
            )
        resolved_specs.append(
            {
                "name": name,
                "prefixes": tuple(prefixes),
                "lr_multiplier": multiplier,
                "weight_decay": resolved_weight_decay,
                "trainable": trainable,
            }
        )

    assignments: Dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        spec["name"]: [] for spec in resolved_specs
    }
    unmatched = []
    seen_parameter_ids = set()
    for parameter_name, parameter in model.named_parameters():
        matched_spec = next(
            (
                spec
                for spec in resolved_specs
                if any(parameter_name.startswith(prefix) for prefix in spec["prefixes"])
            ),
            None,
        )
        if matched_spec is None:
            unmatched.append(parameter_name)
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            raise ValueError(f"model exposes parameter {parameter_name!r} more than once")
        seen_parameter_ids.add(parameter_id)
        assignments[matched_spec["name"]].append((parameter_name, parameter))

    if unmatched:
        raise ValueError(
            f"parameter group specs leave {len(unmatched)} parameters unmatched: "
            + ", ".join(unmatched[:10])
        )

    for spec in resolved_specs:
        if not assignments[spec["name"]]:
            raise ValueError(
                f"parameter group spec {spec['name']!r} matched no parameters"
            )

    trainable_specs = [spec for spec in resolved_specs if spec["trainable"]]
    if not trainable_specs:
        raise ValueError("parameter group specs freeze every model parameter")

    # Only mutate requires_grad after every spec and assignment has validated.
    for spec in resolved_specs:
        for _, parameter in assignments[spec["name"]]:
            parameter.requires_grad_(spec["trainable"])

    parameter_groups: list[Dict[str, object]] = []
    for spec in resolved_specs:
        if not spec["trainable"]:
            continue
        entries = assignments[spec["name"]]
        parameter_groups.append(
            {
                "name": spec["name"],
                "params": [parameter for _, parameter in entries],
                "param_names": tuple(name for name, _ in entries),
                "lr": base_lr * spec["lr_multiplier"],
                "lr_multiplier": spec["lr_multiplier"],
                "weight_decay": spec["weight_decay"],
            }
        )
    validate_parameter_groups(model, parameter_groups)
    return parameter_groups


def _validate_nonnegative_finite(value: float, name: str, *, positive: bool) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or (numeric <= 0.0 if positive else numeric < 0.0):
        qualifier = "positive and finite" if positive else "non-negative and finite"
        raise ValueError(f"{name} must be {qualifier}")
    return numeric


def _validate_group_definition(
    group_prefixes: Mapping[str, Sequence[str]], fallback_group: str
) -> None:
    if not isinstance(fallback_group, str) or not fallback_group:
        raise ValueError("fallback_group must be a non-empty string")
    if fallback_group in group_prefixes:
        raise ValueError("fallback_group must not also appear in group_prefixes")
    for group_name, prefixes in group_prefixes.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("parameter group names must be non-empty strings")
        if not prefixes:
            raise ValueError(f"parameter group {group_name!r} must define prefixes")
        for prefix in prefixes:
            if not isinstance(prefix, str) or not prefix:
                raise ValueError(
                    f"parameter group {group_name!r} has an invalid empty prefix"
                )


def build_named_parameter_groups(
    model: torch.nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    lr_multipliers: Optional[Mapping[str, float]] = None,
    group_prefixes: Mapping[str, Sequence[str]] = DEFAULT_GROUP_PREFIXES,
    fallback_group: str = DEFAULT_FALLBACK_GROUP,
) -> list[Dict[str, object]]:
    """Build disjoint, exhaustive optimizer groups for trainable parameters.

    Prefixes are matched against names returned by ``model.named_parameters``.
    A trainable parameter matching multiple named groups is rejected instead of
    depending on dictionary order.  Unmatched parameters enter ``fallback_group``.
    Parameters with ``requires_grad=False`` are excluded completely.
    """

    base_lr = _validate_nonnegative_finite(base_lr, "base_lr", positive=True)
    weight_decay = _validate_nonnegative_finite(
        weight_decay, "weight_decay", positive=False
    )
    _validate_group_definition(group_prefixes, fallback_group)

    all_group_names = set(group_prefixes) | {fallback_group}
    resolved_multipliers = dict(DEFAULT_LR_MULTIPLIERS)
    if lr_multipliers is not None:
        unknown = sorted(set(lr_multipliers) - all_group_names)
        if unknown:
            raise ValueError("unknown LR multiplier groups: " + ", ".join(unknown))
        resolved_multipliers.update(lr_multipliers)
    missing_multipliers = sorted(all_group_names - set(resolved_multipliers))
    if missing_multipliers:
        raise ValueError(
            "missing LR multipliers for groups: " + ", ".join(missing_multipliers)
        )
    for group_name in all_group_names:
        resolved_multipliers[group_name] = _validate_nonnegative_finite(
            resolved_multipliers[group_name],
            f"lr_multipliers[{group_name!r}]",
            positive=True,
        )

    grouped: Dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        name: [] for name in group_prefixes
    }
    grouped[fallback_group] = []
    seen_parameter_ids = set()
    trainable_parameter_ids = set()

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in trainable_parameter_ids:
            raise ValueError(
                f"trainable parameter {parameter_name!r} is exposed more than once by the model"
            )
        trainable_parameter_ids.add(parameter_id)

        matches = [
            group_name
            for group_name, prefixes in group_prefixes.items()
            if any(parameter_name.startswith(prefix) for prefix in prefixes)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"parameter {parameter_name!r} matches multiple groups: {sorted(matches)}"
            )
        group_name = matches[0] if matches else fallback_group
        if parameter_id in seen_parameter_ids:
            raise ValueError(f"parameter {parameter_name!r} was assigned more than once")
        seen_parameter_ids.add(parameter_id)
        grouped[group_name].append((parameter_name, parameter))

    if not trainable_parameter_ids:
        raise ValueError("model has no trainable parameters")
    if seen_parameter_ids != trainable_parameter_ids:
        raise RuntimeError("internal error: optimizer parameter grouping is not exhaustive")

    parameter_groups: list[Dict[str, object]] = []
    # Preserve declared order for stable optimizer state and logging.
    for group_name in list(group_prefixes) + [fallback_group]:
        entries = grouped[group_name]
        if not entries:
            continue
        multiplier = resolved_multipliers[group_name]
        parameter_groups.append(
            {
                "name": group_name,
                "params": [parameter for _, parameter in entries],
                "param_names": tuple(name for name, _ in entries),
                "lr": base_lr * multiplier,
                "lr_multiplier": multiplier,
                "weight_decay": weight_decay,
            }
        )

    validate_parameter_groups(model, parameter_groups)
    return parameter_groups


def validate_parameter_groups(
    model: torch.nn.Module,
    parameter_groups: Sequence[Mapping[str, object]],
) -> None:
    """Reject duplicate, omitted, frozen, or foreign optimizer parameters."""

    expected = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    frozen = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    encountered: Dict[int, str] = {}
    group_names = set()

    for group_index, group in enumerate(parameter_groups):
        group_name = group.get("name")
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(f"optimizer group {group_index} has no valid name")
        if group_name in group_names:
            raise ValueError(f"duplicate optimizer group name: {group_name}")
        group_names.add(group_name)
        parameters = group.get("params")
        if not isinstance(parameters, (list, tuple)) or not parameters:
            raise ValueError(f"optimizer group {group_name!r} has no parameters")
        param_names = group.get("param_names")
        if param_names is not None and len(param_names) != len(parameters):
            raise ValueError(
                f"optimizer group {group_name!r} param_names length mismatch"
            )

        for parameter_index, parameter in enumerate(parameters):
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(
                    f"optimizer group {group_name!r} contains a non-Parameter value"
                )
            parameter_id = id(parameter)
            if parameter_id in frozen:
                raise ValueError(
                    f"frozen parameter {frozen[parameter_id]!r} is present in optimizer"
                )
            if parameter_id not in expected:
                raise ValueError(
                    f"optimizer group {group_name!r} contains a parameter outside the model"
                )
            if parameter_id in encountered:
                raise ValueError(
                    f"parameter {expected[parameter_id]!r} appears in both "
                    f"{encountered[parameter_id]!r} and {group_name!r}"
                )
            encountered[parameter_id] = group_name
            if param_names is not None and param_names[parameter_index] != expected[parameter_id]:
                raise ValueError(
                    f"optimizer group {group_name!r} records the wrong parameter name "
                    f"for {expected[parameter_id]!r}"
                )

    omitted = sorted(expected[parameter_id] for parameter_id in set(expected) - set(encountered))
    if omitted:
        raise ValueError(
            f"optimizer groups omit {len(omitted)} trainable parameters: "
            + ", ".join(omitted[:10])
        )


def nonfinite_gradient_names(model: torch.nn.Module) -> tuple[str, ...]:
    """Return parameter names whose existing gradients contain NaN or Inf."""

    invalid = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        if not bool(torch.isfinite(values).all().item()):
            invalid.append(name)
    return tuple(invalid)


def gradients_are_finite(model: torch.nn.Module) -> bool:
    """Return whether all gradients that currently exist are finite."""

    return not nonfinite_gradient_names(model)


def set_frozen_batch_norm_eval(model: torch.nn.Module) -> tuple[str, ...]:
    """Put BatchNorm modules with frozen affine parameters in evaluation mode.

    ``requires_grad=False`` does not stop BatchNorm running statistics from
    changing after ``model.train()``. Call this helper immediately after each
    ``model.train()`` transition when a backbone is frozen. Affine-free
    BatchNorm modules are left unchanged because their ownership cannot be
    inferred from parameters alone.
    """

    frozen_names = []
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        affine_parameters = list(module.parameters(recurse=False))
        if affine_parameters and all(
            not parameter.requires_grad for parameter in affine_parameters
        ):
            module.eval()
            frozen_names.append(module_name)
    return tuple(frozen_names)


def set_batch_norm_eval(
    module: torch.nn.Module, *, name_prefix: str = ""
) -> tuple[str, ...]:
    """Freeze running-stat updates for every BatchNorm below ``module``.

    This deliberately changes only module training mode. Affine parameters keep
    their existing ``requires_grad`` flags, so the optimizer protocol remains
    the sole authority for whether gamma/beta are updated.
    """

    names = []
    for module_name, child in module.named_modules():
        if not isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            continue
        child.eval()
        qualified_name = ".".join(
            part for part in (name_prefix.rstrip("."), module_name) if part
        )
        names.append(qualified_name)
    return tuple(names)


def validate_and_clip_gradients(
    model: torch.nn.Module,
    *,
    max_norm: float,
    norm_type: float = 2.0,
) -> GradientClipResult:
    """Reject non-finite gradients, then clip the global trainable gradient norm.

    This helper is intended immediately after ``loss.backward()`` and before
    ``optimizer.step()``.  It reports the pre-clipping norm returned by PyTorch.
    """

    max_norm = _validate_nonnegative_finite(max_norm, "max_norm", positive=True)
    if math.isinf(float(norm_type)):
        normalized_norm_type = float(norm_type)
        if normalized_norm_type < 0:
            raise ValueError("norm_type must be positive and finite, or positive infinity")
    else:
        normalized_norm_type = _validate_nonnegative_finite(
            norm_type, "norm_type", positive=True
        )

    invalid = nonfinite_gradient_names(model)
    if invalid:
        raise FloatingPointError(
            f"non-finite gradients in {len(invalid)} parameters: "
            + ", ".join(invalid[:10])
        )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters:
        return GradientClipResult(
            total_norm=0.0,
            max_norm=max_norm,
            was_clipped=False,
            parameter_count=0,
        )
    total_norm_tensor = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=normalized_norm_type,
        error_if_nonfinite=True,
    )
    total_norm = float(total_norm_tensor.detach().cpu().item())
    if not math.isfinite(total_norm):
        raise FloatingPointError("global gradient norm is non-finite")
    return GradientClipResult(
        total_norm=total_norm,
        max_norm=max_norm,
        was_clipped=total_norm > max_norm,
        parameter_count=len(parameters),
    )
