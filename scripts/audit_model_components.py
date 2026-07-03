#!/usr/bin/env python3
"""Audit ForceAwareACT parameter counts by architectural component."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.models import (  # noqa: E402
    ACTPolicyBaseline,
    ForceAwareACTMotionCVAEPolicy,
    ForceAwareACTPolicy,
)


COMPONENT_ORDER = (
    "vision_backbone",
    "state_projection",
    "transformer_encoder",
    "transformer_decoder",
    "action_queries_action_head",
    "motion_latent_modules",
    "force_temporal_encoder",
    "force_vision_fusion",
    "force_head",
    "contact_latent_prior_posterior",
    "other_unclassified",
)

DEFAULT_SYNTHETIC_CONFIG: dict[str, Any] = {
    "pretrained_resnet18": False,
    "freeze_resnet18": False,
    "d_model": 128,
    "z_dim": 16,
    "q_dim": 7,
    "action_dim": 7,
    "force_dim": 6,
    "chunk_len": 10,
    "nhead": 4,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dim_feedforward": 256,
    "dropout": 0.0,
    "max_force_window_len": 20,
}

DEFAULT_ACT_SYNTHETIC_CONFIG: dict[str, Any] = {
    key: value
    for key, value in DEFAULT_SYNTHETIC_CONFIG.items()
    if key not in {"force_dim", "max_force_window_len"}
}

ACT_BOUNDARY_COMPONENTS = {
    "vision_backbone",
    "state_projection",
    "transformer_encoder",
    "transformer_decoder",
    "action_queries_action_head",
    "motion_latent_modules",
}


def policy_variant_from_checkpoint(path: Path) -> str:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dict")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        return "force_aware_act"
    return str(config.get("policy_variant", "force_aware_act"))


def model_config_from_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dict")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a dict")
    model_config = dict(config.get("model", {}))
    if not model_config:
        raise KeyError("checkpoint config is missing model settings")
    if "pretrained_vision" in model_config and "pretrained_resnet18" not in model_config:
        model_config["pretrained_resnet18"] = model_config.pop("pretrained_vision")
    model_config.setdefault("pretrained_resnet18", False)
    model_config.setdefault("dropout", 0.0)
    model_config.setdefault(
        "max_force_window_len",
        max(int(config.get("force_window_len", model_config.get("chunk_len", 20))), 20),
    )
    model_config.setdefault("freeze_resnet18", False)
    if str(config.get("policy_variant", "force_aware_act")) == "act_baseline":
        model_config.pop("force_dim", None)
        model_config.pop("max_force_window_len", None)
    return model_config


def build_model(
    config: dict[str, Any],
    device: str = "cpu",
    policy_variant: str = "force_aware_act",
):
    safe_config = dict(config)
    safe_config["pretrained_resnet18"] = False
    if policy_variant == "act_baseline":
        safe_config.pop("force_dim", None)
        safe_config.pop("max_force_window_len", None)
        return ACTPolicyBaseline(**safe_config).to(torch.device(device))
    if policy_variant == "force_aware_act":
        return ForceAwareACTPolicy(**safe_config).to(torch.device(device))
    if policy_variant == "force_aware_motion_cvae":
        return ForceAwareACTMotionCVAEPolicy(**safe_config).to(torch.device(device))
    raise ValueError(
        "policy_variant must be 'force_aware_act', 'force_aware_motion_cvae', or 'act_baseline'"
    )


def classify_parameter(name: str) -> str:
    if name.startswith("vision_encoder."):
        return "vision_backbone"
    if name.startswith("joint_encoder."):
        return "state_projection"
    if name.startswith("policy_encoder."):
        return "transformer_encoder"
    if name.startswith("policy_decoder."):
        return "transformer_decoder"
    if name == "future_queries" or name.startswith("action_head."):
        return "action_queries_action_head"
    if name.startswith("motion_posterior.") or name.startswith("motion_latent_proj."):
        return "motion_latent_modules"
    if name.startswith("force_encoder."):
        return "force_temporal_encoder"
    if name.startswith("force_vision_cross_attention."):
        return "force_vision_fusion"
    if name.startswith("force_head."):
        return "force_head"
    if (
        name.startswith("contact_posterior.")
        or name.startswith("contact_prior.")
        or name.startswith("contact_latent_proj.")
    ):
        return "contact_latent_prior_posterior"
    return "other_unclassified"


def _empty_component(name: str) -> dict[str, Any]:
    return {
        "component": name,
        "total_parameters": 0,
        "trainable_parameters": 0,
        "percentage_of_total": 0.0,
        "parameter_names": [],
    }


def parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    components = {name: _empty_component(name) for name in COMPONENT_ORDER}
    seen: set[str] = set()
    duplicates: list[str] = []

    for name, parameter in model.named_parameters():
        if name in seen:
            duplicates.append(name)
        seen.add(name)
        component_name = classify_parameter(name)
        component = components[component_name]
        count = int(parameter.numel())
        component["total_parameters"] += count
        if parameter.requires_grad:
            component["trainable_parameters"] += count
        component["parameter_names"].append(name)

    total_parameters = sum(item["total_parameters"] for item in components.values())
    trainable_parameters = sum(item["trainable_parameters"] for item in components.values())
    for component in components.values():
        if total_parameters:
            component["percentage_of_total"] = (
                100.0 * component["total_parameters"] / total_parameters
            )

    unclassified_names = list(components["other_unclassified"]["parameter_names"])
    act_boundary = {
        "included_components": sorted(ACT_BOUNDARY_COMPONENTS),
        "excluded_components": [
            name for name in COMPONENT_ORDER if name not in ACT_BOUNDARY_COMPONENTS
        ],
        "included_total_parameters": sum(
            components[name]["total_parameters"] for name in ACT_BOUNDARY_COMPONENTS
        ),
        "included_trainable_parameters": sum(
            components[name]["trainable_parameters"] for name in ACT_BOUNDARY_COMPONENTS
        ),
        "excluded_parameter_names": [
            parameter_name
            for parameter_name, _parameter in model.named_parameters()
            if classify_parameter(parameter_name) not in ACT_BOUNDARY_COMPONENTS
        ],
    }

    return {
        "components": components,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "duplicates": duplicates,
        "unclassified_parameter_names": unclassified_names,
        "act_boundary_simulation": act_boundary,
    }


def create_audit(
    config: dict[str, Any],
    device: str = "cpu",
    checkpoint_path: Path | None = None,
    policy_variant: str = "force_aware_act",
) -> dict[str, Any]:
    model = build_model(config, device=device, policy_variant=policy_variant)
    report = parameter_report(model)
    return {
        "policy_variant": policy_variant,
        "model_configuration": dict(config),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "pretrained_weights_downloaded": False,
        **report,
    }


def create_comparison_audit(device: str = "cpu") -> dict[str, Any]:
    force_audit = create_audit(
        config=dict(DEFAULT_SYNTHETIC_CONFIG),
        device=device,
        policy_variant="force_aware_act",
    )
    act_audit = create_audit(
        config=dict(DEFAULT_ACT_SYNTHETIC_CONFIG),
        device=device,
        policy_variant="act_baseline",
    )
    return {
        "force_aware_act": force_audit,
        "act_baseline": act_audit,
        "parameter_count_difference_force_minus_act": (
            force_audit["total_parameters"] - act_audit["total_parameters"]
        ),
        "trainable_parameter_count_difference_force_minus_act": (
            force_audit["trainable_parameters"] - act_audit["trainable_parameters"]
        ),
    }


def _json_ready(audit: dict[str, Any], include_parameter_names: bool) -> dict[str, Any]:
    if "components" not in audit:
        return {
            key: _json_ready(value, include_parameter_names)
            if isinstance(value, dict) and "components" in value
            else value
            for key, value in audit.items()
        }
    ready = dict(audit)
    components = {}
    for name, component in audit["components"].items():
        component_copy = dict(component)
        if not include_parameter_names:
            component_copy.pop("parameter_names", None)
        components[name] = component_copy
    ready["components"] = components
    if not include_parameter_names:
        ready["act_boundary_simulation"] = dict(audit["act_boundary_simulation"])
        ready["act_boundary_simulation"].pop("excluded_parameter_names", None)
    return ready


def _print_text_report(audit: dict[str, Any], show_parameter_names: bool) -> None:
    print(f"{audit.get('policy_variant', 'force_aware_act')} component parameter audit")
    print(f"total_parameters={audit['total_parameters']}")
    print(f"trainable_parameters={audit['trainable_parameters']}")
    print(f"checkpoint_path={audit['checkpoint_path']}")
    for name in COMPONENT_ORDER:
        component = audit["components"][name]
        print(
            f"{name}: total={component['total_parameters']} "
            f"trainable={component['trainable_parameters']} "
            f"pct={component['percentage_of_total']:.6f}"
        )
        if show_parameter_names:
            for parameter_name in component["parameter_names"]:
                print(f"  {parameter_name}")
    if audit["duplicates"]:
        print("duplicate_parameter_names:")
        for name in audit["duplicates"]:
            print(f"  {name}")
    if audit["unclassified_parameter_names"]:
        print("unclassified_parameter_names:")
        for name in audit["unclassified_parameter_names"]:
            print(f"  {name}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-from-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--policy-variant",
        choices=("force_aware_act", "force_aware_motion_cvae", "act_baseline", "both"),
        default="force_aware_act",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--include-frozen",
        action="store_true",
        help="Accepted for audit ergonomics; total and trainable counts are always reported.",
    )
    parser.add_argument("--show-parameter-names", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    checkpoint_path = args.config_from_checkpoint.expanduser() if args.config_from_checkpoint else None
    if args.policy_variant == "both" and checkpoint_path is not None:
        raise ValueError("--policy-variant both cannot be combined with --config-from-checkpoint")
    if args.policy_variant == "both":
        audit = create_comparison_audit(device=args.device)
        _print_text_report(audit["force_aware_act"], show_parameter_names=args.show_parameter_names)
        _print_text_report(audit["act_baseline"], show_parameter_names=args.show_parameter_names)
        print(
            "parameter_count_difference_force_minus_act="
            f"{audit['parameter_count_difference_force_minus_act']}"
        )
    else:
        policy_variant = (
            policy_variant_from_checkpoint(checkpoint_path)
            if checkpoint_path is not None
            else args.policy_variant
        )
        default_config = (
            DEFAULT_ACT_SYNTHETIC_CONFIG
            if policy_variant == "act_baseline"
            else DEFAULT_SYNTHETIC_CONFIG
        )
        config = (
            model_config_from_checkpoint(checkpoint_path)
            if checkpoint_path is not None
            else dict(default_config)
        )
        audit = create_audit(
            config=config,
            device=args.device,
            checkpoint_path=checkpoint_path,
            policy_variant=policy_variant,
        )
        _print_text_report(audit, show_parameter_names=args.show_parameter_names)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w") as handle:
            json.dump(
                _json_ready(audit, include_parameter_names=args.show_parameter_names),
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        print(f"saved_json={args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
