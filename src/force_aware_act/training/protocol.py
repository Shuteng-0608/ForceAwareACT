"""Validated, hashable configuration for staged visual-force training.

The protocol is deliberately JSON-only so experiments do not depend on an
optional YAML parser.  Relative paths are resolved from the protocol file,
while the protocol hash is computed from the original JSON document.  This
keeps an experiment identity stable when the repository is moved.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PROTOCOL_SCHEMA_VERSION = 2
POLICY_VARIANTS = (
    "force_aware_act",
    "force_aware_motion_cvae",
    "force_aware_contact_cvae",
)
ACTION_MODES = (
    "joint_pos",
    "action",
    "joint_pos_command",
    "delta_joint_cmd",
    "delta_joint_pos_command",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically for hashing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"protocol contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str):
    raise ValueError(f"protocol contains non-finite JSON value {value}")


def protocol_sha256(document: Mapping[str, Any]) -> str:
    """Return the canonical content hash of a protocol document."""

    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    policy_variant: str
    pretrained_resnet18: bool
    d_model: int
    z_dim: int
    action_dim: int
    force_dim: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float


@dataclass(frozen=True)
class DatasetSpec:
    action_mode: str
    chunk_len: int
    force_window_len: int
    force_window_duration: float
    image_size: Tuple[int, int]
    camera_names: Tuple[str, ...]
    imagenet_normalize: bool
    strict_lengths: bool
    image_alignment: str
    max_image_lag_seconds: float


@dataclass(frozen=True)
class DatasetManifestSpec:
    path: Path
    expected_sha256: Optional[str]


@dataclass(frozen=True)
class NormalizationSpec:
    stats_path: Path
    population_episode_lists: Tuple[Path, ...]
    domain_weights: Mapping[str, float]
    expected_sha256: Optional[str]


@dataclass(frozen=True)
class ParameterGroupSpec:
    name: str
    prefixes: Tuple[str, ...]
    lr_multiplier: float
    weight_decay: Optional[float]
    trainable: bool


@dataclass(frozen=True)
class OptimizerSpec:
    base_lr: float
    weight_decay: float
    max_grad_norm: Optional[float]
    parameter_groups: Tuple[ParameterGroupSpec, ...]


@dataclass(frozen=True)
class ObjectiveSpec:
    lambda_force: float
    lambda_prior: float
    prior_loss_mode: str
    beta_motion_max: float
    beta_contact_max: float
    warmup_steps: int
    train_latent_mode: str
    train_contact_latent_mode: str
    validation_deployment_mode: str


@dataclass(frozen=True)
class SourceSpec:
    name: str
    domain: str
    episode_list: Path
    expected_episode_count: int
    batch_quota: int
    phase_quotas: Mapping[str, int]
    min_episodes_per_phase: Optional[int]
    sample_catalog: Optional[Path]
    sample_catalog_sha256: Optional[str]


@dataclass(frozen=True)
class ValidationDomainSpec:
    name: str
    domain: str
    episode_list: Path
    expected_episode_count: int


@dataclass(frozen=True)
class TestEpisodeListSpec:
    domain: str
    episode_list: Path
    expected_episode_count: int


@dataclass(frozen=True)
class MonitorSpec:
    primary_domain: str
    metric: str
    aggregation: str
    retention_domain: Optional[str]
    max_retention_regression: Optional[float]
    patience: int
    min_validations: int
    min_delta: float


MONITOR_METRICS = ("deploy_loss", "action_l1", "force_l1")
VALIDATION_AGGREGATIONS = ("episode_uniform",)


@dataclass(frozen=True)
class StageSpec:
    name: str
    sources: Tuple[SourceSpec, ...]
    validation_domains: Tuple[ValidationDomainSpec, ...]
    freeze_vision_batch_norm: bool
    batch_size: int
    samples_per_epoch: int
    max_steps: int
    validation_every_steps: int
    checkpoint_every_steps: int
    optimizer: OptimizerSpec
    objective: ObjectiveSpec
    monitor: MonitorSpec


@dataclass(frozen=True)
class ResolvedProtocol:
    source_path: Path
    schema_version: int
    run_name: str
    seed: int
    deterministic: bool
    model: ModelSpec
    dataset: DatasetSpec
    dataset_manifest: Optional[DatasetManifestSpec]
    normalization: NormalizationSpec
    stages: Tuple[StageSpec, ...]
    test_episode_lists: Mapping[str, TestEpisodeListSpec]
    content_sha256: str
    raw_document: Mapping[str, Any]

    def stage(self, name: str) -> StageSpec:
        """Return one named stage, rejecting unknown names explicitly."""

        matches = [stage for stage in self.stages if stage.name == name]
        if not matches:
            available = ", ".join(stage.name for stage in self.stages)
            raise KeyError(f"unknown stage {name!r}; available stages: {available}")
        return matches[0]


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: Sequence[str], context: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {', '.join(unknown)}")


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required key {key!r}")
    return mapping[key]


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _finite_float(value: Any, context: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _resolve_path(value: Any, base_dir: Path, context: str) -> Path:
    text = _nonempty_string(value, context)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _parse_model(value: Any) -> ModelSpec:
    context = "model"
    data = _require_mapping(value, context)
    allowed = (
        "policy_variant",
        "pretrained_resnet18",
        "d_model",
        "z_dim",
        "action_dim",
        "force_dim",
        "nhead",
        "num_encoder_layers",
        "num_decoder_layers",
        "dim_feedforward",
        "dropout",
    )
    _reject_unknown(data, allowed, context)
    variant = _nonempty_string(_required(data, "policy_variant", context), "model.policy_variant")
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"model.policy_variant must be one of: {', '.join(POLICY_VARIANTS)}")
    pretrained = data.get("pretrained_resnet18", False)
    if not isinstance(pretrained, bool):
        raise ValueError("model.pretrained_resnet18 must be boolean")
    dropout = _finite_float(data.get("dropout", 0.0), "model.dropout", minimum=0.0)
    if dropout >= 1.0:
        raise ValueError("model.dropout must be < 1")
    spec = ModelSpec(
        policy_variant=variant,
        pretrained_resnet18=pretrained,
        d_model=_positive_int(data.get("d_model", 128), "model.d_model"),
        z_dim=_positive_int(data.get("z_dim", 16), "model.z_dim"),
        action_dim=_positive_int(data.get("action_dim", 7), "model.action_dim"),
        force_dim=_positive_int(data.get("force_dim", 6), "model.force_dim"),
        nhead=_positive_int(data.get("nhead", 4), "model.nhead"),
        num_encoder_layers=_positive_int(
            data.get("num_encoder_layers", 1), "model.num_encoder_layers"
        ),
        num_decoder_layers=_positive_int(
            data.get("num_decoder_layers", 1), "model.num_decoder_layers"
        ),
        dim_feedforward=_positive_int(
            data.get("dim_feedforward", 256), "model.dim_feedforward"
        ),
        dropout=dropout,
    )
    if spec.d_model % spec.nhead != 0:
        raise ValueError("model.d_model must be divisible by model.nhead")
    return spec


def _parse_dataset(value: Any) -> DatasetSpec:
    context = "dataset"
    data = _require_mapping(value, context)
    allowed = (
        "action_mode",
        "chunk_len",
        "force_window_len",
        "force_window_duration",
        "image_size",
        "camera_names",
        "imagenet_normalize",
        "strict_lengths",
        "image_alignment",
        "max_image_lag_seconds",
    )
    _reject_unknown(data, allowed, context)
    action_mode = _nonempty_string(
        _required(data, "action_mode", context), "dataset.action_mode"
    )
    if action_mode not in ACTION_MODES:
        raise ValueError(f"dataset.action_mode must be one of: {', '.join(ACTION_MODES)}")
    image_size = data.get("image_size", [224, 224])
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError("dataset.image_size must be a two-element JSON array")
    cameras = data.get("camera_names", ["ee_cam", "base_top_cam"])
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("dataset.camera_names must be a non-empty JSON array")
    camera_names = tuple(
        _nonempty_string(camera, f"dataset.camera_names[{index}]")
        for index, camera in enumerate(cameras)
    )
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("dataset.camera_names contains duplicates")
    imagenet_normalize = data.get("imagenet_normalize", False)
    strict_lengths = data.get("strict_lengths", True)
    if not isinstance(imagenet_normalize, bool):
        raise ValueError("dataset.imagenet_normalize must be boolean")
    if not isinstance(strict_lengths, bool):
        raise ValueError("dataset.strict_lengths must be boolean")
    image_alignment = _nonempty_string(
        _required(data, "image_alignment", context), "dataset.image_alignment"
    )
    if image_alignment != "latest_past":
        raise ValueError(
            "formal staged protocols require dataset.image_alignment='latest_past'"
        )
    return DatasetSpec(
        action_mode=action_mode,
        chunk_len=_positive_int(data.get("chunk_len", 10), "dataset.chunk_len"),
        force_window_len=_positive_int(
            data.get("force_window_len", 20), "dataset.force_window_len"
        ),
        force_window_duration=_finite_float(
            data.get("force_window_duration", 0.25),
            "dataset.force_window_duration",
            minimum=0.0,
        ),
        image_size=(
            _positive_int(image_size[0], "dataset.image_size[0]"),
            _positive_int(image_size[1], "dataset.image_size[1]"),
        ),
        camera_names=camera_names,
        imagenet_normalize=imagenet_normalize,
        strict_lengths=strict_lengths,
        image_alignment=image_alignment,
        max_image_lag_seconds=_finite_float(
            _required(data, "max_image_lag_seconds", context),
            "dataset.max_image_lag_seconds",
            minimum=0.0,
        ),
    )


def _parse_normalization(value: Any, base_dir: Path) -> NormalizationSpec:
    context = "normalization"
    data = _require_mapping(value, context)
    allowed = ("stats_path", "population_episode_lists", "domain_weights", "sha256")
    _reject_unknown(data, allowed, context)
    populations = _required(data, "population_episode_lists", context)
    if not isinstance(populations, list) or not populations:
        raise ValueError("normalization.population_episode_lists must be a non-empty array")
    expected_sha256 = data.get("sha256")
    if expected_sha256 is not None:
        expected_sha256 = _nonempty_string(expected_sha256, "normalization.sha256").lower()
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise ValueError("normalization.sha256 must be a 64-character lowercase hex digest")
    raw_domain_weights = _require_mapping(
        _required(data, "domain_weights", context),
        "normalization.domain_weights",
    )
    if not raw_domain_weights:
        raise ValueError("normalization.domain_weights must not be empty")
    domain_weights = {
        _nonempty_string(name, "normalization.domain_weights key"): _finite_float(
            value,
            f"normalization.domain_weights[{name!r}]",
            minimum=0.0,
        )
        for name, value in raw_domain_weights.items()
    }
    if any(value <= 0.0 for value in domain_weights.values()):
        raise ValueError("normalization.domain_weights values must be positive")
    if not math.isclose(math.fsum(domain_weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("normalization.domain_weights must sum to 1")
    return NormalizationSpec(
        stats_path=_resolve_path(
            _required(data, "stats_path", context), base_dir, "normalization.stats_path"
        ),
        population_episode_lists=tuple(
            _resolve_path(item, base_dir, f"normalization.population_episode_lists[{index}]")
            for index, item in enumerate(populations)
        ),
        domain_weights=MappingProxyType(dict(sorted(domain_weights.items()))),
        expected_sha256=expected_sha256,
    )


def _parse_sha256(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    digest = _nonempty_string(value, context).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{context} must be a 64-character lowercase hex digest")
    return digest


def _parse_dataset_manifest(value: Any, base_dir: Path) -> Optional[DatasetManifestSpec]:
    if value is None:
        return None
    context = "dataset_manifest"
    data = _require_mapping(value, context)
    _reject_unknown(data, ("path", "sha256"), context)
    return DatasetManifestSpec(
        path=_resolve_path(_required(data, "path", context), base_dir, f"{context}.path"),
        expected_sha256=_parse_sha256(data.get("sha256"), f"{context}.sha256"),
    )


def _parse_parameter_group(value: Any, index: int) -> ParameterGroupSpec:
    context = f"optimizer.parameter_groups[{index}]"
    data = _require_mapping(value, context)
    allowed = ("name", "prefixes", "lr_multiplier", "weight_decay", "trainable")
    _reject_unknown(data, allowed, context)
    prefixes = _required(data, "prefixes", context)
    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError(f"{context}.prefixes must be a non-empty array")
    parsed_prefixes_list = []
    for prefix_index, prefix in enumerate(prefixes):
        if not isinstance(prefix, str):
            raise ValueError(f"{context}.prefixes[{prefix_index}] must be a string")
        # The empty prefix is an intentional catch-all group.  Optimizer group
        # resolution uses first-match wins, so it is normally listed last.
        parsed_prefixes_list.append(prefix)
    parsed_prefixes = tuple(parsed_prefixes_list)
    weight_decay = data.get("weight_decay")
    parsed_weight_decay = (
        None
        if weight_decay is None
        else _finite_float(weight_decay, f"{context}.weight_decay", minimum=0.0)
    )
    trainable = data.get("trainable", True)
    if not isinstance(trainable, bool):
        raise ValueError(f"{context}.trainable must be boolean")
    return ParameterGroupSpec(
        name=_nonempty_string(_required(data, "name", context), f"{context}.name"),
        prefixes=parsed_prefixes,
        lr_multiplier=_finite_float(
            data.get("lr_multiplier", 1.0), f"{context}.lr_multiplier", minimum=0.0
        ),
        weight_decay=parsed_weight_decay,
        trainable=trainable,
    )


def _parse_optimizer(value: Any, stage_context: str) -> OptimizerSpec:
    context = f"{stage_context}.optimizer"
    data = _require_mapping(value, context)
    allowed = ("base_lr", "weight_decay", "max_grad_norm", "parameter_groups")
    _reject_unknown(data, allowed, context)
    groups = data.get(
        "parameter_groups",
        [{"name": "all", "prefixes": [""], "lr_multiplier": 1.0}],
    )
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{context}.parameter_groups must be a non-empty array")
    parsed_groups = tuple(_parse_parameter_group(item, index) for index, item in enumerate(groups))
    names = [group.name for group in parsed_groups]
    if len(set(names)) != len(names):
        raise ValueError(f"{context}.parameter_groups contains duplicate names")
    max_grad_norm = data.get("max_grad_norm", 1.0)
    parsed_max_grad_norm = (
        None
        if max_grad_norm is None
        else _finite_float(max_grad_norm, f"{context}.max_grad_norm", minimum=0.0)
    )
    if parsed_max_grad_norm == 0:
        raise ValueError(f"{context}.max_grad_norm must be positive or null")
    base_lr = _finite_float(
        _required(data, "base_lr", context), f"{context}.base_lr", minimum=0.0
    )
    if base_lr == 0.0:
        raise ValueError(f"{context}.base_lr must be positive")
    return OptimizerSpec(
        base_lr=base_lr,
        weight_decay=_finite_float(
            data.get("weight_decay", 0.0), f"{context}.weight_decay", minimum=0.0
        ),
        max_grad_norm=parsed_max_grad_norm,
        parameter_groups=parsed_groups,
    )


def _parse_objective(value: Any, stage_context: str) -> ObjectiveSpec:
    context = f"{stage_context}.objective"
    data = _require_mapping(value, context)
    allowed = (
        "lambda_force",
        "lambda_prior",
        "prior_loss_mode",
        "beta_motion_max",
        "beta_contact_max",
        "warmup_steps",
        "train_latent_mode",
        "train_contact_latent_mode",
        "validation_deployment_mode",
    )
    _reject_unknown(data, allowed, context)
    prior_loss_mode = _nonempty_string(
        data.get("prior_loss_mode", "mse_mu"), f"{context}.prior_loss_mode"
    )
    if prior_loss_mode not in ("mse_mu", "kl_q_to_p"):
        raise ValueError(f"{context}.prior_loss_mode must be 'mse_mu' or 'kl_q_to_p'")
    train_latent_mode = _nonempty_string(
        data.get("train_latent_mode", "posterior"), f"{context}.train_latent_mode"
    )
    if train_latent_mode not in ("posterior", "zero"):
        raise ValueError(f"{context}.train_latent_mode must be 'posterior' or 'zero'")
    train_contact_latent_mode = _nonempty_string(
        data.get("train_contact_latent_mode", "posterior"),
        f"{context}.train_contact_latent_mode",
    )
    if train_contact_latent_mode != "posterior":
        raise ValueError(f"{context}.train_contact_latent_mode currently supports only 'posterior'")
    deployment_mode = _nonempty_string(
        data.get("validation_deployment_mode", "auto"),
        f"{context}.validation_deployment_mode",
    )
    if deployment_mode not in ("auto", "zero", "prior"):
        raise ValueError(
            f"{context}.validation_deployment_mode must be 'auto', 'zero', or 'prior'"
        )
    return ObjectiveSpec(
        lambda_force=_finite_float(
            data.get("lambda_force", 0.1), f"{context}.lambda_force", minimum=0.0
        ),
        lambda_prior=_finite_float(
            data.get("lambda_prior", 0.0), f"{context}.lambda_prior", minimum=0.0
        ),
        prior_loss_mode=prior_loss_mode,
        beta_motion_max=_finite_float(
            data.get("beta_motion_max", 1.0e-4),
            f"{context}.beta_motion_max",
            minimum=0.0,
        ),
        beta_contact_max=_finite_float(
            data.get("beta_contact_max", 1.0e-4),
            f"{context}.beta_contact_max",
            minimum=0.0,
        ),
        warmup_steps=_nonnegative_int(
            data.get("warmup_steps", 100), f"{context}.warmup_steps"
        ),
        train_latent_mode=train_latent_mode,
        train_contact_latent_mode=train_contact_latent_mode,
        validation_deployment_mode=deployment_mode,
    )


def _parse_source(value: Any, index: int, stage_context: str, base_dir: Path) -> SourceSpec:
    context = f"{stage_context}.sources[{index}]"
    data = _require_mapping(value, context)
    allowed = (
        "name",
        "domain",
        "episode_list",
        "expected_episode_count",
        "batch_quota",
        "phase_quotas",
        "min_episodes_per_phase",
        "sample_catalog",
        "sample_catalog_sha256",
    )
    _reject_unknown(data, allowed, context)
    phase_quotas_value = data.get("phase_quotas", {})
    if not isinstance(phase_quotas_value, Mapping):
        raise ValueError(f"{context}.phase_quotas must be a JSON object")
    phase_quotas: Dict[str, int] = {}
    for phase, quota in phase_quotas_value.items():
        phase_name = _nonempty_string(phase, f"{context}.phase_quotas key")
        phase_quotas[phase_name] = _positive_int(
            quota, f"{context}.phase_quotas[{phase_name!r}]"
        )
    batch_quota = _positive_int(
        _required(data, "batch_quota", context), f"{context}.batch_quota"
    )
    if phase_quotas and sum(phase_quotas.values()) != batch_quota:
        raise ValueError(f"{context}.phase_quotas must sum to batch_quota={batch_quota}")
    minimum_phase_episodes_value = data.get("min_episodes_per_phase")
    if phase_quotas:
        minimum_phase_episodes = _positive_int(
            _required(data, "min_episodes_per_phase", context),
            f"{context}.min_episodes_per_phase",
        )
    else:
        if minimum_phase_episodes_value is not None:
            raise ValueError(
                f"{context}.min_episodes_per_phase requires phase_quotas"
            )
        minimum_phase_episodes = None
    catalog = data.get("sample_catalog")
    catalog_sha256_value = data.get("sample_catalog_sha256")
    if phase_quotas:
        if catalog is None:
            raise ValueError(f"{context}.phase_quotas requires sample_catalog")
        catalog_sha256 = _parse_sha256(
            _required(data, "sample_catalog_sha256", context),
            f"{context}.sample_catalog_sha256",
        )
    else:
        if catalog is not None:
            raise ValueError(f"{context}.sample_catalog requires phase_quotas")
        if catalog_sha256_value is not None:
            raise ValueError(
                f"{context}.sample_catalog_sha256 requires phase_quotas and sample_catalog"
            )
        catalog_sha256 = None
    return SourceSpec(
        name=_nonempty_string(_required(data, "name", context), f"{context}.name"),
        domain=_nonempty_string(_required(data, "domain", context), f"{context}.domain"),
        episode_list=_resolve_path(
            _required(data, "episode_list", context), base_dir, f"{context}.episode_list"
        ),
        expected_episode_count=_positive_int(
            _required(data, "expected_episode_count", context),
            f"{context}.expected_episode_count",
        ),
        batch_quota=batch_quota,
        phase_quotas=MappingProxyType(dict(sorted(phase_quotas.items()))),
        min_episodes_per_phase=minimum_phase_episodes,
        sample_catalog=(
            None
            if catalog is None
            else _resolve_path(catalog, base_dir, f"{context}.sample_catalog")
        ),
        sample_catalog_sha256=catalog_sha256,
    )


def _parse_validation_domain(
    value: Any, index: int, stage_context: str, base_dir: Path
) -> ValidationDomainSpec:
    context = f"{stage_context}.validation_domains[{index}]"
    data = _require_mapping(value, context)
    _reject_unknown(
        data,
        ("name", "domain", "episode_list", "expected_episode_count"),
        context,
    )
    name = _nonempty_string(_required(data, "name", context), f"{context}.name")
    return ValidationDomainSpec(
        name=name,
        domain=_nonempty_string(data.get("domain", name), f"{context}.domain"),
        episode_list=_resolve_path(
            _required(data, "episode_list", context), base_dir, f"{context}.episode_list"
        ),
        expected_episode_count=_positive_int(
            _required(data, "expected_episode_count", context),
            f"{context}.expected_episode_count",
        ),
    )


def _parse_monitor(value: Any, stage_context: str, validation_names: Sequence[str]) -> MonitorSpec:
    context = f"{stage_context}.monitor"
    data = _require_mapping(value, context)
    allowed = (
        "primary_domain",
        "metric",
        "aggregation",
        "retention_domain",
        "max_retention_regression",
        "patience",
        "min_validations",
        "min_delta",
    )
    _reject_unknown(data, allowed, context)
    primary = _nonempty_string(
        _required(data, "primary_domain", context), f"{context}.primary_domain"
    )
    if primary not in validation_names:
        raise ValueError(f"{context}.primary_domain {primary!r} is not a validation domain")
    retention = data.get("retention_domain")
    regression = data.get("max_retention_regression")
    if retention is None:
        if regression is not None:
            raise ValueError(
                f"{context}.max_retention_regression requires retention_domain"
            )
        parsed_retention = None
        parsed_regression = None
    else:
        parsed_retention = _nonempty_string(retention, f"{context}.retention_domain")
        if parsed_retention not in validation_names:
            raise ValueError(
                f"{context}.retention_domain {parsed_retention!r} is not a validation domain"
            )
        if parsed_retention == primary:
            raise ValueError(f"{context}.retention_domain must differ from primary_domain")
        parsed_regression = _finite_float(
            _required(data, "max_retention_regression", context),
            f"{context}.max_retention_regression",
            minimum=0.0,
        )
    min_delta = _finite_float(data.get("min_delta", 0.005), f"{context}.min_delta", minimum=0.0)
    if min_delta >= 1.0:
        raise ValueError(f"{context}.min_delta must be < 1")
    metric = _nonempty_string(data.get("metric", "deploy_loss"), f"{context}.metric")
    if metric not in MONITOR_METRICS:
        raise ValueError(
            f"{context}.metric must be one of: {', '.join(MONITOR_METRICS)}"
        )
    aggregation = _nonempty_string(
        _required(data, "aggregation", context), f"{context}.aggregation"
    )
    if aggregation not in VALIDATION_AGGREGATIONS:
        raise ValueError(
            f"{context}.aggregation must be one of: "
            + ", ".join(VALIDATION_AGGREGATIONS)
        )
    return MonitorSpec(
        primary_domain=primary,
        metric=metric,
        aggregation=aggregation,
        retention_domain=parsed_retention,
        max_retention_regression=parsed_regression,
        patience=_positive_int(data.get("patience", 8), f"{context}.patience"),
        min_validations=_nonnegative_int(
            data.get("min_validations", 1), f"{context}.min_validations"
        ),
        min_delta=min_delta,
    )


def _parse_stage(value: Any, index: int, base_dir: Path) -> StageSpec:
    context = f"stages[{index}]"
    data = _require_mapping(value, context)
    allowed = (
        "name",
        "sources",
        "validation_domains",
        "freeze_vision_batch_norm",
        "batch_size",
        "samples_per_epoch",
        "max_steps",
        "validation_every_steps",
        "checkpoint_every_steps",
        "optimizer",
        "objective",
        "monitor",
    )
    _reject_unknown(data, allowed, context)
    sources_value = _required(data, "sources", context)
    if not isinstance(sources_value, list) or not sources_value:
        raise ValueError(f"{context}.sources must be a non-empty array")
    sources = tuple(
        _parse_source(item, source_index, context, base_dir)
        for source_index, item in enumerate(sources_value)
    )
    source_names = [source.name for source in sources]
    source_domains = [source.domain for source in sources]
    if len(set(source_names)) != len(source_names):
        raise ValueError(f"{context}.sources contains duplicate names")
    if len(set(source_domains)) != len(source_domains):
        raise ValueError(
            f"{context}.sources must use unique domains so quotas remain auditable"
        )
    validation_value = _required(data, "validation_domains", context)
    if not isinstance(validation_value, list) or not validation_value:
        raise ValueError(f"{context}.validation_domains must be a non-empty array")
    validation = tuple(
        _parse_validation_domain(item, val_index, context, base_dir)
        for val_index, item in enumerate(validation_value)
    )
    validation_names = [item.name for item in validation]
    if len(set(validation_names)) != len(validation_names):
        raise ValueError(f"{context}.validation_domains contains duplicate names")
    batch_size = _positive_int(_required(data, "batch_size", context), f"{context}.batch_size")
    if sum(source.batch_quota for source in sources) != batch_size:
        raise ValueError(f"{context} source batch_quotas must sum to batch_size={batch_size}")
    samples_per_epoch = _positive_int(
        _required(data, "samples_per_epoch", context), f"{context}.samples_per_epoch"
    )
    if samples_per_epoch % batch_size != 0:
        raise ValueError(f"{context}.samples_per_epoch must be divisible by batch_size")
    freeze_vision_batch_norm = _required(
        data, "freeze_vision_batch_norm", context
    )
    if not isinstance(freeze_vision_batch_norm, bool):
        raise ValueError(f"{context}.freeze_vision_batch_norm must be boolean")
    max_steps = _positive_int(
        _required(data, "max_steps", context), f"{context}.max_steps"
    )
    validation_every_steps = _positive_int(
        _required(data, "validation_every_steps", context),
        f"{context}.validation_every_steps",
    )
    checkpoint_every_steps = _positive_int(
        _required(data, "checkpoint_every_steps", context),
        f"{context}.checkpoint_every_steps",
    )
    if checkpoint_every_steps != validation_every_steps:
        raise ValueError(
            f"{context}.checkpoint_every_steps must equal validation_every_steps "
            "so every formally validated point has one immutable periodic candidate"
        )
    if max_steps % validation_every_steps != 0:
        raise ValueError(
            f"{context}.max_steps must be divisible by validation_every_steps so "
            "the final validation is also an immutable periodic candidate"
        )
    monitor = _parse_monitor(
        _required(data, "monitor", context), context, validation_names
    )
    scheduled_validations = (
        max_steps + validation_every_steps - 1
    ) // validation_every_steps
    if scheduled_validations < monitor.min_validations:
        raise ValueError(
            f"{context} schedules only {scheduled_validations} validations, below "
            f"monitor.min_validations={monitor.min_validations}"
        )
    return StageSpec(
        name=_nonempty_string(_required(data, "name", context), f"{context}.name"),
        sources=sources,
        validation_domains=validation,
        freeze_vision_batch_norm=freeze_vision_batch_norm,
        batch_size=batch_size,
        samples_per_epoch=samples_per_epoch,
        max_steps=max_steps,
        validation_every_steps=validation_every_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        optimizer=_parse_optimizer(_required(data, "optimizer", context), context),
        objective=_parse_objective(data.get("objective", {}), context),
        monitor=monitor,
    )


def load_protocol(path: Path) -> ResolvedProtocol:
    """Load and strictly validate one staged-training JSON protocol."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"protocol file does not exist: {source_path}")
    try:
        document = json.loads(
            source_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid protocol JSON at {source_path}: {error}") from error
    root = _require_mapping(document, "protocol")
    allowed = (
        "schema_version",
        "run_name",
        "seed",
        "deterministic",
        "model",
        "dataset",
        "dataset_manifest",
        "normalization",
        "stages",
        "test_episode_lists",
    )
    _reject_unknown(root, allowed, "protocol")
    schema_version = _required(root, "schema_version", "protocol")
    if schema_version != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported protocol schema_version={schema_version!r}; "
            f"expected {PROTOCOL_SCHEMA_VERSION}"
        )
    deterministic = root.get("deterministic", True)
    if not isinstance(deterministic, bool):
        raise ValueError("protocol.deterministic must be boolean")
    stages_value = _required(root, "stages", "protocol")
    if not isinstance(stages_value, list) or not stages_value:
        raise ValueError("protocol.stages must be a non-empty array")
    base_dir = source_path.parent
    stages = tuple(
        _parse_stage(item, stage_index, base_dir)
        for stage_index, item in enumerate(stages_value)
    )
    stage_names = [stage.name for stage in stages]
    if len(set(stage_names)) != len(stage_names):
        raise ValueError("protocol.stages contains duplicate names")
    tests_value = root.get("test_episode_lists", {})
    if not isinstance(tests_value, Mapping):
        raise ValueError("protocol.test_episode_lists must be a JSON object")
    tests: Dict[str, TestEpisodeListSpec] = {}
    for name, test_value in tests_value.items():
        test_name = _nonempty_string(name, "protocol.test_episode_lists key")
        context = f"protocol.test_episode_lists[{test_name!r}]"
        test_data = _require_mapping(test_value, context)
        _reject_unknown(
            test_data,
            ("domain", "episode_list", "expected_episode_count"),
            context,
        )
        tests[test_name] = TestEpisodeListSpec(
            domain=_nonempty_string(
                _required(test_data, "domain", context), f"{context}.domain"
            ),
            episode_list=_resolve_path(
                _required(test_data, "episode_list", context),
                base_dir,
                f"{context}.episode_list",
            ),
            expected_episode_count=_positive_int(
                _required(test_data, "expected_episode_count", context),
                f"{context}.expected_episode_count",
            ),
        )
    model = _parse_model(_required(root, "model", "protocol"))
    dataset = _parse_dataset(_required(root, "dataset", "protocol"))
    # ContactForceHDF5Dataset has a fixed seven-joint action space and a
    # six-axis wrench stream. Reject a model/data shape mismatch at protocol
    # load time rather than during the first optimizer update.
    if model.action_dim != 7:
        raise ValueError("model.action_dim must be 7 for ContactForceHDF5Dataset")
    if model.force_dim != 6:
        raise ValueError("model.force_dim must be 6 for ContactForceHDF5Dataset")
    normalization = _parse_normalization(
        _required(root, "normalization", "protocol"), base_dir
    )
    source_domains = {
        source.domain for stage in stages for source in stage.sources
    }
    if set(normalization.domain_weights) != source_domains:
        raise ValueError(
            "normalization.domain_weights must exactly match all staged source "
            f"domains: expected={sorted(source_domains)} "
            f"actual={sorted(normalization.domain_weights)}"
        )
    raw_copy = json.loads(json.dumps(root, ensure_ascii=False, allow_nan=False))
    return ResolvedProtocol(
        source_path=source_path,
        schema_version=PROTOCOL_SCHEMA_VERSION,
        run_name=_nonempty_string(
            _required(root, "run_name", "protocol"), "protocol.run_name"
        ),
        seed=_nonnegative_int(root.get("seed", 0), "protocol.seed"),
        deterministic=deterministic,
        model=model,
        dataset=dataset,
        dataset_manifest=_parse_dataset_manifest(root.get("dataset_manifest"), base_dir),
        normalization=normalization,
        stages=stages,
        test_episode_lists=MappingProxyType(dict(sorted(tests.items()))),
        content_sha256=protocol_sha256(root),
        raw_document=MappingProxyType(raw_copy),
    )
