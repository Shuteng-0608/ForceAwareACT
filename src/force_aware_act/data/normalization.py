"""Normalization statistics utilities for ForceAwareACT datasets."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from force_aware_act.data.contact_force_hdf5_dataset import (
    ACTION_MODE_TO_DATASET,
    DELTA_ACTION_MODES,
    TIMESTAMP_FORCE_KEYS,
    TIMESTAMP_STATE_KEYS,
)
from force_aware_act.data.manifest import canonical_json_sha256, sha256_file


STAT_KEYS = (
    "qpos_mean",
    "qpos_std",
    "action_mean",
    "action_std",
    "force_mean",
    "force_std",
)

BALANCED_NORMALIZATION_IMPLEMENTATION_VERSION = 1
BALANCED_NORMALIZATION_METHOD = "domain_episode_time_equal_raw_hdf5_v1"
_LEGACY_EPISODE_UUID_NAMESPACE = uuid.UUID("b739db7e-5dd7-4fb8-8832-083078a7ec37")
_ACTION_DIM = 7
_FORCE_DIM = 6

ACTION_ALIGNMENT_DESCRIPTIONS = {
    "joint_pos": "observations/joint_pos[1:N]; legacy next-state offset=1",
    "action": "action[0:N]; command aligned to current state timestamp",
    "joint_pos_command": (
        "actions/joint_pos_command[0:N]; command aligned to current state timestamp"
    ),
    "delta_joint_cmd": "action[t] - observations/joint_pos[t]; same-time pairing",
    "delta_joint_pos_command": (
        "actions/joint_pos_command[t] - observations/joint_pos[t]; same-time pairing"
    ),
}


def validate_normalization_provenance_hashes(
    stats: Mapping[str, Any],
    *,
    require_components: bool,
) -> Dict[str, str]:
    """Verify that normalization provenance objects match their recorded hashes."""

    component_specs = (
        ("normalization_config", "normalization_config_sha256", Mapping),
        ("population_identities", "population_sha256", (list, tuple)),
    )
    verified: Dict[str, str] = {}
    for value_key, digest_key, expected_type in component_specs:
        value = stats.get(value_key)
        recorded = stats.get(digest_key)
        if value is None or recorded is None:
            if require_components:
                raise ValueError(
                    "normalization stats must record both "
                    f"{value_key} and {digest_key}"
                )
            continue
        if not isinstance(value, expected_type):
            raise ValueError(f"normalization {value_key} has an invalid type")
        if (
            not isinstance(recorded, str)
            or len(recorded) != 64
            or any(character not in "0123456789abcdef" for character in recorded)
        ):
            raise ValueError(f"normalization {digest_key} must be a SHA256 hex digest")
        actual = canonical_json_sha256(value)
        if actual != recorded:
            raise ValueError(
                f"normalization {value_key} SHA256 mismatch: "
                f"recorded={recorded} actual={actual}"
            )
        verified[digest_key] = actual
    return verified


def validate_balanced_normalization_contract(
    stats: Mapping[str, Any],
    *,
    expected_action_mode: str,
    expected_domain_weights: Mapping[str, float],
    strict_lengths: bool,
) -> None:
    """Validate the exact domain→episode→time raw-statistics semantics."""

    if stats.get("normalization_estimator") != "balanced_raw":
        raise ValueError("formal training requires normalization_estimator='balanced_raw'")
    if stats.get("normalization_method") != BALANCED_NORMALIZATION_METHOD:
        raise ValueError("normalization_method is not the balanced raw implementation")
    if (
        stats.get("normalization_implementation_version")
        != BALANCED_NORMALIZATION_IMPLEMENTATION_VERSION
    ):
        raise ValueError("normalization implementation version mismatch")
    config = stats.get("normalization_config")
    if not isinstance(config, Mapping):
        raise ValueError("normalization_config must be a mapping")
    expected_config_fields = {
        "implementation_version": BALANCED_NORMALIZATION_IMPLEMENTATION_VERSION,
        "method": BALANCED_NORMALIZATION_METHOD,
        "weighting_hierarchy": ["domain", "episode", "time_point"],
        "action_mode": expected_action_mode,
        "action_dataset": ACTION_MODE_TO_DATASET[expected_action_mode],
        "action_alignment": ACTION_ALIGNMENT_DESCRIPTIONS[expected_action_mode],
        "action_offset": 1 if expected_action_mode == "joint_pos" else 0,
        "accumulation_dtype": "float64",
        "tolerate_length_mismatch": not strict_lengths,
        "max_length_mismatch": 0 if strict_lengths else 1,
    }
    mismatches = [
        f"{key}: stats={config.get(key)!r} expected={expected!r}"
        for key, expected in expected_config_fields.items()
        if config.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "normalization_config balanced-raw semantics mismatch: "
            + "; ".join(mismatches)
        )

    def parsed_weights(value: Any, context: str) -> Dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{context} must be a mapping")
        if set(value) != set(expected_domain_weights):
            raise ValueError(
                f"{context} domain keys mismatch: "
                f"stats={sorted(value)} expected={sorted(expected_domain_weights)}"
            )
        result = {}
        for name in expected_domain_weights:
            raw = value[name]
            if isinstance(raw, bool):
                raise ValueError(f"{context}[{name!r}] must be finite and positive")
            numeric = float(raw)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{context}[{name!r}] must be finite and positive")
            result[name] = numeric
        return result

    expected_weights = parsed_weights(expected_domain_weights, "expected_domain_weights")
    for context, value in (
        ("normalization_config.domain_weights", config.get("domain_weights")),
        ("normalization domain_weights", stats.get("domain_weights")),
    ):
        actual_weights = parsed_weights(value, context)
        for name, expected in expected_weights.items():
            if not math.isclose(
                actual_weights[name], expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{context}[{name!r}] mismatch: "
                    f"stats={actual_weights[name]} expected={expected}"
                )
    domain_paths = stats.get("domain_episode_paths")
    domain_counts = stats.get("domain_episode_counts")
    if not isinstance(domain_paths, Mapping) or set(domain_paths) != set(
        expected_domain_weights
    ):
        raise ValueError("balanced normalization domain_episode_paths is invalid")
    if not isinstance(domain_counts, Mapping) or set(domain_counts) != set(
        expected_domain_weights
    ):
        raise ValueError("balanced normalization domain_episode_counts is invalid")
    expected_identity_keys = set()
    for domain, raw_paths in domain_paths.items():
        if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
            raise ValueError(
                f"balanced normalization domain {domain!r} has no episode paths"
            )
        if domain_counts.get(domain) != len(raw_paths):
            raise ValueError(
                f"balanced normalization domain {domain!r} episode count mismatch"
            )
        for path in raw_paths:
            if not isinstance(path, str) or not path:
                raise ValueError("normalization episode paths must be non-empty strings")
            expected_identity_keys.add((domain, str(Path(path).expanduser().resolve())))
    population_identities = stats.get("population_identities")
    if not isinstance(population_identities, (list, tuple)):
        raise ValueError("balanced normalization population_identities is invalid")
    recorded_identity_keys = {
        (row.get("domain"), str(Path(row.get("path", "")).expanduser().resolve()))
        for row in population_identities
        if isinstance(row, Mapping)
    }
    if recorded_identity_keys != expected_identity_keys or len(
        population_identities
    ) != len(expected_identity_keys):
        raise ValueError(
            "balanced normalization domain paths disagree with population identities"
        )
    timepoint_counts = stats.get("episode_timepoint_counts")
    if not isinstance(timepoint_counts, (list, tuple)) or not timepoint_counts:
        raise ValueError("balanced normalization episode_timepoint_counts is invalid")
    recorded_timepoint_keys = set()
    for row in timepoint_counts:
        if not isinstance(row, Mapping):
            raise ValueError("normalization timepoint-count rows must be mappings")
        key = (
            row.get("domain"),
            str(Path(row.get("path", "")).expanduser().resolve()),
        )
        for stream in ("qpos", "action", "force"):
            value = row.get(stream)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"normalization timepoint count {stream!r} must be positive"
                )
        if key in recorded_timepoint_keys:
            raise ValueError("duplicate normalization timepoint-count episode")
        recorded_timepoint_keys.add(key)
    if recorded_timepoint_keys != expected_identity_keys:
        raise ValueError(
            "balanced normalization timepoint counts disagree with domain paths"
        )


class RunningStats:
    """Accumulate feature-wise moments with float64 Chan/Welford updates.

    The legacy batch-based public API still returns float32 tensors, but its
    internal accumulation no longer relies on cancellation-prone ``sum_sq -
    mean**2`` arithmetic.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().to(device="cpu", dtype=torch.float64)
        values = values.reshape(-1, values.shape[-1])
        if not torch.isfinite(values).all():
            raise ValueError("normalization values must all be finite")
        batch_count = int(values.shape[0])
        batch_mean = values.mean(dim=0)
        batch_m2 = (values - batch_mean).square().sum(dim=0)
        if self.mean is None or self.m2 is None:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
        else:
            if self.mean.shape != batch_mean.shape:
                raise ValueError(
                    "normalization feature dimension changed between batches: "
                    f"expected={tuple(self.mean.shape)}; actual={tuple(batch_mean.shape)}"
                )
            combined_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * (batch_count / combined_count)
            self.m2 = (
                self.m2
                + batch_m2
                + delta.square() * (self.count * batch_count / combined_count)
            )
            self.count = combined_count

    def mean_std(self, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_eps(eps)
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("cannot compute statistics from zero samples")
        variance = torch.clamp(self.m2 / self.count, min=0.0)
        std = torch.sqrt(variance).clamp_min(eps)
        return self.mean.float(), std.float()


@dataclass(frozen=True)
class _DistributionMoments:
    mean: np.ndarray
    variance: np.ndarray
    count: int


class _Float64ArrayMoments:
    """Streaming population moments for two-dimensional NumPy chunks."""

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self.count = 0
        self.mean = np.zeros(feature_dim, dtype=np.float64)
        self.m2 = np.zeros(feature_dim, dtype=np.float64)

    def update(self, values: np.ndarray, description: str) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                f"{description} must have shape [N, {self.feature_dim}], got {values.shape}"
            )
        if values.shape[0] == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError(f"{description} contains NaN or Inf")
        batch_count = int(values.shape[0])
        batch_mean = values.mean(axis=0, dtype=np.float64)
        batch_m2 = np.square(values - batch_mean).sum(axis=0, dtype=np.float64)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / combined_count)
        self.m2 = (
            self.m2
            + batch_m2
            + np.square(delta) * (self.count * batch_count / combined_count)
        )
        self.count = combined_count

    def finish(self, description: str) -> _DistributionMoments:
        if self.count <= 0:
            raise ValueError(f"cannot compute {description} statistics from zero time points")
        return _DistributionMoments(
            mean=self.mean.copy(),
            variance=np.maximum(self.m2 / self.count, 0.0),
            count=self.count,
        )


def compute_normalization_stats_from_batches(
    batches: Iterable[Mapping[str, torch.Tensor]],
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Compute qpos/action/force normalization stats from dataset batches."""

    qpos_stats = RunningStats()
    action_stats = RunningStats()
    force_stats = RunningStats()

    for batch in batches:
        _require_batch_key(batch, "qpos")
        _require_batch_key(batch, "action_chunk")
        _require_batch_key(batch, "force_window")
        _require_batch_key(batch, "future_force_chunk")
        qpos_stats.update(batch["qpos"])
        action_stats.update(batch["action_chunk"])
        force_stats.update(batch["force_window"])
        force_stats.update(batch["future_force_chunk"])

    qpos_mean, qpos_std = qpos_stats.mean_std(eps)
    action_mean, action_std = action_stats.mean_std(eps)
    force_mean, force_std = force_stats.mean_std(eps)
    return {
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "force_mean": force_mean,
        "force_std": force_std,
    }


def compute_normalization_stats(
    dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Compute normalization statistics by iterating over a dataset."""

    if len(dataset) == 0:
        raise ValueError("cannot compute normalization statistics for an empty dataset")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return compute_normalization_stats_from_batches(dataloader, eps=eps)


def compute_balanced_normalization_stats(
    domain_episode_paths: Mapping[str, Sequence[Union[str, Path]]],
    *,
    action_mode: str = "joint_pos",
    domain_weights: Optional[Mapping[str, float]] = None,
    eps: float = 1.0e-6,
    tolerate_length_mismatch: bool = True,
    max_length_mismatch: int = 1,
    read_chunk_size: int = 65536,
    output_dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Compute hierarchy-balanced statistics directly from raw HDF5 streams.

    The estimator defines a mixture distribution independently for qpos,
    action, and force:

    1. every raw time point has equal weight within its episode;
    2. every episode has equal weight within its domain;
    3. domains use equal weight unless ``domain_weights`` is supplied.

    This deliberately avoids overlapping ACT chunks and resampled force
    windows, both of which multiply-count interior time points.  Delta action
    modes use same-time command/qpos pairs.  ``joint_pos`` retains the legacy
    next-state target convention and therefore starts at index one.

    Float64 Chan/Welford updates are used for stream and mixture moments.  The
    returned tensors remain float32 by default for checkpoint compatibility;
    the exact accumulation and output dtypes are recorded in provenance.
    """

    _validate_balanced_arguments(
        action_mode=action_mode,
        eps=eps,
        max_length_mismatch=max_length_mismatch,
        read_chunk_size=read_chunk_size,
        output_dtype=output_dtype,
    )
    domains = _canonicalize_domain_paths(domain_episode_paths)
    normalized_domain_weights = _normalize_domain_weights(domains, domain_weights)

    episode_moments: Dict[str, Dict[Path, Dict[str, _DistributionMoments]]] = {}
    population_identities = []
    episode_timepoint_counts = []
    seen_file_sha256: Dict[str, Path] = {}
    for domain_name, episode_paths in domains.items():
        episode_moments[domain_name] = {}
        for episode_path in episode_paths:
            moments = _compute_episode_raw_moments(
                episode_path,
                action_mode=action_mode,
                tolerate_length_mismatch=tolerate_length_mismatch,
                max_length_mismatch=max_length_mismatch,
                read_chunk_size=read_chunk_size,
            )
            episode_moments[domain_name][episode_path] = moments

            file_sha256 = sha256_file(episode_path)
            if file_sha256 in seen_file_sha256:
                raise ValueError(
                    "normalization population contains duplicate file content: "
                    f"first_path={seen_file_sha256[file_sha256]}; "
                    f"duplicate_path={episode_path}; sha256={file_sha256}"
                )
            seen_file_sha256[file_sha256] = episode_path
            derived_uuid = str(
                uuid.uuid5(
                    _LEGACY_EPISODE_UUID_NAMESPACE,
                    f"force-aware-act:episode-sha256:{file_sha256}",
                )
            )
            population_identities.append(
                {
                    "domain": domain_name,
                    "episode_uuid": derived_uuid,
                    "identity_scheme": "uuid5(file_sha256)",
                    "path": str(episode_path),
                    "file_sha256": file_sha256,
                }
            )
            episode_timepoint_counts.append(
                {
                    "domain": domain_name,
                    "path": str(episode_path),
                    "qpos": moments["qpos"].count,
                    "action": moments["action"].count,
                    "force": moments["force"].count,
                }
            )

    stats: Dict[str, Any] = {}
    stream_to_stat_names = {
        "qpos": ("qpos_mean", "qpos_std"),
        "action": ("action_mean", "action_std"),
        "force": ("force_mean", "force_std"),
    }
    for stream_name, (mean_name, std_name) in stream_to_stat_names.items():
        domain_distributions = []
        for domain_name, paths in domains.items():
            episode_weight = 1.0 / len(paths)
            domain_distribution = _mix_distributions(
                [
                    (episode_moments[domain_name][path][stream_name], episode_weight)
                    for path in paths
                ],
                description=f"{domain_name}/{stream_name}",
            )
            domain_distributions.append(
                (domain_distribution, normalized_domain_weights[domain_name])
            )
        population_distribution = _mix_distributions(
            domain_distributions,
            description=f"population/{stream_name}",
        )
        mean = torch.from_numpy(population_distribution.mean.copy()).to(dtype=output_dtype)
        std_values = np.maximum(
            np.sqrt(np.maximum(population_distribution.variance, 0.0)),
            eps,
        )
        std = torch.from_numpy(std_values).to(dtype=output_dtype)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError(
                f"{stream_name} statistics overflowed output dtype {output_dtype}"
            )
        stats[mean_name] = mean
        stats[std_name] = std

    action_offset = 1 if action_mode == "joint_pos" else 0
    normalization_config = {
        "implementation_version": BALANCED_NORMALIZATION_IMPLEMENTATION_VERSION,
        "method": BALANCED_NORMALIZATION_METHOD,
        "weighting_hierarchy": ["domain", "episode", "time_point"],
        "action_mode": action_mode,
        "action_dataset": ACTION_MODE_TO_DATASET[action_mode],
        "action_alignment": ACTION_ALIGNMENT_DESCRIPTIONS[action_mode],
        "action_offset": action_offset,
        "accumulation_dtype": "float64",
        "output_dtype": str(output_dtype),
        "eps": float(eps),
        "tolerate_length_mismatch": bool(tolerate_length_mismatch),
        "max_length_mismatch": int(max_length_mismatch),
        "read_chunk_size": int(read_chunk_size),
        "domain_weights": normalized_domain_weights,
    }
    population_paths = [
        str(path) for domain_name in domains for path in domains[domain_name]
    ]
    population_sha256 = canonical_json_sha256(population_identities)
    config_sha256 = canonical_json_sha256(normalization_config)
    stats.update(
        {
            "normalization_method": BALANCED_NORMALIZATION_METHOD,
            "normalization_implementation_version": (
                BALANCED_NORMALIZATION_IMPLEMENTATION_VERSION
            ),
            "normalization_config": normalization_config,
            "normalization_config_sha256": config_sha256,
            "population_paths": population_paths,
            "population_identities": population_identities,
            "population_sha256": population_sha256,
            "domain_episode_paths": {
                name: [str(path) for path in paths] for name, paths in domains.items()
            },
            "domain_weights": normalized_domain_weights,
            "domain_episode_counts": {
                name: len(paths) for name, paths in domains.items()
            },
            "episode_timepoint_counts": episode_timepoint_counts,
            "action_mode": action_mode,
            "action_alignment": ACTION_ALIGNMENT_DESCRIPTIONS[action_mode],
            "action_offset": action_offset,
            "accumulation_dtype": "float64",
        }
    )
    content_descriptor = {
        "normalization_config_sha256": config_sha256,
        "population_sha256": population_sha256,
        "statistics": {
            key: {
                "dtype": str(stats[key].dtype),
                "shape": list(stats[key].shape),
                "values": stats[key].detach().cpu().tolist(),
            }
            for key in STAT_KEYS
        },
    }
    stats["normalization_content_sha256"] = canonical_json_sha256(content_descriptor)
    return stats


def _validate_balanced_arguments(
    *,
    action_mode: str,
    eps: float,
    max_length_mismatch: int,
    read_chunk_size: int,
    output_dtype: torch.dtype,
) -> None:
    if action_mode not in ACTION_MODE_TO_DATASET:
        supported = ", ".join(sorted(ACTION_MODE_TO_DATASET))
        raise ValueError(f"unsupported action_mode={action_mode!r}; supported: {supported}")
    _validate_eps(eps)
    if (
        isinstance(max_length_mismatch, bool)
        or not isinstance(max_length_mismatch, int)
        or max_length_mismatch < 0
    ):
        raise ValueError("max_length_mismatch must be a non-negative integer")
    if (
        isinstance(read_chunk_size, bool)
        or not isinstance(read_chunk_size, int)
        or read_chunk_size <= 0
    ):
        raise ValueError("read_chunk_size must be a positive integer")
    if output_dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ValueError(f"output_dtype must be a floating torch dtype, got {output_dtype}")


def _canonicalize_domain_paths(
    domain_episode_paths: Mapping[str, Sequence[Union[str, Path]]],
) -> Dict[str, Tuple[Path, ...]]:
    if not isinstance(domain_episode_paths, Mapping) or not domain_episode_paths:
        raise ValueError("domain_episode_paths must be a non-empty mapping")
    if not all(isinstance(name, str) for name in domain_episode_paths):
        raise ValueError("domain names must be non-empty strings without surrounding whitespace")
    canonical: Dict[str, Tuple[Path, ...]] = {}
    seen_paths: Dict[Path, str] = {}
    for raw_name in sorted(domain_episode_paths):
        if not isinstance(raw_name, str) or not raw_name or raw_name != raw_name.strip():
            raise ValueError("domain names must be non-empty strings without surrounding whitespace")
        raw_paths = domain_episode_paths[raw_name]
        if isinstance(raw_paths, (str, Path)):
            raise TypeError(f"domain {raw_name!r} paths must be a sequence, not one path")
        resolved_paths = []
        for raw_path in raw_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"normalization episode does not exist: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"normalization episode is not a file: {path}")
            if path in seen_paths:
                raise ValueError(
                    "normalization episode path appears more than once: "
                    f"path={path}; first_domain={seen_paths[path]!r}; "
                    f"duplicate_domain={raw_name!r}"
                )
            seen_paths[path] = raw_name
            resolved_paths.append(path)
        if not resolved_paths:
            raise ValueError(f"normalization domain {raw_name!r} has no episodes")
        canonical[raw_name] = tuple(sorted(resolved_paths, key=str))
    return canonical


def _normalize_domain_weights(
    domains: Mapping[str, Sequence[Path]],
    domain_weights: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    domain_names = tuple(domains)
    if domain_weights is None:
        equal_weight = 1.0 / len(domain_names)
        return {name: equal_weight for name in domain_names}
    if not isinstance(domain_weights, Mapping):
        raise TypeError("domain_weights must be a mapping")
    missing = sorted(set(domain_names) - set(domain_weights))
    unexpected = sorted(set(domain_weights) - set(domain_names))
    if missing or unexpected:
        raise ValueError(
            "domain_weights keys must exactly match domains: "
            f"missing={missing}; unexpected={unexpected}"
        )
    parsed: Dict[str, float] = {}
    for name in domain_names:
        value = domain_weights[name]
        if isinstance(value, bool):
            raise ValueError(f"domain weight for {name!r} must be positive and finite")
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"domain weight for {name!r} must be positive and finite")
        parsed[name] = weight
    total = math.fsum(parsed.values())
    return {name: parsed[name] / total for name in domain_names}


def _compute_episode_raw_moments(
    episode_path: Path,
    *,
    action_mode: str,
    tolerate_length_mismatch: bool,
    max_length_mismatch: int,
    read_chunk_size: int,
) -> Dict[str, _DistributionMoments]:
    with h5py.File(episode_path, "r") as handle:
        state_timestamp_key = _first_existing_hdf5_key(
            handle,
            TIMESTAMP_STATE_KEYS,
            episode_path,
        )
        force_timestamp_key = _first_existing_hdf5_key(
            handle,
            TIMESTAMP_FORCE_KEYS,
            episode_path,
        )
        qpos = _require_matrix_dataset(
            handle,
            "observations/joint_pos",
            feature_dim=_ACTION_DIM,
            episode_path=episode_path,
        )
        state_timestamps = _require_vector_dataset(
            handle,
            state_timestamp_key,
            episode_path=episode_path,
        )
        state_len = _safe_stream_length(
            episode_path,
            "state",
            {
                "observations/joint_pos": len(qpos),
                state_timestamp_key: len(state_timestamps),
            },
            tolerate_length_mismatch=tolerate_length_mismatch,
            max_length_mismatch=max_length_mismatch,
        )

        force = _require_matrix_dataset(
            handle,
            "observations/ft_wrench",
            feature_dim=_FORCE_DIM,
            episode_path=episode_path,
        )
        force_timestamps = _require_vector_dataset(
            handle,
            force_timestamp_key,
            episode_path=episode_path,
        )
        force_len = _safe_stream_length(
            episode_path,
            "force",
            {
                "observations/ft_wrench": len(force),
                force_timestamp_key: len(force_timestamps),
            },
            tolerate_length_mismatch=tolerate_length_mismatch,
            max_length_mismatch=max_length_mismatch,
        )

        action_key = ACTION_MODE_TO_DATASET[action_mode]
        action_source = _require_matrix_dataset(
            handle,
            action_key,
            feature_dim=_ACTION_DIM,
            episode_path=episode_path,
        )
        action_len = _safe_stream_length(
            episode_path,
            f"action/{action_mode}",
            {action_key: len(action_source), "safe_state": state_len},
            tolerate_length_mismatch=tolerate_length_mismatch,
            max_length_mismatch=max_length_mismatch,
        )

        qpos_moments = _dataset_moments(
            qpos,
            start=0,
            stop=state_len,
            feature_dim=_ACTION_DIM,
            read_chunk_size=read_chunk_size,
            description=f"{episode_path}:observations/joint_pos",
        )
        if action_mode == "joint_pos":
            if action_len <= 1:
                raise ValueError(
                    f"{episode_path}: joint_pos action normalization requires at least two states"
                )
            action_moments = _dataset_moments(
                action_source,
                start=1,
                stop=action_len,
                feature_dim=_ACTION_DIM,
                read_chunk_size=read_chunk_size,
                description=f"{episode_path}:{action_key}[1:N]",
            )
        elif action_mode in DELTA_ACTION_MODES:
            action_moments = _dataset_moments(
                action_source,
                start=0,
                stop=action_len,
                feature_dim=_ACTION_DIM,
                read_chunk_size=read_chunk_size,
                description=f"{episode_path}:{action_key}-observations/joint_pos",
                subtract_dataset=qpos,
            )
        else:
            action_moments = _dataset_moments(
                action_source,
                start=0,
                stop=action_len,
                feature_dim=_ACTION_DIM,
                read_chunk_size=read_chunk_size,
                description=f"{episode_path}:{action_key}",
            )
        force_moments = _dataset_moments(
            force,
            start=0,
            stop=force_len,
            feature_dim=_FORCE_DIM,
            read_chunk_size=read_chunk_size,
            description=f"{episode_path}:observations/ft_wrench",
        )
    return {
        "qpos": qpos_moments,
        "action": action_moments,
        "force": force_moments,
    }


def _first_existing_hdf5_key(
    handle: h5py.File,
    candidates: Sequence[str],
    episode_path: Path,
) -> str:
    for key in candidates:
        if key in handle:
            return key
    raise KeyError(
        f"{episode_path}: missing required HDF5 dataset; tried: {', '.join(candidates)}"
    )


def _require_matrix_dataset(
    handle: h5py.File,
    key: str,
    *,
    feature_dim: int,
    episode_path: Path,
) -> h5py.Dataset:
    if key not in handle:
        raise KeyError(f"{episode_path}: missing required HDF5 dataset: {key}")
    dataset = handle[key]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{episode_path}: {key} is not an HDF5 dataset")
    if dataset.ndim != 2 or dataset.shape[1] != feature_dim:
        raise ValueError(
            f"{episode_path}: {key} must have shape [N, {feature_dim}], got {dataset.shape}"
        )
    return dataset


def _require_vector_dataset(
    handle: h5py.File,
    key: str,
    *,
    episode_path: Path,
) -> h5py.Dataset:
    if key not in handle:
        raise KeyError(f"{episode_path}: missing required HDF5 dataset: {key}")
    dataset = handle[key]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{episode_path}: {key} is not an HDF5 dataset")
    if dataset.ndim != 1:
        raise ValueError(f"{episode_path}: {key} must have shape [N], got {dataset.shape}")
    return dataset


def _safe_stream_length(
    episode_path: Path,
    stream_name: str,
    lengths: Mapping[str, int],
    *,
    tolerate_length_mismatch: bool,
    max_length_mismatch: int,
) -> int:
    minimum = min(lengths.values())
    maximum = max(lengths.values())
    allowed = max_length_mismatch if tolerate_length_mismatch else 0
    if maximum - minimum > allowed:
        details = ", ".join(f"{key}={value}" for key, value in lengths.items())
        raise ValueError(
            f"{episode_path}: {stream_name} length mismatch {maximum - minimum} "
            f"exceeds max_length_mismatch={allowed} ({details})"
        )
    if minimum <= 0:
        raise ValueError(f"{episode_path}: {stream_name} has no usable time points")
    return minimum


def _dataset_moments(
    dataset: h5py.Dataset,
    *,
    start: int,
    stop: int,
    feature_dim: int,
    read_chunk_size: int,
    description: str,
    subtract_dataset: Optional[h5py.Dataset] = None,
) -> _DistributionMoments:
    accumulator = _Float64ArrayMoments(feature_dim)
    for chunk_start in range(start, stop, read_chunk_size):
        chunk_stop = min(chunk_start + read_chunk_size, stop)
        values = np.asarray(dataset[chunk_start:chunk_stop], dtype=np.float64)
        if subtract_dataset is not None:
            values = values - np.asarray(
                subtract_dataset[chunk_start:chunk_stop],
                dtype=np.float64,
            )
        accumulator.update(values, description)
    return accumulator.finish(description)


def _mix_distributions(
    weighted_distributions: Sequence[Tuple[_DistributionMoments, float]],
    *,
    description: str,
) -> _DistributionMoments:
    if not weighted_distributions:
        raise ValueError(f"cannot mix an empty distribution collection: {description}")
    total_weight = 0.0
    mean = None
    weighted_m2 = None
    representative_count = 0
    for distribution, raw_weight in weighted_distributions:
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{description} mixture weights must be positive and finite")
        if mean is None or weighted_m2 is None:
            mean = distribution.mean.copy()
            weighted_m2 = distribution.variance * weight
            total_weight = weight
        else:
            if mean.shape != distribution.mean.shape:
                raise ValueError(f"{description} mixture feature dimensions do not match")
            combined_weight = total_weight + weight
            delta = distribution.mean - mean
            weighted_m2 = (
                weighted_m2
                + distribution.variance * weight
                + np.square(delta) * (total_weight * weight / combined_weight)
            )
            mean = mean + delta * (weight / combined_weight)
            total_weight = combined_weight
        representative_count += distribution.count
    assert mean is not None and weighted_m2 is not None
    return _DistributionMoments(
        mean=mean,
        variance=np.maximum(weighted_m2 / total_weight, 0.0),
        count=representative_count,
    )


def _validate_eps(eps: float) -> None:
    if isinstance(eps, bool):
        raise ValueError("eps must be positive and finite")
    try:
        parsed = float(eps)
    except (TypeError, ValueError) as error:
        raise ValueError("eps must be positive and finite") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("eps must be positive and finite")


def normalize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize tensor features along the last dimension."""

    return (x - _view_stats_for_tensor(mean, x)) / _view_stats_for_tensor(std, x)


def denormalize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert ``normalize_tensor`` for matching mean and std."""

    return x * _view_stats_for_tensor(std, x) + _view_stats_for_tensor(mean, x)


def _view_stats_for_tensor(stats: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if stats.ndim != 1:
        raise ValueError("mean/std must be 1D tensors")
    if x.shape[-1] != stats.shape[0]:
        raise ValueError(
            f"tensor last dimension {x.shape[-1]} does not match stats dimension {stats.shape[0]}"
        )
    return stats.to(device=x.device, dtype=x.dtype).view(*([1] * (x.ndim - 1)), -1)


def _require_batch_key(batch: Mapping[str, torch.Tensor], key: str) -> None:
    if key not in batch:
        raise KeyError(f"batch is missing required key: {key}")
    if not torch.is_tensor(batch[key]):
        raise ValueError(f"batch[{key!r}] must be a torch.Tensor")
