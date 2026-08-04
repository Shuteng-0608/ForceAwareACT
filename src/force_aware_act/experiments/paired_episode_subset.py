"""Build a distribution-audited paired episode subset for model comparison."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from force_aware_act.act_aligned_training.schema import (
    load_state_aligned_force,
)
from force_aware_act.act_aligned_training.split import (
    EpisodeRecord,
    discover_episodes,
)


PAIRED_EPISODE_SUBSET_VERSION = "paired_episode_subset_v1"
SELECTION_ALGORITHM = "chronological_strata_distribution_match_v1"
SELECTION_FEATURE_NAMES = (
    "num_steps",
    "duration_sim",
    "action_step_delta_mean",
    "action_range_norm",
    "force_norm_mean",
    "force_norm_p95",
    "force_norm_max",
    "torque_norm_p95",
    "contact_onset_fraction_5n",
    "contact_fraction_5n",
    "contact_fraction_20n",
)


@dataclass(frozen=True)
class EpisodeSelectionFeatures:
    """Compact trajectory/contact descriptors used only for subset selection."""

    episode_id: str
    num_steps: int
    duration_sim: float
    action_step_delta_mean: float
    action_range_norm: float
    force_norm_mean: float
    force_norm_p95: float
    force_norm_max: float
    torque_norm_p95: float
    contact_onset_fraction_5n: float
    contact_fraction_5n: float
    contact_fraction_20n: float

    def vector(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in SELECTION_FEATURE_NAMES],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedPairedEpisodeSubset:
    """Validated records and compact provenance for a fixed experiment."""

    path: Path
    dataset_fingerprint: str
    algorithm: str
    seed: int
    train_episodes: tuple[EpisodeRecord, ...]
    validation_episodes: tuple[EpisodeRecord, ...]
    holdout_episodes: tuple[EpisodeRecord, ...]

    def checkpoint_provenance(self) -> dict[str, Any]:
        return {
            "format_version": PAIRED_EPISODE_SUBSET_VERSION,
            "path": str(self.path),
            "dataset_fingerprint": self.dataset_fingerprint,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "train_episode_ids": [
                record.episode_id for record in self.train_episodes
            ],
            "validation_episode_ids": [
                record.episode_id for record in self.validation_episodes
            ],
            "holdout_episode_ids": [
                record.episode_id for record in self.holdout_episodes
            ],
        }


def extract_episode_selection_features(
    data_root: Path,
    episodes: Sequence[EpisodeRecord],
) -> tuple[EpisodeSelectionFeatures, ...]:
    """Read non-image trajectory statistics for every validated episode."""

    data_root = Path(data_root).resolve()
    features = []
    for record in episodes:
        metadata_path = record.resolve(data_root).parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "auto_stop_task_success":
            raise ValueError(
                f"episode {record.episode_id} is not a successful demonstration"
            )
        with h5py.File(record.resolve(data_root), "r") as handle:
            action = np.asarray(handle["action"][...], dtype=np.float64)
            wrench = np.asarray(
                load_state_aligned_force(handle),
                dtype=np.float64,
            )
        if action.shape != (record.num_steps, 7):
            raise ValueError(f"episode {record.episode_id} action shape mismatch")
        if wrench.shape != (record.num_steps, 6):
            raise ValueError(f"episode {record.episode_id} force shape mismatch")

        force_norm = np.linalg.norm(wrench[:, :3], axis=1)
        torque_norm = np.linalg.norm(wrench[:, 3:], axis=1)
        contact_indices = np.flatnonzero(force_norm >= 5.0)
        contact_onset = (
            float(contact_indices[0] / max(record.num_steps - 1, 1))
            if contact_indices.size
            else 1.0
        )
        action_step_delta_mean = (
            float(np.linalg.norm(np.diff(action, axis=0), axis=1).mean())
            if record.num_steps > 1
            else 0.0
        )
        item = EpisodeSelectionFeatures(
            episode_id=record.episode_id,
            num_steps=record.num_steps,
            duration_sim=float(metadata["duration_sim"]),
            action_step_delta_mean=action_step_delta_mean,
            action_range_norm=float(np.linalg.norm(np.ptp(action, axis=0))),
            force_norm_mean=float(force_norm.mean()),
            force_norm_p95=float(np.quantile(force_norm, 0.95)),
            force_norm_max=float(force_norm.max()),
            torque_norm_p95=float(np.quantile(torque_norm, 0.95)),
            contact_onset_fraction_5n=contact_onset,
            contact_fraction_5n=float((force_norm >= 5.0).mean()),
            contact_fraction_20n=float((force_norm >= 20.0).mean()),
        )
        if not np.isfinite(item.vector()).all():
            raise ValueError(
                f"episode {record.episode_id} has non-finite selection features"
            )
        features.append(item)
    return tuple(features)


def select_paired_episode_subset(
    episodes: Sequence[EpisodeRecord],
    features: Sequence[EpisodeSelectionFeatures],
    *,
    temporal_strata: int = 10,
    selected_per_stratum: int = 5,
    validation_per_stratum: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Select matched halves, then a representative validation subset.

    Episodes are ordered chronologically by their identifier. Within each
    stratum, exhaustive combination search chooses the selected half whose
    standardized feature moments best match the held-out half. A deterministic
    local search then assigns validation episodes while preserving one
    validation example per chronological stratum.
    """

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if temporal_strata <= 0 or selected_per_stratum <= 0:
        raise ValueError("strata and selected count must be positive")
    if validation_per_stratum <= 0:
        raise ValueError("validation count must be positive")
    if len(episodes) != len(features):
        raise ValueError("episodes and features must have the same length")
    if len(episodes) % temporal_strata != 0:
        raise ValueError("episode count must be divisible by temporal_strata")
    stratum_size = len(episodes) // temporal_strata
    if selected_per_stratum >= stratum_size:
        raise ValueError("selected_per_stratum must be smaller than stratum size")
    if validation_per_stratum >= selected_per_stratum:
        raise ValueError(
            "validation_per_stratum must be smaller than selected_per_stratum"
        )

    records_by_id = {record.episode_id: record for record in episodes}
    features_by_id = {item.episode_id: item for item in features}
    if len(records_by_id) != len(episodes) or len(features_by_id) != len(features):
        raise ValueError("episode identifiers must be unique")
    if set(records_by_id) != set(features_by_id):
        raise ValueError("episode and feature identifiers must match")

    ordered_ids = sorted(records_by_id)
    matrix = np.stack([features_by_id[item].vector() for item in ordered_ids])
    standardized, center, scale = _robust_standardize(matrix)
    standardized_by_id = {
        episode_id: standardized[index]
        for index, episode_id in enumerate(ordered_ids)
    }

    strata: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    holdout_ids: list[str] = []
    for stratum_index in range(temporal_strata):
        start = stratum_index * stratum_size
        episode_ids = ordered_ids[start : start + stratum_size]
        chosen = _best_matched_subset(
            episode_ids,
            standardized_by_id,
            selected_per_stratum,
            seed=seed,
            stratum_index=stratum_index,
        )
        held_out = sorted(set(episode_ids) - set(chosen))
        selected_ids.extend(chosen)
        holdout_ids.extend(held_out)
        strata.append(
            {
                "stratum_index": stratum_index,
                "all_episode_ids": episode_ids,
                "selected_episode_ids": chosen,
                "holdout_episode_ids": held_out,
            }
        )

    validation_ids = _select_validation_ids(
        strata,
        selected_ids,
        standardized_by_id,
        validation_per_stratum=validation_per_stratum,
        seed=seed,
    )
    train_ids = sorted(set(selected_ids) - set(validation_ids))
    selected_ids = sorted(selected_ids)
    validation_ids = sorted(validation_ids)
    holdout_ids = sorted(holdout_ids)
    for stratum in strata:
        validation_in_stratum = sorted(
            set(stratum["selected_episode_ids"]) & set(validation_ids)
        )
        stratum["validation_episode_ids"] = validation_in_stratum
        stratum["train_episode_ids"] = sorted(
            set(stratum["selected_episode_ids"]) - set(validation_in_stratum)
        )

    groups = {
        "all": ordered_ids,
        "selected": selected_ids,
        "train": train_ids,
        "validation": validation_ids,
        "holdout": holdout_ids,
    }
    selection = {
        "format_version": PAIRED_EPISODE_SUBSET_VERSION,
        "algorithm": SELECTION_ALGORITHM,
        "seed": seed,
        "feature_names": list(SELECTION_FEATURE_NAMES),
        "robust_center": center.tolist(),
        "robust_scale": scale.tolist(),
        "temporal_strata": temporal_strata,
        "stratum_size": stratum_size,
        "selected_per_stratum": selected_per_stratum,
        "validation_per_stratum": validation_per_stratum,
        "groups": groups,
        "strata": strata,
        "audit": _selection_audit(groups, features_by_id, standardized_by_id),
    }
    _validate_selection(selection, expected_ids=set(ordered_ids))
    return selection


def build_paired_episode_subset_manifest(
    data_root: Path,
    *,
    seed: int = 0,
    expected_episode_count: int = 100,
) -> dict[str, Any]:
    """Discover, select, and serialize the fixed paired50 experiment split."""

    data_root = Path(data_root)
    episodes = discover_episodes(data_root)
    if len(episodes) != expected_episode_count:
        raise ValueError(
            f"expected {expected_episode_count} episodes, found {len(episodes)}"
        )
    features = extract_episode_selection_features(data_root, episodes)
    selection = select_paired_episode_subset(episodes, features, seed=seed)
    records_by_id = {record.episode_id: record for record in episodes}
    groups = selection["groups"]
    manifest = {
        "format_version": PAIRED_EPISODE_SUBSET_VERSION,
        "data_root_hint": str(data_root),
        "dataset_fingerprint": _dataset_fingerprint(episodes),
        "selection": {
            key: value
            for key, value in selection.items()
            if key not in {"format_version", "groups"}
        },
        "all_episode_count": len(episodes),
        "selected_episode_count": len(groups["selected"]),
        "train_episode_count": len(groups["train"]),
        "validation_episode_count": len(groups["validation"]),
        "holdout_episode_count": len(groups["holdout"]),
        "train_episodes": _records(groups["train"], records_by_id),
        "validation_episodes": _records(
            groups["validation"],
            records_by_id,
        ),
        "holdout_episodes": _records(groups["holdout"], records_by_id),
        "selected_episodes": _records(groups["selected"], records_by_id),
        "episode_features": {
            item.episode_id: item.to_dict() for item in features
        },
    }
    validate_paired_episode_subset_manifest(
        manifest,
        data_root=data_root,
    )
    return manifest


def load_paired_episode_subset_manifest(
    path: Path,
    *,
    data_root: Path,
) -> LoadedPairedEpisodeSubset:
    """Load a paired50 manifest and bind it to the current dataset root."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"experiment manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_paired_episode_subset_manifest(
        manifest,
        data_root=data_root,
    )
    selection = manifest["selection"]
    return LoadedPairedEpisodeSubset(
        path=path,
        dataset_fingerprint=str(manifest["dataset_fingerprint"]),
        algorithm=str(selection["algorithm"]),
        seed=int(selection["seed"]),
        train_episodes=_episode_records(manifest["train_episodes"]),
        validation_episodes=_episode_records(
            manifest["validation_episodes"]
        ),
        holdout_episodes=_episode_records(manifest["holdout_episodes"]),
    )


def validate_checkpoint_experiment_provenance(
    checkpoint_provenance: Mapping[str, Any] | None,
    subset: LoadedPairedEpisodeSubset,
) -> None:
    """Reject a resume request that supplies a different experiment split."""

    if checkpoint_provenance is None:
        raise ValueError(
            "checkpoint does not record an experiment manifest; resume "
            "without --experiment-manifest or use a matching experiment checkpoint"
        )
    expected = subset.checkpoint_provenance()
    for key in (
        "format_version",
        "dataset_fingerprint",
        "algorithm",
        "seed",
        "train_episode_ids",
        "validation_episode_ids",
        "holdout_episode_ids",
    ):
        if checkpoint_provenance.get(key) != expected[key]:
            raise ValueError(
                f"checkpoint experiment manifest mismatch for {key}"
            )


def write_paired_episode_subset_manifest(
    manifest: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Atomically write a validated, deterministic JSON manifest."""

    validate_paired_episode_subset_manifest(manifest)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)


def validate_paired_episode_subset_manifest(
    manifest: Mapping[str, Any],
    *,
    data_root: Path | None = None,
) -> None:
    """Validate disjointness, counts, records, and optional dataset identity."""

    if manifest.get("format_version") != PAIRED_EPISODE_SUBSET_VERSION:
        raise ValueError("unsupported paired episode subset format")
    expected_counts = {
        "selected_episodes": 50,
        "train_episodes": 40,
        "validation_episodes": 10,
        "holdout_episodes": 50,
    }
    id_sets = {}
    for key, expected_count in expected_counts.items():
        records = manifest.get(key)
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(f"{key} must contain {expected_count} records")
        identifiers = [record.get("episode_id") for record in records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{key} contains duplicate episode identifiers")
        id_sets[key] = set(identifiers)
    if id_sets["train_episodes"] & id_sets["validation_episodes"]:
        raise ValueError("train and validation episodes must be disjoint")
    if (
        id_sets["train_episodes"] | id_sets["validation_episodes"]
        != id_sets["selected_episodes"]
    ):
        raise ValueError("train and validation must partition selected episodes")
    if id_sets["selected_episodes"] & id_sets["holdout_episodes"]:
        raise ValueError("selected and holdout episodes must be disjoint")
    if len(id_sets["selected_episodes"] | id_sets["holdout_episodes"]) != 100:
        raise ValueError("selected and holdout must partition 100 episodes")

    strata = manifest.get("selection", {}).get("strata")
    if not isinstance(strata, list) or len(strata) != 10:
        raise ValueError("manifest must contain ten chronological strata")
    for stratum in strata:
        if len(stratum.get("all_episode_ids", [])) != 10:
            raise ValueError("each stratum must contain ten episodes")
        if len(stratum.get("selected_episode_ids", [])) != 5:
            raise ValueError("each stratum must select five episodes")
        if len(stratum.get("train_episode_ids", [])) != 4:
            raise ValueError("each stratum must contain four train episodes")
        if len(stratum.get("validation_episode_ids", [])) != 1:
            raise ValueError("each stratum must contain one validation episode")

    if data_root is not None:
        discovered = discover_episodes(data_root)
        if _dataset_fingerprint(discovered) != manifest.get(
            "dataset_fingerprint"
        ):
            raise ValueError("manifest dataset fingerprint mismatch")


def _best_matched_subset(
    episode_ids: Sequence[str],
    standardized_by_id: Mapping[str, np.ndarray],
    selected_count: int,
    *,
    seed: int,
    stratum_index: int,
) -> list[str]:
    best_score = math.inf
    best_tie = ""
    best: tuple[str, ...] | None = None
    for chosen in itertools.combinations(episode_ids, selected_count):
        held_out = tuple(sorted(set(episode_ids) - set(chosen)))
        selected_values = np.stack([standardized_by_id[item] for item in chosen])
        holdout_values = np.stack([standardized_by_id[item] for item in held_out])
        score = _moment_distance(selected_values, holdout_values)
        tie = _tie_key(seed, stratum_index, chosen)
        if score < best_score - 1.0e-12 or (
            abs(score - best_score) <= 1.0e-12 and tie < best_tie
        ):
            best_score = score
            best_tie = tie
            best = chosen
    if best is None:
        raise RuntimeError("could not select a matched stratum subset")
    return sorted(best)


def _select_validation_ids(
    strata: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    standardized_by_id: Mapping[str, np.ndarray],
    *,
    validation_per_stratum: int,
    seed: int,
) -> list[str]:
    if validation_per_stratum != 1:
        raise ValueError("v1 supports one validation episode per stratum")
    target = np.stack([standardized_by_id[item] for item in selected_ids])
    target_mean = target.mean(axis=0)
    validation = []
    for stratum_index, stratum in enumerate(strata):
        candidates = stratum["selected_episode_ids"]
        validation.append(
            min(
                candidates,
                key=lambda item: (
                    float(
                        np.square(
                            standardized_by_id[item] - target_mean
                        ).sum()
                    ),
                    _tie_key(seed, stratum_index, (item,)),
                ),
            )
        )

    def score(values: Sequence[str]) -> float:
        validation_values = np.stack(
            [standardized_by_id[item] for item in values]
        )
        return _moment_distance(validation_values, target)

    current_score = score(validation)
    while True:
        best_score = current_score
        best_tie = ""
        best_values: list[str] | None = None
        for stratum_index, stratum in enumerate(strata):
            for candidate in stratum["selected_episode_ids"]:
                proposal = list(validation)
                proposal[stratum_index] = candidate
                proposal_score = score(proposal)
                tie = _tie_key(seed, stratum_index, tuple(proposal))
                if proposal_score < best_score - 1.0e-12 or (
                    abs(proposal_score - best_score) <= 1.0e-12
                    and best_values is not None
                    and tie < best_tie
                ):
                    best_score = proposal_score
                    best_tie = tie
                    best_values = proposal
        if best_values is None:
            break
        validation = best_values
        current_score = best_score
    return sorted(validation)


def _robust_standardize(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    scale = np.quantile(values, 0.75, axis=0) - np.quantile(
        values,
        0.25,
        axis=0,
    )
    standard_deviation = values.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, standard_deviation)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (values - center) / scale, center, scale


def _moment_distance(left: np.ndarray, right: np.ndarray) -> float:
    mean_distance = np.square(left.mean(axis=0) - right.mean(axis=0)).mean()
    std_distance = np.square(left.std(axis=0) - right.std(axis=0)).mean()
    return float(mean_distance + 0.25 * std_distance)


def _selection_audit(
    groups: Mapping[str, Sequence[str]],
    features_by_id: Mapping[str, EpisodeSelectionFeatures],
    standardized_by_id: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    summaries = {}
    for group_name, identifiers in groups.items():
        raw = np.stack([features_by_id[item].vector() for item in identifiers])
        summaries[group_name] = {
            "episode_count": len(identifiers),
            "total_timesteps": int(
                sum(features_by_id[item].num_steps for item in identifiers)
            ),
            "features": {
                name: {
                    "min": float(values.min()),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "mean": float(values.mean()),
                    "q75": float(np.quantile(values, 0.75)),
                    "max": float(values.max()),
                    "std": float(values.std()),
                }
                for name, values in zip(SELECTION_FEATURE_NAMES, raw.T)
            },
        }

    def standardized_mean_distance(left: str, right: str) -> float:
        left_values = np.stack(
            [standardized_by_id[item] for item in groups[left]]
        )
        right_values = np.stack(
            [standardized_by_id[item] for item in groups[right]]
        )
        return float(
            np.sqrt(
                np.square(left_values.mean(axis=0) - right_values.mean(axis=0))
                .mean()
            )
        )

    return {
        "summaries": summaries,
        "standardized_mean_rms": {
            "selected_vs_all": standardized_mean_distance("selected", "all"),
            "holdout_vs_all": standardized_mean_distance("holdout", "all"),
            "train_vs_selected": standardized_mean_distance(
                "train",
                "selected",
            ),
            "validation_vs_selected": standardized_mean_distance(
                "validation",
                "selected",
            ),
        },
    }


def _validate_selection(
    selection: Mapping[str, Any],
    *,
    expected_ids: set[str],
) -> None:
    groups = selection["groups"]
    selected = set(groups["selected"])
    train = set(groups["train"])
    validation = set(groups["validation"])
    holdout = set(groups["holdout"])
    if train & validation or selected & holdout:
        raise RuntimeError("selection groups are not disjoint")
    if train | validation != selected:
        raise RuntimeError("train and validation do not partition selected")
    if selected | holdout != expected_ids:
        raise RuntimeError("selected and holdout do not partition all episodes")


def _records(
    identifiers: Sequence[str],
    records_by_id: Mapping[str, EpisodeRecord],
) -> list[dict[str, Any]]:
    return [asdict(records_by_id[item]) for item in sorted(identifiers)]


def _episode_records(values: Sequence[Mapping[str, Any]]) -> tuple[EpisodeRecord, ...]:
    return tuple(
        EpisodeRecord(
            episode_id=str(item["episode_id"]),
            relative_hdf5_path=str(item["relative_hdf5_path"]),
            num_steps=int(item["num_steps"]),
            camera_names=tuple(str(name) for name in item["camera_names"]),
        )
        for item in values
    )


def _dataset_fingerprint(episodes: Sequence[EpisodeRecord]) -> str:
    payload = [asdict(record) for record in sorted(episodes, key=lambda x: x.episode_id)]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _tie_key(seed: int, stratum_index: int, identifiers: Sequence[str]) -> str:
    text = f"{seed}:{stratum_index}:{','.join(identifiers)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
