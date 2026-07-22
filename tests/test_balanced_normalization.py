from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from force_aware_act.data.normalization import compute_balanced_normalization_stats
from scripts.compute_normalization_stats import (
    _resolve_domain_inputs,
    main as normalization_main,
    parse_args,
)


def _rows(values, width):
    values = np.asarray(values, dtype=np.float64)
    return np.repeat(values[:, None], width, axis=1)


def _write_raw_episode(
    path: Path,
    qpos_values,
    *,
    action_values=None,
    force_values=None,
) -> Path:
    qpos_values = np.asarray(qpos_values, dtype=np.float64)
    if action_values is None:
        action_values = qpos_values
    action_values = np.asarray(action_values, dtype=np.float64)
    if force_values is None:
        force_values = qpos_values
    force_values = np.asarray(force_values, dtype=np.float64)

    with h5py.File(path, "w") as handle:
        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset(
            "state_episode",
            data=np.arange(len(qpos_values), dtype=np.float64),
        )
        timestamps.create_dataset(
            "force_episode",
            data=np.arange(len(force_values), dtype=np.float64),
        )
        observations = handle.create_group("observations")
        observations.create_dataset("joint_pos", data=_rows(qpos_values, 7))
        observations.create_dataset("ft_wrench", data=_rows(force_values, 6))
        handle.create_dataset("action", data=_rows(action_values, 7))
        actions = handle.create_group("actions")
        actions.create_dataset("joint_pos_command", data=_rows(action_values, 7))
    return path.resolve()


def test_domain_episode_time_hierarchy_is_not_length_or_count_biased(tmp_path):
    short = _write_raw_episode(
        tmp_path / "r60_short.hdf5",
        np.zeros(2),
        force_values=np.zeros(3),
    )
    long = _write_raw_episode(
        tmp_path / "r60_long.hdf5",
        np.full(100, 10.0),
        force_values=np.full(500, 10.0),
    )
    r2 = _write_raw_episode(
        tmp_path / "r2.hdf5",
        np.full(7, 20.0),
        force_values=np.full(11, 20.0),
    )

    stats = compute_balanced_normalization_stats(
        {"r60": [short, long], "r2": [r2]},
        action_mode="action",
        read_chunk_size=13,
    )

    # R60 episode mixture mean=(0+10)/2=5. Domains are then equally weighted:
    # population mean=(5+20)/2=12.5, independent of all stream lengths.
    for key in ("qpos_mean", "action_mean", "force_mean"):
        torch.testing.assert_close(stats[key], torch.full_like(stats[key], 12.5))
    expected_std = np.sqrt(68.75)
    for key in ("qpos_std", "action_std", "force_std"):
        torch.testing.assert_close(
            stats[key],
            torch.full_like(stats[key], expected_std),
        )
    assert stats["domain_weights"] == {"r2": 0.5, "r60": 0.5}
    assert stats["domain_episode_counts"] == {"r2": 1, "r60": 2}
    assert [row["qpos"] for row in stats["episode_timepoint_counts"]] == [7, 100, 2]


def test_time_points_are_uniform_inside_each_episode(tmp_path):
    episode = _write_raw_episode(tmp_path / "episode.hdf5", [0.0, 2.0, 10.0])

    stats = compute_balanced_normalization_stats(
        {"one_domain": [episode]},
        action_mode="action",
    )

    expected_mean = 4.0
    expected_std = np.sqrt(56.0 / 3.0)
    torch.testing.assert_close(
        stats["qpos_mean"], torch.full((7,), expected_mean, dtype=torch.float32)
    )
    torch.testing.assert_close(
        stats["qpos_std"], torch.full((7,), expected_std, dtype=torch.float32)
    )


def test_explicit_domain_weights_are_normalized(tmp_path):
    first = _write_raw_episode(tmp_path / "first.hdf5", [0.0, 0.0])
    second = _write_raw_episode(tmp_path / "second.hdf5", [10.0, 10.0])

    stats = compute_balanced_normalization_stats(
        {"first": [first], "second": [second]},
        action_mode="action",
        domain_weights={"first": 1.0, "second": 3.0},
    )

    torch.testing.assert_close(stats["qpos_mean"], torch.full((7,), 7.5))
    torch.testing.assert_close(
        stats["qpos_std"], torch.full((7,), np.sqrt(18.75))
    )
    assert stats["domain_weights"] == {"first": 0.25, "second": 0.75}


def test_joint_pos_offset_and_delta_same_time_alignment(tmp_path):
    episode = _write_raw_episode(
        tmp_path / "episode.hdf5",
        [1.0, 2.0, 4.0],
        action_values=[2.0, 5.0, 8.0],
    )

    joint_pos_stats = compute_balanced_normalization_stats(
        {"domain": [episode]},
        action_mode="joint_pos",
    )
    delta_stats = compute_balanced_normalization_stats(
        {"domain": [episode]},
        action_mode="delta_joint_cmd",
    )

    torch.testing.assert_close(joint_pos_stats["action_mean"], torch.full((7,), 3.0))
    assert joint_pos_stats["action_offset"] == 1
    assert "offset=1" in joint_pos_stats["action_alignment"]
    torch.testing.assert_close(
        delta_stats["action_mean"],
        torch.full((7,), 8.0 / 3.0),
    )
    assert delta_stats["action_offset"] == 0
    assert "same-time pairing" in delta_stats["action_alignment"]


def test_float64_stream_accumulation_preserves_small_variance(tmp_path):
    base = 1.0e12
    episode = _write_raw_episode(
        tmp_path / "large_values.hdf5",
        [base, base + 1.0, base + 2.0],
    )

    stats = compute_balanced_normalization_stats(
        {"domain": [episode]},
        action_mode="action",
        output_dtype=torch.float64,
        read_chunk_size=1,
    )

    torch.testing.assert_close(
        stats["qpos_mean"],
        torch.full((7,), base + 1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        stats["qpos_std"],
        torch.full((7,), np.sqrt(2.0 / 3.0), dtype=torch.float64),
    )
    assert stats["accumulation_dtype"] == "float64"


def test_provenance_and_content_hash_are_canonical(tmp_path):
    r60 = _write_raw_episode(tmp_path / "r60.hdf5", [1.0, 2.0])
    r2 = _write_raw_episode(tmp_path / "r2.hdf5", [3.0, 4.0])

    first = compute_balanced_normalization_stats(
        {"r60": [r60], "r2": [r2]},
        action_mode="action",
    )
    second = compute_balanced_normalization_stats(
        {"r2": [r2], "r60": [r60]},
        action_mode="action",
    )

    assert first["normalization_config_sha256"] == second["normalization_config_sha256"]
    assert first["population_sha256"] == second["population_sha256"]
    assert first["normalization_content_sha256"] == second["normalization_content_sha256"]
    assert len(first["population_identities"]) == 2
    assert all(len(row["file_sha256"]) == 64 for row in first["population_identities"])
    assert all(row["identity_scheme"] == "uuid5(file_sha256)" for row in first["population_identities"])
    assert first["normalization_config"]["implementation_version"] == 1


def test_duplicate_episode_path_across_domains_is_rejected(tmp_path):
    episode = _write_raw_episode(tmp_path / "episode.hdf5", [1.0, 2.0])

    with pytest.raises(ValueError, match="appears more than once"):
        compute_balanced_normalization_stats(
            {"r60": [episode], "r2": [episode]},
            action_mode="action",
        )


def test_cli_repeated_domain_lists_save_balanced_provenance(tmp_path):
    r60 = _write_raw_episode(tmp_path / "r60.hdf5", [1.0, 2.0])
    r2 = _write_raw_episode(tmp_path / "r2.hdf5", [3.0, 4.0])
    r60_list = tmp_path / "r60.txt"
    r2_list = tmp_path / "r2.txt"
    r60_list.write_text(f"{r60}\n", encoding="utf-8")
    r2_list.write_text(f"{r2}\n", encoding="utf-8")
    output = tmp_path / "stats.pt"

    exit_code = normalization_main(
        [
            "--domain",
            f"r60={r60_list}",
            "--domain",
            f"r2={r2_list}",
            "--action-mode",
            "action",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    stats = torch.load(output, map_location="cpu")
    assert stats["normalization_estimator"] == "balanced_raw"
    assert stats["domain_weights"] == {"r2": 0.5, "r60": 0.5}
    assert stats["domain_episode_lists"] == {
        "r2": str(r2_list.resolve()),
        "r60": str(r60_list.resolve()),
    }
    assert stats["episode_paths"] == [str(r2), str(r60)]
    assert len(stats["normalization_content_sha256"]) == 64

    original_bytes = output.read_bytes()
    assert normalization_main(
        [
            "--domain",
            f"r60={r60_list}",
            "--domain",
            f"r2={r2_list}",
            "--action-mode",
            "action",
            "--output",
            str(output),
        ]
    ) == 1
    assert output.read_bytes() == original_bytes


def test_legacy_positional_cli_keeps_historical_estimator(tmp_path):
    episode = _write_raw_episode(tmp_path / "episode.hdf5", [1.0, 2.0])
    args = parse_args([str(episode)])

    assert _resolve_domain_inputs(args) is None
    assert args.estimator == "legacy_chunked"
