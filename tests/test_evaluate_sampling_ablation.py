from pathlib import Path

import pytest

from scripts.evaluate_sampling_ablation import (
    POLICY_ORDER,
    _comparison_report,
    interaction_difference,
    paired_bootstrap_summary,
    parse_args,
)


def test_paired_bootstrap_is_deterministic_and_preserves_constant_difference():
    first = paired_bootstrap_summary([0.25, 0.25, 0.25], resamples=100, seed=7)
    second = paired_bootstrap_summary([0.25, 0.25, 0.25], resamples=100, seed=7)

    assert first == second
    assert first["mean_difference"] == pytest.approx(0.25)
    assert first["mean_difference_ci95"] == pytest.approx([0.25, 0.25])
    assert first["first_wins"] == 3
    assert first["second_wins"] == 0


def test_interaction_is_difference_of_contact_effects_across_sampling_modes():
    values = {
        "official_episodic": 1.0,
        "official_dense": 2.0,
        "contact_dense": 2.5,
        "contact_episodic": 0.75,
    }

    assert interaction_difference(values) == pytest.approx(0.75)


def test_comparison_report_contains_all_six_pairs_and_handles_missing_stratum():
    rows = []
    for episode_index in range(3):
        row = {"category": "clean_100"}
        for policy_index, policy_id in enumerate(POLICY_ORDER):
            row[f"{policy_id}_action_chunk_l1_physical"] = (
                episode_index + policy_index / 10
            )
        rows.append(row)
    global_means = {
        f"{policy_id}_action_chunk_l1_physical": float(index)
        for index, policy_id in enumerate(POLICY_ORDER)
    }

    report = _comparison_report(
        rows,
        global_means,
        bootstrap_resamples=100,
        bootstrap_seed=0,
    )

    assert len(report["all"]["pairwise"]) == 6
    assert len(report["clean_100"]["pairwise"]) == 6
    assert report["collision_100"] == {
        "episode_count": 0,
        "pairwise": {},
        "interaction": None,
    }


def _required_cli_paths(tmp_path: Path) -> list[str]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    arguments = [str(data_root), "--experiment-manifest", str(manifest)]
    for policy_id in POLICY_ORDER:
        checkpoint = tmp_path / f"{policy_id}.pt"
        checkpoint.write_bytes(b"checkpoint")
        arguments.extend(
            [f"--{policy_id.replace('_', '-')}-checkpoint", str(checkpoint)]
        )
    arguments.extend(["--output-dir", str(tmp_path / "output")])
    return arguments


def test_cli_requires_explicit_holdout_acknowledgement(tmp_path):
    arguments = _required_cli_paths(tmp_path)

    with pytest.raises(SystemExit):
        parse_args([*arguments, "--split", "holdout"])

    parsed = parse_args(
        [
            *arguments,
            "--split",
            "holdout",
            "--confirm-holdout-evaluation",
        ]
    )
    assert parsed.split == "holdout"
    assert parsed.confirm_holdout_evaluation is True


def test_cli_validation_does_not_accept_holdout_acknowledgement(tmp_path):
    arguments = _required_cli_paths(tmp_path)

    with pytest.raises(SystemExit):
        parse_args(
            [
                *arguments,
                "--split",
                "validation",
                "--confirm-holdout-evaluation",
            ]
        )
