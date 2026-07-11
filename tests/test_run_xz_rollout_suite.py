import argparse

import pytest

from scripts.run_xz_rollout_suite import (
    MODEL_SPECS,
    build_grid_command,
    build_parser,
    output_dir_from_args,
    resolved_point_set_seed,
    resolved_rollout_seed_base,
    selected_models,
    target_map_limit_mm,
    validate_inputs,
)


def _args(*extra: str) -> argparse.Namespace:
    return build_parser().parse_args(list(extra))


def _model(key: str):
    return selected_models([key])[0]


def test_default_suite_includes_contact_zero_and_prior():
    args = _args()

    assert "contact_cvae" in args.models
    assert "contact_cvae_prior" in args.models
    assert _model("contact_cvae").contact_latent_mode == "zero"
    assert _model("contact_cvae_prior").contact_latent_mode == "prior"


def test_default_output_name_is_backward_compatible():
    args = _args()

    output = output_dir_from_args(args, _model("act_baseline"), "temporal")

    assert output.name == (
        "hole_lhs_50_xz_6mm_act_baseline100k_"
        "temporal_d03_dq002_maxsteps900"
    )


def test_output_name_reflects_cli_protocol_values():
    args = _args(
        "--num-points",
        "25",
        "--offset-mm",
        "4.5",
        "--max-rollout-steps",
        "600",
    )

    output = output_dir_from_args(args, _model("contact_cvae_prior"), "mid")

    assert output.name == (
        "hole_lhs_25_xz_4p5mm_contact_cvae100k_prior_"
        "mid_dq002_maxsteps600"
    )


def test_grid_command_forwards_protocol_and_prior_mode():
    args = _args(
        "--num-points",
        "25",
        "--offset-mm",
        "4",
        "--max-rollout-steps",
        "600",
    )

    command = build_grid_command(args, _model("contact_cvae_prior"), "mid")

    assert command[command.index("--num-points") + 1] == "25"
    assert command[command.index("--x-min") + 1] == "-0.004000"
    assert command[command.index("--x-max") + 1] == "0.004000"
    assert command[command.index("--max-rollout-steps") + 1] == "600"
    assert command[command.index("--contact-latent-mode") + 1] == "prior"
    assert command[command.index("--point-set-seed") + 1] == "20260702"
    assert command[command.index("--rollout-seed-base") + 1] == "20260702"
    assert "hole_lhs_25_xz_4mm_contact_cvae100k_prior" in command[
        command.index("--output-root") + 1
    ]


def test_grid_command_forwards_separated_seeds():
    args = _args(
        "--point-set-seed",
        "101",
        "--rollout-seed-base",
        "900",
    )

    command = build_grid_command(args, _model("motion_cvae"), "mid")

    assert resolved_point_set_seed(args) == 101
    assert resolved_rollout_seed_base(args) == 900
    assert command[command.index("--point-set-seed") + 1] == "101"
    assert command[command.index("--rollout-seed-base") + 1] == "900"


def test_default_target_map_limit_contains_square_sampling_range():
    args = _args()

    assert target_map_limit_mm(args) == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (("--num-points", "0"), "num-points"),
        (("--offset-mm", "0"), "offset-mm"),
        (("--max-rollout-steps", "0"), "max-rollout-steps"),
        (("--target-map-ring-step-mm", "0"), "ring-step"),
        (("--target-map-max-radius-mm", "8"), "full square"),
        (("--target-map-formats", "jpg"), "unsupported"),
    ],
)
def test_invalid_protocol_values_are_rejected(flags, message, tmp_path):
    args = _args(*flags)
    args.normalization_stats = tmp_path / "stats.pt"
    args.model_xml = tmp_path / "model.xml"

    with pytest.raises(ValueError, match=message):
        validate_inputs(args, MODEL_SPECS)


def test_validate_inputs_accepts_existing_artifacts(tmp_path):
    args = _args("--models", "contact_cvae_prior")
    checkpoint = tmp_path / "checkpoint.pt"
    stats = tmp_path / "stats.pt"
    xml = tmp_path / "model.xml"
    for path in (checkpoint, stats, xml):
        path.touch()
    model = _model("contact_cvae_prior")
    model = type(model)(
        key=model.key,
        output_token=model.output_token,
        checkpoint=checkpoint,
        contact_latent_mode=model.contact_latent_mode,
    )
    args.normalization_stats = stats
    args.model_xml = xml

    validate_inputs(args, [model])
