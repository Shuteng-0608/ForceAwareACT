import argparse
import csv
import json
from pathlib import Path

from scripts.run_paired50_q1_temporal_rollout_pilot import (
    PILOT_VERSION,
    build_pilot_specs,
    build_rollout_command,
    main,
    validate_completed_summary,
    write_aggregate,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path / "pilot",
        model_xml=tmp_path / "model.xml",
        device="cuda",
        max_rollout_steps=600,
        hole_offset_x=0.0,
        hole_offset_y=0.0,
        hole_offset_z=0.0,
        seed=0,
        save_videos=True,
    )


def test_pilot_matrix_is_paired_and_interleaved_by_executor(tmp_path):
    official = tmp_path / "official.pt"
    contact = tmp_path / "contact.pt"
    specs = build_pilot_specs(official, contact)

    assert [spec.configuration_id for spec in specs] == [
        "official_act__signed_km0p01",
        "highrate_contact_v3__signed_km0p01",
        "official_act__signed_k0p0",
        "highrate_contact_v3__signed_k0p0",
        "official_act__signed_kp0p01",
        "highrate_contact_v3__signed_kp0p01",
        "official_act__signed_kp0p03",
        "highrate_contact_v3__signed_kp0p03",
        "official_act__signed_kp0p05",
        "highrate_contact_v3__signed_kp0p05",
        "official_act__signed_kp0p1",
        "highrate_contact_v3__signed_kp0p1",
        "official_act__signed_kp0p2",
        "highrate_contact_v3__signed_kp0p2",
        "official_act__signed_kp0p3",
        "highrate_contact_v3__signed_kp0p3",
        "official_act__latest_only",
        "highrate_contact_v3__latest_only",
    ]
    assert {spec.model_id for spec in specs} == {
        "official_act",
        "highrate_contact_v3",
    }
    assert {spec.executor_id for spec in specs} == {
        "signed_km0p01",
        "signed_k0p0",
        "signed_kp0p01",
        "signed_kp0p03",
        "signed_kp0p05",
        "signed_kp0p1",
        "signed_kp0p2",
        "signed_kp0p3",
        "latest_only",
    }


def test_rollout_command_locks_fairness_contract(tmp_path):
    args = _args(tmp_path)
    spec = build_pilot_specs(tmp_path / "official.pt", tmp_path / "contact.pt")[0]
    command = build_rollout_command(args, spec)

    assert command[command.index("--action-select-mode") + 1] == "signed_temporal"
    assert command[command.index("--temporal-agg-decay") + 1] == "-0.01"
    assert command[command.index("--policy-rate-hz") + 1] == "30"
    assert command[command.index("--max-rollout-steps") + 1] == "600"
    assert command[command.index("--force-stop-threshold") + 1] == "100"
    assert command[command.index("--safe-force-threshold") + 1] == "40"
    assert command[command.index("--contact-latent-mode") + 1] == "zero"
    assert "--execute-actions" in command
    assert "--save-videos" in command


def test_latest_only_command_does_not_inject_irrelevant_decay(tmp_path):
    args = _args(tmp_path)
    spec = build_pilot_specs(tmp_path / "official.pt", tmp_path / "contact.pt")[-1]
    command = build_rollout_command(args, spec)

    assert command[command.index("--action-select-mode") + 1] == "latest_only"
    assert "--temporal-agg-decay" not in command


def test_completed_summary_contract_accepts_matching_q1_run(tmp_path):
    args = _args(tmp_path)
    args.model_xml.touch()
    spec = build_pilot_specs(tmp_path / "official.pt", tmp_path / "contact.pt")[0]
    spec.checkpoint.touch()
    summary = {
        "rollout_protocol_version": "paired_action_executor_rollout_v3",
        "checkpoint": str(spec.checkpoint.resolve()),
        "model_xml": str(args.model_xml.resolve()),
        "rollout_mode": "execute",
        "seed": 0,
        "action_mode": "joint_pos",
        "action_select_mode": "signed_temporal",
        "policy_query_interval": 1,
        "contact_latent_mode": "zero",
        "policy_rate_hz": 30.0,
        "max_rollout_steps": 600,
        "ema_alpha": 1.0,
        "max_delta_q": 0.02,
        "force_stop_threshold": 100.0,
        "safe_force_threshold": 40.0,
        "hole_offset_x": 0.0,
        "hole_offset_y": 0.0,
        "hole_offset_z": 0.0,
        "temporal_signed_decay_equivalent": -0.01,
        "temporal_endpoint": None,
    }

    assert validate_completed_summary(summary, args, spec) == []
    summary["policy_query_interval"] = 2
    assert any(
        "policy_query_interval" in error
        for error in validate_completed_summary(summary, args, spec)
    )


def test_plan_only_main_writes_eighteen_run_manifest_without_launching(tmp_path):
    official = tmp_path / "official.pt"
    contact = tmp_path / "contact.pt"
    model_xml = tmp_path / "model.xml"
    for path in (official, contact, model_xml):
        path.touch()
    output_dir = tmp_path / "pilot"

    result = main(
        [
            "--official-checkpoint",
            str(official),
            "--contact-checkpoint",
            str(contact),
            "--model-xml",
            str(model_xml),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    plan = json.loads((output_dir / "pilot_plan.json").read_text())
    assert plan["pilot_version"] == PILOT_VERSION
    assert plan["execution_enabled"] is False
    assert plan["fairness_contract"]["policy_query_interval"] == 1
    assert len(plan["specifications"]) == 18
    assert not list(output_dir.glob("*/summary.json"))


def test_aggregate_writer_has_one_row_per_completed_configuration(tmp_path):
    rows = [
        {
            "configuration_id": "official_act__latest_only",
            "model_id": "official_act",
            "executor_id": "latest_only",
            "signed_decay": None,
            "success": False,
            "safe_success": False,
            "stop_reason": "max_rollout_steps",
            "steps_executed": 600,
            "min_peg_to_hole_dist": 0.1,
            "final_peg_to_hole_dist": 0.2,
            "max_force_norm": 3.0,
            "policy_inference_time_ms_p95": 8.0,
            "policy_step_compute_time_ms_p95": 12.0,
            "policy_deadline_miss_fraction": 0.0,
        }
    ]
    tmp_path.mkdir(exist_ok=True)

    path = write_aggregate(tmp_path, rows)

    with path.open() as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert written[0]["configuration_id"] == "official_act__latest_only"
