"""The batch conversion binds reviewed files and publishes six-axis raw data only."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import h5py
import pytest

pytest.importorskip("mcap_ros2.writer")


def _load(relative, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    helper = _load("tests/test_real_robot_conversion.py", "reviewed_sensor_fixture")
    converter = helper._mcap(source/"trial.mcap", include_ft=True)
    review = {"status": "inspection_complete", "source_root": str(source), "episodes": [
        {"episode_id": "trial", "source_sha256": converter.sha256(source/"trial.mcap"),
         "task_success": True, "disposition": "successful_demonstration_candidate"},
        {"episode_id": "deleted_short", "source_sha256": "f"*64,
         "task_success": None, "disposition": "hold_for_manual_review"}]}
    return source, review


@pytest.mark.parametrize("damage", ["extra_file", "duplicate_identity", "unknown_success", "wrong_root"])
def test_review_inventory_must_match_current_successful_files(tmp_path, damage):
    source, review = _fixture(tmp_path)
    if damage == "extra_file":
        (source/"unexpected.mcap").write_bytes(b"unexpected")
    elif damage == "duplicate_identity":
        review["episodes"].append(dict(review["episodes"][0]))
    elif damage == "unknown_success":
        review["episodes"][0]["task_success"] = None
    else:
        review["source_root"] = str(tmp_path/"different")
    batch = _load("scripts/convert_reviewed_mcap_dataset.py", "reviewed_batch_validation")
    with pytest.raises(ValueError):
        batch.reviewed_sources(source, review, 1)


def test_batch_cli_converts_only_current_reviewed_files_and_keeps_raw_status(tmp_path):
    source, review = _fixture(tmp_path)
    review_path, output = tmp_path/"review.json", tmp_path/"converted"
    review_path.write_text(json.dumps(review))
    subprocess.run([sys.executable, str(Path(__file__).parents[1]/"scripts/convert_reviewed_mcap_dataset.py"),
        "--source-root", str(source), "--output-root", str(output), "--review-manifest", str(review_path),
        "--expected-episodes", "1", "--workers", "1", "--ft-gravity-compensated",
        "--ft-semantics-provenance", "synthetic user confirmation"], check=True, capture_output=True, text=True)
    manifest = json.loads((output/"conversion_manifest.json").read_text())
    assert manifest["status"] == "converted_verified"
    assert manifest["primary_force_topic"] == "/io_teleop/right/ft_sensor"
    assert not manifest["training_ready"] and not manifest["training_started"]
    assert manifest["selected_message_counts"]["ft_sensor"] == 2
    assert len(manifest["episodes"]) == 1
    row = manifest["episodes"][0]
    assert row["source_sha256"] == review["episodes"][0]["source_sha256"]
    assert len(row["output_sha256"]) == 64
    assert not list(output.rglob("*.partial"))
    with h5py.File(output/"trial/episode.hdf5", "r") as h:
        assert h["observations/ee_wrench"].shape == (2, 6)
        assert h.attrs["source_comparison"] == "passed: complete second read"
    assert not (output/"normalization.json").exists()
