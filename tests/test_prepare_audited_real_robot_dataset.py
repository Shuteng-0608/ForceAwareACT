"""Generic audited preparation freezes complete recordings without training."""

import importlib.util
import json
from pathlib import Path

import pytest
pytest.importorskip("mcap_ros2.writer")


def _module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preparer():
    return _module("scripts/prepare_audited_real_robot_dataset.py", "audited_preparation_test")


def _rows():
    return [{"episode_id": f"recording_{g}_{i}", "group": f"point_{g}",
             "source_fingerprint": {"sha256": f"source_{g}_{i}", "bytes": 1},
             "status": "passed", "errors": [], "task_success": True}
            for g in range(10) for i in range(5)]


def test_ten_position_split_is_recording_balanced_and_order_independent():
    module = _preparer()
    rows = _rows()
    split = module.balanced_recording_split(rows)
    assert module.balanced_recording_split(list(reversed(rows))) == split
    sets = [set(split[f"{role}_episode_ids"]) for role in ("train", "validation", "test")]
    assert list(map(len, sets)) == [30, 10, 10]
    assert set.union(*sets) == {r["episode_id"] for r in rows}
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    for role,count in (("train",3),("validation",1),("test",1)):
        for group in set(split["condition_groups"].values()):
            assert sum(split["condition_groups"][x] == group for x in split[f"{role}_episode_ids"]) == count


@pytest.mark.parametrize("damage", ["duplicate_id", "duplicate_source", "incomplete_group"])
def test_incomplete_or_leaking_recordings_cannot_be_prepared(damage):
    rows = _rows()
    if damage == "duplicate_id":
        rows[1]["episode_id"] = rows[0]["episode_id"]
    elif damage == "duplicate_source":
        rows[1]["source_fingerprint"] = dict(rows[0]["source_fingerprint"])
    else:
        rows.pop()
    with pytest.raises(ValueError, match="duplicate|five accepted"):
        _preparer().balanced_recording_split(rows)


@pytest.mark.parametrize("damage", ["dataset_failed", "recording_failed", "success_unconfirmed"])
def test_failed_audit_or_unconfirmed_success_blocks_preparation(damage):
    audit = {"status": "passed", "episodes": _rows()}
    if damage == "dataset_failed":
        audit["status"] = "failed"
    elif damage == "recording_failed":
        audit["episodes"][0]["status"] = "failed"
    else:
        audit["episodes"][0]["task_success"] = False
    with pytest.raises(ValueError, match="passed|failed|success"):
        _preparer().require_passed_audit(audit)


def test_audited_conversion_statistics_and_data_batches_complete_without_training(tmp_path):
    helper = _module("tests/test_audit_mcap_right_arm_dataset.py", "audited_conversion_fixture")
    preparer = _preparer()
    audit_dir = tmp_path/"audit"
    episodes = []
    for index in range(5):
        auditor, original, row, config = helper._encoded_source(tmp_path/f"fixture{index}", nonce=str(index))
        source = original.with_name(f"recording_{index}_x13y08.mcap")
        source.write_bytes(original.read_bytes())
        result = auditor.audit_recording(source, row, audit_dir/"episodes"/source.stem, config.to_dict(),
            task_success=True, task_success_provenance="synthetic successful fixture")
        assert result["status"] == "passed", result["errors"]
        episodes.append(result)
    audit = {"status": "passed", "episodes": episodes, "output_dir": str(audit_dir),
             "source_root": str(tmp_path), "data_config": config.to_dict(),
             "data_contract": json.loads(json.dumps(config.contract_dict()))}
    auditor.write_json(audit_dir/"audit_manifest.json", audit)
    output, configs = tmp_path/"converted", tmp_path/"configs"
    manifest = preparer.main(["--audit-manifest", str(audit_dir/"audit_manifest.json"),
        "--output-root", str(output), "--dataset-config-dir", str(configs), "--workers", "2"])
    assert manifest["status"] == "ready"
    assert len(manifest["episodes"]) == 5
    assert all(r["audit_reader_indices_equal"] and r["verification"]["status"] == "passed" for r in manifest["episodes"])
    normalization = json.loads((configs/"normalization.json").read_text())
    assert len(normalization["training_episode_ids"]) == 3
    assert normalization["internal_torque_included"] is False
    loader = json.loads((output/"data_loader_validation.json").read_text())
    assert loader["windows_checked"] == 15
    assert loader["model_instantiated"] is False
    assert loader["optimizer_instantiated"] is False
    assert loader["training_started"] is False
    assert (configs/"dataset.json").is_file()
    assert (configs/"validation_indices_stride10.json").is_file()
    assert not (configs/"optimization.json").exists()
    assert not (configs/"queue.json").exists()
