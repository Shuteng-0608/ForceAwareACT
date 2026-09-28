"""Recording splits and fingerprints must remain independent of training code."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_1mm_gap_dataset.py"


def _preparer():
    spec = importlib.util.spec_from_file_location("prepare_1mm_gap_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    return [{"group": f"x{x}y08", "dataset_name": f"episode_x{x}_{i}"}
            for x in range(13, 17) for i in range(5)]


def test_balanced_split_is_order_independent_and_condition_stratified():
    module = _preparer()
    rows = _rows()
    split = module.balanced_split(rows)
    assert module.balanced_split(list(reversed(rows))) == split
    sets = [set(split[f"{role}_episode_ids"]) for role in ("train", "validation", "test")]
    assert list(map(len, sets)) == [12, 4, 4]
    assert set.union(*sets) == {row["dataset_name"] for row in rows}
    assert all(not left & right for i, left in enumerate(sets) for right in sets[i + 1:])
    groups = split["condition_groups"]
    for role, count in (("train", 3), ("validation", 1), ("test", 1)):
        ids = split[f"{role}_episode_ids"]
        for group in set(groups.values()):
            assert sum(groups[episode] == group for episode in ids) == count


@pytest.mark.parametrize("rows", [_rows()[:-1], [row for row in _rows() if row["group"] != "x13y08"]])
def test_split_rejects_incomplete_condition_groups(rows):
    with pytest.raises(ValueError, match="four condition groups with five episodes"):
        _preparer().balanced_split(rows)


def test_source_fingerprint_records_exact_content_and_size(tmp_path):
    path = tmp_path / "source.mcap"
    content = b"original recording\x00\xff"
    path.write_bytes(content)
    expected = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    assert _preparer().source_fingerprint(path) == expected
    path.write_bytes(content[:-1] + b"!")
    assert _preparer().source_fingerprint(path)["sha256"] != expected["sha256"]


def test_preparation_help_does_not_import_training_or_checkpoint_modules():
    code = """
import json, runpy, sys
sys.argv = [sys.argv[1], '--help']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    if exc.code:
        raise
print(json.dumps([name for name in sys.modules if name in (
    'force_aware_act.real_robot.training',
    'force_aware_act.real_robot.checkpoint')]))
"""
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPT)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
