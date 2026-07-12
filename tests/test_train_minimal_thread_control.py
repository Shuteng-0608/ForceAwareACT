import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.train_minimal import _config_from_args, build_checkpoint_payload, parse_args


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_thread_probe(arguments: list[str]) -> dict[str, int]:
    code = """
import json
import torch
from scripts.train_minimal import configure_cpu_threads

before = [torch.get_num_threads(), torch.get_num_interop_threads()]
intra = None if len(sys.argv) < 2 or sys.argv[1] == "none" else int(sys.argv[1])
interop = None if len(sys.argv) < 3 or sys.argv[2] == "none" else int(sys.argv[2])
after = configure_cpu_threads(intra, interop)
print(json.dumps({"before_intra": before[0], "before_interop": before[1], "after_intra": after[0], "after_interop": after[1]}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-c", "import sys\n" + code, *arguments],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _run_fingerprint_probe(intra_threads: int, interop_threads: int) -> str:
    code = """
import torch
from scripts.train_minimal import compute_initial_model_sha256, configure_cpu_threads, configure_reproducibility

configure_cpu_threads(int(sys.argv[1]), int(sys.argv[2]))
configure_reproducibility(17)
print(compute_initial_model_sha256(torch.nn.Linear(4, 3)))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n" + code,
            str(intra_threads),
            str(interop_threads),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_help_exposes_thread_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        parse_args(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--torch-num-threads" in help_text
    assert "--torch-num-interop-threads" in help_text


@pytest.mark.parametrize(
    "option",
    ["--torch-num-threads", "--torch-num-interop-threads"],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_thread_values_are_rejected(option, value):
    with pytest.raises(SystemExit) as error:
        parse_args([option, value])
    assert error.value.code == 2


def test_explicit_thread_values_are_applied_in_fresh_process():
    values = _run_thread_probe(["3", "2"])
    assert values["after_intra"] == 3
    assert values["after_interop"] == 2


def test_omitted_thread_values_preserve_defaults_in_fresh_process():
    values = _run_thread_probe(["none", "none"])
    assert values["after_intra"] == values["before_intra"]
    assert values["after_interop"] == values["before_interop"]


def test_thread_settings_do_not_change_seeded_initial_fingerprint():
    assert _run_fingerprint_probe(1, 1) == _run_fingerprint_probe(3, 2)


def test_checkpoint_payload_contains_resolved_thread_metadata():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    args = parse_args(["episode.hdf5"])
    args.resolved_torch_num_threads = 16
    args.resolved_torch_num_interop_threads = 2
    config = _config_from_args(args)

    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        torch_num_threads=16,
        torch_num_interop_threads=2,
    )

    assert payload["torch_num_threads"] == 16
    assert payload["torch_num_interop_threads"] == 2
    assert payload["config"]["torch_num_threads"] == 16
    assert payload["config"]["torch_num_interop_threads"] == 2
