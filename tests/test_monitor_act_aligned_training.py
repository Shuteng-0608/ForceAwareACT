import json
from datetime import datetime, timezone

import pytest

from scripts.monitor_act_aligned_training import (
    estimate_eta,
    read_jsonl_records,
)


def test_read_jsonl_records_ignores_partial_final_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first = {"record_type": "step", "global_step": 10}
    second = {"record_type": "step", "global_step": 20}
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n{\"partial\":",
        encoding="utf-8",
    )

    assert read_jsonl_records(path) == [first, second]


def test_estimate_eta_uses_process_elapsed_time():
    now = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)

    estimate = estimate_eta(
        current_step=1200,
        start_step=0,
        target_steps=24000,
        elapsed_seconds=600.0,
        now=now,
    )

    assert estimate is not None
    assert estimate.completed_steps == 1200
    assert estimate.steps_per_second == pytest.approx(2.0)
    assert estimate.seconds_per_step == pytest.approx(0.5)
    assert estimate.remaining_seconds == pytest.approx(11400.0)
    assert estimate.estimated_total_seconds == pytest.approx(12000.0)
    assert estimate.estimated_finish == datetime(
        2026,
        7,
        29,
        15,
        10,
        tzinfo=timezone.utc,
    )


def test_estimate_eta_supports_resumed_training_start_step():
    estimate = estimate_eta(
        current_step=3000,
        start_step=2000,
        target_steps=24000,
        elapsed_seconds=500.0,
        now=datetime(2026, 7, 29, tzinfo=timezone.utc),
    )

    assert estimate is not None
    assert estimate.completed_steps == 1000
    assert estimate.steps_per_second == pytest.approx(2.0)
