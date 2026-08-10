import json
from pathlib import Path

from scripts.monitor_paired_fibonacci_temporal_rollouts import (
    format_status,
    read_status,
)


def test_monitor_reads_completed_progress(tmp_path: Path) -> None:
    (tmp_path / "experiment_plan.json").write_text(
        json.dumps({"planned_runs": 10, "protocol_fingerprint": "abc123"})
    )
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "runner_pid": 999999999,
                "completed_rollouts": 10,
                "failed_rollouts": 0,
                "current_run_id": None,
                "average_seconds_per_new_rollout": 12.5,
                "eta_seconds": 0.0,
                "updated_at_utc": "2026-08-10T00:00:00+00:00",
            }
        )
    )

    status = read_status(tmp_path)
    rendered = format_status(status)

    assert status["status"] == "COMPLETED"
    assert status["percent"] == 100.0
    assert "10/10 (100.00%)" in rendered
    assert "abc123" in rendered
