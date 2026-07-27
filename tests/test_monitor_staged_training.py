import argparse
from dataclasses import replace
from pathlib import Path

from scripts import monitor_staged_training as monitor
from force_aware_act.training.protocol import load_protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "staged_visual_force_r60_r2_v2_long.json"
)


def test_csv_tail_returns_recent_complete_rows(tmp_path):
    path = tmp_path / "train_log.csv"
    path.write_text(
        "stage_step,loss_total\n"
        "1,1.0\n"
        "2,0.8\n"
        "3,0.6\n",
        encoding="utf-8",
    )

    assert monitor.csv_tail(path, 2) == [
        {"stage_step": "2", "loss_total": "0.8"},
        {"stage_step": "3", "loss_total": "0.6"},
    ]


def test_validation_summary_matches_minimum_step_patience_semantics():
    stage = load_protocol(PROTOCOL_PATH).stage("spatial_r60")
    stage = replace(
        stage,
        monitor=replace(stage.monitor, min_stage_steps=2, patience=2),
    )
    rows = [
        {
            "validation_index": "1",
            "stage_step": "1",
            "domain": "r60_spatial_val",
            "deploy_loss": "1.0",
            "selected": "True",
        },
        {
            "validation_index": "1",
            "stage_step": "1",
            "domain": "r2_contact_val",
            "deploy_loss": "2.0",
            "selected": "True",
        },
        {
            "validation_index": "2",
            "stage_step": "2",
            "domain": "r60_spatial_val",
            "deploy_loss": "1.1",
            "selected": "False",
        },
        {
            "validation_index": "2",
            "stage_step": "2",
            "domain": "r2_contact_val",
            "deploy_loss": "2.1",
            "selected": "False",
        },
        {
            "validation_index": "3",
            "stage_step": "3",
            "domain": "r60_spatial_val",
            "deploy_loss": "1.2",
            "selected": "False",
        },
    ]

    progress = monitor.summarize_validations(rows, stage)

    assert progress.count == 2
    assert progress.without_selection == 1
    assert progress.last_stage_step == 2
    assert progress.best_primary["stage_step"] == "1"
    assert progress.latest_primary["stage_step"] == "2"


def test_earliest_plateau_stop_respects_floor_and_existing_patience():
    assert monitor.earliest_plateau_stop_step(
        current_step=0,
        last_validation_step=0,
        min_stage_steps=20,
        validation_every_steps=1,
        patience=12,
        without_selection=0,
        max_steps=100,
    ) == 31
    assert monitor.earliest_plateau_stop_step(
        current_step=25,
        last_validation_step=25,
        min_stage_steps=20,
        validation_every_steps=5,
        patience=4,
        without_selection=2,
        max_steps=100,
    ) == 35


def test_render_before_training_is_read_only_and_informative(tmp_path):
    output = tmp_path / "not_started"
    args = argparse.Namespace(
        protocol=PROTOCOL_PATH,
        stage="spatial_r60",
        output_dir=output,
        pid=99999999,
        recent_window=20,
        eta_min_steps=10,
        stale_after=900.0,
        watch=False,
        interval=10.0,
        no_gpu=True,
    )

    rendered = monitor.render(args)

    assert "status: NOT STARTED" in rendered
    assert "minimum floor: 0/33460" in rendered
    assert "earliest plateau stop: 51863" in rendered
    assert not output.exists()
