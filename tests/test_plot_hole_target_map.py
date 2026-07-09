import csv

import pytest

from scripts.plot_hole_target_map import (
    compute_symmetric_plot_limit,
    create_target_figure,
    load_target_data,
    main as plot_target_main,
    normalize_formats,
    parse_success_series,
)


def _write_grid_summary(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = ["point_index", "hole_offset_x", "hole_offset_z", "success"]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def _rows(success_values=(True, False)):
    return [
        {"point_index": index, "hole_offset_x": x, "hole_offset_z": z, "success": success}
        for index, (x, z, success) in enumerate(
            zip((-0.002, 0.003), (0.001, -0.004), success_values),
            start=1,
        )
    ]


def _safe_success_rows():
    return [
        {
            "point_index": 1,
            "hole_offset_x": -0.002,
            "hole_offset_z": 0.001,
            "success": True,
            "safe_success": True,
        },
        {
            "point_index": 2,
            "hole_offset_x": 0.003,
            "hole_offset_z": -0.004,
            "success": True,
            "safe_success": False,
        },
        {
            "point_index": 3,
            "hole_offset_x": 0.001,
            "hole_offset_z": 0.002,
            "success": False,
            "safe_success": False,
        },
    ]


def test_mixed_success_failure_data_creates_png(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    output_dir = tmp_path / "plots"
    _write_grid_summary(csv_path, _rows())

    exit_code = plot_target_main(
        [
            "--grid-summary-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--formats",
            "png",
            "--dpi",
            "80",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "hole_target_map.png").is_file()


def test_mixed_success_failure_data_creates_pdf(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    output_dir = tmp_path / "plots"
    _write_grid_summary(csv_path, _rows())

    assert (
        plot_target_main(
            [
                "--grid-summary-csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                "--formats",
                "pdf",
            ]
        )
        == 0
    )
    assert (output_dir / "hole_target_map.pdf").is_file()


def test_multiple_requested_formats_are_created(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    output_dir = tmp_path / "plots"
    _write_grid_summary(csv_path, _rows())

    plot_target_main(
        [
            "--grid-summary-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--formats",
            "png",
            "pdf",
            "svg",
        ]
    )

    assert (output_dir / "hole_target_map.png").is_file()
    assert (output_dir / "hole_target_map.pdf").is_file()
    assert (output_dir / "hole_target_map.svg").is_file()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([True, False], [True, False]),
        ([1, 0], [True, False]),
        (["true", "false"], [True, False]),
        (["yes", "no"], [True, False]),
        (["success", "failure"], [True, False]),
    ],
)
def test_success_values_parse_correctly(values, expected):
    import pandas as pd

    parsed = parse_success_series(pd.Series(values))

    assert parsed.tolist() == expected


def test_unknown_success_strings_raise_clear_error():
    import pandas as pd

    with pytest.raises(ValueError, match="unknown value"):
        parse_success_series(pd.Series(["maybe"]))


def test_safe_success_data_is_loaded_and_counted(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    _write_grid_summary(
        csv_path,
        _safe_success_rows(),
        fieldnames=["point_index", "hole_offset_x", "hole_offset_z", "success", "safe_success"],
    )

    data = load_target_data(csv_path)

    assert data.successful_points == 2
    assert data.safe_successful_points == 1
    assert data.unsafe_successful_points == 1


def test_safe_success_categories_appear_in_legend(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    _write_grid_summary(
        csv_path,
        _safe_success_rows(),
        fieldnames=["point_index", "hole_offset_x", "hole_offset_z", "success", "safe_success"],
    )
    data = load_target_data(csv_path)

    figure = create_target_figure(
        data,
        title="",
        ring_step_mm=2.0,
        plot_limit_mm=6.0,
        marker_size=40.0,
        show_point_index=False,
        show_sampling_boundary=False,
    )
    try:
        labels = figure.axes[0].get_legend_handles_labels()[1]
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)

    assert "Failure (1)" in labels
    assert "Safe success (1)" in labels
    assert "Task success, not safe (1)" in labels


def test_safe_success_rate_appears_in_title(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    _write_grid_summary(
        csv_path,
        _safe_success_rows(),
        fieldnames=["point_index", "hole_offset_x", "hole_offset_z", "success", "safe_success"],
    )
    data = load_target_data(csv_path)

    figure = create_target_figure(
        data,
        title="Synthetic",
        ring_step_mm=2.0,
        plot_limit_mm=6.0,
        marker_size=40.0,
        show_point_index=False,
        show_sampling_boundary=False,
    )
    try:
        title = figure.axes[0].get_title()
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)

    assert title == "Synthetic\n1/3 safe successful — 33.3%"


def test_safe_success_cannot_be_true_for_task_failure(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    rows = [
        {
            "point_index": 1,
            "hole_offset_x": 0.0,
            "hole_offset_z": 0.0,
            "success": False,
            "safe_success": True,
        }
    ]
    _write_grid_summary(
        csv_path,
        rows,
        fieldnames=["point_index", "hole_offset_x", "hole_offset_z", "success", "safe_success"],
    )

    with pytest.raises(ValueError, match="task-failure"):
        load_target_data(csv_path)


def test_missing_safe_success_column_preserves_legacy_behavior(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    _write_grid_summary(csv_path, _rows())

    data = load_target_data(csv_path)

    assert data.safe_success is None
    assert data.safe_successful_points is None
    assert data.unsafe_successful_points is None


@pytest.mark.parametrize(
    ("fieldnames", "match"),
    [
        (["point_index", "hole_offset_z", "success"], "x-offset"),
        (["point_index", "hole_offset_x", "success"], "z-offset"),
        (["point_index", "hole_offset_x", "hole_offset_z"], "success column"),
    ],
)
def test_missing_required_columns_raise_clear_errors(tmp_path, fieldnames, match):
    csv_path = tmp_path / "grid_summary.csv"
    _write_grid_summary(
        csv_path,
        [{"point_index": 1, "hole_offset_x": 0.0, "hole_offset_z": 0.0, "success": True}],
        fieldnames=fieldnames,
    )

    with pytest.raises(ValueError, match=match):
        load_target_data(csv_path)


def test_empty_csv_raises_clear_error(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    csv_path.write_text("")

    with pytest.raises(ValueError, match="empty"):
        load_target_data(csv_path)


@pytest.mark.parametrize(
    ("x_value", "z_value", "match"),
    [
        ("inf", "0.0", "non-finite"),
        ("not-a-number", "0.0", "non-numeric"),
    ],
)
def test_bad_offsets_raise_clear_errors(tmp_path, x_value, z_value, match):
    csv_path = tmp_path / "grid_summary.csv"
    _write_grid_summary(
        csv_path,
        [{"point_index": 1, "hole_offset_x": x_value, "hole_offset_z": z_value, "success": True}],
    )

    with pytest.raises(ValueError, match=match):
        load_target_data(csv_path)


def test_non_finite_z_value_raises_clear_error(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    _write_grid_summary(
        csv_path,
        [{"point_index": 1, "hole_offset_x": "0.0", "hole_offset_z": "-inf", "success": True}],
    )

    with pytest.raises(ValueError, match="non-finite"):
        load_target_data(csv_path)


def test_ring_step_mm_must_be_positive():
    with pytest.raises(ValueError, match="ring-step"):
        compute_symmetric_plot_limit([0.0], [0.0], 0.0, None)


def test_max_radius_mm_must_be_positive():
    with pytest.raises(ValueError, match="max-radius"):
        compute_symmetric_plot_limit([0.0], [0.0], 2.0, 0.0)


def test_no_output_formats_raises_clear_error():
    with pytest.raises(ValueError, match="at least one"):
        normalize_formats([])


def test_unsupported_output_formats_raise_clear_error():
    with pytest.raises(ValueError, match="unsupported"):
        normalize_formats(["png", "jpg"])


def test_automatic_limits_are_symmetric_rounded_and_contain_points():
    x_mm = [-10.0, 1.0]
    z_mm = [10.0, -1.0]

    limit = compute_symmetric_plot_limit(x_mm, z_mm, 2.0, None)

    assert limit == pytest.approx(16.0)
    assert limit % 2.0 == pytest.approx(0.0)
    assert max(abs(value) for value in x_mm) <= limit
    assert max(abs(value) for value in z_mm) <= limit


def test_requested_limit_that_would_clip_points_raises():
    with pytest.raises(ValueError, match="clip"):
        compute_symmetric_plot_limit([4.0], [4.0], 2.0, 5.0)


def test_show_point_index_completes_successfully(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    output_dir = tmp_path / "plots"
    _write_grid_summary(csv_path, _rows())

    assert (
        plot_target_main(
            [
                "--grid-summary-csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                "--show-point-index",
                "--formats",
                "png",
            ]
        )
        == 0
    )


def test_show_sampling_boundary_completes_successfully(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    output_dir = tmp_path / "plots"
    _write_grid_summary(csv_path, _rows())

    assert (
        plot_target_main(
            [
                "--grid-summary-csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                "--show-sampling-boundary",
                "--formats",
                "png",
            ]
        )
        == 0
    )


def test_output_directory_is_created_automatically(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    output_dir = tmp_path / "new" / "plots"
    _write_grid_summary(csv_path, _rows())

    plot_target_main(
        [
            "--grid-summary-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--formats",
            "png",
        ]
    )

    assert output_dir.is_dir()


@pytest.mark.parametrize("success_values", [(True, True), (False, False)])
def test_all_success_or_all_failure_input_is_handled(tmp_path, success_values):
    csv_path = tmp_path / "grid_summary.csv"
    output_dir = tmp_path / "plots"
    _write_grid_summary(csv_path, _rows(success_values))

    plot_target_main(
        [
            "--grid-summary-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--formats",
            "png",
        ]
    )

    assert (output_dir / "hole_target_map.png").is_file()


def test_origin_only_synthetic_input_is_handled(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    output_dir = tmp_path / "plots"
    _write_grid_summary(
        csv_path,
        [{"point_index": 1, "hole_offset_x": 0.0, "hole_offset_z": 0.0, "success": "success"}],
    )

    data = load_target_data(csv_path)
    limit = compute_symmetric_plot_limit(data.x_mm, data.z_mm, 2.0, None)
    plot_target_main(
        [
            "--grid-summary-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--formats",
            "png",
        ]
    )

    assert limit == pytest.approx(2.0)
    assert (output_dir / "hole_target_map.png").is_file()


def test_duplicate_point_index_raises_clear_error(tmp_path):
    csv_path = tmp_path / "grid_summary.csv"
    _write_grid_summary(
        csv_path,
        [
            {"point_index": 1, "hole_offset_x": 0.0, "hole_offset_z": 0.0, "success": True},
            {"point_index": 1, "hole_offset_x": 0.001, "hole_offset_z": 0.0, "success": False},
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_target_data(csv_path)
