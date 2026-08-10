import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("cv2")

from force_aware_act.visualization.force_feedback_overlay import (  # noqa: E402
    ForceFeedbackConfig,
)
from force_aware_act.visualization.rollout_force_hud import (  # noqa: E402
    RolloutForceHUDAdapter,
    RolloutForceHUDConfig,
    RolloutForceHUDIntervalPeakTracker,
    draw_rollout_force_hud_rgb,
)


MINIMAL_HUD_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <camera name="cctv_cam" pos="0 -1 0" xyaxes="1 0 0 0 0 1"/>
    <body name="peg_tool" pos="0 0 0">
      <freejoint/>
      <geom type="sphere" size="0.01" mass="1.0"/>
      <site name="ft_site" pos="0 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <force name="peg_ft_force" site="ft_site"/>
    <torque name="peg_ft_torque" site="ft_site"/>
  </sensor>
</mujoco>
"""


def _model_and_data():
    model = mujoco.MjModel.from_xml_string(MINIMAL_HUD_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _feedback_config(**overrides) -> ForceFeedbackConfig:
    values = {
        "enabled": True,
        "insertion_axis_world": [0.0, -1.0, 0.0],
        "force_guidance_basis_mode": "camera_screen",
        "force_guidance_screen_right_sign": 1.0,
        "force_guidance_screen_up_sign": 1.0,
        "smoothing_alpha": 1.0,
    }
    values.update(overrides)
    return ForceFeedbackConfig(**values)


def test_snapshot_contains_raw_compensated_and_cctv_projection() -> None:
    model, data = _model_and_data()
    adapter = RolloutForceHUDAdapter(
        model,
        _feedback_config(),
        RolloutForceHUDConfig(gravity_world=(0.0, 0.0, 0.0)),
    )
    raw = np.array([3.0, 4.0, 0.0, 0.1, 0.2, 0.3])

    snapshot = adapter.update(data, raw, timestamp=0.0)

    np.testing.assert_allclose(snapshot.raw_wrench, raw)
    np.testing.assert_allclose(snapshot.gravity_wrench, np.zeros(6), atol=1e-12)
    np.testing.assert_allclose(snapshot.compensated_wrench, raw)
    np.testing.assert_allclose(snapshot.primary_wrench, raw)
    np.testing.assert_allclose(snapshot.smoothed_primary_force, raw[:3])
    np.testing.assert_allclose(snapshot.guidance_right_world, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(snapshot.guidance_up_world, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(snapshot.feedback["lateral_uv"], [3.0, 0.0])
    np.testing.assert_allclose(snapshot.feedback["torque_uv"], [0.1, 0.3])
    assert snapshot.feedback["axial"] == pytest.approx(-4.0)
    assert snapshot.feedback["lateral"] == pytest.approx(3.0)
    assert snapshot.feedback["basis_label"] == "cctv_cam_screen"
    assert snapshot.primary_source_label == "comp"
    assert adapter.update_count == 1


def test_gravity_compensation_and_raw_primary_are_both_explicit() -> None:
    model, data = _model_and_data()
    adapter = RolloutForceHUDAdapter(
        model,
        _feedback_config(),
        RolloutForceHUDConfig(primary_wrench="raw"),
    )
    raw = np.array([3.0, 4.0, 0.0, 0.1, 0.2, 0.3])

    snapshot = adapter.update(data, raw, timestamp=0.0)

    np.testing.assert_allclose(snapshot.gravity_wrench[:3], [0.0, 0.0, 9.81])
    np.testing.assert_allclose(
        snapshot.compensated_wrench[:3],
        [3.0, 4.0, -9.81],
    )
    np.testing.assert_allclose(snapshot.primary_wrench, raw)
    assert snapshot.primary_source_label == "raw"
    assert snapshot.raw_force_norm == pytest.approx(5.0)
    assert snapshot.compensated_force_norm == pytest.approx(
        np.linalg.norm([3.0, 4.0, -9.81])
    )
    assert snapshot.feedback["force_norm"] == pytest.approx(5.0)


def test_smoothing_and_simulation_time_trend_update_once_per_policy_frame() -> None:
    model, data = _model_and_data()
    adapter = RolloutForceHUDAdapter(
        model,
        _feedback_config(
            smoothing_alpha=0.25,
            trend_window_sec=0.8,
            trend_rising_threshold=1.0,
            trend_falling_threshold=-1.0,
        ),
        RolloutForceHUDConfig(
            primary_wrench="raw",
            gravity_world=(0.0, 0.0, 0.0),
        ),
    )

    first = adapter.update(data, [4, 0, 0, 0, 0, 0], timestamp=0.0)
    second = adapter.update(data, [0, 0, 0, 0, 0, 0], timestamp=0.4)

    np.testing.assert_allclose(first.smoothed_primary_force, [4.0, 0.0, 0.0])
    np.testing.assert_allclose(second.smoothed_primary_force, [3.0, 0.0, 0.0])
    assert first.feedback["trend"] == "STABLE"
    assert second.feedback["trend"] == "FALLING"
    assert adapter.update_count == 2

    adapter.reset()
    reset = adapter.update(data, [0, 0, 0, 0, 0, 0], timestamp=0.0)
    np.testing.assert_allclose(reset.smoothed_primary_force, [0.0, 0.0, 0.0])
    assert reset.feedback["trend"] == "STABLE"
    assert adapter.update_count == 1


def test_timestamp_and_wrench_contract_fail_closed() -> None:
    model, data = _model_and_data()
    adapter = RolloutForceHUDAdapter(model, _feedback_config())
    adapter.update(data, np.zeros(6), timestamp=1.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        adapter.update(data, np.zeros(6), timestamp=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        adapter.update(data, np.zeros(6), timestamp=0.5)

    adapter.reset()
    with pytest.raises(ValueError, match="finite 6-vector"):
        adapter.update(data, np.zeros(5), timestamp=0.0)
    with pytest.raises(ValueError, match="finite 6-vector"):
        adapter.update(data, [0, 0, np.nan, 0, 0, 0], timestamp=0.0)
    with pytest.raises(ValueError, match="timestamp must be finite"):
        adapter.update(data, np.zeros(6), timestamp=np.nan)


def test_missing_camera_and_tool_body_fail_at_construction() -> None:
    model, _ = _model_and_data()

    with pytest.raises(ValueError, match="HUD camera is unavailable"):
        RolloutForceHUDAdapter(
            model,
            _feedback_config(),
            RolloutForceHUDConfig(camera_name="missing_camera"),
        )
    with pytest.raises(ValueError, match="gravity tool bodies are unavailable"):
        RolloutForceHUDAdapter(
            model,
            _feedback_config(),
            RolloutForceHUDConfig(gravity_tool_body_names=("missing_tool",)),
        )


def test_missing_torque_sensor_fails_at_construction() -> None:
    model, _ = _model_and_data()

    with pytest.raises(ValueError, match="torque sensor is unavailable"):
        RolloutForceHUDAdapter(
            model,
            _feedback_config(),
            RolloutForceHUDConfig(torque_sensor_name="missing_torque"),
        )


def test_failed_data_update_does_not_advance_episode_state() -> None:
    model, data = _model_and_data()
    adapter = RolloutForceHUDAdapter(model, _feedback_config())
    original_camera_rotation = data.cam_xmat[0].copy()
    data.cam_xmat[0] = np.nan

    with pytest.raises(ValueError, match="camera world rotation"):
        adapter.update(data, np.zeros(6), timestamp=0.0)
    assert adapter.update_count == 0

    data.cam_xmat[0] = original_camera_rotation
    snapshot = adapter.update(data, np.zeros(6), timestamp=0.0)
    assert snapshot.timestamp == 0.0
    assert adapter.update_count == 1


def test_interval_peak_tracks_direct_raw_and_compensated_samples() -> None:
    model, data = _model_and_data()
    adapter = RolloutForceHUDAdapter(
        model,
        _feedback_config(),
        RolloutForceHUDConfig(),
    )
    snapshot = adapter.update(data, [0, 0, 9.81, 0, 0, 0], timestamp=1.0)
    tracker = RolloutForceHUDIntervalPeakTracker(adapter, snapshot)

    tracker.observe(data, [0, 0, 0, 0, 0, 0], timestamp=1.002)
    tracker.observe(data, [0, 0, 4, 0, 0, 0], timestamp=1.004)
    peak = tracker.result()

    assert peak.start_timestamp == 1.0
    assert peak.end_timestamp == 1.004
    assert peak.sample_count == 3
    assert peak.raw_force_norm == pytest.approx(9.81)
    assert peak.raw_timestamp == pytest.approx(1.0)
    assert peak.compensated_force_norm == pytest.approx(9.81)
    assert peak.compensated_timestamp == pytest.approx(1.002)
    assert peak.primary_force_norm == pytest.approx(9.81)
    assert peak.primary_timestamp == pytest.approx(1.002)
    assert peak.primary_source_label == "comp"

    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.observe(data, np.zeros(6), timestamp=1.004)


def test_interval_peak_integrates_threshold_duration_and_exposure() -> None:
    model, data = _model_and_data()
    adapter = RolloutForceHUDAdapter(
        model,
        _feedback_config(),
        RolloutForceHUDConfig(gravity_world=(0.0, 0.0, 0.0)),
    )
    snapshot = adapter.update(data, [6, 0, 0, 0, 0, 0], timestamp=1.0)
    tracker = RolloutForceHUDIntervalPeakTracker(adapter, snapshot, threshold=5.0)

    tracker.observe(data, [4, 0, 0, 0, 0, 0], timestamp=1.002)
    tracker.observe(data, [7, 0, 0, 0, 0, 0], timestamp=1.005)
    tracker.observe(data, [0, 0, 0, 0, 0, 0], timestamp=1.006)
    result = tracker.result()

    # Piecewise-constant integration uses the previous physical observation
    # over each strictly increasing timestamp interval.
    assert result.raw_above_threshold_duration == pytest.approx(0.003)
    assert result.compensated_above_threshold_duration == pytest.approx(0.003)
    assert result.raw_excess_force_exposure == pytest.approx(0.004)
    assert result.compensated_excess_force_exposure == pytest.approx(0.004)
    with pytest.raises(ValueError, match="positive and finite"):
        RolloutForceHUDIntervalPeakTracker(adapter, snapshot, threshold=0.0)


def test_draw_rollout_force_hud_returns_rgb_without_mutating_input() -> None:
    model, data = _model_and_data()
    feedback_config = _feedback_config(enable_task_force_guidance_hud=True)
    adapter = RolloutForceHUDAdapter(
        model,
        feedback_config,
        RolloutForceHUDConfig(gravity_world=(0.0, 0.0, 0.0)),
    )
    snapshot = adapter.update(data, [3, 4, 0, 0, 0, 0], timestamp=0.0)
    tracker = RolloutForceHUDIntervalPeakTracker(adapter, snapshot)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    original = frame.copy()

    output = draw_rollout_force_hud_rgb(
        frame,
        snapshot,
        feedback_config,
        tracker.result(),
        output_width=400,
        output_height=300,
    )

    assert output.shape == (300, 400, 3)
    assert output.dtype == np.uint8
    assert output.flags.c_contiguous
    assert np.any(output != 0)
    np.testing.assert_array_equal(frame, original)

    with pytest.raises(ValueError, match="HxWx3 uint8"):
        draw_rollout_force_hud_rgb(
            frame.astype(np.float32),
            snapshot,
            feedback_config,
            tracker.result(),
        )


@pytest.mark.parametrize("field", ["primary_wrench", "compensation_mode"])
def test_invalid_rollout_force_hud_configuration_is_rejected(field: str) -> None:
    kwargs = {field: "invalid"}

    with pytest.raises(ValueError, match=field):
        RolloutForceHUDConfig(**kwargs)
