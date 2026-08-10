import numpy as np
import pytest


pytest.importorskip("cv2")

from force_aware_act.visualization.force_feedback_overlay import (  # noqa: E402
    ForceFeedbackConfig,
    ForceFeedbackSmoother,
    compute_force_feedback,
    draw_force_feedback_overlay,
    make_force_feedback_hud,
    risk_band,
    trend_label,
)


def test_force_feedback_threshold_boundaries() -> None:
    config = ForceFeedbackConfig()

    assert risk_band(9.99, config)["label"] == "SAFE"
    assert risk_band(40.0, config)["label"] == "MED"
    assert risk_band(80.0, config)["label"] == "HIGH"
    assert risk_band(100.0, config)["label"] == "EXCESS"
    assert trend_label(5.0, config) == "RISING"
    assert trend_label(-5.0, config) == "FALLING"
    assert trend_label(0.0, config) == "STABLE"


def test_force_feedback_smoother_updates_once_per_sample() -> None:
    smoother = ForceFeedbackSmoother(alpha=0.25)

    np.testing.assert_allclose(smoother.update([4.0, 0.0, 0.0]), [4.0, 0.0, 0.0])
    np.testing.assert_allclose(smoother.update([0.0, 0.0, 0.0]), [3.0, 0.0, 0.0])


def test_compute_force_feedback_world_axial_lateral_and_camera_projection() -> None:
    config = ForceFeedbackConfig(
        insertion_axis_world=[0.0, -1.0, 0.0],
        force_guidance_basis_mode="camera_screen",
    )

    feedback = compute_force_feedback(
        force_sensor=np.array([3.0, 4.0, 0.0]),
        config=config,
        source_label="raw",
        R_ws=np.eye(3),
        trend="STABLE",
        torque_sensor=np.array([0.1, 0.2, 0.3]),
        guidance_right_world=np.array([1.0, 0.0, 0.0]),
        guidance_up_world=np.array([0.0, 0.0, 1.0]),
        guidance_basis_label="test_camera_screen",
    )

    assert feedback["force_norm"] == pytest.approx(5.0)
    assert feedback["axial"] == pytest.approx(-4.0)
    assert feedback["lateral"] == pytest.approx(3.0)
    np.testing.assert_allclose(feedback["force_world"], [3.0, 4.0, 0.0])
    np.testing.assert_allclose(feedback["lateral_uv"], [3.0, 0.0])
    np.testing.assert_allclose(feedback["torque_uv"], [0.1, 0.3])
    assert feedback["basis_label"] == "test_camera_screen"
    assert feedback["source_label"] == "raw"


def test_sensor_debug_correction_semantics_are_explicit() -> None:
    config = ForceFeedbackConfig(
        force_guidance_basis_mode="sensor_debug",
        force_guidance_vector_semantics="correction",
        force_guidance_correction_sign=-1.0,
    )

    feedback = compute_force_feedback(
        force_sensor=np.array([3.0, 4.0, 12.0]),
        torque_sensor=np.array([0.2, -0.3, 0.4]),
        config=config,
    )

    assert feedback["axial"] == pytest.approx(12.0)
    assert feedback["lateral"] == pytest.approx(5.0)
    np.testing.assert_allclose(feedback["lateral_uv"], [-3.0, -4.0])
    np.testing.assert_allclose(feedback["torque_uv"], [-0.2, 0.3])
    assert feedback["basis_label"] == "sensor_debug"


@pytest.mark.parametrize("ring_mode", [False, True])
def test_force_feedback_overlay_draws_without_changing_frame_contract(
    ring_mode: bool,
) -> None:
    config = ForceFeedbackConfig(
        enabled=True,
        enable_task_force_guidance_hud=ring_mode,
        task_force_guidance_mode="ring",
        force_guidance_hud_anchor="top_left",
        show_torque_ring=False,
    )
    feedback = compute_force_feedback(
        force_sensor=np.array([12.0, 25.0, 8.0]),
        torque_sensor=np.array([0.1, 0.2, 0.3]),
        config=config,
        source_label="comp",
        R_ws=np.eye(3),
        trend="RISING",
        guidance_right_world=np.array([1.0, 0.0, 0.0]),
        guidance_up_world=np.array([0.0, 0.0, 1.0]),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    output = draw_force_feedback_overlay(
        frame_bgr=frame,
        feedback=feedback,
        config=config,
        camera_name="cctv_cam",
    )

    assert output is frame
    assert output.shape == (480, 640, 3)
    assert output.dtype == np.uint8
    assert np.count_nonzero(output) > 0


def test_standalone_force_feedback_hud_has_stable_shape() -> None:
    config = ForceFeedbackConfig(enabled=True)
    feedback = compute_force_feedback(
        force_sensor=np.array([1.0, 2.0, 3.0]),
        config=config,
        source_label="raw",
    )

    hud = make_force_feedback_hud(feedback, config)

    assert hud.shape == (170, 340, 3)
    assert hud.dtype == np.uint8
    assert np.count_nonzero(hud) > 0
