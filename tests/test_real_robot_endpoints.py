import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.analyze_real_robot_endpoints import (
    condition_name, flange_fk, grid_stats, lag_speed, load_model,
    pose_summary, group_stats, latest_stable_window,
    causal_frame_index,
    continuous_intervals,
)


def poses(p, r=None):
    p = np.asarray(p)
    ts = np.tile(np.eye(4), (len(p), 1, 1))
    ts[:, :3, 3] = p
    if r is not None:
        ts[:, :3, :3] = r
    return ts


def test_condition_is_terminal_and_not_timestamp():
    assert condition_name("Tianji_Marvin_M6CCS_261003_191602_0_x17y07") == "x17y07"
    with pytest.raises(ValueError):
        condition_name("record_x17y07_incomplete")


def test_unvalidated_model_rejected(tmp_path):
    config = tmp_path / "model"
    config.write_text("1017,0,0,0")
    with pytest.raises(ValueError, match="validated"):
        load_model(config)


def test_fk_zero_pose_geometry_and_joint1_rotation():
    dh = np.zeros((8, 4))
    dh[0, 2], dh[2, 2], dh[4, 2], dh[3, 1], dh[7, 2] = 174.5, 287, 314, 18, 95
    q = np.zeros((2, 7))
    q[1, 0] = np.pi/2
    t = flange_fk(q, dh)
    # Straight arm: shoulder + upper arm + forearm + flange lengths on Z.
    np.testing.assert_allclose(t[0, :3, 3], [0, 0, 870.5], atol=1e-10)
    rz = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    np.testing.assert_allclose(t[1, :3, :3], rz @ t[0, :3, :3], atol=1e-10)
    np.testing.assert_allclose(t[1, :3, 3], rz @ t[0, :3, 3], atol=1e-10)
    np.testing.assert_allclose(t[:, :3, :3] @ t[:, :3, :3].transpose(0,2,1), np.tile(np.eye(3),(2,1,1)), atol=1e-10)
    with pytest.raises(ValueError):
        flange_fk(np.zeros((2, 6)), dh)


def test_rotation_mean_handles_pi_wrap():
    ts = poses([[1,2,3], [1,2,3]], Rotation.from_euler("z", [179, -179], degrees=True).as_matrix())
    s = pose_summary(ts)
    assert s["orientation_max_deg"] == pytest.approx(1)
    assert s["radius_max_mm"] == 0


def test_grid_fit_retains_25mm_scale_and_detects_displaced_point():
    rot = Rotation.from_euler("xyz", [30,40,20], degrees=True).as_matrix()
    groups = {}
    for x in range(13,18):
        for y in (7,8):
            p = rot @ np.array([(x-13)*25., (y-7)*25., 0.]) + [40,50,60]
            groups[f"x{x}y{y:02}"] = {"center_mm": p.tolist()}
    g = grid_stats(groups)
    assert g["x_step_mm"] == pytest.approx(25)
    assert g["y_step_mm"] == pytest.approx(25)
    assert g["xy_angle_deg"] == pytest.approx(90)
    assert g["rigid_25mm_rms_mm"] < 1e-10
    assert len(g["adjacent_edges"]) == 13
    groups["x17y08"]["center_mm"][0] += 5
    assert grid_stats(groups)["rigid_25mm_rms_mm"] > .5


def test_lag_speed_uses_elapsed_time_and_rejects_gap():
    times = np.r_[np.arange(0., .2, .01), np.arange(.3, .5, .01)]
    t = poses(np.column_stack([times*10, times*0, times*0]))
    speed, angular = lag_speed(times, t)
    np.testing.assert_allclose(speed[np.isfinite(speed)], 10.)
    np.testing.assert_allclose(angular[np.isfinite(angular)], 0.)
    assert not np.isfinite(speed[20:24]).any()


def test_equal_episode_weighting_not_sample_count():
    records=[]
    for x, n in [(0,100), (1,100), (2,100000)]:
        summary=pose_summary(poses([[x,0,0]]))
        summary.update(n=n, q_median_rad=[0]*7)
        records.append({"condition":"x13y07", "window":summary})
    g=group_stats(records,"window")["x13y07"]
    assert g["center_mm"][0] == 1
    assert g["max_pairwise_mm"] == 2
    assert max(g["leave_one_out_errors_mm"]) == 1.5


def test_stability_search_rejects_moving_tail_and_finds_earlier_hold():
    t=np.arange(-5., .0001, 1/300)
    p=np.column_stack([np.maximum(t+.3,0)*100, np.zeros((len(t),2))])
    matrices=poses(p)
    e={"t":t,"tf":t,"q":np.zeros((len(t),7)),"dq":np.zeros((len(t),7)),"tau":np.zeros((len(t),7))}
    speed, angular=lag_speed(t,matrices)
    stable=latest_stable_window(e,matrices,speed,angular)
    assert stable is not None
    assert stable["end_s_from_end"] <= -.299


def test_image_evidence_selects_past_frame_not_future_frame():
    t=np.array([100,200,300],dtype=np.int64)
    assert causal_frame_index(t,299)==1
    assert causal_frame_index(t,300)==2
    with pytest.raises(ValueError):
        causal_frame_index(t,99)


def test_dwell_does_not_bridge_false_samples_or_timestamp_gaps():
    times=np.array([0.,.01,.02,.03,.04,.05,.15,.16,.17])
    valid=np.array([True,True,False,True,True,True,True,True,True])
    runs=continuous_intervals(times,valid)
    assert [x["n"] for x in runs]==[2,3,3]
    assert [x["duration_s"] for x in runs]==pytest.approx([.01,.02,.02])
