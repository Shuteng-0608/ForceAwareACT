import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from force_aware_act.visualization.ft_wrench_utils import (  # noqa: E402
    body_ids,
    compensated_ft_wrench,
    ft_sensor_pose_world,
    ft_sensor_site_id,
    gravity_wrench_sensor_frame,
    raw_ft_wrench,
    sensor_vec,
)


MINIMAL_FT_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="tool" pos="0 0 0">
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


def _minimal_ft_model_and_data():
    model = mujoco.MjModel.from_xml_string(MINIMAL_FT_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_compensated_ft_wrench_modes() -> None:
    raw = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    gravity = np.array([0.5, 1.0, 1.5, 0.01, 0.02, 0.03])

    np.testing.assert_allclose(compensated_ft_wrench(raw, gravity, "none"), raw)
    np.testing.assert_allclose(
        compensated_ft_wrench(raw, gravity, "gravity"),
        raw - gravity,
    )
    assert compensated_ft_wrench(raw, gravity, "unsupported") is None
    assert compensated_ft_wrench(raw, None, "gravity") is None


def test_sensor_lookup_and_raw_wrench_contract() -> None:
    model, data = _minimal_ft_model_and_data()

    assert sensor_vec(model, data, "missing") is None
    np.testing.assert_allclose(sensor_vec(model, data, "peg_ft_force"), np.zeros(3))
    np.testing.assert_allclose(raw_ft_wrench(model, data), np.zeros(6))
    assert body_ids(model, ["tool", "missing"]) == [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
    ]


def test_sensor_pose_and_gravity_wrench_are_in_sensor_frame() -> None:
    model, data = _minimal_ft_model_and_data()
    site_id = ft_sensor_site_id(model)

    assert site_id == mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "ft_site",
    )
    position, rotation = ft_sensor_pose_world(data, site_id)
    np.testing.assert_allclose(position, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)

    gravity = gravity_wrench_sensor_frame(
        model=model,
        data=data,
        ft_site_id=site_id,
        tool_body_ids=body_ids(model, ["tool"]),
        gravity_world=[0.0, 0.0, -9.81],
        sensor_sign=-1.0,
    )

    np.testing.assert_allclose(gravity[:3], [0.0, 0.0, 9.81], atol=1e-9)
    np.testing.assert_allclose(gravity[3:], np.zeros(3), atol=1e-9)


def test_missing_sensor_pose_has_explicit_sentinel() -> None:
    _, data = _minimal_ft_model_and_data()

    position, rotation = ft_sensor_pose_world(data, -1)

    assert np.isnan(position).all()
    np.testing.assert_allclose(rotation, np.eye(3))
