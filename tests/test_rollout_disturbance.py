import math

import numpy as np
import pytest

from force_aware_act.rollout_disturbance import LateralPulse, MujocoPulse, PulseConfig
from scripts.run_mujoco_policy_rollout import parse_args


def test_trigger_shape_impulse_release_and_no_retrigger():
    pulse = LateralPulse(PulseConfig(6, .5, .025, "zm"))
    assert not pulse.force(0, .03, .001).any()
    assert not pulse.force(0, -.001, .001).any()
    assert not pulse.force(0, .025, .001, allowed=False).any()
    assert not pulse.force(1, .025, .001).any()
    assert pulse.trigger_time == 1
    assert pulse.force(1.25, .04, .001) == pytest.approx([0, 0, -6])
    impulse = sum(np.linalg.norm(pulse.force(1 + i * .001, .02, .001)) * .001
                  for i in range(500))
    assert impulse == pytest.approx(1.5)
    assert not pulse.force(1.5, .02, .001).any()
    assert pulse.release_time == 1.5
    assert not pulse.force(2, .02, .001).any()
    assert pulse.trigger_time == 1
    assert pulse.summary(True, 3)["recovered"]
    assert not pulse.summary(True, 1.4)["recovered"]
    assert not pulse.summary(False, None)["recovered"]


@pytest.mark.parametrize("kwargs", [dict(peak_n=-1), dict(peak_n=math.nan),
    dict(peak_n=11), dict(duration_s=0), dict(duration_s=math.inf),
    dict(remaining_m=.001), dict(remaining_m=math.nan), dict(direction="y")])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        PulseConfig(**kwargs)


def test_disabled_untriggered_and_early_abort_are_not_recovery():
    pulse = LateralPulse(PulseConfig())
    pulse.force(1, .01, .001)
    assert not pulse.summary(True, 2)["triggered"]
    pulse = LateralPulse(PulseConfig(2))
    pulse.force(1, .01, .001)
    assert not pulse.summary(True, 2)["recovered"]


def test_contact_distinguished_from_external_load():
    pulse = LateralPulse(PulseConfig(2))
    pulse.force(1, .025, .001)
    pulse.force(1.25, .025, .002)
    pulse.observe(timestamp=1.251, timestep=.001, remaining_m=.025, lateral_m=.002,
                  force_n=2, contact_force_n=0, contact_count=0)
    pulse.force(1.5, .02, .001)
    assert pulse.summary(True, 3)["recovered"]
    assert not pulse.summary(True, 3)["contact_recovery"]
    assert pulse.summary(True, 3)["delivered_impulse_ns"] == .002


@pytest.mark.parametrize("extra", [[], ["--task-profile", "moving_hole_fixed_peg"],
    ["--execute-actions"], ["--task-profile", "moving_hole_fixed_peg", "--execute-actions", "--hole-offset-x", ".001"],
    ["--task-profile", "moving_hole_fixed_peg", "--execute-actions", "--enable-axial-push"]])
def test_cli_rejects_unsafe_or_ambiguous_combinations(extra):
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "unused.pt", "--disturbance-peak-n", "2", *extra])


def test_real_mujoco_force_causes_motion_sensor_response_and_restores_wrench():
    mujoco = pytest.importorskip("mujoco")
    xml = '''<mujoco><option timestep="0.001" gravity="0 0 0"/>
      <worldbody><geom name="fixed_cylindrical_peg" type="sphere" size=".01" pos="5 0 0"/>
       <body><freejoint/><geom type="sphere" size=".02" mass="1"/>
        <body name="hole_tool" pos="0 .05 0"><geom type="sphere" size=".01" mass="1"/>
         <site name="ft"/></body></body></worldbody>
      <sensor><force name="ft_force" site="ft"/></sensor></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fixed = data.geom_xpos[0].copy()
    binding = MujocoPulse(model, data, PulseConfig(2))
    initial = data.xpos[binding.body_id].copy()
    max_sensor = 0.
    task = {"peg_to_hole_axial_error": -.025, "peg_to_hole_lateral_error": 0.}
    try:
        for _ in range(510):
            binding.apply(task)
            mujoco.mj_step(model, data)
            max_sensor = max(max_sensor, np.linalg.norm(data.sensordata))
            binding.observe(task, np.linalg.norm(data.sensordata))
        assert data.xpos[binding.body_id, 0] > initial[0] + .01
        assert max_sensor > .1
        assert data.geom_xpos[0] == pytest.approx(fixed)
        assert binding.pulse.impulse_ns == pytest.approx(.5)
        assert not data.xfrc_applied.any()
    finally:
        binding.close()
    # Early-exit cleanup must also remove an active pulse without dropping an
    # unrelated pre-existing wrench.
    data.xfrc_applied[binding.body_id, 5] = .7
    binding = MujocoPulse(model, data, PulseConfig(2))
    binding.apply(task)
    data.time += .25
    binding.apply(task)
    assert data.xfrc_applied[binding.body_id, 0] == pytest.approx(2)
    binding.close()
    assert data.xfrc_applied[binding.body_id] == pytest.approx([0, 0, 0, 0, 0, .7])
