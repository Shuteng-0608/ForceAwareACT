"""Configuration consistency and nominal arithmetic; no fault injection or SDK."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from force_aware_act.control.contracts import Feedback, Limits, ShadowConfig
from force_aware_act.control.execution import ActionLimiter

ROOT=Path(__file__).resolve().parents[1]
PROFILE=ROOT/'configs/deployment/profiles/contact_right_step_v1'
PATHS=[PROFILE/'base.json',*sorted((PROFILE/'goals').glob('*.json'))]
SOFT=(17.5,6.5,5.,11.,1.,6.5,2.)
HARD=(20.,6.75,5.5,12.,1.2,7.,2.5)


@pytest.mark.parametrize('path',PATHS,ids=lambda p:p.stem)
def test_each_profile_loads_exact_selected_values(path):
    config=ShadowConfig.read(path);limits=config.limits
    assert limits.authority=='operator_reviewed'
    assert limits.torque_soft_nm==SOFT and limits.torque_hard_nm==HARD
    assert limits.torque_release_nm==tuple(float(Decimal(str(v))*Decimal('.95')) for v in SOFT)
    assert limits.torque_soft_duration_ms==100.
    assert limits.motion_guard=='step_reject' and limits.max_command_step_rad==(.03,)*7
    assert limits.first_action_error_rad==(.05,)*7
    assert limits.following_error_rad==(.05,)*7
    assert limits.velocity_rad_s is None and limits.acceleration_rad_s2 is None
    assert limits.max_consecutive_limited_ms is None
    assert limits.missing()==[]
    assert config.mode=='shadow' and config.allow_motion is False
    assert not config.readiness()['motion_possible']
    assert not config.readiness()['host_velocity_guard_enabled']
    assert not config.readiness()['controller_limits_readback_verified']
    assert ShadowConfig.from_dict(json.loads(json.dumps(config.to_dict())))==config
    if path.parent.name=='goals':
        old=ShadowConfig.read(ROOT/'configs/deployment/goals'/path.name)
        assert config.goal==old.goal
        assert config.goal.radius_m==.003 and config.goal.dwell_ms==500.


def test_profile_manifest_and_controller_baseline():
    manifest=json.loads((PROFILE/'manifest.json').read_text())
    assert len(PATHS)==11 and len(manifest['goal_configs'])==10
    for path,digest in manifest['files_sha256'].items():
        assert hashlib.sha256((PROFILE/path).read_bytes()).hexdigest()==digest
    for path,digest in manifest['runtime_sources_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    baseline=json.loads((PROFILE/'controller_baseline.json').read_text())
    assert baseline['velocity_ratio_percent']==10 and baseline['acceleration_ratio_percent']==10
    assert baseline['apply_on_load'] is False and baseline['controller_readback_verified'] is False


def feedback(q,t):
    return Feedback(t,t//1_000_000+1,tuple(q),(0.,)*7,(0.,)*7,t,True,'nominal_arithmetic_only')


def test_step_guard_preserves_ordinary_targets_without_derivative_shaping():
    config=ShadowConfig.read(PROFILE/'base.json');limiter=ActionLimiter(config.limits)
    q=(np.asarray(config.limits.joint_min_rad)+config.limits.joint_max_rad)/2
    # First alignment and later per-step checks are distinct accepted cases.
    target=q+.04
    first=limiter.apply(target,feedback(q,0),0)
    np.testing.assert_array_equal(first['action_rad'],target)
    assert first['step_reference']=='initial_actual_feedback' and first['step_rad'] is None
    second=limiter.apply(target+.005,feedback(q,10_000_000),10_000_000)
    np.testing.assert_array_equal(second['action_rad'],target+.005)
    np.testing.assert_allclose(second['step_rad'],.005)
    assert second['fault'] is None and not second['limited'] and second['velocity_rad_s'] is None
    third=limiter.apply(target+.035,feedback(target+.005,20_000_000),20_000_000)
    assert third['fault'] is None and third['step_reference']=='previous_accepted_shadow_target'
    np.testing.assert_allclose(third['step_rad'],.03,atol=1e-12)


def test_old_profile_defaults_and_nominal_legacy_calculation_are_preserved():
    old=ShadowConfig.read(ROOT/'configs/deployment/contact_right_shadow.json')
    assert old.limits.motion_guard=='velocity_acceleration'
    assert old.limits.max_command_step_rad is None
    limits=Limits(joint_min_rad=(-2.,)*7,joint_max_rad=(2.,)*7,
                  velocity_rad_s=(1.,)*7,acceleration_rad_s2=(2.,)*7,
                  following_error_rad=(.3,)*7,first_action_error_rad=(.2,)*7)
    first=ActionLimiter(limits).apply([.1]*7,feedback([0.]*7,0),0)
    np.testing.assert_allclose(first['action_rad'],.0002)
    np.testing.assert_allclose(first['velocity_rad_s'],.02)
