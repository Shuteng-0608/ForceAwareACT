import numpy as np
import json
from pathlib import Path
import pytest
from scripts.analyze_sensor_deployment_reference import spans, threshold_crossings
from scripts.deployment.run_sensor55_x14y08 import controller_arguments, main, EXPECTED_MODEL_SHA
from force_aware_act.control.goal_selection import select_goal


def test_contiguous_duration_does_not_bridge_gaps_or_false_samples():
    assert spans([0,.01,.02,.5,.51,.52],[1,1,0,1,1,1])==[(0,1,.01),(3,5,.020000000000000018)]
    assert spans([0,.01],[0,0])==[]


def test_hard_threshold_uses_absolute_inclusive_axes():
    w=np.zeros((4,6));w[1:3,2]=[-10,11]
    result=threshold_crossings(np.array([0,.01,.02,.03]),w,[10]*6)
    assert result['first_axes']==[2] and result['first_s']==.01
    assert result['counts'][2]==2 and result['longest_s'][2]==.01


def test_new_model_pinned_and_preview_never_connects(capsys):
    args=controller_arguments([])
    assert '--enable-control' not in args
    assert 'contact_wrench6_ah20_fh20_seed0_b8_es3000_p20_r001' in args[args.index('--checkpoint')+1]
    assert main([])==0
    result=json.loads(capsys.readouterr().out)
    assert not result['hardware_connected'] and not result['commands_sent']
    assert result['checkpoint_sha256']==EXPECTED_MODEL_SHA
    with pytest.raises(SystemExit):controller_arguments(['--checkpoint','old.pt'])
    with pytest.raises(SystemExit):controller_arguments(['--seconds','60'])


def test_new_profile_only_selects_new_x14y08_and_guards_are_explicit():
    root=Path('configs/deployment/profiles/contact_sensor55_x14y08_v1')
    s=select_goal('x14y08',root)
    np.testing.assert_allclose(s.config.goal.target_position_m,[.5084240171297213,-.3148913705400862,.2267064691240277])
    assert s.config.goal.radius_m==.003 and s.config.goal.dwell_ms==500
    assert set(s.config.limits.first_action_error_rad)=={.1}
    assert set(s.config.limits.following_error_rad)=={.1}
    assert set(s.config.limits.max_command_step_rad)=={.03}
    with pytest.raises(ValueError,match='unknown goal'):select_goal('x13y07',root)
    session=json.loads(Path('configs/deployment/command_session_sensor55_candidate.json').read_text())
    assert session['allow_hardware'] is False and session['servo_off_after_standstill'] is True
    assert session['ft_force_hard_n']==[26,10,83]
    assert session['ft_torque_hard_nm']==[1.15,2.55,.24]
