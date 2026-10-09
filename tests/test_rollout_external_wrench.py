"""Independent geometry, virtual work and least-squares residual verification."""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.analyze_rollout_external_wrench import flange_jacobian,equivalent_wrench,rotate_wrench_to_flange,moving_mean,analysis_windows
from force_aware_act.control.kinematics import right_flange_transform

Q=np.array([-1.5,-1.34,1.58,-1.43,.22,-.065,.034])


def test_jacobian_matches_independent_fk_finite_differences():
    epsilon=1e-6; j=flange_jacobian(Q)
    for k in range(7):
        plus=Q.copy();minus=Q.copy();plus[k]+=epsilon;minus[k]-=epsilon
        a=right_flange_transform(plus);b=right_flange_transform(minus)
        linear=(a[:3,3]-b[:3,3])/1000/(2*epsilon)
        angular=Rotation.from_matrix(a[:3,:3]@b[:3,:3].T).as_rotvec()/(2*epsilon)
        np.testing.assert_allclose(j[:3,k],linear,atol=1e-9,rtol=0)
        np.testing.assert_allclose(j[3:,k],angular,atol=1e-9,rtol=0)


def test_known_wrench_is_recovered_and_virtual_work_is_preserved():
    j=flange_jacobian(Q);expected=np.array([12.,-4.,21.,.8,-1.2,.5]);tau=j.T@expected
    for length in (.01,.1,1.):
        out=equivalent_wrench(j,tau,length_scale_m=length)
        assert out['rank']==6 and out['relative_residual']<1e-12
        np.testing.assert_allclose(out['wrench_base'],expected,atol=1e-10)
    dq=np.array([.1,-.2,.03,.04,.01,-.1,.05])
    np.testing.assert_allclose(tau@dq,expected@(j@dq),atol=1e-12)


def test_seventh_joint_null_component_is_retained_as_unexplained_torque():
    j=flange_jacobian(Q);_,_,vt=np.linalg.svd(j,full_matrices=True)
    outside=vt[-1];tau=j.T@np.array([1.,2.,3.,.1,.2,.3])+outside
    out=equivalent_wrench(j,tau)
    np.testing.assert_allclose(out['unexplained_torque_nm'],outside,atol=1e-12)
    np.testing.assert_allclose(j@out['unexplained_torque_nm'],0,atol=1e-12)
    assert out['relative_residual']>0


def test_wrench_rotation_preserves_both_norms_and_virtual_work():
    j=flange_jacobian(Q);r=right_flange_transform(Q)[:3,:3]
    w=np.array([12.,-4.,21.,.8,-1.2,.5]);local=rotate_wrench_to_flange(w,r)
    local_j=np.r_[r.T@j[:3],r.T@j[3:]]
    np.testing.assert_allclose(local_j.T@local,j.T@w,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(local[:3]),np.linalg.norm(w[:3]))
    np.testing.assert_allclose(np.linalg.norm(local[3:]),np.linalg.norm(w[3:]))


def test_zero_estimate_and_rank_deficiency_are_explicit():
    out=equivalent_wrench(flange_jacobian(Q),np.zeros(7))
    np.testing.assert_array_equal(out['wrench_base'],0);assert out['relative_residual']==0
    out=equivalent_wrench(np.zeros((6,7)),np.ones(7))
    assert out['rank']==0 and np.isinf(out['scaled_condition'])
    np.testing.assert_array_equal(out['unexplained_torque_nm'],1)


def test_display_mean_keeps_sample_count_and_constant_level():
    for n in (1,2,15,16):
        a=np.full((100,7),2.)
        np.testing.assert_allclose(moving_mean(a,n),a,atol=1e-14)


def test_statistics_windows_cover_short_and_long_actual_runs():
    assert analysis_windows(3.)==[(0.,3.)]
    assert analysis_windows(9.999665)==[(0.,5.),(5.,9.999665)]
    assert analysis_windows(30.)[-1]==(25.,30.)
    for duration in (3.,9.999665,30.,30.001):
        windows=analysis_windows(duration)
        assert windows[0][0]==0 and windows[-1][1]==duration
        assert all(a<b for a,b in windows)
        assert all(windows[i][1]==windows[i+1][0] for i in range(len(windows)-1))
