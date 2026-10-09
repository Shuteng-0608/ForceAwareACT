"""Force-only conversion, causal stream identity, masks and live/offline agreement."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from force_aware_act.real_robot.flange_force import (
    _geometry, joint_torque_to_flange_force,
)
from force_aware_act.real_robot.config import RealRobotDataConfig
from force_aware_act.real_robot.reader import RawEpisode, discover_real_robot_episodes
from force_aware_act.real_robot.normalization import compute_real_robot_normalization_stats
from force_aware_act.real_robot.data import RealRobotHighRateDataset
from force_aware_act.real_robot.sampling import pack_torque_history, pack_future_torque
from force_aware_act.real_robot.training import make_model_config
from force_aware_act.real_robot.evaluation import aggregate_episode_metrics
from force_aware_act.inference.real_robot import RealRobotObservationBuffer, RealRobotPolicyAdapter, validate_data_contract

Q = np.array([-1.5, -1.34, 1.58, -1.43, .22, -.065, .034])


def _module(filename):
    spec = importlib.util.spec_from_file_location("force3_" + Path(filename).stem,
                                                Path(__file__).parents[1] / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode(root, name="episode", *, missing_state=False, delayed=False):
    helper = _module("tests/test_real_robot_data.py")
    path = helper._write_raw(root, name, missing_state_ms=(50,) if missing_state else (),
                             missing_disturbance_ms=())
    with h5py.File(path, "r+") as f:
        state = f['timestamps/state/header_ns'][()]
        disturbance = f['timestamps/disturbance/header_ns'][()]
        start = state[0]
        def pose(times):
            return Q[None] + np.sin((times-start)[:, None] / 1e9) * np.arange(1,8)[None] * .01
        f['observations/joint_pos'][:] = pose(state)
        dq = pose(disturbance)
        j, _ = _geometry(dq)
        expected_w = np.tile([2., -3., 5., .1, -.2, .3], (len(dq), 1))
        f['observations/joint_external_torque'][:] = np.einsum('nji,nj->ni',j,expected_w)
        if delayed:
            f['timestamps/state/log_ns'][3] = int(f['timestamps/state/log_ns'][3]) + 500_000
        f.attrs['source_sha256'] = name
    return path, replace(helper._config(), force_representation="flange_force3")


def test_scalar_batch_and_existing_reference_agree_including_nonzero_moments():
    reference = _module("scripts/analyze_rollout_external_wrench.py")
    q = np.stack([Q, Q+.01, Q-.05])
    tau = np.array([[1.,2.,3.,4.,5.,6.,7.], [-1.,1.,.2,.4,.6,.8,-1.], [0.]*7])
    batch = joint_torque_to_flange_force(q,tau)
    for i in range(3):
        single = joint_torque_to_flange_force(q[i],tau[i])
        j = reference.flange_jacobian(q[i])
        w = reference.equivalent_wrench(j,tau[i])['wrench_base']
        from force_aware_act.control.kinematics import right_flange_transform
        local = reference.rotate_wrench_to_flange(w,right_flange_transform(q[i])[:3,:3])
        np.testing.assert_allclose(single.force, local[:3], atol=1e-10)
        np.testing.assert_allclose(batch.force[i], single.force, atol=1e-11)
        np.testing.assert_allclose(batch.relative_residual[i],single.relative_residual,atol=1e-12)
    j, rotation = _geometry(Q[None])
    known = np.array([2.,-3.,5.,.5,-.8,.3])
    out = joint_torque_to_flange_force(Q,j[0].T@known)
    np.testing.assert_allclose(out.force,rotation[0].T@known[:3],atol=1e-11)
    assert out.valid and out.relative_residual < 1e-12


def test_geometry_matches_independent_fk_finite_difference():
    from force_aware_act.control.kinematics import right_flange_transform
    j,rotation = _geometry(Q[None]);epsilon=1e-6
    for k in range(7):
        a=Q.copy();b=Q.copy();a[k]+=epsilon;b[k]-=epsilon
        plus=right_flange_transform(a);minus=right_flange_transform(b)
        linear=(plus[:3,3]-minus[:3,3])/(2000*epsilon)
        omega=((plus[:3,:3]-minus[:3,:3])/(2*epsilon))@rotation[0].T
        angular=np.array([omega[2,1]-omega[1,2],omega[0,2]-omega[2,0],omega[1,0]-omega[0,1]])/2
        np.testing.assert_allclose(j[0,:,k],np.r_[linear,angular],atol=1e-9)


def test_invalid_inputs_and_singular_pose_are_explicit():
    with pytest.raises(ValueError,match="shapes"):
        joint_torque_to_flange_force(Q,np.zeros(6))
    with pytest.raises(ValueError,match="finite"):
        joint_torque_to_flange_force(Q,np.full(7,np.nan))
    result=joint_torque_to_flange_force(np.zeros(7),np.ones(7))
    assert not result.valid
    np.testing.assert_array_equal(result.force,np.zeros(3))
    empty=joint_torque_to_flange_force(np.empty((0,7)),np.empty((0,7)))
    assert empty.force.shape==(0,3)


def test_exact_pairing_masks_and_delayed_source_never_enter_earlier_history(tmp_path):
    path,config=_episode(tmp_path,missing_state=True,delayed=True)
    raw=RawEpisode(path,config);stream=raw.force
    assert stream.diagnostics['unpaired_samples']==1
    assert stream.diagnostics['availability_delayed_samples']==1
    source_times=raw.timestamps['disturbance']
    # Force at 10ms needs a state recorded at 10.5ms, so it is not yet available.
    early=pack_torque_history(raw.torque_timestamps_ns,stream,source_times[3],config)
    late=pack_torque_history(raw.torque_timestamps_ns,stream,source_times[3]+500_000,config)
    assert late.valid_sample_count==early.valid_sample_count+1
    missing=np.flatnonzero(stream.state_indices<0)[0]
    values,valid=stream.read(np.array([missing]))
    assert not valid[0] and not values.any()
    future=pack_future_torque(raw.torque_timestamps_ns,stream,source_times[0]+40_000_000,3,config)
    assert future.sample_padding_mask[0,2]  # Missing 50ms endpoint stays in its slot.
    assert not future.sample_padding_mask[1,2]
    # On demand: no force cache hides changes to the source array.
    a,_=stream.read(np.array([0]));stream.torque[0]*=2;b,_=stream.read(np.array([0]))
    np.testing.assert_allclose(b,2*a)


def test_force3_normalization_uses_only_valid_native_training_samples(tmp_path):
    _,config=_episode(tmp_path,'train',missing_state=True)
    _episode(tmp_path,'validation')
    records=discover_real_robot_episodes(tmp_path,config)
    train=[r for r in records if r.episode_id=='train']
    stats=compute_real_robot_normalization_stats(tmp_path,train,config)
    raw=RawEpisode(train[0].resolve(tmp_path),config)
    force,valid=raw.force.read(slice(None))
    np.testing.assert_allclose(stats.force_mean,force[valid].mean(axis=0),atol=1e-12)
    np.testing.assert_allclose(stats.force_std,np.maximum(force[valid].std(axis=0),.01),atol=1e-12)
    with h5py.File(tmp_path/'validation/episode.hdf5','r+') as f:
        f['observations/joint_external_torque'][:]=99999.
    assert stats==compute_real_robot_normalization_stats(tmp_path,train,config)
    model_config=make_model_config('contact',config,compact=True,chunk_len=3,pretrained=False)
    dataset=RealRobotHighRateDataset(tmp_path,train,stats,model_config,config)
    sample=dataset[dataset.index.index((0,4))]
    assert sample.online_force_intervals.shape==(20,8,3)
    assert sample.future_force_intervals.shape==(3,8,3)
    assert sample.future_force_target.shape==(3,3)
    assert sample.future_force_target_padding_mask[0]
    assert not sample.future_force_target[0].any()
    trace=dataset.sample_trace(dataset.index.index((0,4)))
    assert trace['future_force']['endpoint_padding_mask'][0]
    assert not trace['future_force']['valid_conversion_slots'][0][2]


def test_future_pose_and_torque_changes_do_not_change_force_history(tmp_path):
    path,config=_episode(tmp_path)
    raw=RawEpisode(path,config)
    anchor=int(raw.anchor_timestamps_ns[12])
    first=pack_torque_history(raw.torque_timestamps_ns,raw.force,anchor,config)
    raw.qpos[raw.state_timestamps_ns>anchor]+=1.
    raw.torque[raw.torque_timestamps_ns>anchor]+=100.
    second=pack_torque_history(raw.torque_timestamps_ns,raw.force,anchor,config)
    np.testing.assert_array_equal(first.values,second.values)
    np.testing.assert_array_equal(first.sample_padding_mask,second.sample_padding_mask)


def test_live_invalid_conversion_is_masked_in_snapshot():
    config=RealRobotDataConfig(force_representation='flange_force3')
    buffer=RealRobotObservationBuffer(config)
    image=np.zeros((config.image_height,config.image_width,3),dtype=np.uint8)
    for camera in config.camera_names:buffer.append_image(camera,0,image)
    buffer.append_state(0,Q,np.ones(7))
    buffer.append_external_torque(3_333_333,np.ones(7),qpos=np.zeros(7))
    observation=buffer.observation_at(10_000_000)
    assert observation.torque_valid_mask.tolist()==[True,False]
    np.testing.assert_array_equal(observation.torque_values[1],np.zeros(3))


def test_live_same_frame_conversion_and_offline_history_agree(tmp_path):
    _,config=_episode(tmp_path,delayed=True)
    records=discover_real_robot_episodes(tmp_path,config)
    raw=RawEpisode(records[0].resolve(tmp_path),config)
    stats=compute_real_robot_normalization_stats(tmp_path,records,config)
    model_config=make_model_config('contact',config,compact=True,chunk_len=3,pretrained=False)
    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__();self.config=model_config;self.anchor=torch.nn.Parameter(torch.zeros(()))
    adapter=RealRobotPolicyAdapter(Capture(),model_kind='contact',normalization=stats.to_dict(),data_config=config,data_contract=config.contract_dict())
    buffer=RealRobotObservationBuffer(config)
    for camera in config.camera_names:
        for i,t in enumerate(raw.timestamps[camera]):
            with h5py.File(raw.path) as f:image=f[f'observations/images/{camera}/rgb'][i]
            buffer.append_image(camera,int(t),image)
    for t,q in zip(raw.state_timestamps_ns,raw.qpos):buffer.append_state(int(t),q)
    for i,t in enumerate(raw.torque_timestamps_ns):
        qi=raw.qpos[raw.force.state_indices[i]]
        buffer.append_external_torque(int(t),raw.torque[i],qpos=qi)
    anchor=int(raw.anchor_timestamps_ns[12])
    inputs=adapter.prepare_observation(buffer,anchor)
    packed=pack_torque_history(raw.torque_timestamps_ns,raw.force,anchor,config)
    expected=stats.normalize_force(torch.from_numpy(packed.values))
    expected[packed.sample_padding_mask]=0
    torch.testing.assert_close(inputs.online_force_intervals[0],expected,atol=1e-6,rtol=1e-6)
    with pytest.raises(ValueError,match='same-frame'):
        RealRobotObservationBuffer(config).append_external_torque(anchor,np.zeros(7))
    wrong=RealRobotDataConfig(image_height=2,image_width=3)
    with pytest.raises(ValueError,match='contract'):
        validate_data_contract(config.contract_dict(),wrong)
    assert 'force_representation' not in wrong.to_dict()


@pytest.mark.parametrize('model_kind', ['contact', 'motion'])
def test_force3_cli_training_resume_and_newton_metrics(tmp_path,capsys,model_kind):
    helper=_module('tests/test_real_robot_cli.py')
    _,config=_episode(tmp_path/'data','a');_episode(tmp_path/'data','b')
    config_path=tmp_path/'config.json'
    import json
    config_path.write_text(json.dumps(config.to_dict()))
    args=helper._args(tmp_path/'data',config_path,tmp_path/'run',1,sampling_mode='episodic_uniform')
    args[args.index('--model')+1]=model_kind
    args[args.index('--mode')+1]='test'
    first=helper._runner().main(args)
    assert first['checkpoint_round_trip']=='passed'
    metrics=first['final_metrics']['held_out_validation']
    assert metrics['force_unit']=='N' and metrics['force_valid_scalars']>0
    assert len(metrics['force_mae_per_axis_n'])==3 and 'torque_mae_nm' not in metrics
    aggregate=aggregate_episode_metrics({'one':metrics,'two':metrics})
    assert aggregate['episode_macro_force_mae_n']==metrics['force_mae_n']
    args[args.index('--steps')+1]='2'
    resumed=helper._runner().main(args+['--resume',str(tmp_path/'run/last.pt')])
    assert resumed['completed_steps']==2 and resumed['checkpoint_round_trip']=='passed'
    from force_aware_act.real_robot.checkpoint import read_checkpoint
    from force_aware_act.inference.real_robot import RealRobotPolicyAdapter
    checkpoint=read_checkpoint(tmp_path/'run/last.pt')
    adapter=RealRobotPolicyAdapter.from_checkpoint(checkpoint)
    assert adapter.model.config.force_dim==3
    assert checkpoint['model_config']['force_dim']==3
    if model_kind=='motion':
        assert hasattr(adapter.model,'motion_posterior')
        assert not hasattr(adapter.model,'contact_posterior')
        assert 'loss_prior_match' not in resumed['final_metrics']['train']
    capsys.readouterr()
