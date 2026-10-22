"""Measured wrench source identity, normalization, causal masks and mixed-unit metrics."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from force_aware_act.real_robot.config import RealRobotDataConfig, WRENCH_AXES, WRENCH_UNITS, WRENCH_TOPIC, WRENCH_SCHEMA_VERSION
from force_aware_act.real_robot.reader import RawEpisode, discover_real_robot_episodes
from force_aware_act.real_robot.normalization import compute_real_robot_normalization_stats
from force_aware_act.real_robot.data import RealRobotHighRateDataset
from force_aware_act.real_robot.training import make_model_config, build_model, build_criterion, forward_training, RealRobotOptimizationConfig
from force_aware_act.real_robot.wrench_metrics import last_observed_wrench
from force_aware_act.real_robot.evaluation import aggregate_episode_metrics
from force_aware_act.act_aligned_training.high_rate_data import collate_high_rate_samples


def config():
    return RealRobotDataConfig(image_height=2, image_width=3, force_representation='end_effector_wrench6',
        wrench_component_units=WRENCH_UNITS, wrench_frame='right_ft_measurement',
        force_std_floor=(.01,.01,.01,.001,.001,.001))


def episode(root,name='train',offset=0):
    spec=importlib.util.spec_from_file_location('raw_helper',Path(__file__).with_name('test_real_robot_data.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    path=helper._write_raw(root,name,missing_disturbance_ms=())
    with h5py.File(path,'r+') as h:
        h.attrs.update(schema_version=WRENCH_SCHEMA_VERSION,force_representation='end_effector_wrench6',
                       primary_force_dataset='observations/ee_wrench',gravity_compensation='applied_upstream',source_sha256=name)
        times=helper.EPOCH + np.rint(np.arange(119)*1e9/300).astype(np.int64)+500000
        for clock in ['log_ns','publish_ns','header_ns']:
            h.create_dataset('timestamps/ft_sensor/'+clock,data=times-(100000000 if clock=='header_ns' else 0))
        h.create_dataset('timestamps/ft_sensor/frame_id',data=np.asarray([b'right_ft_measurement']*len(times)))
        h['timestamps/ft_sensor'].attrs['source_topic']=WRENCH_TOPIC
        values=np.arange(len(times))[:,None]*np.array([.1,.2,.3,.01,.02,0.])[None]+offset
        h.create_dataset('metadata/ee_wrench_axes',data=np.asarray(WRENCH_AXES,dtype='S'))
        ds=h.create_dataset('observations/ee_wrench',data=values)
        ds.attrs.update(source_topic=WRENCH_TOPIC,source_field='effort',timestamp_group='/timestamps/ft_sensor',
                        unit='source_native; no unit conversion',gravity_compensation='applied_upstream')
    return path


def dataset(root,cfg=None):
    cfg=cfg or config();records=discover_real_robot_episodes(root,cfg)
    stats=compute_real_robot_normalization_stats(root,records,cfg)
    mc=make_model_config('contact',cfg,compact=True,chunk_len=20,pretrained=False,
                         force_prediction_steps=20,strict_force_horizon=True)
    return RealRobotHighRateDataset(root,records,stats,mc,cfg),stats,mc


def test_config_requires_explicit_units_frame_and_floors_and_preserves_legacy():
    with pytest.raises(ValueError,match='units'):RealRobotDataConfig(force_representation='end_effector_wrench6')
    for change in [{'wrench_component_units':('N',)*6},{'wrench_frame':''},{'force_std_floor':(0.,)*6}]:
        with pytest.raises(ValueError):replace(config(),**change)
    assert RealRobotDataConfig.from_dict(config().to_dict())==config()
    assert 'force_std_floor' not in RealRobotDataConfig().to_dict()
    contract=config().contract_dict()
    assert contract['force_timestamp_stream']=='ft_sensor' and contract['force_component_units']==WRENCH_UNITS


@pytest.mark.parametrize('damage',['axes','topic','clock','frame','unit','compensation','schema'])
def test_mislabelled_sensor_is_rejected(tmp_path,damage):
    p=episode(tmp_path)
    with h5py.File(p,'r+') as h:
        if damage=='axes':h['metadata/ee_wrench_axes'][0]=b'Tx'
        elif damage=='topic':h['observations/ee_wrench'].attrs['source_topic']='wrong'
        elif damage=='clock':h['observations/ee_wrench'].attrs['timestamp_group']='/timestamps/disturbance'
        elif damage=='frame':h['timestamps/ft_sensor/frame_id'][0]=b'base'
        elif damage=='unit':h['observations/ee_wrench'].attrs['unit']='Nmm'
        elif damage=='compensation':h.attrs['gravity_compensation']='unknown'
        else:h.attrs['schema_version']='real_robot_right_arm_raw_v2'
    with pytest.raises(ValueError):RawEpisode(p,config())


def test_source_independent_of_auxiliary_joint_streams_and_no_future_history(tmp_path):
    p=episode(tmp_path);ds,stats,mc=dataset(tmp_path)
    index=ds.index.index((0,10));sample=ds[index];trace=ds.sample_trace(index)
    assert trace['force_timestamp_stream']=='ft_sensor'
    assert trace['force_source_topic']==WRENCH_TOPIC
    assert sample.online_force_intervals.shape==(20,8,6)
    assert sample.future_force_intervals.shape==(20,8,6)
    assert sample.future_force_target.shape==(20,6)
    anchor=trace['observation']['anchor_ns']
    with h5py.File(p,'r+') as h:
        # Invalid auxiliary values/clocks must not participate in a sensor-wrench run.
        h['observations/joint_external_torque'][:]=np.nan
        h['observations/joint_internal_torque'][:]=999999
        h['timestamps/disturbance/log_ns'][:]=0
    raw=RawEpisode(p,config())
    assert raw.force.shape==(119,6) and len(raw.qpos)==121
    assert compute_real_robot_normalization_stats(tmp_path,ds.episodes,config())==stats
    changed=RealRobotHighRateDataset(tmp_path,ds.episodes,stats,mc,config())[index]
    torch.testing.assert_close(changed.future_force_intervals,sample.future_force_intervals,rtol=0,atol=0)
    with h5py.File(p,'r+') as h:
        times=h['timestamps/ft_sensor/log_ns'][...]
        values=h['observations/ee_wrench'][...];values[times>anchor]+=123.
        h['observations/ee_wrench'][:]=values
    after=RealRobotHighRateDataset(tmp_path,ds.episodes,stats,mc,config())[index]
    torch.testing.assert_close(after.online_force_intervals,sample.online_force_intervals,rtol=0,atol=0)
    assert not torch.equal(after.future_force_intervals,sample.future_force_intervals)
    indices=np.array(trace['future_force']['hdf5_indices']);valid=indices>=0
    assert (raw.torque_timestamps_ns[indices[valid]]>anchor).all()
    assert (raw.torque_timestamps_ns[indices[valid]]<=anchor+200000000).all()


def test_normalization_uses_training_wrench_and_per_axis_floors_only(tmp_path):
    p=episode(tmp_path,'train');episode(tmp_path,'validation',offset=1e6)
    cfg=config();records=[r for r in discover_real_robot_episodes(tmp_path,cfg) if r.episode_id=='train']
    stats=compute_real_robot_normalization_stats(tmp_path,records,cfg)
    with h5py.File(p,'r') as h:v=h['observations/ee_wrench'][...]
    np.testing.assert_allclose(stats.force_mean,v.mean(0))
    np.testing.assert_allclose(stats.force_std,np.maximum(v.std(0),cfg.force_std_floor))
    assert stats.force_std[-1]==pytest.approx(.001)
    with h5py.File(tmp_path/'validation/episode.hdf5','r+') as h:h['observations/ee_wrench'][:]=9e8
    assert compute_real_robot_normalization_stats(tmp_path,records,cfg)==stats


@pytest.mark.parametrize('kind',['contact','motion'])
def test_six_axis_model_forward_backward_and_loss_components(tmp_path,kind):
    episode(tmp_path);ds,_,mc=dataset(tmp_path)
    mc=make_model_config(kind,config(),compact=True,chunk_len=20,pretrained=False,force_prediction_steps=20,strict_force_horizon=True)
    batch=collate_high_rate_samples([ds[10],ds[-1]])
    model=build_model(kind,vars(mc));settings=RealRobotOptimizationConfig()
    outputs,loss=forward_training(kind,model,build_criterion(kind,settings),batch,sample_posterior=False)
    assert outputs['pred_force'].shape==(2,20,6)
    assert outputs['pred_force_highrate'].shape==(2,20,8,6)
    torch.testing.assert_close(loss['loss_force'],.5*(loss['loss_force_xyz']+loss['loss_moment_xyz']))
    torch.testing.assert_close(loss['loss_force_highrate'],.5*(loss['loss_force_highrate_xyz']+loss['loss_moment_highrate_xyz']))
    loss['loss_total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_physical_metrics_keep_newtons_and_moments_separate(tmp_path,monkeypatch):
    from force_aware_act.real_robot import training
    episode(tmp_path);ds,stats,mc=dataset(tmp_path)
    batch=collate_high_rate_samples([ds[10],ds[-1]])
    shift=torch.tensor([2.,2.,2.,.2,.2,.2])/torch.tensor(stats.force_std)
    outputs={'pred_action':batch.action_chunk.clone(),
             'pred_force':batch.future_force_target+shift,
             'pred_force_highrate':batch.future_force_intervals+shift}
    # Masked NaN labels never enter metrics.
    outputs['pred_force_highrate'][batch.future_force_sample_padding_mask]=float('nan')
    monkeypatch.setattr(training,'deployment_predictions',lambda *args, **kwargs:outputs)
    model=torch.nn.Identity();model.config=mc
    row=training.evaluate('contact',model,[batch],stats,'cpu')
    assert row['force_mae_n']==pytest.approx(2.,abs=1e-6)
    assert row['moment_mae_nm']==pytest.approx(.2,abs=1e-6)
    assert row['force_endpoint_mae_n']==pytest.approx(2.,abs=1e-6)
    assert 'torque_mae_nm' not in row
    result=aggregate_episode_metrics({'one':row,'two':row})
    assert result['episode_macro_moment_mae_nm']==pytest.approx(.2,abs=1e-6)
    assert result['force_mae_n_valid_scalars']==2*row['force_mae_n_valid_scalars']
    hold=last_observed_wrench(batch)
    for i in range(2):
        values=batch.online_force_intervals[i][~batch.online_force_sample_padding_mask[i]]
        torch.testing.assert_close(hold[i],values[-1])


def test_act_can_share_raw_v3_anchors_without_consuming_wrench(tmp_path):
    from force_aware_act.real_robot.training import deployment_predictions, evaluate
    episode(tmp_path);ds,stats,_=dataset(tmp_path)
    batch=collate_high_rate_samples([ds[10]])
    mc=make_model_config('act',config(),compact=True,chunk_len=20,pretrained=False)
    model=build_model('act',vars(mc)).eval()
    with torch.no_grad():
        expected=deployment_predictions('act',model,batch)['pred_action']
        batch.online_force_intervals.fill_(float('nan'))
        batch.future_force_intervals.fill_(float('nan'))
        actual=deployment_predictions('act',model,batch)['pred_action']
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    metrics=evaluate('act',model,[batch],stats,'cpu')
    assert np.isfinite(metrics['action_mae_rad']) and 'force_mae_n' not in metrics
