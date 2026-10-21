"""Sensor force-only ablation: provenance, channel isolation, masks and gradients."""

from dataclasses import fields, replace
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from force_aware_act.act_aligned_training.high_rate_data import collate_high_rate_samples
from force_aware_act.real_robot.config import RealRobotDataConfig, WRENCH_SCHEMA_VERSION
from force_aware_act.real_robot.data import RealRobotHighRateDataset
from force_aware_act.real_robot.evaluation import aggregate_episode_metrics, evaluate_episodes, fixed_indices, index_manifest
from force_aware_act.real_robot.normalization import compute_real_robot_normalization_stats
from force_aware_act.real_robot.reader import RawEpisode, discover_real_robot_episodes
from force_aware_act.real_robot.training import (
    RealRobotOptimizationConfig, build_criterion, build_model, deployment_predictions,
    evaluate, forward_training, make_model_config,
)


def helper():
    path = Path(__file__).with_name('test_real_robot_wrench6.py')
    spec = importlib.util.spec_from_file_location('sensor_wrench_test_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return replace(helper().config(), force_representation='end_effector_force3',
                   wrench_component_units=('N',) * 3, force_std_floor=(.01,) * 3)


def dataset(root, kind='contact'):
    cfg = config()
    records = discover_real_robot_episodes(root, cfg)
    stats = compute_real_robot_normalization_stats(root, records, cfg)
    mc = make_model_config(kind, cfg, compact=True, chunk_len=20, pretrained=False,
                           force_prediction_steps=20, strict_force_horizon=True)
    return RealRobotHighRateDataset(root, records, stats, mc, cfg), stats, mc


def test_source_contract_is_sensor_not_derived_force():
    cfg = config()
    assert cfg.force_dim == 3 and cfg.uses_wrench_sensor
    assert cfg.raw_schema_version == WRENCH_SCHEMA_VERSION and cfg.force_timestamp_stream == 'ft_sensor'
    assert RealRobotDataConfig.from_dict(cfg.to_dict()) == cfg
    contract = cfg.contract_dict()
    assert contract['force_axes'] == ('Fx', 'Fy', 'Fz')
    assert contract['source_force_axes'] == ('Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz')
    assert contract['force_channel_indices'] == (0, 1, 2)
    assert contract['force_coordinate_frame'] == 'right_ft_measurement'
    assert contract['moment_input'] is False and contract['moment_target'] is False
    assert 'force_conversion' not in contract and 'torque_unit' not in contract
    old = RealRobotDataConfig(force_representation='flange_force3')
    assert not old.uses_wrench_sensor and old.force_timestamp_stream == 'disturbance'
    for kwargs in ({'wrench_component_units': ('N',) * 6}, {'force_std_floor': (.01,) * 6},
                   {'force_std_floor': (.01, 0, .01)}, {'wrench_frame': ''}):
        with pytest.raises(ValueError):
            replace(cfg, **kwargs)


@pytest.mark.parametrize('damage', ['axes', 'frame', 'shape', 'clock', 'topic', 'schema', 'force_nan'])
def test_invalid_selected_source_is_rejected(tmp_path, damage):
    path = helper().episode(tmp_path)
    with h5py.File(path, 'r+') as h:
        if damage == 'axes': h['metadata/ee_wrench_axes'][0] = b'Tx'
        elif damage == 'frame': h['timestamps/ft_sensor/frame_id'][0] = b'base'
        elif damage == 'clock': h['timestamps/ft_sensor/log_ns'][1] = h['timestamps/ft_sensor/log_ns'][0]
        elif damage == 'topic': h['observations/ee_wrench'].attrs['source_topic'] = 'wrong'
        elif damage == 'schema': h.attrs['schema_version'] = 'real_robot_right_arm_raw_v2'
        elif damage == 'force_nan': h['observations/ee_wrench'][5, 0] = np.nan
        else:
            attrs = dict(h['observations/ee_wrench'].attrs)
            values = h['observations/ee_wrench'][:, :3]
            del h['observations/ee_wrench']
            h.create_dataset('observations/ee_wrench', data=values).attrs.update(attrs)
    with pytest.raises(ValueError):
        RawEpisode(path, config())


def test_sensor_three_and_six_share_anchors_masks_normalization_and_timestamps(tmp_path):
    helper().episode(tmp_path)
    three, s3, _ = dataset(tmp_path)
    six, s6, _ = helper().dataset(tmp_path)
    assert s3.force_mean == s6.force_mean[:3] and s3.force_std == s6.force_std[:3]
    for name in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std'):
        assert getattr(s3, name) == getattr(s6, name)
    assert three.index == six.index
    assert index_manifest(three, fixed_indices(three, 3), 3) == index_manifest(six, fixed_indices(six, 3), 3)
    for index in (0, 10, len(three)-1):
        a, b = three[index], six[index]
        for field in fields(a):
            x, y = getattr(a, field.name), getattr(b, field.name)
            if torch.is_tensor(x):
                if field.name in ('online_force_intervals', 'future_force_intervals', 'future_force_target'):
                    y = y[..., :3]
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        trace = three.sample_trace(index)
        assert trace['future_force'] == six.sample_trace(index)['future_force']
        assert trace['force_channel_indices'] == [0, 1, 2]
        assert trace['force_representation'] == 'end_effector_force3'
        assert 'force_conversion' not in trace and 'torque_unit' not in trace


@pytest.mark.parametrize('kind', ['contact', 'motion'])
def test_unused_moments_cannot_affect_samples_latent_predictions_or_losses(tmp_path, kind):
    path = helper().episode(tmp_path)
    ds, stats, mc = dataset(tmp_path, kind)
    before = collate_high_rate_samples([ds[10], ds[-1]])
    model = build_model(kind, vars(mc)).eval()
    settings = RealRobotOptimizationConfig(force_loss_weight=.5, high_rate_force_loss_weight=.25)
    criterion = build_criterion(kind, settings)
    outputs, losses = forward_training(kind, model, criterion, before, sample_posterior=False)
    with torch.no_grad():
        online_before = deployment_predictions(kind, model, before)
    with h5py.File(path, 'r+') as h:
        h['observations/ee_wrench'][:, 3:] = np.array([np.nan, np.inf, -1e30])
        h['observations/joint_external_torque'][:] = np.nan
        h['timestamps/disturbance/log_ns'][:] = 0
    after_ds = RealRobotHighRateDataset(tmp_path, ds.episodes, stats, mc, config())
    after = collate_high_rate_samples([after_ds[10], after_ds[-1]])
    assert compute_real_robot_normalization_stats(tmp_path, ds.episodes, config()) == stats
    for name, value in vars(before).items():
        if torch.is_tensor(value): torch.testing.assert_close(value, getattr(after, name), rtol=0, atol=0)
    changed, changed_losses = forward_training(kind, model, criterion, after, sample_posterior=False)
    with torch.no_grad():
        online_after = deployment_predictions(kind, model, after)
    for original, modified in ((outputs, changed), (losses, changed_losses), (online_before, online_after)):
        for key, value in original.items():
            if torch.is_tensor(value): torch.testing.assert_close(value, modified[key], rtol=0, atol=0)
    assert outputs['pred_action'].shape == (2, 20, 7)
    assert outputs['pred_force'].shape == (2, 20, 3)
    assert outputs['pred_force_highrate'].shape == (2, 20, 8, 3)
    expected = (losses['loss_action'] + .5 * losses['loss_force'] + .25 * losses['loss_force_highrate']
                + 10 * losses['loss_posterior_kl'] + losses.get('loss_prior_match', 0))
    torch.testing.assert_close(losses['loss_total'], expected)
    losses['loss_total'].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.high_rate_force_encoder.input_projection.in_features == 3


def test_future_changes_leave_history_untouched_and_train_stats_exclude_holdout(tmp_path):
    path = helper().episode(tmp_path, 'train')
    helper().episode(tmp_path, 'validation', offset=1e6)
    cfg = config()
    train = [r for r in discover_real_robot_episodes(tmp_path, cfg) if r.episode_id == 'train']
    stats = compute_real_robot_normalization_stats(tmp_path, train, cfg)
    mc = make_model_config('contact', cfg, compact=True, chunk_len=20, pretrained=False,
                           force_prediction_steps=20, strict_force_horizon=True)
    ds = RealRobotHighRateDataset(tmp_path, train, stats, mc, cfg)
    index = ds.index.index((0, 10))
    sample = ds[index]
    trace = ds.sample_trace(index)
    anchor = trace['observation']['anchor_ns']
    with h5py.File(tmp_path/'validation/episode.hdf5', 'r+') as h:
        h['observations/ee_wrench'][:] = -1e12
    assert compute_real_robot_normalization_stats(tmp_path, train, cfg) == stats
    with h5py.File(path, 'r+') as h:
        times = h['timestamps/ft_sensor/log_ns'][...]
        values = h['observations/ee_wrench'][...]
        values[times > anchor, :3] += 100
        h['observations/ee_wrench'][:] = values
    changed = RealRobotHighRateDataset(tmp_path, train, stats, mc, cfg)[index]
    torch.testing.assert_close(sample.online_force_intervals, changed.online_force_intervals, rtol=0, atol=0)
    assert not torch.equal(sample.future_force_intervals, changed.future_force_intervals)
    indices = np.asarray(trace['future_force']['hdf5_indices'])
    actual_times = times[indices[indices >= 0]]
    assert (actual_times > anchor).all() and (actual_times <= anchor + 200000000).all()


def test_sensor_force_metrics_and_episode_aggregation_keep_source_and_newtons(tmp_path, monkeypatch):
    from force_aware_act.real_robot import training
    helper().episode(tmp_path)
    ds, stats, mc = dataset(tmp_path)
    def predict(kind, model, batch):
        offset = batch.qpos.new_tensor([1., 2., 3.]) / batch.qpos.new_tensor(stats.force_std)
        return {'pred_action': batch.action_chunk,
                'pred_force': batch.future_force_target + offset,
                'pred_force_highrate': batch.future_force_intervals + offset}
    monkeypatch.setattr(training, 'deployment_predictions', predict)
    model = torch.nn.Identity(); model.config = mc
    result = evaluate_episodes('contact', model, ds, fixed_indices(ds, 10), stats, 'cpu', 2)
    assert result['force_representation'] == 'end_effector_force3'
    assert result['force_component_units'] == ['N'] * 3
    assert result['episode_macro_force_mae_n'] == pytest.approx(2., abs=1e-6)
    assert result['episode_macro_force_endpoint_mae_n'] == pytest.approx(2., abs=1e-6)
    assert result['force_mae_n_per_axis'] == pytest.approx([1., 2., 3.], abs=1e-6)
    assert not any('moment' in k or 'torque' in k or 'wrench' in k for k in result)
    one = next(iter(result['per_episode'].values()))
    with pytest.raises(ValueError, match='mixed'):
        aggregate_episode_metrics({'a': one, 'b': dict(one, force_representation='flange_force3')})


def test_sensor_force_experiment_weights_and_shared_split_are_explicit():
    repo = Path(__file__).parents[1]
    settings = json.loads((repo/'configs/experiments/1mm_x14y08_55_sensor_force3_v1/optimization.json').read_text())
    baseline = json.loads((repo/'configs/experiments/1mm_x14y08_55_wrench6_v1/optimization.json').read_text())
    assert settings['force_loss_weight'] == baseline['force_loss_weight'] / 2
    assert settings['high_rate_force_loss_weight'] == baseline['high_rate_force_loss_weight'] / 2
    for key in baseline.keys() - {'force_loss_weight', 'high_rate_force_loss_weight'}:
        assert settings[key] == baseline[key]
    launcher = (repo/'scripts/training/train_x14y08_sensor_force3.sh').read_text()
    assert 'configs/datasets/1mm_x14y08_55_wrench6_v1/split.json' in launcher


def test_act_can_evaluate_sensor_force3_without_using_force(tmp_path):
    helper().episode(tmp_path)
    ds, stats, _ = dataset(tmp_path)
    batch = collate_high_rate_samples([ds[10]])
    mc = make_model_config('act', config(), compact=True, chunk_len=20, pretrained=False)
    model = build_model('act', vars(mc)).eval()
    with torch.no_grad():
        before = deployment_predictions('act', model, batch)['pred_action']
        batch.online_force_intervals.fill_(float('nan'))
        after = deployment_predictions('act', model, batch)['pred_action']
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    result = evaluate('act', model, [batch], stats, 'cpu', force_representation='end_effector_force3')
    assert np.isfinite(result['action_mae_rad']) and 'force_mae_n' not in result
