"""Independent history packing, unchanged future labels, and model compatibility."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from force_aware_act.real_robot.config import RealRobotDataConfig
from force_aware_act.real_robot.sampling import pack_torque_history, pack_future_torque
from force_aware_act.real_robot.training import (make_model_config, build_model, build_criterion,
    forward_training, deployment_predictions, RealRobotOptimizationConfig)
from force_aware_act.real_robot.data import RealRobotHighRateDataset
from force_aware_act.real_robot.reader import discover_real_robot_episodes
from force_aware_act.real_robot.normalization import compute_real_robot_normalization_stats
from force_aware_act.act_aligned_training.high_rate_data import collate_high_rate_samples
from force_aware_act.models.act_aligned.config import canonical_model_config
from force_aware_act.inference.real_robot import RealRobotPolicyAdapter, RealRobotObservation

ROOT = Path(__file__).parents[1]
ORIGIN = 1_700_000_000_000_000_000
MS = 1_000_000


def helper(name):
    spec = importlib.util.spec_from_file_location('history50_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coarse(config):
    return replace(config, history_interval_ms=50, history_max_samples_per_interval=20)


@pytest.mark.parametrize('changes', [dict(history_interval_ms=0), dict(history_interval_ms=True),
    dict(history_interval_ms=50.0), dict(history_max_samples_per_interval=False),
    dict(history_max_samples_per_interval=-1), dict(history_interval_ms=50),
    dict(history_interval_ms=30, history_max_samples_per_interval=20)])
def test_invalid_history_contract(changes):
    with pytest.raises(ValueError):
        RealRobotDataConfig(**changes)


def test_default_serialization_and_legacy_model_config_are_compatible():
    old = RealRobotDataConfig()
    assert 'history_interval_ms' not in old.to_dict()
    assert 'history_max_samples_per_interval' not in old.to_dict()
    assert old.history_intervals == 20 and old.history_sample_capacity == 8
    cfg = coarse(old)
    assert cfg.history_intervals == 4 and cfg.history_period_ns == 50 * MS
    assert RealRobotDataConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.contract_dict()['sampling_version'] != old.contract_dict()['sampling_version']
    mc = make_model_config('contact', old, compact=True)
    legacy = asdict(mc)
    legacy.pop('max_online_force_samples_per_interval')
    legacy.pop('online_force_interval_duration_s')
    assert canonical_model_config(legacy) == canonical_model_config(asdict(mc))
    model = build_model('contact', legacy, initialize_pretrained=False)
    model.load_state_dict(build_model('contact', asdict(mc), initialize_pretrained=False).state_dict())


def test_exact_boundaries_slide_at_10ms_and_never_use_future_samples():
    cfg = coarse(RealRobotDataConfig())
    offsets = np.array([-1, 0, 1, 49, 50, 51, 99, 100, 101, 149, 150, 151, 199, 200, 201])
    times = ORIGIN + offsets * MS
    values = np.repeat(offsets[:, None], 7, axis=1).astype(np.float32)
    packed = pack_torque_history(times, values, ORIGIN + 200 * MS, cfg)
    assert packed.values.shape == (4, 20, 7)
    assert packed.sample_counts.tolist() == [3, 3, 3, 3]
    np.testing.assert_array_equal(packed.values[~packed.sample_padding_mask, 0], offsets[2:-1])
    np.testing.assert_allclose(packed.relative_times[~packed.sample_padding_mask], [.001, .049, .05] * 4)
    later = pack_torque_history(times, values, ORIGIN + 210 * MS, cfg)
    np.testing.assert_array_equal(later.values[~later.sample_padding_mask, 0], offsets[(offsets > 10) & (offsets <= 210)])
    changed = values.copy(); changed[times > ORIGIN + 200 * MS] = np.nan
    after = pack_torque_history(times, changed, ORIGIN + 200 * MS, cfg)
    np.testing.assert_array_equal(after.values, packed.values)
    with pytest.raises(ValueError, match='capacity exceeded'):
        pack_torque_history(ORIGIN + np.arange(1, 22) * MS, np.ones((21, 7)), ORIGIN + 200 * MS, cfg)


def test_empty_history_bins_and_future_slots_are_independent():
    old = RealRobotDataConfig(); cfg = coarse(old)
    times = ORIGIN + np.arange(1, 181) * (MS * 10 // 3)
    values = np.ones((len(times), 7), dtype=np.float32)
    sparse = pack_torque_history(times[:1], values[:1], ORIGIN + 200 * MS, cfg)
    assert sparse.interval_padding_mask.tolist() == [False, True, True, True]
    assert (sparse.values[sparse.sample_padding_mask] == 0).all()
    for anchor in (ORIGIN + 100 * MS, ORIGIN + 200 * MS, times[-1]):
        a = pack_future_torque(times, values, anchor, 20, old, strict_horizon=True)
        b = pack_future_torque(times, values, anchor, 20, cfg, strict_horizon=True)
        for field in ('values', 'relative_times', 'sample_padding_mask', 'interval_padding_mask'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        assert b.values.shape == (20, 8, 7)


def test_auditor_counts_independent_history_bins_at_each_action_anchor():
    spec = importlib.util.spec_from_file_location('history50_audit', ROOT/'scripts/audit_mcap_right_arm_dataset.py')
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    old = RealRobotDataConfig()
    cfg = coarse(old)
    times = dict(action=ORIGIN + np.arange(101) * 10 * MS,
                 state=ORIGIN + np.arange(1001) * MS,
                 disturbance=ORIGIN + np.rint(np.arange(331) * 1e9 / 300).astype(np.int64))
    times.update({camera: ORIGIN + np.arange(35) * 30 * MS for camera in cfg.camera_names})
    before, old_indices = auditor.alignment_metrics(times, old)
    after, new_indices = auditor.alignment_metrics(times, cfg)
    for field in old_indices:
        np.testing.assert_array_equal(old_indices[field], new_indices[field])
    anchors = new_indices['candidate_anchor_log_ns'][new_indices['valid_anchor_grid_indices']]
    maximum = max(pack_torque_history(times['disturbance'], np.ones((331,7)), int(anchor), cfg)
                  .sample_counts.max() for anchor in anchors)
    assert after['max_samples_per_history_interval'] == maximum
    assert maximum > before['max_samples_per_history_interval']
    assert after['history_sample_count'] == before['history_sample_count']
    with pytest.raises(ValueError, match='capacity exceeded'):
        auditor.alignment_metrics(dict(times, disturbance=ORIGIN + np.arange(1001) * MS), cfg)


def pair(tmp_path, dimension, kind='contact'):
    fixture = helper('test_real_robot_wrench6')
    path = fixture.episode(tmp_path)
    cfg = fixture.config()
    if dimension == 3:
        cfg = replace(cfg, force_representation='end_effector_force3', wrench_component_units=('N',)*3,
                      force_std_floor=(.01,)*3)
    records = discover_real_robot_episodes(tmp_path, cfg)
    stats = compute_real_robot_normalization_stats(tmp_path, records, cfg)
    datasets = []
    for data_config in (cfg, coarse(cfg)):
        mc = make_model_config(kind, data_config, compact=True, chunk_len=20, pretrained=False,
                              force_prediction_steps=20, strict_force_horizon=True)
        datasets.append(RealRobotHighRateDataset(tmp_path, records, stats, mc, data_config))
    return datasets, stats, path


@pytest.mark.parametrize('dimension', [3, 6])
def test_dataset_preserves_anchors_raw_samples_labels_and_normalization(tmp_path, dimension):
    (old, new), stats, _ = pair(tmp_path, dimension)
    assert old.index == new.index
    assert compute_real_robot_normalization_stats(tmp_path, new.episodes, new.data_config) == stats
    for index in (0, 10, len(new)-1):
        a, b = old[index], new[index]
        for field in ('images','qpos','action_chunk','action_padding_mask','future_force_intervals',
                      'future_force_relative_time','future_force_sample_padding_mask',
                      'future_force_interval_padding_mask','future_force_target','future_force_target_padding_mask'):
            torch.testing.assert_close(getattr(a,field),getattr(b,field),rtol=0,atol=0)
        assert b.online_force_intervals.shape == (4,20,dimension)
        torch.testing.assert_close(a.online_force_intervals[~a.online_force_sample_padding_mask],
                                   b.online_force_intervals[~b.online_force_sample_padding_mask],rtol=0,atol=0)
        assert a.online_raw_valid_count == b.online_raw_valid_count
        trace = new.sample_trace(index)
        assert trace['history_interval_ns'] == 50 * MS
        assert sum(trace['history_sample_counts']) == b.online_raw_valid_count


@pytest.mark.parametrize('kind', ['contact','motion'])
@pytest.mark.parametrize('dimension', [3,6])
def test_models_gradients_snapshot_inference_and_checkpoint_identity(tmp_path, kind, dimension):
    (old, ds), stats, path = pair(tmp_path, dimension, kind)
    config = ds.model_config
    batch = collate_high_rate_samples([ds[10],ds[-1]])
    batch.validate(config)
    model = build_model(kind, asdict(config), initialize_pretrained=False)
    base_model = build_model(kind, asdict(old.model_config), initialize_pretrained=False)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in base_model.parameters())
    assert model.high_rate_force_head.weight.shape == base_model.high_rate_force_head.weight.shape
    assert model.high_rate_force_encoder.output_projection.in_features == 3 * config.local_force_dim
    assert config.force_encoder_token_count == 5 and config.contact_posterior_token_count == 22
    outputs, losses = forward_training(kind,model,build_criterion(kind,RealRobotOptimizationConfig()),batch,
                                      sample_posterior=False)
    assert outputs['pred_force_highrate'].shape == (2,20,8,dimension)
    assert outputs['pred_force'].shape == (2,20,dimension)
    assert outputs['pred_action'].shape == (2,20,7)
    assert outputs['online_force_interval_tokens'].shape == (2,4,config.d_model)
    if kind == 'contact':
        assert outputs['future_force_interval_tokens'].shape == (2,20,config.d_model)
    losses['loss_total'].backward()
    assert model.high_rate_force_encoder.input_projection.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        before = deployment_predictions(kind,model,batch)
        corrupted = replace(batch,future_force_intervals=torch.full_like(batch.future_force_intervals,float('nan')),
                            action_chunk=torch.full_like(batch.action_chunk,float('nan')))
        after = deployment_predictions(kind,model,corrupted)
    torch.testing.assert_close(before['pred_action'],after['pred_action'],rtol=0,atol=0)
    payload = dict(format_version='real_robot_training_v1',model_kind=kind,model_config=asdict(config),
                   model_state=model.state_dict(),data_config=ds.data_config.to_dict(),
                   data_contract=ds.data_config.contract_dict(),normalization=stats.to_dict())
    adapter = RealRobotPolicyAdapter.from_checkpoint(payload)
    with pytest.raises(ValueError):
        RealRobotPolicyAdapter.from_checkpoint(payload,expected_data_config=old.data_config)
    index = len(ds)-1
    trace = ds.sample_trace(index); anchor=trace['observation']['anchor_ns']
    raw=ds.raw_episodes[0]; state,_=raw.observation_indices(anchor)
    observation=RealRobotObservation(anchor_ns=anchor,images=tuple(raw.images_at(anchor)),
        image_timestamps_ns=(anchor,anchor),qpos=raw.qpos[state],state_timestamp_ns=anchor,
        torque_timestamps_ns=raw.torque_timestamps_ns,torque_values=raw.force)
    prepared=adapter.prepare_snapshot(observation)
    sample=ds[index]
    torch.testing.assert_close(prepared.online_force_intervals[0],sample.online_force_intervals,rtol=0,atol=0)
    torch.testing.assert_close(prepared.online_force_relative_time[0],sample.online_force_relative_time,rtol=0,atol=0)


def test_new_launch_config_changes_only_history_and_delegates_training_options(tmp_path):
    script=ROOT/'scripts/training/train_x14y08_history50ms.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    # Intercept the delegated bash; never execute either training launcher.
    stub=tmp_path/'bash'
    stub.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    stub.chmod(0o755)
    import os
    for representation in ('wrench6','sensor_force3'):
        old=json.loads((ROOT/f'configs/real_robot/right_arm_{representation}.json').read_text())
        new=json.loads((ROOT/f'configs/real_robot/right_arm_{representation}_history50ms.json').read_text())
        assert new == dict(old,history_interval_ms=50,history_max_samples_per_interval=20)
        for kind in ('contact','motion'):
            output=subprocess.check_output(['/bin/bash',str(script),representation,kind,'train'],text=True,
                env={**os.environ,'PATH':str(tmp_path)+os.pathsep+os.environ['PATH']})
            args=json.loads(output)
            assert args[:3] == [str(ROOT/f'scripts/training/train_x14y08_{representation}.sh'),kind,'train']
            assert args[3::2] == ['--data-config','--output-dir']
            assert 'history50ms_multimetric' in args[-1]


def test_new_grouping_cli_resume_and_rejection_of_old_grouping(tmp_path, monkeypatch, capsys):
    fixture=helper('test_real_robot_multi_metric_cli')
    runner,args=fixture.setup(tmp_path,monkeypatch,'contact',6)
    path=tmp_path/'config.json'
    cfg=json.loads(path.read_text()); cfg.update(history_interval_ms=50,history_max_samples_per_interval=20)
    path.write_text(json.dumps(cfg))
    output=tmp_path/'out'
    runner.main(args(output,2))
    runner.main(args(output,3)+['--resume',str(output/'last.pt')])
    manifest=json.loads((output/'run_manifest.json').read_text())
    assert manifest['model_config']['max_force_samples_per_interval'] == 8
    assert manifest['model_config']['max_online_force_samples_per_interval'] == 20
    assert manifest['prediction_horizons']['force_history_intervals'] == 4
    cfg.pop('history_interval_ms'); cfg.pop('history_max_samples_per_interval'); path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='another run|differs'):
        runner.main(args(output,4)+['--resume',str(output/'last.pt')])
