"""Independent force horizons: causal labels, full action posterior, and resume."""
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from force_aware_act.act_aligned_training.high_rate_batch import ACTAlignedHighRateBatch
from force_aware_act.models.act_aligned.config import ACTAlignedHighRateConfig, canonical_model_config
from force_aware_act.models.act_aligned.contact_latent import ACTAlignedHighRateContactPosterior
from force_aware_act.real_robot.config import RealRobotDataConfig
from force_aware_act.real_robot.data import RealRobotHighRateDataset
from force_aware_act.real_robot.reader import discover_real_robot_episodes
from force_aware_act.real_robot.normalization import compute_real_robot_normalization_stats
from force_aware_act.real_robot.sampling import pack_future_torque
from force_aware_act.real_robot.training import (
    RealRobotOptimizationConfig, build_model, build_criterion, make_model_config,
    forward_training, deployment_predictions, evaluate,
)


def _helper(filename):
    spec = importlib.util.spec_from_file_location(filename, Path(__file__).with_name(filename + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(kind='contact', force_dim=3):
    data = RealRobotDataConfig(force_representation='flange_force3' if force_dim == 3 else 'joint_torque7')
    return replace(make_model_config(kind, data, compact=True, force_prediction_steps=20),
                   image_height=32, image_width=32, dropout=0.)


def _batch(config):
    b, k, f, s, h = 1, config.chunk_len, config.force_prediction_len, 8, 20
    def intervals(n):
        mask = torch.ones(b, n, s, dtype=torch.bool)
        mask[:, :, :3] = False
        force = torch.randn(b, n, s, config.force_dim).masked_fill(mask.unsqueeze(-1), 0.)
        time = (torch.arange(1, s + 1).float() / 300).expand(b, n, s).clone()
        return force, time, mask, mask.all(-1)
    past, future = intervals(h), intervals(f)
    return ACTAlignedHighRateBatch(
        torch.randn(b, 2, 3, 32, 32), torch.randn(b, 7), torch.randn(b, k, 7),
        torch.zeros(b, k, dtype=torch.bool), *past, *future,
        future[0][:, :, 2].clone(), future[2][:, :, 2].clone())


@pytest.mark.parametrize('value', [0, -1, 101, True, 20.5])
def test_horizon_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='force_prediction_steps'):
        ACTAlignedHighRateConfig(force_prediction_steps=value)


def test_legacy_config_means_full_horizon_and_preserves_state_shapes():
    old = asdict(_config())
    old.pop('force_prediction_steps')
    legacy = ACTAlignedHighRateConfig(**old)
    assert legacy.force_prediction_len == legacy.chunk_len == 100
    explicit = replace(legacy, force_prediction_steps=100)
    assert canonical_model_config(old) == canonical_model_config(asdict(explicit))
    assert canonical_model_config(old) != canonical_model_config(asdict(_config()))
    with pytest.raises(ValueError, match='ACT has no force'):
        make_model_config('act', RealRobotDataConfig(), force_prediction_steps=20)


@pytest.mark.parametrize('kind', ['contact', 'motion', 'dual-zero'])
@pytest.mark.parametrize('force_dim', [3, 7])
def test_training_inference_and_gradients_with_short_force(kind, force_dim):
    torch.manual_seed(7)
    config = _config(kind, force_dim)
    model = build_model(kind, asdict(config)).eval()
    batch = _batch(config)
    # Missing endpoint, then entirely missing interval; payloads may be NaN.
    batch.future_force_sample_padding_mask[:, -1] = True
    batch.future_force_interval_padding_mask[:, -1] = True
    batch.future_force_target_padding_mask[:, -1] = True
    batch.future_force_intervals[:, -1] = float('nan')
    batch.future_force_target[:, -1] = float('nan')
    batch.future_force_sample_padding_mask[:, -2, 2] = True
    batch.future_force_target_padding_mask[:, -2] = True
    batch.future_force_intervals[:, -2, 2] = float('nan')
    batch.future_force_target[:, -2] = float('nan')
    criterion = build_criterion(kind, RealRobotOptimizationConfig())
    outputs, losses = forward_training(kind, model, criterion, batch, sample_posterior=False)
    assert outputs['pred_action'].shape == (1, 100, 7)
    assert outputs['pred_force'].shape == (1, 20, force_dim)
    assert outputs['pred_force_highrate'].shape == (1, 20, 8, force_dim)
    assert all(torch.isfinite(v).all() for v in losses.values())
    losses['loss_total'].backward()
    assert model.force_head.weight.grad.abs().sum() > 0
    assert model.high_rate_force_head.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    online = deployment_predictions(kind, model, batch)
    assert online['pred_action'].shape[1] == 100
    assert online['pred_force'].shape[1] == 20


def test_contact_posterior_keeps_tail_actions_and_ignores_missing_force():
    torch.manual_seed(1)
    config = _config()
    posterior = ACTAlignedHighRateContactPosterior(config).eval()
    q = torch.randn(1, 7)
    actions = torch.randn(1, 100, 7, requires_grad=True)
    force = torch.randn(1, 20, config.d_model, requires_grad=True)
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[:, 2] = True
    mean, _, _ = posterior(q, actions, force, future_force_interval_padding_mask=mask, sample=False)
    mean.square().sum().backward()
    assert actions.grad[:, 20:].abs().sum() > 0
    assert force.grad[:, :2].abs().sum() > 0
    assert force.grad[:, 2].eq(0).all()
    changed = force.detach().clone()
    changed[:, 2] = float('nan')
    second = posterior(q, actions.detach(), changed, future_force_interval_padding_mask=mask, sample=False)[0]
    torch.testing.assert_close(mean, second, rtol=0, atol=0)
    with pytest.raises(ValueError, match='future_force_tokens'):
        posterior(q, actions, torch.randn(1, 100, config.d_model))


def test_short_data_never_reads_force_after_horizon_and_keeps_action_grid(tmp_path):
    helpers = _helper('test_real_robot_data')
    path = helpers._write_raw(tmp_path)
    data_config = helpers._config()
    config = replace(helpers._model(data_config), chunk_len=100, force_prediction_steps=20)
    records = discover_real_robot_episodes(tmp_path, data_config)
    stats = compute_real_robot_normalization_stats(tmp_path, records, data_config)
    dataset = RealRobotHighRateDataset(tmp_path, records, stats, config, data_config)
    index = dataset.index.index((0, 4))
    sample = dataset[index]
    assert sample.action_chunk.shape == (100, 7)
    assert sample.future_force_intervals.shape == (20, 8, 7)
    assert sample.future_force_target_padding_mask[-1].item() is False
    # Mutate only this synthetic fixture; stats intentionally remain frozen.
    with h5py.File(path, 'r+') as h:
        times = h['timestamps/disturbance/log_ns'][:]
        cutoff = helpers.EPOCH + 40_000_000 + 200_000_000
        values = h['observations/joint_external_torque'][:]
        values[times > cutoff] += 10000
        h['observations/joint_external_torque'][:] = values
    changed = RealRobotHighRateDataset(tmp_path, records, stats, config, data_config)
    assert changed.index == dataset.index
    other = changed[index]
    for name in ('action_chunk', 'online_force_intervals', 'future_force_intervals', 'future_force_target'):
        torch.testing.assert_close(getattr(sample, name), getattr(other, name), rtol=0, atol=0)
    tail = changed[len(changed) - 1]
    assert tail.future_force_sample_padding_mask.any()
    assert changed.sample_trace(index)['future_torque']['prediction_steps'] == 20


def test_matching_tolerance_cannot_extend_short_force_horizon():
    data = RealRobotDataConfig()
    times = np.arange(1, 61, dtype=np.int64) * 3_333_333
    times[-1] = 201_000_000
    force = np.ones((60, 7), dtype=np.float32)
    legacy = pack_future_torque(times, force, 0, 20, data)
    short = pack_future_torque(times, force, 0, 20, data, strict_horizon=True)
    assert not legacy.sample_padding_mask[-1, 2]
    assert short.sample_padding_mask[-1, 2]
    empty = pack_future_torque(np.array([], dtype=np.int64), np.zeros((0, 7)), 0, 20, data, strict_horizon=True)
    assert empty.sample_padding_mask.all()


def test_evaluation_can_compare_old_model_on_same_force_prefix():
    config = replace(_config(force_dim=7), force_prediction_steps=None)
    batch = _batch(config)
    batch = replace(batch, action_chunk=torch.zeros_like(batch.action_chunk),
                    future_force_intervals=torch.zeros_like(batch.future_force_intervals),
                    future_force_target=torch.zeros_like(batch.future_force_target))
    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
        def forward(self, *args, **kwargs):
            endpoint = torch.ones_like(batch.future_force_target)
            high = torch.ones_like(batch.future_force_intervals)
            endpoint[:, 20:] = 100
            high[:, 20:] = 100
            return {'pred_action': torch.ones_like(batch.action_chunk),
                    'pred_force': endpoint, 'pred_force_highrate': high}
    stats = SimpleNamespace(action_std=[1.] * 7, force_std=[1.] * 7)
    result = evaluate('contact', FixedModel(), [batch], stats, 'cpu', force_evaluation_steps=20)
    assert result['torque_mae_nm'] == result['torque_endpoint_mae_nm'] == 1
    assert result['action_valid_scalars'] == 700
    assert result['torque_valid_scalars'] == 20 * 3 * 7
    assert result['force_evaluation_duration_s'] == .2


def test_short_cli_resume_records_horizon_and_rejects_cross_horizon(tmp_path):
    helpers = _helper('test_real_robot_cli')
    runner = helpers._runner()
    root, config = helpers._data(tmp_path)
    output = tmp_path / 'run'
    args = helpers._args(root, config, output, 1) + ['--force-prediction-steps', '1']
    runner.main(args)
    manifest = json.loads((output / 'run_manifest.json').read_text())
    assert manifest['prediction_horizons']['action_steps'] == 3
    assert manifest['prediction_horizons']['force_prediction_steps'] == 1
    resumed = helpers._args(root, config, output, 2) + ['--resume', str(output / 'last.pt')]
    result = runner.main(resumed)
    assert result['checkpoint_round_trip'] == 'passed'
    with pytest.raises(ValueError, match='model_config'):
        runner.main(helpers._args(root, config, output, 3) + ['--resume', str(output / 'last.pt'),
                                                          '--force-prediction-steps', '3'])


def test_legacy_cli_resume_accepts_missing_horizon_field(tmp_path):
    helpers = _helper('test_real_robot_cli')
    runner = helpers._runner()
    root, config = helpers._data(tmp_path)
    output = tmp_path / 'legacy'
    runner.main(helpers._args(root, config, output, 1))
    payload = torch.load(output / 'last.pt', weights_only=False)
    payload['model_config'].pop('force_prediction_steps')
    torch.save(payload, output / 'last.pt')
    manifest = json.loads((output / 'run_manifest.json').read_text())
    manifest['model_config'].pop('force_prediction_steps')
    (output / 'run_manifest.json').write_text(json.dumps(manifest))
    result = runner.main(helpers._args(root, config, output, 2) + ['--resume', str(output / 'last.pt')])
    assert result['checkpoint_round_trip'] == 'passed'
