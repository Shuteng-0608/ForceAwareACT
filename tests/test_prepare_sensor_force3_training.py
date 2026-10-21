"""Preparation preserves reviewed split and rejects source/statistics mismatches."""
import importlib.util
import json
from pathlib import Path

import pytest

from force_aware_act.real_robot.normalization import compute_real_robot_normalization_stats
from force_aware_act.real_robot.reader import discover_real_robot_episodes

_helpers_spec = importlib.util.spec_from_file_location(
    'sensor_force3_test_helpers', Path(__file__).with_name('test_real_robot_sensor_force3.py'))
_helpers = importlib.util.module_from_spec(_helpers_spec)
_helpers_spec.loader.exec_module(_helpers)
config, helper = _helpers.config, _helpers.helper


def module():
    path = Path(__file__).parents[1]/'scripts/prepare_sensor_force3_training.py'
    spec = importlib.util.spec_from_file_location('prepare_sensor_force3_test', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def fixture(tmp_path):
    root = tmp_path/'data'
    for name in ('train', 'validation', 'test'): helper().episode(root, name)
    cfg = tmp_path/'config.json'; cfg.write_text(json.dumps(config().to_dict()))
    split = tmp_path/'split.json'
    split.write_text(json.dumps({**{r+'_episode_ids': [r] for r in ('train', 'validation', 'test')},
                                 'source_groups': {r: r for r in ('train', 'validation', 'test')}}))
    records = [r for r in discover_real_robot_episodes(root, helper().config()) if r.episode_id == 'train']
    stats = compute_real_robot_normalization_stats(root, records, helper().config())
    reference = tmp_path/'reference.json'; reference.write_text(json.dumps(stats.to_dict()))
    return ['--data-root', str(root), '--data-config', str(cfg), '--split', str(split),
            '--reference-normalization', str(reference), '--output-dir', str(tmp_path/'out')]


def test_preparation_is_read_only_and_refuses_overwrite(tmp_path):
    args = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in (tmp_path/'data').rglob('*.hdf5')}
    result = module().main(args)
    assert result['split_counts'] == {'train': 1, 'validation': 1, 'test': 1}
    assert result['normalization_matches_wrench6_force_channels']
    assert result['data_contract']['force_channel_indices'] == (0, 1, 2)
    assert all(e['probe_values_times_and_masks_equal'] for e in result['episodes'].values())
    assert before == {str(p): p.read_bytes() for p in (tmp_path/'data').rglob('*.hdf5')}
    with pytest.raises(ValueError, match='existing'):
        module().main(args)


@pytest.mark.parametrize('damage', ['source', 'statistics', 'split'])
def test_preparation_rejects_mismatched_reference(tmp_path, damage):
    args = fixture(tmp_path)
    path = tmp_path/('reference.json' if damage == 'statistics' else 'split.json')
    value = json.loads(path.read_text())
    if damage == 'statistics': value['force_mean'][0] += 10
    elif damage == 'source': value['source_groups']['train'] = 'wrong'
    else: value['validation_episode_ids'] = ['train']
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): module().main(args)
    assert not (tmp_path/'out/preparation.json').exists()
