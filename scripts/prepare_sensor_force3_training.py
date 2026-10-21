#!/usr/bin/env python3
"""Record a force-only view of the reviewed raw_v3 experiment without editing HDF5."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from force_aware_act.real_robot.config import RealRobotDataConfig, WRENCH_UNITS
from force_aware_act.real_robot.normalization import compute_real_robot_normalization_stats
from force_aware_act.real_robot.reader import RawEpisode, discover_real_robot_episodes
from force_aware_act.real_robot.sampling import pack_torque_history, pack_future_torque


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify_alignment(force3, wrench6):
    """Compare complete time/anchor mappings and three representative packed windows."""
    for name in ('anchor_timestamps_ns', 'action_slot_indices', 'valid_anchor_indices', 'torque_timestamps_ns'):
        np.testing.assert_array_equal(getattr(force3, name), getattr(wrench6, name))
    np.testing.assert_array_equal(force3.force, wrench6.force[:, :3])
    probes = sorted({int(force3.valid_anchor_indices[i]) for i in (0, len(force3.valid_anchor_indices)//2, -1)})
    for grid in probes:
        anchor = int(force3.anchor_timestamps_ns[grid])
        for pack in (
            lambda raw: pack_torque_history(raw.torque_timestamps_ns, raw.force, anchor, raw.config),
            lambda raw: pack_future_torque(raw.torque_timestamps_ns, raw.force, anchor, 20, raw.config, strict_horizon=True),
        ):
            three, six = pack(force3), pack(wrench6)
            for name in ('relative_times', 'sample_padding_mask', 'interval_padding_mask'):
                np.testing.assert_array_equal(getattr(three, name), getattr(six, name))
            np.testing.assert_array_equal(three.values, six.values[..., :3])
    return {'all_timestamps_anchors_and_action_indices_equal': True, 'native_force_values_equal': True,
            'packed_window_probe_grid_indices': probes, 'probe_values_times_and_masks_equal': True,
            'valid_anchor_count': len(force3.valid_anchor_indices),
            'valid_anchor_ns_sha256': hashlib.sha256(force3.valid_anchor_ns.tobytes()).hexdigest(),
            'action_grid_phase': force3.diagnostics['action_grid']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--data-config', type=Path, default=REPO/'configs/real_robot/right_arm_sensor_force3.json')
    parser.add_argument('--split', type=Path, default=REPO/'configs/datasets/1mm_x14y08_55_wrench6_v1/split.json')
    parser.add_argument('--reference-normalization', type=Path,
                        default=REPO/'configs/experiments/1mm_x14y08_55_wrench6_v1/normalization.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if any((args.output_dir/name).exists() for name in ('preparation.json', 'normalization.json')):
        raise ValueError('choose new preparation outputs; existing records are preserved')
    cfg = RealRobotDataConfig.from_dict(json.loads(args.data_config.read_text()))
    if cfg.force_representation != 'end_effector_force3':
        raise ValueError('preparation requires end_effector_force3')
    split = json.loads(args.split.read_text())
    roles = {role: split[role + '_episode_ids'] for role in ('train', 'validation', 'test')}
    all_ids = sum(roles.values(), [])
    if any(not ids for ids in roles.values()) or len(all_ids) != len(set(all_ids)):
        raise ValueError('split must contain disjoint nonempty train/validation/test recordings')
    records = discover_real_robot_episodes(args.data_root, cfg)
    if set(all_ids) != {r.episode_id for r in records}:
        raise ValueError('dataset recordings differ from the reviewed split')
    if len(set(split['source_groups'][key] for key in all_ids)) != len(all_ids):
        raise ValueError('source recordings must not be duplicated across the split')
    identity = {r.episode_id: {'bytes': r.resolve(args.data_root).stat().st_size,
                              'mtime_ns': r.resolve(args.data_root).stat().st_mtime_ns} for r in records}
    stats = compute_real_robot_normalization_stats(args.data_root,
                [r for r in records if r.episode_id in roles['train']], cfg)
    actual = json.loads(json.dumps(stats.to_dict()))
    reference = json.loads(args.reference_normalization.read_text())
    projected = {key: value[:3] if key in ('force_mean', 'force_std') else value for key, value in reference.items()}
    if actual != projected:
        raise ValueError('normalization differs from the reviewed wrench6 force channels')
    # The comparison view differs only in channel selection; selected floors stay identical.
    cfg6 = replace(cfg, force_representation='end_effector_wrench6', wrench_component_units=WRENCH_UNITS,
                   force_std_floor=tuple(cfg.force_std_floor) + (.001,) * 3)
    episodes = {}
    for record in records:
        path = record.resolve(args.data_root)
        three, six = RawEpisode(path, cfg), RawEpisode(path, cfg6)
        source = three.provenance['source_sha256']
        if source != split['source_groups'][record.episode_id]:
            raise ValueError('source recording identity differs from the reviewed split: ' + record.episode_id)
        episodes[record.episode_id] = {
            'role': next(role for role, ids in roles.items() if record.episode_id in ids),
            'relative_hdf5_path': record.relative_hdf5_path, 'source_mcap_sha256': source,
            **verify_alignment(three, six)}
        current = path.stat()
        if identity[record.episode_id] != {'bytes': current.st_size, 'mtime_ns': current.st_mtime_ns}:
            raise RuntimeError('source changed during preparation: ' + str(path))
    report = {'version': 'sensor_force3_training_preparation_v1', 'data_root': str(args.data_root.resolve()),
              'data_contract': cfg.contract_dict(), 'split_reference': str(args.split),
              'split_sha256': hashlib.sha256(args.split.read_bytes()).hexdigest(),
              'split_counts': {role: len(ids) for role, ids in roles.items()},
              'reference_normalization': str(args.reference_normalization),
              'normalization_matches_wrench6_force_channels': True,
              'normalization_source': 'training recordings only; native selected samples once',
              'normalization': stats.to_dict(), 'episodes': episodes, 'source_identity': identity,
              'dataset_modified': False, 'formal_training_started': False}
    write_json(args.output_dir/'normalization.json', stats.to_dict())
    write_json(args.output_dir/'preparation.json', report)
    print(json.dumps({'split_counts': report['split_counts'], 'recordings_verified': len(episodes),
                      'force_mean': stats.force_mean, 'force_std': stats.force_std,
                      'normalization_matches_wrench6_force_channels': True, 'output': str(args.output_dir)}, indent=2))
    return report


if __name__ == '__main__':
    main()
