"""Strict checkpoint/GPU smoke test with explicitly synthetic force history."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None, *, model_kind='contact'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args(argv)
    digest = sha256(args.checkpoint)
    if digest != args.expected_sha256:
        raise ValueError('checkpoint SHA256 mismatch')
    import numpy as np
    import torch
    from force_aware_act.real_robot.checkpoint import read_checkpoint
    from force_aware_act.inference.real_robot import RealRobotPolicyAdapter, RealRobotObservationBuffer
    torch.set_num_threads(4)
    payload = read_checkpoint(args.checkpoint)
    adapter = RealRobotPolicyAdapter.from_checkpoint(payload, device='cuda')
    if adapter.model_kind != model_kind:
        raise ValueError(f'{model_kind} checkpoint required')
    buffer = RealRobotObservationBuffer(adapter.data_config)
    anchor = 1_000_000_000
    for timestamp in np.arange(anchor-210_000_000, anchor, 3_333_333):
        buffer.append_state(int(timestamp), np.asarray(adapter.normalization['qpos_mean']))
        buffer.append_external_torque(int(timestamp), np.asarray(adapter.normalization['force_mean']))
    for camera in adapter.data_config.camera_names:
        buffer.append_image(camera, anchor-1_000_000, np.zeros((480,640,3), dtype=np.uint8))
    inputs = adapter.prepare_observation(buffer, anchor)
    times = []
    for i in range(8):
        torch.cuda.synchronize(); start = time.perf_counter()
        result = adapter.predict(buffer, anchor, contact_latent_mode='zero')
        torch.cuda.synchronize(); elapsed = (time.perf_counter()-start)*1000
        if i >= 3:
            times.append(elapsed)
    if result['pred_action'].shape != (100, 7) or not np.isfinite(result['pred_action']).all():
        raise ValueError('model must produce finite [100,7] joint targets')
    report = {'sha256': digest, 'model_kind': adapter.model_kind,
              'global_step': payload['global_step'], 'strict_state_load': True,
              'synthetic_input_only': True, 'robot_connected': False,
              'normalization': payload['normalization'], 'model_config': payload['model_config'],
              'data_config': payload['data_config'], 'data_contract': payload['data_contract'],
              'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'input_shapes': {'images': list(inputs.images.shape), 'qpos': list(inputs.qpos.shape),
                               'force_intervals': (list(inputs.online_force_intervals.shape)
                                                   if inputs.online_force_intervals is not None else None)},
              'output_shapes': {k:list(v.shape) for k,v in result.items()},
              'warm_latency_ms': {'samples': times, 'mean':float(np.mean(times)), 'max':max(times)},
              'first_synthetic_predicted_action_rad': result['pred_action'][0].tolist(),
              'commands_sent': False}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    args.checkpoint.with_name('model_metadata.json').write_text(json.dumps({
        key: payload[key] for key in ['model_kind','global_step','model_config','data_config','data_contract','normalization']
    }, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['normalization','model_config','data_config','data_contract']}, indent=2))


if __name__ == '__main__':
    main()
