"""Explicit checkpoint contract for sensor models; no hardware imports."""
import hashlib
import json
from pathlib import Path


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def adapter_contract(adapter):
    config = adapter.data_config
    if not config.uses_wrench_sensor:
        raise ValueError('sensor deployment requires a sensor-force training contract')
    if config.action_rate_hz != 100 or tuple(config.camera_names) != ('d455', 'd405'):
        raise ValueError('sensor deployment requires 100 Hz actions and d455/d405 RGB')
    if config.wrench_frame != 'right_ft_measurement':
        raise ValueError('sensor deployment requires right_ft_measurement frame')
    if not 2 <= adapter.model.config.chunk_len <= 100:
        raise ValueError('supported action horizon is 2..100 points')
    return dict(model_kind=adapter.model_kind, force_representation=config.force_representation,
                force_dim=config.force_dim, force_frame=config.wrench_frame,
                action_chunk_len=adapter.model.config.chunk_len, data_contract=json.loads(json.dumps(config.contract_dict())))


def validate_manifest(manifest):
    if manifest.get('schema_version') != 'sensor_deployment_model_v1':
        raise ValueError('unsupported sensor model manifest')
    if manifest.get('model_kind') not in ('contact', 'motion', 'dual-zero', 'act'):
        raise ValueError('unsupported sensor model kind')
    digest = manifest.get('checkpoint_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid checkpoint digest')
    return manifest
