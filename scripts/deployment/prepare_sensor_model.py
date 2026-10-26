"""Inspect a trained sensor checkpoint on CPU, and write its deployment contract.

This does not approve model performance or open any robot/camera device.
"""
import argparse
import json
from pathlib import Path

from force_aware_act.control.deployment_model import adapter_contract, checkpoint_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists')
    from force_aware_act.real_robot.checkpoint import read_checkpoint
    from force_aware_act.inference.real_robot import RealRobotPolicyAdapter
    adapter = RealRobotPolicyAdapter.from_checkpoint(read_checkpoint(args.checkpoint), device='cpu')
    report = {'schema_version': 'sensor_deployment_model_v1',
              'checkpoint_sha256': checkpoint_digest(args.checkpoint), **adapter_contract(adapter)}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
