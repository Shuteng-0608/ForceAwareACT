"""Sensor55 model-library ACT #1, using the shared guarded execution path.

Default is a device-free preview. RGB and qpos are the policy inputs; the
compensated/tared FT stream remains available for protection and the HUD.
"""
from pathlib import Path

from scripts.deployment.run_sensor55_x14y08 import controller_arguments as _arguments

ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / 'configs/deployment/sensor55_act_model_binding.json'
EXPECTED_MODEL_SHA = 'be3f8756534df3b7622db5d6dc0e86143a0d6fa036c27bfe299762b58e2c9b08'


def controller_arguments(argv=None):
    return _arguments(argv, binding_path=BINDING,
                      expected_model_sha=EXPECTED_MODEL_SHA, model_kind='act')


def main(argv=None):
    from scripts.deployment.run_act_controller import main as act_main
    return act_main(controller_arguments(argv))


if __name__ == '__main__':
    raise SystemExit(main())
