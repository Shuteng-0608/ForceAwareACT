"""Environment and GPU arithmetic check. No robot connections or camera streaming."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def collect():
    report = {"python": sys.version, "executable": sys.executable,
              "system": platform.platform(), "packages": {}}
    for name in ["numpy", "torch", "torchvision", "h5py", "pyrealsense2", "pillow", "pytest"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    import torch
    import torchvision
    import h5py
    import numpy as np
    report["torch_cuda_runtime"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    if report["cuda_available"]:
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        with torch.inference_mode():
            a = torch.eye(16, device="cuda")
            b = a @ a
            torch.cuda.synchronize()
            report["gpu_arithmetic_ok"] = bool(torch.equal(a, b))
    report["numpy_tensor_bridge_ok"] = torch.from_numpy(np.ones(7, dtype=np.float32)).shape == (7,)
    report["torchvision_import_ok"] = bool(torchvision.__version__)
    report["h5py_import_ok"] = bool(h5py.__version__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(not report.get("gpu_arithmetic_ok"))


if __name__ == "__main__":
    raise SystemExit(main())
