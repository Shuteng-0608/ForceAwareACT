"""Compile only. Never load the SDK, connect, or change a robot parameter."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from force_aware_act.control.command_session import SessionOptions


def build(sdk_directory, output, options):
    sdk_directory, output = Path(sdk_directory).resolve(), Path(output).resolve()
    source = Path(__file__).with_name("right_arm_transport.cpp").resolve()
    sdk = sdk_directory/"libMarvinSDK.so"
    digest = hashlib.sha256(sdk.read_bytes()).hexdigest()
    if digest != options.expected_sdk_sha256:
        raise ValueError("SDK hash differs from validated deployment library; do not substitute local SDK versions")
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("build output already exists; use a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
               "-I", str(sdk_directory), str(source), str(sdk), "-Wl,-z,defs", "-o", str(output)]
    subprocess.run(command, check=True)
    # Refuse SDKs with a SONAME that would replace the absolute dependency.
    dynamic = subprocess.run(["readelf", "-d", str(output)], check=True, text=True, capture_output=True).stdout
    if f"[{sdk}]" not in dynamic:
        raise ValueError("linker did not pin an absolute SDK dependency; output is not usable")
    manifest = {"abi": 2, "sdk_path": str(sdk), "sdk_sha256": digest,
                "headers_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(sdk_directory.glob("*.h"))},
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "shim_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "command": command,
                "hardware_connected": False}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session-config", type=Path, default=Path("configs/deployment/command_session.json"))
    args = parser.parse_args(argv)
    print(json.dumps(build(args.sdk_directory, args.output, SessionOptions.read(args.session_config)), indent=2))


if __name__ == "__main__":
    main()
