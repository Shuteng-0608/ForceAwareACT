"""Compile only. Never load the SDK, connect, or change a robot parameter."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from force_aware_act.control.command_session import SessionOptions


def build(sdk_directory, output, options, *, sensor_ft=False, read_only=False):
    if read_only and not sensor_ft:
        raise ValueError("read-only FT build requires --sensor-ft")
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
    if sensor_ft:
        command.insert(1, "-DFA_SENSOR_FT=1")
    if read_only:
        command.insert(1, "-DFA_READ_ONLY=1")
    subprocess.run(command, check=True)
    # Refuse SDKs with a SONAME that would replace the absolute dependency.
    dynamic = subprocess.run(["readelf", "-d", str(output)], check=True, text=True, capture_output=True).stdout
    if f"[{sdk}]" not in dynamic:
        raise ValueError("linker did not pin an absolute SDK dependency; output is not usable")
    readonly_symbols = None
    if read_only:
        # Do not load the SDK to inspect it. Audit actual linked API references.
        symbols = subprocess.run(["nm", "-D", "-C", str(output)], check=True, text=True, capture_output=True).stdout
        import re
        readonly_symbols = sorted(set(re.findall(r"\b(?:On\w+|CheckSDKTypeCompat)(?=\(|\s|$)", symbols)))
        expected = ["CheckSDKTypeCompat", "OnGetBuf", "OnGetSDKVersion", "OnLinkTo", "OnRelease", "OnSetUserSpcfData_B"]
        if readonly_symbols != sorted(expected):
            raise ValueError(f"read-only SDK symbol audit failed: {readonly_symbols}")
        if any(name in symbols for name in ("fa_target", "fa_configure", "fa_enter_impedance", "fa_servo_off")):
            raise ValueError("read-only build exposes motor controls")
    manifest = {"read_only": read_only, "readonly_sdk_symbols": readonly_symbols, "abi": 3 if sensor_ft else 2, "sdk_path": str(sdk), "sdk_sha256": digest,
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
    parser.add_argument("--sensor-ft", action="store_true", help="build sensor feedback ABI 3")
    parser.add_argument("--read-only", action="store_true", help="compile out all motion/configuration/servo interfaces")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.sdk_directory, args.output, SessionOptions.read(args.session_config),
                           sensor_ft=args.sensor_ft, read_only=args.read_only), indent=2))


if __name__ == "__main__":
    main()
