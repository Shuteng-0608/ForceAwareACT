"""Explicitly opened SDK transport. Imports and default construction do not connect."""
import ctypes
import hashlib
import ipaddress
import json
from pathlib import Path
import time

from .contracts import vector


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_feedback(values, ns):
    result = dict(zip(("frame_serial", "state", "error_code", "impedance_type", "velocity_ratio_percent",
                       "acceleration_ratio_percent", "low_speed_flag"), map(int, values[:7])))
    for name, start, size in (("qpos_rad", 7, 7), ("qvel_rad_s", 14, 7),
                              ("internal_torque_nm", 21, 7), ("external_raw_nm", 28, 7),
                              ("joint_stiffness_nm_per_degree", 35, 7), ("joint_damping_ratio", 42, 7),
                              ("tool_kine", 49, 6), ("tool_dyn", 55, 10), ("controller_target_rad", 65, 7)):
        result[name] = list(values[start:start+size])
    if len(values) == 79:
        result["ft_channel"] = values[72]
        result["ft_raw_counts"] = list(values[73:79])
    result["received_monotonic_ns"] = ns
    return result


class RightArmTransport:
    hardware = True

    def __init__(self, shim, sdk, options, *, enable_control=False):
        if enable_control is not True or options.allow_hardware is not True:
            raise ValueError("hardware transport disabled; config and explicit flag are both required")
        shim, sdk = Path(shim).resolve(), Path(sdk).resolve()
        manifest = json.loads(shim.with_suffix(".json").read_text())
        if (manifest.get("read_only", False) or manifest["abi"] not in (2, 3) or manifest["shim_sha256"] != sha256(shim)
                or manifest["sdk_sha256"] != sha256(sdk)
                or manifest["sdk_sha256"] != options.expected_sdk_sha256
                or manifest["sdk_path"] != str(sdk)):
            raise ValueError("transport/SDK build provenance mismatch; rebuild against the validated SDK")
        self.sensor_enabled = bool(options.ft_calibration_file)
        if self.sensor_enabled and manifest["abi"] != 3:
            raise ValueError("sensor FT requires a transport built with --sensor-ft")
        # Build uses an absolute dependency on this exact SDK, not LD_LIBRARY_PATH.
        self.lib = ctypes.CDLL(str(shim))
        self.lib.fa_abi_version.restype = ctypes.c_int
        self.lib.fa_sdk_version.restype = ctypes.c_long
        if self.lib.fa_abi_version() != manifest["abi"] or self.lib.fa_sdk_version() != options.expected_sdk_version:
            raise ValueError("SDK/transport ABI version mismatch")
        self.lib.fa_last_packet_detail.argtypes = []
        self.lib.fa_last_packet_detail.restype = ctypes.c_char_p
        self.last_submission_detail = None
        ptr = ctypes.POINTER(ctypes.c_double)
        signatures = {"fa_open": ([ctypes.c_int]*5, ctypes.c_int),
                      "fa_read": ([ptr, ctypes.c_int], ctypes.c_int),
                      "fa_configure": ([ctypes.c_int, ctypes.c_int, ptr, ptr], ctypes.c_int),
                      "fa_enter_impedance": ([], ctypes.c_int), "fa_target": ([ptr, ctypes.c_longlong], ctypes.c_int),
                      "fa_revoke_targets": ([], None), "fa_servo_off": ([], ctypes.c_int),
                      "fa_close": ([], ctypes.c_int)}
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = result
        if self.sensor_enabled:
            self.lib.fa_select_ft_telemetry.argtypes = []
            self.lib.fa_select_ft_telemetry.restype = ctypes.c_int
        self.connected = False

    def open(self, address):
        parts = tuple(ipaddress.IPv4Address(address).packed)
        if self.lib.fa_open(*parts, 1) != 1:
            raise RuntimeError("SDK connection failed; no recovery attempted")
        self.connected = True
        if self.sensor_enabled:
            try:
                if self.lib.fa_select_ft_telemetry() != 1:
                    raise RuntimeError("cannot select FT telemetry channel 216")
                deadline, previous, good = time.monotonic()+2., None, 0
                while time.monotonic() < deadline:
                    row = self.read()
                    if row["frame_serial"] != previous:
                        previous = row["frame_serial"]
                        good = good+1 if row["ft_channel"] == 216 else 0
                        if good >= 2:
                            break
                    time.sleep(.002)
                else:
                    raise RuntimeError("FT telemetry channel 216 was not confirmed")
            except Exception:
                self.close()
                raise

    def read(self):
        length = 79 if self.sensor_enabled else 72
        values = (ctypes.c_double*length)()
        if self.lib.fa_read(values, length) != 1:
            raise RuntimeError("SDK feedback read failed")
        ns = time.monotonic_ns()
        return decode_feedback(values, ns)

    @staticmethod
    def _result(code):
        return {1: "submitted", 0: "busy", 2: "expired", -1: "error"}.get(code, "error")

    def _packet_result(self, code):
        result = self._result(code)
        self.last_submission_detail = {"result": result,
            "detail": self.lib.fa_last_packet_detail().decode("utf-8", errors="replace")}
        return result

    def configure(self, baseline):
        k = (ctypes.c_double*7)(*vector(baseline["joint_stiffness_nm_per_degree"], "K", positive=True))
        d = (ctypes.c_double*7)(*vector(baseline["joint_damping_ratio"], "D", positive=True))
        return self._packet_result(self.lib.fa_configure(baseline["velocity_ratio_percent"], baseline["acceleration_ratio_percent"], k, d))

    def enter_impedance(self):
        return self._packet_result(self.lib.fa_enter_impedance())

    def target(self, q_rad, *, deadline_ns):
        values = (ctypes.c_double*7)(*vector(q_rad, "target"))
        result = self._packet_result(self.lib.fa_target(values, deadline_ns))
        self.last_submit_finished_ns = time.monotonic_ns()
        return result

    def revoke_targets(self):
        self.lib.fa_revoke_targets()  # Local flag only, no SDK command.

    def servo_off(self):
        return self._packet_result(self.lib.fa_servo_off())

    def close(self):
        result = self.lib.fa_close()
        self.connected = False
        if result != 1:
            raise RuntimeError("SDK release failed; physical state remains unconfirmed")
