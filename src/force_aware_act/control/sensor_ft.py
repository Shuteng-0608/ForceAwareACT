"""Same-frame sensor compensation using the vendored acquisition C++ core."""
import ctypes as C
import hashlib
from pathlib import Path
import time

import numpy as np
import yaml

from .kinematics import MODEL_SHA256, right_flange_transform

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "third_party/tianji_ft/ccs_m6_40.MvKDCfg"


def load_calibration(path):
    """Read this boot's parameters. Calibration hashes are intentionally not pinned."""
    d = yaml.safe_load(Path(path).read_text())
    if not isinstance(d, dict) or d.get("validation_passed") is not True:
        raise ValueError("a successfully validated calibration is required")
    if d.get("schema_version") != 1 or d.get("kind") != "real_telemetry":
        raise ValueError("unsupported calibration format")
    if d.get("model_sha256") != MODEL_SHA256:
        raise ValueError("calibration FK model differs from deployment")
    if d["sensor"]["frame_id"] != "right_ft_measurement":
        raise ValueError("calibration must use right_ft_measurement")
    return d


class SensorFT:
    def __init__(self, calibration_file, library, *, window_s=5., gap_s=.2):
        self.calibration_file = str(Path(calibration_file).resolve())
        before = Path(self.calibration_file).read_bytes()
        self.parameters = load_calibration(self.calibration_file)
        if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("bundled FK model differs from deployment geometry")
        self.lib = C.CDLL(str(Path(library).resolve()))
        ptr = C.POINTER(C.c_double)
        signatures = {
            "ft_error": ([], C.c_char_p),
            "ft_create": ([C.c_char_p, C.c_char_p, C.c_double, C.c_double], C.c_void_p),
            "ft_destroy": ([C.c_void_p], None), "ft_reset": ([C.c_void_p], None),
            "ft_process": ([C.c_void_p, ptr, ptr, ptr, ptr], C.c_int),
            "ft_observe": ([C.c_void_p, C.c_double, C.c_int, C.c_longlong, C.c_int,
                            C.c_int, ptr, ptr, ptr], None),
            "ft_capture": ([C.c_void_p, C.c_double, C.c_longlong, C.c_char_p,
                            ptr, ptr, ptr], C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes, fn.restype = args, result
        self.handle = self.lib.ft_create(self.calibration_file.encode(), str(MODEL).encode(), window_s, gap_s)
        if not self.handle:
            raise ValueError(self.lib.ft_error().decode())
        if before != Path(self.calibration_file).read_bytes():
            self.close()
            raise ValueError("calibration file changed while loading; restart preparation")
        self.tare = None
        self.waiting_reason = "waiting for fresh stationary sensor samples"
        self.last = None

    @staticmethod
    def _array(value, n):
        a = np.asarray(value, dtype=float)
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError(f"expected finite [{n}] data")
        return (C.c_double * n)(*a)

    def process(self, row, *, collect=False):
        started = time.perf_counter_ns()
        if row.get("ft_channel") != 216:
            raise ValueError("FT telemetry channel must be 216")
        measured = np.asarray(row["ft_raw_counts"], dtype=float) * .0001
        rotation = right_flange_transform(row["qpos_rad"])[:3, :3]
        compensated, corrected = (C.c_double * 6)(), (C.c_double * 6)()
        status = self.lib.ft_process(self.handle, self._array(measured, 6),
                    self._array(rotation.ravel(), 9), compensated, corrected)
        if status < 0:
            raise ValueError(self.lib.ft_error().decode())
        ns = row["received_monotonic_ns"]
        if collect:
            self.lib.ft_observe(self.handle, ns/1e9, row["frame_serial"], ns, row["state"],
                                row["low_speed_flag"], self._array(np.rad2deg(row["qpos_rad"]), 7),
                                self._array(np.rad2deg(row["qvel_rad_s"]), 7), compensated)
        self.last = {"frame_id": "right_ft_measurement", "reference_point": "sensor",
                     "frame_serial": row["frame_serial"], "source_ns": ns,
                     "measured_si": measured.tolist(), "compensated_si": list(compensated),
                     "wrench_si": list(corrected) if status == 1 else None,
                     "valid": status == 1, "gravity_compensation": "applied_upstream",
                     "sensor_to_flange_rotation": self.parameters["sensor"]["R_flange_sensor"],
                     "processing_ms": (time.perf_counter_ns()-started)/1e6}
        row["sensor_ft"] = self.last
        return self.last

    def capture(self, ns, directory):
        if self.tare is not None:
            return True
        bias, std, size = (C.c_double * 6)(), (C.c_double * 6)(), (C.c_double * 2)()
        result = self.lib.ft_capture(self.handle, ns/1e9, ns, str(Path(directory).resolve()).encode(), bias, std, size)
        if result < 0:
            raise ValueError(self.lib.ft_error().decode())
        if not result:
            self.waiting_reason = self.lib.ft_error().decode()
        if result:
            self.waiting_reason = None
            self.tare = {"bias_N_Nm": list(bias), "std_N_Nm": list(std),
                         "samples": int(size[0]), "span_s": size[1], "captured_ns": ns}
        return bool(result)

    def report(self):
        return {"calibration_file": self.calibration_file,
                "parameters": self.parameters["parameters"], "sensor": self.parameters["sensor"],
                "gravity_base_m_s2": self.parameters["gravity_base_m_s2"], "tare": self.tare,
                "waiting_reason": self.waiting_reason, "timestamp_domain": "host_monotonic_ns",
                "input_source": "SDK right arm channel 216",
                "tare_record_timestamp_note": "reused core ros_stamp_ns field contains host monotonic ns here"}

    def close(self):
        if getattr(self, "handle", None):
            self.lib.ft_destroy(self.handle); self.handle = None
