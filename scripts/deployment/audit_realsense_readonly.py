"""Bounded RGBD reception audit. Imports do not open any hardware."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time


def summarize_frames(rows, elapsed_s):
    """Summarize reception timing; device timestamps retain their own clock domain."""
    if not rows:
        return {"frames": 0, "received_hz": 0, "status": "no_frames"}
    intervals = [(b["received_monotonic_ns"] - a["received_monotonic_ns"]) / 1e6
                 for a, b in zip(rows, rows[1:])]
    gaps = sum(max(0, b["frame_number"] - a["frame_number"] - 1)
               for a, b in zip(rows, rows[1:]))
    backwards = sum(b["frame_number"] <= a["frame_number"] for a, b in zip(rows, rows[1:]))
    def quantile(p):
        return sorted(intervals)[int((len(intervals) - 1) * p)] if intervals else None
    return {"frames": len(rows), "received_hz": len(rows) / elapsed_s,
            "frame_number_gaps": gaps, "nonincreasing_frame_numbers": backwards,
            "interval_ms_p50": quantile(.5), "interval_ms_p95": quantile(.95),
            "interval_ms_p99": quantile(.99), "interval_ms_max": quantile(1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600:
        parser.error("seconds must be in (0, 3600]")
    import numpy as np
    import pyrealsense2 as rs
    from PIL import Image
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "started_unix_ns": time.time_ns(),
              "sdk_version": importlib.metadata.version("pyrealsense2"),
              "expected": {"d455": "261822304024", "d405": "260322275961"},
              "devices": [], "streams": {}, "errors": []}
    active = []
    try:
        context = rs.context()
        for device in context.query_devices():
            report["devices"].append({name: device.get_info(getattr(rs.camera_info, name))
                                      for name in ["name", "serial_number", "firmware_version", "usb_type_descriptor"]
                                      if device.supports(getattr(rs.camera_info, name))})
        present = {d.get("serial_number") for d in report["devices"]}
        for label, serial in report["expected"].items():
            if serial not in present:
                report["streams"][label] = {"status": "missing_device", "serial": serial}
                continue
            pipeline = rs.pipeline(context)
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(config)
            # Register immediately so errors after start still stop the camera.
            entry = {"label": label, "pipeline": pipeline, "rows": [], "depth_rows": [],
                     "last_color": None, "last_depth": None, "color_duplicates": 0,
                     "depth_duplicates": 0, "first_image": False, "options": []}
            active.append(entry)
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                    entry["options"].append({"sensor": sensor.get_info(rs.camera_info.name),
                                             "auto_exposure": sensor.get_option(rs.option.enable_auto_exposure)})
            print("Started", label, serial, flush=True)
            time.sleep(3)
        start = time.monotonic()
        while active and time.monotonic() - start < args.seconds:
            for entry in active:
                frames = entry["pipeline"].poll_for_frames()
                if not frames:
                    continue
                received = time.monotonic_ns()
                for kind, frame, key in [("color", frames.get_color_frame(), "rows"),
                                         ("depth", frames.get_depth_frame(), "depth_rows")]:
                    if not frame:
                        continue
                    number = frame.get_frame_number()
                    if number == entry["last_" + kind]:
                        entry[kind + "_duplicates"] += 1
                        continue
                    entry["last_" + kind] = number
                    row = {"received_monotonic_ns": received, "received_unix_ns": time.time_ns(),
                           "device_timestamp_ms": frame.get_timestamp(),
                           "timestamp_domain": str(frame.get_frame_timestamp_domain()),
                           "frame_number": number}
                    entry[key].append(row)
                    if kind == "color" and not entry["first_image"]:
                        pixels = np.asanyarray(frame.get_data()).copy()
                        row["shape"] = list(pixels.shape)
                        row["dtype"] = str(pixels.dtype)
                        Image.fromarray(pixels).save(args.output / (entry["label"] + "_first_rgb.png"))
                        entry["first_image"] = True
            time.sleep(.001)
        elapsed = max(time.monotonic() - start, .001)
        for entry in active:
            summary = summarize_frames(entry["rows"], elapsed)
            summary["depth"] = summarize_frames(entry["depth_rows"], elapsed)
            summary["color_duplicate_frames"] = entry["color_duplicates"]
            summary["depth_duplicate_frames"] = entry["depth_duplicates"]
            summary["sensor_options"] = entry["options"]
            summary["first_frame"] = entry["rows"][0] if entry["rows"] else None
            summary["last_frame"] = entry["rows"][-1] if entry["rows"] else None
            report["streams"][entry["label"]] = summary
            with (args.output / (entry["label"] + "_timing.jsonl")).open("w") as handle:
                for row in entry["rows"]:
                    handle.write(json.dumps(row) + "\n")
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        for entry in reversed(active):
            try:
                entry["pipeline"].stop()
            except Exception as exc:
                report["errors"].append({"stage": "stop", "message": str(exc)})
        (args.output / "camera_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return int(bool(report["errors"]) or len(active) != 2
               or any(not e["rows"] or not e["depth_rows"] for e in active))


if __name__ == "__main__":
    raise SystemExit(main())
