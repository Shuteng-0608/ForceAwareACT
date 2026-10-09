"""Postprocess bounded reception audits; no hardware or model calls."""
import argparse
import bisect
import json
from pathlib import Path


def causal_age_audit(timestamps, max_age_ms=100, hz=100):
    """Use past reception timestamps only, excluding the first second of startup."""
    if len(timestamps) < 2:
        return {"status": "insufficient_data"}
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        return {"status": "invalid_timestamp_order"}
    start, end = timestamps[0] + 1_000_000_000, timestamps[-1]
    ages = []
    for anchor in range(start, end + 1, round(1e9 / hz)):
        index = bisect.bisect_right(timestamps, anchor) - 1
        ages.append((anchor - timestamps[index]) / 1e6)
    if not ages:
        return {"status": "insufficient_duration"}
    ordered = sorted(ages)
    valid = sum(age <= max_age_ms for age in ages)
    return {"status": "measured", "grid_hz": hz, "age_limit_ms": max_age_ms,
            "anchors": len(ages), "valid_anchors": valid,
            "valid_fraction": valid / len(ages), "age_ms_p99": ordered[int((len(ordered)-1)*.99)],
            "age_ms_max": ordered[-1], "startup_excluded_s": 1,
            "meaning": "host reception age; not independently measured sensor latency"}


def read_json_lines(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith('{'):
            rows.append(json.loads(line))
    return rows


def summarize(root, camera_dir="camera_2min", robot_file="robot_10min.jsonl"):
    camera = json.loads((root / camera_dir / "camera_report.json").read_text())
    robot_rows = read_json_lines(root / robot_file)
    report = {"camera": camera, "robot": next((r for r in reversed(robot_rows) if r.get("kind") == "summary"), None),
              "robot_first_feedback": next((r for r in robot_rows if r.get("kind") == "feedback"), None),
              "camera_causal_availability": {},
              "external_force_calibration": "unperformed; raw estimates only; do not use as calibrated model input"}
    for name in ["d455", "d405"]:
        timing = root / camera_dir / (name + "_timing.jsonl")
        if timing.exists():
            rows = read_json_lines(timing)
            report["camera_causal_availability"][name] = causal_age_audit([r["received_monotonic_ns"] for r in rows])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    result = summarize(args.report_dir)
    target = args.report_dir / "combined_report.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
