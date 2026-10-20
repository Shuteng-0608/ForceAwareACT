#!/usr/bin/env python3
"""Convert reviewed local MCAP recordings to sensor-first raw_v3, without training preparation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import cv2

from convert_mcap_right_arm import (SENSOR_SCHEMA, FT_SENSOR, FT_AXES, FT_DATASET,
                                    convert_verified, sha256)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reviewed_sources(source_root, review, expected_episodes):
    """Bind current files to the successful review entries, excluding held records."""
    if source_root.resolve() != Path(review["source_root"]).resolve():
        raise ValueError("review source directory differs from requested source")
    if review.get("status") != "inspection_complete":
        raise ValueError("review must be complete")
    selected = [r for r in review["episodes"]
                if r.get("disposition") == "successful_demonstration_candidate" and r.get("task_success") is True]
    ids = [r["episode_id"] for r in selected]
    if len(selected) != expected_episodes or len(set(ids)) != len(ids):
        raise ValueError("reviewed unique recording count differs from expected count")
    for row in selected:
        if (Path(row["episode_id"]).name != row["episode_id"]
                or row["episode_id"] in (".", "..")
                or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])):
            raise ValueError("invalid reviewed identity")
    if len({r["source_sha256"] for r in selected}) != len(selected):
        raise ValueError("duplicate source contents in review")
    if {p.stem for p in source_root.glob("*.mcap")} != set(ids):
        raise ValueError("current MCAP inventory differs from reviewed successful candidates")
    return sorted(selected, key=lambda row: row["episode_id"])


def convert_one(source_root, output_root, row, gravity_compensated, provenance):
    cv2.setNumThreads(1)
    source = Path(source_root) / (row["episode_id"] + ".mcap")
    directory = Path(output_root) / row["episode_id"]
    directory.mkdir()
    output = directory / "episode.hdf5"
    with (directory / "conversion.log").open("x") as log, redirect_stdout(log):
        report = convert_verified(source, output, include_ft_sensor=True,
            ft_gravity_compensated=gravity_compensated, ft_semantics_provenance=provenance,
            expected_source_sha256=row["source_sha256"])
    return {"episode_id": row["episode_id"], "task_success": True,
            "review_notes": row.get("notes", []), "source": str(source),
            "source_sha256": report["source_sha256"], "source_bytes": source.stat().st_size,
            "output": str(output), "output_bytes": report["output_bytes"],
            "output_sha256": sha256(output), "schema_version": report["schema_version"],
            "verification": report["verification"], "quality_report": str(directory / "quality_report.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ft-gravity-compensated", action="store_true")
    parser.add_argument("--ft-semantics-provenance", required=True)
    args = parser.parse_args(argv)
    if args.expected_episodes <= 0 or args.workers <= 0:
        raise ValueError("expected count and workers must be positive")
    source, output, review_path = args.source_root.resolve(), args.output_root.resolve(), args.review_manifest.resolve()
    review = json.loads(review_path.read_text())
    rows = reviewed_sources(source, review, args.expected_episodes)
    if output.exists():
        raise FileExistsError("output root already exists; preserve previous versions")
    output.mkdir(parents=True)
    provenance_dir = output / "provenance"
    provenance_dir.mkdir()
    shutil.copyfile(review_path, provenance_dir / "source_review_manifest.json")
    code_hashes = {}
    for relative in ("scripts/convert_mcap_right_arm.py", "scripts/convert_reviewed_mcap_dataset.py"):
        path = REPO / relative
        shutil.copyfile(path, provenance_dir / path.name)
        code_hashes[relative] = sha256(path)
    manifest = {"version": "reviewed_mcap_sensor_conversion_v1", "status": "converting",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(source), "output_root": str(output),
        "review_manifest": str(review_path), "review_manifest_sha256": sha256(review_path),
        "raw_schema_version": SENSOR_SCHEMA, "expected_episodes": len(rows),
        "primary_force_dataset": FT_DATASET, "primary_force_topic": FT_SENSOR, "axis_order": FT_AXES,
        "gravity_compensation": "applied_upstream" if args.ft_gravity_compensated else "unverified",
        "force_semantics_provenance": args.ft_semantics_provenance,
        "processing": "native streams, canonical named axes; original float64, clocks, JPEG and RGB; no interpolation, resampling, filtering or additional gravity correction",
        "conversion_only": True, "training_ready": False, "training_started": False,
        "code_sha256": code_hashes, "episodes": []}
    write_json(output / "source_inventory.json", {"source_root": str(source), "episodes": rows})
    write_json(output / "conversion_manifest.json", manifest)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(convert_one, str(source), str(output), row,
                args.ft_gravity_compensated, args.ft_semantics_provenance) for row in rows]
            for future in as_completed(futures):
                result = future.result()
                manifest["episodes"].append(result)
                write_json(output / "conversion_manifest.json", manifest)
                print(json.dumps({"completed": len(manifest["episodes"]), "episode": result["episode_id"],
                    "ft_samples": result["verification"]["counts"]["ft_sensor"], "output_bytes": result["output_bytes"]}), flush=True)
        manifest["episodes"].sort(key=lambda row: row["episode_id"])
        counts = {}
        for row in manifest["episodes"]:
            for stream, count in row["verification"]["counts"].items():
                counts[stream] = counts.get(stream, 0) + count
        manifest.update(status="converted_verified", completed_utc=datetime.now(timezone.utc).isoformat(),
            total_output_bytes=sum(r["output_bytes"] for r in manifest["episodes"]),
            total_source_bytes=sum(r["source_bytes"] for r in manifest["episodes"]), selected_message_counts=counts)
        write_json(output / "conversion_manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "conversion_manifest.json", manifest)
        raise
    print(json.dumps({"status": manifest["status"], "episodes": len(rows),
        "selected_message_counts": counts, "bytes": manifest["total_output_bytes"], "training_started": False}), flush=True)
    return manifest


if __name__ == "__main__":
    main()
