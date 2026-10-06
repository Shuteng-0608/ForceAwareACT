#!/usr/bin/env python3
"""Run the approved real-robot experiments sequentially and persist progress."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from force_aware_act.real_robot.config import RAW_SCHEMA_VERSION, RealRobotDataConfig


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def build_jobs(config):
    if config["models"] != ["contact", "act", "motion"]:
        raise ValueError("approved initial model list is contact, act, motion")
    jobs = [(kind, 0) for kind in config["models"]]
    jobs.extend((kind, seed) for seed in (1, 2) for kind in ("contact", "act"))
    return [{"model": kind, "seed": seed, "id": f"{kind}_seed{seed}", "status": "pending"} for kind, seed in jobs]


def training_command(config, job, output):
    per_epoch = (config["train_episodes"] + config["batch_size"] - 1) // config["batch_size"]
    return [sys.executable, "-u", str(REPO / "scripts/train_real_robot.py"),
            "--data-root", config["data_root"], "--data-config", config["data_config"],
            "--optimization-config", config["optimization_config"], "--split", config["split"],
            "--output-dir", str(output), "--mode", "train", "--model", job["model"],
            "--max-sampling-epochs", str(config["max_sampling_epochs"]),
            "--minimum-sampling-epochs", str(config["minimum_sampling_epochs"]),
            "--early-stop-patience", str(config["early_stop_patience"]),
            "--early-stop-min-relative-improvement", str(config["minimum_relative_improvement"]),
            "--validation-interval", str(per_epoch * config["validation_interval_epochs"]),
            "--checkpoint-interval", "100", "--validation-anchor-stride", str(config["validation_anchor_stride"]),
            "--selection-metric", "episode_macro_action_mae_rad", "--train-sampling-mode", "episodic_uniform",
            "--batch-size", str(config["batch_size"]), "--chunk-len", "100", "--seed", str(job["seed"]),
            "--device", "cuda", "--threads", "4", "--record-timing"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config.get("execution_status") != "approved_to_start":
        raise ValueError("formal training is not authorized to start; the user requested a test report first")
    output = Path(config["output_root"]).resolve()
    state_path = output / "queue_status.json"
    if state_path.exists():
        raise ValueError("queue already exists; inspect it before creating another run")
    conversion_path = Path(config["conversion_manifest"])
    conversion = json.loads(conversion_path.read_text())
    if conversion["status"] != "ready" or len(conversion["episodes"]) != 20:
        raise ValueError("the verified dataset is not ready")
    contract = RealRobotDataConfig.from_dict(json.loads(Path(config["data_config"]).read_text())).contract_dict()
    # JSON canonicalization normalizes tuple/list representations.
    if (conversion.get("raw_schema_version") != RAW_SCHEMA_VERSION
            or conversion.get("data_contract") != json.loads(json.dumps(contract))):
        raise ValueError("dataset must contain the corrected external disturbance source and matching sampling contract")
    split = json.loads(Path(config["split"]).read_text())
    if len(split["train_episode_ids"]) != config["train_episodes"]:
        raise ValueError("epoch conversion differs from the frozen split")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "experiment_config.json", config)
    files = list((REPO / "src/force_aware_act").rglob("*.py"))
    files.extend(REPO / "scripts" / name for name in ("train_real_robot.py", "evaluate_real_robot.py",
                                                      "prepare_1mm_gap_dataset.py", "run_real_robot_experiment_queue.py",
                                                      "convert_mcap_right_arm.py", "replay_real_robot.py"))
    fingerprints = {}
    for path in files:
        relative = path.relative_to(REPO)
        destination = output / "code_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        fingerprints[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "provenance.json", {"code_sha256": fingerprints,
               "conversion_manifest_sha256": hashlib.sha256(conversion_path.read_bytes()).hexdigest(),
               "configuration_files_sha256": {config[key]: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                                               for key in ("data_config", "optimization_config", "split")},
               "user_amendments": {"dual_zero": "excluded", "max_sampling_epochs": 20000}})
    state = {"version": "1mm_gap_experiment_queue_v1", "pid": os.getpid(), "status": "running",
             "started_utc": datetime.now(timezone.utc).isoformat(), "jobs": build_jobs(config)}
    write_json(state_path, state)
    current = None
    try:
        for job in state["jobs"]:
            current = job
            for relative, digest in fingerprints.items():
                if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"code changed during the frozen experiment: {relative}")
            run_dir = output / "runs" / job["id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            command = training_command(config, job, run_dir)
            job.update(status="training", started_utc=datetime.now(timezone.utc).isoformat(), output=str(run_dir), command=command)
            with (run_dir / "run.log").open("x") as log:
                child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                job["pid"] = child.pid
                write_json(state_path, state)
                code = child.wait()
            if code:
                raise RuntimeError(f"training {job['id']} exited with {code}; see {run_dir / 'run.log'}")
            job.update(status="evaluating", summary=json.loads((run_dir / "summary.json").read_text()))
            write_json(state_path, state)
            for role in ("validation", "test"):
                eval_command = [sys.executable, "-u", str(REPO / "scripts/evaluate_real_robot.py"),
                                "--checkpoint", str(run_dir / "best.pt"), "--data-root", config["data_root"],
                                "--split-role", role, "--output", str(run_dir / f"full_{role}.json"),
                                "--batch-size", str(config["batch_size"]), "--device", "cuda", "--threads", "4"]
                with (run_dir / f"full_{role}.log").open("x") as log:
                    child = subprocess.Popen(eval_command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                    job["pid"] = child.pid
                    write_json(state_path, state)
                    code = child.wait()
                if code:
                    raise RuntimeError(f"{role} evaluation {job['id']} exited with {code}")
                report = json.loads((run_dir / f"full_{role}.json").read_text())
                job[f"full_{role}_macro_action_mae_rad"] = report["metrics"]["episode_macro_action_mae_rad"]
            job.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
            write_json(state_path, state)
        state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
    except Exception as error:
        if current is not None:
            current.update(status="failed", error=str(error))
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(state_path, state)
        raise
    return state


if __name__ == "__main__":
    main()
