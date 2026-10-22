#!/usr/bin/env python3
"""Read-only terminal/JSON monitoring for train_real_robot.py (stdlib only)."""

from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO / "outputs/1mm_gap_20_v2/runs/contact_seed0"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def argument(argv, flag, default=None):
    # Launch wrappers append overrides; argparse uses the last occurrence.
    for index in range(len(argv) - 1, -1, -1):
        item = argv[index]
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return default


def process_output_dir(argv, cwd):
    # Exact argument matching avoids mistaking a shell or the monitor for training.
    if not argv or not Path(argv[0]).name.startswith("python"):
        return None
    if not any(Path(item).name == "train_real_robot.py" for item in argv[1:]):
        return None
    output = argument(argv, "--output-dir")
    if output is None:
        return None
    path = Path(output)
    return (path if path.is_absolute() else cwd / path).resolve()


def scan_processes():
    """Inspect Linux process metadata; do not signal or attach to any process."""
    processes = []
    try:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        ticks = os.sysconf("SC_CLK_TCK")
        entries = list(Path("/proc").iterdir())
    except (OSError, ValueError):
        return processes
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            argv = [x.decode("utf-8", "replace") for x in (entry / "cmdline").read_bytes().split(b"\0") if x]
            if not argv or not Path(argv[0]).name.startswith("python"):
                continue
            if not any(Path(x).name == "train_real_robot.py" for x in argv[1:]):
                continue
            cwd = Path(os.readlink(entry / "cwd"))
            output = process_output_dir(argv, cwd)
            stat = (entry / "stat").read_text().rpartition(")")[2].split()
            if output is not None and stat[0] != "Z":
                processes.append({"pid": int(entry.name), "parent_pid": int(stat[1]),
                                  "elapsed_s": max(0.0, uptime - int(stat[19]) / ticks),
                                  "output_dir": str(output), "argv": argv})
        except (OSError, ValueError, IndexError):
            continue  # A process can exit between these individual reads.
    return processes


def matching_processes(run_dir, processes):
    matches = [p for p in processes if p["output_dir"] == str(run_dir.resolve())]
    pids = {p["pid"] for p in matches}
    return [p for p in matches if p["parent_pid"] not in pids]


class IncrementalMetrics:
    """Read only newly appended complete JSONL rows, with bounded metric history."""

    def __init__(self, path, recent_window=100):
        self.path = Path(path)
        self.recent_window = recent_window
        self.identity = None
        self.reset()

    def reset(self):
        self.offset = 0
        self.pending = b""
        self.recent = deque(maxlen=self.recent_window)
        self.latest = {}
        self.last_validation = {}
        self.best = None
        self.rows = 0
        self.invalid_lines = 0
        self.nonfinite_rows = 0
        self.mtime = None

    def poll(self, selection_metric):
        try:
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.reset()
                    self.identity = identity
                stream.seek(self.offset)
                data = stream.read()
                self.offset = stream.tell()
                self.mtime = stat.st_mtime
        except OSError:
            return
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                self.invalid_lines += 1
                continue
            if not isinstance(row, dict) or not isinstance(row.get("step"), int):
                self.invalid_lines += 1
                continue
            self.rows += 1
            self.latest = row
            self.recent.append(row)
            if any(isinstance(v, (int, float)) and not isinstance(v, bool) and not finite(v)
                   for v in row.get("train", {}).values()):
                self.nonfinite_rows += 1
            metrics = row.get("held_out_validation")
            if isinstance(metrics, dict):
                self.last_validation = row
                value = metrics.get(selection_metric)
                if finite(value) and (self.best is None or value < self.best["value"]):
                    self.best = {"step": row["step"], "value": value}


class WallRate:
    """Observe progress against wall time, including validation/save pauses."""

    def __init__(self):
        self.points = deque(maxlen=25)

    def observe(self, now, step):
        if self.points and step < self.points[-1][1]:
            self.points.clear()
        self.points.append((now, step))
        elapsed = now - self.points[0][0]
        delta = step - self.points[0][1]
        return delta / elapsed if elapsed >= 15 and delta > 0 else None


def _nvidia_query(fields, kind="gpu"):
    try:
        result = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=2, check=False)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True)) if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        return []


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def gpu_snapshot(pid):
    rows = _nvidia_query("uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu")
    own = {}
    if pid is not None and rows:
        for row in _nvidia_query("gpu_uuid,pid,used_memory", "compute-apps"):
            if len(row) == 3 and row[1].strip() == str(pid):
                own[row[0]] = _number(row[2])
    return [{"name": row[1], "utilization_percent": _number(row[2]),
             "used_mib": _number(row[3]), "total_mib": _number(row[4]),
             "temperature_c": _number(row[5]), "training_process_mib": own.get(row[0])}
            for row in rows if len(row) == 6]


def tail_error(path):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 32_768))
            lines = stream.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    markers = ("Traceback (most recent call last):", "RuntimeError:", "ValueError:",
               "FloatingPointError:", "CUDA out of memory", "KeyboardInterrupt")
    return next((line[-300:] for line in reversed(lines) if any(m in line for m in markers)), None)


def checkpoint_info(path, now):
    try:
        stat = path.stat()
        return {"exists": True, "bytes": stat.st_size, "age_s": max(0.0, now - stat.st_mtime)}
    except OSError:
        return {"exists": False}


def build_report(run_dir, reader, rate, *, now=None, processes=None, include_gpu=True, stale_after=120):
    now = time.time() if now is None else now
    manifest = read_json(run_dir / "run_manifest.json")
    config = manifest.get("run_config", {})
    metric = config.get("selection_metric", "held_out_validation.episode_macro_action_mae_rad").rsplit(".", 1)[-1]
    reader.poll(metric)
    matches = matching_processes(run_dir, scan_processes() if processes is None else processes)
    process = matches[0] if len(matches) == 1 else None
    latest = reader.latest
    step = latest.get("step", 0)
    budget = config.get("steps")
    per_epoch = config.get("steps_per_sampling_epoch")
    completed_epochs = step // per_epoch if isinstance(per_epoch, int) and per_epoch > 0 else None
    summary = read_json(run_dir / "summary.json")
    error = tail_error(run_dir / "train.log")
    age = max(0.0, now - reader.mtime) if reader.mtime is not None else None
    warnings = []
    if len(matches) > 1:
        warnings.append("同一目录检测到多个训练进程：" + ", ".join(str(p["pid"]) for p in matches))
    if reader.invalid_lines:
        warnings.append(f"日志包含{reader.invalid_lines}条完整但无效的JSON记录")
    if reader.nonfinite_rows:
        warnings.append(f"发现{reader.nonfinite_rows}条非有限训练指标")
    if process:
        status = "running" if latest else "initializing"
        if age is not None and age > stale_after:
            warnings.append("进程仍存在但指标长时间未更新；请检查是否在验证、保存或等待I/O")
    elif len(matches) > 1:
        status = "multiple_processes"
    elif (summary.get("completed_steps") == step and summary.get("step_budget") == budget
          and summary.get("termination") in ("early_stopping", "step_budget")):
        status = "early_stopped" if summary["termination"] == "early_stopping" else "completed"
    else:
        status = "error" if error else "process_not_found"
    if error:
        warnings.append("日志末尾包含：" + error)
    window_rate = rate.observe(time.monotonic(), step)
    speed = None
    if process:
        speed = window_rate
        if speed is None and not argument(process["argv"], "--resume") and step and process["elapsed_s"] > 0:
            speed = step / process["elapsed_s"]
    remaining = ((budget - step) / speed if finite(budget) and budget > step and speed and speed > 0 else None)
    means = {}
    for row in reader.recent:
        for key, value in row.get("train", {}).items():
            if finite(value):
                means.setdefault(key, []).append(value)
    timings = [r["data_seconds"] + r["train_seconds"] for r in reader.recent
               if finite(r.get("data_seconds")) and finite(r.get("train_seconds"))]
    validation = reader.last_validation
    stopping = validation.get("early_stopping", {})
    stopping_config = config.get("early_stopping_config", {})
    objectives = config.get("validation_objectives", {})
    checkpoint_names = [spec["checkpoint"] for spec in objectives.get("metrics", {}).values()] or ["best.pt"]
    checkpoint_names.append("last.pt")
    interval = _number(argument(process["argv"], "--validation-interval")) if process else None
    next_validation = (int((step // interval + 1) * interval) if interval and interval > 0 else None)
    if next_validation is not None and finite(budget):
        next_validation = min(next_validation, budget)
    return json_safe({"checked_at": datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
            "run_dir": str(run_dir), "model": manifest.get("model_kind"), "status": status,
            "pid": process["pid"] if process else None, "elapsed_s": process["elapsed_s"] if process else None,
            "step": step, "step_budget": budget, "progress_percent": 100 * step / budget if budget else None,
            "completed_epochs": completed_epochs, "epoch_budget": config.get("max_sampling_epochs"),
            "sampling_epoch": latest.get("sampling_epoch"), "batch_size": latest.get("batch_size_actual"),
            "sampling_mode": config.get("train_sampling_mode"), "last_update_age_s": age,
            "recent_window_rows": len(reader.recent), "latest_losses": latest.get("train", {}),
            "mean_losses": {k: sum(v) / len(v) for k, v in means.items()},
            "mean_read_train_seconds": sum(timings) / len(timings) if timings else None,
            "wall_steps_per_second": speed, "estimated_remaining_to_budget_s": remaining,
            "validation_step": validation.get("step"), "validation_metric": metric,
            "contact_latent_mode": config.get("contact_latent_config", {}).get("evaluation", "zero"),
            "zero_latent_reference": validation.get("zero_latent_reference", {}),
            "validation_value": validation.get("held_out_validation", {}).get(metric),
            "validation": validation.get("held_out_validation", {}), "best_validation": reader.best,
            "next_validation_step": next_validation, "early_stopping_config": stopping_config,
            "early_stopping_state": stopping,
            "validation_objectives": objectives,
            "validation_objective_values": validation.get("validation_objective_values", {}),
            "best_checkpoints": validation.get("best_checkpoints", {}),
            "force_source": manifest.get("data_contract", {}).get("force_source_topic"),
            "force_representation": manifest.get("data_config", {}).get("force_representation", "joint_torque7"),
            "force_coordinate_frame": manifest.get("data_contract", {}).get("force_coordinate_frame"),
            "checkpoints": {n: checkpoint_info(run_dir / n, now) for n in checkpoint_names},
            "gpus": gpu_snapshot(process["pid"] if process else None) if include_gpu else [],
            "warnings": warnings})


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if finite(value) else "—"


def duration(seconds):
    return str(timedelta(seconds=round(seconds))) if finite(seconds) else "—"


def render_report(report):
    states = {"running": "训练进程运行中", "initializing": "初始化中", "completed": "已达到训练上限",
              "early_stopped": "已早停", "process_not_found": "未检测到训练进程", "error": "进程退出且日志含错误",
              "multiple_processes": "同一目录有多个训练进程"}
    progress = report["progress_percent"] or 0
    filled = min(28, max(0, int(progress / 100 * 28)))
    lines = ["实机训练进度监测 | " + report["checked_at"], report["run_dir"],
             f"状态：{states[report['status']]}   模型：{report['model']}   PID：{report['pid'] or '—'}",
             f"更新：{report['step']:,} / {report['step_budget'] or '—'}   "
             f"完整采样epoch：{report['completed_epochs']} / {report['epoch_budget'] or '—'}",
             f"[{('#' * filled).ljust(28, '-')}] {progress:.2f}%（按训练上限）",
             f"采样：{report['sampling_mode']}   batch：{report['batch_size']}   "
             f"日志更新距今：{fmt(report['last_update_age_s'], 1)}秒",
             f"运行时间：{duration(report['elapsed_s'])}   墙钟速度：{fmt(report['wall_steps_per_second'], 2)} 更新/秒",
             f"跑到设定上限预计还需：{duration(report['estimated_remaining_to_budget_s'])}（早停可能提前结束）",
             f"最近{report['recent_window_rows']}步均值 / 最新值："]
    labels = {"loss_total": "总损失", "loss_action": "动作L1", "loss_force": "力矩端点L1",
              "loss_force_highrate": "高频力矩L1", "loss_posterior_kl": "后验KL", "loss_prior_match": "先验匹配"}
    if report.get("force_representation") in ("flange_force3", "end_effector_force3"):
        labels.update(loss_force="三维力端点L1", loss_force_highrate="高频三维力L1")
    if report.get("force_representation") == "end_effector_wrench6":
        labels.update(loss_force="六维端点L1", loss_force_highrate="六维区间L1",
                      loss_force_xyz="三维力端点L1", loss_moment_xyz="三维力矩端点L1",
                      loss_force_highrate_xyz="三维力区间L1", loss_moment_highrate_xyz="三维力矩区间L1")
    for key, label in labels.items():
        lines.append(f"  {label:<10} {fmt(report['mean_losses'].get(key))} / {fmt(report['latest_losses'].get(key))}")
    lines.append(f"平均取样+训练：{fmt(report['mean_read_train_seconds'], 3)}秒/更新（不含验证和保存）")
    lines.append(f"验证：step {report['validation_step'] or '—'}   {report['validation_metric']} = "
                 f"{fmt(report['validation_value'], 6)} rad")
    if report.get("model") == "contact":
        lines.append(f"Contact：训练 posterior 采样；验证/早停/最佳 checkpoint 使用 {report.get('contact_latent_mode', 'zero')}")
        reference = report.get("zero_latent_reference", {})
        if reference:
            lines.append(f"  zero 对照动作误差：{fmt(reference.get(report['validation_metric']), 6)} rad（仅记录，不参与早停）")
            for key, label, unit in (("force", "力", "N"), ("moment", "力矩", "N·m"), ("torque", "关节力矩", "N·m")):
                suffix = "nm" if key != "force" else "n"
                endpoint = reference.get(f"episode_macro_{key}_endpoint_mae_{suffix}")
                interval = reference.get(f"episode_macro_{key}_mae_{suffix}")
                if endpoint is not None or interval is not None:
                    lines.append(f"  zero 对照{label}：端点 {fmt(endpoint, 6)}；区间 {fmt(interval, 6)} {unit}")
    best = report["best_validation"] or {}
    lines.append(f"最优验证：{fmt(best.get('value'), 6)} rad @ step {best.get('step', '—')}   "
                 f"下次验证：{report['next_validation_step'] or '—'}")
    config, state = report["early_stopping_config"], report["early_stopping_state"]
    objectives = report.get("validation_objectives", {})
    if config and config.get("patience", 0):
        warmup = (report["completed_epochs"] or 0) <= config["minimum_epochs"]
        if objectives:
            lines.append(f"多指标早停：{'预热阶段' if warmup else '已启用计数'}；预热{config['minimum_epochs']} epoch；"
                         f"全部指标连续 {config['patience']} 次验证未充分改善才停止；"
                         f"所需相对改善 {config['minimum_relative_improvement'] * 100:.2f}%")
        else:
            lines.append(f"早停：{'预热阶段' if warmup else '已启用计数'}；预热{config['minimum_epochs']} epoch；"
                         f"最近验证计数 {state.get('bad_checks', 0)}/{config['patience']}；"
                         f"所需相对改善 {config['minimum_relative_improvement'] * 100:.2f}%")
        lines.append("  早停计数和参考值按最近一次验证更新。")
    else:
        lines.append("早停：未启用")
    objective_labels = {"action": "动作", "force": "三维力", "moment": "三维力矩", "torque": "关节外力矩"}
    for name, spec in objectives.get("metrics", {}).items():
        current = report.get("validation_objective_values", {}).get(name)
        saved = report.get("best_checkpoints", {}).get(name) or {}
        control = state.get("metrics", {}).get(name, {}).get("controller", {})
        label = objective_labels.get(name, name)
        aggregation = "MAE" if name == "action" else "综合MAE（端点2/3＋区间1/3）"
        lines.append(f"  {label}{aggregation}：当前 {fmt(current, 6)} {spec['unit']}；"
                     f"最佳 {fmt(saved.get('value'), 6)} @ epoch {saved.get('epoch', '—')} → {spec['checkpoint']}")
        if config.get("patience", 0):
            lines.append(f"    未充分改善 {control.get('bad_checks', '—')}/{config['patience']}；"
                         f"早停参考值 {fmt(control.get('reference_metric'), 6)} {spec['unit']}")
    lines.append("外力来源：" + str(report["force_source"] or "—"))
    if report.get("force_representation") == "flange_force3":
        lines.append("力表示：法兰坐标系 Fx/Fy/Fz；物理指标单位 N")
    if report.get("force_representation") == "end_effector_force3":
        lines.append("力表示：传感器 Fx/Fy/Fz；坐标系 " + str(report.get("force_coordinate_frame") or "—") +
                     "；训练L1为无量纲；物理指标单位 N")
        val = report.get("validation", {})
        for label, suffix in (("区间", ""), ("端点", "_endpoint")):
            force = val.get(f"episode_macro_force{suffix}_mae_n", val.get(f"force{suffix}_mae_n"))
            lines.append(f"  验证{label}：力 {fmt(force, 5)} N")
    if report.get("force_representation") == "end_effector_wrench6":
        lines.append("力表示：传感器 Fx/Fy/Fz/Tx/Ty/Tz；训练L1为无量纲；物理误差分别为N和N·m")
        val = report.get("validation", {})
        for label, prefix in (("区间", ""), ("端点", "_endpoint")):
            f = val.get(f"episode_macro_force{prefix}_mae_n", val.get(f"force{prefix}_mae_n"))
            m = val.get(f"episode_macro_moment{prefix}_mae_nm", val.get(f"moment{prefix}_mae_nm"))
            lines.append(f"  验证{label}：力 {fmt(f, 5)} N；力矩 {fmt(m, 5)} N·m")
    for name, info in report["checkpoints"].items():
        lines.append(f"{name}：" + (f"{info['bytes'] / 1024**3:.2f} GiB；{duration(info['age_s'])}前更新" if info["exists"] else "尚未生成"))
    for gpu in report["gpus"]:
        lines.append(f"GPU {gpu['name']}：利用率{fmt(gpu['utilization_percent'], 0)}%；"
                     f"显存{fmt(gpu['used_mib'], 0)}/{fmt(gpu['total_mib'], 0)} MiB；"
                     f"训练进程{fmt(gpu['training_process_mib'], 0)} MiB；{fmt(gpu['temperature_c'], 0)}℃")
    if not report["gpus"]:
        lines.append("GPU状态：不可用或已关闭查询")
    lines.extend("提示：" + warning for warning in report["warnings"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--watch", action="store_true", help="refresh until completion; Ctrl+C stops this monitor")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--recent-window", type=int, default=100)
    parser.add_argument("--stale-after", type=float, default=120.0)
    parser.add_argument("--json", action="store_true", help="JSON snapshot; NDJSON when combined with --watch")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0 or args.recent_window <= 0 or not math.isfinite(args.stale_after) or args.stale_after <= 0:
        parser.error("interval, recent-window and stale-after must be finite and positive")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"run directory does not exist: {run_dir}")
    reader = IncrementalMetrics(run_dir / "metrics.jsonl", args.recent_window)
    rate = WallRate()
    try:
        while True:
            report = build_report(run_dir, reader, rate, include_gpu=not args.no_gpu, stale_after=args.stale_after)
            if args.watch and not args.json and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(json.dumps(report, ensure_ascii=False, allow_nan=False) if args.json else render_report(report), flush=True)
            if not args.watch or report["status"] in ("completed", "early_stopped"):
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if not args.json:
            print("\n监测已退出。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
