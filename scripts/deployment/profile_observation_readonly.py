"""Camera/GPU/IPC timing experiment with synthetic joints; no robot SDK or socket.

Predictions go only to a local multiprocessing queue. The receiver records
timings and NEVER constructs a robot transport or publishes a control target.
"""
import argparse
import gc
import json
import multiprocessing as mp
from pathlib import Path
import queue
import resource
import threading
import time


def intervals_ms(timestamps):
    import numpy as np
    values = np.diff(np.asarray(timestamps, dtype=np.int64))/1e6
    if not len(values):
        return {}
    return dict(zip(("p50", "p99", "max"), np.percentile(values, (50, 99, 100)).tolist()))


def overlap_ns(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end)-max(a_start, b_start))


def summarize(trace, worker):
    import numpy as np
    result = {"robot_sdk_loaded": False, "robot_connections": 0,
              "robot_control_packets": 0, "synthetic_joint_input": True,
              "prediction_outputs": "discarded by local timing receiver",
              "gc": trace["gc"], "errors": trace["errors"], "worker": worker}
    for key in ("camera", "feedback", "inference"):
        result[key] = {"rows": len(trace[key])}
    for name in ("d455", "d405"):
        rows = [r for r in trace["camera"] if r["name"] == name]
        gaps = []
        for a, b in zip(rows, rows[1:]):
            interval = (b["received_ns"]-a["received_ns"])/1e6
            if interval > 50:
                gaps.append({"interval_ms": interval, "start_ns": a["received_ns"],
                             "end_ns": b["received_ns"],
                             "frame_number_delta": b["frame_number"]-a["frame_number"],
                             "device_interval_ms": b["device_ms"]-a["device_ms"],
                             "gc_overlaps": [g for g in trace["gc"] if overlap_ns(
                                 a["received_ns"], b["received_ns"], g["start_ns"], g["end_ns"]) > 0]})
        result[name] = {
            "frames": len(rows), "receive_intervals_ms": intervals_ms([r["received_ns"] for r in rows]),
            "gaps_above_50ms": gaps,
            "stage_max_ms": {k: max((r[k] for r in rows), default=0)/1e6
                             for k in ("poll_ns", "copy_ns", "buffer_wait_ns", "buffer_append_ns", "publish_ns")}}
    result["inference"]["anchor_intervals_ms"] = intervals_ms([r["anchor_ns"] for r in trace["inference"]])
    result["inference"]["stage_max_ms"] = {k: max((r[k] for r in trace["inference"]), default=0)/1e6
        for k in ("snapshot_wait_ns", "snapshot_ns", "prepare_ns", "model_ns", "publish_ns")}
    result["gc_summary"] = {
        str(g): {"collections": len(rows), "max_ms": max((r["duration_ns"] for r in rows), default=0)/1e6,
                 "collected": sum(r["collected"] for r in rows)}
        for g in (0, 1, 2) for rows in [[r for r in trace["gc"] if r["generation"] == g]]}
    result["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result


def _mock_receiver(inbox, feedback, stopping, output, qpos, force):
    """Receive action arrays for IPC timing, discard them; no hardware import."""
    records = []
    next_feedback = time.monotonic_ns()
    while not stopping.is_set():
        now = time.monotonic_ns()
        if now >= next_feedback:
            try:
                feedback.put_nowait({"timestamp_ns": now, "qpos": qpos, "force": force})
            except queue.Full:
                records.append({"kind": "feedback_queue_full", "at_ns": now})
            next_feedback = max(next_feedback+3_333_333, time.monotonic_ns()+1)
        for _ in range(32):
            try:
                message = inbox.get_nowait()
            except queue.Empty:
                break
            received = time.monotonic_ns()
            records.append({"kind": message["kind"], "name": message.get("name"),
                            "source_ns": message["source_ns"], "published_ns": message["published_ns"],
                            "consumed_ns": received})
        time.sleep(max(0, min(.001, (next_feedback-time.monotonic_ns())/1e9)))
    Path(output).write_text(json.dumps(records)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--gc-policy", choices=("auto", "freeze"), default="auto")
    parser.add_argument("--gc-probe-seconds", default="",
                        help="optional comma-separated times for explicit generation-2 timing probes; no robot SDK")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probes = [float(x) for x in args.gc_probe_seconds.split(",")] if args.gc_probe_seconds else []
    if not 0 < args.seconds <= 180 or args.output.exists():
        parser.error("seconds must be in (0,180]; output must be a new directory")
    if probes != sorted(set(probes)) or any(not 0 < x < args.seconds for x in probes):
        parser.error("GC probe times must be unique increasing times within the diagnostic window")
    import numpy as np
    import pyrealsense2 as rs
    import torch
    from force_aware_act.inference.real_robot import (
        RealRobotObservationBuffer, RealRobotPolicyAdapter, ObservationUnavailableError)
    from force_aware_act.real_robot.checkpoint import read_checkpoint
    from force_aware_act.inference.realtime_gc import StaticGraphGCGuard

    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    adapter = RealRobotPolicyAdapter.from_checkpoint(read_checkpoint(args.checkpoint), device="cuda")
    qpos = np.asarray(adapter.normalization["qpos_mean"])
    force = np.asarray(adapter.normalization["force_mean"])
    warm = RealRobotObservationBuffer(adapter.data_config, image_capacity=4)
    for t in range(790_000_000, 1_000_000_000, 3_333_333):
        warm.append_state(t, qpos)
        warm.append_external_torque(t, force)
    for name in adapter.data_config.camera_names:
        warm.append_image(name, 999_000_000, np.zeros((480, 640, 3), dtype=np.uint8))
    for _ in range(5):
        adapter.predict(warm, 1_000_000_000)
    torch.cuda.synchronize()
    active = []
    stopping = threading.Event()
    context = mp.get_context("spawn")
    child_stop = context.Event()
    inbox, feedback = context.Queue(128), context.Queue(256)
    worker = context.Process(target=_mock_receiver, args=(inbox, feedback, child_stop,
        str(args.output/"ipc.json"), qpos.tolist(), force.tolist()))
    lock = threading.Lock()
    buffer = RealRobotObservationBuffer(adapter.data_config, image_capacity=4)
    trace = {"camera": [], "feedback": [], "inference": [], "gc": [], "errors": [], "skips": [],
             "explicit_gc_probes": [], "memory_samples": []}
    collecting = {}

    def gc_event(phase, info):
        key = (threading.get_ident(), info["generation"])
        now = time.monotonic_ns()
        if phase == "start":
            collecting[key] = now
        elif key in collecting:
            began = collecting.pop(key)
            trace["gc"].append({"generation": info["generation"], "thread_id": key[0],
                "start_ns": began, "end_ns": now, "duration_ns": now-began,
                "collected": info["collected"], "uncollectable": info["uncollectable"]})

    def publish(kind, source_ns, **payload):
        began = time.monotonic_ns()
        inbox.put_nowait({"kind": kind, "source_ns": source_ns, "published_ns": began, **payload})
        return time.monotonic_ns()-began

    def camera_loop():
        last = {}
        while not stopping.is_set():
            for name, pipeline in active:
                poll_start = time.monotonic_ns()
                frames = pipeline.poll_for_frames()
                frame = frames.get_color_frame() if frames else None
                if not frame or last.get(name) == frame.get_frame_number():
                    continue
                received = time.monotonic_ns()
                pixels = np.asanyarray(frame.get_data()).copy()
                available = time.monotonic_ns()
                wait_start = time.monotonic_ns()
                with lock:
                    acquired = time.monotonic_ns()
                    buffer.append_image(name, received, pixels, available_ns=available)
                    appended = time.monotonic_ns()
                publish_ns = publish("camera", received, name=name)
                trace["camera"].append({"name": name, "received_ns": received,
                    "frame_number": frame.get_frame_number(), "device_ms": frame.get_timestamp(),
                    "timestamp_domain": str(frame.get_frame_timestamp_domain()),
                    "poll_ns": received-poll_start, "copy_ns": available-received,
                    "buffer_wait_ns": acquired-wait_start, "buffer_append_ns": appended-acquired,
                    "publish_ns": publish_ns})
                last[name] = frame.get_frame_number()
            time.sleep(.001)

    def feedback_loop():
        while not stopping.is_set():
            try:
                item = feedback.get(timeout=.02)
            except queue.Empty:
                continue
            received = time.monotonic_ns()
            with lock:
                buffer.append_state(item["timestamp_ns"], np.asarray(item["qpos"]), available_ns=received)
                buffer.append_external_torque(item["timestamp_ns"], np.asarray(item["force"]), available_ns=received)
            trace["feedback"].append({"source_ns": item["timestamp_ns"], "received_ns": received})

    def guarded(fn):
        try:
            fn()
        except Exception as exc:
            trace["errors"].append({"worker": fn.__name__, "error": repr(exc)})
            stopping.set()

    threads = []
    gc_guard = StaticGraphGCGuard()
    try:
        # Same initialization order as the deployment entry point: collect and
        # freeze the warmed model graph before any device or worker is started.
        if args.gc_policy == "freeze":
            gc_guard.start()
        for name, serial in (("d455", "261822304024"), ("d405", "260322275961")):
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(config)
            active.append((name, pipeline))
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
            time.sleep(3)
        worker.start()
        # Collection stays enabled; only the pre-initialized graph is frozen.
        gc.callbacks.append(gc_event)
        origin = time.monotonic_ns()
        trace["origin_ns"] = origin
        trace["gc_policy"] = args.gc_policy
        trace["gc_threshold"] = gc.get_threshold()
        trace["frozen_objects"] = gc.get_freeze_count()
        for fn in (camera_loop, feedback_loop):
            thread = threading.Thread(target=guarded, args=(fn,), daemon=True)
            thread.start()
            threads.append(thread)
        deadline = origin+int(args.seconds*1e9)
        next_memory_sample = origin
        while not stopping.is_set() and time.monotonic_ns() < deadline:
            now = time.monotonic_ns()
            if probes and now-origin >= probes[0]*1e9:
                scheduled_s = probes.pop(0)
                began = time.monotonic_ns()
                collected = gc.collect(2)
                trace["explicit_gc_probes"].append({"scheduled_s": scheduled_s,
                    "at_ns": began, "duration_ns": time.monotonic_ns()-began,
                    "collected": collected})
                now = time.monotonic_ns()
            if now >= next_memory_sample:
                status = Path("/proc/self/status").read_text()
                rss = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
                trace["memory_samples"].append({"at_ns": now, "rss_kib": rss,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved()})
                next_memory_sample = now+10_000_000_000
            anchor = origin+(now-origin)//20_000_000*20_000_000
            try:
                wait_start = time.monotonic_ns()
                with lock:
                    acquired = time.monotonic_ns()
                    snapshot = buffer.observation_at(anchor)
                    snap_done = time.monotonic_ns()
                inputs = adapter.prepare_snapshot(snapshot)
                prepare_done = time.monotonic_ns()
                result = adapter.predict_prepared(inputs, contact_latent_mode="zero")
                torch.cuda.synchronize()
                completed = time.monotonic_ns()
                duration = publish("prediction", anchor, actions=result["pred_action"].tolist())
                trace["inference"].append({"anchor_ns": anchor, "available_ns": completed,
                    "snapshot_wait_ns": acquired-wait_start, "snapshot_ns": snap_done-acquired,
                    "prepare_ns": prepare_done-snap_done, "model_ns": completed-prepare_done,
                    "publish_ns": duration})
            except ObservationUnavailableError as exc:
                trace["skips"].append({"at_ns": time.monotonic_ns(), "reason": str(exc)})
            now = time.monotonic_ns()
            next_query = origin+((now-origin)//20_000_000+1)*20_000_000
            stopping.wait(max(0, (next_query-time.monotonic_ns())/1e9))
            if len(trace["inference"]) and len(trace["inference"]) % 500 == 0:
                print("predictions", len(trace["inference"]), "elapsed_s", round((now-origin)/1e9, 2), flush=True)
    except Exception as exc:
        trace["errors"].append({"worker": "main", "error": repr(exc)})
    finally:
        stopping.set()
        for thread in threads:
            thread.join(2)
        if gc_event in gc.callbacks:
            gc.callbacks.remove(gc_event)
        gc_guard.close()
        trace["gc_guard"] = gc_guard.snapshot()
        child_stop.set()
        if worker.pid is not None:
            worker.join(5)
            if worker.is_alive():
                trace["errors"].append({"worker": "mock_receiver", "error": "did not exit"})
        for _, pipeline in reversed(active):
            pipeline.stop()
        for q in (inbox, feedback):
            q.cancel_join_thread()
            q.close()
    (args.output/"trace.json").write_text(json.dumps(trace)+"\n")
    ipc = json.loads((args.output/"ipc.json").read_text()) if (args.output/"ipc.json").exists() else []
    report = summarize(trace, ipc)
    # Keep detailed traces separate from the compact report.
    report.pop("worker")
    report.pop("gc")
    report["gc_guard"] = trace["gc_guard"]
    report["explicit_gc_probes"] = trace["explicit_gc_probes"]
    report["memory_samples"] = trace["memory_samples"]
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    return int(bool(trace["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
