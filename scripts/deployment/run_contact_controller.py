"""Contact controller entry point. Default: print plan, without loading devices."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import queue
import signal
import threading
import time

import numpy as np

from force_aware_act.control.command_session import SessionOptions
from force_aware_act.control.goal_selection import DEFAULT_PROFILE, select_goal

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "a5999aa40c8a39111366fbbcd9ae99301e70b302a435cffd5b1f63bfc194a855"
VALIDATED_CHECKPOINTS = {
    "contact": CHECKPOINT_SHA256,
    "act": "a4d6abb6fa8b1674cca088086d10523ff279050c0d113371c67b7f87495a6989",
}


def validate_checkpoint(path, policy_kind, manifest=None):
    """Stream the pinned training checkpoint before importing device libraries."""
    expected = manifest["checkpoint_sha256"] if manifest else VALIDATED_CHECKPOINTS[policy_kind]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"checkpoint differs from the validated {policy_kind} model")
    return expected


def parse_arguments(argv=None, *, policy_kind="contact"):
    VALIDATED_CHECKPOINTS[policy_kind]
    parser = argparse.ArgumentParser(description=f"{policy_kind.upper()} controller; default is disabled preview")
    parser.add_argument("--goal-id", required=True)
    parser.add_argument("--controller-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--session-config", type=Path, default=ROOT/"configs/deployment/command_session.json")
    parser.add_argument("--enable-control", action="store_true",
                        help="future explicit hardware run: config must also enable hardware; mode/parameter commands are included")
    parser.add_argument("--operator-confirmed-no-contact", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="with explicit hardware authorization: configure/verify/zero, then stop; no policy targets")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-manifest", type=Path, help="CPU-inspected sensor checkpoint contract")
    parser.add_argument("--ft-calibration", type=Path, help="latest independently calibrated parameter file")
    parser.add_argument("--ft-library", type=Path, help="local compiled compensation core")
    parser.add_argument("--transport", type=Path)
    parser.add_argument("--sdk", type=Path)
    parser.add_argument("--robot-ip", default="192.168.1.190")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float, default=5.)
    parser.add_argument("--hud", action="store_true", help="optional independent localhost force HUD")
    parser.add_argument("--hud-config", type=Path, default=None)
    parser.add_argument("--hud-port", type=int, default=8787)
    args = parser.parse_args(argv)
    try:
        selected = select_goal(args.goal_id, args.controller_profile)
        options = SessionOptions.read(args.session_config)
        args.model_manifest_data = None
        if args.model_manifest:
            from force_aware_act.control.deployment_model import validate_manifest
            args.model_manifest_data = validate_manifest(json.loads(args.model_manifest.read_text()))
            if policy_kind == "act" and args.model_manifest_data["model_kind"] != "act":
                raise ValueError("ACT entry requires an ACT checkpoint")
        if args.ft_calibration or args.ft_library:
            options = replace(options,
                ft_calibration_file=str(args.ft_calibration.resolve()) if args.ft_calibration else options.ft_calibration_file,
                ft_compensation_library=str(args.ft_library.resolve()) if args.ft_library else options.ft_compensation_library)
        if args.model_manifest and not options.ft_calibration_file:
            raise ValueError("sensor model requires FT calibration and compensation library")
        if args.hud_config is None:
            args.hud_config = ROOT/("configs/deployment/force_hud_sensor.json" if options.ft_calibration_file
                                    else "configs/deployment/force_hud.json")
        if args.hud and not 1 <= args.hud_port <= 65535:
            raise ValueError("hud-port must be 1..65535")
        if args.hud:
            # Inspect presentation settings only; no device/process startup here.
            config = json.loads(args.hud_config.read_text())
            if config.get('schema_version') != 'deployment_force_hud_v1':
                raise ValueError('unsupported HUD config')
            if bool(options.ft_calibration_file) != (config.get('source_kind') == 'right_ft_sensor'):
                raise ValueError('HUD source must match session FT/legacy source')
        if args.enable_control:
            if not options.allow_hardware or not args.operator_confirmed_no_contact:
                raise ValueError("hardware disabled: config allow_hardware=true and a current no-contact confirmation are required")
            if not all(getattr(args, k) is not None for k in ("checkpoint", "transport", "sdk", "output")):
                raise ValueError("enabled run requires --checkpoint --transport --sdk --output")
            if not np.isfinite(args.seconds) or not 0 < args.seconds <= 180:
                raise ValueError("seconds must be in (0,180]")
            if args.prepare_only and args.seconds > 5:
                raise ValueError("initial preparation-only observation is limited to 5 seconds")
            if args.output.exists():
                raise ValueError("output exists; each run needs a new directory")
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return args, selected, options


def with_model_horizon(selected, chunk_len):
    """Use the checkpoint horizon consistently in previews and live execution."""
    return replace(selected, config=replace(selected.config,
        action_chunk_len=chunk_len,
        max_prediction_age_ms=min(selected.config.max_prediction_age_ms, chunk_len*10.)))


def describe(selected, options, *, prepare_only=False, policy_kind="contact", manifest=None):
    if manifest is not None:
        selected = with_model_horizon(selected, manifest['action_chunk_len'])
    return {"mode": "disabled_preview", "hardware_connected": False, "commands_sent": False,
            "model_kind": policy_kind, "checkpoint_sha256": manifest["checkpoint_sha256"] if manifest else VALIDATED_CHECKPOINTS[policy_kind],
            "model_manifest": manifest,
            "policy_inputs": ["RGB d455", "RGB d405", "qpos_rad"] +
                             ([] if policy_kind == "act" else ["sensor_wrench_history" if manifest else "external_torque_history"]),
            "external_torque_safety_enabled": True,
            "requested_future_mode": "prepare_only" if prepare_only else "policy_targets",
            "goal_selection": selected.to_dict(), "session": asdict(options),
            "sequence": ["fresh stationary feedback", "right parameters packet", "parameter echo +250ms",
                         "pure right impedance mode packet", "mode and parameter echo +250ms",
                         ("5s or longer no-contact sensor tare in final mode" if options.ft_calibration_file else "2s no-contact software calibration in final mode"),
                         f"fresh observations and [{selected.config.action_chunk_len},7] prediction",
                         "50Hz inference / 100Hz aggregation and guarded target submission",
                         "revoke targets / observe standstill / optional servo off"],
            "stop_limit": "ceasing targets can retain the last target; communication loss cannot confirm standstill"}


def run_live(args, selected, options, *, policy_kind="contact"):
    """One owner across new controller runs sharing the validated SDK directory."""
    if not args.enable_control or not options.allow_hardware or not args.operator_confirmed_no_contact:
        raise ValueError("live execution disabled")
    import fcntl
    with (args.sdk.resolve().parent/".right_arm_control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_live_owned(args, selected, options, policy_kind=policy_kind)


def _run_live_owned(args, selected, options, *, policy_kind="contact"):
    """Devices are initialized only after explicit CLI/config gates pass."""
    if not args.enable_control or not options.allow_hardware or not args.operator_confirmed_no_contact:
        raise ValueError("live execution disabled")
    manifest = getattr(args, "model_manifest_data", None)
    checkpoint_sha256 = validate_checkpoint(args.checkpoint, policy_kind, manifest)
    # Preflight transport files before starting cameras or creating a child.
    build = json.loads(args.transport.with_suffix(".json").read_text())
    if (hashlib.sha256(args.sdk.read_bytes()).hexdigest() != options.expected_sdk_sha256
            or build["sdk_sha256"] != options.expected_sdk_sha256
            or build["sdk_path"] != str(args.sdk.resolve())
            or build["shim_sha256"] != hashlib.sha256(args.transport.read_bytes()).hexdigest()):
        raise ValueError("transport/SDK fingerprint mismatch")
    if options.ft_calibration_file:
        from force_aware_act.control.sensor_ft import SensorFT
        if build["abi"] != 3:
            raise ValueError("sensor deployment requires transport ABI 3")
        preflight = SensorFT(options.ft_calibration_file, options.ft_compensation_library,
                             window_s=options.ft_tare_window_s)
        preflight.close()
    import torch
    import pyrealsense2 as rs
    from force_aware_act.control.command_worker import CommandProcess
    from force_aware_act.real_robot.checkpoint import read_checkpoint
    from force_aware_act.inference.real_robot import RealRobotPolicyAdapter, RealRobotObservationBuffer, ObservationUnavailableError
    from force_aware_act.inference.realtime_gc import StaticGraphGCGuard
    from scripts.deployment.shadow_contact_readonly import ReceiveClock

    torch.set_num_threads(4)
    adapter = RealRobotPolicyAdapter.from_checkpoint(read_checkpoint(args.checkpoint), device="cuda")
    if adapter.model_kind != policy_kind:
        raise ValueError(f"{policy_kind} checkpoint required")
    if (not 2 <= adapter.model.config.chunk_len <= 100 or adapter.data_config.action_rate_hz != 100
            or tuple(adapter.data_config.camera_names) != ("d455", "d405")):
        raise ValueError("checkpoint differs from the 100Hz/two-camera execution contract")
    if adapter.data_config.uses_wrench_sensor:
        from force_aware_act.control.deployment_model import adapter_contract
        contract = adapter_contract(adapter)
        if not manifest or any(manifest.get(k) != v for k, v in contract.items()):
            raise ValueError("checkpoint and sensor deployment manifest disagree")
        if not options.ft_calibration_file:
            raise ValueError("sensor checkpoint requires current calibration parameters")
    elif manifest:
        raise ValueError("sensor manifest cannot be used with joint-estimate checkpoint")
    selected = with_model_horizon(selected, adapter.model.config.chunk_len)
    # CUDA initialization completes before any robot connection or mode packet.
    warm = RealRobotObservationBuffer(adapter.data_config)
    anchor = 1_000_000_000
    for t in range(anchor-210_000_000, anchor, 3_333_333):
        warm.append_state(t, np.asarray(adapter.normalization["qpos_mean"]))
        if adapter.data_config.uses_wrench_sensor:
            warm.append_sensor_wrench(t, np.zeros(6), frame_id=adapter.data_config.wrench_frame)
        else:
            warm.append_external_torque(t, np.zeros(7), qpos=np.asarray(adapter.normalization["qpos_mean"]))
    for name in adapter.data_config.camera_names:
        warm.append_image(name, anchor-1_000_000, np.zeros((480,640,3), dtype=np.uint8))
    for _ in range(5):
        adapter.predict(warm, anchor)
    torch.cuda.synchronize()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"run_config.json").write_text(json.dumps({"goal": selected.to_dict(), "session": asdict(options),
        "model_kind": policy_kind, "checkpoint_sha256": checkpoint_sha256,
        "model_manifest": manifest, "transport_build": build, "seconds": args.seconds,
        "hud": {"enabled":args.hud,"port":args.hud_port,
                "config_sha256":hashlib.sha256(args.hud_config.read_bytes()).hexdigest() if args.hud else None}}, indent=2)+"\n")
    buffer = RealRobotObservationBuffer(adapter.data_config, image_capacity=4)
    clock = ReceiveClock()
    lock, done, calibrated = threading.Lock(), threading.Event(), threading.Event()
    errors, pipelines, threads = [], [], []
    gc_guard = StaticGraphGCGuard()
    monitor = None
    hud, hud_channel, hud_report = None, None, None
    predictions, skips, latencies = 0, 0, []
    first_sensor_ns = None

    def request_stop(*_):
        done.set()
        if monitor is not None:
            monitor.stop()

    old_signals = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}

    def guarded(fn):
        try:
            fn()
        except Exception as exc:
            errors.append({"worker": fn.__name__, "message": str(exc)})
            request_stop()

    def receive_feedback():
        nonlocal first_sensor_ns
        while not done.is_set():
            try:
                item = monitor.feedback_out.get(timeout=.02)
            except queue.Empty:
                if not monitor.process.is_alive():
                    request_stop()
                continue
            f = item["feedback"]
            calibrated.set()
            received, available = clock.map(f["timestamp_ns"]), clock.map(time.monotonic_ns())
            with lock:
                buffer.append_state(received, np.asarray(f["q_rad"]), available_ns=available)
                if adapter.data_config.uses_wrench_sensor:
                    sensor = item["record"]["sensor_ft"]
                    if not sensor["valid"] or sensor["source_ns"] != f["timestamp_ns"]:
                        raise ValueError("model FT sample is not valid or from the same frame")
                    buffer.append_sensor_wrench(received, np.asarray(sensor["wrench_si"]),
                                                frame_id=sensor["frame_id"], available_ns=available)
                    if first_sensor_ns is None:
                        first_sensor_ns = received
                else:
                    buffer.append_external_torque(received, np.asarray(f["external_nm"]),
                                                  qpos=np.asarray(f["q_rad"]), available_ns=available)

    def receive_cameras():
        last = {}
        while not done.is_set():
            for name, pipeline in pipelines:
                frames = pipeline.poll_for_frames()
                frame = frames.get_color_frame() if frames else None
                if not frame or last.get(name) == frame.get_frame_number():
                    continue
                received = time.monotonic_ns()
                rgb = np.asanyarray(frame.get_data()).copy()
                available = time.monotonic_ns()
                with lock:
                    buffer.append_image(name, clock.map(received), rgb, available_ns=clock.map(available))
                if monitor is not None:
                    monitor.publish("camera", name=name, received_ns=received,
                                    available_ns=available, published_ns=time.monotonic_ns(),
                                    frame_number=frame.get_frame_number(),
                                    device_timestamp_ms=frame.get_timestamp(),
                                    timestamp_domain=str(frame.get_frame_timestamp_domain()))
                last[name] = frame.get_frame_number()
            time.sleep(.001)

    try:
        if args.hud:
            from force_aware_act.control.hud_telemetry import HudTelemetry
            from force_aware_act.visualization.deployment_hud import HudProcess, read_config
            hud_channel = HudTelemetry()
            hud = HudProcess(hud_channel,read_config(args.hud_config),selected.to_dict(),
                             args.output/'hud',args.hud_port)
            print(f'Force HUD: open {hud.url} on this PC (display only)',flush=True)
        # The expensive initialized model/module graph is collected and frozen
        # before camera startup or ANY robot connection/parameter/mode packet.
        # New runtime objects still use ordinary automatic cyclic collection.
        gc_guard.start()
        serials = {"d455": "261822304024", "d405": "260322275961"}
        for name in adapter.data_config.camera_names:
            if done.is_set():
                break
            pipeline = rs.pipeline(); config = rs.config(); config.enable_device(serials[name])
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(config); pipelines.append((name, pipeline))
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
            done.wait(3)
            frame = pipeline.wait_for_frames(timeout_ms=1500).get_color_frame()
            if not frame or np.asanyarray(frame.get_data()).shape != (480,640,3):
                raise RuntimeError(f"camera {name} has no valid RGB observation before robot startup")
        if done.is_set():
            return 1
        monitor = CommandProcess(selected, options, args.transport, args.sdk, args.robot_ip, args.output/"controller",
                                 enable_control=True, operator_confirmed_no_contact=True, hud_channel=hud_channel)
        for fn in (receive_feedback, receive_cameras):
            thread = threading.Thread(target=guarded, args=(fn,), daemon=True); thread.start(); threads.append(thread)
        first_prediction_ns = None
        with (args.output/"predictions.jsonl").open("w", buffering=262144) as log:
            while not done.is_set() and monitor.process.is_alive():
                now = time.monotonic_ns()
                if first_prediction_ns is not None and now-first_prediction_ns >= args.seconds*1e9:
                    break
                query_ns = monitor.origin_ns+(now-monitor.origin_ns)//20_000_000*20_000_000
                if args.prepare_only:
                    if calibrated.is_set() and first_prediction_ns is None:
                        first_prediction_ns = now
                    done.wait(.02)
                    continue
                try:
                    with lock:
                        anchor_ns = clock.map(query_ns)
                        if (adapter.data_config.uses_wrench_sensor and policy_kind != "act"
                                and (first_sensor_ns is None or
                                     anchor_ns-first_sensor_ns < round(adapter.data_config.history_duration_s*1e9))):
                            raise ObservationUnavailableError("waiting for post-tare sensor history")
                        snapshot = buffer.observation_at(anchor_ns)
                    inputs = adapter.prepare_snapshot(snapshot)
                    started = time.perf_counter_ns()
                    result = adapter.predict_prepared(inputs, contact_latent_mode="zero")
                    torch.cuda.synchronize()
                    completed = time.monotonic_ns()
                    elapsed = (time.perf_counter_ns()-started)/1e6
                    predictions += 1; latencies.append(elapsed)
                    if first_prediction_ns is None:
                        first_prediction_ns = completed
                    metadata = {"query_id": predictions, "start_ns": query_ns, "observed_ns": query_ns,
                                "available_ns": completed, "run_id": monitor.run_id}
                    monitor.publish("prediction", actions=result["pred_action"].tolist(), metadata=metadata)
                    log.write(json.dumps({"metadata": metadata, "model_ms": elapsed,
                                          "pred_action_rad": result["pred_action"].tolist()})+"\n")
                except ObservationUnavailableError:
                    skips += 1
                current = time.monotonic_ns()
                next_query = monitor.origin_ns+((current-monitor.origin_ns)//20_000_000+1)*20_000_000
                done.wait(max(0, (next_query-time.monotonic_ns())/1e9))
    except Exception as exc:
        errors.append({"worker": "main", "message": str(exc)})
    finally:
        request_stop()
        for thread in threads:
            thread.join(1)
        if monitor is not None:
            try:
                monitor.close()
            except Exception as exc:
                errors.append({"stage": "stop", "message": str(exc)})
        if hud is not None:
            # Give the independent renderer time to display the final state.
            time.sleep(.08)
            hud_report = hud.close()
        gc_guard.close()
        for _, pipeline in reversed(pipelines):
            pipeline.stop()
        for s, handler in old_signals.items():
            signal.signal(s, handler)
        report_path = args.output/"controller/report.json"
        control_report = json.loads(report_path.read_text()) if report_path.exists() else None
        report = {"model_kind": policy_kind, "checkpoint_sha256": checkpoint_sha256,
                  "checkpoint": str(args.checkpoint.resolve()), "model_manifest": manifest,
                  "predictions": predictions, "prepare_only": args.prepare_only,
                  "skipped_observations": skips, "errors": errors,
                  "model_ms_p99": float(np.percentile(latencies, 99)) if latencies else None,
                  "realtime_gc": gc_guard.snapshot(),
                  "hud": hud_report,
                  "controller": control_report, "clock_offset_ns": clock.offset_ns}
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return int(bool(errors) or not control_report
               or control_report["state"] not in ("STOPPED_HOLDING_LAST_TARGET", "SERVO_OFF_CONFIRMED")
               or control_report["outcome"] == "aborted" or bool(control_report["errors"])
               or control_report["first_safety_fault"] is not None
               or (predictions == 0 and not args.prepare_only) or control_report["calibration"] is None)


def main(argv=None, *, policy_kind="contact"):
    args, selected, options = parse_arguments(argv, policy_kind=policy_kind)
    if args.model_manifest_data:
        policy_kind = args.model_manifest_data["model_kind"]
    if not args.enable_control:
        preview = describe(selected, options, prepare_only=args.prepare_only, policy_kind=policy_kind,
                           manifest=args.model_manifest_data)
        preview['hud'] = {'requested':args.hud,'started':False,'url':f'http://127.0.0.1:{args.hud_port}'}
        print(json.dumps(preview, indent=2))
        return 0
    return run_live(args, selected, options, policy_kind=policy_kind)


if __name__ == "__main__":
    raise SystemExit(main())
