"""CPU process owning feedback, supervision and the SDK command connection."""
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
import uuid

from .command_session import CommandSession, SessionOptions
from .contracts import PERIOD_NS


def run_loop(session, transport, inbox, feedback_out, stop_event, overflow, output, hud_channel=None):
    """Used by the hardware worker and nominal offline tests; no CUDA here."""
    output = Path(output)
    errors = []
    next_read = time.monotonic_ns()
    next_tick = session.origin_ns + ((next_read-session.origin_ns)//PERIOD_NS+1)*PERIOD_NS
    last_frame = None
    hud_next_ns, hud_dropped, hud_errors = 0, 0, 0
    hud_record = None
    if hud_channel is not None:
        from .hud_telemetry import session_snapshot
    try:
        with (output/"events.jsonl").open("w", buffering=262144) as events, \
             (output/"execution.jsonl").open("w", buffering=262144) as ticks:
            while not session.terminal:
                now = time.monotonic_ns()
                if stop_event.is_set():
                    session.stop("operator_or_parent_stop", now)
                if overflow.is_set():
                    session.stop("ipc_overflow", now)
                # Stop and direct fresh feedback take priority over prediction backlog.
                if now >= next_read:
                    record = transport.read()
                    hud_record = record
                    received = record["received_monotonic_ns"]
                    if record["frame_serial"] != last_frame:
                        last_frame = record["frame_serial"]
                        session.feedback(record, time.monotonic_ns())
                        events.write(json.dumps({"kind": "feedback", "value": record})+"\n")
                        if session.bias is not None and not session.stop_request:
                            corrected = session.as_feedback(record)
                            try:
                                feedback_out.put_nowait({"record": record, "feedback": asdict(corrected)})
                            except queue.Full:
                                overflow.set(); session.stop("feedback_ipc_overflow", time.monotonic_ns())
                    next_read = max(next_read+3_333_333, time.monotonic_ns()+1)
                for _ in range(32):
                    if session.stop_request:
                        break
                    try:
                        message = inbox.get_nowait()
                    except queue.Empty:
                        break
                    now = time.monotonic_ns()
                    events.write(json.dumps({**message, "consumed_ns": now})+"\n")
                    if message["kind"] == "camera":
                        session.camera(message["name"], message["received_ns"], now)
                    elif message["kind"] == "prediction":
                        session.prediction(message["actions"], **message["metadata"], now_ns=now)
                    else:
                        session.stop("unknown_input_event", now)
                now = time.monotonic_ns()
                session.poll(now)
                if now >= next_tick:
                    row = session.tick(next_tick, now)
                    if row:
                        ticks.write(json.dumps(row)+"\n")
                    # No back-to-back catch-up targets. A late running tick trips the core.
                    next_tick = session.origin_ns + ((time.monotonic_ns()-session.origin_ns)//PERIOD_NS+1)*PERIOD_NS
                # Optional display work is after safety/control evaluation. Never
                # wait on the reader, compute FK/SVD or enqueue a display backlog.
                if hud_channel is not None and now >= hud_next_ns:
                    try:
                        if not hud_channel.publish(session_snapshot(session, hud_record, now)):
                            hud_dropped += 1
                    except Exception:
                        hud_errors += 1  # A display issue cannot alter send/stop decisions.
                    hud_next_ns = now+33_333_333
                time.sleep(max(0., min(.001, (next_read-time.monotonic_ns())/1e9)))
    except BaseException as exc:
        errors.append({"type": type(exc).__name__, "message": str(exc)})
        session.stop("worker_exception", time.monotonic_ns())
        # No new command is attempted on exception; do not claim a confirmed stop.
    finally:
        if hud_channel is not None:
            try:
                hud_channel.publish(session_snapshot(session, hud_record, time.monotonic_ns()))
            except Exception:
                hud_errors += 1
        transport.revoke_targets()
        try:
            transport.close()
        except Exception as exc:
            errors.append({"stage": "release", "message": str(exc)})
        report = session.report()
        report["errors"] = errors
        if hud_channel is not None:
            report['hud_telemetry'] = {'dropped_snapshots':hud_dropped,'publish_errors':hud_errors,
                                     'reader_can_block_control':False}
        (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def _worker(selection, options, shim, sdk, address, origin_ns, run_id,
            inbox, feedback_out, stop_event, overflow, output, hud_channel):
    # Imported only in the explicitly enabled child, never in the default CLI.
    from .sdk_transport import RightArmTransport
    transport = None
    try:
        transport = RightArmTransport(shim, sdk, options, enable_control=True)
        session = CommandSession(selection, options, transport, run_id=run_id, origin_ns=origin_ns,
                                 enable_control=True, operator_confirmed_no_contact=True)
        transport.open(address)
        run_loop(session, transport, inbox, feedback_out, stop_event, overflow, output, hud_channel)
    except BaseException as exc:
        if transport is not None:
            transport.revoke_targets()
            transport.close()
        (Path(output)/"startup_error.json").write_text(json.dumps({"error": str(exc), "physical_stop_confirmed": False})+"\n")
        raise


class CommandProcess:
    def __init__(self, selection, options, shim, sdk, address, output, *,
                 enable_control=False, operator_confirmed_no_contact=False, hud_channel=None):
        if not (enable_control is True and options.allow_hardware is True and operator_confirmed_no_contact is True):
            raise ValueError("command process requires explicit control/no-contact authorization and enabled config")
        output = Path(output); output.mkdir(parents=True, exist_ok=False)
        self.options = options
        self.origin_ns, self.run_id = time.monotonic_ns(), str(uuid.uuid4())
        context = mp.get_context("spawn")
        self.inbox, self.feedback_out = context.Queue(128), context.Queue(256)
        self.stop_event, self.overflow = context.Event(), context.Event()
        self.process = context.Process(target=_worker, args=(selection, options, str(shim), str(sdk), address,
            self.origin_ns, self.run_id, self.inbox, self.feedback_out, self.stop_event, self.overflow, str(output), hud_channel))
        self.process.start()

    def publish(self, kind, **payload):
        if self.stop_event.is_set():
            return
        if not self.process.is_alive():
            raise RuntimeError("command worker exited; no automatic restart")
        try:
            self.inbox.put_nowait({"kind": kind, **payload})
        except queue.Full:
            self.overflow.set()
            raise RuntimeError("command input queue full; send permission revoked")

    def stop(self):
        self.stop_event.set()  # Independent of queued predictions.

    def close(self):
        self.stop()
        self.process.join(self.options.stop_observe_timeout_s+3)
        if self.process.is_alive():
            # Do not kill a connection owner while implying that this stops a robot.
            raise RuntimeError("SDK worker did not exit; physical stop unconfirmed; local operator must intervene")
        for q in (self.inbox, self.feedback_out):
            q.cancel_join_thread(); q.close()
        if self.process.exitcode:
            raise RuntimeError(f"command worker failed: {self.process.exitcode}")
