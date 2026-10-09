"""Optional, best-effort latest-value HUD telemetry; standard library only.

The control writer never waits for the display reader. Missed display snapshots
are harmless: this channel is not a control input, logger or safety heartbeat.
"""
import ctypes
import json
import multiprocessing as mp


class HudTelemetry:
    def __init__(self, context=None):
        context = context or mp.get_context('spawn')
        self.data = context.RawArray(ctypes.c_ubyte, 16384)
        self.size = context.RawValue(ctypes.c_int, 0)
        self.lock = context.Lock()

    def publish(self, payload):
        encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
        if len(encoded) > len(self.data):
            return False
        if not self.lock.acquire(False):
            return False
        try:
            ctypes.memmove(ctypes.addressof(self.data), encoded, len(encoded))
            self.size.value = len(encoded)
        finally:
            self.lock.release()
        return True

    def read(self):
        if not self.lock.acquire(False):
            return None
        try:
            encoded = ctypes.string_at(ctypes.addressof(self.data), self.size.value)
        finally:
            self.lock.release()
        return json.loads(encoded) if encoded else None


def session_snapshot(session, record, now_ns):
    """One packet's measured q/raw torque plus THIS session's calibration."""
    calibration = session.calibration
    bias_valid = bool(calibration and record and record['state']==calibration['mode']
                      and record['impedance_type']==calibration['impedance_type']
                      and (tuple(record['tool_kine']),tuple(record['tool_dyn']))==session.tool_context)
    return {'schema_version': 1, 'published_ns': now_ns,
            'record': record, 'state': session.state,
            'bias_nm': calibration['bias_nm'] if calibration else None,
            'bias_valid': bias_valid,
            'calibration_status': session.calibration_status['status'],
            'stop_reason': (session.stop_request or {}).get('reason'),
            'target_packets': session.packet_counts['target']}
