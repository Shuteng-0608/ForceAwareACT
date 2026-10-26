"""Feedback-only adapter. Requires a native build with motor APIs compiled out."""
import ctypes as C
import ipaddress
import json
from pathlib import Path
import time

from .sdk_transport import decode_feedback, sha256


class SensorReadOnlyTransport:
    def __init__(self, shim, sdk, options):
        shim, sdk = Path(shim).resolve(), Path(sdk).resolve()
        manifest = json.loads(shim.with_suffix('.json').read_text())
        if (manifest.get('read_only') is not True or manifest['abi'] != 3
                or manifest['shim_sha256'] != sha256(shim)
                or manifest['sdk_sha256'] != sha256(sdk)
                or manifest['sdk_sha256'] != options.expected_sdk_sha256
                or manifest['sdk_path'] != str(sdk)):
            raise ValueError('feedback-only build/SDK provenance mismatch')
        self.lib = C.CDLL(str(shim))
        signatures = {'fa_read_only':([],C.c_int), 'fa_abi_version':([],C.c_int),
            'fa_sdk_version':([],C.c_long), 'fa_open':([C.c_int]*5,C.c_int),
            'fa_select_ft_telemetry':([],C.c_int),
            'fa_read':([C.POINTER(C.c_double),C.c_int],C.c_int),'fa_close':([],C.c_int)}
        for name,(args,result) in signatures.items():
            fn=getattr(self.lib,name); fn.argtypes=args; fn.restype=result
        if self.lib.fa_read_only()!=1 or self.lib.fa_abi_version()!=3:
            raise ValueError('library is not feedback-only ABI 3')
        if any(hasattr(self.lib,name) for name in ('fa_target','fa_configure','fa_enter_impedance','fa_servo_off')):
            raise ValueError('library unexpectedly exports motor controls')
        if self.lib.fa_sdk_version()!=options.expected_sdk_version:
            raise ValueError('SDK version mismatch')
        self.connected=False

    def open(self, address):
        parts=tuple(ipaddress.IPv4Address(address).packed)
        # close also cleans up a partially failed connection; never changes robot mode.
        self.connected=True
        try:
            if self.lib.fa_open(*parts,1)!=1:
                raise RuntimeError('feedback connection failed; no recovery attempted')
            if self.lib.fa_select_ft_telemetry()!=1:
                raise RuntimeError('cannot select telemetry channel 216')
            deadline=time.monotonic()+5.; last=None; good=0
            while time.monotonic()<deadline:
                row=self.read()
                if row['frame_serial']!=last:
                    last=row['frame_serial'];good=good+1 if row['ft_channel']==216 else 0
                    if good>=2:return
                time.sleep(.003)
            raise RuntimeError('no fresh channel 216 feedback')
        except BaseException:
            self.close(); raise

    def read(self):
        values=(C.c_double*79)()
        if self.lib.fa_read(values,79)!=1:
            raise RuntimeError('feedback read failed')
        return decode_feedback(values,time.monotonic_ns())

    def close(self):
        if self.connected:
            self.connected=False
            if self.lib.fa_close()!=1:raise RuntimeError('SDK release failed')
