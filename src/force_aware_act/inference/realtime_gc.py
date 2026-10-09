"""Exclude initialized static graphs from GC during a dedicated receive loop.

Process-wide CPython policy: call start before device/worker startup and close
after the receive loop stops. Automatic GC and its thresholds are unchanged.
An already-frozen process is rejected rather than unfreezing someone else's
objects during cleanup. This module imports no model, SDK or device driver.
"""
from collections import deque
import gc
import threading
import time


class StaticGraphGCGuard:
    def __init__(self):
        self.active = False
        self.frozen_objects = 0
        self.precollect_ns = 0
        self.enabled_at_start = None
        self._starts = {}
        self._recent = deque(maxlen=32)
        self._counts = [0, 0, 0]
        self._max_ns = [0, 0, 0]
        self._callback = self._observe

    def start(self):
        if self.active:
            raise RuntimeError("GC guard is already active")
        if gc.get_freeze_count():
            raise RuntimeError("GC guard requires a dedicated process without existing frozen objects")
        self.enabled_at_start = gc.isenabled()
        began = time.monotonic_ns()
        gc.collect()
        self.precollect_ns = time.monotonic_ns()-began
        gc.freeze()
        self.frozen_objects = gc.get_freeze_count()
        self.active = True
        gc.callbacks.append(self._callback)
        return self

    def _observe(self, phase, info):
        key = (threading.get_ident(), info["generation"])
        now = time.monotonic_ns()
        if phase == "start":
            self._starts[key] = now
        elif phase == "stop" and key in self._starts:
            began = self._starts.pop(key)
            generation = info["generation"]
            self._counts[generation] += 1
            self._max_ns[generation] = max(self._max_ns[generation], now-began)
            self._recent.append({"generation": generation, "start_ns": began,
                "end_ns": now, "duration_ns": now-began,
                "thread_name": threading.current_thread().name,
                "collected": info["collected"], "uncollectable": info["uncollectable"]})

    def snapshot(self):
        return {"policy": "precollect_and_freeze_initialized_graph_v1",
                "active": self.active, "frozen_objects_at_start": self.frozen_objects,
                "precollect_ms_before_device_start": self.precollect_ns/1e6,
                "automatic_gc_enabled_at_start": self.enabled_at_start,
                "automatic_gc_enabled_now": gc.isenabled(),
                "collections_by_generation": list(self._counts),
                "max_collection_ms_by_generation": [x/1e6 for x in self._max_ns],
                "recent_collections": list(self._recent)}

    def close(self):
        if not self.active:
            return
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)
        gc.unfreeze()
        self.active = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
