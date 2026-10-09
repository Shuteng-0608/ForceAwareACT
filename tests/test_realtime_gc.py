import pytest
from force_aware_act.inference import realtime_gc as module


class FakeGC:
    def __init__(self, enabled=True):
        self.callbacks = []
        self.frozen = 0
        self.enabled = enabled
        self.collected = 0

    def get_freeze_count(self): return self.frozen
    def isenabled(self): return self.enabled
    def collect(self): self.collected += 1
    def freeze(self): self.frozen = 123
    def unfreeze(self): self.frozen = 0


@pytest.mark.parametrize("enabled", [True, False])
def test_guard_restores_frozen_state_on_exception_without_changing_auto_gc(monkeypatch, enabled):
    fake = FakeGC(enabled)
    monkeypatch.setattr(module, "gc", fake)
    guard = module.StaticGraphGCGuard()
    with pytest.raises(ValueError):
        with guard:
            assert fake.collected == 1 and fake.frozen == 123
            assert fake.enabled == enabled
            raise ValueError("camera initialization failed")
    assert fake.frozen == 0 and fake.callbacks == []
    assert fake.enabled == enabled and not guard.active
    guard.close()


def test_guard_rejects_existing_freeze_and_nested_activation(monkeypatch):
    fake = FakeGC()
    monkeypatch.setattr(module, "gc", fake)
    fake.frozen = 10
    with pytest.raises(RuntimeError, match="existing frozen"):
        module.StaticGraphGCGuard().start()
    assert fake.frozen == 10 and fake.collected == 0
    fake.frozen = 0
    with module.StaticGraphGCGuard() as guard:
        with pytest.raises(RuntimeError, match="already active"):
            guard.start()


def test_collection_trace_is_bounded_and_attributed(monkeypatch):
    fake = FakeGC()
    monkeypatch.setattr(module, "gc", fake)
    now = iter(range(100))
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: next(now))
    with module.StaticGraphGCGuard() as guard:
        for _ in range(40):
            fake.callbacks[0]("start", {"generation": 2})
            fake.callbacks[0]("stop", {"generation": 2, "collected": 3, "uncollectable": 0})
        report = guard.snapshot()
        assert report["collections_by_generation"] == [0, 0, 40]
        assert len(report["recent_collections"]) == 32
        assert report["max_collection_ms_by_generation"][2] == 1e-6
        assert report["recent_collections"][-1]["collected"] == 3
