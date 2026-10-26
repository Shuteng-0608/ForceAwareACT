"""Timestamped temporal aggregation and constrained *proposed* joint targets."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .contracts import PERIOD_NS, Limits, ShadowConfig, timestamp, vector


class ActionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Chunk:
    query_id: int
    start_ns: int
    observed_ns: int
    available_ns: int
    actions: np.ndarray


class TemporalExecutor:
    """Only fuse predictions for the SAME grid tick; no terminal holding.

    Availability is tested against the logical execution tick, not a late wall time.
    Freshness is tested at evaluation time; expired chunks cannot shorten the
    transport deadline of an otherwise fresh aggregate to an already past tick.
The caller skips missed ticks instead of issuing a burst of old proposals.
"""
    def __init__(self, origin_ns: int, run_id: str, config: ShadowConfig):
        self.origin_ns = timestamp(origin_ns)
        if not run_id:
            raise ValueError("run_id is required")
        self.run_id, self.config = run_id, config
        self.chunks = []
        self.seen_ids = set()
        self.last_tick = None
        self.closed = False

    def grid_index(self, tick_ns):
        tick_ns = timestamp(tick_ns)
        if tick_ns < self.origin_ns or (tick_ns - self.origin_ns) % PERIOD_NS:
            raise ValueError("action must lie on the 100 Hz execution grid")
        return (tick_ns - self.origin_ns) // PERIOD_NS

    def submit(self, actions, *, query_id, start_ns, observed_ns, available_ns, run_id, now_ns):
        if self.closed:
            raise ValueError("executor is latched closed")
        if run_id != self.run_id:
            raise ValueError("prediction belongs to another run")
        query_id = timestamp(query_id, "query_id")
        if query_id in self.seen_ids:
            raise ValueError("duplicate query id")
        self.grid_index(start_ns)
        observed_ns, available_ns, now_ns = map(timestamp, (observed_ns, available_ns, now_ns))
        if not self.origin_ns <= observed_ns <= start_ns <= available_ns <= now_ns:
            raise ValueError("prediction has inconsistent/future timestamps")
        arr = np.asarray(actions, dtype=np.float64)
        if arr.shape != (self.config.action_chunk_len, 7) or not np.isfinite(arr).all():
            raise ValueError(f"prediction must contain finite absolute joint positions [{self.config.action_chunk_len},7]")
        self.seen_ids.add(query_id)
        self.chunks.append(Chunk(query_id, start_ns, observed_ns, available_ns, arr.copy()))
        cutoff = now_ns - int(self.config.max_prediction_age_ms * 1e6)
        self.chunks = [c for c in self.chunks if c.observed_ns >= cutoff]
        if len(self.chunks) > 128:
            raise ValueError("prediction buffer overflow")

    def sample(self, tick_ns, *, now_ns=None):
        self.grid_index(tick_ns)
        now_ns = tick_ns if now_ns is None else timestamp(now_ns)
        if self.closed:
            raise ActionUnavailable("executor is latched closed")
        if now_ns < tick_ns or now_ns - tick_ns > self.config.execution_lateness_ms * 1e6:
            raise ActionUnavailable("execution tick is future or too late")
        if self.last_tick is not None and tick_ns <= self.last_tick:
            raise ValueError("execution ticks must increase; no duplicate/catch-up execution")
        self.last_tick = tick_ns
        candidates = sorted((c for c in self.chunks
                             if c.available_ns <= tick_ns
                             and 0 <= now_ns - c.observed_ns < self.config.max_prediction_age_ms * 1e6
                             and 0 <= (tick_ns - c.start_ns) // PERIOD_NS < len(c.actions)),
                            key=lambda c: (c.observed_ns, c.start_ns, c.query_id))
        if not candidates:
            raise ActionUnavailable("no fresh available prediction covers this tick")
        ages = np.asarray([(tick_ns - c.observed_ns) / 1e6 for c in candidates])
        indices = [(tick_ns - c.start_ns) // PERIOD_NS for c in candidates]
        if self.config.aggregation == "newest":
            weights = np.zeros(len(candidates)); weights[-1] = 1
        else:
            logw = (-self.config.official_decay * np.arange(len(candidates))
                    if self.config.aggregation == "official" else -ages / self.config.decay_ms)
            weights = np.exp(logw - logw.max()); weights /= weights.sum()
        proposed = np.sum(np.stack([c.actions[i] for c, i in zip(candidates, indices)]) * weights[:, None], axis=0)
        return {"tick_ns": tick_ns, "action_rad": proposed.tolist(),
                "query_ids": [c.query_id for c in candidates], "chunk_indices": indices,
                "prediction_age_ms": ages.tolist(), "weights": weights.tolist(),
                "weighted_age_ms": float(weights @ ages), "aggregation": self.config.aggregation}

    def close(self):
        self.closed = True
        self.chunks.clear()


class ActionLimiter:
    """Validate shadow targets with the selected motion guard.

Unknown limits yield no approved proposal. State records shadow targets only;
there is intentionally no send/acknowledgement interface in this class.
    In step_reject mode, accepted targets are unmodified. This shadow reference
    is not a successfully transmitted target; a future hardware sender must own
    its own acknowledged-command reference and recheck the step before sending.
"""
    def __init__(self, limits: Limits):
        self.limits = limits
        self.previous_q = self.previous_v = self.previous_ns = None
        self.limited_since = None

    def apply(self, target, feedback, tick_ns):
        target = vector(target, "target")
        q = vector(feedback.q_rad, "q")
        dq = vector(feedback.dq_rad_s, "dq")
        timestamp(tick_ns)
        missing = self.limits.missing_motion()
        if missing:
            return {"action_rad": None, "limited": False, "fault": None,
                    "unassessed": missing}
        low, high = np.asarray(self.limits.joint_min_rad), np.asarray(self.limits.joint_max_rad)
        if np.any(target < low) or np.any(target > high):
            return {"action_rad": None, "fault": "target_joint_limit", "limited": False}
        error_limit = (self.limits.first_action_error_rad if self.previous_q is None
                       else self.limits.following_error_rad)
        if np.any(np.abs(target - q) > error_limit):
            return {"action_rad": None, "fault": "target_feedback_error", "limited": False}
        if self.previous_q is not None and np.any(np.abs(self.previous_q - q) > self.limits.following_error_rad):
            return {"action_rad": None, "fault": "following_error", "limited": False}
        if self.previous_ns is not None and tick_ns <= self.previous_ns:
            raise ValueError("limiter tick must strictly increase")
        if self.limits.motion_guard == "step_reject":
            # The first target has its separate feedback-alignment gate above.
            # Subsequent targets are compared to the last accepted shadow target.
            step = None if self.previous_q is None else target - self.previous_q
            if step is not None and np.any(np.abs(step) > np.asarray(self.limits.max_command_step_rad) + 1e-12):
                return {"action_rad": None, "fault": "target_step_limit", "limited": False}
            self.previous_q, self.previous_ns = target.copy(), tick_ns
            return {"action_rad": target.tolist(), "velocity_rad_s": None,
                    "limited": False, "fault": None, "unassessed": [],
                    "motion_guard": "step_reject", "step_rad": None if step is None else step.tolist(),
                    "step_reference": "initial_actual_feedback" if step is None else "previous_accepted_shadow_target"}
        dt = PERIOD_NS / 1e9 if self.previous_ns is None else (tick_ns - self.previous_ns) / 1e9
        base = q if self.previous_q is None else self.previous_q
        previous_v = dq if self.previous_v is None else self.previous_v
        max_v, max_a = np.asarray(self.limits.velocity_rad_s), np.asarray(self.limits.acceleration_rad_s2)
        v_min = np.maximum(-max_v, previous_v - max_a * dt)
        v_max = np.minimum(max_v, previous_v + max_a * dt)
        # Include joint bounds in the feasible interval. Never silently break an
        # acceleration constraint by clipping position after integrating velocity.
        v_min = np.maximum(v_min, (low - base) / dt)
        v_max = np.minimum(v_max, (high - base) / dt)
        if np.any(v_min > v_max):
            return {"action_rad": None, "fault": "infeasible_motion_constraints", "limited": False}
        v = np.clip((target - base) / dt, v_min, v_max)
        result = base + v * dt
        if np.any(np.abs(result - q) > self.limits.following_error_rad):
            return {"action_rad": None, "fault": "limited_target_feedback_error", "limited": False}
        limited = bool(np.any(np.abs(result - target) > 1e-9))
        if limited:
            self.limited_since = tick_ns if self.limited_since is None else self.limited_since
        else:
            self.limited_since = None
        if self.limited_since is not None and tick_ns - self.limited_since >= self.limits.max_consecutive_limited_ms * 1e6:
            return {"action_rad": None, "fault": "sustained_action_limiting", "limited": True}
        self.previous_q, self.previous_v, self.previous_ns = result.copy(), v.copy(), tick_ns
        return {"action_rad": result.tolist(), "velocity_rad_s": v.tolist(),
                "limited": limited, "fault": None, "unassessed": []}
