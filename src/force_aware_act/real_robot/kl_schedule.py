"""Stateless posterior KL schedules, indexed by completed optimizer updates.

Only KL(q || N(0,I)) is scheduled. Conditional prior matching is unchanged.
Legacy runs without this contract retain their saved constant KL weight.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PosteriorKLSchedule:
    kind: str = "constant"
    target_weight: float = 10.0
    warmup_epochs: int = 0
    steps_per_epoch: int = 0
    version: str = "posterior_kl_completed_updates_v1"

    def __post_init__(self):
        if self.version != "posterior_kl_completed_updates_v1" or self.kind not in ("constant", "linear_warmup"):
            raise ValueError("invalid posterior KL schedule version/kind")
        if isinstance(self.target_weight, bool) or not isinstance(self.target_weight, (int, float)) or not math.isfinite(self.target_weight) or self.target_weight < 0:
            raise ValueError("posterior KL target weight must be finite and nonnegative")
        for name in ("warmup_epochs", "steps_per_epoch"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"posterior KL {name} must be a nonnegative integer")
        if self.kind == "constant" and (self.warmup_epochs or self.steps_per_epoch):
            raise ValueError("constant posterior KL schedule cannot have warmup")
        if self.kind == "linear_warmup" and (not self.warmup_epochs or not self.steps_per_epoch):
            raise ValueError("linear posterior KL schedule requires positive warmup epochs and steps per epoch")

    def weight(self, completed_updates):
        """Coefficient for the NEXT optimizer update; first update uses weight(0)."""
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError("completed updates must be a nonnegative integer")
        if self.kind == "constant":
            return float(self.target_weight)
        return self.target_weight * min(completed_updates / (self.warmup_epochs * self.steps_per_epoch), 1.0)

    def to_dict(self):
        return asdict(self)

    def state(self, completed_updates):
        next_weight = self.weight(completed_updates)
        return {"completed_updates": completed_updates,
                "last_update_weight": self.weight(completed_updates - 1) if completed_updates else None,
                "next_update_weight": next_weight}


def schedule_from_metadata(metadata):
    saved = metadata.get("run_config", {}).get("posterior_kl_schedule")
    weight = metadata.get("training_config", {}).get("posterior_kl_weight", 10.0)
    result = PosteriorKLSchedule(**saved) if saved is not None else PosteriorKLSchedule(target_weight=weight)
    if result.target_weight != weight:
        raise ValueError("posterior KL schedule target differs from training_config")
    return result


def resolve_kl_schedule(kind, warmup_epochs, *, target_weight, steps_per_epoch, episodic_uniform, saved_metadata=None):
    saved = schedule_from_metadata(saved_metadata) if saved_metadata is not None else None
    kind = kind or (saved.kind if saved else "constant")
    if warmup_epochs is None:
        warmup_epochs = saved.warmup_epochs if saved and saved.kind == kind else 0
    if kind == "linear_warmup" and not episodic_uniform:
        raise ValueError("posterior KL epoch warmup requires episodic_uniform")
    result = PosteriorKLSchedule(kind, target_weight, warmup_epochs,
                                steps_per_epoch if kind == "linear_warmup" else 0)
    if saved is not None and saved != result:
        raise ValueError("cannot change posterior_kl_schedule when resuming; use a new run")
    return result


def validate_kl_schedule_identity(saved, expected):
    schedule = schedule_from_metadata(saved)
    if schedule != schedule_from_metadata(expected):
        raise ValueError("checkpoint/output posterior_kl_schedule differs from this run")
    if "posterior_kl_schedule_state" in saved:
        if saved["posterior_kl_schedule_state"] != schedule.state(saved["global_step"]):
            raise ValueError("checkpoint posterior_kl_schedule_state is inconsistent with global_step")
