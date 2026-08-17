"""Runtime training horizons that can safely extend immutable checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


TRAINING_HORIZON_VERSION = "training_horizon_v1"


@dataclass(frozen=True)
class TrainingHorizon:
    """Absolute optimizer-step target for one training invocation.

    Optimizer and objective settings remain owned by the checkpoint's training
    configuration.  The horizon is deliberately runtime state so a completed
    checkpoint can be continued without pretending its original budget was
    different.
    """

    source_optimizer_step_limit: int
    target_optimizer_steps: int
    start_global_step: int
    steps_per_epoch: Optional[int] = None
    target_epochs: Optional[int] = None
    version: str = TRAINING_HORIZON_VERSION

    @property
    def is_extension(self) -> bool:
        return self.target_optimizer_steps > self.source_optimizer_step_limit

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["is_extension"] = self.is_extension
        return values


def resolve_training_horizon(
    *,
    source_optimizer_step_limit: int,
    start_global_step: int,
    requested_target_optimizer_steps: Optional[int],
    steps_per_epoch: Optional[int] = None,
) -> TrainingHorizon:
    """Resolve and validate an absolute runtime optimizer-step target."""

    for name, value in (
        ("source_optimizer_step_limit", source_optimizer_step_limit),
        ("start_global_step", start_global_step),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if source_optimizer_step_limit <= 0:
        raise ValueError("source_optimizer_step_limit must be positive")
    if start_global_step < 0:
        raise ValueError("start_global_step must be non-negative")
    if requested_target_optimizer_steps is not None and (
        not isinstance(requested_target_optimizer_steps, int)
        or isinstance(requested_target_optimizer_steps, bool)
        or requested_target_optimizer_steps <= 0
    ):
        raise ValueError("target_optimizer_steps must be positive or None")
    if steps_per_epoch is not None and (
        not isinstance(steps_per_epoch, int)
        or isinstance(steps_per_epoch, bool)
        or steps_per_epoch <= 0
    ):
        raise ValueError("steps_per_epoch must be positive or None")

    target = (
        source_optimizer_step_limit
        if requested_target_optimizer_steps is None
        else requested_target_optimizer_steps
    )
    if target <= start_global_step:
        raise ValueError(
            "target_optimizer_steps must be greater than checkpoint "
            f"global_step ({start_global_step}); pass an explicit larger "
            "target to extend a completed run"
        )

    target_epochs = None
    if steps_per_epoch is not None:
        if target % steps_per_epoch != 0:
            raise ValueError(
                "target_optimizer_steps must be divisible by steps_per_epoch "
                "for the episodic Official ACT trainer"
            )
        if start_global_step % steps_per_epoch != 0:
            raise ValueError(
                "Official ACT checkpoint global_step must be at an epoch "
                "boundary"
            )
        target_epochs = target // steps_per_epoch

    return TrainingHorizon(
        source_optimizer_step_limit=source_optimizer_step_limit,
        target_optimizer_steps=target,
        start_global_step=start_global_step,
        steps_per_epoch=steps_per_epoch,
        target_epochs=target_epochs,
    )
