"""Validation-driven convergence monitoring for extended training runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


CONVERGENCE_MONITOR_VERSION = "validation_convergence_monitor_v1"


@dataclass(frozen=True)
class ConvergenceUpdate:
    checkpoint_improved: bool
    meaningful_improvement: bool
    should_stop: bool


@dataclass
class ValidationConvergenceMonitor:
    """Track exact best checkpoints separately from plateau evidence."""

    metric_name: str
    minimum_optimizer_steps: int
    patience_validations: int
    min_relative_improvement: float
    best_metric: Optional[float] = None
    best_step: Optional[int] = None
    plateau_reference_metric: Optional[float] = None
    validations_seen: int = 0
    validations_without_meaningful_improvement: int = 0
    version: str = CONVERGENCE_MONITOR_VERSION

    def __post_init__(self) -> None:
        if self.version != CONVERGENCE_MONITOR_VERSION:
            raise ValueError("unsupported convergence monitor version")
        if not self.metric_name:
            raise ValueError("metric_name must be non-empty")
        if self.minimum_optimizer_steps < 0:
            raise ValueError("minimum_optimizer_steps must be non-negative")
        if self.patience_validations <= 0:
            raise ValueError("patience_validations must be positive")
        if not 0.0 <= self.min_relative_improvement < 1.0:
            raise ValueError(
                "min_relative_improvement must be in [0, 1)"
            )
        if self.validations_seen < 0:
            raise ValueError("validations_seen must be non-negative")
        if self.validations_without_meaningful_improvement < 0:
            raise ValueError(
                "validations_without_meaningful_improvement must be "
                "non-negative"
            )

    def update(self, metric: float, *, global_step: int) -> ConvergenceUpdate:
        if not math.isfinite(metric):
            raise ValueError("convergence metric must be finite")
        if global_step < 0:
            raise ValueError("global_step must be non-negative")

        checkpoint_improved = (
            self.best_metric is None or metric < self.best_metric
        )
        if checkpoint_improved:
            self.best_metric = float(metric)
            self.best_step = int(global_step)

        meaningful_improvement = self.plateau_reference_metric is None
        if self.plateau_reference_metric is not None:
            required = (
                abs(self.plateau_reference_metric)
                * self.min_relative_improvement
            )
            meaningful_improvement = (
                metric < self.plateau_reference_metric - required
            )
        if meaningful_improvement:
            self.plateau_reference_metric = float(metric)
            self.validations_without_meaningful_improvement = 0
        elif global_step >= self.minimum_optimizer_steps:
            self.validations_without_meaningful_improvement += 1

        self.validations_seen += 1
        should_stop = (
            global_step >= self.minimum_optimizer_steps
            and not meaningful_improvement
            and self.validations_without_meaningful_improvement
            >= self.patience_validations
        )
        return ConvergenceUpdate(
            checkpoint_improved=checkpoint_improved,
            meaningful_improvement=meaningful_improvement,
            should_stop=should_stop,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> "ValidationConvergenceMonitor":
        return cls(**dict(values))

    def assert_protocol(
        self,
        *,
        metric_name: str,
        minimum_optimizer_steps: int,
        patience_validations: int,
        min_relative_improvement: float,
    ) -> None:
        expected = {
            "metric_name": metric_name,
            "minimum_optimizer_steps": minimum_optimizer_steps,
            "patience_validations": patience_validations,
            "min_relative_improvement": min_relative_improvement,
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError(
                "resume convergence protocol does not match checkpoint: "
                f"expected={actual}, requested={expected}"
            )


def resolve_convergence_monitor(
    *,
    metric_name: str,
    minimum_optimizer_steps: Optional[int],
    patience_validations: Optional[int],
    min_relative_improvement: float,
    default_minimum_optimizer_steps: int,
    prior_state: Optional[Mapping[str, Any]],
) -> Optional[ValidationConvergenceMonitor]:
    """Create, restore, or deliberately disable convergence monitoring."""

    if patience_validations is None:
        if minimum_optimizer_steps is not None:
            raise ValueError(
                "minimum_optimizer_steps requires early-stop patience"
            )
        if prior_state is None:
            return None
        return ValidationConvergenceMonitor.from_dict(prior_state)

    minimum = (
        default_minimum_optimizer_steps
        if minimum_optimizer_steps is None
        else minimum_optimizer_steps
    )
    if prior_state is None:
        return ValidationConvergenceMonitor(
            metric_name=metric_name,
            minimum_optimizer_steps=minimum,
            patience_validations=patience_validations,
            min_relative_improvement=min_relative_improvement,
        )
    monitor = ValidationConvergenceMonitor.from_dict(prior_state)
    monitor.assert_protocol(
        metric_name=metric_name,
        minimum_optimizer_steps=minimum,
        patience_validations=patience_validations,
        min_relative_improvement=min_relative_improvement,
    )
    return monitor
