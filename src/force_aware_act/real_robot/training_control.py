"""Serializable validation-driven stopping for episodic real-robot training."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class EarlyStoppingConfig:
    minimum_epochs: int = 500
    patience: int = 0  # zero keeps legacy fixed-step training
    minimum_relative_improvement: float = 0.005

    def __post_init__(self):
        if type(self.minimum_epochs) is not int or self.minimum_epochs < 0:
            raise ValueError("minimum epochs must be a nonnegative integer")
        if type(self.patience) is not int or self.patience < 0:
            raise ValueError("patience must be a nonnegative integer")
        if not math.isfinite(self.minimum_relative_improvement) or not 0 <= self.minimum_relative_improvement < 1:
            raise ValueError("minimum relative improvement must be in [0,1)")


class EarlyStoppingController:
    def __init__(self, config):
        self.config = config
        self.reference_metric = None
        self.bad_checks = 0
        self.last_validation_step = 0
        self.stopped = False

    def update(self, metric, *, completed_epochs, step):
        if not math.isfinite(metric) or metric < 0:
            raise ValueError("early stopping metric must be finite and nonnegative")
        if step <= self.last_validation_step or completed_epochs < 0:
            raise ValueError("validation steps must increase")
        self.last_validation_step = step
        if completed_epochs <= self.config.minimum_epochs:
            self.reference_metric = metric if self.reference_metric is None else min(metric, self.reference_metric)
            self.bad_checks = 0
        elif self.reference_metric is None or metric < self.reference_metric * (1 - self.config.minimum_relative_improvement):
            self.reference_metric = metric
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        self.stopped = self.config.patience > 0 and self.bad_checks >= self.config.patience
        return self.stopped

    def state_dict(self):
        return {"version": "real_robot_early_stopping_v1", "config": asdict(self.config),
                "reference_metric": self.reference_metric, "bad_checks": self.bad_checks,
                "last_validation_step": self.last_validation_step, "stopped": self.stopped}

    def load_state_dict(self, state):
        if state["version"] != "real_robot_early_stopping_v1" or state["config"] != asdict(self.config):
            raise ValueError("checkpoint early stopping configuration differs")
        metric = state["reference_metric"]
        if metric is not None and (not math.isfinite(metric) or metric < 0):
            raise ValueError("invalid early stopping reference metric")
        for key in ("bad_checks", "last_validation_step"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError("invalid early stopping counter")
        if type(state["stopped"]) is not bool:
            raise ValueError("invalid early stopping flag")
        if state["stopped"] != (self.config.patience > 0 and state["bad_checks"] >= self.config.patience):
            raise ValueError("early stopping flag disagrees with counters")
        self.reference_metric = metric
        self.bad_checks = state["bad_checks"]
        self.last_validation_step = state["last_validation_step"]
        self.stopped = state["stopped"]
