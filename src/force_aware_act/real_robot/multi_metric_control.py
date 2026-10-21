"""Independent validation objectives, best checkpoints, and all-objective stopping.

All values are deployment-mode held-out episode-macro errors. Physical units are
never mixed: endpoint and interval errors are combined only within one quantity.
"""

from copy import deepcopy
from dataclasses import asdict
import math

from .training_control import EarlyStoppingConfig, EarlyStoppingController


def resolve_metric_mode(requested, saved_run_config=None):
    saved = "multi" if (saved_run_config or {}).get("validation_objectives") else "action"
    if requested is not None and saved_run_config is not None and requested != saved:
        raise ValueError("cannot change early-stop-metrics when resuming a checkpoint")
    return requested or saved


def validation_objectives(model_kind, force_representation, selection_metric):
    if selection_metric != "episode_macro_action_mae_rad":
        raise ValueError("multi-metric selection requires episode_macro_action_mae_rad")
    metrics = {"action": {"unit": "rad", "checkpoint": "best.pt",
                          "components": {selection_metric: 1.0}}}
    if model_kind != "act":
        if force_representation in ("flange_force3", "end_effector_force3", "end_effector_wrench6"):
            quantities = [("force", "n", "N")]
            if force_representation == "end_effector_wrench6":
                quantities.append(("moment", "nm", "N*m"))
        elif force_representation == "joint_torque7":
            quantities = [("torque", "nm", "N*m")]
        else:
            raise ValueError("unsupported multi-metric force representation")
        for name, suffix, unit in quantities:
            metrics[name] = {
                "unit": unit, "checkpoint": f"best_{name}.pt",
                "components": {f"episode_macro_{name}_endpoint_mae_{suffix}": 2.0 / 3.0,
                               f"episode_macro_{name}_mae_{suffix}": 1.0 / 3.0},
            }
    return {"version": "real_robot_validation_objectives_v1", "mode": "multi",
            "stop_rule": "all_objectives_stalled", "metrics": metrics}


class MultiMetricController:
    """Track exact minima for saving, significant improvements for patience.

An individual stalled metric can become active again after an improvement.
Missing, nonfinite, and negative inputs fail before any state is changed.
"""

    def __init__(self, config, objectives):
        if objectives.get("version") != "real_robot_validation_objectives_v1" or not objectives.get("metrics"):
            raise ValueError("invalid validation objectives")
        self.config = config
        self.objectives = deepcopy(objectives)
        self.controls = {name: EarlyStoppingController(config) for name in objectives["metrics"]}
        self.best = {name: None for name in self.controls}
        self.values = {}
        self.improved_metrics = []
        self.last_validation_step = 0
        self.completed_epochs = 0
        self.stopped = False

    def _values(self, evaluated):
        values = {}
        for name, spec in self.objectives["metrics"].items():
            parts = []
            for key, weight in spec["components"].items():
                value = evaluated.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"validation objective {name} requires finite nonnegative {key}")
                parts.append(value * weight)
            value = sum(parts)
            if not math.isfinite(value):
                raise ValueError(f"nonfinite validation objective {name}")
            values[name] = value
        return values

    def update(self, evaluated, *, completed_epochs, step):
        if type(step) is not int or step <= self.last_validation_step:
            raise ValueError("validation steps must increase")
        if type(completed_epochs) is not int or completed_epochs < self.completed_epochs:
            raise ValueError("completed epochs must not decrease")
        values = self._values(evaluated)
        improved = []
        for name, value in values.items():
            self.controls[name].update(value, completed_epochs=completed_epochs, step=step)
            if self.best[name] is None or value < self.best[name]["value"]:
                spec = self.objectives["metrics"][name]
                self.best[name] = {"value": value, "step": step, "epoch": completed_epochs,
                                   "unit": spec["unit"], "checkpoint": spec["checkpoint"],
                                   "components": {key: evaluated[key] for key in spec["components"]}}
                improved.append(name)
        self.values = values
        self.improved_metrics = improved
        self.last_validation_step = step
        self.completed_epochs = completed_epochs
        self.stopped = all(control.stopped for control in self.controls.values())
        return self.stopped

    def state_dict(self):
        return {"version": "real_robot_multi_metric_stopping_v1", "config": asdict(self.config),
                "objectives": deepcopy(self.objectives), "metrics": {
                    name: {"controller": control.state_dict(), "best": deepcopy(self.best[name]),
                           "current_value": self.values.get(name)} for name, control in self.controls.items()},
                "last_validation_step": self.last_validation_step, "completed_epochs": self.completed_epochs,
                "stopped": self.stopped}

    def load_state_dict(self, state):
        if (state.get("version") != "real_robot_multi_metric_stopping_v1"
                or state.get("config") != asdict(self.config) or state.get("objectives") != self.objectives):
            raise ValueError("multi-metric checkpoint configuration differs")
        if set(state.get("metrics", {})) != set(self.controls):
            raise ValueError("multi-metric checkpoint metrics differ")
        step, epoch = state.get("last_validation_step"), state.get("completed_epochs")
        if type(step) is not int or step < 0 or type(epoch) is not int or epoch < 0:
            raise ValueError("invalid multi-metric checkpoint progress")
        controls, best, values = {}, {}, {}
        for name, saved in state["metrics"].items():
            control = EarlyStoppingController(self.config)
            control.load_state_dict(saved["controller"])
            if control.last_validation_step != step:
                raise ValueError("multi-metric checkpoint validation steps disagree")
            if control.bad_checks > step or (epoch <= self.config.minimum_epochs and control.bad_checks):
                raise ValueError("multi-metric checkpoint counter disagrees with progress")
            item, value = saved.get("best"), saved.get("current_value")
            if step == 0:
                if item is not None or value is not None or control.reference_metric is not None or epoch != 0:
                    raise ValueError("invalid uninitialized multi-metric checkpoint")
            else:
                spec = self.objectives["metrics"][name]
                if not isinstance(item, dict):
                    raise ValueError("missing best metric in multi-metric checkpoint")
                for v in (value, item.get("value"), control.reference_metric):
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                        raise ValueError("invalid best/current/reference metric in multi-metric checkpoint")
                if (type(item.get("step")) is not int or not 0 < item["step"] <= step
                        or type(item.get("epoch")) is not int or not 0 <= item["epoch"] <= epoch
                        or item.get("unit") != spec["unit"] or item.get("checkpoint") != spec["checkpoint"]):
                    raise ValueError("invalid best checkpoint identity")
                if set(item.get("components", {})) != set(spec["components"]):
                    raise ValueError("invalid best checkpoint components")
                parts = item["components"]
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                       for v in parts.values()):
                    raise ValueError("invalid best checkpoint component value")
                score = sum(parts[key] * weight for key, weight in spec["components"].items())
                if not math.isclose(score, item["value"], rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("best checkpoint score differs from components")
                if item["value"] > value or item["value"] > control.reference_metric:
                    raise ValueError("best checkpoint is worse than current/reference metric")
                values[name] = value
            controls[name], best[name] = control, deepcopy(item)
        stopped = all(control.stopped for control in controls.values())
        if type(state.get("stopped")) is not bool or state["stopped"] != stopped:
            raise ValueError("multi-metric checkpoint stopping flag disagrees with counters")
        self.controls, self.best, self.values = controls, best, values
        self.last_validation_step, self.completed_epochs, self.stopped = step, epoch, stopped
        self.improved_metrics = []


def restore_multi_metric_controller(payload):
    """Validate persisted control before loading model, optimizer, or RNG state."""
    run = payload["run_config"]
    contract = run.get("validation_objectives")
    if contract is None:
        return None
    controller = MultiMetricController(EarlyStoppingConfig(**run["early_stopping_config"]), contract)
    state = payload.get("multi_metric_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing multi_metric_state")
    controller.load_state_dict(state)
    if controller.last_validation_step > payload["global_step"]:
        raise ValueError("multi-metric checkpoint state is newer than model")
    action_best = controller.best["action"]
    if payload.get("best_metric") != (action_best["value"] if action_best else None):
        raise ValueError("checkpoint action best_metric disagrees with multi-metric state")
    if controller.config.patience and payload.get("early_stopping_state") != state:
        raise ValueError("checkpoint early stopping state disagrees with multi-metric state")
    return controller
