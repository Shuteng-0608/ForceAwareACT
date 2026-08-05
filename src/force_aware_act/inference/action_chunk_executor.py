"""Official-code-equivalent temporal execution for ACT action chunks."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


OFFICIAL_TEMPORAL_AGGREGATION_VERSION = "official_act_temporal_ensemble_v1"
OFFICIAL_TEMPORAL_AGGREGATION_DECAY = 0.01
OFFICIAL_TEMPORAL_CANDIDATE_ORDER = "oldest_prediction_to_newest_prediction"
OFFICIAL_TEMPORAL_WEIGHT_FORMULA = "exp(-k*candidate_index)"


@dataclass(frozen=True)
class TemporalAggregationResult:
    """One temporally ensembled action plus auditable alignment metadata."""

    action: np.ndarray
    prediction_steps: tuple[int, ...]
    chunk_indices: tuple[int, ...]
    ages: tuple[int, ...]
    weights: tuple[float, ...]
    weighted_mean_age: float

    @property
    def num_predictions(self) -> int:
        return len(self.prediction_steps)

    @property
    def oldest_weight(self) -> float:
        return self.weights[0]

    @property
    def newest_weight(self) -> float:
        return self.weights[-1]


class OfficialTemporalActionChunkExecutor:
    """Query every step and ensemble actions aligned to one absolute timestep.

    The official ACT implementation stores each predicted chunk on a row of an
    absolute-time action matrix. At timestep ``t`` it reads column ``t`` in row
    order (oldest prediction first) and applies ``exp(-k*i)`` over that order.
    This class reproduces that populated-candidate computation without relying
    on an all-zero tensor as an ambiguous "not populated" sentinel.
    """

    version = OFFICIAL_TEMPORAL_AGGREGATION_VERSION
    candidate_order = OFFICIAL_TEMPORAL_CANDIDATE_ORDER
    weight_formula = OFFICIAL_TEMPORAL_WEIGHT_FORMULA

    def __init__(
        self,
        *,
        decay: float = OFFICIAL_TEMPORAL_AGGREGATION_DECAY,
    ) -> None:
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError("decay must be finite and non-negative")
        self.decay = float(decay)
        self._predictions: deque[tuple[int, np.ndarray]] = deque()
        self._chunk_len: int | None = None
        self._action_dim: int | None = None
        self._last_step: int | None = None

    @property
    def chunk_len(self) -> int | None:
        return self._chunk_len

    @property
    def action_dim(self) -> int | None:
        return self._action_dim

    @property
    def stored_prediction_count(self) -> int:
        return len(self._predictions)

    def update(
        self,
        current_step: int,
        action_chunk: np.ndarray,
    ) -> TemporalAggregationResult:
        """Add the current prediction and return its official temporal ensemble."""

        if not isinstance(current_step, int) or isinstance(current_step, bool):
            raise TypeError("current_step must be an int")
        expected_step = 0 if self._last_step is None else self._last_step + 1
        if current_step != expected_step:
            raise ValueError(
                "official temporal execution requires one policy query per step: "
                f"expected step {expected_step}, got {current_step}"
            )
        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] <= 0 or chunk.shape[1] <= 0:
            raise ValueError("action_chunk must be non-empty with shape [K, action_dim]")
        if self._chunk_len is None:
            self._chunk_len = int(chunk.shape[0])
            self._action_dim = int(chunk.shape[1])
        elif chunk.shape != (self._chunk_len, self._action_dim):
            raise ValueError(
                "action_chunk shape changed during rollout: expected "
                f"{(self._chunk_len, self._action_dim)}, got {tuple(chunk.shape)}"
            )

        self._predictions.append((current_step, chunk.copy()))
        self._last_step = current_step
        while (
            self._predictions
            and current_step - self._predictions[0][0] >= self._chunk_len
        ):
            self._predictions.popleft()

        prediction_steps: list[int] = []
        chunk_indices: list[int] = []
        ages: list[int] = []
        aligned_actions: list[np.ndarray] = []
        for prediction_step, prediction in self._predictions:
            chunk_index = current_step - prediction_step
            if 0 <= chunk_index < self._chunk_len:
                prediction_steps.append(prediction_step)
                chunk_indices.append(chunk_index)
                ages.append(chunk_index)
                aligned_actions.append(prediction[chunk_index])
        if not aligned_actions:
            raise RuntimeError(
                f"no temporally aligned action exists at step {current_step}"
            )

        unnormalized_weights = np.exp(
            -self.decay * np.arange(len(aligned_actions), dtype=np.float64)
        )
        weights = unnormalized_weights / unnormalized_weights.sum()
        action = np.sum(
            np.asarray(aligned_actions, dtype=np.float64) * weights[:, None],
            axis=0,
        )
        weighted_mean_age = float(
            np.sum(np.asarray(ages, dtype=np.float64) * weights)
        )
        return TemporalAggregationResult(
            action=action,
            prediction_steps=tuple(prediction_steps),
            chunk_indices=tuple(chunk_indices),
            ages=tuple(ages),
            weights=tuple(float(value) for value in weights),
            weighted_mean_age=weighted_mean_age,
        )
