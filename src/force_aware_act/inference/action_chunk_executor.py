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
RECENCY_TEMPORAL_AGGREGATION_VERSION = "recency_temporal_ensemble_v1"
RECENCY_TEMPORAL_CANDIDATE_ORDER = "oldest_prediction_to_newest_prediction"
RECENCY_TEMPORAL_WEIGHT_FORMULA = "exp(-k*prediction_age)"
SIGNED_AGE_TEMPORAL_AGGREGATION_VERSION = "signed_age_temporal_ensemble_v1"
SIGNED_AGE_TEMPORAL_CANDIDATE_ORDER = "oldest_prediction_to_newest_prediction"
SIGNED_AGE_TEMPORAL_WEIGHT_FORMULA = "softmax(-signed_k*prediction_age)"
TEMPORAL_ENDPOINT_AGGREGATION_VERSION = "temporal_endpoint_ensemble_v1"
RECEDING_CHUNK_EXECUTION_VERSION = "receding_chunk_execution_v1"


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
        """Add the current prediction and return the configured ensemble."""

        if not isinstance(current_step, int) or isinstance(current_step, bool):
            raise TypeError("current_step must be an int")
        expected_step = 0 if self._last_step is None else self._last_step + 1
        if current_step != expected_step:
            raise ValueError(
                "temporal execution requires one policy query per step: "
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

        weights = self._normalized_weights(ages)
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

    def _log_weights(self, ages: list[int]) -> np.ndarray:
        return -self.decay * np.arange(len(ages), dtype=np.float64)

    def _normalized_weights(self, ages: list[int]) -> np.ndarray:
        log_weights = np.asarray(self._log_weights(ages), dtype=np.float64)
        if log_weights.shape != (len(ages),) or not np.isfinite(log_weights).all():
            raise FloatingPointError("temporal executor produced invalid log weights")
        shifted = log_weights - log_weights.max()
        unnormalized = np.exp(shifted)
        denominator = unnormalized.sum()
        if not math.isfinite(float(denominator)) or denominator <= 0.0:
            raise FloatingPointError("temporal executor produced invalid weights")
        return unnormalized / denominator


class RecencyTemporalActionChunkExecutor(OfficialTemporalActionChunkExecutor):
    """Query every step and favor predictions made most recently.

    Candidates remain in the same explicit oldest-to-newest order as the
    official executor.  Unlike the official candidate-index weighting, this
    executor weights the actual prediction age, so a positive decay gives a
    newer prediction a larger weight.
    """

    version = RECENCY_TEMPORAL_AGGREGATION_VERSION
    candidate_order = RECENCY_TEMPORAL_CANDIDATE_ORDER
    weight_formula = RECENCY_TEMPORAL_WEIGHT_FORMULA

    def _log_weights(self, ages: list[int]) -> np.ndarray:
        return -self.decay * np.asarray(ages, dtype=np.float64)


class SignedAgeTemporalActionChunkExecutor(OfficialTemporalActionChunkExecutor):
    """Query every step and sweep old/new preference with one signed parameter.

    The normalized weight of a prediction with age ``a`` is proportional to
    ``exp(-signed_decay * a)``. Positive values favor recent predictions,
    negative values favor old predictions, and zero gives uniform weights.
    Log weights are centered before exponentiation, so deliberately extreme
    diagnostic values remain numerically stable.
    """

    version = SIGNED_AGE_TEMPORAL_AGGREGATION_VERSION
    candidate_order = SIGNED_AGE_TEMPORAL_CANDIDATE_ORDER
    weight_formula = SIGNED_AGE_TEMPORAL_WEIGHT_FORMULA

    def __init__(self, *, signed_decay: float) -> None:
        if not math.isfinite(signed_decay):
            raise ValueError("signed_decay must be finite")
        super().__init__(decay=0.0)
        self.signed_decay = float(signed_decay)

    def _log_weights(self, ages: list[int]) -> np.ndarray:
        return -self.signed_decay * np.asarray(ages, dtype=np.float64)


class TemporalEndpointActionChunkExecutor(OfficialTemporalActionChunkExecutor):
    """Exact newest-only or oldest-valid temporal diagnostic endpoint."""

    version = TEMPORAL_ENDPOINT_AGGREGATION_VERSION

    def __init__(self, *, preference: str) -> None:
        if preference not in {"newest", "oldest"}:
            raise ValueError("preference must be 'newest' or 'oldest'")
        super().__init__(decay=0.0)
        self.preference = preference
        self.weight_formula = f"{preference}_valid_prediction_only"

    def _normalized_weights(self, ages: list[int]) -> np.ndarray:
        weights = np.zeros(len(ages), dtype=np.float64)
        weights[-1 if self.preference == "newest" else 0] = 1.0
        return weights


@dataclass(frozen=True)
class RecedingChunkResult:
    """One action selected from the currently active open-loop chunk."""

    action: np.ndarray
    query_step: int
    chunk_index: int

    @property
    def prediction_age(self) -> int:
        return self.chunk_index


class RecedingChunkActionExecutor:
    """Query every ``query_interval`` steps and execute the chunk in order.

    ``query_interval=1`` is fresh first-action receding-horizon control.
    ``query_interval=chunk_len`` matches ACT's non-temporal execution: query
    one chunk and execute every element before querying again.
    """

    version = RECEDING_CHUNK_EXECUTION_VERSION

    def __init__(self, *, query_interval: int) -> None:
        if (
            not isinstance(query_interval, int)
            or isinstance(query_interval, bool)
            or query_interval <= 0
        ):
            raise ValueError("query_interval must be a positive int")
        self.query_interval = query_interval
        self._active_chunk: np.ndarray | None = None
        self._query_step: int | None = None
        self._chunk_len: int | None = None
        self._action_dim: int | None = None
        self._last_step: int | None = None
        self._query_count = 0

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def chunk_len(self) -> int | None:
        return self._chunk_len

    @property
    def action_dim(self) -> int | None:
        return self._action_dim

    def should_query(self, current_step: int) -> bool:
        if not isinstance(current_step, int) or isinstance(current_step, bool):
            raise TypeError("current_step must be an int")
        if current_step < 0:
            raise ValueError("current_step must be non-negative")
        return current_step % self.query_interval == 0

    def update(
        self,
        current_step: int,
        action_chunk: np.ndarray | None,
    ) -> RecedingChunkResult:
        expected_step = 0 if self._last_step is None else self._last_step + 1
        if current_step != expected_step:
            raise ValueError(
                "receding chunk execution requires consecutive control steps: "
                f"expected step {expected_step}, got {current_step}"
            )
        query_now = self.should_query(current_step)
        if query_now:
            if action_chunk is None:
                raise ValueError("a new action chunk is required on a query step")
            chunk = np.asarray(action_chunk, dtype=np.float64)
            if chunk.ndim != 2 or chunk.shape[0] <= 0 or chunk.shape[1] <= 0:
                raise ValueError(
                    "action_chunk must be non-empty with shape [K, action_dim]"
                )
            if self.query_interval > chunk.shape[0]:
                raise ValueError(
                    "query_interval cannot exceed the action chunk length"
                )
            if self._chunk_len is None:
                self._chunk_len = int(chunk.shape[0])
                self._action_dim = int(chunk.shape[1])
            elif chunk.shape != (self._chunk_len, self._action_dim):
                raise ValueError(
                    "action_chunk shape changed during rollout: expected "
                    f"{(self._chunk_len, self._action_dim)}, got {tuple(chunk.shape)}"
                )
            self._active_chunk = chunk.copy()
            self._query_step = current_step
            self._query_count += 1
        elif action_chunk is not None:
            raise ValueError("action_chunk must be omitted on a non-query step")
        if self._active_chunk is None or self._query_step is None:
            raise RuntimeError("no active action chunk is available")
        chunk_index = current_step - self._query_step
        if not 0 <= chunk_index < self._active_chunk.shape[0]:
            raise RuntimeError("active action chunk was exhausted before the next query")
        self._last_step = current_step
        return RecedingChunkResult(
            action=self._active_chunk[chunk_index].copy(),
            query_step=self._query_step,
            chunk_index=chunk_index,
        )
