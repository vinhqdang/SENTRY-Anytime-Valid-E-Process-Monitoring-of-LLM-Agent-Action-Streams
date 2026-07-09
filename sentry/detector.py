"""SR / CUSUM e-detector recursions, algorithm.md §4 Phase 3.

Given a baseline increment L_n (sentry.baseline), the Shiryaev-Roberts (SR)
and CUSUM e-detectors are the O(1) recursions (E-detectors Def. 2.8-2.9):

    M_n^SR = L_n * (M_{n-1}^SR + 1),      M_0^SR = 0
    M_n^CU = L_n * max(M_{n-1}^CU, 1),    M_0^CU = 0

Thresholding either at 1/alpha controls ARL >= 1/alpha under any P in the
pre-change class, with no i.i.d. assumption (E-detectors Thm 2.4). Restarting
the statistic at 0 after an alarm re-applies the same guarantee from the
restart point (E-detectors §6.2 generalized Lorden argument), which is what
lets SENTRY run continuously across an unbounded deployment stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

from sentry.baseline import MixtureBaselineIncrement

Kind = Literal["SR", "CUSUM"]


class SRDetector:
    """M_n^SR = L_n * (M_{n-1}^SR + 1)."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.value = 0.0
        self.t = 0

    def update(self, l_n: float) -> float:
        self.t += 1
        self.value = l_n * (self.value + 1.0)
        return self.value

    @property
    def alarmed(self) -> bool:
        return self.value >= self.threshold

    def reset(self) -> None:
        self.value = 0.0


class CUSUMDetector:
    """M_n^CU = L_n * max(M_{n-1}^CU, 1)."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.value = 0.0
        self.t = 0

    def update(self, l_n: float) -> float:
        self.t += 1
        self.value = l_n * max(self.value, 1.0)
        return self.value

    @property
    def alarmed(self) -> bool:
        return self.value >= self.threshold

    def reset(self) -> None:
        self.value = 0.0


@dataclass
class MixtureSRCUSUMDetector:
    """Mixture e-detector: each mixture component k keeps its own SR/CUSUM
    recursion over its own baseline increment L_t(lambda_k), and the
    reported/thresholded statistic is the weighted sum

        M_t = sum_k omega_k * M_t(k)

    algorithm.md §4 Phase 3. A convex combination of valid e-detectors is
    itself a valid e-detector (linearity of expectation), so this is exactly
    as valid as any single component while adapting to unknown drift
    magnitude (E-detectors §3, method-of-mixtures betting).

    On alarm (M_t >= threshold) every component resets to 0, restarting the
    ARL guarantee from t+1 (algorithm.md §4 Phase 3 pseudocode).
    """

    mixture: MixtureBaselineIncrement
    threshold: float
    kind: Kind = "SR"

    _state: np.ndarray = field(init=False, default=None)
    value: float = field(init=False, default=0.0)
    t: int = field(init=False, default=0)
    alarm_times: list = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._state = np.zeros(len(self.mixture.lambdas))

    def _weights(self) -> np.ndarray:
        k_active = self.mixture.active_k()
        w = np.zeros(len(self.mixture.weights))
        active = np.asarray(self.mixture.weights[:k_active], dtype=float)
        w[:k_active] = active / active.sum()
        return w

    def update(self, x: float) -> float:
        self.t += 1
        l_t = self.mixture.per_component_increments(x)
        # keep MixtureBaselineIncrement's internal step counter (for K(n))
        # in sync even though we bypass its combined .increment() here.
        self.mixture._n += 1

        if self.kind == "SR":
            self._state = l_t * (self._state + 1.0)
        else:
            self._state = l_t * np.maximum(self._state, 1.0)

        w = self._weights()
        self.value = float(np.dot(w, self._state))

        if self.value >= self.threshold:
            self.alarm_times.append(self.t)
            self._state[:] = 0.0
            self.value = 0.0
        return self.value

    @property
    def alarmed_last_step(self) -> bool:
        return bool(self.alarm_times) and self.alarm_times[-1] == self.t

    def reset(self) -> None:
        self._state[:] = 0.0
        self.value = 0.0


def run_stream(
    detector,
    stream: Sequence[float],
    step_fn: Callable[[object], float] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Run a detector over a full stream, recording the statistic path and
    alarm times, restarting after every alarm (algorithm.md §4 Phase 3)."""
    values = np.empty(len(stream))
    alarms: list[int] = []
    for i, x in enumerate(stream, start=1):
        v = detector.update(step_fn(x) if step_fn else x)
        values[i - 1] = v
        if isinstance(detector, MixtureSRCUSUMDetector):
            if detector.alarmed_last_step:
                alarms.append(i)
        elif detector.alarmed:
            alarms.append(i)
            detector.reset()
    return values, alarms
