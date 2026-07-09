"""Baseline increments L_n, algorithm.md §3.

A baseline increment is any nonnegative sequence L_n with
sup_{P in P} E_P[L_n | F_{n-1}] <= 1 under the pre-change class P
(E-detectors Def. 2.1/§2.4). Feeding a baseline increment into the SR or
CUSUM recursion (sentry.detector) yields a valid e-detector (E-detectors
Prop. 2.3): ARL >= 1/alpha at threshold 1/alpha, with no i.i.d. assumption.

Two constructions are provided, matching algorithm.md §3:
  * ExponentialBaselineIncrement -- parametric mean-shift test on a scalar
    surprise score, valid under a sub-Gaussian/bounded tail assumption.
  * ConformalBaselineIncrement -- nonparametric, built from a conformal
    p-value inverted to an e-value via a Vovk-Wang power calibrator
    (E-detectors Example 2, "change from exchangeability").

MixtureBaselineIncrement combines several exponential baselines over a
grid of tilts lambda_k (unknown drift magnitude/direction), with an
"adaptive" mode that grows the number of active components as
K(n) = O(log n), per E-detectors §3.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

Family = Literal["gaussian", "bounded"]


def _psi(lam: float, family: Family, variance_proxy: float) -> float:
    """Cumulant-generating-function bound psi(lambda) such that
    E[exp(lambda*(X-mean0))] <= exp(psi(lambda)) under the pre-change class,
    for the two tail families algorithm.md §3 calls out ("sub-exponential/
    sub-Gaussian tail assumption on s_t under nominal behavior").
    """
    if family == "gaussian":
        # sub-Gaussian with parameter sqrt(variance_proxy) (Hoeffding-style).
        return 0.5 * lam * lam * variance_proxy
    if family == "bounded":
        # Hoeffding's lemma for a [0,1]-bounded variable: variance proxy 1/4.
        return (lam * lam) / 8.0
    raise ValueError(f"unknown family: {family}")


@dataclass
class ExponentialBaselineIncrement:
    """L_n^(lambda) = exp(lambda * (s(X_n) - mean0) - psi(lambda)), algorithm.md
    §3 eq. for the exponential baseline increment, instantiated as a mean-shift
    test (E-detectors §5.2): pre-change class P = {mean surprise <= mean0},
    post-change class Q = {mean surprise >= mean0 + delta}.

    ``lam`` is the tilt parameter; the optimal tilt for detecting a shift of
    size ``delta`` under family "gaussian" is lam* = delta / variance_proxy.
    """

    lam: float
    mean0: float = 0.0
    variance_proxy: float = 1.0
    family: Family = "gaussian"

    def increment(self, x: float) -> float:
        s = x - self.mean0
        log_l = self.lam * s - _psi(self.lam, self.family, self.variance_proxy)
        return math.exp(log_l)

    @staticmethod
    def optimal_lambda(delta: float, variance_proxy: float, family: Family = "gaussian") -> float:
        if family == "gaussian":
            return delta / variance_proxy
        if family == "bounded":
            return 4.0 * delta
        raise ValueError(f"unknown family: {family}")


def conformal_p_value(cal_nonconformity: Sequence[float], test_nonconformity: float) -> float:
    """Conformal p-value for a new nonconformity score against a calibration
    set, valid under exchangeability of {cal_nonconformity..., test_nonconformity}
    (standard split-conformal p-value, e.g. Vovk et al. 2005 Alg. 2.1)."""
    m = len(cal_nonconformity)
    count = sum(1 for c in cal_nonconformity if c >= test_nonconformity)
    return (1.0 + count) / (m + 1.0)


def vovk_wang_e_value(p: float, kappa: float = 0.5) -> float:
    """Power calibrator e = kappa * p^(kappa-1), kappa in (0,1) (Vovk & Wang
    2021, "E-values: Calibrating Valid p-values"). Turns any valid p-value
    into a valid e-value: E_P[e] <= 1 for any P consistent with the null."""
    if not 0.0 < kappa < 1.0:
        raise ValueError("kappa must be in (0, 1)")
    p = min(max(p, 1e-12), 1.0)
    return kappa * p ** (kappa - 1.0)


@dataclass
class ConformalBaselineIncrement:
    """Nonparametric baseline increment via conformal p-value -> e-value
    (algorithm.md §3, "conformal baseline increment"). ``nonconformity`` maps
    a raw action feature to a scalar nonconformity score; ``cal_scores`` is
    the calibration set of nonconformity scores from D_cal (nominal
    trajectories), assumed exchangeable with the deployment stream under the
    pre-change class (E-detectors Example 2)."""

    cal_scores: Sequence[float]
    nonconformity: Callable[[object], float]
    kappa: float = 0.5

    def increment(self, x: object) -> float:
        nc = self.nonconformity(x)
        p = conformal_p_value(self.cal_scores, nc)
        return vovk_wang_e_value(p, self.kappa)


@dataclass
class MixtureBaselineIncrement:
    """Finite or adaptively-growing mixture over unknown drift magnitude,
    algorithm.md §3 "Mixture over both constructions and over unknown drift
    magnitude": L_n = (1/K) sum_k L_n^(lambda_k), or an adaptive-K(n) version
    (E-detectors §3.1) that keeps the per-step update O(log n).

    A convex combination of valid baseline increments is itself a valid
    baseline increment: E[sum_k w_k L_k | F] = sum_k w_k E[L_k | F] <= sum_k w_k = 1.
    """

    lambdas: Sequence[float]
    weights: Sequence[float]
    mean0: float = 0.0
    variance_proxy: float = 1.0
    family: Family = "gaussian"
    adaptive: bool = False

    _components: list = field(init=False, default_factory=list)
    _n: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if len(self.lambdas) != len(self.weights):
            raise ValueError("lambdas and weights must have the same length")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-9):
            raise ValueError("weights must sum to 1")
        self._components = [
            ExponentialBaselineIncrement(lam, self.mean0, self.variance_proxy, self.family)
            for lam in self.lambdas
        ]

    def active_k(self) -> int:
        """K(n) = ceil(log(n+2)) active components under the adaptive
        schedule (E-detectors §3.1); all K components otherwise."""
        if not self.adaptive:
            return len(self._components)
        return max(1, min(len(self._components), math.ceil(math.log(self._n + 2))))

    def increment(self, x: float) -> float:
        self._n += 1
        k_active = self.active_k()
        w = np.asarray(self.weights[:k_active], dtype=float)
        w = w / w.sum()
        ls = np.array([c.increment(x) for c in self._components[:k_active]])
        return float(np.dot(w, ls))

    def per_component_increments(self, x: float) -> np.ndarray:
        """Return each component's own L_n^(lambda_k) (used by
        MixtureSRCUSUMDetector, which keeps per-component recursive state)."""
        return np.array([c.increment(x) for c in self._components])
