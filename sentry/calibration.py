"""Phase 1-2 calibration, algorithm.md §4 "Algorithm SENTRY-Detect".

Phase 1 fits the mixture (lambda grid + weights) to a guessed range of drift
magnitudes [Delta_L, Delta_U] (E-detectors Algorithm 1/2, "computeBaseline").

Phase 2 replaces the exact Ville threshold c_alpha = 1/alpha with a PAC
threshold (E-valuator Algorithm 1 / Proposition 1, adapted here to the
e-detector's running max rather than a single end-of-trajectory value):
run the mixture SR/CUSUM detector to completion (no early stopping) on each
of a held-out nominal split D_thresh, record the running maximum M^(i), and
take a distribution-free upper confidence bound on the (1-alpha)-quantile of
that maximum via a binomial-tail order statistic (Wilks-style nonparametric
tolerance limit). This yields

    Pr_{D_cal}( Pr_{H_N}[ exists t: M_t > c_alpha ] <= alpha ) >= 1 - delta,

algorithm.md §4 eq. in "Guarantee".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.stats import beta

from sentry.baseline import ExponentialBaselineIncrement, MixtureBaselineIncrement
from sentry.detector import MixtureSRCUSUMDetector

Kind = Literal["SR", "CUSUM"]


def fit_mixture(
    delta_range: tuple[float, float],
    k: int,
    mean0: float = 0.0,
    variance_proxy: float = 1.0,
    family: Literal["gaussian", "bounded"] = "gaussian",
    adaptive: bool = False,
) -> MixtureBaselineIncrement:
    """Phase 1: build a mixture over K tilts lambda_k whose optimal-tilt
    drift magnitudes span [Delta_L, Delta_U] on a log grid (algorithm.md §4
    Phase 1: "using a guessed range (Delta_L, Delta_U) for drift magnitude,
    elicited from red-team/injected-attack severity"). Weights are uniform
    over the grid, the noninformative choice when no prior on drift
    magnitude/direction is available.
    """
    delta_l, delta_u = delta_range
    if delta_l <= 0 or delta_u <= delta_l:
        raise ValueError("delta_range must be a positive, increasing (Delta_L, Delta_U)")
    deltas = np.geomspace(delta_l, delta_u, num=k)
    lambdas = [
        ExponentialBaselineIncrement.optimal_lambda(d, variance_proxy, family) for d in deltas
    ]
    weights = (np.ones(k) / k).tolist()
    return MixtureBaselineIncrement(
        lambdas=lambdas,
        weights=weights,
        mean0=mean0,
        variance_proxy=variance_proxy,
        family=family,
        adaptive=adaptive,
    )


def select_order_statistic(m: int, alpha: float, conf_delta: float) -> int:
    """Smallest 1-indexed order-statistic index k in {1,...,m} such that,
    for X_1,...,X_m iid continuous, the k-th order statistic X_(k) upper
    bounds the (1-alpha)-quantile of the underlying distribution with
    confidence >= 1 - conf_delta:

        Pr[ F(X_(k)) >= 1 - alpha ] >= 1 - conf_delta.

    F(X_(k)) ~ Beta(k, m-k+1) for iid continuous X_i (order-statistic CDF
    identity), so this reduces to a Binomial-tail / Beta computation --
    the "binomial-tail upper confidence bound on the (1-alpha)-quantile"
    of algorithm.md §4 Phase 2 (E-valuator Proposition 1 / Algorithm 1).
    Returns m (most conservative available choice) if no k in range
    satisfies the guarantee, meaning D_thresh is too small for (alpha,delta).
    """
    if not 0 < alpha < 1 or not 0 < conf_delta < 1:
        raise ValueError("alpha and conf_delta must be in (0, 1)")
    for k in range(1, m + 1):
        # Pr[Beta(k, m-k+1) >= 1-alpha] = beta.sf(1-alpha, k, m-k+1)
        if beta.sf(1.0 - alpha, k, m - k + 1) >= 1.0 - conf_delta:
            return k
    return m


@dataclass
class PacThresholdResult:
    c_alpha: float
    ville_c_alpha: float
    order_statistic_index: int
    running_maxima: np.ndarray


def pac_threshold(
    d_thresh_streams: Sequence[Sequence[float]],
    mixture: MixtureBaselineIncrement,
    alpha: float,
    conf_delta: float,
    kind: Kind = "SR",
) -> PacThresholdResult:
    """Phase 2: compute the PAC threshold c_alpha from a held-out nominal
    calibration split D_thresh (algorithm.md §4 Phase 2).

    ``mixture`` should be a *fresh* MixtureBaselineIncrement (its adaptive
    step counter resets per trajectory) -- callers get this for free by
    passing the output of fit_mixture() and letting this function build one
    fresh MixtureSRCUSUMDetector per trajectory.
    """
    ville_c_alpha = 1.0 / alpha
    maxima = np.empty(len(d_thresh_streams))
    for i, stream in enumerate(d_thresh_streams):
        fresh_mixture = MixtureBaselineIncrement(
            lambdas=list(mixture.lambdas),
            weights=list(mixture.weights),
            mean0=mixture.mean0,
            variance_proxy=mixture.variance_proxy,
            family=mixture.family,
            adaptive=mixture.adaptive,
        )
        # never let the "detector" alarm-and-reset during calibration: we
        # want the free-running max over the whole trajectory, so set the
        # threshold to +inf.
        det = MixtureSRCUSUMDetector(mixture=fresh_mixture, threshold=float("inf"), kind=kind)
        running_max = 0.0
        for x in stream:
            v = det.update(x)
            running_max = max(running_max, v)
        maxima[i] = running_max

    order = np.sort(maxima)
    m = len(order)
    k_star = select_order_statistic(m, alpha, conf_delta)
    c_alpha = float(order[k_star - 1])
    return PacThresholdResult(
        c_alpha=c_alpha,
        ville_c_alpha=ville_c_alpha,
        order_statistic_index=k_star,
        running_maxima=order,
    )
