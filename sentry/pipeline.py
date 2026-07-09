"""End-to-end "Algorithm SENTRY-Detect", algorithm.md §4.

    SentryDetect.calibrate(...)  -- Phase 1 (fit mixture) + Phase 2 (PAC threshold)
    SentryMonitor.step(x_t)      -- Phase 3 (online deployment monitor)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from sentry.baseline import MixtureBaselineIncrement
from sentry.calibration import PacThresholdResult, fit_mixture, pac_threshold
from sentry.detector import MixtureSRCUSUMDetector

Kind = Literal["SR", "CUSUM"]


@dataclass
class SentryMonitor:
    """Phase 3: online deployment monitor. Wraps a MixtureSRCUSUMDetector at
    the calibrated PAC threshold; ``alarms`` accumulates every alarm time,
    and the detector restarts (state -> 0) after each one, per algorithm.md
    §4 Phase 3 pseudocode."""

    detector: MixtureSRCUSUMDetector
    threshold_info: PacThresholdResult

    def step(self, x_t: float) -> tuple[float, bool]:
        value = self.detector.update(x_t)
        return value, self.detector.alarmed_last_step

    @property
    def alarms(self) -> list[int]:
        return self.detector.alarm_times

    def fresh_copy(self) -> "SentryMonitor":
        """A new monitor at the same calibrated mixture/threshold (Phases
        1-2 already done), with detector state reset to 0. Use this to
        evaluate the *same* calibration against many independent streams
        (algorithm.md §7 items 1-2) without repeating Phase 2's PAC-threshold
        search on every stream."""
        m = self.detector.mixture
        fresh_mixture = MixtureBaselineIncrement(
            lambdas=list(m.lambdas),
            weights=list(m.weights),
            mean0=m.mean0,
            variance_proxy=m.variance_proxy,
            family=m.family,
            adaptive=m.adaptive,
        )
        fresh_detector = MixtureSRCUSUMDetector(
            mixture=fresh_mixture, threshold=self.detector.threshold, kind=self.detector.kind
        )
        return SentryMonitor(detector=fresh_detector, threshold_info=self.threshold_info)


class SentryDetect:
    """Phases 1-2: calibrate a mixture SR/CUSUM e-detector against a guessed
    drift-magnitude range and a PAC false-alarm target, then hand back a
    SentryMonitor ready for Phase 3 (algorithm.md §4)."""

    @staticmethod
    def calibrate(
        d_cal_streams: Sequence[Sequence[float]],
        d_thresh_streams: Sequence[Sequence[float]],
        alpha: float,
        conf_delta: float,
        delta_range: tuple[float, float],
        k: int = 8,
        mean0: float | None = None,
        variance_proxy: float | None = None,
        family: Literal["gaussian", "bounded"] = "gaussian",
        adaptive: bool = False,
        kind: Kind = "SR",
    ) -> SentryMonitor:
        """algorithm.md §4 Phase 1: "compute the stream of surprise/
        conformal scores" from D_cal. mean0/variance_proxy default to the
        empirical mean/variance of the pooled D_cal scores (the nominal
        reference the exponential baseline increment is centered on);
        pass them explicitly to override with, e.g., known distribution
        parameters on a synthetic toy stream.
        """
        if mean0 is None or variance_proxy is None:
            pooled = np.concatenate([np.asarray(s, dtype=float) for s in d_cal_streams])
            mean0 = float(pooled.mean()) if mean0 is None else mean0
            variance_proxy = float(pooled.var()) if variance_proxy is None else variance_proxy
            variance_proxy = max(variance_proxy, 1e-6)
        mixture = fit_mixture(
            delta_range=delta_range,
            k=k,
            mean0=mean0,
            variance_proxy=variance_proxy,
            family=family,
            adaptive=adaptive,
        )
        threshold_info = pac_threshold(
            d_thresh_streams=d_thresh_streams,
            mixture=mixture,
            alpha=alpha,
            conf_delta=conf_delta,
            kind=kind,
        )
        deployment_mixture = MixtureBaselineIncrement(
            lambdas=list(mixture.lambdas),
            weights=list(mixture.weights),
            mean0=mixture.mean0,
            variance_proxy=mixture.variance_proxy,
            family=mixture.family,
            adaptive=mixture.adaptive,
        )
        detector = MixtureSRCUSUMDetector(
            mixture=deployment_mixture, threshold=threshold_info.c_alpha, kind=kind
        )
        return SentryMonitor(detector=detector, threshold_info=threshold_info)
