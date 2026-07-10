"""SENTRY: Sequential E-process monitoriNg for Trustworthy autonomous AgencY.

Implementation of the algorithm described in ``algorithm.md``: an e-detector
(Shin, Ramdas & Rinaldo) whose e_j-processes are built from a causal
surprise / conformal nonconformity score, with PAC-style thresholding
(Sadhuka et al.) applied because the reference score is learned rather than
known exactly.
"""

from sentry.baseline import (
    ExponentialBaselineIncrement,
    ConformalBaselineIncrement,
    MixtureBaselineIncrement,
    MultiSignalConformalIncrement,
)
from sentry.detector import SRDetector, CUSUMDetector, MixtureSRCUSUMDetector
from sentry.calibration import fit_mixture, pac_threshold, select_order_statistic
from sentry.pipeline import SentryDetect, SentryMonitor

__all__ = [
    "ExponentialBaselineIncrement",
    "ConformalBaselineIncrement",
    "MixtureBaselineIncrement",
    "MultiSignalConformalIncrement",
    "SRDetector",
    "CUSUMDetector",
    "MixtureSRCUSUMDetector",
    "fit_mixture",
    "pac_threshold",
    "select_order_statistic",
    "SentryDetect",
    "SentryMonitor",
]
