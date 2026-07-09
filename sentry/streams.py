"""Synthetic Bernoulli/Gaussian toy streams, algorithm.md §8 "Immediate next
steps" item 3: "Implement Phase 1-2 calibration first against a synthetic
Bernoulli/Gaussian toy stream (replicate E-detectors' Cavaliers example
structurally) to validate the ARL guarantee end-to-end before wiring in the
LLM agent harness".

A ``rng`` (numpy Generator) must always be passed explicitly by the caller
-- there is no module-level global RNG / seed, so callers control
reproducibility.
"""

from __future__ import annotations

import numpy as np


def gaussian_stream(
    n: int,
    rng: np.random.Generator,
    mean0: float = 0.0,
    sigma: float = 1.0,
    changepoint: int | None = None,
    delta: float = 0.0,
) -> np.ndarray:
    """N(mean0, sigma^2) for t < changepoint, N(mean0+delta, sigma^2) after.
    changepoint=None gives a pure nominal (no-drift) stream, for ARL/FAR
    validation (algorithm.md §7 item 1)."""
    x = rng.normal(loc=mean0, scale=sigma, size=n)
    if changepoint is not None:
        x[changepoint:] += delta
    return x


def bernoulli_stream(
    n: int,
    rng: np.random.Generator,
    p0: float = 0.5,
    changepoint: int | None = None,
    p1: float | None = None,
) -> np.ndarray:
    """Bernoulli(p0) for t < changepoint, Bernoulli(p1) after."""
    p = np.full(n, p0)
    if changepoint is not None:
        p[changepoint:] = p1 if p1 is not None else p0
    return (rng.random(n) < p).astype(float)
