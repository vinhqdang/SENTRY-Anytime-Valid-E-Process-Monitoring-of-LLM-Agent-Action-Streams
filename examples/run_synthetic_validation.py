"""Reproduce algorithm.md §7 items 1-2 on the synthetic Gaussian toy stream
(§8 next-step 3): empirical FAR-vs-alpha under continuous monitoring, and a
Cavaliers-style (E-detectors) plot of the e-detector value vs. step with the
log(1/alpha) threshold line, for a stream with a mid-run mean shift.

Run: python examples/run_synthetic_validation.py
Writes PNGs to examples/output/.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sentry.pipeline import SentryDetect
from sentry.streams import gaussian_stream

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def calibrate(rng: np.random.Generator, alpha: float):
    d_cal = [gaussian_stream(300, rng).tolist() for _ in range(150)]
    d_thresh = [gaussian_stream(300, rng).tolist() for _ in range(300)]
    return SentryDetect.calibrate(
        d_cal_streams=d_cal,
        d_thresh_streams=d_thresh,
        alpha=alpha,
        conf_delta=0.1,
        delta_range=(0.5, 3.0),
        k=6,
        family="gaussian",
        kind="SR",
    )


def far_vs_alpha(rng: np.random.Generator, alphas, n_streams=100, length=1000):
    """§7 item 1: empirical FAR as a function of stream length, under
    continuous monitoring, for a range of nominal alpha targets."""
    empirical_far = []
    for alpha in alphas:
        monitor = calibrate(rng, alpha)
        total_alarms = 0
        for _ in range(n_streams):
            m = monitor.fresh_copy()
            for x in gaussian_stream(length, rng):
                m.step(float(x))
            total_alarms += len(m.alarms)
        empirical_far.append(total_alarms / (n_streams * length))
    return np.array(empirical_far)


def detection_delay_trace(rng: np.random.Generator, alpha=0.05, changepoint=150, delta=2.0, length=500):
    """§7 item 2: a single trace of the e-detector statistic across an
    injected mean-shift, for the Cavaliers-style figure."""
    monitor = calibrate(rng, alpha)
    m = monitor.fresh_copy()
    stream = gaussian_stream(length, rng, changepoint=changepoint, delta=delta)
    values = np.empty(length)
    alarm_t = None
    for t, x in enumerate(stream):
        v, alarmed = m.step(float(x))
        values[t] = v
        if alarmed and alarm_t is None:
            alarm_t = t + 1
    return values, alarm_t, m.threshold_info.c_alpha


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    alphas = np.array([0.2, 0.1, 0.05, 0.02, 0.01])
    empirical_far = far_vs_alpha(rng, alphas)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(alphas, empirical_far, "o-", label="empirical FAR")
    ax.plot(alphas, alphas, "k--", label="target alpha")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("target alpha")
    ax.set_ylabel("empirical per-step FAR")
    ax.set_title("SENTRY-Detect ARL/FAR validation (Gaussian toy stream)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "arl_far_validation.png"), dpi=150)

    values, alarm_t, c_alpha = detection_delay_trace(rng)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.log(values + 1e-12), label="log M_t (SR e-detector)")
    ax.axhline(np.log(c_alpha), color="red", linestyle="--", label="log c_alpha (PAC threshold)")
    ax.axvline(150, color="gray", linestyle=":", label="changepoint")
    if alarm_t is not None:
        ax.axvline(alarm_t, color="green", linestyle=":", label=f"alarm (t={alarm_t})")
    ax.set_xlabel("step t")
    ax.set_ylabel("log e-detector value")
    ax.set_title("Detection delay after injected mean shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "detection_delay.png"), dpi=150)

    print(f"alphas:        {alphas}")
    print(f"empirical FAR: {empirical_far}")
    print(f"alarm at t={alarm_t} (changepoint at t=150), c_alpha={c_alpha:.2f}")
    print(f"wrote plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
