"""End-to-end validation on the synthetic Gaussian toy stream, algorithm.md
§8 next-step 3 ("validate the ARL guarantee end-to-end ... before wiring in
the LLM agent harness") and §7 items 1-2 (ARL/FAR validation, detection
delay)."""

import numpy as np
import pytest

from sentry.pipeline import SentryDetect
from sentry.streams import gaussian_stream


@pytest.fixture(scope="module")
def calibrated_monitor_factory():
    rng = np.random.default_rng(42)
    d_cal = [gaussian_stream(300, rng).tolist() for _ in range(100)]
    d_thresh = [gaussian_stream(300, rng).tolist() for _ in range(150)]

    monitor = SentryDetect.calibrate(
        d_cal_streams=d_cal,
        d_thresh_streams=d_thresh,
        alpha=0.05,
        conf_delta=0.1,
        delta_range=(0.5, 3.0),
        k=6,
        family="gaussian",
        kind="SR",
    )

    return monitor, monitor.fresh_copy


def test_empirical_far_is_controlled_on_pure_nominal_streams(calibrated_monitor_factory):
    monitor, factory = calibrated_monitor_factory
    rng = np.random.default_rng(7)
    alpha = 0.05
    n_streams, length = 60, 500

    total_alarms = 0
    for _ in range(n_streams):
        m = factory()
        stream = gaussian_stream(length, rng)
        for x in stream:
            m.step(float(x))
        total_alarms += len(m.alarms)

    empirical_far = total_alarms / (n_streams * length)
    # PAC guarantee is Pr_thresh(Pr_stream[exists t: M_t > c_alpha] <= alpha) >= 1-delta,
    # a per-stream tail bound, not a rate bound -- allow generous slack so the
    # test is a sanity check on order of magnitude, not the PAC statement itself.
    assert empirical_far <= 5 * alpha


def test_detection_delay_after_injected_drift(calibrated_monitor_factory):
    _, factory = calibrated_monitor_factory
    rng = np.random.default_rng(11)
    changepoint = 100
    delta = 2.0
    n_streams = 30

    delays = []
    false_early_alarms = 0
    for _ in range(n_streams):
        m = factory()
        stream = gaussian_stream(400, rng, changepoint=changepoint, delta=delta)
        alarmed_at = None
        for t, x in enumerate(stream, start=1):
            _, alarmed = m.step(float(x))
            if alarmed:
                alarmed_at = t
                break
        if alarmed_at is not None:
            if alarmed_at < changepoint:
                false_early_alarms += 1
            else:
                delays.append(alarmed_at - changepoint)

    assert len(delays) > 0, "detector should eventually flag a sustained mean shift"
    assert np.mean(delays) < 200
    assert false_early_alarms <= n_streams * 0.2
