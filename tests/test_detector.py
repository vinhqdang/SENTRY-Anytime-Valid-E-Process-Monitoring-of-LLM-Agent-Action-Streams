import numpy as np
import pytest

from sentry.baseline import MixtureBaselineIncrement
from sentry.detector import CUSUMDetector, MixtureSRCUSUMDetector, SRDetector, run_stream


def test_sr_recursion_matches_definition():
    det = SRDetector(threshold=1e9)
    ls = [1.1, 0.9, 1.5, 0.8]
    expected = 0.0
    for l in ls:
        expected = l * (expected + 1.0)
        v = det.update(l)
        assert v == pytest.approx(expected)


def test_cusum_recursion_matches_definition():
    det = CUSUMDetector(threshold=1e9)
    ls = [1.1, 0.9, 1.5, 0.8]
    expected = 0.0
    for l in ls:
        expected = l * max(expected, 1.0)
        v = det.update(l)
        assert v == pytest.approx(expected)


def test_sr_detector_alarms_and_resets():
    det = SRDetector(threshold=5.0)
    values, alarms = run_stream(det, [2.0] * 10)
    assert len(alarms) > 0
    # value right after an alarm-triggered reset should be small again
    assert values[alarms[0] - 1] >= 5.0


def test_mixture_sr_matches_manual_per_component_recursion():
    lambdas = [0.2, 0.6]
    weights = [0.4, 0.6]
    mix = MixtureBaselineIncrement(lambdas=lambdas, weights=weights)
    det = MixtureSRCUSUMDetector(mixture=mix, threshold=1e9, kind="SR")

    manual_state = np.zeros(2)
    from sentry.baseline import ExponentialBaselineIncrement

    comps = [ExponentialBaselineIncrement(l) for l in lambdas]
    xs = [0.1, -0.2, 0.5, 0.3, -0.1]
    for x in xs:
        ls = np.array([c.increment(x) for c in comps])
        manual_state = ls * (manual_state + 1.0)
        expected = float(np.dot(weights, manual_state))
        v = det.update(x)
        assert v == pytest.approx(expected, rel=1e-6)


def test_mixture_detector_restarts_after_alarm():
    lambdas = [1.0]
    weights = [1.0]
    mix = MixtureBaselineIncrement(lambdas=lambdas, weights=weights, mean0=0.0, variance_proxy=1.0)
    det = MixtureSRCUSUMDetector(mixture=mix, threshold=3.0, kind="SR")
    rng = np.random.default_rng(0)
    alarmed_once = False
    for _ in range(2000):
        det.update(float(rng.normal(3.0, 1.0)))  # strongly drifted stream
        if det.alarmed_last_step:
            alarmed_once = True
            assert det.value == 0.0
            break
    assert alarmed_once
