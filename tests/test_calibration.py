import numpy as np
import pytest

from sentry.calibration import fit_mixture, pac_threshold, select_order_statistic


def test_select_order_statistic_monotone_in_confidence():
    k_loose = select_order_statistic(m=200, alpha=0.05, conf_delta=0.5)
    k_strict = select_order_statistic(m=200, alpha=0.05, conf_delta=0.01)
    assert k_strict >= k_loose


def test_select_order_statistic_within_range():
    m = 100
    k = select_order_statistic(m, alpha=0.05, conf_delta=0.1)
    assert 1 <= k <= m


def test_fit_mixture_rejects_bad_range():
    with pytest.raises(ValueError):
        fit_mixture(delta_range=(1.0, 0.5), k=4)
    with pytest.raises(ValueError):
        fit_mixture(delta_range=(-1.0, 2.0), k=4)


def test_pac_threshold_is_looser_than_ville_when_estimation_noise_present():
    rng = np.random.default_rng(0)
    mixture = fit_mixture(delta_range=(0.2, 2.0), k=4, mean0=0.0, variance_proxy=1.0)
    d_thresh = [rng.normal(0, 1, 300).tolist() for _ in range(150)]
    result = pac_threshold(d_thresh, mixture, alpha=0.05, conf_delta=0.1, kind="SR")
    assert result.c_alpha > 0
    # PAC threshold should generally exceed the exact-Ville threshold 1/alpha
    # once finite-sample estimation noise is accounted for.
    assert result.ville_c_alpha == pytest.approx(20.0)


def test_pac_threshold_running_maxima_sorted():
    rng = np.random.default_rng(1)
    mixture = fit_mixture(delta_range=(0.2, 2.0), k=3)
    d_thresh = [rng.normal(0, 1, 100).tolist() for _ in range(50)]
    result = pac_threshold(d_thresh, mixture, alpha=0.1, conf_delta=0.1)
    assert np.all(np.diff(result.running_maxima) >= 0)
