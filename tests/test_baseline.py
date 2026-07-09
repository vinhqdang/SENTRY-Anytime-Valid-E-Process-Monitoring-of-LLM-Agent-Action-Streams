import numpy as np
import pytest

from sentry.baseline import (
    ConformalBaselineIncrement,
    ExponentialBaselineIncrement,
    MixtureBaselineIncrement,
    conformal_p_value,
    vovk_wang_e_value,
)


def test_exponential_baseline_mean_condition_gaussian():
    """E_P[L_n] <= 1 under the nominal (mean0, variance_proxy) Gaussian,
    the defining condition of a baseline increment (E-detectors Def 2.1)."""
    rng = np.random.default_rng(0)
    inc = ExponentialBaselineIncrement(lam=0.5, mean0=0.0, variance_proxy=1.0, family="gaussian")
    x = rng.normal(0, 1, 200_000)
    l = np.array([inc.increment(v) for v in x])
    assert l.mean() == pytest.approx(1.0, rel=0.05)


def test_exponential_baseline_bounded_family():
    rng = np.random.default_rng(1)
    inc = ExponentialBaselineIncrement(lam=0.3, mean0=0.5, variance_proxy=0.25, family="bounded")
    x = rng.uniform(0, 1, 200_000)
    l = np.array([inc.increment(v) for v in x])
    assert l.mean() <= 1.0 + 0.05


def test_optimal_lambda_gaussian():
    assert ExponentialBaselineIncrement.optimal_lambda(2.0, 4.0, "gaussian") == pytest.approx(0.5)


def test_conformal_p_value_uniform_under_exchangeability():
    rng = np.random.default_rng(2)
    cal = rng.normal(size=500)
    ps = []
    for _ in range(2000):
        test = rng.normal()
        ps.append(conformal_p_value(cal, test))
    ps = np.array(ps)
    assert 0.0 < ps.mean() < 1.0
    assert (ps <= 1.0).all() and (ps > 0.0).all()


def test_vovk_wang_e_value_expectation_under_uniform_p():
    """A p-value that is exactly Uniform(0,1) under the null gives
    E[e] = 1 for the power calibrator (Vovk & Wang 2021)."""
    rng = np.random.default_rng(3)
    p = rng.uniform(1e-6, 1, 500_000)
    e = np.array([vovk_wang_e_value(pi, kappa=0.5) for pi in p])
    assert e.mean() == pytest.approx(1.0, rel=0.02)


def test_conformal_baseline_increment_end_to_end():
    rng = np.random.default_rng(4)
    cal_scores = rng.normal(size=300).tolist()
    inc = ConformalBaselineIncrement(cal_scores=cal_scores, nonconformity=lambda x: x)
    vals = [inc.increment(x) for x in rng.normal(size=1000)]
    assert all(v > 0 for v in vals)


def test_mixture_baseline_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        MixtureBaselineIncrement(lambdas=[0.1, 0.2], weights=[0.5, 0.6])


def test_mixture_baseline_is_convex_combination():
    lambdas = [0.1, 0.3, 0.5]
    weights = [0.2, 0.3, 0.5]
    mix = MixtureBaselineIncrement(lambdas=lambdas, weights=weights)
    x = 0.7
    expected = sum(w * ExponentialBaselineIncrement(lam=l).increment(x) for w, l in zip(weights, lambdas))
    assert mix.increment(x) == pytest.approx(expected)


def test_mixture_baseline_adaptive_grows_active_components():
    mix = MixtureBaselineIncrement(
        lambdas=[0.1] * 10, weights=[0.1] * 10, adaptive=True
    )
    active_early = mix.active_k()
    for _ in range(1000):
        mix.increment(0.0)
    active_late = mix.active_k()
    assert active_late >= active_early
    assert active_late <= 10
