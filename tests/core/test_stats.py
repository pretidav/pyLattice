"""Tests of pylattice.core.stats, on series with known answers. The main one is the AR(1) process
x[t] = phi x[t-1] + noise, whose autocorrelation is rho(t) = phi**t and whose integrated
autocorrelation time is exactly tau = (1 + phi) / (2 (1 - phi)).

Run with:  pytest tests/core/test_stats.py
"""
import numpy as np
import pytest
from scipy.signal import lfilter

from pylattice.core import stats


def ar1(n, phi, seed, series=None):
    """AR(1) series (n,) or (n, series): unit-variance noise, started long before the first point"""
    rng = np.random.default_rng(seed)
    shape = (n + 2000,) if series is None else (n + 2000, series)
    return lfilter([1.0], [1.0, -phi], rng.standard_normal(shape), axis=0)[2000:]


def tau_exact(phi):
    return (1 + phi) / (2 * (1 - phi))


# ------------------------------------------------------------------- mean_error
def test_mean_error_of_independent_points_is_std_over_sqrt_n():
    x = np.random.default_rng(1).standard_normal((20000, 40))
    errors = [stats.mean_error(x[:, k])[1] for k in range(40)]
    assert abs(np.mean(errors) / (1 / np.sqrt(20000)) - 1) < 0.1             # 20 blocks: each error is good to ~16%, the mean of 40 to ~3%
    mean, error = stats.mean_error(x[:, 0])
    assert mean == x[:, 0].mean() and type(mean) is np.float64 and type(error) is np.float64


def test_mean_error_takes_the_correlation_into_account():
    phi, n = 0.9, 20000
    x = ar1(n, phi, 2, series=40)
    errors = np.array([stats.mean_error(x[:, k])[1] for k in range(40)])
    true = np.sqrt(1 / (1 - phi ** 2) * 2 * tau_exact(phi) / n)               # variance of the mean = var * 2 tau / n
    naive = x.std(axis=0).mean() / np.sqrt(n)
    assert abs(errors.mean() / true - 1) < 0.15, (errors.mean(), true)
    assert errors.mean() > 3 * naive                                          # std/sqrt(n) would be 4.4 times too small


def test_mean_error_of_a_matrix_has_one_value_per_column():
    x = np.random.default_rng(3).standard_normal((1000, 3)) * [1.0, 2.0, 5.0] + [0, 10, -4]
    mean, error = stats.mean_error(x, blocks=10)
    assert mean.shape == error.shape == (3,)
    assert np.allclose(mean, x.mean(axis=0))
    for k in range(3):
        assert np.allclose((mean[k], error[k]), stats.mean_error(x[:, k], blocks=10), rtol=1e-12, atol=0)
    assert error[2] > error[1] > error[0]                                     # larger scatter, larger error


def test_mean_error_blocks_and_the_points_that_do_not_fit():
    x = np.arange(45, dtype=float)
    mean, error = stats.mean_error(x, blocks=20)                              # pieces of 2: the last 5 points are left out of the error
    pieces = x[:40].reshape(20, 2).mean(axis=1)
    assert mean == x.mean() and np.isclose(error, pieces.std(ddof=1) / np.sqrt(20))
    mean, error = stats.mean_error(np.full(100, 3.0))
    assert mean == 3.0 and error == 0.0                                       # no scatter, no error
    mean, error = stats.mean_error([1.0, 2.0, 3.0, 4.0], blocks=2)            # a list will do
    assert mean == 2.5 and np.isclose(error, np.std([1.5, 3.5], ddof=1) / np.sqrt(2))


def test_mean_error_refuses_what_it_cannot_do():
    with pytest.raises(ValueError):
        stats.mean_error(np.ones(10), blocks=20)                              # fewer points than blocks
    with pytest.raises(ValueError):
        stats.mean_error(np.ones(10), blocks=1)
    with pytest.raises(ValueError):
        stats.mean_error(np.ones((4, 4, 4)))


# ------------------------------------------------------------------------- uwerr
def test_uwerr_finds_the_autocorrelation_time_and_the_error_of_ar1():
    for phi, tol in ((0.0, 0.05), (0.5, 0.05), (0.9, 0.1)):
        results = [stats.uwerr(ar1(20000, phi, 100 + s)) for s in range(20)]
        errors, taus = (np.array([r[i] for r in results]) for i in (1, 2))
        assert abs(taus.mean() / tau_exact(phi) - 1) < tol, (phi, taus.mean(), tau_exact(phi))
        exact_error = np.sqrt(1 / (1 - phi ** 2) * 2 * tau_exact(phi) / 20000)    # sqrt(variance * 2 tau / N)
        assert abs(errors.mean() / exact_error - 1) < 0.05, (phi, errors.mean(), exact_error)
        assert all(r[3] > 0 for r in results)


def test_uwerr_returns_four_floats_and_takes_replica():
    a, b = ar1(5000, 0.5, 1), ar1(5000, 0.5, 2)
    out = stats.uwerr(a)
    assert len(out) == 4 and all(type(v) is float for v in out)
    assert out == stats.tauint([[a]], 0)[:4]                                  # a series is one replicum of one observable
    assert stats.uwerr([a, b]) == stats.tauint([[a, b]], 0)[:4]               # a list of series: replica
    assert stats.uwerr(np.array([a, b])) == stats.uwerr([a, b])               # as a 2-d array too
    assert stats.uwerr(list(a)) == out                                        # and a list of numbers
    mean = stats.uwerr([a, b])[0]
    assert np.isclose(mean, np.concatenate([a, b]).mean())


def test_uwerr_refuses_what_it_cannot_do():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')                                            # a clear error, and no numpy warnings on the way
        for constant in (np.full(100, 2.0), [np.full(50, 1.0), np.full(50, 1.0)]):
            with pytest.raises(ValueError, match='constant'):
                stats.uwerr(constant)
    with pytest.raises(ValueError, match='empty'):
        stats.uwerr([])
    with pytest.raises(ValueError, match='constant'):
        stats.uwerr([1.0])                                                        # one point has no scatter
    with pytest.raises(ValueError, match='too short'):
        stats.uwerr([1.0, 2.0])                                                   # no window can be found
    with pytest.raises(ValueError, match='1-d'):
        stats.uwerr(np.ones((2, 3, 4)))


def test_uwerr_agrees_with_the_batch_means_error():
    # two different estimators of the same thing: the error of the mean of a correlated series
    x = ar1(40000, 0.8, 3, series=20)
    gamma_method = np.mean([stats.uwerr(x[:, k])[1] for k in range(20)])
    batch_means = np.mean([stats.mean_error(x[:, k])[1] for k in range(20)])
    assert abs(gamma_method / batch_means - 1) < 0.1


# ------------------------------------------------------------ the code of py-uwerr
def test_means_and_datainfo():
    data = [[np.linspace(0, 10, 20)], [np.linspace(10, 20, 20)]]
    assert np.allclose(stats.means(data), [5.0, 15.0])
    info = stats.DataInfo([[np.arange(20.0), np.arange(20.0)], [np.ones(20), np.ones(20)]])
    assert (info.nobs, info.R, info.Nr, info.N) == (2, 2, [20, 20], 40)
    with pytest.raises(stats.DataSanityCheckFail):
        stats.DataInfo([[np.arange(5.0)], [np.arange(6.0)]])                    # not the same number of measurements


def test_deriv_is_the_numerical_derivative():
    f = lambda a, b: a * a * b
    assert np.isclose(stats.deriv(f, 0, 1e-4)(3.0, 5.0), 2 * 3.0 * 5.0)
    assert np.isclose(stats.deriv(f, 1, 1e-4)(3.0, 5.0), 3.0 ** 2)
    assert stats.idf(1)(10, 20, 30) == 20


def test_gamma_is_the_autocovariance():
    x = ar1(2000, 0.7, 4)
    G, info, mean = stats.gamma([[x]], 0)
    y = x - x.mean()
    direct = np.array([np.sum(y[:len(y) - t] * y[t:]) / (len(y) - t) for t in range(len(y))])
    assert np.allclose(G, direct) and np.isclose(mean, x.mean()) and (info.R, info.N) == (1, 2000)
    assert np.isclose(G[0], x.var())


def test_tauint_full_output_and_the_window():
    x = ar1(20000, 0.9, 5)
    mean, error, tau, dtau, G, W = stats.tauint([[x]], 0, full_output=True)
    assert (mean, error, tau, dtau) == stats.tauint([[x]], 0)
    assert 1 <= W < len(G) and np.isclose(G[0], x.var())
    assert np.isclose(error, np.sqrt(2 * tau * G[0] / len(x)))                   # error = sqrt(2 tau var / N)
    assert W >= 4 * tau                                                          # the window is several tau


def test_the_error_of_a_function_of_two_observables():
    # f = a / b with independent a, b: sigma_f^2 = (sigma_a / mu_b)^2 + (mu_a sigma_b / mu_b^2)^2
    mu_a, mu_b, s_a, s_b, n = 2.0, 3.0, 0.5, 0.7, 20000
    predicted = np.sqrt((s_a / mu_b) ** 2 + (mu_a * s_b / mu_b ** 2) ** 2) / np.sqrt(n)
    results = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        data = [[mu_a + s_a * rng.standard_normal(n)], [mu_b + s_b * rng.standard_normal(n)]]
        results.append(stats.tauint(data, lambda a, b: a / b)[1:3])
    errors, taus = np.array(results).T
    assert abs(errors.mean() / predicted - 1) < 0.08, (errors.mean(), predicted)
    assert abs(taus.mean() - 0.5) < 0.05                                          # independent points


def test_the_numpy_error_settings_are_left_as_they_were():
    old = np.seterr(over='raise', under='ignore', divide='warn', invalid='warn')
    try:
        before = np.geterr()
        stats.uwerr(ar1(3000, 0.6, 6))
        stats.tauint([[ar1(3000, 0.6, 7)]], 0)
        assert np.geterr() == before
    finally:
        np.seterr(**old)


def test_no_warnings_from_the_code():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        stats.uwerr(ar1(5000, 0.6, 8))
        stats.tauint([[ar1(2000, 0.3, 9), ar1(2000, 0.3, 10)]], 0)


def test_plots_flag_draws_the_window_figure():
    import matplotlib.pyplot as plt
    shown = []
    original = stats.plt.show
    stats.plt.show = lambda *a, **k: shown.append(True)                           # no window in a test
    try:
        stats.tauint([[ar1(5000, 0.8, 11)]], 0, plots=True)
    finally:
        stats.plt.show = original
    assert shown == [True] and len(plt.get_fignums()) == 1
    assert len(plt.gcf().axes) == 2                                               # tau_int against W, and Gamma
    plt.close('all')


def test_correlated_data_has_the_layout_of_the_code():
    data = stats.correlated_data(5, 300)
    assert len(data) == 1 and len(data[0]) == 1 and data[0][0].shape == (300,)    # data[observable][replicum][measurement]
    assert np.array_equal(data[0][0], stats.correlated_data(5, 300)[0][0])         # the same every time
    assert not np.array_equal(data[0][0], stats.correlated_data(2, 300)[0][0])
