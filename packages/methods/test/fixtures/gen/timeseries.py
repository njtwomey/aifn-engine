"""Reference values for aifn-methods/timeseries by direct numpy/scipy computation (statsmodels is not installed).

- ARMA and seasonal ARIMA: the exact Gaussian log-likelihood as a dense multivariate normal density with the Toeplitz
  autocovariance of the (differenced) series, the autocovariance from ψ weights (scipy.signal.lfilter of an impulse);
  maximum-likelihood fits of the profile likelihood (σ² concentrated out, mean the sample mean as aifn does) by
  scipy.optimize from a grid of starts.
- GARCH(1,1): the variance recursion backcast at the sample variance, and its maximum-likelihood fit.
- Differencing: numpy.diff.
Series are simulated here with numpy and stored, so both sides see the same data.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, signal, stats
from scipy.linalg import toeplitz


def psi_weights(ar, ma, n):
    impulse = np.zeros(n)
    impulse[0] = 1
    return np.asarray(signal.lfilter(np.r_[1, ma], np.r_[1, -np.asarray(ar, float)], impulse))


def autocovariance(ar, ma, sigma, n, terms=4000):
    psi = psi_weights(ar, ma, terms + n)
    return sigma**2 * np.array([psi[:terms] @ psi[h : terms + h] for h in range(n)])


def exact_loglik(z, ar, ma, sigma=None):
    """log N(z; 0, Γ); with sigma None, the profile likelihood with σ² = zᵀΓ₁⁻¹z / n."""
    n = len(z)
    g1 = toeplitz(autocovariance(ar, ma, 1.0, n))
    if sigma is not None:
        return stats.multivariate_normal(np.zeros(n), sigma**2 * g1).logpdf(z)
    c = np.linalg.cholesky(g1)
    w = np.linalg.solve(c, z)
    s2 = w @ w / n
    return -0.5 * n * (np.log(2 * np.pi * s2) + 1) - np.log(np.diag(c)).sum()


def stationary(ar):
    return len(ar) == 0 or np.all(np.abs(np.roots(np.r_[-np.asarray(ar)[::-1], 1])) > 1)


def fit(
    negloglik: Callable[[NDArray[np.float64]], float],
    starts: Sequence[ArrayLike],
    bounds: Sequence[tuple[float | None, float | None]],
) -> optimize.OptimizeResult:
    best: optimize.OptimizeResult | None = None
    for s in starts:
        r = optimize.minimize(
            negloglik, s, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000}
        )
        r = optimize.minimize(negloglik, r.x, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-15, "gtol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None, "no starting points"
    return best


def simulate_arma(rng, ar, ma, sigma, n, mean=0.0, burn=500):
    e = sigma * rng.standard_normal(n + burn)
    return mean + np.asarray(signal.lfilter(np.r_[1, ma], np.r_[1, -np.asarray(ar, float)], e))[burn:]


def arma_cases(rng):
    out = []
    for ar_t, ma_t, n, mean in [([0.6], [0.3], 200, 2.0), ([0.5, -0.3], [], 150, 0.0), ([], [0.4, 0.25], 160, -1.0)]:
        x = simulate_arma(rng, ar_t, ma_t, 1.0, n, mean)
        p, q = len(ar_t), len(ma_t)
        z = x - x.mean()

        # Defaults bind this iteration's p and z; fit() calls nll only within the iteration.
        def nll(u: NDArray[np.float64], p: int = p, z: NDArray[np.float64] = z) -> float:
            ar, ma = u[:p], u[p:]
            if not stationary(ar) or not stationary(-np.asarray(ma)):
                return 1e10
            return -exact_loglik(z, ar, ma)

        starts = [np.r_[ar_t, ma_t], np.zeros(p + q), np.r_[ar_t, ma_t] * 0.5]
        best = fit(nll, starts, [(-0.99, 0.99)] * (p + q))
        spec = {"ar": [0.4] * p, "ma": [0.2] * q, "sigma": 1.1, "mean": float(mean)}
        out.append(
            {
                "x": x,
                "p": p,
                "q": q,
                "spec": spec,
                "logLikelihoodAtSpec": exact_loglik(x - mean, spec["ar"], spec["ma"], 1.1),
                "fit": {"ar": best.x[:p], "ma": best.x[p:], "logLikelihood": -best.fun, "mean": x.mean()},
            }
        )
    return out


def sarima_cases(rng):
    # The airline model SARIMA(0,1,1)(0,1,1)₁₂ on 144 points: w = ∇∇₁₂x is MA with (1 + θz)(1 + Θz¹²).
    theta, Theta, n, period = -0.4, -0.6, 144, 12
    seasonal = np.zeros(period + 1)
    seasonal[0], seasonal[period] = 1, Theta
    ma_full = np.convolve([1, theta], seasonal)  # 1 + θz + Θz¹² + θΘz¹³
    e = 0.5 * rng.standard_normal(n + 300)
    w = signal.lfilter(ma_full, [1], e)
    integ = np.convolve([1, -1], np.r_[1, np.zeros(period - 1), -1])  # (1 − z)(1 − z¹²)
    x = signal.lfilter([1], integ, w)[300:]
    d = np.diff(np.r_[x], 1)
    d = d[period:] - d[:-period]

    def ll(th, Th, sigma=None):
        s = np.zeros(period + 1)
        s[0], s[period] = 1, Th
        return exact_loglik(d, [], np.convolve([1, th], s)[1:], sigma)

    def nll(u):
        if abs(u[0]) >= 0.999 or abs(u[1]) >= 0.999:
            return 1e10
        return -ll(u[0], u[1])

    best = fit(nll, [[theta, Theta], [0, 0], [-0.2, -0.3]], [(-0.99, 0.99)] * 2)
    return [
        {
            "x": x,
            "period": period,
            "spec": {"ma": [-0.3], "seasonalMa": [-0.5], "sigma": 0.6},
            "logLikelihoodAtSpec": ll(-0.3, -0.5, 0.6),
            "fit": {"ma": best.x[0], "seasonalMa": best.x[1], "logLikelihood": -best.fun},
        }
    ]


def garch_loglik(r, omega, alpha, beta, mean):
    d = r - mean
    v = prev = d @ d / len(d)
    ll = 0.0
    for x in d:
        v = omega + alpha * prev + beta * v
        ll += -0.5 * (np.log(2 * np.pi) + np.log(v) + x * x / v)
        prev = x * x
    return ll


def garch_cases(rng):
    n, omega, alpha, beta = 1500, 0.1, 0.1, 0.85
    e = rng.standard_normal(n + 500)
    v, prev, r = omega / (1 - alpha - beta), omega / (1 - alpha - beta), []
    for t in range(n + 500):
        v = omega + alpha * prev + beta * v
        x = np.sqrt(v) * e[t]
        prev = x * x
        r.append(x)
    r = np.array(r[500:]) + 0.05
    mean = r.mean()

    def nll(u):
        om, a, b = u
        if om <= 0 or a < 0 or b < 0 or a + b >= 1:
            return 1e10
        return -garch_loglik(r, om, a, b, mean)

    best = fit(nll, [[omega, alpha, beta], [0.05, 0.05, 0.9], [0.2, 0.15, 0.7]], [(1e-8, None), (0, 1), (0, 1)])
    spec = {"omega": 0.2, "alpha": 0.15, "beta": 0.7, "mean": 0.0}
    return [
        {
            "r": r,
            "spec": spec,
            "logLikelihoodAtSpec": garch_loglik(r, 0.2, 0.15, 0.7, 0.0),
            "fit": {
                "omega": best.x[0],
                "alpha": best.x[1],
                "beta": best.x[2],
                "mean": mean,
                "logLikelihood": -best.fun,
            },
        }
    ]


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(20261001)
    x = rng.standard_normal(30).cumsum()
    return {
        "arma": arma_cases(rng),
        "sarima": sarima_cases(rng),
        "garch": garch_cases(rng),
        "difference": [
            {"x": x, "lag": 1, "order": 1, "value": np.diff(x)},
            {"x": x, "lag": 1, "order": 2, "value": np.diff(x, 2)},
            {"x": x, "lag": 4, "order": 1, "value": x[4:] - x[:-4]},
        ],
    }
