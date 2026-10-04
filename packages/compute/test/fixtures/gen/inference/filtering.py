"""Golden values for aifn-compute/timeseries from numpy and scipy (statsmodels is not installed).

References are computed independently of the algorithms under test where possible: Toeplitz solves for Levinson,
Yule–Walker and the PACF, dense multivariate-normal densities for the exact ARMA likelihood, and dense Gaussian
conditioning of the joint distribution for the Kalman filter and smoother.
"""

import numpy as np
import scipy.linalg as sla
from scipy.stats import multivariate_normal


def acov(x: np.ndarray, max_lag: int) -> np.ndarray:
    d = x - x.mean()
    n = len(x)
    return np.array([np.dot(d[: n - k], d[k:]) / n for k in range(max_lag + 1)])


def arma_acov(ar: list[float], ma: list[float], sigma: float, max_lag: int, terms: int = 5000) -> np.ndarray:
    psi = np.zeros(terms)
    psi[0] = 1.0
    for j in range(1, terms):
        v = ma[j - 1] if j - 1 < len(ma) else 0.0
        for i, phi in enumerate(ar):
            if j - 1 - i >= 0:
                v += phi * psi[j - 1 - i]
        psi[j] = v
    return np.array([sigma**2 * np.dot(psi[: terms - h], psi[h:]) for h in range(max_lag + 1)])


def simulate_arma(rng: np.random.Generator, ar: list[float], ma: list[float], n: int) -> np.ndarray:
    burn = 300
    e = rng.normal(size=n + burn)
    x = np.zeros(n + burn)
    for t in range(n + burn):
        v = e[t]
        for i, phi in enumerate(ar):
            if t - 1 - i >= 0:
                v += phi * x[t - 1 - i]
        for j, theta in enumerate(ma):
            if t - 1 - j >= 0:
                v += theta * e[t - 1 - j]
        x[t] = v
    return x[burn:]


def burg(x: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    d = x - x.mean()
    n = len(d)
    f = d.copy()
    b = d.copy()
    power = np.dot(d, d) / n
    phi = np.zeros(0)
    for m in range(1, order + 1):
        num = np.dot(f[m:], b[m - 1 : n - 1])
        den = np.dot(f[m:], f[m:]) + np.dot(b[m - 1 : n - 1], b[m - 1 : n - 1])
        k = 2 * num / den
        nf = f.copy()
        nb = b.copy()
        nf[m:] = f[m:] - k * b[m - 1 : n - 1]
        nb[m:] = b[m - 1 : n - 1] - k * f[m:]
        f, b = nf, nb
        phi = np.concatenate([phi - k * phi[::-1], [k]])
        power *= 1 - k * k
    return phi, power


def kalman_reference(
    a: np.ndarray, c: np.ndarray, q: np.ndarray, r: np.ndarray, m0: np.ndarray, p0: np.ndarray, y: np.ndarray
):
    """Exact filtered and smoothed moments by conditioning the joint Gaussian of (z_1..z_T, y_1..y_T)."""
    t_len, m = y.shape
    n = a.shape[0]
    # z_t = A^t z_0 + Σ_{s≤t} A^{t−s} w_s.
    mean_z = np.zeros((t_len, n))
    cov_z = np.zeros((t_len * n, t_len * n))
    powers = [np.linalg.matrix_power(a, k) for k in range(t_len + 1)]
    for t in range(t_len):
        mean_z[t] = powers[t + 1] @ m0
    for t in range(t_len):
        for s in range(t_len):
            block = powers[t + 1] @ p0 @ powers[s + 1].T
            for k in range(min(t, s) + 1):
                block = block + powers[t - k] @ q @ powers[s - k].T
            cov_z[t * n : (t + 1) * n, s * n : (s + 1) * n] = block
    big_c = np.kron(np.eye(t_len), c)
    mean_y = (big_c @ mean_z.reshape(-1)).reshape(t_len, m)
    cov_y = big_c @ cov_z @ big_c.T + np.kron(np.eye(t_len), r)
    cov_zy = cov_z @ big_c.T
    loglik = multivariate_normal(mean_y.reshape(-1), cov_y).logpdf(y.reshape(-1))  # pyright: ignore[reportArgumentType]  # scipy is unstubbed: cov inferred as int from its default

    def condition(k: int):
        idx = np.arange(k * m)
        gain = cov_zy[:, idx] @ np.linalg.inv(cov_y[np.ix_(idx, idx)])
        mu = mean_z.reshape(-1) + gain @ (y.reshape(-1)[idx] - mean_y.reshape(-1)[idx])
        cv = cov_z - gain @ cov_zy[:, idx].T
        return mu.reshape(t_len, n), cv

    filtered_mean = np.zeros((t_len, n))
    filtered_cov = np.zeros((t_len, n, n))
    for t in range(t_len):
        mu, cv = condition(t + 1)
        filtered_mean[t] = mu[t]
        filtered_cov[t] = cv[t * n : (t + 1) * n, t * n : (t + 1) * n]
    mu, cv = condition(t_len)
    smoothed_cov = np.array([cv[t * n : (t + 1) * n, t * n : (t + 1) * n] for t in range(t_len)])
    lag_cov = np.array([cv[t * n : (t + 1) * n, (t - 1) * n : t * n] for t in range(1, t_len)])
    return filtered_mean, filtered_cov, mu, smoothed_cov, lag_cov, loglik


def holt_winters(y: np.ndarray, alpha: float, beta: float, gamma: float, phi: float, m: int, horizon: int):
    """Additive damped Holt–Winters with the same initial-state heuristic as aifn (a direct re-implementation)."""
    first = y[:m].mean()
    second = y[m : 2 * m].mean()
    b = (second - first) / m
    s = [y[j] - (first + b * (j - (m - 1) / 2)) for j in range(m)]
    level = first - b * ((m - 1) / 2 + 1)
    fitted = []
    for v in y:
        base = level + phi * b
        s_old = s[0]
        fitted.append(base + s_old)
        new_level = alpha * (v - s_old) + (1 - alpha) * base
        new_b = beta * (new_level - level) + (1 - beta) * phi * b
        s = [*s[1:], gamma * (v - base) + (1 - gamma) * s_old]
        level, b = new_level, new_b
    forecast = []
    damp = 0.0
    for h in range(1, horizon + 1):
        damp += phi**h
        forecast.append(level + damp * b + s[(h - 1) % m])
    return np.array(fitted), np.array(forecast)


def garch_loglik(x: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    d = x - 0.0
    v = np.mean(d**2)
    prev2 = v
    ll = 0.0
    for r in d:
        v = omega + alpha * prev2 + beta * v
        ll += -0.5 * (np.log(2 * np.pi) + np.log(v) + r * r / v)
        prev2 = r * r
    return float(ll)


def cases() -> dict[str, object]:
    rng = np.random.default_rng(11)
    ar2 = simulate_arma(rng, [0.6, -0.3], [], 400)
    g = acov(ar2, 12)
    pacf = [sla.solve_toeplitz(g[:k], g[1 : k + 1])[-1] for k in range(1, 11)]
    yw = sla.solve_toeplitz(g[:3], g[1:4])
    burg_phi, burg_power = burg(ar2, 3)

    arma = simulate_arma(rng, [0.5], [0.4], 60)
    spec = {"ar": [0.5], "ma": [0.4], "sigma": 1.3}
    gam = arma_acov(spec["ar"], spec["ma"], spec["sigma"], 59)
    cov = sla.toeplitz(gam)
    exact = multivariate_normal(np.zeros(60), cov).logpdf(arma)  # pyright: ignore[reportArgumentType]  # scipy is unstubbed: cov inferred as int from its default

    # Kalman: constant velocity in 1-D with position observed, 12 steps; R with a deterministic part in 2-D below.
    a = np.array([[1.0, 1.0], [0.0, 1.0]])
    c = np.array([[1.0, 0.0]])
    q = np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]]) * 0.1
    r = np.array([[0.5]])
    m0 = np.array([0.0, 0.5])
    p0 = np.eye(2)
    t_len = 12
    z = m0.copy()
    ys = []
    for _ in range(t_len):
        z = a @ z + rng.multivariate_normal(np.zeros(2), q)
        ys.append(c @ z + rng.normal(scale=np.sqrt(0.5), size=1))
    y = np.array(ys)
    fm, fc, sm, sc, lc, ll = kalman_reference(a, c, q, r, m0, p0, y)

    t = np.arange(48)
    season = np.array([3.0, -1.0, -4.0, 2.0])
    hw_y = 10 + 0.3 * t + season[t % 4] + rng.normal(scale=0.5, size=48)
    fitted, forecast = holt_winters(hw_y, 0.4, 0.2, 0.3, 0.95, 4, 8)

    garch_x = rng.normal(size=200) * 0.8
    return {
        "ar2": ar2,
        "acf": g / g[0],
        "acov": g,
        "pacf": pacf,
        "yuleWalker": {"ar": yw, "sigma2": g[0] - np.dot(yw, g[1:4])},
        "burg": {"ar": burg_phi, "sigma2": burg_power},
        "armaAcov": {"spec": spec, "acov": arma_acov(spec["ar"], spec["ma"], spec["sigma"], 10)},
        "armaExact": {"x": arma, "spec": spec, "logLikelihood": exact},
        "kalman": {
            "A": a,
            "C": c,
            "Q": q,
            "R": r,
            "m0": m0,
            "P0": p0,
            "y": y,
            "filteredMean": fm,
            "filteredCov": fc,
            "smoothedMean": sm,
            "smoothedCov": sc,
            "lagOneCov": lc,
            "logLikelihood": ll,
        },
        "holtWinters": {
            "y": hw_y,
            "alpha": 0.4,
            "beta": 0.2,
            "gamma": 0.3,
            "phi": 0.95,
            "fitted": fitted,
            "forecast": forecast,
        },
        "garch": {
            "x": garch_x,
            "omega": 0.05,
            "alpha": 0.1,
            "beta": 0.85,
            "logLikelihood": garch_loglik(garch_x, 0.05, 0.1, 0.85),
        },
    }
