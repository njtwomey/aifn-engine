"""Reference values for aifn-methods/optimisation/tree-parzen: the good and bad Parzen densities l and g of TPE.

For fixed observations (points and losses), the best ceil(gamma n) form the good group. Each group's density is a
mixture, weights 1/(N + 1) each, of a prior N(1/2, 1) and one Gaussian per observation, every one truncated to the unit
interval (scipy.stats.truncnorm); in two dimensions each component is a product of truncated Gaussians. Bandwidths:
Scott's rule as in Watanabe (2023), 1.059 min(sd, IQR / 1.34) N^(-1/5) (sd with ddof 1, numpy's linear
percentiles), or hyperopt's neighbour rule, clipped to [max(0.03, (N + 1)^-2), 1]. A log-scale dimension is modelled in
u = log(x / lo) / log(hi / lo).
"""

import math

import numpy as np
from scipy.stats import truncnorm


def bandwidths(x: np.ndarray, rule: str) -> np.ndarray:
    n = len(x)
    if rule == "scott":
        if n < 2:
            raw = np.zeros(n)
        else:
            sd = np.std(x, ddof=1)
            iqr = np.percentile(x, 75) - np.percentile(x, 25)
            spread = min(sd, iqr / 1.34) if iqr > 0 else sd
            raw = np.full(n, 1.059 * spread * n**-0.2)
    else:
        order = np.argsort(x, kind="stable")
        raw = np.ones(n)
        for r, i in enumerate(order):
            left = x[i] - x[order[r - 1]] if r > 0 else -np.inf
            right = x[order[r + 1]] - x[i] if r < n - 1 else -np.inf
            raw[i] = 1.0 if n == 1 else max(left, right)
    return np.clip(raw, max(0.03, (n + 1) ** -2.0), 1.0)


def kernel(x: np.ndarray, mu: float, b: float) -> np.ndarray:
    return truncnorm.pdf(x, (0 - mu) / b, (1 - mu) / b, loc=mu, scale=b)


def density(rows: np.ndarray, points: np.ndarray, rule: str) -> np.ndarray:
    """rows: N x m observations; points: P x m evaluation points."""
    N, m = rows.shape
    bws = [bandwidths(rows[:, j], rule) for j in range(m)]
    out = np.ones(len(points)) / (N + 1)
    for j in range(m):
        out = out * kernel(points[:, j], 0.5, 1.0)
    for i in range(N):
        comp = np.ones(len(points)) / (N + 1)
        for j in range(m):
            comp = comp * kernel(points[:, j], rows[i, j], bws[j][i])
        out = out + comp
    return out


def case(u: np.ndarray, losses: np.ndarray, gamma: float, rule: str, grid: np.ndarray) -> dict:
    n = len(losses)
    order = np.argsort(losses, kind="stable")
    n_good = max(1, math.ceil(gamma * n - 1e-12))
    good, bad = order[:n_good], order[n_good:]
    return {
        "gamma": gamma,
        "rule": rule,
        "losses": losses,
        "good": good,
        "l": density(u[good], grid, rule),
        "g": density(u[bad], grid, rule),
    }


def cases() -> dict:
    rng = np.random.default_rng(2011)
    x1 = rng.uniform(0, 1, 15)
    loss1 = (x1 - 0.3) ** 2 + 0.01 * rng.normal(size=15)
    grid1 = np.linspace(0, 1, 11)[:, None]
    lo, hi = 1e-3, 1.0
    x_log = np.exp(rng.uniform(np.log(lo), np.log(hi), 12))
    u_log = np.log(x_log / lo) / np.log(hi / lo)
    loss_log = (np.log10(x_log) + 2) ** 2
    x2 = rng.uniform(0, 1, (20, 2))
    loss2 = (x2[:, 0] - 0.7) ** 2 + 2 * (x2[:, 1] - 0.2) ** 2
    g = np.linspace(0, 1, 5)
    grid2 = np.array([[a, b] for a in g for b in g])
    return {
        "scott": {"x": x1, **case(x1[:, None], loss1, 0.25, "scott", grid1)},
        "hyperopt": {"x": x1, **case(x1[:, None], loss1, 0.25, "hyperopt", grid1)},
        "log": {"x": x_log, **case(u_log[:, None], loss_log, 0.25, "scott", grid1)},
        "joint": {"x": x2, **case(x2, loss2, 0.2, "scott", grid2)},
    }
