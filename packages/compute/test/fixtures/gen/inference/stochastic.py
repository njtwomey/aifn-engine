"""Golden values for aifn-compute/mcmc diagnostics.

ArviZ is not a dependency, so its estimators (arviz.stats.diagnostics: _ess, _rhat, _z_scale, _split_chains,
_ess_tail, _mcse_mean, _mcse_quantile, _rhat_rank) are transcribed here from ArviZ 0.17 in plain numpy/scipy and
applied to fixed AR(1) chains, heavy-tailed chains and chains with shifted means.
"""

import numpy as np
from scipy import stats


def _autocov(x: np.ndarray) -> np.ndarray:
    """Biased (divide by n) autocovariance of each row, all lags, by direct summation."""
    n = x.shape[-1]
    d = x - x.mean(axis=-1, keepdims=True)
    return np.stack([np.array([np.dot(r[: n - k], r[k:]) / n for k in range(n)]) for r in d])


def _ess(ary: np.ndarray) -> float:
    ary = np.atleast_2d(np.asarray(ary, dtype=float))
    if np.max(ary) - np.min(ary) < np.finfo(float).resolution:
        return float(ary.size)
    n_chain, n_draw = ary.shape
    acov = _autocov(ary)
    chain_mean = ary.mean(axis=1)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += np.var(chain_mean, ddof=1)
    rho_hat_t = np.zeros(n_draw)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho_hat_t[1] = rho_hat_odd
    t = 1
    while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2
    max_t = t - 2
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2
    ess = n_chain * n_draw
    tau_hat = -1.0 + 2.0 * np.sum(rho_hat_t[: max_t + 1]) + np.sum(rho_hat_t[max_t + 1 : max_t + 2])
    tau_hat = max(tau_hat, 1 / np.log10(ess))
    return float(ess / tau_hat)


def _split(ary: np.ndarray) -> np.ndarray:
    half = ary.shape[1] // 2
    return np.vstack([ary[:, :half], ary[:, -half:]])


def _z_scale(ary: np.ndarray) -> np.ndarray:
    size = ary.size
    rank = stats.rankdata(ary, method="average").reshape(ary.shape)
    return stats.norm.ppf((rank - 0.375) / (size + 0.25))


def _rhat(ary: np.ndarray) -> float:
    n = ary.shape[1]
    between = n * np.var(ary.mean(axis=1), ddof=1)
    within = np.mean(np.var(ary, axis=1, ddof=1))
    return float(np.sqrt((between / within + n - 1) / n))


def _rhat_rank(ary: np.ndarray) -> float:
    s = _split(ary)
    bulk = _rhat(_z_scale(s))
    folded = np.abs(s - np.median(s))
    return max(bulk, _rhat(_z_scale(folded)))


def _ess_tail(ary: np.ndarray) -> float:
    def q(p: float) -> float:
        quant = np.quantile(ary, p)
        return _ess(_split((ary <= quant).astype(float)))

    return min(q(0.05), q(0.95))


def _mcse_quantile(ary: np.ndarray, prob: float) -> float:
    """ArviZ's _mcse_quantile: half the width of the ±1 sd interval of the quantile's order statistic, from a Beta
    approximation with the ESS of the indicator I(x ≤ q̂) on split chains."""
    quant = np.quantile(ary, prob)
    ess = _ess(_split((ary <= quant).astype(float)))
    p = np.array([0.1586553, 0.8413447])
    a = stats.beta.ppf(p, ess * prob + 1, ess * (1 - prob) + 1)
    s = np.sort(ary.ravel())
    size = s.size
    th1 = s[int(np.rint(np.nanmax((a[0] * size, 0))))]
    th2 = s[int(np.rint(np.nanmin((a[1] * size, size - 1))))]
    return float((th2 - th1) / 2)


def _ar1(rng: np.random.Generator, m: int, n: int, phi: float) -> np.ndarray:
    out = np.zeros((m, n))
    for j in range(m):
        x = rng.standard_normal()
        for t in range(n):
            x = phi * x + np.sqrt(1 - phi * phi) * rng.standard_normal()
            out[j, t] = x
    return out


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    chains = {
        "ar1": _ar1(rng, 4, 301, 0.8),
        "iid": rng.standard_normal((3, 200)),
        "antithetic": _ar1(rng, 2, 250, -0.6),
        "heavy": rng.standard_cauchy((4, 150)),
        "shifted": _ar1(rng, 4, 200, 0.5) + np.array([[0.0], [0.0], [0.0], [1.5]]),
        "one": _ar1(rng, 1, 400, 0.9),
    }
    out: dict[str, object] = {}
    for name, x in chains.items():
        n = x.shape[1]
        out[name] = {
            "draws": x,
            "ess_bulk": _ess(_z_scale(_split(x))),
            "ess_mean": _ess(_split(x)),
            "ess_tail": _ess_tail(x),
            "iact": x.size / _ess(x),
            "rhat_rank": _rhat_rank(x),
            "rhat_split": _rhat(_split(x)),
            "rhat_basic": _rhat(x) if x.shape[0] > 1 else float("nan"),
            "mcse_mean": float(np.std(x, ddof=1) / np.sqrt(_ess(_split(x)))),
            "mcse_quantile": [_mcse_quantile(x, p) for p in (0.05, 0.5, 0.95)],
            "acf": (_autocov(x[:1])[0] / _autocov(x[:1])[0][0])[: min(n, 30)],
        }
    return out
