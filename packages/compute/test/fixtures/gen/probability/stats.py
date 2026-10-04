"""Golden values for aifn-compute/stats, from numpy and scipy (direct sums where neither has the function), and the
minimum
covariance determinant from scikit-learn (gen/probability/_robust.py)."""

import sys
from pathlib import Path
from typing import Any, Literal, Protocol, cast, get_args

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _robust import mcd_cases

QuantileMethod = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]
QUANTILE_METHODS: list[QuantileMethod] = list(get_args(QuantileMethod))


class Statistic(Protocol):
    """scipy's correlation results are built dynamically, so Pyright cannot see their `statistic`."""

    statistic: Any


def statistic(result: object) -> Any:
    return cast(Statistic, result).statistic


def lagged(x: np.ndarray, y: np.ndarray, max_lag: int, adjusted: bool = False) -> np.ndarray:
    """(1/n) Σ_t (x_t − x̄)(y_{t+k} − ȳ) for k = −max_lag … max_lag, by direct summation."""
    n = len(x)
    a, b = x - x.mean(), y - y.mean()
    out = []
    for k in range(-max_lag, max_lag + 1):
        s = sum(a[t] * b[t + k] for t in range(max(0, -k), min(n, n - k)))
        out.append(s / (n - abs(k) if adjusted else n))
    return np.array(out)


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    x = rng.gamma(2.0, 1.5, size=37)
    y = 0.6 * x + rng.normal(0, 1, size=37)
    w = rng.uniform(0.1, 2.0, size=37)
    ties = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0, 9.0, 7.0, 9.0, 3.0])
    ties_y = np.array([2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0, 2.0, 8.0, 4.0, 5.0, 9.0, 0.0, 4.0, 5.0])
    fweights = rng.integers(1, 5, size=37)

    probabilities = np.array([0.0, 0.01, 0.1, 0.25, 1 / 3, 0.5, 0.6, 0.75, 0.9, 0.99, 1.0])
    samples = {
        "x": x,
        "ties": ties,
        "small": np.array([7.0, 1.0, 3.0, 5.0]),
        "one": np.array([2.5]),
        "two": np.array([1.0, 4.0]),
    }
    # Grids hitting exact order statistics: k/n and k/(n−1) for the 4-sample.
    exact = np.array([k / 4 for k in range(5)] + [k / 3 for k in range(4)] + [0.125, 0.375, 0.625, 0.875])
    quantiles = {
        name: {
            "x": data,
            "q": np.concatenate([probabilities, exact]),
            "methods": {
                m.replace("_", "-"): np.quantile(data, np.concatenate([probabilities, exact]), method=m)
                for m in QUANTILE_METHODS
            },
        }
        for name, data in samples.items()
    }

    histograms = {
        "count": {"x": x, "bins": 7, **hist(x, 7)},
        "sturges": {"x": x, "bins": "sturges", **hist(x, "sturges")},
        "freedman-diaconis": {"x": x, "bins": "freedman-diaconis", **hist(x, "fd")},
        "range": {"x": x, "bins": 5, "range": [1.0, 4.0], **hist(x, 5, (1.0, 4.0))},
        "edges": {"x": x, "bins": [0.0, 0.5, 2.0, 3.5, 8.0], **hist(x, np.array([0.0, 0.5, 2.0, 3.5, 8.0]))},
        "weighted": {"x": x, "bins": 6, "weights": w, **hist(x, 6, weights=w)},
        "ties-at-last-edge": {"x": ties, "bins": 4, **hist(ties, 4)},
        "constant": {"x": np.array([2.0, 2.0, 2.0]), "bins": 3, **hist(np.array([2.0, 2.0, 2.0]), 3)},
        "sturges-integers": {"x": ties, "bins": "sturges", **hist(ties, "sturges")},
    }

    grid = np.linspace(-2, 12, 29)
    kdes = {
        rule: {"bandwidth": float(np.sqrt(k.covariance[0, 0])), "density": k(grid)}
        for rule in ("scott", "silverman")
        for k in [stats.gaussian_kde(x, bw_method=rule)]
    }
    kw = stats.gaussian_kde(x, weights=w)
    kdes["weighted"] = {"bandwidth": float(np.sqrt(kw.covariance[0, 0])), "density": kw(grid)}

    e = stats.ecdf(ties)
    series = np.cumsum(rng.normal(0, 1, size=80)) + 0.3 * rng.normal(0, 1, size=80)
    other = np.roll(series, 3) + rng.normal(0, 0.5, size=80)
    acov = lagged(series, series, 20)[20:]
    return {
        "mcd": mcd_cases(),
        "x": x,
        "y": y,
        "w": w,
        "fweights": fweights,
        "ties": ties,
        "tiesY": ties_y,
        "moments": {
            "mean": x.mean(),
            "var": x.var(),
            "varSample": x.var(ddof=1),
            "std": x.std(),
            "stdSample": x.std(ddof=1),
            "weightedMean": np.average(x, weights=w),
            "weightedVarPopulation": np.cov(x, aweights=w, ddof=0),
            "weightedVarReliability": np.cov(x, aweights=w, ddof=1),
            "weightedVarFrequency": np.cov(x, fweights=fweights),
            "skew": stats.skew(x),
            "skewUnbiased": stats.skew(x, bias=False),
            "kurtosis": stats.kurtosis(x),
            "kurtosisPearson": stats.kurtosis(x, fisher=False),
            "kurtosisUnbiased": stats.kurtosis(x, bias=False),
            "cov": np.cov(x, y, ddof=0)[0, 1],
            "covSample": np.cov(x, y)[0, 1],
            "pearson": statistic(stats.pearsonr(x, y)),
            "spearman": statistic(stats.spearmanr(x, y)),
            "kendall": statistic(stats.kendalltau(x, y)),
            "spearmanTies": statistic(stats.spearmanr(ties, ties_y)),
            "kendallTies": statistic(stats.kendalltau(ties, ties_y)),
            "zscore": stats.zscore(x),
            "zscoreSample": stats.zscore(x, ddof=1),
            "median": np.median(x),
            "iqr": stats.iqr(x),
            "mode": {"value": stats.mode(ties).mode, "count": stats.mode(ties).count},
        },
        "ranks": {m: stats.rankdata(ties, method=m) for m in ("average", "min", "max", "dense", "ordinal")},
        "quantiles": quantiles,
        "histograms": histograms,
        "kde": {"grid": grid, **kdes},
        "ecdf": {"values": e.cdf.quantiles, "probabilities": e.cdf.probabilities},
        "series": {
            "x": series,
            "y": other,
            "autocovariance": acov,
            "autocovarianceAdjusted": lagged(series, series, 20, adjusted=True)[20:],
            "autocorrelation": acov / acov[0],
            "crossCovariance": lagged(series, other, 6),
            "crossCorrelation": lagged(series, other, 6) / (series.std() * other.std()),
        },
    }


def hist(
    x: np.ndarray,
    bins: object,
    range_: tuple[float, float] | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, object]:
    counts, edges = np.histogram(x, bins=bins, range=range_, weights=weights)  # pyright: ignore[reportArgumentType, reportCallIssue]
    density, _ = np.histogram(x, bins=bins, range=range_, weights=weights, density=True)  # pyright: ignore[reportArgumentType, reportCallIssue]
    return {"edges": edges, "counts": counts.astype(float), "density": density}
