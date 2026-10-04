"""Reference values for aifn-compute/probability/extremes from scipy.stats: maximum-likelihood generalised Pareto fits
(genpareto.fit with the location fixed at 0) of excess samples with a heavy, an exponential-like and a bounded tail,
the peaks-over-threshold quantities built on them, and the mean excess function."""

from typing import Any

import numpy as np
from scipy import stats


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    fits: list[dict[str, Any]] = []
    for name, shape, scale, n in [("heavy", 0.35, 1.5, 400), ("light", 0.02, 0.8, 300), ("bounded", -0.3, 2.0, 300)]:
        y = stats.genpareto(shape, 0, scale).rvs(n, random_state=rng)
        c, _, s = stats.genpareto.fit(y, floc=0)
        ll = float(np.sum(stats.genpareto.logpdf(y, c, 0, s)))
        fits.append({"name": name, "excesses": y, "shape": c, "scale": s, "logLikelihood": ll})
    # Peaks over threshold of a Student-t sample at its 0.9 quantile (NumPy's linear interpolation).
    x = stats.t(4).rvs(1000, random_state=rng)
    u = float(np.quantile(x, 0.9))
    exc = x[x > u] - u
    c, _, s = stats.genpareto.fit(exc, floc=0)
    rate = len(exc) / len(x)
    probes = [u + 0.5, u + 2.0, u + 6.0]
    risks = [0.05, 0.01, 1e-3]
    pot = {
        "values": x,
        "threshold": u,
        "shape": c,
        "scale": s,
        "rate": rate,
        "probes": probes,
        "tailProbability": [rate * float(stats.genpareto(c, u, s).sf(p)) for p in probes],
        "risks": risks,
        "tailQuantile": [float(stats.genpareto(c, u, s).isf(q / rate)) for q in risks],
    }
    thresholds = [-1.0, 0.0, 1.0, 2.5]
    mean_excess = [float(np.mean(x[x > t] - t)) for t in thresholds]
    return {"fits": fits, "pot": pot, "meanExcess": {"thresholds": thresholds, "values": mean_excess}}
