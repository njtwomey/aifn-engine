"""Golden values for aifn-methods theory/shrinkage: the exact risk of the James-Stein estimator.

The risk is sigma^2 (p - k^2 E[1 / X]) with X noncentral chi-squared, k = p - 2 and p degrees of freedom towards a fixed
target, k = p - 3 and p - 1 degrees of freedom towards the grand mean. E[1 / X] is integrated numerically against
scipy's noncentral (or, at noncentrality 0, central) chi-squared density, independently of the Poisson-mixture series
the engine sums.
"""

import itertools

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2, ncx2


def inverse_mean(df: int, nc: float) -> float:
    """E[1 / X] for X noncentral chi-squared with df degrees of freedom and noncentrality nc, by quadrature."""
    pdf = (lambda v: chi2.pdf(v, df)) if nc == 0 else (lambda v: ncx2.pdf(v, df, nc))
    # On (0, 1) substitute v = u^2, which removes the 1 / v singularity: the integrand becomes 2 pdf(u^2) / u.
    head = quad(lambda u: 2 * pdf(u * u) / u, 0, 1, epsabs=1e-17, epsrel=1e-13, limit=500)[0]
    # Beyond 1, break the range at the bulk of the density, which a large noncentrality pushes far out.
    mean, sd = df + nc, np.sqrt(2 * (df + 2 * nc))
    edges = sorted({1.0, *(max(1.0, mean + z * sd) for z in (-12, -4, 0, 4, 12, 40))})
    tail = sum(
        quad(lambda v: pdf(v) / v, a, b, epsabs=0, epsrel=1e-13, limit=500)[0] for a, b in itertools.pairwise(edges)
    )
    tail += quad(lambda v: pdf(v) / v, edges[-1], np.inf, epsabs=0, epsrel=1e-13, limit=500)[0]
    return head + tail


def cases() -> dict[str, object]:
    out = []
    for p, target, sigma2 in [(3, "fixed", 1.0), (10, "fixed", 1.0), (10, "mean", 1.0), (25, "fixed", 2.5),
                              (6, "mean", 0.5)]:  # fmt: skip
        k, df = (p - 3, p - 1) if target == "mean" else (p - 2, p)
        distances = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0]
        risks = [sigma2 * (p - k * k * inverse_mean(df, dist * dist / sigma2)) for dist in distances]
        out.append({"p": p, "target": target, "sigma2": sigma2, "distances": distances, "risk": risks})
    return {"exact_risk": out}
