"""Golden values for aifn-methods theory/adaptive-holdout: the expected maximum of J standard normals.

E max = integral of x J phi(x) Phi(x)^(J - 1) over the real line, the mean against the density of the maximum,
integrated by scipy's quad with log-space powers. The engine integrates the distribution function instead, so the two
share no formula. J = 2 has the closed form 1 / sqrt(pi).
"""

import itertools

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def expected_maximum(j: float) -> float:
    """E max of j independent standard normals, by quadrature of x times the density of the maximum."""

    def density(x: float) -> float:
        return x * j * np.exp(norm.logpdf(x) + (j - 1) * norm.logcdf(x))

    mode = np.sqrt(2 * np.log(max(j, 2.0)))
    edges = [-np.inf, -10.0, 0.0, mode, mode + 12, np.inf]
    return sum(quad(density, a, b, epsabs=1e-15, epsrel=1e-13, limit=500)[0] for a, b in itertools.pairwise(edges))


def cases() -> dict[str, object]:
    js = [1, 1.5, 2, 3, 5, 10, 20, 50, 100, 1000, 10_000, 1_000_000]
    return {"expected_maximum": [{"J": j, "value": expected_maximum(float(j))} for j in js]}
