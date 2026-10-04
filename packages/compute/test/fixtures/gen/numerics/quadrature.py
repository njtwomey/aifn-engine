"""Golden values for aifn-compute/numerics/quadrature: integrals from scipy.integrate.quad at tight tolerance (finite
and
infinite limits, smooth, peaked, oscillatory, kinked and endpoint-singular integrands), Gauss rules from numpy and
scipy.special, composite rules from scipy.integrate.{trapezoid,simpson}, unscrambled Halton and Sobol points from
scipy.stats.qmc, and box integrals for Monte Carlo from scipy.integrate.nquad. The integrands are named; the test holds
the same expressions under the same names."""

import warnings

import numpy as np
from numpy.polynomial import hermite, hermite_e, laguerre, legendre
from scipy import integrate, special, stats

# name → (f, a, b, singular): `singular` integrands have an integrable endpoint singularity (Romberg is skipped).
INTEGRANDS = {
    "exp": (np.exp, 0.0, 1.0, False),
    "cos": (np.cos, -1.0, 2.0, False),
    "poly7": (lambda x: 3 * x**7 - 2 * x**4 + x - 5, -1.5, 2.0, False),
    "runge": (lambda x: 1 / (1 + 25 * x**2), -1.0, 1.0, False),
    "oscillatory": (lambda x: np.sin(20 * x) * np.exp(-x), 0.0, 3.0, False),
    "gaussian": (lambda x: np.exp(-0.5 * x**2), -3.0, 4.0, False),
    "kink": (lambda x: np.abs(x - 0.3), 0.0, 1.0, False),
    "sqrt": (np.sqrt, 0.0, 1.0, False),
    "power15": (lambda x: x**1.5, 0.0, 2.0, False),
    "log": (lambda x: np.log(x) if x > 0 else 0.0, 0.0, 1.0, True),
    "inverseSqrt": (lambda x: 1 / np.sqrt(x) if x > 0 else 0.0, 0.0, 1.0, True),
    "logistic": (lambda x: 1 / (1 + np.exp(-x)), -10.0, 10.0, False),
    "reversed": (np.sin, np.pi, 0.0, False),
}

INFINITE = {
    "gaussianLine": (lambda x: np.exp(-(x**2)), -np.inf, np.inf),
    "cauchyHalf": (lambda x: 1 / (1 + x**2), 0.0, np.inf),
    "gammaThree": (lambda x: x**2 * np.exp(-x), 0.0, np.inf),
    "dampedCos": (lambda x: np.exp(-x) * np.cos(x), 0.0, np.inf),
    "leftTail": (lambda x: np.exp(x), -np.inf, 1.0),
    "normalCdf": (lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), -np.inf, -2.0),
}


def quad(f, a, b) -> float:
    # At 1e-13 quad warns of roundoff on some integrands; its value is still exact to ~1e-15 relative.
    warnings.simplefilter("ignore", integrate.IntegrationWarning)
    value, _ = integrate.quad(f, a, b, epsabs=0, epsrel=1e-13, limit=1000)
    return float(value)


def cases() -> dict[str, object]:
    out: dict[str, object] = {}
    finite = {
        name: {"a": a, "b": b, "value": quad(f, a, b), "singular": s} for name, (f, a, b, s) in INTEGRANDS.items()
    }
    # Exact values where known, to keep the reference honest.
    finite["exp"]["value"] = np.e - 1
    finite["sqrt"]["value"] = 2 / 3
    finite["log"]["value"] = -1.0
    finite["inverseSqrt"]["value"] = 2.0
    finite["kink"]["value"] = (0.3**2 + 0.7**2) / 2
    out["integrals"] = finite
    out["infinite"] = {name: {"a": a, "b": b, "value": quad(f, a, b)} for name, (f, a, b) in INFINITE.items()}

    # One group per registered algorithm, naming the integrands it is held to (Romberg only on smooth ones).
    smooth = [k for k in ("exp", "cos", "poly7", "runge", "oscillatory", "gaussian", "logistic", "reversed")]
    out["adaptiveSimpson"] = {"integrands": [k for k, v in INTEGRANDS.items() if not v[3]]}
    out["gaussKronrod"] = {"integrands": list(INTEGRANDS)}
    out["romberg"] = {"integrands": smooth}

    rules: dict[str, object] = {}
    rules["legendre"] = {
        str(n): dict(zip(("nodes", "weights"), legendre.leggauss(n), strict=True)) for n in (1, 2, 7, 20, 64)
    }
    rules["hermite"] = {
        str(n): dict(zip(("nodes", "weights"), hermite.hermgauss(n), strict=True)) for n in (1, 3, 20, 60)
    }
    rules["hermiteProbabilists"] = {
        str(n): dict(zip(("nodes", "weights"), hermite_e.hermegauss(n), strict=True)) for n in (2, 10, 40)
    }
    rules["laguerre"] = {
        str(n): dict(zip(("nodes", "weights"), laguerre.laggauss(n), strict=True)) for n in (1, 4, 15, 40)
    }
    gen = {}
    for alpha in (-0.5, 1.5, 4.0):
        for n in (3, 12):
            x, w = special.roots_genlaguerre(n, alpha)
            gen[f"{alpha}/{n}"] = {"alpha": alpha, "n": n, "nodes": x, "weights": w}
    rules["genLaguerre"] = gen
    # Gauss–Legendre on [a, b] and normal expectations.
    rules["integrateGauss"] = {"a": -0.5, "b": 2.5, "n": 12, "value": quad(lambda x: np.exp(np.sin(x)), -0.5, 2.5)}
    rules["normalExpectation"] = {
        "mean": 0.7,
        "sd": 1.3,
        "value": quad(lambda x: np.cos(x) * stats.norm.pdf(x, 0.7, 1.3), -np.inf, np.inf),
    }
    out["rules"] = rules

    # Composite rules on equal panels, and trapezoid on given samples.
    xs = np.linspace(0.2, 2.2, 13)
    rng = np.random.default_rng(20261001)
    xu = np.sort(rng.uniform(0, 3, 11))
    out["composite"] = {
        "a": 0.2,
        "b": 2.2,
        "n": 12,
        "trapezoid": integrate.trapezoid(np.exp(np.sin(xs)), xs),
        "simpson": integrate.simpson(np.exp(np.sin(xs)), x=xs),
        "samplesX": xu,
        "samplesY": np.cos(xu) ** 2,
        "trapezoidSamples": integrate.trapezoid(np.cos(xu) ** 2, xu),
    }

    out["halton"] = {"n": 20, "d": 6, "points": stats.qmc.Halton(6, scramble=False).random(20)}
    out["sobol"] = {"n": 64, "d": 12, "points": stats.qmc.Sobol(12, scramble=False).random(64)}

    # Box integrals: ∫ over [lo, hi] of named functions of x ∈ ℝᵈ.
    def box(f, lo, hi):
        return float(
            integrate.nquad(lambda *x: f(np.array(x)), list(zip(lo, hi, strict=True)), opts={"epsrel": 1e-12})[0]
        )

    out["monteCarlo"] = {
        "expSum2": {"lo": [0, 0], "hi": [1, 1], "value": (np.e - 1) ** 2},
        "gaussian3": {
            "lo": [-1, 0, -2],
            "hi": [2, 1, 1],
            "value": box(lambda x: np.exp(-0.5 * np.sum(x**2)), [-1, 0, -2], [2, 1, 1]),
        },
        "product4": {
            "lo": [0, 0, 0, 0],
            "hi": [1, 2, 1, 1],
            "value": (1 - np.cos(1)) * (1 - np.cos(2)) * (1 - np.cos(1)) ** 2,
        },
    }
    out["integrate2d"] = {
        "x": [0, 1.5],
        "y": [-1, 1],
        "value": integrate.dblquad(lambda y, x: np.exp(-x * y) * np.cos(x + y), 0, 1.5, -1, 1, epsabs=0, epsrel=1e-13)[
            0
        ],
    }
    return out
