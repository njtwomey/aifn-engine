"""Golden values for aifn-compute/numerics/roots: scalar roots from scipy.optimize.brentq at xtol 1e-15 (with brentq's
own
iteration and call counts), roots of systems from scipy.optimize.root (MINPACK hybrd) at tight tolerance, fixed points
from scipy.optimize.fixed_point, and minima from scipy.optimize.minimize_scalar (Brent and bounded). The functions are
named; the test holds the same expressions under the same names."""

from typing import Protocol, cast

import numpy as np
from scipy import optimize, special


class _ScalarMinimum(Protocol):
    """The fields read from minimize_scalar's OptimizeResult (declared as object)."""

    x: float
    fun: float


SCALAR = {
    "cubic": (lambda x: x**3 - 2 * x - 5, 2.0, 3.0, 2.0),
    "cosMinusX": (lambda x: np.cos(x) - x, 0.0, 1.0, 0.5),
    "kepler": (lambda x: x - 0.9 * np.sin(x) - 0.5, 0.0, np.pi, 1.0),
    "lambert": (lambda x: x * np.exp(x) - 3, 0.0, 2.0, 1.0),
    "logShift": (lambda x: np.log(x) + x - 2, 0.5, 3.0, 1.0),
    "tanhShift": (lambda x: np.tanh(x - 0.3), -5.0, 10.0, 0.0),
    "wilkinson": (lambda x: np.prod([x - k for k in range(1, 6)]), 2.5, 3.4, 3.2),
    "steepExp": (lambda x: np.exp(10 * x) - 1e3, 0.0, 1.0, 0.8),
}


def broyden_tridiagonal(x):
    xp = np.r_[0.0, x[:-1]]
    xn = np.r_[x[1:], 0.0]
    return (3 - 2 * x) * x - xp - 2 * xn + 1


SYSTEMS = {
    "circleLine": (lambda v: [v[0] ** 2 + v[1] ** 2 - 4, v[0] - v[1] - 0.5], [1.0, 1.0]),
    "rosenbrockGradient": (lambda v: [10 * (v[1] - v[0] ** 2), 1 - v[0]], [-1.2, 1.0]),
    "burdenFaires": (
        lambda v: [
            3 * v[0] - np.cos(v[1] * v[2]) - 0.5,
            v[0] ** 2 - 81 * (v[1] + 0.1) ** 2 + np.sin(v[2]) + 1.06,
            np.exp(-v[0] * v[1]) + 20 * v[2] + (10 * np.pi - 3) / 3,
        ],
        [0.1, 0.1, -0.1],
    ),
    "broydenTridiagonal": (broyden_tridiagonal, [-1.0] * 6),
}

FIXED = {
    "dottie": (lambda x: np.cos(x), [1.0]),
    "planar": (lambda v: np.array([0.5 * np.cos(v[1]), 0.5 * np.sin(v[0]) + 0.2]), [0.0, 0.0]),
}

MINIMA = {
    "quartic": (lambda x: (x - 1.3) ** 4 + 0.5 * (x - 1.3) ** 2 - 2, (0.0, 1.0), (-3.0, 4.0)),
    "sinPlus": (lambda x: np.sin(x) + 0.1 * x, (3.0, 4.0), (2.0, 6.0)),
    "gammaLog": (lambda x: special.gammaln(x), (1.0, 2.0), (0.5, 4.0)),
}


def cases() -> dict[str, object]:
    out: dict[str, object] = {}
    scalar = {}
    for name, (f, lo, hi, x0) in SCALAR.items():
        root, r = optimize.brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, full_output=True)
        scalar[name] = {
            "lo": lo,
            "hi": hi,
            "x0": x0,
            "root": root,
            "brentqIterations": r.iterations,
            "brentqCalls": r.function_calls,
        }
    scalar["lambert"]["root"] = special.lambertw(3).real
    out["scalar"] = scalar
    # Each registered scalar method is held to every function (the open ones from x0, or x0 and the bracket's ends).
    for key in ("bisection", "regulaFalsi", "brent", "secant", "newtonRoot"):
        out[key] = {"functions": list(SCALAR)}

    systems = {}
    for name, (F, x0) in SYSTEMS.items():
        sol = optimize.root(lambda v, F=F: np.asarray(F(v), dtype=float), x0, method="hybr", tol=1e-14)
        assert sol.success, name
        systems[name] = {"x0": x0, "root": sol.x}
    out["systems"] = systems
    for key in ("newtonSystem", "broyden", "continuation"):
        out[key] = {"systems": list(SYSTEMS)}

    out["fixedPoint"] = {
        name: {
            "x0": x0,
            "point": np.atleast_1d(np.asarray(optimize.fixed_point(g, np.asarray(x0), xtol=1e-14, maxiter=5000))),
        }
        for name, (g, x0) in FIXED.items()
    }

    minima = {}
    for name, (f, bracket, bounds) in MINIMA.items():
        brent = cast(_ScalarMinimum, optimize.minimize_scalar(f, bracket=bracket, method="brent", tol=1e-12))
        bounded = cast(
            _ScalarMinimum, optimize.minimize_scalar(f, bounds=bounds, method="bounded", options={"xatol": 1e-12})
        )
        minima[name] = {"bracket": bracket, "bounds": bounds, "x": brent.x, "value": brent.fun, "boundedX": bounded.x}
    out["minimizeScalar"] = minima
    return out
