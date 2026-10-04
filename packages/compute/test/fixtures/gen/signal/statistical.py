"""Golden values for aifn-compute/signal/statistical's adaptive filters: LMS, normalised LMS and RLS run directly in
numpy (the
updates of Sayed, 2008, "Adaptive Filters", with the a-priori error and RLS's symmetrised inverse correlation matrix)
on a system-identification problem, an unknown 5-tap FIR plant driven by coloured noise and observed in white noise.
The weights are recorded at checkpoints; the least-squares solution on the whole record is the identification target."""

import numpy as np
from scipy import signal


def regressor(x, t, m):
    u = np.zeros(m)
    for k in range(m):
        if t - k >= 0:
            u[k] = x[t - k]
    return u


def run(x, d, m, update, checkpoints):
    w = np.zeros(m)
    out = {}
    errors = []
    for t in range(len(x)):
        u = regressor(x, t, m)
        e = d[t] - w @ u
        w = update(w, u, e)
        errors.append(e)
        if t + 1 in checkpoints:
            out[str(t + 1)] = w.copy()
    return {"weights": out, "errors": np.array(errors)}


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    n, m = 600, 5
    plant = np.array([0.8, -0.4, 0.25, 0.1, -0.05])
    x = signal.lfilter([1.0], [1.0, -0.6], rng.normal(size=n))  # coloured (AR(1)) input
    d = signal.lfilter(plant, [1.0], x) + 0.01 * rng.normal(size=n)
    checkpoints = [1, 2, 10, 50, 200, 600]
    U = np.array([regressor(x, t, m) for t in range(n)])
    ls = np.linalg.lstsq(U, d, rcond=None)[0]

    mu = 0.02
    lms = run(x, d, m, lambda w, u, e: w + mu * e * u, checkpoints)
    mun, eps = 0.5, 1e-6
    nlms = run(x, d, m, lambda w, u, e: w + mun / (eps + u @ u) * e * u, checkpoints)

    lam, delta = 0.995, 0.01
    P = np.eye(m) / delta
    w = np.zeros(m)
    rls = {"weights": {}, "errors": []}
    for t in range(n):
        u = regressor(x, t, m)
        Pu = P @ u
        k = Pu / (lam + u @ Pu)
        e = d[t] - w @ u
        w = w + k * e
        P = (P - np.outer(k, Pu)) / lam
        P = 0.5 * (P + P.T)
        rls["errors"].append(e)
        if t + 1 in checkpoints:
            rls["weights"][str(t + 1)] = w.copy()
    rls["P"] = P
    return {
        "x": x,
        "d": d,
        "order": m,
        "plant": plant,
        "leastSquares": ls,
        "lms": {"stepSize": mu, **lms},
        "nlms": {"stepSize": mun, "epsilon": eps, **nlms},
        "rls": {"forgetting": lam, "delta": delta, **rls},
    }
