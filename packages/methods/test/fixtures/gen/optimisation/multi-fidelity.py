"""Reference values for aifn-methods/optimisation/multi-fidelity.

- Hyperband brackets by Li et al. (2018), Algorithm 1, written out directly: s_max = floor(log_eta(R)) (computed
  in integers), B = (s_max + 1) R, n = ceil(B / R * eta^s / (s + 1)), r = R eta^-s, n_i = floor(n eta^-i),
  r_i = r eta^i. R in units of the minimum budget.
- Power-law fits a + b t^-c by scipy.optimize.curve_fit (Levenberg-Marquardt), from several starts, keeping the
  smallest residual.
"""

import math

import numpy as np
from scipy.optimize import curve_fit


def s_max_of(ratio: float, eta: int) -> int:
    s = 0
    while eta ** (s + 1) <= ratio * (1 + 1e-12):
        s += 1
    return s


def hyperband(R: float, eta: int) -> list[dict]:
    s_max = s_max_of(R, eta)
    B = (s_max + 1) * R
    out = []
    for s in range(s_max, -1, -1):
        n = math.ceil(B / R * eta**s / (s + 1) - 1e-9)
        r = R * eta ** (-s)
        out.append(
            {
                "s": s,
                "n": [math.floor(n * eta ** (-i) + 1e-9) for i in range(s + 1)],
                "r": [r * eta**i for i in range(s + 1)],
            }
        )
    return out


def power(t, a, b, c):
    return a + b * t ** (-c)


def fit(t: np.ndarray, y: np.ndarray) -> dict:
    best = None
    for c0 in (0.1, 0.3, 0.5, 1.0, 2.0):
        try:
            p, _ = curve_fit(power, t, y, p0=(y[-1], y[0] - y[-1], c0), maxfev=20000, xtol=1e-14, ftol=1e-14)
        except RuntimeError:
            continue
        rss = float(np.sum((y - power(t, *p)) ** 2))
        if p[2] > 0 and (best is None or rss < best[1]):
            best = (p, rss)
    assert best is not None
    p, rss = best
    return {"t": t, "y": y, "a": p[0], "b": p[1], "c": p[2], "rss": rss}


def cases() -> dict:
    rng = np.random.default_rng(2015)
    t = np.arange(1, 21, dtype=float)
    noisy = 0.15 + 0.6 * t**-0.8 + rng.normal(scale=0.01, size=t.size)
    t2 = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    slow = 0.3 + 1.2 * t2**-0.25 + rng.normal(scale=0.005, size=t2.size)
    return {
        "hyperband": {
            "R81eta3": hyperband(81, 3),
            "R27eta3": hyperband(27, 3),
            "R16eta2": hyperband(16, 2),
            "R100eta3": hyperband(100, 3),
        },
        "powerLaw": [fit(t, noisy), fit(t2, slow)],
    }
