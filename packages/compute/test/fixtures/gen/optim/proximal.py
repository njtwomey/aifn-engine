"""Golden values for aifn-compute/optim/proximal: the soft, hard, SCAD (Fan and Li 2001) and MCP (Zhang 2010)
thresholding rules and penalties in closed form, each rule checked to minimise 1/2 (z - theta)^2 + p(theta) by brute
force on a fine grid; and exact 1-D total variation denoising and the fused lasso signal approximator from their duals,
min ||D^T u (+ v) - y||^2 over the box |u| <= lambda2 (|v| <= lambda1), solved by scipy.optimize.lsq_linear's BVLS (an
active-set method, exact to rounding), x = y - D^T u (- v)."""

import numpy as np
from scipy.optimize import lsq_linear


def threshold(z: np.ndarray, rule: str, lam: float, gamma: float) -> np.ndarray:
    a, sign = np.abs(z), np.sign(z)
    soft = sign * np.maximum(a - lam, 0)
    if rule == "soft":
        return soft
    if rule == "hard":
        return np.where(a > lam, z, 0.0)
    if rule == "scad":
        mid = ((gamma - 1) * z - sign * gamma * lam) / (gamma - 2)
        return np.where(a <= 2 * lam, soft, np.where(a <= gamma * lam, mid, z))
    return np.where(a <= gamma * lam, soft / (1 - 1 / gamma), z)


def penalty(t: np.ndarray, rule: str, lam: float, gamma: float) -> np.ndarray:
    a = np.abs(t)
    if rule == "soft":
        return lam * a
    if rule == "hard":
        return np.where(a == 0, 0.0, 0.5 * lam**2)
    if rule == "scad":
        mid = (2 * gamma * lam * a - a**2 - lam**2) / (2 * (gamma - 1))
        return np.where(a <= lam, lam * a, np.where(a <= gamma * lam, mid, lam**2 * (gamma + 1) / 2))
    return np.where(a <= gamma * lam, lam * a - a**2 / (2 * gamma), gamma * lam**2 / 2)


def brute_force(z: np.ndarray, rule: str, lam: float, gamma: float) -> np.ndarray:
    grid = np.linspace(-12, 12, 480001)  # spacing 5e-5
    pen = penalty(grid, rule, lam, gamma)
    return np.array([grid[np.argmin(0.5 * (zi - grid) ** 2 + pen)] for zi in z])


def thresholding_cases() -> dict[str, object]:
    # Avoid the hard rule's tie at |z| = lambda, where 0 and z both minimise.
    z = np.array([-7.3, -4.1, -3.2, -2.6, -1.7, -0.9, -0.2, 0.0, 0.35, 1.25, 1.95, 2.4, 3.05, 3.6, 5.5, 8.0])
    out: dict[str, object] = {"z": z, "theta": np.linspace(-6, 6, 25), "rules": []}
    rules = []
    for rule, lam, gamma in [
        ("soft", 1.5, 0.0),
        ("hard", 1.5, 0.0),
        ("scad", 1.0, 3.7),
        ("scad", 1.2, 2.5),
        ("mcp", 1.0, 3.0),
        ("mcp", 1.2, 1.5),
    ]:
        value = threshold(z, rule, lam, gamma)
        assert np.allclose(brute_force(z, rule, lam, gamma), value, atol=1e-4), (rule, lam, gamma)
        rules.append(
            {
                "rule": rule,
                "lambda": lam,
                "gamma": gamma,
                "value": value,
                "penalty": penalty(np.asarray(out["theta"]), rule, lam, gamma),
            }
        )
    out["rules"] = rules
    return out


def difference(n: int) -> np.ndarray:
    return np.diff(np.eye(n), axis=0)


def tv_dual(y: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    n = len(y)
    dt = difference(n).T  # n x (n - 1)
    cols = [dt] + ([np.eye(n)] if lam1 > 0 else [])
    m = np.hstack(cols)
    bound = np.r_[np.full(n - 1, lam2), np.full(n if lam1 > 0 else 0, lam1)]
    res = lsq_linear(m, y, bounds=(-bound, bound), method="bvls", tol=1e-15, max_iter=10000)
    return y - m @ res.x


def tv_cases() -> dict[str, object]:
    rng = np.random.default_rng(3)
    t = np.arange(60)
    blocky = np.where(t < 15, 0.0, np.where(t < 30, 2.0, np.where(t < 42, -1.0, 0.5))) + 0.4 * rng.standard_normal(60)
    signals = {
        "blocky": blocky,
        "random": rng.standard_normal(25),
        "ramp": np.linspace(0, 3, 12) + 0.1 * rng.standard_normal(12),
        "pair": np.array([1.0, -1.0]),
        "single": np.array([2.5]),
        "constant": np.full(7, 1.25),
    }
    tv = []
    for name, y in signals.items():
        for lam in [0.05, 0.3, 1.0, 50.0]:
            x = tv_dual(y, 0.0, lam) if len(y) > 1 else y.copy()
            tv.append({"signal": name, "lambda": lam, "x": x})
    flsa = []
    for lam1, lam2 in [(0.2, 0.5), (0.5, 0.1), (1.0, 2.0), (0.3, 0.0)]:
        y = signals["blocky"]
        x = tv_dual(y, lam1, lam2) if lam2 > 0 else np.sign(y) * np.maximum(np.abs(y) - lam1, 0)
        flsa.append({"signal": "blocky", "lambda1": lam1, "lambda2": lam2, "x": x})
    return {"signals": signals, "tv": tv, "fused": flsa}


def cases() -> dict[str, object]:
    return {"thresholding": thresholding_cases(), "totalVariation": tv_cases()}
