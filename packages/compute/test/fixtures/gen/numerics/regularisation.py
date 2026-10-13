"""Golden values for aifn-compute/numerics/regularisation, on a discretised Gaussian blur (a small discrete ill-posed
problem) with deterministic noise. Standard-form Tikhonov from numpy's SVD, checked against scikit-learn's Ridge
(alpha = lambda, no intercept); general-form Tikhonov from numpy.linalg.lstsq on the stacked system; the L-curve's
curvature from the parametric formula with analytic derivatives in lambda (independent of the closed form the engine
uses); the discrepancy principle by scipy.optimize.brentq on log lambda; Landweber iterates by the plain recursion,
checked against their spectral filter factors. General-form paths (second differences, a square and a tall L) from
numpy.linalg.lstsq on the stacked system, the generalised singular values of a square L as the singular values of
A L^-1, and convolution matrices column by column from scipy.ndimage.convolve of unit vectors."""

import numpy as np
from scipy import ndimage, optimize
from sklearn.linear_model import Ridge


def blur(m: int, n: int, width: float) -> np.ndarray:
    t = (np.arange(m) + 0.5) / m
    u = (np.arange(n) + 0.5) / n
    return np.exp(-((t[:, None] - u[None, :]) ** 2) / (2 * width**2)) / n


def problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m, n = 16, 12
    a = blur(m, n, 0.15)
    u = (np.arange(n) + 0.5) / n
    x_true = np.sin(np.pi * u) + (u > 0.5)
    noise = 1e-3 * np.random.default_rng(0).standard_normal(m)
    return a, a @ x_true + noise, x_true, noise


def spectral(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    beta = u.T @ b
    return s, vt.T, beta, float(np.sum((b - u @ beta) ** 2))


def standard(a: np.ndarray, b: np.ndarray, lam: float) -> dict[str, object]:
    s, v, beta, _ = spectral(a, b)
    cutoff = max(a.shape) * np.finfo(float).eps * s[0]
    f = s**2 / (s**2 + lam) if lam > 0 else (s > cutoff).astype(float)
    x = v @ np.where(f > 0, f * beta / np.where(s > 0, s, 1), 0)
    return {
        "lambda": lam,
        "x": x,
        "residualNorm": np.linalg.norm(a @ x - b),
        "solutionNorm": np.linalg.norm(x),
        "filterFactors": f,
        "singularValues": s,
    }


def curvature(s: np.ndarray, beta: np.ndarray, outside: float, lam: float) -> float:
    """Signed curvature of (log ||r||, log ||x||) from first and second derivatives in lambda."""
    f = s**2 / (s**2 + lam)
    df = -(s**2) / (s**2 + lam) ** 2
    d2f = 2 * s**2 / (s**2 + lam) ** 3
    rho = np.sum(((1 - f) * beta) ** 2) + outside
    eta = np.sum((f * beta / s) ** 2)
    rho1 = np.sum(-2 * (1 - f) * df * beta**2)
    rho2 = np.sum((2 * df**2 - 2 * (1 - f) * d2f) * beta**2)
    eta1 = np.sum(2 * f * df * beta**2 / s**2)
    eta2 = np.sum((2 * df**2 + 2 * f * d2f) * beta**2 / s**2)
    x1, x2 = rho1 / (2 * rho), (rho2 * rho - rho1**2) / (2 * rho**2)
    y1, y2 = eta1 / (2 * eta), (eta2 * eta - eta1**2) / (2 * eta**2)
    return float((x1 * y2 - x2 * y1) / (x1**2 + y1**2) ** 1.5)


def l_curve(a: np.ndarray, b: np.ndarray, lambdas: np.ndarray) -> dict[str, object]:
    s, _, beta, outside = spectral(a, b)
    rows = [standard(a, b, lam) for lam in lambdas]
    kappa = np.array([curvature(s, beta, outside, lam) for lam in lambdas])
    return {
        "lambdas": lambdas,
        "residualNorm": [r["residualNorm"] for r in rows],
        "solutionNorm": [r["solutionNorm"] for r in rows],
        "curvature": kappa,
        "cornerIndex": int(np.argmax(kappa)),
    }


def discrepancy(a: np.ndarray, b: np.ndarray, target: float) -> dict[str, object]:
    s, _, beta, outside = spectral(a, b)

    def g(t: float) -> float:
        f = s**2 / (s**2 + np.exp(t))
        return float(np.sum(((1 - f) * beta) ** 2) + outside - target**2)

    t = optimize.brentq(g, np.log(s[0] ** 2) - 60, np.log(s[0] ** 2) + 10, xtol=1e-14, rtol=4 * np.finfo(float).eps)
    lam = float(np.exp(t))
    return {"target": target, **standard(a, b, lam)}


def landweber(a: np.ndarray, b: np.ndarray, steps: list[int]) -> dict[str, object]:
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    omega = 1 / s[0] ** 2
    x = np.zeros(a.shape[1])
    out: dict[str, object] = {"step": omega, "iterates": {}}
    iterates: dict[str, object] = {}
    for k in range(1, max(steps) + 1):
        x = x + omega * a.T @ (b - a @ x)
        if k in steps:
            phi = 1 - (1 - omega * s**2) ** k
            assert np.allclose(vt.T @ (phi * (u.T @ b) / s), x, rtol=1e-10, atol=1e-12)
            iterates[str(k)] = {
                "x": x.copy(),
                "residualNorm": np.linalg.norm(a @ x - b),
                "solutionNorm": np.linalg.norm(x),
                "filterFactors": phi,
            }
    out["iterates"] = iterates
    return out


def stacked_path(a: np.ndarray, b: np.ndarray, ell: np.ndarray, lambdas: list[float]) -> dict[str, object]:
    rows = []
    for lam in lambdas:
        stacked = np.vstack([a, np.sqrt(lam) * ell])
        x = np.linalg.lstsq(stacked, np.r_[b, np.zeros(ell.shape[0])], rcond=None)[0]
        rows.append(x)
    x = np.array(rows)
    return {
        "L": ell,
        "lambdas": lambdas,
        "x": x,
        "residualNorm": np.linalg.norm(x @ a.T - b, axis=1),
        "solutionNorm": np.linalg.norm(x @ ell.T, axis=1),
    }


def convolution_cases() -> list[dict[str, object]]:
    modes = {"zero": "constant", "periodic": "wrap", "reflect": "reflect"}
    out = []
    for kernel, n in [
        ([0.1, 0.2, 0.4, 0.2, 0.1], 7),
        ([0.5, 0.3, 0.2], 6),
        ([1.0, 2.0, 3.0, 4.0], 7),
        ([0.2, 0.3, 0.5, 0.7, 1.1, 1.3, 1.7], 3),
        ([2.0], 4),
    ]:
        for boundary, mode in modes.items():
            k = np.array(kernel)
            matrix = np.column_stack([ndimage.convolve(np.eye(n)[j], k, mode=mode) for j in range(n)])
            out.append({"kernel": kernel, "n": n, "boundary": boundary, "matrix": matrix})
    return out


def cases() -> dict[str, object]:
    a, b, x_true, noise = problem()
    lambdas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    tik = [standard(a, b, lam) for lam in lambdas]
    for row in tik[1:]:
        lam = float(row["lambda"])  # pyright: ignore[reportArgumentType]
        ridge = Ridge(alpha=lam, fit_intercept=False, solver="svd", tol=1e-15).fit(a, b).coef_
        assert np.allclose(ridge, row["x"], rtol=1e-8, atol=1e-10), lam
        row["ridge"] = ridge

    n = a.shape[1]
    diff = np.diff(np.eye(n), axis=0)
    general = []
    for lam in [1e-8, 1e-6, 1e-3]:
        stacked = np.vstack([a, np.sqrt(lam) * diff])
        x = np.linalg.lstsq(stacked, np.r_[b, np.zeros(n - 1)], rcond=None)[0]
        general.append(
            {"lambda": lam, "x": x, "residualNorm": np.linalg.norm(a @ x - b), "solutionNorm": np.linalg.norm(diff @ x)}
        )

    wide = a[:6]
    wide_b = b[:6]

    s = np.linalg.svd(a, compute_uv=False)
    lo = max(s[-1], 16 * np.finfo(float).eps * s[0])
    default_grid = (s[0] * (lo / s[0]) ** (np.arange(100) / 99)) ** 2

    path_lambdas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]
    second = np.diff(np.eye(n), 2, axis=0)
    square = np.eye(n) - 0.5 * np.eye(n, k=1)
    tall = np.vstack([np.eye(n), np.diff(np.eye(n), axis=0)])
    general_path = {
        "second": stacked_path(a, b, second, path_lambdas[1:]),
        "square": stacked_path(a, b, square, path_lambdas[1:])
        | {"gammas": np.linalg.svd(a @ np.linalg.inv(square), compute_uv=False)},
        "tall": stacked_path(a, b, tall, path_lambdas[1:]),
    }

    delta = float(np.linalg.norm(noise))
    return {
        "A": a,
        "b": b,
        "xTrue": x_true,
        "noiseNorm": delta,
        "tikhonov": tik,
        "general": general,
        "wide": {"A": wide, "b": wide_b, "cases": [standard(wide, wide_b, lam) for lam in [0.0, 1e-6, 1e-3]]},
        "lCurve": l_curve(a, b, np.logspace(-14, 0, 57)),
        "lCurveDefault": l_curve(a, b, default_grid),
        "discrepancy": {
            "tau1": discrepancy(a, b, delta),
            "tau1.5": discrepancy(a, b, 1.5 * delta),
            "sigma": discrepancy(a, b, 1e-3 * np.sqrt(a.shape[0])),
        },
        "landweber": landweber(a, b, [1, 5, 50, 500]),
        "generalPath": general_path,
        "convolution": convolution_cases(),
    }
