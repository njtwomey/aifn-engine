"""Reference values for aifn-compute/numerics/factorisation from scikit-learn: NMF by multiplicative updates (solver
'mu') and
by coordinate descent (solver 'cd', HALS) from a given initial W and H, after a fixed number of sweeps (the tolerance
set to 0 so every sweep runs), with the objective as scikit-learn's beta divergence; the Johnson-Lindenstrauss
dimension (`johnson_lindenstrauss_min_dim`); CCA in closed form (numpy: the SVD of the whitened cross-covariance,
classical and ridge-regularised) with scikit-learn's `CCA` scores for the first pair."""

from typing import Any

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import (
    _beta_divergence,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]
)
from sklearn.random_projection import johnson_lindenstrauss_min_dim


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(11)
    x = rng.uniform(0, 1, size=(8, 6)) @ rng.uniform(0, 1, size=(6, 7)) + 0.05
    k = 3
    w0 = rng.uniform(0.1, 1, size=(8, k))
    h0 = rng.uniform(0.1, 1, size=(k, 7))
    out: list[dict[str, Any]] = []
    for loss in ["frobenius", "kullback-leibler"]:
        for steps in [1, 5, 40]:
            # scikit-learn's stubs type n_components as "auto" only.
            m = NMF(n_components=k, solver="mu", beta_loss=loss, init="custom", max_iter=steps, tol=0.0)  # pyright: ignore[reportArgumentType]
            w = m.fit_transform(x, W=w0.copy(), H=h0.copy())
            h = m.components_
            beta = 2 if loss == "frobenius" else 1
            obj = float(_beta_divergence(x, w, h, beta))  # pyright: ignore[reportUnknownArgumentType]
            out.append({"loss": loss, "steps": steps, "W": w, "H": h, "objective": obj})
    hals: list[dict[str, Any]] = []
    for steps in [1, 5, 40]:
        m = NMF(n_components=k, solver="cd", init="custom", max_iter=steps, tol=0.0, shuffle=False)  # pyright: ignore[reportArgumentType]
        w = m.fit_transform(x, W=w0.copy(), H=h0.copy())
        h = m.components_
        obj = float(_beta_divergence(x, w, h, 2))  # pyright: ignore[reportUnknownArgumentType]
        hals.append({"steps": steps, "W": w, "H": h, "objective": obj})
    jl = [
        {"n": n, "eps": eps, "k": int(johnson_lindenstrauss_min_dim(n, eps=eps))}
        for n in [10, 100, 1000, 100000]
        for eps in [0.1, 0.3, 0.5, 0.9]
    ]
    return {"x": x, "W0": w0, "H0": h0, "runs": out, "hals": hals, "jl": jl, "cca": cca_cases()}


def inverse_sqrt(s: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(s)
    return (vectors / np.sqrt(values)) @ vectors.T


def cca_reference(x: np.ndarray, y: np.ndarray, rx: float, ry: float) -> dict[str, Any]:
    n = x.shape[0]
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    wx = inverse_sqrt(xc.T @ xc / (n - 1) + rx * np.eye(x.shape[1]))
    wy = inverse_sqrt(yc.T @ yc / (n - 1) + ry * np.eye(y.shape[1]))
    u, s, vt = np.linalg.svd(wx @ (xc.T @ yc / (n - 1)) @ wy)
    r = min(x.shape[1], y.shape[1])
    a = wx @ u[:, :r]
    b = wy @ vt.T[:, :r]
    for c in range(r):
        if a[np.argmax(np.abs(a[:, c])), c] < 0:
            a[:, c] *= -1
            b[:, c] *= -1
    return {"rx": rx, "ry": ry, "A": a, "B": b, "correlations": s[:r]}


def cca_cases() -> dict[str, Any]:
    rng = np.random.default_rng(5)
    n = 60
    z = rng.normal(size=(n, 2))
    x = z @ rng.normal(size=(2, 4)) + 0.5 * rng.normal(size=(n, 4))
    y = z @ rng.normal(size=(2, 3)) + 0.5 * rng.normal(size=(n, 3))
    sk = CCA(n_components=1, scale=False, max_iter=5000, tol=1e-12).fit(x, y)
    xs, ys = sk.transform(x, y)  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
    first = float(np.corrcoef(np.ravel(xs), np.ravel(ys))[0, 1])  # pyright: ignore[reportUnknownArgumentType]
    return {
        "x": x,
        "y": y,
        "fits": [cca_reference(x, y, 0.0, 0.0), cca_reference(x, y, 0.5, 0.2)],
        "sklearnFirst": abs(first),
    }
