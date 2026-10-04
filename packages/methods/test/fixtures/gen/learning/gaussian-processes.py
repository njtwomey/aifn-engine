"""Golden values for aifn-compute/kernels and aifn-compute/gp, from scikit-learn's kernels, GaussianProcessRegressor and
GaussianProcessClassifier (Laplace, logistic), and for the probit link (which scikit-learn lacks) from direct numpy
implementations of Rasmussen & Williams' Algorithms 3.1–3.2 (Laplace) and 3.5 (EP)."""

from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

# sklearn ships no type stubs, so Pyright takes a parameter's type from its default (length_scale=1.0,
# optimizer="fmin_l_bfgs_b"); the reportArgumentType ignores below mark ARD length scales and optimizer=None.


class LaplaceBinary(Protocol):
    """The fitted binary Laplace classifier inside GaussianProcessClassifier (a private sklearn class)."""

    f_cached: NDArray[np.float64]
    log_marginal_likelihood_value_: float


def lml_with_gradient(gp: GaussianProcessRegressor) -> tuple[float, NDArray[np.float64]]:
    """The log marginal likelihood and its gradient at the fitted kernel's θ (sklearn's return type omits the tuple)."""
    theta = cast(Kernel, gp.kernel_).theta
    return cast(tuple[float, NDArray[np.float64]], gp.log_marginal_likelihood(theta, eval_gradient=True))


def probit_laplace(K, y01, Ks, kss):
    """R&W Algorithm 3.1 (mode, log q(y | X)) and 3.2 (predictions) with log p(y | f) = log Φ(y f), y ∈ {−1, +1}."""
    y = 2.0 * y01 - 1.0
    n = len(y)
    f = np.zeros(n)
    for _ in range(200):
        z = y * f
        r = np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z))  # N(z)/Φ(z)
        grad = y * r
        W = r**2 + z * r
        sW = np.sqrt(W)
        L = np.linalg.cholesky(np.eye(n) + sW[:, None] * K * sW[None, :])
        b = W * f + grad
        a = b - sW * linalg.cho_solve((L, True), sW * (K @ b))
        f_new = K @ a
        if np.max(np.abs(f_new - f)) < 1e-14:
            f = f_new
            break
        f = f_new
    z = y * f
    r = np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z))
    grad = y * r
    W = r**2 + z * r
    sW = np.sqrt(W)
    L = np.linalg.cholesky(np.eye(n) + sW[:, None] * K * sW[None, :])
    a = np.linalg.solve(K, f)
    lml = -0.5 * a @ f + np.sum(stats.norm.logcdf(z)) - np.sum(np.log(np.diag(L)))
    mean = Ks.T @ grad
    v = linalg.solve_triangular(L, sW[:, None] * Ks, lower=True)
    var = kss - np.sum(v**2, axis=0)
    return {"mode": f, "lml": lml, "mean": mean, "var": var, "proba": stats.norm.cdf(mean / np.sqrt(1 + var))}


def probit_ep(K, y01, Ks, kss, sweeps=500):
    """R&W Algorithm 3.5 (sequential EP, probit) to convergence; the evidence as the normaliser of prior × sites,
    log Z = Σ log Z̃ᵢ + log N(μ̃; 0, K + Σ̃), and predictions by Algorithm 3.6."""
    y = 2.0 * y01 - 1.0
    n = len(y)
    tau = np.zeros(n)
    nu = np.zeros(n)
    Sigma = K.copy()
    mu = np.zeros(n)
    for _ in range(sweeps):
        old = np.r_[tau, nu].copy()
        for i in range(n):
            t_cav = 1 / Sigma[i, i] - tau[i]
            n_cav = mu[i] / Sigma[i, i] - nu[i]
            m_c, v_c = n_cav / t_cav, 1 / t_cav
            z = y[i] * m_c / np.sqrt(1 + v_c)
            ratio = np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z))
            mu_hat = m_c + y[i] * v_c * ratio / np.sqrt(1 + v_c)
            s2_hat = v_c - v_c**2 * ratio / (1 + v_c) * (z + ratio)
            d_tau = 1 / s2_hat - t_cav - tau[i]
            tau[i] += d_tau
            nu[i] = mu_hat / s2_hat - n_cav
            if d_tau != 0:
                s = Sigma[:, i].copy()
                Sigma -= np.outer(s, s) / (1 / d_tau + s[i])
            mu = Sigma @ nu
        sT = np.sqrt(tau)
        L = np.linalg.cholesky(np.eye(n) + sT[:, None] * K * sT[None, :])
        V = linalg.solve_triangular(L, sT[:, None] * K, lower=True)
        Sigma = K - V.T @ V
        mu = Sigma @ nu
        if np.max(np.abs(np.r_[tau, nu] - old)) < 1e-12:
            break
    # Site normalisers: the tilted normaliser over the cavity–site overlap.
    t_cav = 1 / np.diag(Sigma) - tau
    n_cav = mu / np.diag(Sigma) - nu
    m_c, v_c = n_cav / t_cav, 1 / t_cav
    log_zhat = stats.norm.logcdf(y * m_c / np.sqrt(1 + v_c))
    m_site, v_site = nu / tau, 1 / tau
    log_ztilde = log_zhat + 0.5 * np.log(2 * np.pi * (v_c + v_site)) + (m_c - m_site) ** 2 / (2 * (v_c + v_site))
    lml = np.sum(log_ztilde) + stats.multivariate_normal(np.zeros(n), K + np.diag(v_site)).logpdf(m_site)
    sT = np.sqrt(tau)
    L = np.linalg.cholesky(np.eye(n) + sT[:, None] * K * sT[None, :])
    z = sT * linalg.cho_solve((L, True), sT * (K @ nu))
    mean = Ks.T @ (nu - z)
    v = linalg.solve_triangular(L, sT[:, None] * Ks, lower=True)
    var = kss - np.sum(v**2, axis=0)
    return {"mode": mu, "lml": lml, "mean": mean, "var": var, "proba": stats.norm.cdf(mean / np.sqrt(1 + var))}


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    x = rng.uniform(-2, 2, size=(7, 2))
    y2 = rng.uniform(-2, 2, size=(4, 2))
    ard = np.array([0.7, 1.9])
    kernels = {
        "rbf": (ConstantKernel(1.7) * RBF(0.8), {"lengthscale": 0.8, "variance": 1.7}),
        "rbf_ard": (ConstantKernel(1.3) * RBF(ard), {"lengthscale": ard, "variance": 1.3}),  # pyright: ignore[reportArgumentType]
        "matern12": (ConstantKernel(0.9) * Matern(0.6, nu=0.5), {"lengthscale": 0.6, "variance": 0.9}),
        "matern32": (ConstantKernel(1.1) * Matern(1.2, nu=1.5), {"lengthscale": 1.2, "variance": 1.1}),
        "matern52": (ConstantKernel(2.0) * Matern(ard, nu=2.5), {"lengthscale": ard, "variance": 2.0}),  # pyright: ignore[reportArgumentType]
        "rq": (
            ConstantKernel(1.4) * RationalQuadratic(0.9, alpha=0.7),
            {"lengthscale": 0.9, "alpha": 0.7, "variance": 1.4},
        ),
        "periodic": (
            ConstantKernel(0.8) * ExpSineSquared(1.1, periodicity=2.3),
            {"lengthscale": 1.1, "period": 2.3, "variance": 0.8},
        ),
        "linear": (ConstantKernel(0.5) * DotProduct(sigma_0=0.0) + ConstantKernel(0.3), {"variance": 0.5, "bias": 0.3}),
        "polynomial": (
            (ConstantKernel(0.3) + ConstantKernel(0.5) * DotProduct(sigma_0=0.0)) ** 3,
            {"variance": 0.5, "bias": 0.3},
        ),
    }
    grams = {name: {"params": p, "xy": k(x, y2), "xx": k(x)} for name, (k, p) in kernels.items()}

    # Regression: 1-D inputs, fixed hyperparameters (optimizer=None), noise via alpha.
    n = 12
    xr = np.sort(rng.uniform(0, 5, size=n))[:, None]
    yr = np.sin(1.3 * xr[:, 0]) + 0.2 * rng.normal(size=n)
    xs = np.linspace(-0.5, 5.5, 9)[:, None]
    noise = 0.05
    kr = ConstantKernel(1.5) * RBF(0.7)
    gpr = GaussianProcessRegressor(kernel=kr, alpha=noise, optimizer=None).fit(xr, yr)  # pyright: ignore[reportArgumentType]
    mean, cov = gpr.predict(xs, return_cov=True)
    lml, _grad = lml_with_gradient(gpr)
    # Gradient w.r.t. log noise too: use a WhiteKernel for the noise.
    kw = ConstantKernel(1.5) * RBF(0.7) + WhiteKernel(noise)
    gw = GaussianProcessRegressor(kernel=kw, alpha=0.0, optimizer=None).fit(xr, yr)  # pyright: ignore[reportArgumentType]
    lml_w, grad_w = lml_with_gradient(gw)
    # Fitted hyperparameters (L-BFGS-B from the same start, several restarts).
    fitted = GaussianProcessRegressor(kernel=kw, alpha=0.0, n_restarts_optimizer=5, random_state=0).fit(xr, yr)
    fp: dict[str, Any] = cast(Kernel, fitted.kernel_).get_params()

    # Classification by Laplace, fixed hyperparameters.
    xc = rng.uniform(-3, 3, size=(30, 1))
    yc = (np.sin(xc[:, 0]) + 0.3 * rng.normal(size=30) > 0).astype(int)
    kc = ConstantKernel(2.0) * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kc, optimizer=None).fit(xc, yc)  # pyright: ignore[reportArgumentType]
    # Binary targets: base_estimator_ is the Laplace classifier itself, not a one-vs-rest wrapper.
    base = cast(LaplaceBinary, gpc.base_estimator_)
    xcs = np.linspace(-3, 3, 7)[:, None]
    Kc = kc(xc)
    Kcs = kc(xc, xcs)
    kcss = np.diag(kc(xcs))
    probit = {"laplace": probit_laplace(Kc, yc, Kcs, kcss), "ep": probit_ep(Kc, yc, Kcs, kcss)}

    return {
        "x": x,
        "y": y2,
        "grams": grams,
        "regression": {
            "x": xr[:, 0],
            "y": yr,
            "xs": xs[:, 0],
            "noise": noise,
            "variance": 1.5,
            "lengthscale": 0.7,
            "mean": mean,
            "cov": cov,
            "lml": lml,
            # scikit-learn's theta order: log constant, log lengthscale, log noise.
            "log_gradient": grad_w,
            "lml_white": lml_w,
            "fitted": {
                "variance": fp["k1__k1__constant_value"],
                "lengthscale": fp["k1__k2__length_scale"],
                "noise": fp["k2__noise_level"],
                "lml": fitted.log_marginal_likelihood_value_,
            },
        },
        "classification": {
            "x": xc[:, 0],
            "y": yc,
            "xs": xcs[:, 0],
            "variance": 2.0,
            "lengthscale": 1.0,
            "mode": base.f_cached,
            "lml": base.log_marginal_likelihood_value_,
            "proba": gpc.predict_proba(xcs)[:, 1],
            "probit": probit,
        },
    }
