"""Reference values for the linear latent-variable models of aifn-methods/unsupervised/embedding/linear: scikit-learn's
`FactorAnalysis` at convergence (model covariance and mean log-likelihood), probabilistic PCA in closed form (numpy,
maximum-likelihood covariance with 1/n) and scikit-learn's `PCA` covariance rescaled to 1/n, and `FastICA` (parallel,
log-cosh, unit-variance whitening by SVD) from a given initial unmixing matrix after a fixed number of iterations."""

import warnings
from typing import Any

import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis, FastICA


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(17)
    n, d, q = 200, 6, 2
    z = rng.normal(size=(n, q))
    loadings = rng.normal(size=(q, d))
    noise = rng.uniform(0.1, 0.8, size=d)
    x = z @ loadings + rng.normal(size=(n, d)) * np.sqrt(noise) + rng.normal(size=d)

    fa = FactorAnalysis(n_components=q, tol=1e-12, max_iter=20000).fit(x)
    pca = PCA(n_components=q).fit(x)
    s = np.cov(x.T, bias=True)
    values, vectors = np.linalg.eigh(s)
    values, vectors = values[::-1], vectors[:, ::-1]
    sigma2 = float(values[q:].mean())
    w = vectors[:, :q] * np.sqrt(values[:q] - sigma2)
    ppca_cov = w @ w.T + sigma2 * np.eye(d)

    t = np.linspace(0, 8, 300)
    sources = np.c_[np.sin(2 * t), np.sign(np.sin(3 * t)), ((t * 1.7) % 2) - 1]
    mixing = np.array([[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]])
    xm = sources @ mixing.T
    w0 = rng.normal(size=(3, 3))
    ica: list[dict[str, Any]] = []
    for steps in [1, 3, 30]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = FastICA(
                n_components=3,
                whiten="unit-variance",
                whiten_solver="svd",
                w_init=w0,
                max_iter=steps,
                tol=0.0,
            ).fit(xm)
        ica.append({"steps": steps, "components": m.components_, "mixing": m.mixing_})

    return {
        "x": x,
        "latent": q,
        "factorAnalysis": {"covariance": fa.get_covariance(), "logLikelihood": float(fa.score(x))},
        "ppca": {
            "covariance": ppca_cov,
            "noiseVariance": sigma2,
            "sklearnCovariance": pca.get_covariance() * (n - 1) / n,
        },
        "ica": {"x": xm, "W0": w0, "runs": ica, "sources": sources},
    }
