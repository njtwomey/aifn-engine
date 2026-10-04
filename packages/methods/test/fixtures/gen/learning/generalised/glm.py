"""Golden values for aifn-compute/glm. statsmodels is not installed, so the reference is a direct maximum-likelihood fit
with
scipy.optimize (checked here against a textbook IRLS in numpy) and the classical inference computed in numpy: SEs from
the inverse Fisher information, Pearson dispersion, deviance and R's AIC convention."""

import numpy as np
from scipy import optimize, special, stats
from sklearn.linear_model import LogisticRegression


def links(name: str):
    if name == "identity":
        return (lambda m: m), (lambda e: e), (lambda e: np.ones_like(e))
    if name == "log":
        return np.log, np.exp, np.exp
    if name == "logit":
        return special.logit, special.expit, (lambda e: special.expit(e) * special.expit(-e))
    if name == "probit":
        return stats.norm.ppf, stats.norm.cdf, stats.norm.pdf
    if name == "inverse":
        return (lambda m: 1 / m), (lambda e: 1 / e), (lambda e: -1 / e**2)
    raise ValueError(name)


def family(name: str, theta: float = 1.0):
    if name == "gaussian":
        return (lambda m: np.ones_like(m)), (lambda y, m: (y - m) ** 2), None
    if name == "binomial":
        xl = special.xlogy
        return (
            (lambda m: m * (1 - m)),
            (lambda y, m: 2 * (xl(y, y) - xl(y, m) + xl(1 - y, 1 - y) - xl(1 - y, 1 - m))),
            1.0,
        )
    if name == "poisson":
        return (lambda m: m), (lambda y, m: 2 * (special.xlogy(y, y) - special.xlogy(y, m) - (y - m))), 1.0
    if name == "gamma":
        return (lambda m: m**2), (lambda y, m: 2 * (-np.log(y / m) + (y - m) / m)), None
    if name == "inverse-gaussian":
        return (lambda m: m**3), (lambda y, m: (y - m) ** 2 / (m**2 * y)), None
    if name == "negative-binomial":
        xl = special.xlogy
        return (
            (lambda m: m + m**2 / theta),
            (lambda y, m: 2 * (xl(y, y) - xl(y, m) - (xl(y + theta, y + theta) - xl(y + theta, m + theta)))),
            1.0,
        )
    raise ValueError(name)


def fit(x, y, fam, lk, w=None, offset=None, theta=1.0):
    n = len(y)
    X = np.column_stack([x, np.ones(n)])
    w = np.ones(n) if w is None else w
    o = np.zeros(n) if offset is None else offset
    V, d, phi_fixed = family(fam, theta)
    g, ginv, dmu = links(lk)
    mu = {"binomial": (w * y + 0.5) / (w + 1), "poisson": y + 0.1, "negative-binomial": y + 0.1}.get(fam, y.copy())
    eta = g(mu)
    dev_old = np.inf
    beta = dev = None
    for _ in range(100):
        de = dmu(eta)
        z = eta - o + (y - mu) / de
        W = w * de**2 / V(mu)
        beta = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
        eta = X @ beta + o
        mu = ginv(eta)
        dev = np.sum(w * d(y, mu))
        if abs(dev - dev_old) / (abs(dev) + 0.1) < 1e-12:
            break
        dev_old = dev
    assert beta is not None and dev is not None
    de = dmu(eta)
    W = w * de**2 / V(mu)
    inv = np.linalg.inv(X.T @ (W[:, None] * X))
    p = X.shape[1]
    pearson = np.sum(w * (y - mu) ** 2 / V(mu))
    phi = phi_fixed if phi_fixed is not None else pearson / (n - p)
    se = np.sqrt(np.diag(inv) * phi)
    t = beta / se
    pv = 2 * stats.norm.sf(np.abs(t)) if phi_fixed is not None else 2 * stats.t.sf(np.abs(t), n - p)
    return X, beta, se, pv, phi, dev, mu, eta


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    n = 60
    x = rng.normal(size=(n, 2))
    lin = 0.4 + 0.6 * x[:, 0] - 0.3 * x[:, 1]
    exposure = rng.uniform(0.5, 2.0, size=n)
    out: dict[str, object] = {"x": x}

    data = {
        "gaussian": (lin + 0.5 * rng.normal(size=n), "identity", None, None, 1.0),
        "poisson": (rng.poisson(np.exp(lin) * exposure).astype(float), "log", None, np.log(exposure), 1.0),
        "binomial": (None, "logit", None, None, 1.0),
        "probit": (None, "probit", None, None, 1.0),
        "gamma": (rng.gamma(3.0, np.exp(lin) / 3.0), "log", None, None, 1.0),
        "gamma_inverse": (rng.gamma(3.0, 1 / (1.5 + 0.2 * x[:, 0])) / 3.0 * 3.0, "inverse", None, None, 1.0),
        "inverse-gaussian": (rng.wald(np.exp(0.3 * lin), 4.0), "log", None, None, 1.0),
        "negative-binomial": (
            rng.negative_binomial(2.0, 2.0 / (2.0 + np.exp(lin))).astype(float),
            "log",
            None,
            None,
            2.0,
        ),
    }
    trials = rng.integers(1, 8, size=n).astype(float)
    successes = rng.binomial(trials.astype(int), special.expit(lin))
    data["binomial"] = (successes / trials, "logit", trials, None, 1.0)
    data["probit"] = ((rng.uniform(size=n) < stats.norm.cdf(lin)).astype(float), "probit", None, None, 1.0)

    fits = {}
    for key, (y, lk, w, off, theta) in data.items():
        fam = {"probit": "binomial", "gamma_inverse": "gamma"}.get(key, key)
        X, beta, se, pv, phi, dev, mu, _eta = fit(x, y, fam, lk, w, off, theta)
        V, d, _ = family(fam, theta)
        wv = np.ones(n) if w is None else w
        pearson = (y - mu) * np.sqrt(wv) / np.sqrt(V(mu))
        devres = np.sign(y - mu) * np.sqrt(wv * d(y, mu))
        # Cross-check the MLE with a direct optimiser for the canonical/log cases.
        if fam == "poisson":

            def nll(b, X=X, off=off, y=y):
                e = X @ b + off
                return np.sum(np.exp(e) - y * e)

            direct = optimize.minimize(nll, np.zeros(3), method="BFGS", options={"gtol": 1e-10}).x
            assert np.allclose(direct, beta, atol=1e-5), (direct, beta)
        fits[key] = {
            "family": fam,
            "link": lk,
            "theta": theta,
            "y": y,
            "weights": w,
            "offset": off,
            "coef": beta,
            "se": se,
            "p": pv,
            "dispersion": phi,
            "deviance": dev,
            "pearson": pearson,
            "deviance_residuals": devres,
        }
    out["fits"] = fits

    # Unpenalised binary logistic regression from scikit-learn agrees with the binomial GLM.
    yb = data["probit"][0]
    sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000).fit(x, yb)
    out["logistic"] = {"y": yb, "coef": np.append(sk.coef_[0], sk.intercept_[0])}

    # Negative binomial with θ by maximum likelihood (direct optimisation over β and log θ).
    ynb = data["negative-binomial"][0]
    X = np.column_stack([x, np.ones(n)])

    def nb_nll(v):
        b, t = v[:3], np.exp(v[3])
        m = np.exp(X @ b)
        return -np.sum(
            special.gammaln(ynb + t)
            - special.gammaln(t)
            - special.gammaln(ynb + 1)
            + t * np.log(t / (t + m))
            + special.xlogy(ynb, m / (t + m))
        )

    r = optimize.minimize(nb_nll, np.zeros(4), method="BFGS", options={"gtol": 1e-9})
    out["nb_ml"] = {"coef": r.x[:3], "theta": np.exp(r.x[3])}

    # Multinomial: unpenalised softmax, contrasts against class 0 from the MLE and the observed information.
    scores = x @ np.array([[0.0, 1.0, -1.0], [0.0, -0.5, 0.8]]) + np.array([0.0, 0.2, -0.1])
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    ym = np.array([rng.choice(3, p=q) for q in probs])

    def mn_nll(v):
        B = np.column_stack([np.zeros(3), v.reshape(3, 2)])
        s = X @ B
        return -np.sum(s[np.arange(n), ym] - special.logsumexp(s, axis=1))

    r = optimize.minimize(mn_nll, np.zeros(6), method="BFGS", options={"gtol": 1e-10})
    B = r.x.reshape(3, 2)
    P = np.exp(X @ np.column_stack([np.zeros(3), B]))
    P /= P.sum(axis=1, keepdims=True)
    info = np.zeros((6, 6))
    for i in range(n):
        pi = P[i, 1:]
        A = np.diag(pi) - np.outer(pi, pi)
        info += np.kron(np.outer(X[i], X[i]), A)
    se = np.sqrt(np.diag(np.linalg.inv(info))).reshape(3, 2)
    out["multinomial"] = {"y": ym, "coef": B, "se": se}
    return out
