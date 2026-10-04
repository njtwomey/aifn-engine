"""Golden values for aifn-compute/probability/information: entropies and KL divergences from scipy.stats.entropy
(normalised
counts, zeros, batches of rows, base 2, infinite divergences), Jensen–Shannon from scipy.spatial.distance.jensenshannon,
mutual information from scikit-learn's mutual_info_score on the contingency table, the KSG estimate from scikit-learn's
own estimator (_compute_mi_cc: no scaling, no added noise), the Gaussian closed form from log-determinants, and the
remaining measures (joint and conditional entropy, cross entropy, total variation, Hellinger, the f-divergences,
pointwise mutual information) by direct numpy computation."""

import numpy as np
from scipy import stats
from scipy.spatial import distance
from sklearn.feature_selection._mutual_info import _compute_mi_cc
from sklearn.metrics import mutual_info_score


def ksg(x, y, k):
    """KSG estimator 1 with max-norm distances, directly (O(n²)), for vector-valued x and y."""
    from scipy.special import digamma

    dx = np.abs(x[:, None, :] - x[None, :, :]).max(-1)
    dy = np.abs(y[:, None, :] - y[None, :, :]).max(-1)
    dz = np.maximum(dx, dy)
    np.fill_diagonal(dz, np.inf)
    eps = np.sort(dz, axis=1)[:, k - 1]
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    nx = (dx < eps[:, None]).sum(1)
    ny = (dy < eps[:, None]).sum(1)
    n = len(x)
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    p = np.array([3.0, 1.0, 0.0, 6.0, 2.0])  # counts, with a zero
    q = np.array([1.0, 2.0, 1.0, 4.0, 4.0])
    qz = np.array([1.0, 0.0, 2.0, 4.0, 4.0])  # zero where p has mass
    P = rng.dirichlet(np.ones(4), size=3)
    Q = rng.dirichlet(np.ones(4) * 2, size=3)
    pn, qn = p / p.sum(), q / q.sum()

    def xlogx(v):
        return np.where(v > 0, v * np.log(np.where(v > 0, v, 1)), 0.0)

    joint = rng.integers(0, 12, size=(3, 4)).astype(float)
    joint[0, 1] = 0
    pxy = joint / joint.sum()
    px, py = pxy.sum(1), pxy.sum(0)
    with np.errstate(divide="ignore"):
        pmi = np.log(pxy) - np.log(np.outer(px, py))
        npmi = np.where(pxy > 0, pmi / -np.log(np.where(pxy > 0, pxy, 1)), -1.0)
    hxy = -xlogx(pxy).sum()
    labels_a = np.repeat(np.arange(3), joint.sum(1).astype(int))
    labels_b = np.concatenate([np.repeat(np.arange(4), joint[i].astype(int)) for i in range(3)])

    def f_div(f, slope, qn=qn):
        mask = qn > 0
        rest = pn[~mask].sum()
        return float(np.sum(qn[mask] * f(pn[mask] / qn[mask])) + (slope * rest if rest > 0 else 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        gens = {
            "kl": f_div(lambda t: xlogx(t), np.inf),
            "reverseKl": f_div(lambda t: -np.log(t), 0),
            "totalVariation": f_div(lambda t: 0.5 * np.abs(t - 1), 0.5),
            "squaredHellinger": f_div(lambda t: 0.5 * (np.sqrt(t) - 1) ** 2, 0.5),
            "pearsonChiSquare": f_div(lambda t: (t - 1) ** 2, np.inf),
            "neymanChiSquare": f_div(lambda t: np.where(t > 0, (t - 1) ** 2 / np.where(t > 0, t, 1), np.inf), 1),
            "jensenShannon": f_div(lambda t: 0.5 * (xlogx(t) - (t + 1) * np.log((t + 1) / 2)), 0.5 * np.log(2)),
        }
        qzn = qz / qz.sum()
        # Against q with no mass where p has some: the generators' slopes give the extra terms.
        gens_zero = {
            "totalVariation": f_div(lambda t: 0.5 * np.abs(t - 1), 0.5, qzn),
            "squaredHellinger": f_div(lambda t: 0.5 * (np.sqrt(t) - 1) ** 2, 0.5, qzn),
            "kl": f_div(lambda t: xlogx(t), np.inf, qzn),
            "jensenShannon": f_div(lambda t: 0.5 * (xlogx(t) - (t + 1) * np.log((t + 1) / 2)), 0.5 * np.log(2), qzn),
        }

    d = 4
    M = rng.normal(size=(d, d))
    cov = M @ M.T + 0.3 * np.eye(d)

    def logdet(idx):
        return np.linalg.slogdet(cov[np.ix_(idx, idx)])[1]

    gx, gy = [0, 2], [1, 3]
    n = 200
    z = rng.multivariate_normal(np.zeros(2), [[1, 0.6], [0.6, 1]], size=n)
    x3 = rng.normal(size=(n, 2))
    y3 = np.c_[x3[:, 0] + 0.5 * rng.normal(size=n)]

    return {
        "p": p,
        "q": q,
        "qz": qz,
        "P": P,
        "Q": Q,
        "entropy": {
            "p": stats.entropy(p),
            "pBits": stats.entropy(p, base=2),
            "rows": stats.entropy(P, axis=1),
        },
        "klDivergence": {
            "pq": stats.entropy(p, q),
            "pqBits": stats.entropy(p, q, base=2),
            "pqz": stats.entropy(p, qz),
            "rows": stats.entropy(P, Q, axis=1),
        },
        "crossEntropy": {
            "pq": stats.entropy(p) + stats.entropy(p, q),
            "rows": stats.entropy(P, axis=1) + stats.entropy(P, Q, axis=1),
        },
        "jensenShannonDivergence": {
            "pq": distance.jensenshannon(p, q) ** 2,
            "pqBits": distance.jensenshannon(p, q, base=2) ** 2,
        },
        "jensenShannonDistance": {"pq": distance.jensenshannon(p, q), "pqBits": distance.jensenshannon(p, q, base=2)},
        "totalVariation": {"pq": 0.5 * np.abs(pn - qn).sum()},
        "hellingerDistance": {"pq": np.sqrt(0.5 * np.sum((np.sqrt(pn) - np.sqrt(qn)) ** 2))},
        "fDivergence": gens,
        "fDivergenceZero": gens_zero,
        "joint": joint,
        "jointEntropy": hxy,
        "conditionalEntropy": {"givenX": hxy + xlogx(px).sum(), "givenY": hxy + xlogx(py).sum()},
        "mutualInformation": {
            "nats": mutual_info_score(labels_a, labels_b),
            "bits": mutual_info_score(labels_a, labels_b) / np.log(2),
        },
        "pointwiseMutualInformation": {"pmi": pmi, "npmi": npmi},
        "gaussianMutualInformation": {
            "covariance": cov,
            "x": gx,
            "y": gy,
            "value": 0.5 * (logdet(gx) + logdet(gy) - logdet(gx + gy)),
            "scalar": -0.5 * np.log(1 - (cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])) ** 2),
        },
        "ksgMutualInformation": {
            "x": z[:, 0],
            "y": z[:, 1],
            "k3": _compute_mi_cc(z[:, 0], z[:, 1], 3),
            "k7": _compute_mi_cc(z[:, 0], z[:, 1], 7),
            "x2": x3,
            "y2": y3,
            "k4MultiDim": ksg(x3, y3, 4),
        },
        "differentialEntropy": {"normalBits": stats.norm(0.3, 2.5).entropy() / np.log(2)},
    }
