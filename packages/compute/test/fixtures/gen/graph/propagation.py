"""Golden values for aifn-compute/graph/propagation: label propagation and label spreading from scikit-learn's
`LabelPropagation` and `LabelSpreading` (RBF kernel, run to a tight tolerance), with the affinity matrix they use."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.semi_supervised import LabelPropagation, LabelSpreading


def cases() -> dict[str, object]:
    x, y = make_moons(n_samples=60, noise=0.08, random_state=3)
    labels = np.full(60, -1)
    rng = np.random.default_rng(7)
    for c in (0, 1):
        labels[rng.choice(np.flatnonzero(y == c), size=3, replace=False)] = c
    gamma = 8
    w = rbf_kernel(x, gamma=gamma)
    prop = LabelPropagation(kernel="rbf", gamma=gamma, max_iter=100000, tol=1e-12).fit(x, labels)
    out: dict[str, object] = {
        "x": x,
        "labels": labels,
        "affinity": w,
        "propagation": {"scores": prop.label_distributions_, "labels": prop.transduction_},
    }
    for alpha in (0.2, 0.9):
        spread = LabelSpreading(kernel="rbf", gamma=gamma, alpha=alpha, max_iter=100000, tol=1e-12).fit(x, labels)
        out[f"spreading{alpha}"] = {"scores": spread.label_distributions_, "labels": spread.transduction_}
    return out
