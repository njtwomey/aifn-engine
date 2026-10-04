"""Reference values for aifn-methods/unsupervised/anomaly from scikit-learn: the local outlier factor of training and
new points (LocalOutlierFactor, novelty=True), k-nearest-neighbour distances (NearestNeighbors), the one-class SVM's
decision function (OneClassSVM, whose dual weights and offset are nu * n times aifn's) and isolation-forest scores with
their AUROC (IsolationForest: compared statistically, since the random trees differ)."""

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(4)
    inliers = np.vstack([rng.normal([-1, 0.5], 0.15, size=(70, 2)), rng.normal([1, -0.3], 0.4, size=(70, 2))])
    outliers = rng.uniform(-2.5, 2.5, size=(10, 2))
    x = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(140), np.ones(10)]
    queries = rng.uniform(-2.5, 2.5, size=(12, 2))

    lof = LocalOutlierFactor(n_neighbors=10, novelty=True).fit(x)
    nn = NearestNeighbors(n_neighbors=6).fit(x)
    dist, _ = nn.kneighbors(x)  # the first neighbour of a training point is itself
    qdist, _ = nn.kneighbors(queries, n_neighbors=5)

    oc = OneClassSVM(nu=0.1, gamma=0.8, tol=1e-10).fit(x)  # pyright: ignore[reportArgumentType]
    iso = IsolationForest(n_estimators=200, random_state=0).fit(x)
    iso_scores = -iso.score_samples(x)
    return {
        "x": x,
        "y": y,
        "queries": queries,
        "lof": {"k": 10, "train": -lof.negative_outlier_factor_, "queries": -lof.score_samples(queries)},
        "knn": {"k": 5, "train": dist[:, 5], "queries": qdist[:, 4]},
        "ocsvm": {
            "nu": 0.1,
            "gamma": 0.8,
            "train": oc.decision_function(x),
            "queries": oc.decision_function(queries),
            "scale": 0.1 * len(x),
        },
        "isolation": {"scores": iso_scores, "auroc": roc_auc_score(y, iso_scores)},
    }
