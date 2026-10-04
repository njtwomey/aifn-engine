"""Golden values for aifn-compute/numerics/neighbours: exact k nearest neighbours from scikit-learn's `KDTree`,
`BallTree` and
`NearestNeighbors` (brute force) in three metrics, on Gaussian clusters with continuous coordinates (no ties)."""

import numpy as np
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261002)
    centres = rng.normal(scale=4.0, size=(5, 3))
    data = centres[rng.integers(0, 5, size=300)] + rng.normal(size=(300, 3))
    queries = rng.normal(scale=4.0, size=(25, 3))
    k = 7
    out: dict[str, object] = {"data": data, "queries": queries, "k": k}
    for metric in ("euclidean", "manhattan", "chebyshev"):
        kd_dist, kd_idx = KDTree(data, leaf_size=10, metric=metric).query(queries, k=k)
        ball_dist, ball_idx = BallTree(data, leaf_size=10, metric=metric).query(queries, k=k)
        brute = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric).fit(data)
        brute_dist, brute_idx = brute.kneighbors(queries)
        out[metric] = {
            "kdTree": {"indices": kd_idx, "distances": kd_dist},
            "ballTree": {"indices": ball_idx, "distances": ball_dist},
            "brute": {"indices": brute_idx, "distances": brute_dist},
        }
    # The k-NN graph of the data itself, each point's own row left out.
    self_dist, self_idx = NearestNeighbors(n_neighbors=k + 1).fit(data).kneighbors(data)
    out["self"] = {"indices": self_idx[:, 1:], "distances": self_dist[:, 1:]}
    return out
