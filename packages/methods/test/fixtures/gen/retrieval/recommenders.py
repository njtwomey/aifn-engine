"""Reference values for aifn-methods/retrieval/recommenders: implicit-feedback ALS from the `implicit` library (exact
least-squares solver in float64, from given initial factors, confidences 1 + alpha r passed as the matrix values) after
a few sweeps; explicit ALS-WR (Zhou et al., 2008) by a direct NumPy implementation; and item-based cosine similarities
with scikit-learn."""

from typing import Any

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def explicit_als(r: np.ndarray, mask: np.ndarray, p: np.ndarray, q: np.ndarray, lam: float, sweeps: int):
    k = p.shape[1]
    for _ in range(sweeps):
        for u in range(r.shape[0]):
            idx = np.flatnonzero(mask[u])
            if len(idx) == 0:
                p[u] = 0
                continue
            a = q[idx].T @ q[idx] + lam * len(idx) * np.eye(k)
            p[u] = np.linalg.solve(a, q[idx].T @ r[u, idx])
        for i in range(r.shape[1]):
            idx = np.flatnonzero(mask[:, i])
            if len(idx) == 0:
                q[i] = 0
                continue
            a = p[idx].T @ p[idx] + lam * len(idx) * np.eye(k)
            q[i] = np.linalg.solve(a, p[idx].T @ r[idx, i])
    return p, q


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(2)
    users, items, k = 12, 15, 3
    counts = (rng.uniform(size=(users, items)) < 0.25) * rng.integers(1, 4, size=(users, items))
    alpha, lam, sweeps = 2.0, 0.3, 3
    p0 = rng.normal(0, 0.1, size=(users, k))
    q0 = rng.normal(0, 0.1, size=(items, k))
    # implicit's stubs type dtype as float32 only; float64 makes the solve exact to double precision.
    model = AlternatingLeastSquares(
        factors=k,
        regularization=lam,
        alpha=1.0,
        use_cg=False,
        iterations=sweeps,
        dtype=np.float64,  # pyright: ignore[reportArgumentType]
        random_state=0,
    )
    model.user_factors = p0.copy()
    model.item_factors = q0.copy()
    conf = csr_matrix(np.where(counts > 0, 1 + alpha * counts, 0.0))
    model.fit(conf, show_progress=False)
    rows = [[u, i, int(counts[u, i])] for u in range(users) for i in range(items) if counts[u, i] > 0]

    ratings = rng.integers(1, 6, size=(10, 9)).astype(float)
    mask = rng.uniform(size=(10, 9)) < 0.5
    ep0 = rng.normal(0, 0.1, size=(10, 2))
    eq0 = rng.normal(0, 0.1, size=(9, 2))
    ep, eq = explicit_als(ratings, mask, ep0.copy(), eq0.copy(), 0.1, 4)
    rating_rows = [[u, i, ratings[u, i]] for u in range(10) for i in range(9) if mask[u, i]]

    binary = (counts > 0).astype(float)
    return {
        "implicit": {
            "users": users,
            "items": items,
            "rows": rows,
            "alpha": alpha,
            "regularisation": lam,
            "sweeps": sweeps,
            "P0": p0,
            "Q0": q0,
            "P": np.asarray(model.user_factors),
            "Q": np.asarray(model.item_factors),
        },
        "explicit": {"rows": rating_rows, "P0": ep0, "Q0": eq0, "P": ep, "Q": eq, "regularisation": 0.1, "sweeps": 4},
        "itemCosine": {"rows": rows, "similarity": cosine_similarity(binary.T)},
    }
