"""Reference values for aifn-compute/learning/validate: spatial splitters and cross-validation's class handling.

- `spatialBlocks` and `buffered` leave-one-out and k-fold, by brute force in numpy.
- `nndmLeaveOneOut`: a line-by-line numpy port of `CAST::nndm` (Mila et al., 2022) in geographical space.
- `knndm`: a numpy port of `CAST::knndm` (Linnenbrink et al., 2024) with hierarchical clustering, using scipy's Ward
  linkage (`ward.D2`) and `cut_tree`, scipy's one-sided two-sample Kolmogorov-Smirnov test and its Wasserstein
  distance, and the first principal axis with its largest entry made positive.
- `crossValidate` with classes from all of `y`: a nearest-centroid classifier (as the TypeScript test defines it) on
  unshuffled 3-fold splits whose first test fold holds no positives; per-fold recall from scikit-learn with
  `labels=[0, 1]`, `pos_label=1`, `zero_division=nan`, and per-row log loss, Brier score and accuracy.
"""

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, recall_score


def distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between the rows of a and of b, summed over coordinates in order."""
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def clustered(rng: np.random.Generator, centres: np.ndarray, per: int, spread: float) -> np.ndarray:
    return np.concatenate([c + spread * rng.standard_normal((per, centres.shape[1])) for c in centres])


def grid(m: int, size: float) -> np.ndarray:
    g = np.linspace(0, size, m)
    return np.array([[u, v] for v in g for u in g])


def nndm_ref(t: np.ndarray, p: np.ndarray, phi: float | None, min_train: float) -> dict[str, Any]:
    """CAST::nndm, step for step (0-based)."""
    n = len(t)
    gij = distances(p, t).min(1)
    d = distances(t, t)
    np.fill_diagonal(d, np.nan)
    gj = np.nanmin(d, 1)
    gstar = gj.copy()
    if phi is None:
        phi = max(gij.max(), np.nanmax(d)) + 1e-9

    def row_min(i: int) -> float:
        r = d[i][~np.isnan(d[i])]
        return float(r.min()) if len(r) else np.inf

    rmin = gstar.min()
    jmin = int(np.flatnonzero(gstar == rmin)[0])
    kmin = np.flatnonzero(d[jmin] == rmin)
    while rmin <= phi:
        if (np.sum(gstar <= rmin) - 1) / n >= np.mean(gij <= rmin) and np.sum(~np.isnan(d[jmin])) / n > min_train:
            d[jmin, kmin] = np.nan
            gstar = np.array([row_min(i) for i in range(n)])
            rmin = gstar[gstar >= rmin].min()
        elif np.sum(gstar > rmin) == 0:
            break
        else:
            rmin = gstar[gstar > rmin].min()
        if not np.isfinite(rmin):
            break
        jmin = int(np.flatnonzero(gstar == rmin)[0])
        kmin = np.flatnonzero(d[jmin] == rmin)
    excluded = [[int(k) for k in np.flatnonzero(np.isnan(d[i])) if k != i] for i in range(n)]
    return {"phi": phi, "excluded": excluded, "gij": gij, "gj": gj, "gstar": gstar}


def knndm_ref(t: np.ndarray, p: np.ndarray, k: int, maxp: float, nk_len: int = 100) -> dict[str, Any]:
    """CAST::knndm (geographical, hierarchical ward.D2), with the principal axis's largest entry positive."""
    n = len(t)
    d = distances(t, t)
    gij = distances(p, t).min(1)
    off = d + np.diag(np.full(n, np.inf))
    gj = off.min(1)
    pks = ks_2samp(gj, gij, alternative="greater", method="exact").pvalue
    assert pks < 0.05
    z = linkage(t, method="ward")
    centre = t.mean(0)
    _, vecs = np.linalg.eigh((t - centre).T @ (t - centre))
    axis = vecs[:, -1]
    top = int(np.argmax(np.abs(axis)))
    if axis[top] < 0:
        axis = -axis
    qs: list[int] = []
    for g in range(nk_len):
        q = int(np.floor(np.exp(np.log(k) + g * (np.log(n - 2) - np.log(k)) / (nk_len - 1)) + 0.5))
        if q not in qs:
            qs.append(q)
    best: dict[str, Any] | None = None
    for q in qs:
        raw = cut_tree(z, n_clusters=q).ravel()
        ids: dict[int, int] = {}
        lab = np.array([ids.setdefault(int(c), len(ids)) for c in raw])
        counts = np.bincount(lab, minlength=q)
        proj = np.array([(t[lab == c].mean(0) - centre) @ axis for c in range(q)])
        order = np.argsort(proj, kind="stable")
        label = np.full(q, -1)
        nxt = 0
        for c in order:
            if counts[c] >= n / k:
                label[c] = nxt
                nxt += 1
        free = list(range(nxt, k))
        dealt = 0
        for c in order:
            if label[c] < 0:
                label[c] = free[dealt % len(free)]
                dealt += 1
        fold = label[lab]
        if np.bincount(fold, minlength=k).max() / n > maxp:
            continue
        gstar = np.array([d[i, fold != fold[i]].min() for i in range(n)])
        w = wasserstein_distance(gstar, gij)
        if best is None or w < best["W"]:
            best = {"folds": fold, "W": w, "q": q, "gstar": gstar}
    assert best is not None
    return {**best, "ksPValue": pks, "gij": gij, "gj": gj}


def spatial(rng: np.random.Generator) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Blocks: 3 x 2 cells spanning the coordinates, and cells of side 1.5 from the origin.
    coords = rng.uniform(0, 6, (25, 2))
    lo, hi = coords.min(0), coords.max(0)
    blocks = np.array([3, 2])
    cell = np.minimum(np.floor((coords - lo) / ((hi - lo) / blocks)), blocks - 1).astype(int)
    by_size = np.floor(coords / 1.5).astype(int)
    count = np.floor(hi / 1.5).astype(int) + 1
    out["blocks"] = {
        "coords": coords,
        "byBlocks": cell[:, 0] + blocks[0] * cell[:, 1],
        "bySize": by_size[:, 0] + count[0] * by_size[:, 1],
    }
    # Buffered leave-one-out and 4-fold (unshuffled) at radius 1.2.
    coords = rng.uniform(0, 5, (20, 2))
    d = distances(coords, coords)
    loo = [[j for j in range(20) if j != i and d[i, j] > 1.2] for i in range(20)]
    folds = np.repeat(np.arange(4), 5)
    kfold = [[j for j in range(20) if folds[j] != f and d[j, folds == f].min() > 1.2] for f in range(4)]
    out["buffered"] = {"coords": coords, "radius": 1.2, "looTrain": loo, "kFoldTrain": kfold}
    # NNDM: clustered samples, a grid of prediction points; the default phi and min_train, then a smaller phi.
    t = clustered(rng, np.array([[1.0, 1.0], [4.0, 2.0], [2.0, 4.5]]), 10, 0.35)
    p = grid(12, 5.5)
    out["nndm"] = [
        {"coords": t, "prediction": p, "phi": None, "minTrain": 0.5, **nndm_ref(t, p, None, 0.5)},
        {"coords": t, "prediction": p, "phi": 1.0, "minTrain": 0.8, **nndm_ref(t, p, 1.0, 0.8)},
    ]
    # kNNDM: five clumps of eight, a 15 x 15 prediction grid.
    t = clustered(rng, np.array([[1, 1], [8, 2], [5, 5], [2, 8], [8, 8.5]], dtype=float), 8, 0.6)
    p = grid(15, 10.0)
    out["knndm"] = [
        {"coords": t, "prediction": p, "k": 5, "maxp": 0.5, **knndm_ref(t, p, 5, 0.5)},
        {"coords": t, "prediction": p, "k": 3, "maxp": 0.4, **knndm_ref(t, p, 3, 0.4)},
    ]
    return out


def centroid_fit(x: np.ndarray, y: np.ndarray):
    m0, m1 = x[y == 0].mean(0), x[y == 1].mean(0)

    def prob(z: np.ndarray) -> np.ndarray:
        d0 = ((z - m0) ** 2).sum(1)
        d1 = ((z - m1) ** 2).sum(1)
        return 1 / (1 + np.exp(-(d0 - d1) / 2))

    return prob


def classes(rng: np.random.Generator) -> dict[str, Any]:
    """Unshuffled 3-fold CV of 12 rows; the first fold holds class 0 only."""
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0])
    x = rng.standard_normal((12, 2)) + 1.2 * y[:, None]
    folds = np.repeat(np.arange(3), 4)
    recall, logs, briers, hits, fold_log = [], [], [], [], []
    for f in range(3):
        train, test = folds != f, folds == f
        prob = centroid_fit(x[train], y[train])(x[test])
        decide = (prob >= 0.5).astype(int)
        recall.append(recall_score(y[test], decide, labels=[0, 1], pos_label=1, zero_division=np.nan))
        logs += [log_loss([yi], [pi], labels=[0, 1]) for yi, pi in zip(y[test], prob, strict=True)]
        briers += [brier_score_loss([yi], [pi], pos_label=1) for yi, pi in zip(y[test], prob, strict=True)]
        hits += [accuracy_score([yi], [di]) for yi, di in zip(y[test], decide, strict=True)]
        fold_log.append(log_loss(y[test], prob, labels=[0, 1]))
    return {
        "x": x,
        "y": y,
        "recall": recall,
        "logLoss": logs,
        "brierScore": briers,
        "accuracy": hits,
        "foldLogLoss": fold_log,
    }


def cases() -> dict[str, Any]:
    rng = np.random.default_rng(20261013)
    return {"spatial": spatial(rng), "classes": classes(rng)}
