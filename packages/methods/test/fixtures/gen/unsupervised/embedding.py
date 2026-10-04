"""Golden values for aifn-compute/embed, from scikit-learn and direct numpy computations on small fixed data."""

import numpy as np
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.manifold._t_sne import _joint_probabilities  # pyright: ignore[reportPrivateUsage]
from sklearn.metrics import pairwise_distances


def smacof_reference(delta: np.ndarray, y: np.ndarray, steps: int) -> list[np.ndarray]:
    """Guttman transforms with unit weights (de Leeuw, 1977)."""
    n = len(y)
    out = []
    for _ in range(steps):
        d = pairwise_distances(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(d > 0, -delta / d, 0.0)
        np.fill_diagonal(b, 0)
        np.fill_diagonal(b, -b.sum(axis=1))
        y = b @ y / n
        out.append(y)
    return out


def cases() -> dict[str, object]:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 4)) @ np.array([[2, 0, 0, 0], [0.5, 1, 0, 0], [0, 0.3, 0.5, 0], [0, 0, 0.1, 0.2]])
    xq = rng.normal(size=(5, 4))

    p = PCA(n_components=3).fit(x)
    pw = PCA(n_components=2, whiten=True).fit(x)
    kp = KernelPCA(n_components=2, kernel="rbf", gamma=0.25).fit(x)

    t = np.linspace(0, 3 * np.pi, 40)
    spiral = np.column_stack([t * np.cos(t), t * np.sin(t), rng.normal(scale=0.1, size=40)])
    iso = Isomap(n_neighbors=6, n_components=2).fit(spiral)
    se = SpectralEmbedding(n_components=2, affinity="nearest_neighbors", n_neighbors=7, random_state=0).fit(spiral)
    lle = LocallyLinearEmbedding(n_neighbors=6, n_components=2, eigen_solver="dense").fit(spiral)

    d = pairwise_distances(x)
    j = d.shape[0]
    centring = np.eye(j) - 1 / j
    b = -0.5 * centring @ (d**2) @ centring
    w = np.linalg.eigvalsh(b)
    order = np.argsort(w)[::-1]
    mds_eigen = w[order]
    y0 = rng.normal(size=(30, 2))
    smacof = smacof_reference(d, y0, 3)

    joint = squareform(_joint_probabilities(pairwise_distances(x, squared=True), 5.0, 0))

    return {
        "x": x,
        "xq": xq,
        "pca": {
            "components": p.components_,
            "variance": p.explained_variance_,
            "ratio": p.explained_variance_ratio_,
            "noise": p.noise_variance_,
            "transform": p.transform(xq),
            "whiten": pw.transform(xq),
        },
        "kpca": {"eigenvalues": kp.eigenvalues_, "embedding": kp.transform(x), "transform": kp.transform(xq)},
        "spiral": spiral,
        "isomap": iso.embedding_,
        "spectral": se.embedding_,
        "lle": lle.embedding_,
        "mds_eigenvalues": mds_eigen,
        "smacof": {"init": y0, "steps": smacof},
        "joint": joint,
    }
