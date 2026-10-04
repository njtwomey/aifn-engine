"""Golden values for aifn-compute/cluster, from scikit-learn and SciPy on small fixed data."""

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, OPTICS, KMeans, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture


def cases() -> dict[str, object]:
    rng = np.random.default_rng(11)
    centres = np.array([[0.0, 0.0], [4.0, 0.5], [1.5, 3.5]])
    x = np.vstack([c + rng.normal(scale=0.7, size=(15, 2)) for c in centres])
    init = np.array([[0.5, 0.5], [3.0, 0.0], [2.0, 2.0]])

    # sklearn ships no type stubs, so Pyright takes init's and n_init's types from their defaults ("k-means++", "auto").
    km = KMeans(n_clusters=3, init=init, n_init=1, algorithm="lloyd", tol=0).fit(x)  # pyright: ignore[reportArgumentType]

    gmm = {}
    for kind, sk in (("full", "full"), ("diagonal", "diag"), ("spherical", "spherical")):
        weights = np.array([0.3, 0.3, 0.4])
        if sk == "full":
            covs = np.array([np.eye(2) * 0.5, np.eye(2), [[1.0, 0.3], [0.3, 0.8]]])
            precisions = np.linalg.inv(covs)
        elif sk == "diag":
            var = np.array([[0.5, 0.5], [1.0, 1.0], [1.0, 0.8]])
            covs = np.array([np.diag(v) for v in var])
            precisions = 1 / var
        else:
            var = np.array([0.5, 1.0, 0.9])
            covs = np.array([np.eye(2) * v for v in var])
            precisions = 1 / var
        steps = []
        for it in (1, 3):
            g = GaussianMixture(
                n_components=3,
                covariance_type=sk,
                weights_init=weights,
                means_init=init,
                precisions_init=precisions,
                max_iter=it,
                tol=0,
            ).fit(x)
            fitted = np.asarray(g.covariances_)  # declared Optional; set by fit
            full = (
                fitted
                if sk == "full"
                else np.array([np.diag(v) for v in fitted])
                if sk == "diag"
                else np.array([np.eye(2) * v for v in fitted])
            )
            steps.append({"weights": g.weights_, "means": g.means_, "covariances": full, "lower_bound": g.lower_bound_})
        gmm[kind] = {"weights": weights, "covariances": covs, "steps": steps}

    links = {m: linkage(x, method=m) for m in ("single", "complete", "average", "ward")}
    leaves = dendrogram(links["ward"], no_plot=True)["leaves"]

    db = DBSCAN(eps=0.6, min_samples=4).fit(x)
    op = OPTICS(min_samples=4, max_eps=np.inf).fit(x)

    xs = np.vstack([c + rng.normal(scale=0.3, size=(12, 2)) for c in centres])
    spectral = SpectralClustering(n_clusters=3, affinity="rbf", gamma=0.5, random_state=0).fit(xs).labels_
    ms = MeanShift(bandwidth=1.5).fit(x)

    return {
        "x": x,
        "init": init,
        "kmeans": {"centroids": km.cluster_centers_, "labels": km.labels_, "inertia": km.inertia_},
        "gmm": gmm,
        "linkage": links,
        "leaves": leaves,
        "dbscan": {"labels": db.labels_, "core": db.core_sample_indices_},
        "optics": {
            "ordering": op.ordering_,
            "reachability": op.reachability_,
            "core": op.core_distances_,
            "predecessor": op.predecessor_,
        },
        "xs": xs,
        "spectral": spectral,
        "meanshift": {"centres": ms.cluster_centers_, "labels": ms.labels_},
    }
