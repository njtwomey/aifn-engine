"""Golden values for aifn-compute/geometry: convex hulls from scipy.spatial, meshgrids from numpy, and the projective
fits
(normalised-DLT homography, eight-point fundamental matrix) from scikit-image."""

import numpy as np
from scipy.spatial import ConvexHull
from skimage import transform  # pyright: ignore[reportMissingTypeStubs]


def projective_cases() -> dict[str, object]:
    """The normalised-DLT homography and eight-point fundamental matrix from scikit-image."""
    rng = np.random.default_rng(7)
    h_true = np.array([[1.1, 0.05, 20.0], [-0.04, 0.95, 10.0], [2e-4, -1e-4, 1.0]])
    src = rng.uniform(0, 300, size=(25, 2))
    hom = np.c_[src, np.ones(25)] @ h_true.T
    dst = hom[:, :2] / hom[:, 2:] + rng.normal(0, 0.3, size=(25, 2))
    pt = transform.ProjectiveTransform.from_estimate(src, dst)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    hmat = np.asarray(pt.params)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    # Two views of points in depth: K[I | 0] and K[R | t].
    k = np.array([[400.0, 0, 200], [0, 400, 200], [0, 0, 1]])
    a = 0.2
    r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    t = np.array([-1.0, 0.1, 0.2])
    x3 = np.c_[rng.uniform(-2, 2, size=(30, 2)), rng.uniform(5, 9, size=30)]
    p1 = (x3 @ k.T)[:, :2] / x3[:, 2:]
    c2 = (x3 @ r.T + t) @ k.T
    p2 = c2[:, :2] / c2[:, 2:]
    x1 = p1 + rng.normal(0, 0.2, size=p1.shape)
    x2 = p2 + rng.normal(0, 0.2, size=p2.shape)
    ft = transform.FundamentalMatrixTransform.from_estimate(x1, x2)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    f = np.asarray(ft.params)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    f = f / np.linalg.norm(f)
    f = f * np.sign(f.flat[np.argmax(np.abs(f))])
    return {
        "homography": {"src": src, "dst": dst, "H": hmat / hmat[2, 2]},
        "fundamentalMatrix": {"x1": x1, "x2": x2, "F": f, "K": k, "R": r, "t": t, "X": x3},
    }


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    pts = rng.normal(size=(40, 2))
    hull = ConvexHull(pts)
    x = np.linspace(-1, 1, 4)
    y = np.linspace(0, 2, 3)
    xx, yy = np.meshgrid(x, y)
    xi, yi = np.meshgrid(x, y, indexing="ij")
    return {
        "hull": {"points": pts, "vertices": sorted(hull.vertices.tolist()), "area": hull.volume},
        "meshgrid": {"x": x, "y": y, "xx": xx, "yy": yy, "xi": xi, "yi": yi},
        "logspace": np.logspace(-2, 1, 7),
        **projective_cases(),
    }
