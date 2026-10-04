"""Golden values for aifn-compute/smooth from scipy.interpolate (CubicSpline, PchipInterpolator, Akima1DInterpolator,
make_smoothing_spline, BSpline, make_lsq_spline, BarycentricInterpolator, RBFInterpolator) and a direct numpy
P-spline solve."""

import numpy as np
from scipy import interpolate


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    x = np.sort(rng.uniform(0, 10, size=9))
    y = np.sin(x) + 0.3 * x
    t = np.linspace(-1, 11, 41)
    out: dict[str, object] = {"x": x, "y": y, "t": t}

    cubic = {}
    for bc in ["not-a-knot", "natural", "clamped"]:
        cs = interpolate.CubicSpline(x, y, bc_type=bc)
        cubic[bc] = {"v": cs(t), "d1": cs(t, 1), "d2": cs(t, 2)}
    # scipy is unstubbed: bc_type is inferred as str from its default.
    cs = interpolate.CubicSpline(x, y, bc_type=((1, 0.5), (1, -1.0)))  # pyright: ignore[reportArgumentType]
    cubic["first"] = {"v": cs(t), "d1": cs(t, 1), "d2": cs(t, 2)}
    yp = y.copy()
    yp[-1] = yp[0]
    cs = interpolate.CubicSpline(x, yp, bc_type="periodic")
    cubic["periodic"] = {"y": yp, "v": cs(t), "d1": cs(t, 1), "d2": cs(t, 2)}
    out["cubic"] = cubic
    out["pchip"] = interpolate.PchipInterpolator(x, y)(t)
    out["akima"] = interpolate.Akima1DInterpolator(x, y)(t, extrapolate=True)
    out["makima"] = interpolate.Akima1DInterpolator(x, y, method="makima")(t, extrapolate=True)
    ts = np.linspace(x[0], x[-1], 17)
    out["ts"] = ts
    w = rng.uniform(0.5, 2, size=len(x))
    out["smoothing"] = {
        "lam": 0.8,
        "w": w,
        "v": interpolate.make_smoothing_spline(x, y, lam=0.8)(ts),
        "vw": interpolate.make_smoothing_spline(x, y, w=w, lam=0.8)(ts),
    }
    # It permutes the nodes randomly unless seeded. random_state is the legacy spelling that scipy's rng decorator still
    # accepts (seeding a RandomState, unlike rng=); the decorator hides it from Pyright.
    bary = interpolate.BarycentricInterpolator(x, y, random_state=0)  # pyright: ignore[reportCallIssue]
    out["polynomial"] = {"v": bary(ts), "d1": bary.derivative(ts, 1)}

    # B-spline design matrices (and derivatives) on a non-uniform clamped knot vector.
    k = 3
    knots = np.r_[[0.0] * 4, 1.5, 2.0, 4.5, 7.0, [10.0] * 4]
    nb = len(knots) - k - 1
    xb = np.linspace(0, 10, 23)
    design = {}
    for nu in range(3):
        cols = []
        for j in range(nb):
            c = np.zeros(nb)
            c[j] = 1
            cols.append(interpolate.BSpline(knots, c, k)(xb, nu=nu))
        design[str(nu)] = np.column_stack(cols)
    out["bspline"] = {"knots": knots, "x": xb, "design": design}
    xl = np.sort(rng.uniform(0, 10, size=40))
    yl = np.cos(xl) + 0.1 * rng.normal(size=40)
    lsq = interpolate.make_lsq_spline(xl, yl, knots, k)
    out["lsq"] = {"x": xl, "y": yl, "coef": lsq.c, "v": lsq(xb)}

    # P-spline: uniform knots on [0, 10], 10 segments, second differences, λ = 2, solved directly.
    seg, lam = 10, 2.0
    h = 10 / seg
    pk = np.arange(-k, seg + k + 1) * h
    npb = len(pk) - k - 1
    B = interpolate.BSpline.design_matrix(xl, pk, k).toarray()
    D = np.diff(np.eye(npb), 2, axis=0)
    A = B.T @ B + lam * D.T @ D
    beta = np.linalg.solve(A, B.T @ yl)
    H = B @ np.linalg.solve(A, B.T)
    rss = np.sum((yl - B @ beta) ** 2)
    edf = np.trace(H)
    out["pspline"] = {
        "coef": beta,
        "edf": edf,
        "gcv": len(xl) * rss / (len(xl) - edf) ** 2,
        "lam": lam,
        "segments": seg,
    }

    # Thin-plate spline in two dimensions.
    x2 = rng.uniform(-1, 1, size=(15, 2))
    y2 = np.sin(2 * x2[:, 0]) * x2[:, 1]
    g2 = rng.uniform(-1, 1, size=(6, 2))
    out["tps"] = {
        "x": x2,
        "y": y2,
        "g": g2,
        "v": interpolate.RBFInterpolator(x2, y2, kernel="thin_plate_spline", degree=1)(g2),
        "vs": interpolate.RBFInterpolator(x2, y2, kernel="thin_plate_spline", degree=1, smoothing=0.1)(g2),
    }
    return out
