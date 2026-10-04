"""Golden values for aifn-compute/gam: a penalised additive fit solved directly in numpy (scipy B-spline designs with
the
sum-to-zero constraint imposed through the KKT system), its EDF, the GCV curve over λ for one smooth, and the REML
smoothing parameter of one smooth from the exact restricted likelihood of its mixed-model form (Wood, 2017, §5.8 and
§6.2.5), minimised by scipy; a 0.9-expectile smooth by least asymmetric weighted squares; and an explainable
boosting machine run by a plain numpy loop of cyclic one-split boosting on equal-width bins (Lou, Caruana and Gehrke,
2012), for regression and classification; and fits with a cyclic P-spline (periodic B-splines folded from scipy's
design, wrapped difference penalty), a tensor-product smooth (row-wise Kronecker design, penalties S_A ⊗ I and
I ⊗ S_B; Wood, 2006) and a factor term (indicator columns, optionally ridge-penalised), each centred by a sum-to-zero
constraint in the KKT system, at fixed smoothing parameters."""

from typing import cast

import numpy as np
from scipy import interpolate, optimize


def design(x, lo, hi, k, degree=3):
    seg = k - degree
    h = (hi - lo) / seg
    knots = lo + np.arange(-degree, seg + degree + 1) * h
    return interpolate.BSpline.design_matrix(np.clip(x, lo, hi), knots, degree).toarray()


def fit(blocks, y, lambdas, weights=None):
    """min Σ wᵢ(yᵢ − α − Σ Bⱼβⱼ)ᵢ² + Σ λⱼ‖D βⱼ‖² subject to 1ᵀBⱼβⱼ = 0, by the KKT system (w = 1 by default)."""
    n = len(y)
    w = np.ones(n) if weights is None else weights
    X = np.column_stack([np.ones(n), *blocks])
    p = X.shape[1]
    S = np.zeros((p, p))
    C = []
    o = 1
    for B, lam in zip(blocks, lambdas, strict=True):
        q = B.shape[1]
        D = np.diff(np.eye(q), 2, axis=0)
        S[o : o + q, o : o + q] = lam * D.T @ D
        c = np.zeros(p)
        c[o : o + q] = B.sum(axis=0)
        C.append(c)
        o += q
    C = np.array(C)
    m = len(C)
    XtW = X.T * w
    K = np.block([[XtW @ X + S, C.T], [C, np.zeros((m, m))]])
    sol = np.linalg.solve(K, np.r_[XtW @ y, np.zeros(m)])
    beta = sol[:p]
    fitted = X @ beta
    # EDF = tr of the constrained hat matrix: columns of the KKT inverse's top-left block.
    Kinv = np.linalg.inv(K)[:p, :p]
    edf = np.trace(Kinv @ XtW @ X)
    return fitted, edf, S, X, C


def cyclic_design(x, lo, hi, k, degree=3):
    """k periodic B-splines: the k + degree B-splines of k equal segments, the last `degree` folded onto the first."""
    period = hi - lo
    wrapped = lo + np.mod(x - lo, period)
    B = design(wrapped, lo, hi, k + degree, degree)
    out = B[:, :k].copy()
    out[:, :degree] += B[:, k:]
    return out


def cyclic_penalty(k, order=2):
    D0 = np.diff(np.eye(order + 1), order, axis=0)[0]
    D = np.zeros((k, k))
    for r in range(k):
        for a in range(order + 1):
            D[r, (r + a) % k] += D0[a]
    return D.T @ D


def difference_penalty(q, order=2):
    D = np.diff(np.eye(q), order, axis=0)
    return D.T @ D


def tensor_design(xa, xb, ka, kb, degree=3):
    Ba = design(xa, xa.min(), xa.max(), ka, degree)
    Bb = design(xb, xb.min(), xb.max(), kb, degree)
    return np.einsum("ij,il->ijl", Ba, Bb).reshape(len(xa), -1), Ba.shape[1], Bb.shape[1]


def factor_design(g):
    levels = np.unique(g)
    return (g[:, None] == levels[None, :]).astype(float)


def fit_blocks(blocks, y):
    """min Σ(yᵢ − α − Σ Bⱼβⱼ)² + Σⱼ βⱼᵀ(Σₗ λⱼₗ Sⱼₗ)βⱼ subject to 1ᵀBⱼβⱼ = 0 for every block, by the KKT system.
    Each block is (B, [(λ, S), ...]). Returns the fitted values and the EDF."""
    n = len(y)
    X = np.column_stack([np.ones(n), *[B for B, _ in blocks]])
    p = X.shape[1]
    S = np.zeros((p, p))
    C = []
    o = 1
    for B, pens in blocks:
        q = B.shape[1]
        for lam, P in pens:
            S[o : o + q, o : o + q] += lam * P
        c = np.zeros(p)
        c[o : o + q] = B.sum(axis=0)
        C.append(c)
        o += q
    C = np.array(C)
    m = len(C)
    K = np.block([[X.T @ X + S, C.T], [C, np.zeros((m, m))]])
    sol = np.linalg.solve(K, np.r_[X.T @ y, np.zeros(m)])
    edf = np.trace(np.linalg.inv(K)[:p, :p] @ X.T @ X)
    return X @ sol[:p], edf


def reml_criterion(B, y, log_lam):
    """−2 × the profile restricted log-likelihood (up to a constant) of y = Xf β + Zr b + ε, ε ~ N(0, φI),
    b ~ N(0, (φ/λ) I): the smooth B β with 1ᵀBβ = 0 (orthonormal null-space basis of the constraint) split by the
    eigenvectors of its second-difference penalty into unpenalised (fixed, with the intercept) and penalised (random,
    scaled by the eigenvalues) parts; φ profiled out."""
    n, q = B.shape
    c = B.sum(axis=0)[:, None]
    Q, _ = np.linalg.qr(c, mode="complete")
    Z = Q[:, 1:]
    Bz = B @ Z
    D = np.diff(np.eye(q), 2, axis=0)
    e, U = np.linalg.eigh(Z.T @ D.T @ D @ Z)
    pos = e > 1e-9 * e.max()
    Xf = np.column_stack([np.ones(n), Bz @ U[:, ~pos]])
    Zr = Bz @ U[:, pos] / np.sqrt(e[pos])
    V = np.eye(n) + np.exp(-log_lam) * Zr @ Zr.T
    Vi = np.linalg.inv(V)
    A = Xf.T @ Vi @ Xf
    beta = np.linalg.solve(A, Xf.T @ Vi @ y)
    r = y - Xf @ beta
    p = Xf.shape[1]
    return (n - p) * np.log(r @ Vi @ r / (n - p)) + np.linalg.slogdet(V)[1] + np.linalg.slogdet(A)[1]


def expectile(B, y, lam, tau):
    """Least asymmetric weighted squares (Schnabel and Eilers, 2009): refit with weight τ above the curve and 1 − τ
    below it, from w = ½, until no point changes side."""
    w = np.full(len(y), 0.5)
    f = None
    for _ in range(100):
        f, *_ = fit([B], y, [lam], w)
        nxt = np.where(y > f, tau, 1 - tau)
        if np.array_equal(nxt, w):
            break
        w = nxt
    assert f is not None
    return {"fitted": f, "below": float(np.mean(y < f))}


def ebm(x, y, task, rounds, bins, rate, min_leaf=2):
    """Cyclic boosting of one-split trees per feature on equal-width bins; Newton leaves (squared error: means of
    residuals); shapes centred over the training points at the end, the intercept absorbing the means."""
    n, d = x.shape
    lo, hi = x.min(axis=0), x.max(axis=0)
    b = np.clip(np.floor((x - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    classify = task == "classification"
    m = y.mean()
    intercept = np.log(m / (1 - m)) if classify else m
    shapes = np.zeros((d, bins))
    f = np.full(n, intercept)
    floor = min_leaf * (0.01 if classify else 1)
    for _ in range(rounds):
        for j in range(d):
            if classify:
                p = 1 / (1 + np.exp(-f))
                g, h = p - y, p * (1 - p)
            else:
                g, h = f - y, np.ones(n)
            G = np.bincount(b[:, j], g, bins)
            H = np.bincount(b[:, j], h, bins)
            gl, hl = np.cumsum(G)[:-1], np.cumsum(H)[:-1]
            gt, ht = G.sum(), H.sum()
            ok = (hl >= floor) & (ht - hl >= floor)
            if not ok.any():
                continue
            gain = np.where(ok, gl**2 / np.where(ok, hl, 1) + (gt - gl) ** 2 / np.where(ok, ht - hl, 1), -np.inf)
            cut = int(np.argmax(gain))
            left, right = -rate * gl[cut] / hl[cut], -rate * (gt - gl[cut]) / (ht - hl[cut])
            step = np.where(np.arange(bins) <= cut, left, right)
            shapes[j] += step
            f += step[b[:, j]]
    for j in range(d):
        mean = shapes[j, b[:, j]].mean()
        shapes[j] -= mean
        intercept += mean
    return {"intercept": intercept, "shapes": shapes, "fitted": f}


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    n = 120
    x = rng.uniform(0, 1, size=(n, 2))
    y = np.sin(2 * np.pi * x[:, 0]) + (x[:, 1] - 0.5) ** 2 * 4 + 0.3 * rng.normal(size=n)
    k = 10
    blocks = [design(x[:, j], x[:, j].min(), x[:, j].max(), k) for j in range(2)]
    fitted, edf, *_ = fit(blocks, y, [0.5, 3.0])

    # GCV over λ for a single smooth of x0.
    grid = np.linspace(-4, 4, 161)
    gcv = []
    for log_lam in grid:
        f, e, *_ = fit(blocks[:1], y, [np.exp(log_lam)])
        rss = np.sum((y - f) ** 2)
        gcv.append(n * rss / (n - e) ** 2)
    reml_best = cast(
        optimize.OptimizeResult,
        optimize.minimize_scalar(
            lambda r: reml_criterion(blocks[0], y, r), bounds=(-8, 8), method="bounded", options={"xatol": 1e-8}
        ),
    ).x
    reml_grid = np.linspace(-4, 4, 17)
    reml = [reml_criterion(blocks[0], y, r) for r in reml_grid]
    return {
        "x": x,
        "y": y,
        "k": k,
        "lambdas": [0.5, 3.0],
        "fitted": fitted,
        "edf": edf,
        "gcv_grid": grid,
        "gcv": gcv,
        "gcv_best": grid[int(np.argmin(gcv))],
        "reml_best": reml_best,
        "reml_grid": reml_grid,
        "reml": reml,
        "expectile": {"tau": 0.9, "lambda": 1.0, **expectile(blocks[0], y, 1.0, 0.9)},
        "ebm": {
            "regression": ebm(x, y, "regression", 60, 16, 0.1),
            "classification": ebm(x, (y > 0.5).astype(float), "classification", 60, 16, 0.1),
        },
        "terms": terms_case(rng),
    }


def terms_case(rng) -> dict[str, object]:
    """Gaussian fits with cyclic, tensor-product and factor terms at fixed λ (aifn's `cyclic`, `te`, `factorTerm`)."""
    n = 150
    x = rng.uniform(0, 1, size=(n, 2))
    g = rng.integers(0, 4, size=n).astype(float)
    effect = np.array([-0.6, 0.0, 0.4, 0.9])
    y = np.sin(2 * np.pi * x[:, 0]) + 1.5 * x[:, 0] * x[:, 1] + effect[g.astype(int)] + 0.2 * rng.normal(size=n)
    X = np.column_stack([x, g])
    out: dict[str, object] = {"x": X, "y": y}
    # cyclic(0, range [0, 1], k = 8, λ = 0.7) + factorTerm(2, λ = 2)
    Bc = cyclic_design(x[:, 0], 0.0, 1.0, 8)
    Bf = factor_design(g)
    fitted, edf = fit_blocks([(Bc, [(0.7, cyclic_penalty(8))]), (Bf, [(2.0, np.eye(4))])], y)
    out["cyclic_factor"] = {"fitted": fitted, "edf": edf}
    # te(0, 1, k = [5, 6], λ = [0.3, 1.5]) + factorTerm(2) unpenalised
    Bt, qa, qb = tensor_design(x[:, 0], x[:, 1], 5, 6)
    Sa = np.kron(difference_penalty(qa), np.eye(qb))
    Sb = np.kron(np.eye(qa), difference_penalty(qb))
    fitted, edf = fit_blocks([(Bt, [(0.3, Sa), (1.5, Sb)]), (Bf, [])], y)
    out["tensor_factor"] = {"fitted": fitted, "edf": edf}
    return out
