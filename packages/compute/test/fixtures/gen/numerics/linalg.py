"""Golden values for aifn-compute/linalg from numpy and scipy, including singular and ill-conditioned inputs."""

from typing import cast

import numpy as np
import scipy.linalg as sla


def descending_eigh(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(a)
    return values[::-1], vectors[:, ::-1]


def eigsh_cases() -> dict[str, object]:
    """scipy.sparse.linalg.eigsh (ARPACK) on a path-graph Laplacian and a random sparse symmetric matrix."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    rng = np.random.default_rng(17)
    n = 300
    # scipy is unstubbed: diags' offsets and eigsh's tol are inferred as int from their defaults, and the rng decorator
    # hides the legacy random_state keyword (which seeds differently from rng=, so it stays).
    lap = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]).toarray()  # pyright: ignore[reportArgumentType]
    lap[0, 0] = lap[-1, -1] = 1  # free ends: a graph Laplacian with eigenvalue 0
    m = 200
    r = sp.random(m, m, density=0.03, random_state=rng, data_rvs=rng.standard_normal).toarray()  # pyright: ignore[reportCallIssue]
    sym = r + r.T
    out = {}
    for key, a in {"laplacian": lap, "random": sym}.items():
        rows, cols = np.nonzero(a)
        entry = {"n": a.shape[0], "rows": rows, "cols": cols, "vals": a[rows, cols]}
        for which, label in [("LA", "largest"), ("SA", "smallest"), ("LM", "magnitude")]:
            # ARPACK draws a random start vector unless given one; a seeded v0 makes the fixture reproducible.
            v0 = np.random.default_rng(0).standard_normal(a.shape[0])
            w, v = eigsh(a, k=5, which=which, tol=1e-12, v0=v0)  # pyright: ignore[reportArgumentType]
            order = {"LA": np.argsort(-w), "SA": np.argsort(w), "LM": np.argsort(-np.abs(w))}[which]
            entry[label] = {"values": w[order], "vectors": v[:, order]}
        out[key] = entry
    return out


def cases() -> dict[str, object]:
    rng = np.random.default_rng(7)
    b = rng.normal(size=(5, 5))
    spd = b @ b.T + 5 * np.eye(5)
    general = rng.normal(size=(5, 5))
    rhs = rng.normal(size=(5, 2))
    vec = rng.normal(size=5)
    lower = np.tril(rng.normal(size=(5, 5))) + 3 * np.eye(5)
    upper = lower.T.copy()

    chol = np.linalg.cholesky(spd)
    p, lo, up = cast(tuple[np.ndarray, np.ndarray, np.ndarray], sla.lu(general))  # (p, l, u) without permute_l
    sign, logabs = np.linalg.slogdet(general)
    values, vectors = descending_eigh(spd)

    tall = rng.normal(size=(6, 3))
    wide = rng.normal(size=(3, 5))
    u_t, s_t, vh_t = np.linalg.svd(tall, full_matrices=False)
    u_w, s_w, vh_w = np.linalg.svd(wide, full_matrices=False)
    q_t, r_t = np.linalg.qr(tall)
    q_tc, r_tc = np.linalg.qr(tall, mode="complete")
    q_w, r_w = np.linalg.qr(wide)

    # Rank-deficient positive semi-definite: B Bᵀ with B 4×2, so plain Cholesky must fail.
    c = rng.normal(size=(4, 2))
    psd = c @ c.T

    singular = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    exactly_singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    y = rng.normal(size=3)
    x_ls, _, rank_ls, sv_ls = np.linalg.lstsq(singular, y, rcond=None)

    hilbert = sla.hilbert(8)
    hb = hilbert @ np.ones(8)

    over = rng.normal(size=(8, 3))
    over_b = rng.normal(size=8)
    over_bm = rng.normal(size=(8, 2))
    x_over, res_over, rank_over, sv_over = np.linalg.lstsq(over, over_b, rcond=None)
    x_overm, res_overm, _, _ = np.linalg.lstsq(over, over_bm, rcond=None)

    k1 = rng.normal(size=(2, 3))
    k2 = rng.normal(size=(3, 2))

    two = {
        "sym": np.array([[2.0, 0.5], [0.5, 1.0]]),
        "general": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "rotation": np.array([[0.0, -2.0], [1.0, 0.0]]),
        "nearlySingular": np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]]),
    }
    two_out = {
        "symEig": descending_eigh(two["sym"]),
        "generalEig": np.sort(np.real(np.linalg.eigvals(two["general"])))[::-1],
        "rotationEig": [
            float(np.linalg.eigvals(two["rotation"])[0].real),
            float(abs(np.linalg.eigvals(two["rotation"])[0].imag)),
        ],
        "generalInv": np.linalg.inv(two["general"]),
        "symChol": np.linalg.cholesky(two["sym"]),
        "generalSvd": np.linalg.svd(two["general"])[1],
        "nearlySingularSvd": np.linalg.svd(two["nearlySingular"])[1],
    }

    return {
        "eigsh": eigsh_cases(),
        "spd": spd,
        "general": general,
        "rhs": rhs,
        "vec": vec,
        "lower": lower,
        "upper": upper,
        "cholesky": chol,
        "triangular": {
            "lower": sla.solve_triangular(lower, rhs, lower=True),
            "lowerTrans": sla.solve_triangular(lower, rhs, lower=True, trans=1),  # trans=1 is "T": solve Aᵀ x = b
            "upper": sla.solve_triangular(upper, rhs, lower=False),
            "upperTrans": sla.solve_triangular(upper, vec, lower=False, trans=1),
            "unit": sla.solve_triangular(lower, vec, lower=True, unit_diagonal=True),
        },
        "choSolve": sla.cho_solve((chol, True), rhs),
        "lu": {"P": p.T, "L": lo, "U": up},
        "solve": np.linalg.solve(general, rhs),
        "solveVec": np.linalg.solve(general, vec),
        "inverse": np.linalg.inv(general),
        "det": np.linalg.det(general),
        "logAbsDet": logabs,
        "signDet": sign,
        "spdLogDet": np.linalg.slogdet(spd)[1],
        "eigh": {"values": values, "vectors": vectors},
        "tall": tall,
        "wide": wide,
        "svdTall": {"U": u_t, "S": s_t, "V": vh_t.T},
        "svdWide": {"U": u_w, "S": s_w, "V": vh_w.T},
        "qrTall": {"Q": q_t, "R": r_t},
        "qrTallComplete": {"Q": q_tc, "R": r_tc},
        "qrWide": {"Q": q_w, "R": r_w},
        "pinvTall": np.linalg.pinv(tall),
        "pinvWide": np.linalg.pinv(wide),
        "condGeneral": np.linalg.cond(general),
        "psd": psd,
        "singular": singular,
        "exactlySingular": exactly_singular,
        "singularPinv": np.linalg.pinv(singular),
        "singularSvd": np.linalg.svd(singular)[1],
        "singularRhs": y,
        "singularLstsq": {"x": x_ls, "rank": int(rank_ls), "singularValues": sv_ls},
        "hilbert": hilbert,
        "hilbertRhs": hb,
        "hilbertEigenvalues": np.linalg.eigvalsh(hilbert)[::-1],
        "hilbertCond": np.linalg.cond(hilbert),
        "hilbertSingularValues": np.linalg.svd(hilbert)[1],
        "over": over,
        "overB": over_b,
        "overBm": over_bm,
        "overLstsq": {"x": x_over, "residuals": res_over, "rank": int(rank_over), "singularValues": sv_over},
        "overLstsqMatrix": {"x": x_overm, "residuals": res_overm},
        "k1": k1,
        "k2": k2,
        "kron": np.kron(k1, k2),
        "traceGeneral": np.trace(general),
        "froGeneral": np.linalg.norm(general, "fro"),
        "two": two,
        "twoOut": two_out,
    }
