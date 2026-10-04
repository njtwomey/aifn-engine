"""Golden values for aifn-compute/optim/programming: linear programs from scipy.optimize.linprog (HiGHS: solution,
objective,
status and the marginals of every row and bound), mixed-integer programs from scipy.optimize.milp, assignments from
scipy.optimize.linear_sum_assignment, quadratic programs solved exactly by enumerating active sets (each equality-
constrained QP solved by its KKT system; the optimum is the feasible one with non-negative multipliers) and checked
against scipy.optimize.minimize (SLSQP), box-constrained QPs the same way, a 0/1 knapsack table computed
directly, and the sequence programmes: edit distances from nltk, and LCS lengths and Needleman–Wunsch and
Smith–Waterman scores from Biopython's PairwiseAligner."""

import itertools
from typing import Any

import numpy as np
from scipy import optimize

STATUS = {0: "optimal", 2: "infeasible", 3: "unbounded"}


def lp_case(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None) -> dict[str, object]:
    problem: dict[str, object] = {"c": c}
    if A_ub is not None:
        problem |= {"A_ub": A_ub, "b_ub": b_ub}
    if A_eq is not None:
        problem |= {"A_eq": A_eq, "b_eq": b_eq}
    if bounds is not None:
        problem["bounds"] = [[None if not np.isfinite(v) else v for v in b] for b in bounds]
    r = optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds if bounds is not None else (0, None),
        method="highs",
    )
    out: dict[str, object] = {"problem": problem, "status": STATUS[r.status]}
    if r.status == 0:
        out |= {
            "x": r.x,
            "objective": r.fun,
            "ineq": r.ineqlin.marginals if A_ub is not None else [],
            "eq": r.eqlin.marginals if A_eq is not None else [],
            "lower": r.lower.marginals,
            "upper": r.upper.marginals,
        }
    return out


def linear_programs(rng: np.random.Generator) -> dict[str, dict[str, object]]:
    lps: dict[str, dict[str, object]] = {}
    lps["wyndor"] = lp_case([-3, -5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18])
    # Random dense LPs, feasible by construction (b = A x0 + positive slack) and bounded by finite boxes.
    n, m = 4, 6
    A = rng.normal(size=(m, n))
    x0 = rng.uniform(0.5, 1.5, n)
    lps["randomBox"] = lp_case(-rng.uniform(0.2, 1, n), A, A @ x0 + rng.uniform(0.1, 1, m), bounds=[(0, 3)] * n)
    n, mu, me = 6, 5, 2
    A = rng.normal(size=(mu, n))
    E = rng.normal(size=(me, n))
    x0 = rng.uniform(-1, 1, n)
    bounds = [(-np.inf, np.inf), (-2, np.inf), (-np.inf, 2), (-3, 3), (0, np.inf), (-1, 4)]
    # Bounded by construction: c = A_ubᵀy + A_eqᵀw + (bound duals) with y ≤ 0, the bound terms 0 on free variables, ≥ 0
    # on lower-only and ≤ 0 on upper-only ones (dual feasibility in HiGHS' signs).
    lps["mixed"] = lp_case(
        -A.T @ rng.uniform(0.2, 1, mu) + E.T @ rng.normal(size=me) + np.array([0, 0.3, -0.3, 0.5, 0.4, -0.2]),
        A,
        A @ x0 + rng.uniform(0.1, 1, mu),
        E,
        E @ x0,
        bounds,
    )
    n, mu, me = 12, 10, 3
    A = rng.normal(size=(mu, n))
    E = rng.normal(size=(me, n))
    x0 = rng.uniform(0, 2, n)
    lps["large"] = lp_case(
        rng.normal(size=n), A, A @ x0 + rng.uniform(0.1, 1, mu), E, E @ x0, [(0, 5)] * 8 + [(-5, 5)] * 4
    )
    # Klee–Minty in three dimensions: maximise Σ 2^{n−j} x_j (the simplex method's worst case).
    km_A = [[1, 0, 0], [4, 1, 0], [8, 4, 1]]
    lps["kleeMinty"] = lp_case([-4, -2, -1], km_A, [5, 25, 125])
    lps["infeasible"] = lp_case([1, 1], [[1, 1], [-1, -1]], [1, -3])
    lps["unbounded"] = lp_case([-1, 0], [[1, -1]], [1])
    lps["equalityOnly"] = lp_case([1, 2, 3], A_eq=[[1, 1, 1], [1, -1, 0]], b_eq=[6, 1])
    return lps


def eqp(Q, c, M, r):
    """min ½xᵀQx + cᵀx s.t. M x = r, by its KKT system; returns x and multipliers, or None when singular."""
    n, k = len(c), len(r)
    K = np.block([[Q, M.T], [M, np.zeros((k, k))]]) if k else Q
    rhs = np.r_[-c, r]
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        return None
    return sol[:n], sol[n:]


def qp_exact(Q, c, A, b, E, e):
    """The optimum of a strictly convex QP by enumeration of the active inequality sets."""
    m, p = len(b), len(e)
    best = None
    for size in range(min(m, len(c) - p) + 1):
        for S in itertools.combinations(range(m), size):
            M = np.vstack([E, A[list(S)]]) if p + size else np.zeros((0, len(c)))
            r = np.r_[e, b[list(S)]]
            sol = eqp(Q, c, M, r)
            if sol is None:
                continue
            x, mult = sol
            nu, lam_s = mult[:p], mult[p:]
            if np.any(A @ x > b + 1e-10) or np.any(lam_s < -1e-12):
                continue
            lam = np.zeros(m)
            lam[list(S)] = lam_s
            f = 0.5 * x @ Q @ x + c @ x
            if best is None or f < best[0] - 1e-12:
                best = (f, x, lam, nu)
    assert best is not None
    return best


def quadratic_programs(rng: np.random.Generator) -> dict[str, object]:
    qps: dict[str, object] = {}
    for name, (n, m, p) in {"small": (3, 4, 0), "withEquality": (4, 5, 1), "larger": (6, 8, 2)}.items():
        M = rng.normal(size=(n, n))
        Q = M.T @ M + 0.1 * np.eye(n)
        c = rng.normal(size=n) * 3
        A = rng.normal(size=(m, n))
        x0 = rng.normal(size=n) * 0.3
        b = A @ x0 + rng.uniform(0.05, 0.5, m)
        E = rng.normal(size=(p, n))
        e = E @ x0
        f, x, lam, nu = qp_exact(Q, c, A, b, E, e)
        cons = [{"type": "ineq", "fun": lambda v, A=A, b=b: b - A @ v}]
        if p:
            cons.append({"type": "eq", "fun": lambda v, E=E, e=e: E @ v - e})
        check = optimize.minimize(
            lambda v, Q=Q, c=c: 0.5 * v @ Q @ v + c @ v, x0, constraints=cons, method="SLSQP", options={"ftol": 1e-14}
        )
        assert np.allclose(check.x, x, atol=1e-6), name
        problem: dict[str, object] = {"Q": Q, "c": c, "A": A, "b": b}
        if p:
            problem |= {"E": E, "e": e}
        qps[name] = {"problem": problem, "x": x, "objective": f, "lambda": lam, "nu": nu, "start": x0}
    return qps


def box_programs(rng: np.random.Generator) -> dict[str, object]:
    boxes: dict[str, object] = {}
    for name, n in {"box3": 3, "box6": 6}.items():
        M = rng.normal(size=(n, n))
        Q = M.T @ M + 0.05 * np.eye(n)
        c = rng.normal(size=n) * 4
        lower = rng.uniform(-1, 0, n)
        upper = lower + rng.uniform(0.5, 2, n)
        if n == 6:
            lower[0], upper[1] = -np.inf, np.inf
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.r_[upper, -lower]
        keep = np.isfinite(b)
        f, x, _, _ = qp_exact(Q, c, A[keep], b[keep], np.zeros((0, n)), np.zeros(0))
        boxes[name] = {"problem": {"Q": Q, "c": c, "lower": lower, "upper": upper}, "x": x, "objective": f}
    return boxes


def integer_programs(rng: np.random.Generator) -> dict[str, dict[str, Any]]:
    ips: dict[str, dict[str, Any]] = {}

    def case(c, A, b, integrality, bounds=(0, np.inf)) -> dict[str, Any]:
        lo, hi = bounds
        r = optimize.milp(
            c,
            constraints=optimize.LinearConstraint(np.asarray(A, float), -np.inf, b),
            integrality=integrality,
            bounds=optimize.Bounds(lo, hi),
        )
        assert r.status == 0
        return {"problem": {"c": c, "A_ub": A, "b_ub": b, "integrality": integrality}, "x": r.x, "objective": r.fun}

    ips["textbook"] = case([-1, -1], [[-2, 2], [8, 10]], [1, 13], [1, 1])
    ips["knapsack"] = case(
        [-10, -13, -7, -8, -9],
        [[3, 4, 2, 3, 3], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
        [9, 1, 1, 1, 1, 1],
        [1] * 5,
    )
    A = rng.integers(1, 9, size=(4, 5))
    ips["randomPure"] = case(-rng.integers(1, 10, size=5), A, A.sum(axis=1) * 2 // 3, [1] * 5)
    A = rng.integers(1, 9, size=(3, 4))
    ips["mixed"] = case(-rng.integers(1, 10, size=4).astype(float), A, (A.sum(axis=1) * 3) / 5, [1, 0, 1, 0])
    return ips


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    lps = linear_programs(rng)
    out: dict[str, object] = {"lp": lps}
    optimal = [k for k, v in lps.items() if v["status"] == "optimal"]
    out["simplex"] = {"problems": list(lps)}
    out["linearInteriorPoint"] = {"problems": optimal}
    qps = quadratic_programs(rng)
    out["qp"] = qps
    out["activeSet"] = {"problems": list(qps)}
    out["quadraticInteriorPoint"] = {"problems": list(qps)}
    out["boxQuadraticProgram"] = box_programs(rng)
    ips = integer_programs(rng)
    out["milp"] = ips
    out["branchAndBound"] = {"problems": list(ips)}
    out["gomory"] = {"problems": [k for k, v in ips.items() if all(v["problem"]["integrality"])]}

    assignments = {}
    for name, shape, maximize, integer in [
        ("square", (6, 6), False, False),
        ("wide", (4, 7), False, False),
        ("tall", (7, 4), True, False),
        ("ties", (5, 5), False, True),
    ]:
        C = rng.integers(0, 4, size=shape).astype(float) if integer else rng.normal(size=shape)
        rows, cols = optimize.linear_sum_assignment(C, maximize=maximize)
        assignments[name] = {"cost": C, "maximize": maximize, "rows": rows, "cols": cols, "total": C[rows, cols].sum()}
    out["hungarianSteps"] = assignments

    weights = [3, 4, 2, 5, 1, 6]
    values = [4, 5, 3, 8, 1, 9]
    cap = 12
    T = np.zeros((len(weights) + 1, cap + 1))
    for i in range(1, len(weights) + 1):
        for w in range(cap + 1):
            T[i, w] = T[i - 1, w]
            if weights[i - 1] <= w:
                T[i, w] = max(T[i, w], T[i - 1, w - weights[i - 1]] + values[i - 1])
    out["dynamicProgram"] = {"weights": weights, "values": values, "capacity": cap, "table": T}
    return out | sequences()


PAIRS = [
    ("kitten", "sitting"),
    ("GATTACA", "GCATGCU"),
    ("intention", "execution"),
    ("", "abc"),
    ("TGTTACGG", "GGTTGACTA"),
    ("the quick brown fox", "a quick brown dog"),
    ("AAAA", "AAAA"),
]


def sequences() -> dict[str, object]:
    from Bio.Align import PairwiseAligner
    from nltk.metrics.distance import edit_distance

    def score(a: str, b: str, mode: str, match: float, mismatch: float, gap: float) -> float:
        aligner = PairwiseAligner(mode=mode, match_score=match, mismatch_score=mismatch, gap_score=gap)
        return float(aligner.score(a, b)) if a and b else (gap * (len(a) + len(b)) if mode == "global" else 0.0)

    return {
        "editDistance": [
            {"a": a, "b": b, "unit": edit_distance(a, b), "substitute2": edit_distance(a, b, substitution_cost=2)}
            for a, b in PAIRS
        ],
        # LCS length: a global alignment scoring 1 per match, forbidding mismatches, with free gaps.
        "lcs": [{"a": a, "b": b, "length": score(a, b, "global", 1, -1e6, 0)} for a, b in PAIRS],
        "needlemanWunsch": [
            {"a": a, "b": b, "match": m, "mismatch": x, "gap": g, "score": score(a, b, "global", m, x, g)}
            for a, b in PAIRS
            for m, x, g in [(1, -1, -1), (2, -1, -2)]
        ],
        "smithWaterman": [
            {"a": a, "b": b, "match": m, "mismatch": x, "gap": g, "score": score(a, b, "local", m, x, g)}
            for a, b in PAIRS
            for m, x, g in [(1, -1, -1), (3, -3, -2)]
        ],
    }
