"""aifn-compute/learning/subgroups: pysubgroup's top-k subgroups on its bundled Titanic sample (WRAcc, the Klösgen
quality with
a = 0.5, pysubgroup's LiftQF, which is the added value p − p₀ (a = 0), and the numeric mean shift), scipy's χ² of
chosen subgroups, and exceptional-model statistics (correlation, regression slope, Cook's distance of a deleted
subgroup, the logistic interaction's Wald statistic, Yule's Q) computed with numpy and statsmodels on a seeded
sample."""

import os
from typing import Any

import numpy as np
import pandas as pd
import pysubgroup as ps  # pyright: ignore[reportMissingTypeStubs]
import statsmodels.api as sm
from scipy.stats import chi2_contingency

NOMINAL = ["Sex", "Pclass", "Embarked", "SibSp", "Parch"]


def titanic() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(ps.__file__), "data", "titanic.csv")
    d = pd.read_csv(path, sep="\t")
    d["Embarked"] = d["Embarked"].fillna("U")
    out = pd.DataFrame({c: d[c].astype(str) for c in NOMINAL})
    out["Survived"] = d["Survived"].astype(int)
    out["Fare"] = d["Fare"].astype(float)
    return out


def describe(sg: Any) -> list[list[str]]:
    return sorted([[str(s.attribute_name), str(s.attribute_value)] for s in sg.selectors])


def top_k(data: pd.DataFrame, target: Any, qf: Any, depth: int, k: int = 10) -> list[dict[str, object]]:
    space = ps.create_nominal_selectors(data[NOMINAL])
    task = ps.SubgroupDiscoveryTask(data, target, space, result_set_size=k, depth=depth, qf=qf)
    result = ps.Apriori().execute(task)
    return [{"quality": float(q), "selectors": describe(sg)} for q, sg, *_ in result.results]


def chi_square(data: pd.DataFrame) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    y = data["Survived"].to_numpy()
    for sels in [[["Sex", "female"]], [["Pclass", "3"]], [["Sex", "male"], ["Pclass", "1"]], [["Embarked", "C"]]]:
        mask = np.ones(len(data), dtype=bool)
        for a, v in sels:
            mask &= (data[a] == v).to_numpy()
        table = [
            [int((y[mask] == 1).sum()), int((y[~mask] == 1).sum())],
            [int((y[mask] == 0).sum()), int((y[~mask] == 0).sum())],
        ]
        stat = float(chi2_contingency(table, correction=False)[0])  # pyright: ignore[reportArgumentType]
        out.append({"selectors": sels, "chiSquare": stat})
    return out


def emm() -> dict[str, object]:
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(size=n)
    inside = rng.random(n) < 0.3
    y = np.where(inside, -0.8 * x, 0.6 * x) + 0.5 * rng.normal(size=n)
    label = (rng.random(n) < 1 / (1 + np.exp(-np.where(inside, -2.0 * x, 1.5 * x)))).astype(int)
    a = (rng.random(n) < 0.5).astype(int)
    b = np.where(inside, 1 - a, a) * (rng.random(n) < 0.8) + (rng.random(n) < 0.1)
    b = (b > 0).astype(int)

    def corr(m: np.ndarray) -> float:
        return float(np.corrcoef(x[m], y[m])[0, 1])

    rho_in, rho_out = corr(inside), corr(~inside)
    k, rest = int(inside.sum()), int((~inside).sum())

    def line(m: np.ndarray) -> tuple[float, float, float, float]:
        xm, ym = x[m], y[m]
        design = np.column_stack([np.ones(m.sum()), xm])
        beta, *_ = np.linalg.lstsq(design, ym, rcond=None)
        rss = float(((ym - design @ beta) ** 2).sum())
        ssx = float(((xm - xm.mean()) ** 2).sum())
        return float(beta[0]), float(beta[1]), rss / (m.sum() - 2), ssx

    _, b_in, s2_in, ssx_in = line(inside)
    _, b_out, s2_out, ssx_out = line(~inside)
    full = np.column_stack([np.ones(n), x])
    beta_all, *_ = np.linalg.lstsq(full, y, rcond=None)
    s2_all = float(((y - full @ beta_all) ** 2).sum()) / (n - 2)
    a0, a1, _, _ = line(~inside)
    delta = beta_all - np.array([a0, a1])
    cook = float(delta @ (full.T @ full) @ delta / (2 * s2_all))

    d = inside.astype(float)
    design = np.column_stack([np.ones(n), x, d, x * d])
    logit = sm.Logit(label, design).fit(disp=0)  # pyright: ignore[reportUnknownMemberType]
    wald = abs(float(logit.params[3] / logit.bse[3]))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    def yule(m: np.ndarray) -> float:
        t = np.array([[((a[m] == i) & (b[m] == j)).sum() for j in (0, 1)] for i in (0, 1)], dtype=float) + 0.5
        return float((t[1, 1] * t[0, 0] - t[1, 0] * t[0, 1]) / (t[1, 1] * t[0, 0] + t[1, 0] * t[0, 1]))

    return {
        "x": x,
        "y": y,
        "label": label,
        "a": a,
        "b": b,
        "inside": inside.astype(int),
        "rhoInside": rho_in,
        "rhoOutside": rho_out,
        "fisherZ": abs(np.arctanh(rho_in) - np.arctanh(rho_out)) / np.sqrt(1 / (k - 3) + 1 / (rest - 3)),
        "slopeInside": b_in,
        "slopeOutside": b_out,
        "slopeT": abs(b_in - b_out) / np.sqrt(s2_in / ssx_in + s2_out / ssx_out),
        "cook": cook,
        "wald": wald,
        "yuleInside": yule(inside),
        "yuleOutside": yule(~inside),
    }


def cases() -> dict[str, object]:
    data = titanic()
    binary = ps.BinaryTarget("Survived", 1)
    numeric = ps.NumericTarget("Fare")
    return {
        "titanic": {c: data[c].tolist() for c in [*NOMINAL, "Survived", "Fare"]},
        "wracc": {str(d): top_k(data, binary, ps.WRAccQF(), d) for d in (1, 2, 3)},
        "standardHalf": {str(d): top_k(data, binary, ps.StandardQF(0.5), d) for d in (2, 3)},
        "addedValue": top_k(data, binary, ps.LiftQF(), 2),
        "meanShift": top_k(data, numeric, ps.StandardQFNumeric(0.5), 2),
        "chiSquare": chi_square(data),
        "emm": emm(),
    }
