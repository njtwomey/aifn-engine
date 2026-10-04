"""Golden values for aifn-compute/probability/tests from scipy.stats: the t-tests with their intervals, the exact
binomial test
and proportion intervals, χ² and G tests, Fisher's exact test and odds ratios, Cramér's V, the rank tests, the
Kolmogorov–Smirnov tests (one-sided too), Shapiro–Wilk, one-way ANOVA, the Kaplan–Meier estimate (scipy.stats.ecdf on
CensoredData) and the log-rank test. statsmodels is not installed, so the Ljung–Box statistic, the multiple-testing
adjustments (beyond scipy's false_discovery_control), Grubbs' bound, the effect sizes, Nelson–Aalen and the k-group
log-rank test are direct numpy formulas from their papers. Group-sequential boundaries are solved independently with
scipy's multivariate normal cdf and brentq (no numerical integration), and the F distribution is checked against
scipy.stats.f."""

from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special, stats

ALTS = ["two-sided", "less", "greater"]


# scipy builds its result objects dynamically, so Pyright cannot see their fields; these declare the ones used.
class Interval(Protocol):
    low: Any
    high: Any


class TestResult(Protocol):
    statistic: Any
    pvalue: Any


class TTestResult(TestResult, Protocol):
    df: Any

    def confidence_interval(self, confidence_level: float = ...) -> Interval: ...


class Chi2Result(TestResult, Protocol):
    dof: Any


def ci(r: TTestResult, level: float) -> list[float]:
    c = r.confidence_interval(level)
    return [float(c.low), float(c.high)]


def samples():
    rng = np.random.default_rng(20261001)
    x = np.round(rng.normal(0.4, 1.3, 12), 3)
    y = np.round(rng.normal(-0.2, 0.8, 9), 3)
    small = np.array([1.2, 2.9])
    return x, y, small


def t_tests():
    x, y, small = samples()
    out: dict[str, Any] = {"x": x, "y": y, "small": small, "cases": []}
    for alt in ALTS:
        for level in [0.95, 0.9]:
            r = cast(TTestResult, stats.ttest_1samp(x, 0.3, alternative=alt))
            out["cases"].append(
                {
                    "test": "oneSample",
                    "args": {"mu": 0.3},
                    "alternative": alt,
                    "level": level,
                    "statistic": r.statistic,
                    "df": r.df,
                    "p": r.pvalue,
                    "ci": ci(r, level),
                }
            )
            r = cast(TTestResult, stats.ttest_rel(x[:9], y, alternative=alt))
            out["cases"].append(
                {
                    "test": "paired",
                    "args": {},
                    "alternative": alt,
                    "level": level,
                    "statistic": r.statistic,
                    "df": r.df,
                    "p": r.pvalue,
                    "ci": ci(r, level),
                }
            )
            for equal in [True, False]:
                r = cast(TTestResult, stats.ttest_ind(x, y, equal_var=equal, alternative=alt))
                out["cases"].append(
                    {
                        "test": "pooled" if equal else "welch",
                        "args": {},
                        "alternative": alt,
                        "level": level,
                        "statistic": r.statistic,
                        "df": r.df,
                        "p": r.pvalue,
                        "ci": ci(r, level),
                    }
                )
    r = cast(TTestResult, stats.ttest_1samp(small, 0.0))
    out["cases"].append(
        {
            "test": "oneSampleSmall",
            "args": {"mu": 0},
            "alternative": "two-sided",
            "level": 0.95,
            "statistic": r.statistic,
            "df": r.df,
            "p": r.pvalue,
            "ci": ci(r, 0.95),
        }
    )
    # Effect sizes by their definitions: d with the pooled sd, g with the exact gamma correction.
    m, n = len(x), len(y)
    sp = np.sqrt(((m - 1) * x.var(ddof=1) + (n - 1) * y.var(ddof=1)) / (m + n - 2))
    d = (x.mean() - y.mean()) / sp
    df = m + n - 2
    J = np.exp(special.gammaln(df / 2) - special.gammaln((df - 1) / 2)) / np.sqrt(df / 2)
    out["cohensD"] = d
    out["hedgesG"] = J * d
    out["cohensDOne"] = (x.mean() - 0.3) / x.std(ddof=1)
    # z-test with a known σ = 1.1 against μ₀ = 0.1.
    z = (x.mean() - 0.1) / (1.1 / np.sqrt(m))
    half = stats.norm.isf(0.025) * 1.1 / np.sqrt(m)
    out["z"] = {
        "statistic": z,
        "p": 2 * stats.norm.sf(abs(z)),
        "pGreater": stats.norm.sf(z),
        "ci": [x.mean() - half, x.mean() + half],
    }
    return out


def binomial():
    cases = []
    for k, n, p in [(7, 20, 0.3), (0, 10, 0.2), (10, 10, 0.5), (3, 50, 0.1), (682, 925, 0.75), (12, 30, 0.4)]:
        for alt in ALTS:
            r = stats.binomtest(k, n, p, alternative=alt)
            row: dict[str, Any] = {"k": k, "n": n, "p": p, "alternative": alt, "pValue": r.pvalue, "ci": {}}
            for method, key in [("exact", "clopper-pearson"), ("wilson", "wilson"), ("wilsoncc", "wilson-cc")]:
                for level in [0.95, 0.9]:
                    c = r.proportion_ci(level, method=method)
                    row["ci"][f"{key}@{level}"] = [c.low, c.high]
            cases.append(row)
    # Wald and two-proportion z by formula; Newcombe from scipy's Wilson intervals.
    k1, n1, k2, n2 = 45, 400, 30, 380
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    q = stats.norm.isf(0.025)
    w1 = stats.binomtest(k1, n1).proportion_ci(0.95, method="wilson")
    w2 = stats.binomtest(k2, n2).proportion_ci(0.95, method="wilson")
    d = p1 - p2
    newcombe = [
        d - np.sqrt((p1 - w1.low) ** 2 + (w2.high - p2) ** 2),
        d + np.sqrt((w1.high - p1) ** 2 + (p2 - w2.low) ** 2),
    ]
    wald1 = [7 / 20 - q * np.sqrt(0.35 * 0.65 / 20), 7 / 20 + q * np.sqrt(0.35 * 0.65 / 20)]
    return {
        "cases": cases,
        "twoProportion": {
            "k1": k1,
            "n1": n1,
            "k2": k2,
            "n2": n2,
            "z": z,
            "p": 2 * stats.norm.sf(abs(z)),
            "wald": [d - q * se, d + q * se],
            "newcombe": newcombe,
        },
        "wald": wald1,
    }


def tables():
    obs = np.array([16, 18, 16, 14, 12, 12])
    exp = np.array([16, 16, 16, 16, 16, 8])
    gof = {
        "observed": obs,
        "expected": exp,
        "uniform": stats.chisquare(obs),
        "withExpected": stats.chisquare(obs, exp),
        "ddof": stats.chisquare(obs, ddof=1),
        "g": stats.power_divergence(obs, lambda_="log-likelihood"),
        "gExpected": stats.power_divergence(obs, exp, lambda_="log-likelihood"),
    }
    gof = {k: ([v.statistic, v.pvalue] if hasattr(v, "pvalue") else v) for k, v in gof.items()}
    contingency = []
    for t in [[[12, 5], [7, 9]], [[3, 9], [8, 2]], [[10, 20, 30], [6, 9, 17]], [[5, 1, 2], [3, 8, 4], [2, 3, 9]]]:
        row: dict[str, Any] = {"table": t}
        for corr in [True, False]:
            for lam in ["pearson", "log-likelihood"]:
                r = cast(Chi2Result, stats.chi2_contingency(t, correction=corr, lambda_=lam))
                row[f"{lam}/{corr}"] = [r.statistic, r.pvalue, r.dof]
        row["cramer"] = stats.contingency.association(t, method="cramer")
        contingency.append(row)
    fisher = []
    fisher_tables = [
        [[8, 2], [1, 5]],
        [[1, 9], [11, 3]],
        [[0, 5], [5, 0]],
        [[3, 1], [1, 3]],
        [[7, 12], [0, 15]],
        [[20, 15], [10, 25]],
    ]
    for t in fisher_tables:
        row = {"table": t}
        for alt in ALTS:
            r = cast(TestResult, stats.fisher_exact(t, alternative=alt))
            o = stats.contingency.odds_ratio(t, kind="conditional")
            c = o.confidence_interval(0.95, alternative=alt)
            row[alt] = {"p": r.pvalue, "or": o.statistic, "ci": [c.low, c.high]}
        s = stats.contingency.odds_ratio(t, kind="sample")
        c = s.confidence_interval(0.95)
        row["sample"] = {"or": s.statistic, "ci": [c.low, c.high]}
        fisher.append(row)
    return {"gof": gof, "contingency": contingency, "fisher": fisher}


def ranks():
    a = np.array([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30])
    b = np.array([0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29])
    c = np.array([2.1, 3.4, 1.9, 5.6, 4.4, 3.0])
    d = np.array([1.5, 2.2, 0.9, 1.7, 2.8])
    tied_x = np.array([1, 2, 2, 3, 3, 3, 5, 6, 6, 8, 9, 9])
    tied_y = np.array([2, 3, 4, 4, 4, 5, 7, 7, 7, 9, 10, 11, 12])
    big_x = np.round(np.linspace(0, 1, 11) ** 1.7 * 10, 4) + 0.123
    big_y = np.round(np.sin(np.arange(14)) * 4 + 6, 4)
    mwu = []
    for name, x, y, method, cc in [
        ("exact", c, d, "exact", True),
        ("tiesCorrected", tied_x, tied_y, "asymptotic", True),
        ("tiesPlain", tied_x, tied_y, "asymptotic", False),
        ("bigExact", big_x, big_y, "exact", True),
        ("bigAsymp", big_x, big_y, "asymptotic", True),
    ]:
        for alt in ALTS:
            r = cast(TestResult, stats.mannwhitneyu(x, y, alternative=alt, method=method, use_continuity=cc))
            mwu.append(
                {
                    "name": name,
                    "x": x,
                    "y": y,
                    "method": method,
                    "continuity": cc,
                    "alternative": alt,
                    "U": r.statistic,
                    "p": r.pvalue,
                }
            )
    wil = []
    diffs_tied = np.array([1.5, -0.5, 2.0, 2.0, 0.0, -1.0, 3.5, 1.0, -0.5, 4.0, 2.0, 0.0])
    for name, x, y, method, corr in [
        ("exact", a, b, "exact", False),
        ("approx", a, b, "approx", False),
        ("approxCorrected", a, b, "approx", True),
        ("tied", diffs_tied, None, "approx", False),
        ("tiedCorrected", diffs_tied, None, "approx", True),
    ]:
        for alt in ALTS:
            r = cast(
                TestResult, stats.wilcoxon(x, y, alternative=alt, method=method, correction=corr, zero_method="wilcox")
            )
            plus = cast(
                TestResult,
                stats.wilcoxon(x, y, alternative="greater", method=method, correction=corr, zero_method="wilcox"),
            ).statistic
            wil.append(
                {
                    "name": name,
                    "x": x,
                    "y": y,
                    "method": "exact" if method == "exact" else "asymptotic",
                    "correction": corr,
                    "alternative": alt,
                    "plus": plus,
                    "p": r.pvalue,
                }
            )
    return {"mannWhitney": mwu, "wilcoxon": wil}


def ks():
    x = np.array([0.1, -0.4, 0.3, 1.2, -1.5, 0.8, 2.1, -0.2, 0.05, -0.9])
    y = np.array([0.5, 1.1, 1.9, 0.7, 2.4, 1.3, 0.2, 1.8])
    out: dict[str, Any] = {"x": x, "y": y, "one": {}, "two": {}}
    for alt in ["greater", "less"]:
        r = cast(TestResult, stats.ks_1samp(x, stats.norm.cdf, alternative=alt, method="exact"))
        out["one"][alt] = [r.statistic, r.pvalue]
        r = cast(TestResult, stats.ks_2samp(x, y, alternative=alt, method="exact"))
        out["two"][alt] = [r.statistic, r.pvalue]
    out["smirnov"] = [[d, n, special.smirnov(n, d)] for d, n in [(0.1, 10), (0.3, 10), (0.05, 200), (0.4, 5)]]
    return out


def shapiro():
    rng = np.random.default_rng(7)
    out = []
    for n in [3, 4, 5, 8, 11, 12, 30, 200]:
        x = np.round(rng.gamma(2.0, 1.0, n) if n % 2 else rng.normal(0, 1, n), 4)
        r = stats.shapiro(x)
        out.append({"x": x, "W": r.statistic, "p": r.pvalue})
    return out


def ljung_box():
    rng = np.random.default_rng(3)
    e = rng.normal(size=80)
    x = np.zeros(80)
    for t in range(1, 80):
        x[t] = 0.4 * x[t - 1] + e[t]
    x = np.round(x, 4)
    n = len(x)
    a = x - x.mean()
    acf = np.array([np.sum(a[: n - k] * a[k:]) for k in range(11)]) / np.sum(a * a)
    out = {"x": x, "cases": []}
    for lags, fitted in [(1, 0), (5, 0), (10, 0), (10, 1)]:
        q = n * (n + 2) * np.sum(acf[1 : lags + 1] ** 2 / (n - np.arange(1, lags + 1)))
        bp = n * np.sum(acf[1 : lags + 1] ** 2)
        df = lags - fitted
        out["cases"].append(
            {
                "lags": lags,
                "fitted": fitted,
                "Q": q,
                "p": stats.chi2.sf(q, df),
                "boxPierce": bp,
                "pBoxPierce": stats.chi2.sf(bp, df),
            }
        )
    return out


def grubbs():
    x = np.array([2.1, 2.3, 1.9, 2.2, 2.0, 2.4, 2.1, 3.9, 2.2, 1.8])
    n = len(x)
    m, s = x.mean(), x.std(ddof=1)
    out: dict[str, Any] = {"x": x}
    for alt, g in [
        ("two-sided", np.max(np.abs(x - m)) / s),
        ("greater", (x.max() - m) / s),
        ("less", (m - x.min()) / s),
    ]:
        t = np.sqrt(n * (n - 2) * g**2 / ((n - 1) ** 2 - n * g**2))
        sides = 2 if alt == "two-sided" else 1
        out[alt] = {"G": g, "p": min(1.0, sides * n * stats.t.sf(t, n - 2))}
    # The tabulated two-sided 5% critical value for n = 10 (Grubbs, 1969): t at α/(2n), mapped to G.
    tc = stats.t.isf(0.05 / (2 * n), n - 2)
    out["critical10"] = (n - 1) / np.sqrt(n) * np.sqrt(tc**2 / (n - 2 + tc**2))
    return out


def anova():
    g = [np.array([4.2, 4.8, 5.1, 3.9, 4.4]), np.array([5.6, 6.1, 5.9, 6.4]), np.array([4.9, 5.2, 4.7, 5.5, 5.0, 5.3])]
    r = stats.f_oneway(*g)
    allv = np.concatenate(g)
    ssb = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in g)
    sst = np.sum((allv - allv.mean()) ** 2)
    return {"groups": g, "F": r.statistic, "p": r.pvalue, "eta2": ssb / sst}


def multiple():
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.20, 0.04, 0.5, 0.0001, 0.015, 0.8])
    m = len(p)
    o = np.argsort(p, kind="stable")
    ps = p[o]
    ranks = np.arange(1, m + 1)

    def unsort(v):
        out = np.empty(m)
        out[o] = v
        return out

    holm = unsort(np.minimum(1, np.maximum.accumulate((m - ranks + 1) * ps)))
    hoch = unsort(np.minimum(1, np.minimum.accumulate(((m - ranks + 1) * ps)[::-1])[::-1]))
    bh = unsort(np.minimum(1, np.minimum.accumulate((m / ranks * ps)[::-1])[::-1]))
    c = np.sum(1 / ranks)
    by = unsort(np.minimum(1, np.minimum.accumulate((c * m / ranks * ps)[::-1])[::-1]))
    return {
        "p": p,
        "bonferroni": np.minimum(1, m * p),
        "holm": holm,
        "hochberg": hoch,
        "benjaminiHochberg": bh,
        "benjaminiYekutieli": by,
        "scipyBH": stats.false_discovery_control(p, method="bh"),
        "scipyBY": stats.false_discovery_control(p, method="by"),
    }


def survival():
    time = np.array([6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35], dtype=float)
    event = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    data = stats.CensoredData(uncensored=time[event == 1], right=time[event == 0])
    r = stats.ecdf(data)
    keep = np.isin(r.sf.quantiles, np.unique(time[event == 1]))
    km: dict[str, Any] = {"time": r.sf.quantiles[keep], "survival": r.sf.probabilities[keep]}
    for method in ["linear", "log-log"]:
        c = r.sf.confidence_interval(0.95, method=method)
        km[method] = [c.low.probabilities[keep], c.high.probabilities[keep]]
    # Nelson–Aalen with Aalen's variance, by its definition.
    ut = np.unique(time[event == 1])
    nrisk = np.array([np.sum(time >= t) for t in ut])
    d = np.array([np.sum((time == t) & (event == 1)) for t in ut])
    H = np.cumsum(d / nrisk)
    se = np.sqrt(np.cumsum(d / nrisk**2))
    z = stats.norm.isf(0.025)
    na = {"time": ut, "H": H, "se": se, "lower": H * np.exp(-z * se / H), "upper": H * np.exp(z * se / H)}
    # Log-rank: two groups against scipy; three groups by the k-sample formula.
    t2 = np.array([4, 7, 9, 12, 15, 18, 22, 25, 30, 33, 3, 5, 6, 8, 9, 11, 14, 16, 20, 21], dtype=float)
    e2 = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    g2 = np.array([0] * 10 + [1] * 10)

    def cens(mask):
        return stats.CensoredData(uncensored=t2[mask & (e2 == 1)], right=t2[mask & (e2 == 0)])

    lr = stats.logrank(cens(g2 == 0), cens(g2 == 1))
    g3 = np.array([0, 1, 2] * 6 + [0, 1])

    def k_sample(t, e, g):
        labels = np.unique(g)
        G = len(labels)
        observed = np.zeros(G)
        E = np.zeros(G)
        V = np.zeros((G, G))
        for u in np.unique(t[e == 1]):
            at = t >= u
            n = at.sum()
            dd = ((t == u) & (e == 1)).sum()
            ng = np.array([(at & (g == lab)).sum() for lab in labels])
            dg = np.array([((t == u) & (e == 1) & (g == lab)).sum() for lab in labels])
            observed += dg
            E += dd * ng / n
            if n > 1:
                V += dd * (n - dd) / (n - 1) * (np.diag(ng / n) - np.outer(ng / n, ng / n))
        diff = (observed - E)[:-1]
        stat = diff @ np.linalg.solve(V[:-1, :-1], diff)
        return stat, stats.chi2.sf(stat, G - 1)

    s3, p3 = k_sample(t2, e2, g3)
    return {
        "time": time,
        "event": event,
        "km": km,
        "na": na,
        "logrank": {
            "time": t2,
            "event": e2,
            "group": g2,
            "chi2": lr.statistic**2,
            "p": lr.pvalue,
            "group3": g3,
            "chi2_3": s3,
            "p3": p3,
        },
    }


def boundaries():
    """Group-sequential boundaries solved with the multivariate normal cdf: corr(Z_i, Z_j) = √(t_i/t_j)."""

    def mvn_inside(t: NDArray[np.float64], c: list[float], sides: int) -> float:
        k = len(c)
        cov = np.array([[np.sqrt(min(t[i], t[j]) / max(t[i], t[j])) for j in range(k)] for i in range(k)])
        upper = np.array(c)
        lower = -upper if sides == 2 else np.full(k, -np.inf)
        if k == 1:
            return float(stats.norm.cdf(upper[0]) - stats.norm.cdf(lower[0]))
        # Genz's quasi-Monte Carlo integration draws random shifts; a fixed rng makes the fixture reproducible.
        # scipy's signature infers cov's type from its default (1); a covariance matrix is what it accepts.
        cov_arg = cast(Any, cov)
        p = stats.multivariate_normal.cdf(
            upper, mean=np.zeros(k), cov=cov_arg, lower_limit=lower, abseps=1e-10, releps=1e-10, maxpts=4_000_000, rng=0
        )
        return float(p)

    def excess(ck: float, t: NDArray[np.float64], c: list[float], sides: int, target: float) -> float:
        # P(stopped by look k) = 1 − P(inside at every look ≤ k), less the α to spend by look k.
        return (1 - mvn_inside(t, [*c, ck], sides)) - target

    def spend(fam: str, alpha: float, t: float) -> float:
        if fam == "obrien-fleming":
            return float(2 - 2 * stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t)))
        if fam == "pocock":
            return float(alpha * np.log(1 + (np.e - 1) * t))
        return alpha * t**2

    out = []
    designs: list[tuple[str, list[float], int, float]] = [
        ("obrien-fleming", [1, 2, 3, 4], 2, 0.05),
        ("pocock", [1, 2, 3], 2, 0.05),
        ("power", [0.3, 0.7, 1.0], 1, 0.025),
        ("obrien-fleming", [0.25, 0.6, 1.0], 1, 0.025),
    ]
    for fam, info, sides, alpha in designs:
        t = np.array(info, dtype=float) / info[-1]
        c: list[float] = []
        for k in range(len(t)):
            # Two-sided designs spend α/2 on each side (Lan and DeMets' one-sided function at α/2, doubled).
            target = 2 * spend(fam, alpha / 2, float(t[k])) if sides == 2 else spend(fam, alpha, float(t[k]))
            ck = cast(float, optimize.brentq(excess, 0.5, 12, args=(t[: k + 1], c, sides, target), xtol=1e-10))
            c.append(ck)
        out.append({"family": fam, "information": info, "sides": sides, "alpha": alpha, "z": c})
    # Classical constant boundaries (Jennison and Turnbull, 2000, Tables 2.1 and 2.3), K = 5, two-sided α = 0.05.
    # Lan–DeMets O'Brien–Fleming spending, K = 5 equally spaced, two-sided α = 0.05 (as R's ldbounds and gsDesign).
    return {"spending": out, "pocock5": 2.413, "obrienFleming5": 2.040, "ldObf5": [4.877, 3.357, 2.680, 2.290, 2.031]}


def f_distribution():
    rows = []
    for d1, d2 in [(5, 20), (1, 3), (3.5, 7.2), (12, 40)]:
        # stats.f is an rv_continuous; Pyright infers the frozen type from a generic base without logpdf.
        f = cast(Any, stats.f(d1, d2))
        x = np.array([1e-4, 0.1, 0.5, 1, 2, 5, 20, 100])
        p = np.array([1e-10, 0.01, 0.3, 0.5, 0.9, 0.999])
        rows.append(
            {
                "d1": d1,
                "d2": d2,
                "x": x,
                "logpdf": f.logpdf(x),
                "cdf": f.cdf(x),
                "sf": f.sf(x),
                "p": p,
                "ppf": f.ppf(p),
                "isf": f.isf(p),
                "mean": f.mean(),
                "var": f.var(),
                "entropy": f.entropy(),
            }
        )
    return rows


# ── Second references, keyed by registry name ────────────────────────────────────────────────────────────────────────

SM_ALT = {"two-sided": "two-sided", "less": "smaller", "greater": "larger"}


def interval(c: Any) -> list[float]:
    lo, hi = c
    return [float(lo), float(hi)]


def references() -> dict[str, Any]:
    """Each registered test against a second library where one exists (statsmodels, lifelines), on fresh inputs with
    ties, zeros, small and lopsided samples. Keys are the registry names, so the catalog counts them as covered."""
    from lifelines.statistics import multivariate_logrank_test
    from statsmodels.stats import proportion as smp
    from statsmodels.stats.contingency_tables import Table
    from statsmodels.stats.gof import chisquare as sm_chisquare
    from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

    rng = np.random.default_rng(20261002)
    x = np.round(rng.normal(1.0, 2.0, 15), 2)
    y = np.round(rng.normal(0.2, 0.7, 6), 2)
    x2 = np.round(x[:6] + rng.normal(0.3, 0.5, 6), 2)
    out: dict[str, Any] = {"x": x, "y": y, "x2": x2}

    def t_row(test: str, alt: str, level: float, mu: float, r: Any, c: Any) -> dict[str, Any]:
        stat, pv, df = r
        return {
            "test": test,
            "alternative": alt,
            "level": level,
            "mu": mu,
            "statistic": stat,
            "p": pv,
            "df": df,
            "ci": interval(c),
        }

    one, paired, pooled, welch = [], [], [], []
    for alt in ALTS:
        for level, mu in [(0.95, 0.0), (0.9, 1.5)]:
            # statsmodels annotates the null value `value` as int; it takes any float.
            value = cast(Any, mu)
            d = DescrStatsW(x)
            one.append(
                t_row(
                    "oneSampleTTest",
                    alt,
                    level,
                    mu,
                    d.ttest_mean(value, SM_ALT[alt]),
                    d.tconfint_mean(1 - level, SM_ALT[alt]),
                )
            )
            dd = DescrStatsW(x2 - y)
            r = dd.ttest_mean(value, SM_ALT[alt])
            paired.append(t_row("pairedTTest", alt, level, mu, r, dd.tconfint_mean(1 - level, SM_ALT[alt])))
            cm = CompareMeans(DescrStatsW(x), DescrStatsW(y))
            for usevar, rows in [("pooled", pooled), ("unequal", welch)]:
                r = cm.ttest_ind(SM_ALT[alt], usevar, value)
                # tconfint_diff ignores `value`; the interval is for the difference itself.
                c = cm.tconfint_diff(1 - level, SM_ALT[alt], usevar)
                rows.append(t_row("pooledTTest" if usevar == "pooled" else "welchTTest", alt, level, mu, r, c))
    out["oneSampleTTest"], out["pairedTTest"], out["pooledTTest"], out["welchTTest"] = one, paired, pooled, welch

    # z-tests with a known σ: no library takes σ as given, so the statistic is the formula and the tails scipy's.
    z = []
    for alt in ALTS:
        for sigma, sigma_y, two, mu in [(2.0, None, False, 0.5), (1.5, None, True, 0.0), (2.0, 0.7, True, 0.4)]:
            se = sigma / np.sqrt(len(x)) if not two else np.sqrt(sigma**2 / len(x) + (sigma_y or sigma) ** 2 / len(y))
            est = x.mean() - (y.mean() if two else 0.0)
            zz = (est - mu) / se
            pv = {"two-sided": 2 * stats.norm.sf(abs(zz)), "greater": stats.norm.sf(zz), "less": stats.norm.cdf(zz)}
            z.append(
                {
                    "alternative": alt,
                    "sigma": sigma,
                    "sigmaY": sigma_y,
                    "two": two,
                    "mu": mu,
                    "statistic": zz,
                    "p": pv[alt],
                }
            )
    out["zTest"] = z

    binom = []
    for k, n, p0 in [(7, 20, 0.5), (0, 12, 0.3), (12, 12, 0.8), (31, 200, 0.2), (3, 10, 0.05)]:
        for alt in ALTS:
            r = stats.binomtest(k, n, p0, alternative=alt)
            # proportion_confint names the side of the interval's finite end: `larger` bounds p from above, the
            # interval that goes with the alternative `less`.
            side = {"two-sided": "two-sided", "less": "larger", "greater": "smaller"}[alt]
            ci = smp.proportion_confint(k, n, 0.05, method="beta", alternative=side)
            binom.append({"k": k, "n": n, "p0": p0, "alternative": alt, "p": r.pvalue, "ci": interval(ci)})
    out["binomialTest"] = binom

    two_prop = []
    for k1, n1, k2, n2 in [(45, 100, 30, 100), (3, 40, 9, 35), (0, 20, 4, 25)]:
        for alt in ALTS:
            zz, pv = smp.proportions_ztest([k1, k2], [n1, n2], alternative=SM_ALT[alt])
            row: dict[str, Any] = {"k1": k1, "n1": n1, "k2": k2, "n2": n2, "alternative": alt, "z": zz, "p": pv}
            if alt == "two-sided":
                row["wald"] = interval(smp.confint_proportions_2indep(k1, n1, k2, n2, method="wald", alpha=0.05))
            two_prop.append(row)
    out["twoProportionZTest"] = two_prop

    gof_cases = [
        ([18, 22, 21, 39], None, 0),
        ([18, 22, 21, 39], [0.2, 0.2, 0.2, 0.4], 0),
        ([5, 9, 14, 30, 22, 10, 6, 4], None, 2),
        ([0, 3, 7, 10], [0.1, 0.2, 0.3, 0.4], 1),
    ]
    chi_gof, g_gof = [], []
    for obs, probs, ddof in gof_cases:
        o = np.array(obs, dtype=float)
        e = None if probs is None else np.array(probs) * o.sum()
        stat, pv = cast(tuple[float, float], sm_chisquare(o, e, ddof=ddof))
        chi_gof.append({"observed": obs, "probabilities": probs, "ddof": ddof, "statistic": stat, "p": pv})
        # G = 2 Σ o log(o/e) with 0 log 0 = 0, against the χ² law with k − 1 − ddof degrees of freedom.
        ee = np.full_like(o, o.mean()) if e is None else e
        g = 2 * np.sum(special.xlogy(o, o / ee))
        g_gof.append(
            {
                "observed": obs,
                "probabilities": probs,
                "ddof": ddof,
                "statistic": g,
                "p": stats.chi2.sf(g, len(o) - 1 - ddof),
            }
        )
    out["chiSquareGoodnessOfFit"], out["gTestGoodnessOfFit"] = chi_gof, g_gof

    ind_tables = [
        [[25, 15], [10, 30]],
        [[2, 8], [7, 3]],
        [[12, 7, 3], [5, 11, 9]],
        [[8, 4, 2, 1], [3, 6, 5, 2], [1, 2, 7, 9]],
    ]
    chi_ind, g_ind = [], []
    for t in ind_tables:
        a = np.array(t, dtype=float)
        nominal = cast(TTestResult, Table(a).test_nominal_association())
        row = {"table": t, "plain": [nominal.statistic, nominal.pvalue, nominal.df]}
        if a.shape == (2, 2):
            r = cast(Chi2Result, stats.chi2_contingency(a, correction=True))
            row["yates"] = [r.statistic, r.pvalue, r.dof]
        chi_ind.append(row)
        # G by its definition from the expected counts (Yates-shifted counts for 2 × 2 when corrected).
        e = a.sum(1, keepdims=True) * a.sum(0, keepdims=True) / a.sum()
        df = (a.shape[0] - 1) * (a.shape[1] - 1)
        g_plain = 2 * np.sum(special.xlogy(a, a / e))
        grow: dict[str, Any] = {"table": t, "plain": [g_plain, stats.chi2.sf(g_plain, df), df]}
        if a.shape == (2, 2):
            shift = np.minimum(0.5, np.abs(a - e))
            ac = a - np.sign(a - e) * shift
            g_c = 2 * np.sum(special.xlogy(ac, ac / e))
            grow["yates"] = [g_c, stats.chi2.sf(g_c, df), df]
        g_ind.append(grow)
    out["chiSquareIndependence"], out["gTestIndependence"] = chi_ind, g_ind

    fisher = []
    for t in [[[3, 1], [1, 3]], [[10, 2], [3, 15]], [[0, 7], [6, 1]], [[25, 20], [15, 30]], [[1, 0], [0, 1]]]:
        for alt in ALTS:
            r = cast(TestResult, stats.fisher_exact(t, alternative=alt))
            fisher.append({"table": t, "alternative": alt, "p": r.pvalue})
    out["fisherExact"] = fisher

    ks = []
    u = np.round(rng.normal(0.3, 1.2, 25), 3)
    v = np.round(rng.standard_t(3, 18), 3)
    for alt in ALTS:
        for method in ["exact", "asymp"]:
            r = cast(TestResult, stats.ks_1samp(u, stats.norm(0, 1.5).cdf, alternative=alt, method=method))
            ks.append(
                {
                    "sample": "one",
                    "loc": 0.0,
                    "scale": 1.5,
                    "alternative": alt,
                    "method": method,
                    "statistic": r.statistic,
                    "p": r.pvalue,
                }
            )
            r = cast(TestResult, stats.ks_2samp(u, v, alternative=alt, method=method))
            ks.append({"sample": "two", "alternative": alt, "method": method, "statistic": r.statistic, "p": r.pvalue})
    out["ksTest"] = {"u": u, "v": v, "cases": ks}

    mwu = []
    c = np.round(rng.gamma(2.0, 1.0, 7), 2)
    d = np.round(rng.gamma(3.0, 1.0, 5), 2)
    tx = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
    ty = np.array([2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5])
    for name, a1, b1, method, cc in [
        ("exact", c, d, "exact", True),
        ("auto", c, d, "auto", True),
        ("ties", tx, ty, "asymptotic", True),
        ("tiesPlain", tx, ty, "asymptotic", False),
        ("tiesAuto", tx, ty, "auto", True),
    ]:
        for alt in ALTS:
            r = cast(TestResult, stats.mannwhitneyu(a1, b1, alternative=alt, method=method, use_continuity=cc))
            mwu.append(
                {
                    "name": name,
                    "x": a1,
                    "y": b1,
                    "method": method,
                    "continuity": cc,
                    "alternative": alt,
                    "U": r.statistic,
                    "p": r.pvalue,
                }
            )
    out["mannWhitneyU"] = mwu

    wil = []
    e1 = np.round(rng.normal(0.4, 1.0, 14), 2)
    e2 = np.array([0.5, -1.0, 2.0, 0.5, 0.0, 1.5, -0.5, 3.0, 0.5, 2.0, 0.0, -2.0, 1.0, 1.5, 2.5, -0.5])
    e3 = np.round(rng.normal(0.2, 1.0, 70), 2)  # n > 50: scipy's auto is the normal approximation
    for name, dv, method, corr in [
        ("exact", e1, "exact", False),
        ("auto", e1, "auto", False),
        ("approx", e1, "approx", True),
        ("tiesZeros", e2, "approx", False),
        ("tiesZerosCorrected", e2, "approx", True),
        ("large", e3, "auto", False),
    ]:
        dd = dv[dv != 0]
        rk = stats.rankdata(np.abs(dd))
        for alt in ALTS:
            r = cast(
                TestResult, stats.wilcoxon(dv, alternative=alt, method=method, correction=corr, zero_method="wilcox")
            )
            wil.append(
                {
                    "name": name,
                    "d": dv,
                    "method": "asymptotic" if method == "approx" else method,
                    "correction": corr,
                    "alternative": alt,
                    "plus": float(rk[dd > 0].sum()),
                    "p": r.pvalue,
                }
            )
    out["wilcoxonSignedRank"] = wil

    # Log-rank against lifelines: two groups with tied event times and censoring, and three groups.
    lr = []
    for groups in [2, 3]:
        n = 30
        g = np.arange(n) % groups
        t = np.round(rng.exponential(10.0 / (1 + 0.6 * g), n)).clip(1)
        ev = (rng.random(n) < 0.75).astype(int)
        r = multivariate_logrank_test(t, g, ev)
        lr.append({"time": t, "event": ev, "group": g, "statistic": r.test_statistic, "p": r.p_value, "df": groups - 1})
    out["logRankTest"] = lr
    return out


def cases() -> dict[str, Any]:
    return {
        "t": t_tests(),
        "binomial": binomial(),
        "tables": tables(),
        "ranks": ranks(),
        "ks": ks(),
        "shapiro": shapiro(),
        "ljungBox": ljung_box(),
        "grubbs": grubbs(),
        "anova": anova(),
        "multiple": multiple(),
        "survival": survival(),
        "boundaries": boundaries(),
        "f": f_distribution(),
        "references": references(),
    }
