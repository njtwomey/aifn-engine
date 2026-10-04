"""Golden values for aifn-compute/probability/distributions from scipy.stats (and torch.distributions for KL
divergences): for
every family, log density, cdf, log cdf, survival and log survival over a wide grid that reaches far into both tails,
quantiles at extreme probabilities (down to 1e-300), mean, variance and entropy; batched parameters with broadcasting;
the multivariate families (MultivariateNormal, Dirichlet, Multinomial, Wishart); the compositions (Mixture, Independent,
Transformed, Pushforward); and every registered closed-form KL rule. Families are keyed by registry key."""

from collections.abc import Callable
from typing import Any, Protocol, cast

import mpmath as mp
import numpy as np
import torch
from numpy.typing import ArrayLike
from scipy import special, stats
from torch import distributions as td

P_GRID = [1e-300, 1e-100, 1e-12, 1e-6, 0.01, 0.3, 0.5, 0.9, 0.99, 1 - 1e-6, 1 - 1e-12]

# key → (constructor args, the scipy distribution, x grid). Grids reach where the cdf or survival underflows.
CONTINUOUS = {
    "Normal": ([1.5, 2.0], stats.norm(1.5, 2.0), np.r_[-80, -40, -20, -5, np.linspace(-3, 6, 10), 10, 25, 45, 80]),
    "LogNormal": (
        [0.3, 0.8],
        stats.lognorm(0.8, scale=np.exp(0.3)),
        np.r_[1e-8, 1e-4, 0.01, 0.2, 0.7, 1, 1.5, 3, 10, 50, 1e3, 1e5],
    ),
    "StudentT": (
        [3.5, -1.0, 1.5],
        stats.t(3.5, -1.0, 1.5),
        np.r_[-1e6, -1e3, -30, -5, -1, 0, 0.5, 2, 10, 100, 1e4, 1e8],
    ),
    "Cauchy": ([0.5, 2.0], stats.cauchy(0.5, 2.0), np.r_[-1e10, -1e4, -20, -1, 0.5, 2, 30, 1e5, 1e12]),
    "Laplace": ([1.0, 0.7], stats.laplace(1.0, 0.7), np.r_[-400, -50, -5, 0, 0.9, 1, 1.1, 3, 20, 300, 480]),
    "Logistic": ([-0.5, 1.3], stats.logistic(-0.5, 1.3), np.r_[-900, -100, -30, -3, -0.5, 0, 2, 20, 60, 500, 900]),
    "Uniform": ([-2.0, 3.0], stats.uniform(-2.0, 5.0), np.r_[-1.999999, -1.5, 0, 1.2, 2.99, 2.9999999]),
    "Exponential": ([2.5], stats.expon(scale=1 / 2.5), np.r_[1e-12, 1e-6, 0.01, 0.4, 1, 5, 30, 150, 280]),
    "Gamma": ([2.7, 1.8], stats.gamma(2.7, scale=1 / 1.8), np.r_[1e-30, 1e-8, 0.01, 0.5, 1.5, 4, 20, 100, 400]),
    "GammaWithScale": ([0.4, 3.0], stats.gamma(0.4, scale=3.0), np.r_[1e-40, 1e-10, 1e-3, 0.2, 1, 5, 40, 500, 2000]),
    "ChiSquare": ([7.0], stats.chi2(7.0), np.r_[1e-20, 1e-5, 0.3, 2, 7, 15, 50, 300, 1500]),
    "InverseGamma": ([3.0, 2.0], stats.invgamma(3.0, scale=2.0), np.r_[1e-3, 0.02, 0.1, 0.5, 1, 3, 30, 1e4, 1e8]),
    "Beta": ([2.5, 0.6], stats.beta(2.5, 0.6), np.r_[1e-30, 1e-8, 1e-3, 0.1, 0.5, 0.9, 0.999, 1 - 1e-9, 1 - 1e-14]),
    "Weibull": ([1.7, 2.2], stats.weibull_min(1.7, scale=2.2), np.r_[1e-20, 1e-5, 0.1, 1, 2.2, 5, 20, 60, 150]),
    "Gumbel": ([0.5, 1.5], stats.gumbel_r(0.5, 1.5), np.r_[-8, -4, -1, 0, 0.5, 2, 10, 60, 400, 1000]),
    "GeneralisedPareto": (
        [0.3, 1.0, 2.0],
        stats.genpareto(0.3, 1.0, 2.0),
        np.r_[1.0 + 1e-9, 1.001, 1.5, 2, 4, 10, 100, 1e4, 1e8],
    ),
    "GeneralisedParetoBounded": (
        [-0.4, 0.0, 1.5],
        stats.genpareto(-0.4, 0.0, 1.5),
        np.r_[1e-9, 0.01, 0.5, 1.5, 3, 3.7, 3.7499999],
    ),
    "VonMises": ([0.7, 2.5], stats.vonmises(2.5, loc=0.7), np.r_[-2.4, -1.5, -0.5, 0, 0.7, 1.5, 2.8, 3.8]),
    "TruncatedNormal": (
        [0.5, 1.2, -1.0, 4.0],
        stats.truncnorm((-1.0 - 0.5) / 1.2, (4.0 - 0.5) / 1.2, 0.5, 1.2),
        np.r_[-0.999999, -0.9, 0, 0.5, 1.5, 3, 3.99, 3.9999999],
    ),
    # A far-tail truncation (the TrueSkill and probit cases): N(0, 1) on [8, ∞).
    "TruncatedNormalTail": (
        [0.0, 1.0, 8.0, np.inf],
        stats.truncnorm(8.0, np.inf),
        np.r_[8.000001, 8.01, 8.3, 9, 10, 12],
    ),
}

DISCRETE = {
    "Bernoulli": ([0.3], stats.bernoulli(0.3), [0, 1]),
    "Binomial": ([40, 0.15], stats.binom(40, 0.15), [0, 1, 3, 6, 10, 20, 35, 40]),
    "Poisson": ([3.7], stats.poisson(3.7), [0, 1, 3, 4, 8, 15, 40, 100, 200]),
    "PoissonLarge": ([850.0], stats.poisson(850.0), [600, 780, 849, 850, 900, 1000, 1200]),
    "Geometric": ([0.12], stats.geom(0.12), [1, 2, 5, 10, 40, 150, 400]),
    "NegativeBinomial": ([3.5, 0.4], stats.nbinom(3.5, 0.4), [0, 1, 3, 5, 10, 30, 80, 200]),
    "Hypergeometric": ([50, 12, 20], stats.hypergeom(50, 12, 20), [0, 1, 3, 5, 7, 10, 12]),
    "DiscreteUniform": ([-3, 4], stats.randint(-3, 5), [-3, -1, 0, 2, 4]),
}


class Frozen(Protocol):
    """A frozen scipy distribution. Calling an rv_continuous or rv_discrete is typed as returning either frozen kind,
    so Pyright cannot see logpdf on a continuous one; this names the methods the fixtures use."""

    def pdf(self, x: Any) -> Any: ...
    def logpdf(self, x: Any) -> Any: ...
    def logpmf(self, x: Any) -> Any: ...
    def cdf(self, x: Any) -> Any: ...


def frozen(dist: object) -> Frozen:
    return cast(Frozen, dist)


ALIASES = {
    "TruncatedNormalTail": "TruncatedNormal",
    "PoissonLarge": "Poisson",
    "GeneralisedParetoBounded": "GeneralisedPareto",
}


def student_t_quantile(p: float, nu: float, loc: float, scale: float) -> float:
    """The lower-tail quantile of Student's t at tiny p, by solving log I_{ν/(ν+x²)}(ν/2, 1/2) = log 2p in mpmath."""
    mp.mp.dps = 60
    nu_ = mp.mpf(nu)

    def logcdf(z: Any) -> Any:
        return mp.log(mp.betainc(nu_ / 2, mp.mpf(1) / 2, 0, nu_ / (nu_ + z * z), regularized=True) / 2)

    target = mp.log(mp.mpf(p))
    lo, hi = mp.mpf(-1e300), mp.mpf(-1)
    for _ in range(2000):
        mid = -mp.sqrt(lo * hi)
        if logcdf(mid) < target:
            lo = mid
        else:
            hi = mid
        if abs(hi / lo - 1) < mp.mpf(10) ** -30:
            break
    return float(loc + scale * hi)


def finite_or_none(f: Callable[[], Any]) -> float | None:
    with np.errstate(all="ignore"):
        try:
            v = float(f())
        except (ValueError, OverflowError):
            return None
    return v


def univariate(dist: Any, x: ArrayLike, discrete: bool) -> dict[str, object]:
    with np.errstate(all="ignore"):
        cdf, sf = dist.cdf(x), dist.sf(x)
        out: dict[str, object] = {
            "x": np.asarray(x, float),
            "logProb": dist.logpmf(x) if discrete else dist.logpdf(x),
            "cdf": cdf,
            # scipy's default logcdf and logsf are log(cdf) and log(sf), which round to 0 where the value is near 1:
            # there the reference is log1p of minus the other tail (sf and cdf are accurate in their own tails).
            "logcdf": np.where(cdf > 0.5, np.log1p(-sf), dist.logcdf(x)),
            "survival": sf,
            "logSurvival": np.where(sf > 0.5, np.log1p(-cdf), dist.logsf(x)),
            "p": P_GRID,
            "quantile": dist.ppf(P_GRID),
            "mean": finite_or_none(dist.mean),
            "variance": finite_or_none(dist.var),
            "entropy": finite_or_none(dist.entropy),
        }
    return out


def cases() -> dict[str, object]:
    out: dict[str, Any] = {}
    fams: dict[str, Any] = {}
    for name, (args, dist, x) in CONTINUOUS.items():
        fams[name] = {"family": ALIASES.get(name, name), "args": args, **univariate(dist, x, False)}
    for name, (args, dist, x) in DISCRETE.items():
        fams[name] = {"family": ALIASES.get(name, name), "args": args, **univariate(dist, x, True)}
    # VonMises: scipy's cdf on [μ − π, μ + π] counts from −∞ in periods; aifn's runs from μ − π. Keep density and
    # moments only (scipy reports the circular variance as `var`? it does not; drop it).
    # scipy's t.ppf returns −inf below p ≈ 1e-290 (and loses accuracy before): solve cdf(x) = p in mpmath instead.
    st = fams["StudentT"]
    st["quantile"] = [
        student_t_quantile(p, 3.5, -1.0, 1.5) if p < 1e-50 else q for p, q in zip(P_GRID, st["quantile"], strict=True)
    ]
    # The gamma family's log tails in mpmath: scipy's underflow to −inf (log Q(3.5, 750), log Q(3, 2000)).
    mp.mp.dps = 50

    def log_tail(a: float, t: float, upper: bool) -> float:
        # log of the smaller tail directly, log1p of minus it for the larger (50 digits cannot hold 1 − 1e-75).
        p = mp.gammainc(a, 0, t, regularized=True)
        q = mp.gammainc(a, t, mp.inf, regularized=True)
        own, other = (q, p) if upper else (p, q)
        return float(mp.log(own) if own <= other else mp.log1p(-other))

    def log_p(a: float, t: float) -> float:
        return log_tail(a, t, False)

    def log_q(a: float, t: float) -> float:
        return log_tail(a, t, True)

    gamma_tails = {
        "Gamma": (2.7, lambda v: v * 1.8, False),
        "GammaWithScale": (0.4, lambda v: v / 3.0, False),
        "ChiSquare": (3.5, lambda v: v / 2.0, False),
        "InverseGamma": (3.0, lambda v: 2.0 / v, True),
    }
    for name, (a, t, inverse) in gamma_tails.items():
        f = fams[name]
        lower = [log_p(a, t(v)) for v in f["x"]]
        upper = [log_q(a, t(v)) for v in f["x"]]
        f["logcdf"], f["logSurvival"] = (upper, lower) if inverse else (lower, upper)
    # Uniform: scipy's sf is 1 − cdf, which loses 4e-9 relative near `high`; the exact values are (high − x)/width.
    un = fams["Uniform"]
    exact = [(mp.mpf(v) + 2) / 5 for v in un["x"]]
    un["cdf"] = [float(c) for c in exact]
    un["survival"] = [float(1 - c) for c in exact]
    un["logcdf"] = [float(mp.log(c)) for c in exact]
    un["logSurvival"] = [float(mp.log(1 - c)) for c in exact]
    vm = fams["VonMises"]
    for k in ("cdf", "logcdf", "survival", "logSurvival", "quantile", "variance"):
        vm.pop(k)
    vm["variance"] = 1 - special.i1(2.5) / special.i0(2.5)
    vm["entropy"] = float(stats.vonmises(2.5, loc=0.7).entropy())
    # Categorical: the pmf of one probability vector, by direct computation.
    probs = np.array([0.1, 0.25, 0.05, 0.6])
    fams["Categorical"] = {
        "family": "Categorical",
        "args": [probs],
        "x": [0, 1, 2, 3],
        "logProb": np.log(probs),
        "mean": float(probs @ np.arange(4)),
        "variance": float(probs @ np.arange(4) ** 2 - (probs @ np.arange(4)) ** 2),
        "entropy": float(stats.entropy(probs)),
    }
    out["families"] = fams

    # Batches: parameters of shape [3] (and a scalar broadcast) at x of shape [4, 3].
    rng = np.random.default_rng(20261001)
    loc = np.array([0.0, 1.0, -2.0])
    scale = np.array([1.0, 0.5, 3.0])
    xb = rng.normal(size=(4, 3)) * 2
    shape = np.array([0.5, 2.0, 7.0])
    xg = rng.uniform(0.1, 6, size=(4, 3))
    nb = np.array([5, 10, 20])
    kb = np.array([[0, 3, 7], [2, 5, 20], [5, 0, 1], [1, 9, 13]])
    out["batches"] = {
        "Normal": {
            "args": [loc, scale],
            "x": xb,
            "logProb": frozen(stats.norm(loc, scale)).logpdf(xb),
            "cdf": stats.norm(loc, scale).cdf(xb),
        },
        "NormalBroadcast": {
            "family": "Normal",
            "args": [loc, 2.0],
            "x": xb,
            "logProb": frozen(stats.norm(loc, 2.0)).logpdf(xb),
            "cdf": stats.norm(loc, 2.0).cdf(xb),
        },
        "Gamma": {
            "args": [shape, 1.5],
            "x": xg,
            "logProb": frozen(stats.gamma(shape, scale=1 / 1.5)).logpdf(xg),
            "cdf": stats.gamma(shape, scale=1 / 1.5).cdf(xg),
        },
        "Binomial": {
            "args": [nb, 0.3],
            "x": kb,
            "logProb": frozen(stats.binom(nb, 0.3)).logpmf(kb),
            "cdf": stats.binom(nb, 0.3).cdf(kb),
        },
        "StudentT": {
            "args": [np.array([1.5, 4.0, 30.0]), 0.0, 1.0],
            "x": xb,
            "logProb": frozen(stats.t(np.array([1.5, 4.0, 30.0]))).logpdf(xb),
            "cdf": stats.t(np.array([1.5, 4.0, 30.0])).cdf(xb),
        },
        "Beta": {
            "args": [np.array([0.5, 2.0, 5.0]), np.array([0.5, 3.0, 1.0])],
            "x": rng.uniform(0.01, 0.99, (4, 3)),
            "logProb": None,
            "cdf": None,
        },
    }
    bb = out["batches"]["Beta"]
    bdist = frozen(stats.beta(np.array([0.5, 2.0, 5.0]), np.array([0.5, 3.0, 1.0])))
    bb["logProb"], bb["cdf"] = bdist.logpdf(bb["x"]), bdist.cdf(bb["x"])

    # Multivariate.
    d = 3
    M = rng.normal(size=(d, d))
    cov = M @ M.T + 0.5 * np.eye(d)
    mu = np.array([0.5, -1.0, 2.0])
    xm = rng.normal(size=(5, d)) * 2
    alpha = np.array([0.7, 2.0, 4.5])
    xd = rng.dirichlet(alpha, size=5)
    pm = np.array([0.2, 0.5, 0.3])
    km = np.array([[3, 4, 3], [0, 10, 0], [1, 1, 8], [5, 5, 0]])
    df = 5.5
    # scipy's signature infers cov's type from its default (1); a covariance matrix is what it accepts.
    mvn_cov = cast(Any, cov)
    W = stats.wishart(df, cov).rvs(size=3, random_state=1)
    multi = {
        "MultivariateNormal": {
            "loc": mu,
            "covariance": cov,
            "x": xm,
            "logProb": stats.multivariate_normal(mu, mvn_cov).logpdf(xm),
            "entropy": stats.multivariate_normal(mu, mvn_cov).entropy(),
        },
        "Dirichlet": {
            "concentration": alpha,
            "x": xd,
            "logProb": [stats.dirichlet(alpha).logpdf(v) for v in xd],
            "entropy": stats.dirichlet(alpha).entropy(),
            "mean": stats.dirichlet(alpha).mean(),
            "variance": stats.dirichlet(alpha).var(),
        },
        "Multinomial": {
            "n": 10,
            "p": pm,
            "x": km,
            "logProb": stats.multinomial(10, pm).logpmf(km),
            "entropy": stats.multinomial(10, pm).entropy(),
        },
        "Wishart": {
            "df": df,
            "scale": cov,
            "x": W,
            "logProb": [stats.wishart(df, cov).logpdf(w) for w in W],
            "entropy": stats.wishart(df, cov).entropy(),
            "mean": stats.wishart(df, cov).mean(),
        },
    }
    out["multivariate"] = multi

    xs = np.r_[-6, -2, -0.5, 0, 0.7, 2, 5, 9]
    w = np.array([0.3, 0.5, 0.2])
    comps = [(-2.0, 0.5), (0.5, 1.0), (4.0, 2.0)]
    mix_pdf = sum(wk * frozen(stats.norm(m, s)).pdf(xs) for wk, (m, s) in zip(w, comps, strict=True))
    mix_cdf = sum(wk * stats.norm(m, s).cdf(xs) for wk, (m, s) in zip(w, comps, strict=True))
    out["compose"] = {
        "Mixture": {
            "weights": w,
            "components": comps,
            "x": xs,
            "logProb": np.log(mix_pdf),
            "cdf": mix_cdf,
            "mean": float(sum(wk * m for wk, (m, _) in zip(w, comps, strict=True))),
        },
        "Independent": {
            "loc": loc,
            "scale": scale,
            "x": xb,
            "logProb": frozen(stats.norm(loc, scale)).logpdf(xb).sum(axis=1),
            "entropy": stats.norm(loc, scale).entropy().sum(),
        },
        "Transformed": {
            "x": np.r_[1e-6, 0.05, 0.5, 1, 2, 20, 1e4],
            "logProb": frozen(stats.lognorm(1.0)).logpdf(np.r_[1e-6, 0.05, 0.5, 1, 2, 20, 1e4]),
            "cdf": stats.lognorm(1.0).cdf(np.r_[1e-6, 0.05, 0.5, 1, 2, 20, 1e4]),
        },
        "Pushforward": {
            "x": np.r_[1e-8, 0.01, 0.5, 1, 3, 10, 40],
            "logProb": frozen(stats.chi2(1)).logpdf(np.r_[1e-8, 0.01, 0.5, 1, 3, 10, 40]),
            "cdf": stats.chi2(1).cdf(np.r_[1e-8, 0.01, 0.5, 1, 3, 10, 40]),
        },
    }

    def kl(p: td.Distribution, q: td.Distribution) -> float:
        return float(td.kl_divergence(p, q))

    t = torch.tensor
    dt = torch.float64
    covq = cov + np.diag([1.0, 0.2, 0.5])
    out["kl"] = {
        "Normal|Normal": {
            "p": [0.3, 1.2],
            "q": [-1.0, 0.7],
            "value": kl(td.Normal(t(0.3, dtype=dt), t(1.2, dtype=dt)), td.Normal(t(-1.0, dtype=dt), t(0.7, dtype=dt))),
        },
        "Bernoulli|Bernoulli": {
            "p": [0.2],
            "q": [0.65],
            "value": kl(td.Bernoulli(t(0.2, dtype=dt)), td.Bernoulli(t(0.65, dtype=dt))),
        },
        "Beta|Beta": {
            "p": [2.0, 3.5],
            "q": [0.8, 1.4],
            "value": kl(td.Beta(t(2.0, dtype=dt), t(3.5, dtype=dt)), td.Beta(t(0.8, dtype=dt), t(1.4, dtype=dt))),
        },
        "Categorical|Categorical": {
            "p": [[0.1, 0.6, 0.3]],
            "q": [[0.3, 0.3, 0.4]],
            "value": kl(td.Categorical(t([0.1, 0.6, 0.3], dtype=dt)), td.Categorical(t([0.3, 0.3, 0.4], dtype=dt))),
        },
        "Dirichlet|Dirichlet": {
            "p": [[0.7, 2.0, 4.5]],
            "q": [[1.0, 1.0, 1.0]],
            "value": kl(td.Dirichlet(t([0.7, 2.0, 4.5], dtype=dt)), td.Dirichlet(t([1.0, 1.0, 1.0], dtype=dt))),
        },
        "Exponential|Exponential": {
            "p": [2.5],
            "q": [0.4],
            "value": kl(td.Exponential(t(2.5, dtype=dt)), td.Exponential(t(0.4, dtype=dt))),
        },
        "Gamma|Gamma": {
            "p": [2.7, 1.8],
            "q": [1.2, 0.5],
            "value": kl(td.Gamma(t(2.7, dtype=dt), t(1.8, dtype=dt)), td.Gamma(t(1.2, dtype=dt), t(0.5, dtype=dt))),
        },
        "LogNormal|LogNormal": {
            "p": [0.3, 0.8],
            "q": [-0.2, 1.5],
            "value": kl(
                td.LogNormal(t(0.3, dtype=dt), t(0.8, dtype=dt)), td.LogNormal(t(-0.2, dtype=dt), t(1.5, dtype=dt))
            ),
        },
        "Poisson|Poisson": {
            "p": [3.7],
            "q": [1.1],
            "value": kl(td.Poisson(t(3.7, dtype=dt)), td.Poisson(t(1.1, dtype=dt))),
        },
        "MultivariateNormal|MultivariateNormal": {
            "p": [mu, cov],
            "q": [np.zeros(3), covq],
            "value": kl(td.MultivariateNormal(t(mu), t(cov)), td.MultivariateNormal(t(np.zeros(3)), t(covq))),
        },
    }
    return out
