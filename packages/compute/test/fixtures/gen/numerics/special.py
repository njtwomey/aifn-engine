"""Golden values for aifn-compute/special.

References are scipy.special / scipy.stats where they are accurate, and mpmath at 50 digits where scipy itself loses
accuracy (far tails, truncated-normal moments). Inputs are exact doubles, so the references are exact to double
rounding.
"""

import mpmath as mp
import numpy as np
from scipy import special as sp
from scipy import stats

mp.mp.dps = 50


def mp_map(f, xs):
    return [float(f(mp.mpf(float(x)))) for x in xs]


def unary() -> dict[str, object]:
    x_erf = np.concatenate([-np.logspace(-12, 0.8, 30), [0.0], np.logspace(-12, 0.8, 30), np.linspace(-6, 6, 49)])
    x_tail = np.concatenate([np.linspace(0, 26, 105), np.logspace(1.5, 6, 20)])
    z_cdf = np.concatenate([-np.linspace(0, 37.5, 76), np.linspace(0, 8.5, 18), -np.logspace(-10, 0, 11)])
    z_logcdf = np.concatenate([-np.logspace(-3, 5, 40), np.linspace(-40, 40, 81), np.array([8.0, 10.0, 20.0, 38.0])])
    p = np.concatenate([np.logspace(-300, -1, 60), np.linspace(0.02, 0.98, 49), 1 - np.logspace(-15, -2, 20)])
    y_inv = np.concatenate([-1 + np.logspace(-15, -1, 15), np.linspace(-0.99, 0.99, 67), np.logspace(-12, -1, 12)])
    z_inv = np.concatenate([np.logspace(-300, -1, 30), np.linspace(0.05, 1.95, 39), 2 - np.logspace(-15, -2, 10)])
    t_v = np.concatenate([-np.logspace(-2, 4, 40), np.linspace(-10, 10, 81), np.array([15.0, 20.0, 30.0, 37.0])])
    x_gamma = np.concatenate([np.logspace(-8, 2.2, 60), -np.linspace(0.05, 20.95, 40) - 0.013, [1.0, 2.0, 3.0, 50.0]])
    x_lg_big = np.logspace(0, 300, 30)
    x_dig = np.concatenate([np.logspace(-8, 6, 60), -np.linspace(0.1, 10.9, 25) - 0.013])
    x_stable = np.concatenate([-np.logspace(-10, 2.9, 30), [0.0], np.logspace(-10, 2.9, 30)])
    p_logit = np.concatenate([np.logspace(-300, -1, 25), np.linspace(0.05, 0.95, 19), 1 - np.logspace(-15, -2, 10)])
    x_neg = -np.logspace(-15, 2.5, 40)
    x_pos = np.logspace(-15, 2.8, 40)
    x_l1pmx = np.concatenate([-1 + np.logspace(-10, -0.1, 12), np.linspace(-0.5, 2, 51), np.logspace(-10, 5, 20)])
    p_ent = np.concatenate(
        [[0.0, 1.0], np.logspace(-300, -1, 20), np.linspace(0.05, 0.95, 19), 1 - np.logspace(-15, -2, 8)]
    )

    x_bessel = np.concatenate([np.linspace(-40, 40, 81), np.logspace(-8, 2.8, 20), -np.logspace(-3, 2.5, 8)])
    x_bessel_pos = np.concatenate([[0.0], np.logspace(-8, 5, 40), np.linspace(25, 35, 11)])
    x_ellip = np.concatenate([np.linspace(-5, 0.99, 41), 1 - np.logspace(-12, -2.5, 8)])

    def mp_v(t):
        return mp.npdf(t) / mp.ncdf(t)

    def mp_w(t):
        v = mp_v(t)
        return v * (v + t)

    return {
        "erf": {"x": x_erf, "y": sp.erf(x_erf)},
        "erfc": {"x": np.concatenate([x_tail, -x_tail[:40]]), "y": sp.erfc(np.concatenate([x_tail, -x_tail[:40]]))},
        "erfcx": {
            "x": np.concatenate([x_tail, -np.linspace(0, 26, 27)]),
            "y": sp.erfcx(np.concatenate([x_tail, -np.linspace(0, 26, 27)])),
        },
        "logErfc": {
            "x": np.concatenate([x_tail, np.array([1e8, 1e100]), -x_tail[:40]]),
            "y": mp_map(lambda v: mp.log(mp.erfc(v)), np.concatenate([x_tail, np.array([1e8, 1e100]), -x_tail[:40]])),
        },
        "normalPdf": {"x": np.linspace(-38, 38, 77), "y": stats.norm.pdf(np.linspace(-38, 38, 77))},
        "normalCdf": {"x": z_cdf, "y": mp_map(mp.ncdf, z_cdf)},
        "normalLogCdf": {
            "x": z_logcdf,
            "y": mp_map(lambda v: mp.log1p(-mp.ncdf(-v)) if v > 0 else mp.log(mp.ncdf(v)), z_logcdf),
        },
        "normalQuantile": {"x": p, "y": mp_map(_mp_ndtri, p)},
        "erfinv": {"x": y_inv, "y": mp_map(mp.erfinv, y_inv)},
        "erfcinv": {"x": z_inv, "y": mp_map(lambda v: -_mp_ndtri(v / 2) / mp.sqrt(2), z_inv)},
        "truncatedNormalV": {"x": t_v, "y": mp_map(mp_v, t_v)},
        "truncatedNormalW": {"x": t_v, "y": mp_map(mp_w, t_v)},
        "logGamma": {"x": np.concatenate([x_gamma, x_lg_big]), "y": sp.gammaln(np.concatenate([x_gamma, x_lg_big]))},
        "gamma": {"x": x_gamma[x_gamma < 171], "y": sp.gamma(x_gamma[x_gamma < 171])},
        "digamma": {"x": x_dig, "y": sp.digamma(x_dig)},
        "trigamma": {"x": x_dig, "y": sp.polygamma(1, x_dig)},
        "softplus": {"x": x_stable, "y": np.logaddexp(0.0, x_stable)},
        "sigmoid": {"x": x_stable, "y": sp.expit(x_stable)},
        "logSigmoid": {"x": x_stable, "y": sp.log_expit(x_stable)},
        "logit": {"x": p_logit, "y": mp_map(lambda v: mp.log(v / (1 - v)), p_logit)},
        "log1mexp": {"x": x_neg, "y": mp_map(lambda v: mp.log1p(-mp.exp(v)), x_neg)},
        "logExpm1": {"x": x_pos, "y": mp_map(lambda v: mp.log(mp.expm1(v)), x_pos)},
        "log1pmx": {"x": x_l1pmx, "y": mp_map(lambda v: mp.log1p(v) - v, x_l1pmx)},
        "binaryEntropy": {
            "x": p_ent,
            "y": mp_map(lambda v: 0 if v in (0, 1) else -v * mp.log(v) - (1 - v) * mp.log1p(-v), p_ent),
        },
        "besselI0": {"x": x_bessel, "y": sp.i0(x_bessel)},
        "besselI1": {"x": x_bessel, "y": sp.i1(x_bessel)},
        "logBesselI0": {"x": x_bessel_pos, "y": mp_map(lambda v: mp.log(mp.besseli(0, v)), x_bessel_pos)},
        "ellipk": {"x": x_ellip, "y": sp.ellipk(x_ellip)},
        "ellipe": {"x": x_ellip, "y": sp.ellipe(x_ellip)},
        "besselRatio": {"x": x_bessel_pos, "y": mp_map(lambda v: mp.besseli(1, v) / mp.besseli(0, v), x_bessel_pos)},
    }


def _mp_ndtri(p):
    """Φ⁻¹(p) at working precision; far tails by Newton on log Φ, where erfinv(2p − 1) would lose p."""
    if p > mp.mpf("1e-30"):
        return mp.sqrt(2) * mp.erfinv(2 * p - 1)
    z0 = -mp.sqrt(-2 * mp.log(p))
    return mp.findroot(lambda z: mp.log(mp.ncdf(z)) - mp.log(p), z0)


def binary() -> dict[str, object]:
    rng = np.random.default_rng(0)
    out: dict[str, object] = {}

    a_g = np.concatenate([np.logspace(-2, 3.5, 12)])
    ag, xg = np.meshgrid(a_g, np.logspace(-3, 3.7, 15))
    ag, xg = ag.ravel(), xg.ravel()
    out["regularisedGammaP"] = {"a": ag, "b": xg, "y": sp.gammainc(ag, xg)}
    out["regularisedGammaQ"] = {"a": ag, "b": xg, "y": sp.gammaincc(ag, xg)}

    def log_gamma_tail(upper: bool):
        # log of the smaller tail directly, log1p of minus it for the larger (so values near 0 keep their digits).
        def f(a, x):
            p = mp.gammainc(a, 0, x, regularized=True)
            q = mp.gammainc(a, x, mp.inf, regularized=True)
            own, other = (q, p) if upper else (p, q)
            return mp.log(own) if own <= other else mp.log1p(-other)

        return f

    out["logRegularisedGammaP"] = {"a": ag, "b": xg, "y": mp_map2(log_gamma_tail(False), ag, xg)}
    out["logRegularisedGammaQ"] = {"a": ag, "b": xg, "y": mp_map2(log_gamma_tail(True), ag, xg)}
    pg = np.concatenate(
        [[1e-300, 1e-100, 1e-30], np.logspace(-12, -1, 5), [0.3, 0.5, 0.7], 1 - np.logspace(-12, -2, 4)]
    )
    ai, pi = np.meshgrid(np.logspace(-1.5, 3.5, 9), pg)
    ai, pi = ai.ravel(), pi.ravel()
    out["regularisedGammaPInverse"] = {"a": ai, "b": pi, "y": sp.gammaincinv(ai, pi)}
    out["regularisedGammaQInverse"] = {"a": ai, "b": pi, "y": sp.gammainccinv(ai, pi)}

    a_b = np.concatenate([np.logspace(-2, 4, 25), rng.uniform(0.1, 20, 15)])
    b_b = np.concatenate([np.logspace(4, -2, 25), rng.uniform(0.1, 1e6, 15)])
    out["logBeta"] = {"a": a_b, "b": b_b, "y": mp_map2(lambda a, b: mp.log(mp.beta(a, b)), a_b, b_b)}

    n = np.array([5, 10, 20, 100, 1000, 1e6, 1e9, 1e9, 1e12, 50, 7.5])
    k = np.array([2, 0, 10, 3, 500, 1, 2, 5e8, 3, 50, 2.5])
    out["logChoose"] = {"a": n, "b": k, "y": mp_map2(lambda a, b: mp.log(mp.binomial(a, b)), n, k)}

    orders = np.repeat([1, 2, 3, 5], 12)
    xs = np.tile(np.logspace(-4, 4, 12), 4)
    out["polygamma"] = {"a": orders, "b": xs, "y": sp.polygamma(orders.astype(int), xs)}

    t = np.concatenate([np.linspace(-40, 40, 33), -np.logspace(1, 6, 6)])
    df = np.array([0.5, 1, 2, 3, 5, 10, 30, 100, 1e4])
    tt, dd = np.meshgrid(t, df)
    tt, dd = tt.ravel(), dd.ravel()
    out["studentTCdf"] = {"a": tt, "b": dd, "y": stats.t.cdf(tt, dd)}

    def t_log_cdf(t, nu):
        # The tail ½ I_x(ν/2, ½), x = ν/(ν + t²); the upper half as log1p of minus the lower tail at −t.
        tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, nu / (nu + t * t), regularized=True) / 2
        return mp.log(tail) if t <= 0 else mp.log1p(-tail)

    tl = np.concatenate([t, -np.logspace(10, 200, 5)])
    tl2, dl = np.meshgrid(tl, df)
    tl2, dl = tl2.ravel(), dl.ravel()
    out["studentTLogCdf"] = {"a": tl2, "b": dl, "y": mp_map2(t_log_cdf, tl2, dl)}
    pq = np.concatenate([np.logspace(-50, -1, 12), np.linspace(0.1, 0.9, 9), 1 - np.logspace(-12, -2, 5)])
    pp, dd2 = np.meshgrid(pq, df)
    pp, dd2 = pp.ravel(), dd2.ravel()
    out["studentTQuantile"] = {"a": pp, "b": dd2, "y": stats.t.ppf(pp, dd2)}

    xc = np.logspace(-3, 3, 15)
    kc = np.array([1, 2, 3, 10, 50, 300])
    xx, kk = np.meshgrid(xc, kc)
    xx, kk = xx.ravel(), kk.ravel()
    out["chiSquareCdf"] = {"a": xx, "b": kk, "y": stats.chi2.cdf(xx, kk)}
    out["chiSquareSf"] = {"a": xx, "b": kk, "y": stats.chi2.sf(xx, kk)}

    la = np.array([-3.0, -40, -1, 0.5, 2, -1e3, -5])
    lb = np.array([-2.0, -39, 1, 3, 40, -999, 5])
    out["normalLogIntervalProbability"] = {
        "a": la,
        "b": lb,
        "y": mp_map2(
            lambda lo, hi: mp.log(mp.ncdf(-lo) - mp.ncdf(-hi) if lo > 0 else mp.ncdf(hi) - mp.ncdf(lo)), la, lb
        ),
    }

    tv = np.concatenate([np.linspace(-12, 12, 25), np.array([-30.0, 30.0])])
    ev = np.array([0.1, 0.5, 1.0, 3.0])
    tt3, ee3 = np.meshgrid(tv, ev)
    tt3, ee3 = tt3.ravel(), ee3.ravel()

    def mass(lo, hi):
        return mp.ncdf(-lo) - mp.ncdf(-hi) if lo > 0 else mp.ncdf(hi) - mp.ncdf(lo)

    def mean(t, e):
        lo, hi = -e - t, e - t
        z = mass(lo, hi)
        return (mp.npdf(lo) - mp.npdf(hi)) / z

    def one_minus_var(t, e):
        lo, hi = -e - t, e - t
        z = mass(lo, hi)
        m = (mp.npdf(lo) - mp.npdf(hi)) / z
        var = 1 + (lo * mp.npdf(lo) - hi * mp.npdf(hi)) / z - m * m
        return 1 - var

    mp.mp.dps = 80
    out["truncatedNormalVDraw"] = {"a": tt3, "b": ee3, "y": mp_map2(mean, tt3, ee3)}
    out["truncatedNormalWDraw"] = {"a": tt3, "b": ee3, "y": mp_map2(one_minus_var, tt3, ee3)}
    mp.mp.dps = 50

    la2 = np.array([0.0, -np.inf, 5.0, -3.0, 700.0, -1.0])
    lb2 = np.array([0.0, 2.0, 5.0 - 1e-10, -np.inf, 699.0, -2.0])
    out["logAddExp"] = {"a": la2, "b": lb2, "y": np.logaddexp(la2, lb2)}
    ld_a = np.array([0.0, 5.0, 2.0, 700.0, 1.0])
    ld_b = np.array([-1.0, 5.0 - 1e-10, -np.inf, 699.0, -40.0])
    out["logDiffExp"] = {"a": ld_a, "b": ld_b, "y": mp_map2(lambda a, b: mp.log(mp.exp(a) - mp.exp(b)), ld_a, ld_b)}

    xa = np.array([0.0, 0.0, 0.0, 1.0, 2.5, -1.5, 3.0, 1e-300, 7.0, 0.5])
    ya = np.array([0.0, 2.0, 1e-300, 0.5, 3.0, 4.0, 1e300, 1e-300, 0.1, 1e-12])
    out["xlogy"] = {"a": xa, "b": ya, "y": sp.xlogy(xa, ya)}
    ya1 = np.array([-1.0, 2.0, 1e-20, 0.5, 3.0, 4.0, 1e300, -0.999, 0.1, 1e-12])
    out["xlog1py"] = {"a": xa, "b": ya1, "y": sp.xlog1py(xa, ya1)}
    return out


def mp_map2(f, xs, ys):
    return [float(f(mp.mpf(float(x)), mp.mpf(float(y)))) for x, y in zip(xs, ys, strict=True)]


def ternary() -> dict[str, object]:
    rng = np.random.default_rng(1)
    a = np.concatenate([rng.uniform(0.05, 5, 30), rng.uniform(5, 500, 20), [0.5, 0.5, 1.0, 2.0, 1e4]])
    b = np.concatenate([rng.uniform(0.05, 5, 30), rng.uniform(5, 500, 20), [0.5, 30.0, 1.0, 3.0, 2e4]])
    x = np.concatenate([rng.uniform(0, 1, 50), [1e-10, 0.999, 0.5, 0.2, 0.3333]])
    p = np.concatenate([rng.uniform(0, 1, 40), np.logspace(-30, -2, 10), [1e-10, 0.999, 0.5, 0.2, 0.3333]])
    return {
        "regularisedBeta": {"a": a, "b": b, "c": x, "y": sp.betainc(a, b, x)},
        "regularisedBetaInverse": {"a": a, "b": b, "c": p, "y": sp.betaincinv(a, b, p)},
        "logRegularisedBeta": {
            "a": a,
            "b": b,
            "c": x,
            "y": [float(log_beta_tail(*v)) for v in zip(a, b, x, strict=True)],
        },
    }


def log_beta_tail(a: float, b: float, x: float):
    """log I_x(a, b) at 50 digits: the log of I directly where I ≤ ½, log1p of minus its complement above."""
    a_, b_, x_ = mp.mpf(a), mp.mpf(b), mp.mpf(x)
    try:
        lower = mp.betainc(a_, b_, 0, x_, regularized=True)
        return mp.log(lower) if lower <= 0.5 else mp.log1p(-mp.betainc(b_, a_, 0, 1 - x_, regularized=True))
    except ValueError:
        # mpmath's series does not converge for a, b ~ 1e4 near the mean; there I is moderate and scipy is accurate.
        lower = sp.betainc(a, b, x)
        return np.log(lower) if lower <= 0.5 else np.log1p(-sp.betainc(b, a, 1 - x))


def vector() -> dict[str, object]:
    x = np.array([1.0, -2.0, 3.5, 700.0, 699.0, -np.inf])
    y = np.array([-1000.0, -1001.0, -999.5])
    return {
        "logSumExp": [{"x": x, "y": sp.logsumexp(x)}, {"x": y, "y": sp.logsumexp(y)}],
        "softmax": [
            {"x": x, "t": 1.0, "y": sp.softmax(x)},
            {"x": y, "t": 0.5, "y": sp.softmax(y / 0.5)},
            {"x": y, "t": 20.0, "y": sp.softmax(y / 20.0)},
        ],
        "logSoftmax": [{"x": y, "t": 1.0, "y": sp.log_softmax(y)}, {"x": y, "t": 3.0, "y": sp.log_softmax(y / 3.0)}],
    }


def cases() -> dict[str, object]:
    return {"unary": unary(), "binary": binary(), "ternary": ternary(), "vector": vector(), "softplus": _softplus()}


def _softplus() -> dict[str, object]:
    x = np.array([-50.0, -5.0, -1.0, 0.0, 1.0, 5.0, 50.0])
    return {"x": x, "y": np.logaddexp(0.0, x)}
