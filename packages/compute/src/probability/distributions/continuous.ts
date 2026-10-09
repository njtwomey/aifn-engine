/**
 * Continuous univariate families. Every density is a composition of `aifn-compute/foundation/tensor` and
 * `aifn-compute/numerics/special` primitives, so `logProb` is differentiable in the value and in every parameter; cdfs
 * and quantiles are differentiable wherever the special function they use is (the incomplete gamma and beta functions
 * only in their continuous argument). Numerically inverted quantiles and quadrature cdfs are not differentiable and say
 * so when given traced values.
 *
 * Parameterisations follow scipy.stats (`loc`, `scale`) wherever there is a scale: `Normal(loc, scale)` is scipy's
 * `norm(loc, scale)`, with the standard deviation as scale. Each constructor checks its parameters and throws a
 * `DomainError` naming the family and the parameter when one is out of range; on an interval support, values outside
 * it have log-density $-\infty$. Every family is built by `univariate`, so each has the full `Univariate` interface.
 */

import {
  beta as betaDraws,
  chiSquare as chiSquareDraws,
  gammaVariate as gammaDraws,
  studentT as studentTDraws,
} from 'aifn-compute/probability/samplers'
import { drawEach, units, type Stream } from 'aifn-compute/foundation/random'
import {
  chiSquareCdf,
  chiSquareSf,
  digamma,
  gamma as gammaFunction,
  logBeta,
  logGamma,
  logRegularisedBeta,
  logRegularisedGammaP,
  logRegularisedGammaQ,
  logSigmoid,
  logit,
  log1mexp,
  normalCdf,
  normalLogCdf,
  normalLogIntervalProbability,
  normalLogPdf,
  normalQuantile,
  regularisedBeta,
  regularisedBetaInverse,
  regularisedGammaP,
  regularisedGammaPInverse,
  regularisedGammaQ,
  regularisedGammaQInverse,
  sigmoid,
  softplus,
  studentTCdf,
  studentTLogCdf,
  studentTQuantile,
} from 'aifn-compute/numerics/special'
import {
  abs,
  add,
  cos,
  div,
  exp,
  expm1,
  fromData,
  log,
  log1p,
  maximum,
  minimum,
  mul,
  neg,
  pow,
  sqrt,
  square,
  sub,
  unwrap,
  where,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { besselRatio, logBesselI0 } from 'aifn-compute/numerics/special'
import type { Univariate } from './types'
import {
  atBatch,
  check,
  EULER_GAMMA,
  guard,
  inverseTransform,
  locationScale,
  logFromTails,
  LOG_2PI,
  mask,
  outside,
  raw,
  rawMap,
  rawOnly,
  standardCauchyCdf,
  standardCauchyQuantile,
  univariate,
} from './util'
import { xlog1py, xlogy } from 'aifn-compute/numerics/special'

/**
 * The test of a positive parameter, for `check`.
 *
 * @param x One element of the parameter's raw value.
 * @returns True when $x > 0$ (false for NaN).
 */
const positive = (x: number) => x > 0
/**
 * The test of a finite parameter, for `check`.
 *
 * @param x One element of the parameter's raw value.
 * @returns True unless $x$ is NaN or $\pm\infty$.
 */
const finite = (x: number) => Number.isFinite(x)

/**
 * NaN with the batch shape of `params`: the value of a moment that is undefined for every member of the family.
 *
 * @param params The distribution's parameters (numbers, tensors or traced values); only their broadcast shape is used.
 * @returns NaN as a number when every parameter is a number, else a tensor of NaN of the parameters' broadcast shape.
 */
function nanAt(...params: Value[]): Value {
  return rawMap(params, () => NaN)
}

/**
 * Draws as a tensor (the samplers return a number when every parameter is a number and no shape is given).
 *
 * @param x A sampler's output: a number or a tensor.
 * @returns `x` itself when it is a tensor, else a rank-0 tensor holding it.
 */
const drawn = (x: number | Tensor): Tensor => (typeof x === 'number' ? fromData(new Float64Array([x]), []) : x)

/**
 * A raw draw of a composition (numbers or tensors, never traced) as a tensor of the draw's shape.
 *
 * @param x The draw, a number or a tensor; it is unwrapped first.
 * @param shape The draw's full shape (sample shape, then batch shape). Used only when `x` is a number, which then fills
 *   a tensor of this shape; a tensor `x` is returned as it is.
 * @returns A tensor of the draw.
 */
const drawnAt = (x: Value, shape: number[]): Tensor => {
  const r = unwrap(x)
  return typeof r === 'number' ? fromData(new Float64Array(shape.reduce((a, b) => a * b, 1)).fill(r), shape) : r
}

// ── Normal ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The normal distribution $\Gauss(\mu, \sigma^2)$ with mean $\mu$ and standard deviation $\sigma > 0$ (scipy's `norm`),
 * with density
 * $p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\LP -\frac{(x - \mu)^2}{2\sigma^2} \RP$.
 * The tails of the cdf, log cdf and survival function keep full relative accuracy: they use $\Phi$ and $\log\Phi$,
 * never $1 - \Phi$. `rsample` is the pathwise draw $\mu + \sigma z$ with $z \sim \Gauss(0, 1)$. An exponential family
 * with $\etavec = (\mu/\sigma^2, -1/(2\sigma^2))$ and $T(x) = (x, x^2)$.
 *
 * @param loc The mean $\mu$.
 * @param scale The standard deviation $\sigma$ (not the variance); every element must be positive, or a `DomainError`
 *   is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `loc` and `scale`.
 *
 * @example The density at the mean, and the moments
 * const d = Normal(1, 2)
 * print('p(1) =', d.prob(1), ' 1 / (2 sqrt(2 pi)) =', 1 / (2 * Math.sqrt(2 * Math.PI)))
 * print('mean =', d.mean(), ' variance =', d.variance())
 *
 * @example The cdf and quantile round trip, and a far tail
 * const d = Normal(0, 1)
 * print('cdf(1.96) =', d.cdf(1.96))
 * print('quantile(0.975) =', d.quantile(0.975))
 * print('log survival(40) =', d.logSurvival(40))
 *
 * @example A seeded sample and its moments
 * const x = Normal(3, 0.5).sample(stream(0), { shape: [10000] })
 * print('sample mean =', mean(x), ' sample variance =', variance(x))
 *
 * @example A batch of two, and a gradient in the mean
 * print('log p(0) =', Normal(tensor([0, 1]), 1).logProb(0))
 * print('d/dmu log p(2) at mu = 0.5:', grad((mu) => Normal(mu, 1).logProb(2))(0.5))
 */
export function Normal<M extends Value, S extends Value>(loc: M, scale: S): Univariate<M | S> {
  check('Normal', 'scale', scale, positive, 'positive')
  const z = (x: Value) => div(sub(x, loc), scale)
  return univariate({
    name: 'Normal',
    params: { loc, scale },
    support: { type: 'real' },
    logProb: (x) => sub(normalLogPdf(z(x)), log(scale)),
    cdf: (x) => normalCdf(z(x)),
    logcdf: (x) => normalLogCdf(z(x)),
    survival: (x) => normalCdf(neg(z(x))),
    logSurvival: (x) => normalLogCdf(neg(z(x))),
    quantile: (p) => add(loc, mul(scale, normalQuantile(p))),
    isf: (q) => sub(loc, mul(scale, normalQuantile(q))),
    rsample: (s, shape, scalar) => locationScale(s, shape, scalar, loc, scale),
    mean: () => atBatch(loc, scale),
    variance: () => atBatch(square(scale), loc),
    entropy: () => atBatch(add(0.5 * (1 + LOG_2PI), log(scale)), loc),
    mode: () => atBatch(loc, scale),
    expFamily: {
      naturalParams: () => [div(loc, square(scale)), div(-0.5, square(scale))],
      sufficientStats: (x) => [x, square(x)],
      logPartition: () => add(div(square(loc), mul(2, square(scale))), log(scale)),
      logBaseMeasure: () => -0.5 * LOG_2PI,
    },
  })
}

/**
 * The normal distribution with natural parameters $\eta_1 = \mu/\sigma^2$ and $\eta_2 = -1/(2\sigma^2) < 0$ (as used by
 * EP messages), with density $p(x) \propto \exp(\eta_1 x + \eta_2 x^2)$. It is `Normal` with $\sigma^2 = -1/(2\eta_2)$
 * and $\mu = \eta_1\sigma^2$, so its `params` are the mean and standard deviation.
 *
 * @param eta1 The first natural parameter $\eta_1 = \mu/\sigma^2$, the precision-weighted mean.
 * @param eta2 The second natural parameter $\eta_2 = -1/(2\sigma^2)$, minus half the precision; every element must be
 *   negative, or a `DomainError` is thrown.
 * @returns The normal distribution, with batch shape the broadcast shape of `eta1` and `eta2`.
 *
 * @example Natural parameters to mean and variance
 * const d = normalFromNatural(2, -0.5)
 * print('mean =', d.mean(), ' variance =', d.variance())
 *
 * @example The round trip through Normal's natural parameters
 * const [eta1, eta2] = Normal(3, 2).expFamily.naturalParams()
 * print('eta1 =', eta1, ' eta2 =', eta2)
 * print('back:', normalFromNatural(eta1, eta2).params)
 */
export function normalFromNatural<A extends Value, B extends Value>(eta1: A, eta2: B): Univariate<A | B> {
  check('normalFromNatural', 'eta2', eta2, (x) => x < 0, 'negative')
  const variance = div(-0.5, eta2)
  return Normal(mul(eta1, variance), sqrt(variance)) as unknown as Univariate<A | B>
}

// ── Log-normal ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The log-normal distribution: $\log X \sim \Gauss(\mu, \sigma^2)$ with $\sigma > 0$ (scipy's
 * `lognorm(s=sigma, scale=exp(mu))`), with density
 * $p(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\LP -\frac{(\log x - \mu)^2}{2\sigma^2} \RP$ for $x > 0$.
 * Mean $e^{\mu + \sigma^2/2}$, variance $(e^{\sigma^2} - 1)e^{2\mu + \sigma^2}$ and mode $e^{\mu - \sigma^2}$.
 * `rsample` is the pathwise draw $e^{\mu + \sigma z}$. An exponential family with
 * $\etavec = (\mu/\sigma^2, -1/(2\sigma^2))$ and $T(x) = (\log x, (\log x)^2)$.
 *
 * @param mu The mean $\mu$ of $\log X$ (not of $X$).
 * @param sigma The standard deviation $\sigma$ of $\log X$; every element must be positive, or a `DomainError` is
 *   thrown.
 * @returns The distribution, with batch shape the broadcast shape of `mu` and `sigma`.
 *
 * @example The density, the median and the mean
 * const d = LogNormal(0, 1)
 * print('p(1) =', d.prob(1), ' 1 / sqrt(2 pi) =', 1 / Math.sqrt(2 * Math.PI))
 * print('median =', d.quantile(0.5))
 * print('mean =', d.mean(), ' exp(1/2) =', Math.exp(0.5))
 *
 * @example The logarithm of a seeded sample is normal
 * const x = LogNormal(1, 0.5).sample(stream(1), { shape: [10000] })
 * print('mean of log x =', mean(log(x)), ' variance of log x =', variance(log(x)))
 */
export function LogNormal<M extends Value, S extends Value>(mu: M, sigma: S): Univariate<M | S> {
  check('LogNormal', 'sigma', sigma, positive, 'positive')
  const valid = (x: Value) => mask([x], (v) => v > 0)
  const z = (x: Value, ok: ReturnType<typeof valid>) => div(sub(log(guard(x, ok, 1)), mu), sigma)
  return univariate({
    name: 'LogNormal',
    params: { mu, sigma },
    support: { type: 'interval', lower: 0, upper: Infinity },
    logProb: (x) => {
      const ok = valid(x)
      const xs = guard(x, ok, 1)
      return outside(ok, sub(sub(normalLogPdf(z(x, ok)), log(sigma)), log(xs)), -Infinity)
    },
    cdf: (x) => {
      const ok = valid(x)
      return outside(ok, normalCdf(z(x, ok)), 0)
    },
    logcdf: (x) => {
      const ok = valid(x)
      return outside(ok, normalLogCdf(z(x, ok)), -Infinity)
    },
    survival: (x) => {
      const ok = valid(x)
      return outside(ok, normalCdf(neg(z(x, ok))), 1)
    },
    logSurvival: (x) => {
      const ok = valid(x)
      return outside(ok, normalLogCdf(neg(z(x, ok))), 0)
    },
    quantile: (p) => exp(add(mu, mul(sigma, normalQuantile(p)))),
    isf: (q) => exp(sub(mu, mul(sigma, normalQuantile(q)))),
    rsample: (s, shape, scalar) => exp(locationScale(s, shape, scalar, mu, sigma)),
    mean: () => exp(add(mu, mul(0.5, square(sigma)))),
    variance: () => mul(expm1(square(sigma)), exp(add(mul(2, mu), square(sigma)))),
    entropy: () => add(add(mu, 0.5 * (1 + LOG_2PI)), log(sigma)),
    mode: () => exp(sub(mu, square(sigma))),
    expFamily: {
      naturalParams: () => [div(mu, square(sigma)), div(-0.5, square(sigma))],
      sufficientStats: (x) => [log(x), square(log(x))],
      logPartition: () => add(div(square(mu), mul(2, square(sigma))), log(sigma)),
      logBaseMeasure: (x) => sub(-0.5 * LOG_2PI, log(x)),
    },
  })
}

// ── Student t ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Student's $t$ distribution with $\nu > 0$ degrees of freedom, location $\mu$ and scale $\sigma > 0$ (scipy's
 * `t(df, loc, scale)`), with density
 * $p(x) = \frac{\Gamma((\nu + 1)/2)}{\Gamma(\nu/2)\sqrt{\nu\pi}\,\sigma} \LP 1 + \frac{z^2}{\nu} \RP^{-(\nu + 1)/2}$
 * with $z = (x - \mu)/\sigma$.
 * The mean $\mu$ is NaN for $\nu \le 1$; the variance $\sigma^2\nu/(\nu - 2)$ is $\infty$ for $1 < \nu \le 2$ and NaN
 * for $\nu \le 1$. Differentiable in $x$, $\mu$ and $\sigma$; the cdf and quantile are not differentiable in $\nu$. No
 * `rsample`: the draw divides by a chi-square draw (see `Gamma`).
 *
 * @param df The degrees of freedom $\nu$; every element must be positive, or a `DomainError` is thrown.
 * @param loc The location $\mu$, the median (and the mean when $\nu > 1$).
 * @param scale The scale $\sigma$, not the standard deviation; every element must be positive, or a `DomainError` is
 *   thrown.
 * @returns The distribution, with batch shape the broadcast shape of the three parameters.
 *
 * @example The density at 0 and the moments
 * const d = StudentT(3)
 * print('p(0) =', d.prob(0), ' 2 / (pi sqrt 3) =', 2 / (Math.PI * Math.sqrt(3)))
 * print('mean =', d.mean(), ' variance =', d.variance())
 *
 * @example Moments that do not exist
 * print('df = 2: mean', StudentT(2).mean(), ' variance', StudentT(2).variance())
 * print('df = 1: mean', StudentT(1).mean(), ' variance', StudentT(1).variance())
 *
 * @example The cdf and quantile round trip
 * const d = StudentT(5, 1, 2)
 * const x = d.quantile(0.9)
 * print('quantile(0.9) =', x, ' cdf of it =', d.cdf(x))
 */
export function StudentT<D extends Value, M extends Value = number, S extends Value = number>(
  df: D,
  loc: M = 0 as M,
  scale: S = 1 as S,
): Univariate<D | M | S> {
  check('StudentT', 'df', df, positive, 'positive')
  check('StudentT', 'scale', scale, positive, 'positive')
  const z = (x: Value) => div(sub(x, loc), scale)
  const half = mul(0.5, df)
  const halfPlus = mul(0.5, add(df, 1))
  return univariate({
    name: 'StudentT',
    params: { df, loc, scale },
    support: { type: 'real' },
    logProb: (x) =>
      sub(
        sub(sub(logGamma(halfPlus), logGamma(half)), add(mul(0.5, log(mul(Math.PI, df))), log(scale))),
        mul(halfPlus, log1p(div(square(z(x)), df))),
      ),
    cdf: (x) => studentTCdf(z(x), df),
    logcdf: (x) => studentTLogCdf(z(x), df),
    survival: (x) => studentTCdf(neg(z(x)), df),
    logSurvival: (x) => studentTLogCdf(neg(z(x)), df),
    quantile: (p) => add(loc, mul(scale, studentTQuantile(p, df))),
    isf: (q) => sub(loc, mul(scale, studentTQuantile(q, df))),
    sample: (s, shape) =>
      drawn(studentTDraws(s, raw(df, 'StudentT'), raw(loc, 'StudentT'), raw(scale, 'StudentT'), { shape })),
    mean: () =>
      where(
        mask([df, loc, scale], (v) => v > 1),
        atBatch(loc, df, scale),
        NaN,
      ),
    variance: () => {
      const v = div(mul(square(scale), df), sub(df, 2))
      const regime = rawMap([df, loc, scale], (n) => (n > 2 ? 0 : n > 1 ? Infinity : NaN))
      return where(
        mask([regime], (r) => r === 0),
        atBatch(v, loc),
        regime,
      )
    },
    entropy: () =>
      add(
        add(mul(halfPlus, sub(digamma(halfPlus), digamma(half))), add(mul(0.5, log(df)), logBeta(half, 0.5))),
        atBatch(log(scale), loc),
      ),
    mode: () => atBatch(loc, df, scale),
  })
}

// ── Cauchy ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Cauchy distribution with location $\mu$ and scale $\sigma > 0$ (scipy's `cauchy`), with density
 * $p(x) = \frac{1}{\pi\sigma(1 + z^2)}$ with $z = (x - \mu)/\sigma$.
 * Its mean and variance are NaN (undefined); its median and mode are $\mu$ and its quartiles $\mu \pm \sigma$. The
 * quantile keeps its relative accuracy in both tails, and `rsample` is the pathwise draw through it.
 *
 * @param loc The location $\mu$, the median.
 * @param scale The scale $\sigma$, the half-width of the interquartile range; every element must be positive, or a
 *   `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `loc` and `scale`.
 *
 * @example The density at the median, the quartiles, and no moments
 * const d = Cauchy(1, 2)
 * print('p(1) =', d.prob(1), ' 1 / (2 pi) =', 1 / (2 * Math.PI))
 * print('quartiles:', d.quantile(0.25), d.quantile(0.75))
 * print('mean =', d.mean(), ' variance =', d.variance())
 */
export function Cauchy<M extends Value, S extends Value>(loc: M, scale: S): Univariate<M | S> {
  check('Cauchy', 'scale', scale, positive, 'positive')
  const z = (x: Value) => div(sub(x, loc), scale)
  const quantile = (p: Value) => add(loc, mul(scale, standardCauchyQuantile(p)))
  return univariate({
    name: 'Cauchy',
    params: { loc, scale },
    support: { type: 'real' },
    logProb: (x) => neg(add(add(Math.log(Math.PI), log(scale)), log1p(square(z(x))))),
    cdf: (x) => standardCauchyCdf(z(x)),
    survival: (x) => standardCauchyCdf(neg(z(x))),
    quantile,
    isf: (q) => sub(loc, mul(scale, standardCauchyQuantile(q))),
    rsample: (s, shape, scalar) => inverseTransform(s, shape, scalar, quantile),
    mean: () => nanAt(loc, scale),
    variance: () => nanAt(loc, scale),
    entropy: () => atBatch(add(Math.log(4 * Math.PI), log(scale)), loc),
    mode: () => atBatch(loc, scale),
  })
}

// ── Laplace ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Laplace (double exponential) distribution with location $\mu$ and scale $b > 0$ (scipy's `laplace`), with
 * density
 * $p(x) = \frac{1}{2b} \exp\LP -\frac{\abs{x - \mu}}{b} \RP$.
 * Mean, median and mode $\mu$, variance $2b^2$. Both tails of the cdf, the survival function and the quantile keep
 * their relative accuracy, and `rsample` is the pathwise draw through the quantile.
 *
 * @param loc The location $\mu$.
 * @param scale The scale $b$, the mean absolute deviation from $\mu$; every element must be positive, or a
 *   `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `loc` and `scale`.
 *
 * @example The density at the mode and the moments
 * const d = Laplace(0, 2)
 * print('p(0) =', d.prob(0), ' 1 / (2b) =', 1 / 4)
 * print('mean =', d.mean(), ' variance =', d.variance())
 *
 * @example A far-tail quantile, and the cdf back
 * const d = Laplace(0, 1)
 * const x = d.quantile(1e-300)
 * print('quantile(1e-300) =', x, ' log(2e-300) =', Math.log(2e-300))
 * print('cdf of it =', d.cdf(x))
 */
export function Laplace<M extends Value, S extends Value>(loc: M, scale: S): Univariate<M | S> {
  check('Laplace', 'scale', scale, positive, 'positive')
  const z = (x: Value) => div(sub(x, loc), scale)
  const below = (x: Value) => mask([x, loc], (v, m) => v < m)
  // The standard quantile: log 2p below ½ and −log 2(1 − p) above (1 − p is exact there), so that neither tail
  // passes through 1 − 2|p − ½|, which rounds to 0 for tiny p.
  const standard = (p: Value) => {
    const low = mask([p], (v) => v < 0.5)
    const high = mask([p], (v) => !(v < 0.5))
    return where(low, log(mul(2, guard(p, low, 0.25))), neg(log(mul(2, sub(1, guard(p, high, 0.75))))))
  }
  const quantile = (p: Value) => add(loc, mul(scale, standard(p)))
  return univariate({
    name: 'Laplace',
    params: { loc, scale },
    support: { type: 'real' },
    logProb: (x) => neg(add(log(mul(2, scale)), abs(z(x)))),
    // e^{−|z|} is finite in both branches, so the unused branch never produces ∞ · 0 in a derivative.
    cdf: (x) => {
      const t = mul(0.5, exp(neg(abs(z(x)))))
      return where(below(x), t, sub(1, t))
    },
    logcdf: (x) => {
      const a = abs(z(x))
      return where(below(x), sub(-Math.LN2, a), log1p(mul(-0.5, exp(neg(a)))))
    },
    survival: (x) => {
      const t = mul(0.5, exp(neg(abs(z(x)))))
      return where(below(x), sub(1, t), t)
    },
    logSurvival: (x) => {
      const a = abs(z(x))
      return where(below(x), log1p(mul(-0.5, exp(neg(a)))), sub(-Math.LN2, a))
    },
    quantile,
    isf: (q) => sub(loc, mul(scale, standard(q))),
    rsample: (s, shape, scalar) => inverseTransform(s, shape, scalar, quantile),
    mean: () => atBatch(loc, scale),
    variance: () => atBatch(mul(2, square(scale)), loc),
    entropy: () => atBatch(add(1, log(mul(2, scale))), loc),
    mode: () => atBatch(loc, scale),
  })
}

// ── Logistic ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The logistic distribution with location $\mu$ and scale $s > 0$ (scipy's `logistic`), with density
 * $p(x) = \frac{e^{-z}}{s(1 + e^{-z})^2}$ with $z = (x - \mu)/s$.
 * Its cdf is the sigmoid $\sigma(z)$ and its quantile $\mu + s \operatorname{logit} p$. Mean, median and mode $\mu$,
 * variance $\pi^2 s^2/3$. `rsample` is the pathwise draw through the quantile.
 *
 * @param loc The location $\mu$.
 * @param scale The scale $s$, not the standard deviation; every element must be positive, or a `DomainError` is
 *   thrown.
 * @returns The distribution, with batch shape the broadcast shape of `loc` and `scale`.
 *
 * @example The density at the mode, the cdf and the variance
 * const d = Logistic(0, 1)
 * print('p(0) =', d.prob(0), ' cdf(0) =', d.cdf(0))
 * print('cdf(2) =', d.cdf(2), ' 1 / (1 + e^-2) =', 1 / (1 + Math.exp(-2)))
 * print('variance =', d.variance(), ' pi^2 / 3 =', Math.PI ** 2 / 3)
 */
export function Logistic<M extends Value, S extends Value>(loc: M, scale: S): Univariate<M | S> {
  check('Logistic', 'scale', scale, positive, 'positive')
  const z = (x: Value) => div(sub(x, loc), scale)
  const quantile = (p: Value) => add(loc, mul(scale, logit(p)))
  return univariate({
    name: 'Logistic',
    params: { loc, scale },
    support: { type: 'real' },
    logProb: (x) => {
      const t = z(x)
      return neg(add(add(t, log(scale)), mul(2, softplus(neg(t)))))
    },
    cdf: (x) => sigmoid(z(x)),
    logcdf: (x) => logSigmoid(z(x)),
    survival: (x) => sigmoid(neg(z(x))),
    logSurvival: (x) => logSigmoid(neg(z(x))),
    quantile,
    isf: (q) => sub(loc, mul(scale, logit(q))),
    rsample: (s, shape, scalar) => inverseTransform(s, shape, scalar, quantile),
    mean: () => atBatch(loc, scale),
    variance: () => atBatch(mul((Math.PI * Math.PI) / 3, square(scale)), loc),
    entropy: () => atBatch(add(log(scale), 2), loc),
    mode: () => atBatch(loc, scale),
  })
}

// ── Uniform ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The continuous uniform distribution on $[a, b]$ with $a < b$ (scipy's `uniform(low, high - low)`), with density
 * $p(x) = \frac{1}{b - a}$ for $a \le x \le b$,
 * and 0 elsewhere. Mean $(a + b)/2$, variance $(b - a)^2/12$; the mode is NaN (every point of $[a, b]$ is one).
 * `rsample` is the pathwise draw $a + (b - a)u$ with $u$ uniform on $[0, 1)$.
 *
 * @param low The lower end $a$.
 * @param high The upper end $b$; every element of `high - low` must be positive, or a `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `low` and `high`.
 *
 * @example Density, moments, and the cdf and quantile
 * const d = Uniform(2, 6)
 * print('p(3) =', d.prob(3), ' p(7) =', d.prob(7))
 * print('mean =', d.mean(), ' variance =', d.variance())
 * print('cdf(5) =', d.cdf(5), ' quantile(0.75) =', d.quantile(0.75))
 *
 * @example A seeded sample and its moments
 * const x = Uniform(0, 1).sample(stream(2), { shape: [10000] })
 * print('sample mean =', mean(x), ' sample variance =', variance(x), ' 1/12 =', 1 / 12)
 */
export function Uniform<A extends Value, B extends Value>(low: A, high: B): Univariate<A | B> {
  check('Uniform', 'high − low', sub(unwrap(high), unwrap(low)), positive, 'positive')
  const width = sub(high, low)
  const inside = (x: Value) => mask([x, low, high], (v, a, b) => v >= a && v <= b)
  return univariate({
    name: 'Uniform',
    params: { low, high },
    support: { type: 'interval', lower: low, upper: high },
    logProb: (x) => outside(inside(x), atBatch(neg(log(width)), x), -Infinity),
    cdf: (x) => minimum(maximum(div(sub(x, low), width), 0), 1),
    survival: (x) => minimum(maximum(div(sub(high, x), width), 0), 1),
    quantile: (p) => add(low, mul(p, width)),
    isf: (q) => sub(high, mul(q, width)),
    rsample: (s, shape, scalar) => inverseTransform(s, shape, scalar, (u) => add(low, mul(u, width))),
    mean: () => mul(0.5, add(low, high)),
    variance: () => div(square(width), 12),
    entropy: () => log(width),
    mode: () => nanAt(low, high),
  })
}

// ── Exponential ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The exponential distribution with rate $\lambda > 0$ (scipy's `expon(scale=1/rate)`), with density
 * $p(x) = \lambda e^{-\lambda x}$ for $x \ge 0$.
 * Mean $1/\lambda$, variance $1/\lambda^2$, mode 0. The cdf $1 - e^{-\lambda x}$ is computed with
 * $\operatorname{expm1}$, so it keeps its relative accuracy near 0. `rsample` is the pathwise draw
 * $-\log(1 - u)/\lambda$. An exponential family with $\eta = -\lambda$ and $T(x) = x$.
 *
 * @param rate The rate $\lambda$, the reciprocal of the mean; every element must be positive, or a `DomainError` is
 *   thrown.
 * @returns The distribution, with batch shape the shape of `rate`.
 *
 * @example The density at 0, the median and the mean
 * const d = Exponential(2)
 * print('p(0) =', d.prob(0), ' p(-1) =', d.prob(-1))
 * print('median =', d.quantile(0.5), ' log(2) / 2 =', Math.LN2 / 2)
 * print('mean =', d.mean(), ' variance =', d.variance())
 *
 * @example A seeded sample and its moments
 * const x = Exponential(0.5).sample(stream(3), { shape: [10000] })
 * print('sample mean =', mean(x), ' sample variance =', variance(x))
 */
export function Exponential<R extends Value>(rate: R): Univariate<R> {
  check('Exponential', 'rate', rate, positive, 'positive')
  const valid = (x: Value) => mask([x], (v) => v >= 0)
  // λ · max(x, 0): the clamp only keeps the unused branch finite; outside the support the result is replaced.
  const t = (x: Value) => mul(rate, maximum(x, 0))
  return univariate({
    name: 'Exponential',
    params: { rate },
    support: { type: 'interval', lower: 0, upper: Infinity },
    logProb: (x) => outside(valid(x), sub(log(rate), t(x)), -Infinity),
    cdf: (x) => neg(expm1(neg(t(x)))),
    logcdf: (x) => log1mexp(neg(t(x))),
    survival: (x) => exp(neg(t(x))),
    logSurvival: (x) => neg(t(x)),
    quantile: (p) => div(neg(log1p(neg(p))), rate),
    isf: (q) => div(neg(log(q)), rate),
    rsample: (s, shape, scalar) => inverseTransform(s, shape, scalar, (u) => div(neg(log1p(neg(u))), rate)),
    mean: () => div(1, rate),
    variance: () => div(1, square(rate)),
    entropy: () => sub(1, log(rate)),
    mode: () => mul(0, rate),
    expFamily: {
      naturalParams: () => [neg(rate)],
      sufficientStats: (x) => [x],
      logPartition: () => neg(log(rate)),
      logBaseMeasure: () => 0,
    },
  })
}

// ── Gamma family ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The specification of the gamma distribution with shape $\alpha$ and rate $\beta$, shared by `Gamma` and `ChiSquare`
 * (which replaces its name, parameters, cdf, survival function and sampler). The parameters are not checked here.
 *
 * @param shape The shape $\alpha$, positive.
 * @param rate The rate $\beta$, positive (the reciprocal of the scale).
 * @returns The `UnivariateSpec` of $\GammaD(\alpha, \beta)$, for `univariate`.
 */
function gammaSpec(shape: Value, rate: Value) {
  const valid = (x: Value) => mask([x], (v) => v >= 0)
  const t = (x: Value) => mul(rate, maximum(x, 0))
  return {
    name: 'Gamma',
    params: { shape, rate } as Record<string, Value>,
    support: { type: 'interval' as const, lower: 0, upper: Infinity },
    logProb: (x: Value) => {
      const ok = valid(x)
      const xs = guard(x, ok, 1)
      return outside(
        ok,
        sub(add(sub(mul(shape, log(rate)), logGamma(shape)), xlogy(sub(shape, 1), xs)), mul(rate, xs)),
        -Infinity,
      )
    },
    cdf: (x: Value) => regularisedGammaP(shape, t(x)),
    logcdf: (x: Value) => logRegularisedGammaP(shape, t(x)),
    survival: (x: Value) => regularisedGammaQ(shape, t(x)),
    logSurvival: (x: Value) => logRegularisedGammaQ(shape, t(x)),
    // Above p = ½ the inverse solves Q = 1 − p, so upper quantiles keep their relative accuracy.
    quantile: (p: Value) => div(regularisedGammaPInverse(shape, p), rate),
    isf: (q: Value) => div(regularisedGammaQInverse(shape, q), rate),
    sample: (s: Stream, drawShape: number[]) =>
      drawn(
        gammaDraws(s, raw(shape, 'Gamma'), unwrap(div(1, raw(rate, 'Gamma'))) as number | Tensor, { shape: drawShape }),
      ),
    mean: () => div(shape, rate),
    variance: () => div(shape, square(rate)),
    entropy: () => add(sub(shape, log(rate)), add(logGamma(shape), mul(sub(1, shape), digamma(shape)))),
    mode: () =>
      where(
        mask([shape, rate], (a) => a >= 1),
        div(sub(shape, 1), rate),
        0,
      ),
    expFamily: {
      naturalParams: () => [sub(shape, 1), neg(rate)],
      sufficientStats: (x: Value) => [log(x), x],
      logPartition: () => sub(logGamma(shape), mul(shape, log(rate))),
      logBaseMeasure: () => 0,
    },
  }
}

/**
 * The gamma distribution with shape $\alpha > 0$ and rate $\beta > 0$ (scipy's `gamma(a, scale=1/rate)`), with
 * density $p(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}$ for $x \ge 0$.
 * Mean $\alpha/\beta$, variance $\alpha/\beta^2$ and mode $(\alpha - 1)/\beta$ for $\alpha \ge 1$; for $\alpha < 1$
 * the density is infinite at 0 and the mode is 0. The cdf is the regularised incomplete gamma function
 * $P(\alpha, \beta x)$, and the quantile inverts it numerically (inverting $Q$ above $p = 1/2$): both are
 * differentiable in $x$ (or $p$) and $\beta$, not in $\alpha$. An exponential family with
 * $\etavec = (\alpha - 1, -\beta)$ and $T(x) = (\log x, x)$. No `rsample` yet: a pathwise gamma draw needs implicit
 * reparameterisation (Figurnov, Mohamed and Mnih 2018), which waits for $\partial P(a, x)/\partial a$ in
 * `aifn-compute/numerics/special`.
 *
 * @param shape The shape $\alpha$; every element must be positive, or a `DomainError` is thrown.
 * @param rate The rate $\beta$, the reciprocal of the scale (use `GammaWithScale` to give the scale); every element
 *   must be positive, or a `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `shape` and `rate`.
 *
 * @example The density at a point, and the moments
 * const d = Gamma(2, 3)
 * print('p(1) =', d.prob(1), ' 9 e^-3 =', 9 * Math.exp(-3))
 * print('mean =', d.mean(), ' variance =', d.variance(), ' mode =', d.mode())
 *
 * @example The cdf and quantile round trip
 * const d = Gamma(0.5, 1)
 * const x = d.quantile(0.99)
 * print('quantile(0.99) =', x, ' cdf of it =', d.cdf(x))
 *
 * @example A seeded sample and its moments
 * const x = Gamma(4, 2).sample(stream(4), { shape: [10000] })
 * print('sample mean =', mean(x), ' sample variance =', variance(x))
 */
export function Gamma<A extends Value, B extends Value>(shape: A, rate: B): Univariate<A | B> {
  check('Gamma', 'shape', shape, positive, 'positive')
  check('Gamma', 'rate', rate, positive, 'positive')
  return univariate(gammaSpec(shape, rate))
}

/**
 * The gamma distribution with shape $\alpha$ and scale $\theta = 1/\beta$ (scipy's `gamma(a, scale=theta)`), with
 * density $p(x) = \frac{x^{\alpha - 1} e^{-x/\theta}}{\Gamma(\alpha)\theta^\alpha}$ for $x \ge 0$. It is `Gamma` with
 * rate $1/\theta$, so its `params` hold the shape and the rate, and its `name` is `'Gamma'`.
 *
 * @param shape The shape $\alpha$; every element must be positive, or a `DomainError` is thrown.
 * @param scale The scale $\theta$; every element must be positive, or a `DomainError` is thrown.
 * @returns The gamma distribution, with batch shape the broadcast shape of `shape` and `scale`.
 *
 * @example The same distribution as Gamma with the reciprocal rate
 * const d = GammaWithScale(2, 0.5)
 * print('params:', d.params)
 * print('mean =', d.mean(), ' Gamma(2, 2) mean =', Gamma(2, 2).mean())
 */
export function GammaWithScale<A extends Value, C extends Value>(shape: A, scale: C): Univariate<A | C> {
  check('Gamma', 'scale', scale, positive, 'positive')
  return Gamma(shape, div(1, scale)) as unknown as Univariate<A | C>
}

/**
 * The chi-square distribution $\ChiSq(k)$ with $k > 0$ degrees of freedom, $\GammaD(k/2, 1/2)$ in shape and rate
 * (scipy's `chi2`), with density $p(x) = \frac{x^{k/2 - 1} e^{-x/2}}{2^{k/2}\Gamma(k/2)}$ for $x \ge 0$. Mean $k$,
 * variance $2k$. No `rsample` (see `Gamma`).
 *
 * @param df The degrees of freedom $k$ (need not be an integer); every element must be positive, or a `DomainError`
 *   is thrown.
 * @returns The distribution, with batch shape the shape of `df`.
 *
 * @example The 95% point of one degree of freedom, and the moments
 * const d = ChiSquare(1)
 * print('quantile(0.95) =', d.quantile(0.95), ' 1.96^2 =', 1.959964 ** 2)
 * print('mean =', ChiSquare(4).mean(), ' variance =', ChiSquare(4).variance())
 *
 * @example Two degrees of freedom is the exponential with rate 1/2
 * print('cdf(3) =', ChiSquare(2).cdf(3), ' 1 - e^-1.5 =', 1 - Math.exp(-1.5))
 */
export function ChiSquare<K extends Value>(df: K): Univariate<K> {
  check('ChiSquare', 'df', df, positive, 'positive')
  const spec = gammaSpec(mul(0.5, df), 0.5)
  const valid = (x: Value) => mask([x], (v) => v >= 0)
  return univariate({
    ...spec,
    name: 'ChiSquare',
    params: { df },
    cdf: (x) => chiSquareCdf(maximum(x, 0), df),
    survival: (x) => outside(valid(x), chiSquareSf(maximum(x, 0), df), 1),
    sample: (s, shape) => drawn(chiSquareDraws(s, raw(df, 'ChiSquare'), { shape })),
  })
}

/**
 * The inverse-gamma distribution: the law of $1/X$ with $X \sim \GammaD(\alpha, \beta)$ in shape and rate, for shape
 * $\alpha > 0$ and scale $\beta > 0$ (scipy's `invgamma(a, scale=beta)`), with density
 * $p(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{-\alpha - 1} e^{-\beta/x}$ for $x > 0$.
 * Mean $\beta/(\alpha - 1)$ for $\alpha > 1$ and variance $\beta^2/((\alpha - 1)^2(\alpha - 2))$ for $\alpha > 2$ (NaN
 * otherwise); mode $\beta/(\alpha + 1)$. Its cdf is the upper tail $Q(\alpha, \beta/x)$ of the gamma. An exponential
 * family with $\etavec = (-\alpha - 1, -\beta)$ and $T(x) = (\log x, 1/x)$. No `rsample` (see `Gamma`).
 *
 * @param shape The shape $\alpha$; every element must be positive, or a `DomainError` is thrown.
 * @param scale The scale $\beta$, the rate of the gamma variable it inverts; every element must be positive, or a
 *   `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `shape` and `scale`.
 *
 * @example The moments, and the cdf as a gamma tail
 * const d = InverseGamma(3, 2)
 * print('mean =', d.mean(), ' variance =', d.variance(), ' mode =', d.mode())
 * print('cdf(0.5) =', d.cdf(0.5), ' Gamma(3, 2) survival(2) =', Gamma(3, 2).survival(2))
 *
 * @example A seeded sample and its mean
 * const x = InverseGamma(5, 8).sample(stream(5), { shape: [10000] })
 * print('sample mean =', mean(x), ' mean =', InverseGamma(5, 8).mean())
 */
export function InverseGamma<A extends Value, B extends Value>(shape: A, scale: B): Univariate<A | B> {
  check('InverseGamma', 'shape', shape, positive, 'positive')
  check('InverseGamma', 'scale', scale, positive, 'positive')
  const valid = (x: Value) => mask([x], (v) => v > 0)
  const inverse = (x: Value, ok = valid(x)) => div(scale, guard(x, ok, 1))
  return univariate({
    name: 'InverseGamma',
    params: { shape, scale },
    support: { type: 'interval', lower: 0, upper: Infinity },
    logProb: (x) => {
      const ok = valid(x)
      const xs = guard(x, ok, 1)
      return outside(
        ok,
        sub(sub(sub(mul(shape, log(scale)), logGamma(shape)), mul(add(shape, 1), log(xs))), div(scale, xs)),
        -Infinity,
      )
    },
    cdf: (x) => {
      const ok = valid(x)
      return outside(ok, regularisedGammaQ(shape, inverse(x, ok)), 0)
    },
    logcdf: (x) => {
      const ok = valid(x)
      return outside(ok, logRegularisedGammaQ(shape, inverse(x, ok)), -Infinity)
    },
    survival: (x) => {
      const ok = valid(x)
      return outside(ok, regularisedGammaP(shape, inverse(x, ok)), 1)
    },
    logSurvival: (x) => {
      const ok = valid(x)
      return outside(ok, logRegularisedGammaP(shape, inverse(x, ok)), 0)
    },
    // The p-quantile of 1/X is β over the inverse survival function of Gamma(α, 1) at p, never at 1 − p.
    quantile: (p) => div(scale, regularisedGammaQInverse(shape, p)),
    isf: (q) => div(scale, regularisedGammaPInverse(shape, q)),
    sample: (s, shape_) =>
      drawn(
        unwrap(
          div(raw(scale, 'InverseGamma'), gammaDraws(s, raw(shape, 'InverseGamma'), 1, { shape: shape_ })),
        ) as Tensor,
      ),
    mean: () =>
      where(
        mask([shape, scale], (a) => a > 1),
        div(scale, sub(shape, 1)),
        NaN,
      ),
    variance: () =>
      where(
        mask([shape, scale], (a) => a > 2),
        div(square(scale), mul(square(sub(shape, 1)), sub(shape, 2))),
        NaN,
      ),
    entropy: () => sub(add(add(shape, log(scale)), logGamma(shape)), mul(add(1, shape), digamma(shape))),
    mode: () => div(scale, add(shape, 1)),
    expFamily: {
      naturalParams: () => [sub(-1, shape), neg(scale)],
      sufficientStats: (x) => [log(x), div(1, x)],
      logPartition: () => sub(logGamma(shape), mul(shape, log(scale))),
      logBaseMeasure: () => 0,
    },
  })
}

// ── Beta ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The beta distribution with $a > 0$ and $b > 0$ on $[0, 1]$ (scipy's `beta`), with density
 * $p(x) = \frac{x^{a - 1}(1 - x)^{b - 1}}{B(a, b)}$ for $0 \le x \le 1$.
 * Mean $a/(a + b)$, variance $ab/((a + b)^2(a + b + 1))$. The mode is $(a - 1)/(a + b - 2)$ when $a, b > 1$ and NaN
 * otherwise (it is at a boundary, or not unique). The cdf is the regularised incomplete beta function $I_x(a, b)$ and
 * the survival function $I_{1 - x}(b, a)$. An exponential family with $\etavec = (a - 1, b - 1)$ and
 * $T(x) = (\log x, \log(1 - x))$. No `rsample` yet: its draw is built on gamma draws (see `Gamma`).
 *
 * @param a The first shape $a$, the exponent of $x$ plus one; every element must be positive, or a `DomainError` is
 *   thrown.
 * @param b The second shape $b$, the exponent of $1 - x$ plus one; every element must be positive, or a
 *   `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `a` and `b`.
 *
 * @example The density at a point, and the moments
 * const d = Beta(2, 3)
 * print('p(0.5) =', d.prob(0.5), ' 12 (0.5)(0.25) =', 12 * 0.5 * 0.25)
 * print('mean =', d.mean(), ' variance =', d.variance(), ' mode =', d.mode())
 *
 * @example A seeded sample and its moments
 * const x = Beta(2, 3).sample(stream(6), { shape: [10000] })
 * print('sample mean =', mean(x), ' sample variance =', variance(x))
 */
export function Beta<A extends Value, B extends Value>(a: A, b: B): Univariate<A | B> {
  check('Beta', 'a', a, positive, 'positive')
  check('Beta', 'b', b, positive, 'positive')
  const valid = (x: Value) => mask([x], (v) => v >= 0 && v <= 1)
  const total = add(a, b)
  return univariate({
    name: 'Beta',
    params: { a, b },
    support: { type: 'interval', lower: 0, upper: 1 },
    logProb: (x) => {
      const ok = valid(x)
      const xs = guard(x, ok, 0.5)
      return outside(ok, sub(add(xlogy(sub(a, 1), xs), xlog1py(sub(b, 1), neg(xs))), logBeta(a, b)), -Infinity)
    },
    cdf: (x) => regularisedBeta(a, b, minimum(maximum(x, 0), 1)),
    logcdf: (x) => logRegularisedBeta(a, b, minimum(maximum(x, 0), 1)),
    survival: (x) => regularisedBeta(b, a, minimum(maximum(sub(1, x), 0), 1)),
    // 1 − x drops a tiny x, so where the survival function is near 1 it is log1p(−cdf).
    logSurvival: (x) => {
      const y = minimum(maximum(sub(1, x), 0), 1)
      return logFromTails(
        regularisedBeta(b, a, y),
        regularisedBeta(a, b, minimum(maximum(x, 0), 1)),
        logRegularisedBeta(b, a, y),
      )
    },
    quantile: (p) => regularisedBetaInverse(a, b, p),
    isf: (q) => sub(1, regularisedBetaInverse(b, a, q)),
    sample: (s, shape) => drawn(betaDraws(s, raw(a, 'Beta'), raw(b, 'Beta'), { shape })),
    mean: () => div(a, total),
    variance: () => div(mul(a, b), mul(square(total), add(total, 1))),
    entropy: () =>
      add(
        sub(sub(logBeta(a, b), mul(sub(a, 1), digamma(a))), mul(sub(b, 1), digamma(b))),
        mul(sub(total, 2), digamma(total)),
      ),
    mode: () =>
      where(
        mask([a, b], (u, v) => u > 1 && v > 1),
        div(sub(a, 1), sub(total, 2)),
        NaN,
      ),
    expFamily: {
      naturalParams: () => [sub(a, 1), sub(b, 1)],
      sufficientStats: (x) => [log(x), log1p(neg(x))],
      logPartition: () => logBeta(a, b),
      logBaseMeasure: () => 0,
    },
  })
}

// ── Fisher–Snedecor F ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The $F$ distribution with $d_1 > 0$ and $d_2 > 0$ degrees of freedom (scipy's `f(dfn, dfd)`), with density
 * $p(x) = \frac{1}{x B(d_1/2, d_2/2)} \sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1 + d_2}}}$ for $x \ge 0$:
 * the law of $(X_1/d_1)/(X_2/d_2)$ for independent $X_1 \sim \ChiSq(d_1)$ and $X_2 \sim \ChiSq(d_2)$, and of
 * $(d_2/d_1) B/(1 - B)$ for $B \sim \Beta(d_1/2, d_2/2)$, which gives the cdf
 * $I_{d_1 x/(d_1 x + d_2)}(d_1/2, d_2/2)$, the survival function from the other tail of the same beta, and the
 * quantiles. Mean $d_2/(d_2 - 2)$ for $d_2 > 2$ and variance $2d_2^2(d_1 + d_2 - 2)/(d_1(d_2 - 2)^2(d_2 - 4))$ for
 * $d_2 > 4$ (Infinity otherwise); the null law of an analysis-of-variance ratio.
 *
 * @param df1 The numerator degrees of freedom $d_1$; every element must be positive, or a `DomainError` is thrown.
 * @param df2 The denominator degrees of freedom $d_2$; every element must be positive, or a `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `df1` and `df2`.
 *
 * @example A critical value, and the moments
 * const d = FisherSnedecor(5, 10)
 * print('quantile(0.95) =', d.quantile(0.95))
 * print('mean =', d.mean(), ' 10 / 8 =', 10 / 8, ' variance =', d.variance())
 *
 * @example One numerator degree of freedom is a squared Student t
 * print('F(1, 10) quantile(0.95) =', FisherSnedecor(1, 10).quantile(0.95))
 * print('t(10) quantile(0.975)^2 =', StudentT(10).quantile(0.975) ** 2)
 *
 * @example A seeded sample and its mean
 * const x = FisherSnedecor(5, 10).sample(stream(7), { shape: [10000] })
 * print('sample mean =', mean(x))
 */
export function FisherSnedecor<A extends Value, B extends Value>(df1: A, df2: B): Univariate<A | B> {
  check('FisherSnedecor', 'df1', df1, positive, 'positive')
  check('FisherSnedecor', 'df2', df2, positive, 'positive')
  const a = mul(0.5, df1)
  const b = mul(0.5, df2)
  const valid = (x: Value) => mask([x], (v) => v >= 0)
  // The beta variable B = d₁x/(d₁x + d₂) and its complement d₂/(d₁x + d₂), with x clamped at 0.
  const lower = (x: Value) => {
    const s = mul(df1, maximum(x, 0))
    return div(s, add(s, df2))
  }
  const upper = (x: Value) => div(df2, add(mul(df1, maximum(x, 0)), df2))
  const fromBeta = (y: Value) => div(mul(df2, y), mul(df1, sub(1, y)))
  return univariate({
    name: 'FisherSnedecor',
    params: { df1, df2 },
    support: { type: 'interval', lower: 0, upper: Infinity },
    logProb: (x) => {
      const ok = valid(x)
      const xs = guard(x, ok, 1)
      const body = sub(
        add(mul(a, log(df1)), add(mul(b, log(df2)), xlogy(sub(a, 1), xs))),
        add(mul(add(a, b), log(add(mul(df1, xs), df2))), logBeta(a, b)),
      )
      return outside(ok, body, -Infinity)
    },
    cdf: (x) => regularisedBeta(a, b, lower(x)),
    logcdf: (x) => logRegularisedBeta(a, b, lower(x)),
    survival: (x) => regularisedBeta(b, a, upper(x)),
    logSurvival: (x) => logRegularisedBeta(b, a, upper(x)),
    quantile: (p) => fromBeta(regularisedBetaInverse(a, b, p)),
    // The complement 1 − B = I⁻¹(b, a, q) keeps upper quantiles accurate for tiny q.
    isf: (q) => {
      const c = regularisedBetaInverse(b, a, q)
      return div(mul(df2, sub(1, c)), mul(df1, c))
    },
    sample: (s, shape) => {
      const y = drawn(betaDraws(s, raw(a, 'FisherSnedecor'), raw(b, 'FisherSnedecor'), { shape }))
      return drawnAt(fromBeta(y), shape)
    },
    // Infinite moments are Infinity (the variable is positive), as in scipy.
    mean: () =>
      where(
        mask([df1, df2], (_u, v) => v > 2),
        atBatch(div(df2, sub(df2, 2)), df1),
        Infinity,
      ),
    variance: () =>
      where(
        mask([df1, df2], (_u, v) => v > 4),
        div(mul(mul(2, square(df2)), sub(add(df1, df2), 2)), mul(mul(df1, square(sub(df2, 2))), sub(df2, 4))),
        Infinity,
      ),
    // h = log B(a, b) − (a − 1)ψ(a) − (b + 1)ψ(b) + (a + b)ψ(a + b) + log(d₂/d₁): the beta-prime entropy, shifted
    // by the log of the scale d₂/d₁.
    entropy: () =>
      add(
        sub(sub(logBeta(a, b), mul(sub(a, 1), digamma(a))), mul(add(b, 1), digamma(b))),
        add(mul(add(a, b), digamma(add(a, b))), log(div(df2, df1))),
      ),
    mode: () =>
      where(
        mask([df1, df2], (u) => u > 2),
        div(mul(sub(df1, 2), df2), mul(df1, add(df2, 2))),
        atBatch(0, df1, df2),
      ),
  })
}

// ── Weibull ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Weibull distribution with shape $k > 0$ and scale $\lambda > 0$ (scipy's `weibull_min(c, scale=lambda)`), with
 * density $p(x) = \frac{k}{\lambda} \LP \frac{x}{\lambda} \RP^{k - 1} e^{-(x/\lambda)^k}$ for $x \ge 0$.
 * Survival function $e^{-(x/\lambda)^k}$, mean $\lambda\Gamma(1 + 1/k)$, variance
 * $\lambda^2(\Gamma(1 + 2/k) - \Gamma(1 + 1/k)^2)$, and mode $\lambda((k - 1)/k)^{1/k}$ for $k > 1$ (0 otherwise).
 * `rsample` is the pathwise draw through the quantile $\lambda(-\log(1 - p))^{1/k}$.
 *
 * @param shape The shape $k$; every element must be positive, or a `DomainError` is thrown.
 * @param scale The scale $\lambda$; every element must be positive, or a `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `shape` and `scale`.
 *
 * @example Shape 1 is the exponential with rate 1/scale
 * print('log p(1.5) =', Weibull(1, 2).logProb(1.5), ' exponential:', Exponential(0.5).logProb(1.5))
 *
 * @example The moments and the median
 * const d = Weibull(2, 1)
 * print('mean =', d.mean(), ' sqrt(pi) / 2 =', Math.sqrt(Math.PI) / 2)
 * print('median =', d.quantile(0.5), ' sqrt(log 2) =', Math.sqrt(Math.LN2))
 */
export function Weibull<K extends Value, L extends Value>(shape: K, scale: L): Univariate<K | L> {
  check('Weibull', 'shape', shape, positive, 'positive')
  check('Weibull', 'scale', scale, positive, 'positive')
  const valid = (x: Value) => mask([x], (v) => v >= 0)
  // (x/λ)^k, with x clamped at 0 so the unused branch stays finite.
  const power = (x: Value) => pow(div(maximum(x, 0), scale), shape)
  const quantile = (p: Value) => mul(scale, pow(neg(log1p(neg(p))), div(1, shape)))
  return univariate({
    name: 'Weibull',
    params: { shape, scale },
    support: { type: 'interval', lower: 0, upper: Infinity },
    logProb: (x) => {
      const ok = valid(x)
      const xs = guard(x, ok, 1)
      const r = div(xs, scale)
      return outside(ok, sub(add(sub(log(shape), log(scale)), xlogy(sub(shape, 1), r)), pow(r, shape)), -Infinity)
    },
    cdf: (x) => neg(expm1(neg(power(x)))),
    logcdf: (x) => log1mexp(neg(power(x))),
    survival: (x) => exp(neg(power(x))),
    logSurvival: (x) => neg(power(x)),
    quantile,
    isf: (q) => mul(scale, pow(neg(log(q)), div(1, shape))),
    rsample: (s, shape_, scalar) => inverseTransform(s, shape_, scalar, quantile),
    mean: () => mul(scale, gammaFunction(add(1, div(1, shape)))),
    variance: () =>
      mul(square(scale), sub(gammaFunction(add(1, div(2, shape))), square(gammaFunction(add(1, div(1, shape)))))),
    entropy: () => add(add(mul(EULER_GAMMA, sub(1, div(1, shape))), log(div(scale, shape))), 1),
    mode: () =>
      where(
        mask([shape, scale], (k) => k > 1),
        mul(scale, pow(div(sub(shape, 1), shape), div(1, shape))),
        0,
      ),
  })
}

// ── Gumbel ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Gumbel (maximum) distribution with location $\mu$ and scale $\beta > 0$ (scipy's `gumbel_r`), with density
 * $p(x) = \frac{1}{\beta} \exp(-(z + e^{-z}))$ with $z = (x - \mu)/\beta$, and cdf $\exp(-e^{-z})$.
 * Mean $\mu + \gamma\beta$ ($\gamma$ the Euler–Mascheroni constant), variance $\pi^2\beta^2/6$, mode $\mu$. `rsample`
 * is the pathwise draw through the quantile $\mu - \beta\log(-\log p)$.
 *
 * @param loc The location $\mu$, the mode.
 * @param scale The scale $\beta$; every element must be positive, or a `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `loc` and `scale`.
 *
 * @example The cdf at the mode, and the moments
 * const d = Gumbel(0, 1)
 * print('cdf(0) =', d.cdf(0), ' 1/e =', Math.exp(-1))
 * print('mean =', d.mean(), ' variance =', d.variance())
 *
 * @example A seeded sample and its moments
 * const x = Gumbel(1, 2).sample(stream(8), { shape: [40000] })
 * print('sample mean =', mean(x), ' 1 + 2 gamma =', 1 + 2 * 0.5772156649)
 */
export function Gumbel<M extends Value, S extends Value>(loc: M, scale: S): Univariate<M | S> {
  check('Gumbel', 'scale', scale, positive, 'positive')
  const z = (x: Value) => div(sub(x, loc), scale)
  const quantile = (p: Value) => sub(loc, mul(scale, log(neg(log(p)))))
  return univariate({
    name: 'Gumbel',
    params: { loc, scale },
    support: { type: 'real' },
    logProb: (x) => {
      const t = z(x)
      return neg(add(add(log(scale), t), exp(neg(t))))
    },
    cdf: (x) => exp(neg(exp(neg(z(x))))),
    logcdf: (x) => neg(exp(neg(z(x)))),
    survival: (x) => neg(expm1(neg(exp(neg(z(x)))))),
    logSurvival: (x) => log1mexp(neg(exp(neg(z(x))))),
    quantile,
    isf: (q) => sub(loc, mul(scale, log(neg(log1p(neg(q)))))),
    rsample: (s, shape, scalar) => inverseTransform(s, shape, scalar, quantile),
    mean: () => add(loc, mul(EULER_GAMMA, scale)),
    variance: () => atBatch(mul((Math.PI * Math.PI) / 6, square(scale)), loc),
    entropy: () => atBatch(add(log(scale), EULER_GAMMA + 1), loc),
    mode: () => atBatch(loc, scale),
  })
}

// ── Generalised Pareto ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The generalised Pareto distribution with shape $\xi$, location $\mu$ and scale $\sigma > 0$ (scipy's
 * `genpareto(c, loc, scale)`): the law of exceedances over a high threshold (Pickands, 1975; Balkema and de Haan,
 * 1974). With $z = (x - \mu)/\sigma$ the density is $p(x) = \frac{1}{\sigma}(1 + \xi z)^{-1/\xi - 1}$ and the survival
 * function $(1 + \xi z)^{-1/\xi}$, which are $e^{-z}/\sigma$ and $e^{-z}$ at $\xi = 0$ (the exponential). $\xi > 0$
 * gives a Pareto-like heavy tail, $\xi < 0$ a bounded support up to $\mu - \sigma/\xi$. The mean
 * $\mu + \sigma/(1 - \xi)$ exists for $\xi < 1$ and the variance $\sigma^2/((1 - \xi)^2(1 - 2\xi))$ for $\xi < 1/2$
 * ($\infty$ beyond). `rsample` is the pathwise draw through the quantile.
 *
 * @param shape The shape $\xi$, any finite number (a `DomainError` is thrown for NaN or $\pm\infty$).
 * @param loc The location $\mu$, the lower end of the support.
 * @param scale The scale $\sigma$; every element must be positive, or a `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of the three parameters.
 *
 * @example Shape 0 is the exponential
 * print('log p(1) =', GeneralisedPareto(0, 0, 2).logProb(1), ' exponential:', Exponential(0.5).logProb(1))
 *
 * @example A heavy tail, and a bounded support
 * const heavy = GeneralisedPareto(0.25, 0, 1)
 * print('mean =', heavy.mean(), ' 1 / (1 - 0.25) =', 1 / 0.75)
 * print('survival(3) =', heavy.survival(3), ' (1 + 0.75)^-4 =', 1.75 ** -4)
 * const bounded = GeneralisedPareto(-0.5, 0, 1)
 * print('upper end =', bounded.support.upper, ' cdf(2) =', bounded.cdf(2))
 */
export function GeneralisedPareto<C extends Value, M extends Value, S extends Value>(
  shape: C,
  loc: M,
  scale: S,
): Univariate<C | M | S> {
  check('GeneralisedPareto', 'scale', scale, positive, 'positive')
  check('GeneralisedPareto', 'shape', shape, finite, 'finite')
  const zero = mask([shape], (c) => c === 0)
  // ξ with its zeros replaced, so the unused branch of ξ = 0 stays finite.
  const safeShape = guard(
    shape,
    mask([shape], (c) => c !== 0),
    1,
  )
  const lowerOk = (x: Value) => mask([x, loc], (v, m) => v >= m)
  const upperOk = (x: Value) => mask([x, shape, loc, scale], (v, c, m, s) => c >= 0 || v <= m - s / c)
  const inside = (x: Value) => mask([x, shape, loc, scale], (v, c, m, s) => v >= m && (c >= 0 || v <= m - s / c))
  const z = (x: Value) => div(sub(x, loc), scale)
  // t(z) = log(1 + ξz)/ξ (z at ξ = 0) is −log S(x); arguments outside the support are moved into it.
  const t = (x: Value) => {
    const zs = guard(z(x), inside(x), 0)
    return where(zero, zs, div(log1p(mul(safeShape, zs)), safeShape))
  }
  // The inverse of t: x = μ + σ·(e^{ξL} − 1)/ξ (μ + σL at ξ = 0) for L = −log S.
  const fromTail = (L: Value) => add(loc, mul(scale, where(zero, L, div(expm1(mul(safeShape, L)), safeShape))))
  const quantile = (p: Value) => fromTail(neg(log1p(neg(p))))
  // Below μ the cdf is 0; above the upper end (ξ < 0) it is 1.
  const piecewise = (x: Value, expr: Value, below: number, above: number) =>
    outside(lowerOk(x), outside(upperOk(x), expr, above), below)
  return univariate({
    name: 'GeneralisedPareto',
    params: { shape, loc, scale },
    support: {
      type: 'interval',
      lower: loc,
      upper: where(
        mask([shape], (c) => c < 0),
        sub(loc, div(scale, safeShape)),
        Infinity,
      ),
    },
    logProb: (x) => {
      const ok = inside(x)
      const zs = guard(z(x), ok, 0)
      return outside(ok, neg(add(add(log(scale), t(x)), log1p(mul(shape, zs)))), -Infinity)
    },
    cdf: (x) => piecewise(x, neg(expm1(neg(t(x)))), 0, 1),
    logcdf: (x) => piecewise(x, log1mexp(neg(t(x))), -Infinity, 0),
    survival: (x) => piecewise(x, exp(neg(t(x))), 1, 0),
    logSurvival: (x) => piecewise(x, neg(t(x)), 0, -Infinity),
    quantile,
    isf: (q) => fromTail(neg(log(q))),
    rsample: (s, drawShape, scalar) => inverseTransform(s, drawShape, scalar, quantile),
    mean: () => {
      const ok = mask([shape], (c) => c < 1)
      return where(ok, add(loc, div(scale, sub(1, guard(shape, ok, 0)))), Infinity)
    },
    variance: () => {
      const ok = mask([shape], (c) => c < 0.5)
      const c = guard(shape, ok, 0)
      return where(ok, div(square(scale), mul(square(sub(1, c)), sub(1, mul(2, c)))), Infinity)
    },
    entropy: () => add(add(log(scale), shape), atBatch(1, loc)),
    mode: () => atBatch(loc, shape, scale),
  })
}

// ── Von Mises ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * $\int_{-\pi}^{z} e^{\kappa(\cos t - 1)} \, dt$ by composite Simpson's rule, with enough panels to resolve the peak's
 * width $1/\sqrt\kappa$ (the integrand is scaled by $e^{-\kappa}$ so that it does not overflow).
 *
 * @param z The upper limit, an angle relative to the mean direction; 0 is returned for $z \le -\pi$.
 * @param kappa The concentration $\kappa \ge 0$.
 * @returns The integral.
 */
function vonMisesIntegral(z: number, kappa: number): number {
  const lo = -Math.PI
  if (z <= lo) return 0
  const panels = 2 * Math.ceil((600 * (1 + Math.sqrt(kappa)) * (z - lo)) / (2 * Math.PI) + 2)
  const h = (z - lo) / panels
  let total = 0
  for (let i = 0; i <= panels; i++) {
    const w = i === 0 || i === panels ? 1 : i % 2 === 1 ? 4 : 2
    total += w * Math.exp(kappa * (Math.cos(lo + i * h) - 1))
  }
  return (total * h) / 3
}

/**
 * The von Mises cdf by quadrature: the integral of the density from $\mu - \pi$ to $x$, as the ratio of two
 * `vonMisesIntegral`s. Not periodic: 0 below $\mu - \pi$ and 1 above $\mu + \pi$.
 *
 * @param x The angle at which to evaluate the cdf (not reduced modulo $2\pi$).
 * @param loc The mean direction $\mu$.
 * @param kappa The concentration $\kappa \ge 0$.
 * @returns The probability of an angle in $[\mu - \pi, x]$.
 */
function vonMisesCdf(x: number, loc: number, kappa: number): number {
  const z = x - loc
  if (z <= -Math.PI) return 0
  if (z >= Math.PI) return 1
  return vonMisesIntegral(z, kappa) / vonMisesIntegral(Math.PI, kappa)
}

/**
 * One uniform in $[0, 1)$ (2 words of the stream).
 *
 * @param s The stream to draw from; it is advanced.
 * @returns The uniform.
 */
const unit = (s: Stream): number => units(s, 1)[0]

/**
 * One von Mises draw by the rejection sampler of Best and Fisher (1979), "Efficient simulation of the von Mises
 * distribution", Applied Statistics 28(2): a wrapped Cauchy envelope with acceptance rate above 65% for every
 * $\kappa$. For $\kappa$ below $10^{-6}$ the distribution is uniform to double precision (and the envelope's constants
 * cancel), so draws are uniform.
 *
 * @param s The stream to draw from; it is advanced (by one uniform when the draw is uniform, else three per
 *   proposal).
 * @param loc The mean direction $\mu$.
 * @param kappa The concentration $\kappa \ge 0$.
 * @returns An angle in $[\mu - \pi, \mu + \pi]$.
 */
function vonMisesDraw(s: Stream, loc: number, kappa: number): number {
  if (kappa < 1e-6) return loc + Math.PI * (2 * unit(s) - 1)
  const tau = 1 + Math.sqrt(1 + 4 * kappa * kappa)
  const rho = (tau - Math.sqrt(2 * tau)) / (2 * kappa)
  const r = (1 + rho * rho) / (2 * rho)
  for (;;) {
    const z = Math.cos(Math.PI * unit(s))
    const f = (1 + r * z) / (r + z)
    const c = kappa * (r - f)
    const u2 = unit(s)
    const u3 = unit(s)
    if (c * (2 - c) - u2 > 0 || Math.log(c / u2) + 1 - c >= 0) {
      return loc + Math.sign(u3 - 0.5) * Math.acos(Math.min(1, Math.max(-1, f)))
    }
  }
}

/**
 * The von Mises distribution on the circle with mean direction $\mu$ and concentration $\kappa \ge 0$ (scipy's
 * `vonmises(kappa, loc)`), with density $p(x) = \frac{e^{\kappa\cos(x - \mu)}}{2\pi I_0(\kappa)}$, which is periodic;
 * the cdf runs over $[\mu - \pi, \mu + \pi]$ (0 below, 1 above). `mean()` is $\mu$ and `variance()` the circular
 * variance $1 - I_1(\kappa)/I_0(\kappa)$. The cdf (by quadrature) and quantile (by bisection) are not differentiable;
 * the density is, in $x$, $\mu$ and $\kappa$. No `rsample` (the draw is a rejection sampler, Best and Fisher 1979).
 *
 * @param loc The mean direction $\mu$, in radians.
 * @param concentration The concentration $\kappa$ (0 is the uniform distribution on the circle, and large $\kappa$
 *   approaches $\Gauss(\mu, 1/\kappa)$); every element must be non-negative, or a `DomainError` is thrown.
 * @returns The distribution, with batch shape the broadcast shape of `loc` and `concentration`.
 *
 * @example The density is uniform at zero concentration, and the cdf is one half at the mean
 * print('p(1) at kappa = 0:', VonMises(0, 0).prob(1), ' 1 / (2 pi) =', 1 / (2 * Math.PI))
 * const d = VonMises(0.5, 2)
 * print('cdf(0.5) =', d.cdf(0.5), ' quantile(0.9) =', d.quantile(0.9))
 *
 * @example The circular variance from a seeded sample
 * const d = VonMises(0, 2)
 * const x = d.sample(stream(9), { shape: [10000] })
 * print('1 - mean cos x =', 1 - mean(cos(x)), ' circular variance =', d.variance())
 */
export function VonMises<M extends Value, K extends Value>(loc: M, concentration: K): Univariate<M | K> {
  check('VonMises', 'concentration', concentration, (k) => k >= 0, 'non-negative')
  const kappa = concentration
  return univariate({
    name: 'VonMises',
    params: { loc, concentration },
    support: { type: 'circle', lower: sub(loc, Math.PI), upper: add(loc, Math.PI) },
    logProb: (x) => sub(sub(mul(kappa, cos(sub(x, loc))), LOG_2PI), logBesselI0(kappa)),
    cdf: (x) => rawOnly('VonMises.cdf', [x, loc, kappa], vonMisesCdf),
    // A zero tensor of the draw shape fixes the shape the parameters broadcast to; draws are made in row-major order.
    sample: (s, shape) =>
      drawEach('VonMises', [raw(loc, 'VonMises'), raw(kappa, 'VonMises')], { shape }, s, (e, m, k) =>
        vonMisesDraw(e, m, k),
      ) as Tensor,
    mean: () => atBatch(loc, kappa),
    variance: () => atBatch(sub(1, besselRatio(kappa)), loc),
    entropy: () => atBatch(add(sub(LOG_2PI, mul(kappa, besselRatio(kappa))), logBesselI0(kappa)), loc),
    mode: () => atBatch(loc, kappa),
  })
}

// ── Truncated normal ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The quantile of a standard normal truncated to $[a, b]$, reflected into the lower tail when $a > 0$ for accuracy:
 * $\Phi^{-1}(\Phi(a) + p(\Phi(b) - \Phi(a)))$, clamped to $[a, b]$.
 *
 * @param p The probability, in $[0, 1]$; NaN is returned outside it.
 * @param a The standardised lower bound $(\text{low} - \mu)/\sigma$, possibly $-\infty$.
 * @param b The standardised upper bound $(\text{high} - \mu)/\sigma$, possibly $\infty$.
 * @returns The standardised quantile, in $[a, b]$.
 */
function truncatedStandardQuantile(p: number, a: number, b: number): number {
  if (!(p >= 0 && p <= 1)) return NaN
  if (a > 0) return -truncatedStandardQuantile(1 - p, -b, -a)
  const fa = unwrap(normalCdf(a)) as number
  const fb = unwrap(normalCdf(b)) as number
  return Math.min(b, Math.max(a, unwrap(normalQuantile(fa + p * (fb - fa))) as number))
}

/**
 * The normal distribution $\Gauss(\mu, \sigma^2)$ truncated to $[l, h]$ (either may be infinite; scipy's
 * `truncnorm((low - loc)/scale, (high - loc)/scale, loc, scale)`), with density
 * $p(x) = \frac{\phi(z)}{\sigma(\Phi(\beta) - \Phi(\alpha))}$ for $l \le x \le h$, where $z = (x - \mu)/\sigma$,
 * $\alpha = (l - \mu)/\sigma$ and $\beta = (h - \mu)/\sigma$. The normaliser $Z = \Phi(\beta) - \Phi(\alpha)$ is
 * computed in log space without cancellation, so far-tail truncations (the TrueSkill and probit cases) stay accurate.
 * Mean $\mu + \sigma(\phi(\alpha) - \phi(\beta))/Z$; mode $\mu$ clamped to $[l, h]$. The quantile (and the sampler, by
 * inversion) reflects into the lower tail; it is not differentiable, so there is no `rsample`.
 *
 * @param loc The mean $\mu$ of the normal before truncation (not the mean of the result).
 * @param scale The standard deviation $\sigma$ of the normal before truncation; every element must be positive, or a
 *   `DomainError` is thrown.
 * @param low The lower bound $l$, or `-Infinity`.
 * @param high The upper bound $h$, or `Infinity`; every element of `high - low` must be positive, or a `DomainError` is
 *   thrown.
 * @returns The distribution, with batch shape the broadcast shape of the four parameters.
 *
 * @example The half-normal
 * const d = TruncatedNormal(0, 1, 0, Infinity)
 * print('p(0) =', d.prob(0), ' mean =', d.mean(), ' sqrt(2 / pi) =', Math.sqrt(2 / Math.PI))
 * print('variance =', d.variance(), ' 1 - 2 / pi =', 1 - 2 / Math.PI)
 *
 * @example A far-tail truncation stays accurate
 * const d = TruncatedNormal(0, 1, 10, Infinity)
 * print('mean =', d.mean(), ' log p(10) =', d.logProb(10))
 *
 * @example The cdf and quantile round trip, and a seeded sample inside the bounds
 * const d = TruncatedNormal(1, 2, -1, 2)
 * print('quantile(0.3) =', d.quantile(0.3), ' cdf of it =', d.cdf(d.quantile(0.3)))
 * print('draws:', d.sample(stream(10), { shape: [5] }))
 */
export function TruncatedNormal<M extends Value, S extends Value, A extends Value, B extends Value>(
  loc: M,
  scale: S,
  low: A,
  high: B,
): Univariate<M | S | A | B> {
  check('TruncatedNormal', 'scale', scale, positive, 'positive')
  check('TruncatedNormal', 'high − low', sub(unwrap(high), unwrap(low)), positive, 'positive')
  const alpha = div(sub(low, loc), scale)
  const beta = div(sub(high, loc), scale)
  const logZ = normalLogIntervalProbability(alpha, beta)
  const z = (x: Value) => div(sub(x, loc), scale)
  const inside = (x: Value) => mask([x, low, high], (v, a, b) => v >= a && v <= b)
  const clipped = (x: Value) => div(sub(minimum(maximum(x, low), high), loc), scale)
  const logLower = (x: Value) => sub(normalLogIntervalProbability(alpha, clipped(x)), logZ)
  const logUpper = (x: Value) => sub(normalLogIntervalProbability(clipped(x), beta), logZ)
  const logTail = (own: Value, other: Value) => {
    const small = mask([own], (v) => !(v > -Math.LN2))
    const large = mask([own], (v) => v > -Math.LN2)
    return where(small, own, log1mexp(guard(other, large, -1)))
  }
  // φ(t)/Z and t·φ(t)/Z at the bounds, with infinite bounds contributing 0.
  const density = (t: Value) => exp(sub(normalLogPdf(t), logZ))
  const moment = (t: Value) => {
    const ok = mask([t], finite)
    return outside(ok, mul(guard(t, ok, 0), density(guard(t, ok, 0))), 0)
  }
  const quantile = (p: Value) =>
    add(
      raw(loc, 'TruncatedNormal.quantile'),
      mul(
        raw(scale, 'TruncatedNormal.quantile'),
        rawOnly('TruncatedNormal.quantile', [p, alpha, beta], truncatedStandardQuantile),
      ),
    )
  return univariate({
    name: 'TruncatedNormal',
    params: { loc, scale, low, high },
    support: { type: 'interval', lower: low, upper: high },
    logProb: (x) => {
      const ok = inside(x)
      return outside(ok, sub(sub(normalLogPdf(z(guard(x, ok, 0))), log(scale)), logZ), -Infinity)
    },
    cdf: (x) => exp(logLower(x)),
    // Each log tail is the difference of logs where it is at most ½, and log1p of minus the other tail above.
    logcdf: (x) => logTail(logLower(x), logUpper(x)),
    survival: (x) => exp(logUpper(x)),
    logSurvival: (x) => logTail(logUpper(x), logLower(x)),
    quantile,
    sample: (s, shape) => drawnAt(inverseTransform(s, shape, false, quantile), shape),
    mean: () => add(loc, mul(scale, sub(density(alpha), density(beta)))),
    variance: () => {
      const d = sub(density(alpha), density(beta))
      return mul(square(scale), sub(add(1, sub(moment(alpha), moment(beta))), square(d)))
    },
    entropy: () => add(add(add(0.5 * (1 + LOG_2PI), log(scale)), logZ), mul(0.5, sub(moment(alpha), moment(beta)))),
    mode: () => minimum(maximum(loc, low), high),
  })
}
