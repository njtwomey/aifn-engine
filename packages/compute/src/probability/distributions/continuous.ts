/**
 * Continuous univariate families. Every density is a composition of `aifn-compute/foundation/tensor` and `aifn-compute/numerics/special` primitives, so
 * `logProb` is differentiable in the value and in every parameter; cdfs and quantiles are differentiable wherever the
 * special function they use is (the incomplete gamma and beta functions only in their continuous argument). Numerically
 * inverted quantiles and quadrature cdfs are not differentiable and say so when given traced values.
 *
 * Parameterisations follow scipy.stats (`loc`, `scale`) wherever there is a scale: Normal(loc, scale) is scipy's
 * norm(loc, scale), with the standard deviation as scale.
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

const positive = (x: number) => x > 0
const finite = (x: number) => Number.isFinite(x)

/** NaN with the batch shape of `params`. */
function nanAt(...params: Value[]): Value {
  return rawMap(params, () => NaN)
}

/** Draws as a tensor (the samplers return a number when every parameter is a number and no shape is given). */
const drawn = (x: number | Tensor): Tensor => (typeof x === 'number' ? fromData(new Float64Array([x]), []) : x)

/** A raw draw of a composition (numbers or tensors, never traced) as a tensor of the draw's shape. */
const drawnAt = (x: Value, shape: number[]): Tensor => {
  const r = unwrap(x)
  return typeof r === 'number' ? fromData(new Float64Array(shape.reduce((a, b) => a * b, 1)).fill(r), shape) : r
}

// ── Normal ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The normal distribution N(loc, scale²), with the standard deviation `scale` > 0 (scipy's norm). Tails of the cdf,
 * log cdf and survival function keep full relative accuracy (they use Φ and log Φ, never 1 − Φ). An exponential family
 * with η = (μ/σ², −1/(2σ²)) and T(x) = (x, x²).
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

/** The normal distribution with natural parameters η₁ = μ/σ² and η₂ = −1/(2σ²) < 0 (as used by EP messages). */
export function normalFromNatural<A extends Value, B extends Value>(eta1: A, eta2: B): Univariate<A | B> {
  check('normalFromNatural', 'eta2', eta2, (x) => x < 0, 'negative')
  const variance = div(-0.5, eta2)
  return Normal(mul(eta1, variance), sqrt(variance)) as unknown as Univariate<A | B>
}

// ── Log-normal ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The log-normal distribution: log X ~ N(mu, sigma²), sigma > 0 (scipy's lognorm(s = sigma, scale = e^mu)). Support
 * x > 0.
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
 * Student's t distribution with `df` = ν > 0 degrees of freedom, location and scale (scipy's t(df, loc, scale)).
 * The mean is NaN for ν ≤ 1; the variance is ∞ for 1 < ν ≤ 2 and NaN for ν ≤ 1. Differentiable in x, loc and scale;
 * the cdf and quantile are not differentiable in ν. No `rsample`: the draw divides by a chi-square draw (see `Gamma`).
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

/** The Cauchy distribution with location and scale > 0 (scipy's cauchy). Its mean and variance are NaN (undefined). */
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

/** The Laplace (double exponential) distribution with location and scale b > 0 (scipy's laplace). */
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

/** The logistic distribution with location and scale s > 0 (scipy's logistic); its cdf is the sigmoid. */
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

/** The continuous uniform distribution on [low, high], low < high (scipy's uniform(low, high − low)). Mode: NaN. */
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

/** The exponential distribution with rate λ > 0 (mean 1/λ; scipy's expon(scale = 1/λ)). Support x ≥ 0. */
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
 * The gamma distribution with shape α > 0 and rate β > 0 (mean α/β; scipy's gamma(α, scale = 1/β)). For α < 1 the
 * density is infinite at 0 and the mode is 0. The quantile is found numerically (not differentiable). An exponential
 * family with η = (α − 1, −β) and T(x) = (log x, x). No `rsample` yet: a pathwise gamma draw needs implicit
 * reparameterisation (Figurnov, Mohamed and Mnih 2018), which waits for ∂P(a, x)/∂a in `aifn-compute/numerics/special`.
 */
export function Gamma<A extends Value, B extends Value>(shape: A, rate: B): Univariate<A | B> {
  check('Gamma', 'shape', shape, positive, 'positive')
  check('Gamma', 'rate', rate, positive, 'positive')
  return univariate(gammaSpec(shape, rate))
}

/** The gamma distribution with shape α and scale θ = 1/β (scipy's gamma(α, scale = θ)); `params` holds the rate. */
export function GammaWithScale<A extends Value, C extends Value>(shape: A, scale: C): Univariate<A | C> {
  check('Gamma', 'scale', scale, positive, 'positive')
  return Gamma(shape, div(1, scale)) as unknown as Univariate<A | C>
}

/** The chi-square distribution with k > 0 degrees of freedom: Gamma(k/2, rate 1/2) (scipy's chi2). No `rsample` (see `Gamma`). */
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
 * The inverse-gamma distribution: 1/X with X ~ Gamma(α, rate β), for shape α > 0 and scale β > 0 (scipy's
 * invgamma(α, scale = β)). Mean β/(α − 1) for α > 1 (NaN otherwise), variance for α > 2. An exponential family with
 * η = (−α − 1, −β) and T(x) = (log x, 1/x). No `rsample` (see `Gamma`).
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
 * The beta distribution with a > 0 and b > 0 on [0, 1] (scipy's beta). The mode is (a − 1)/(a + b − 2) when a, b > 1
 * and NaN otherwise (it is at a boundary, or not unique). An exponential family with η = (a − 1, b − 1) and
 * T(x) = (log x, log(1 − x)). No `rsample` yet: its draw is built on gamma draws (see `Gamma`).
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
 * The F distribution with d₁ > 0 and d₂ > 0 degrees of freedom (scipy's f(d₁, d₂)): the law of (X₁/d₁)/(X₂/d₂) for
 * independent X₁ ~ χ²(d₁) and X₂ ~ χ²(d₂), and of (d₂/d₁)·B/(1 − B) for B ~ Beta(d₁/2, d₂/2), which gives the cdf
 * I_{d₁x/(d₁x + d₂)}(d₁/2, d₂/2), the survival function from the other tail of the same beta, and the quantiles.
 * Mean d₂/(d₂ − 2) for d₂ > 2 and the variance for d₂ > 4 (Infinity otherwise, as scipy); the null law of an
 * analysis-of-variance ratio.
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

/** The Weibull distribution with shape k > 0 and scale λ > 0 (scipy's weibull_min(k, scale = λ)). Support x ≥ 0. */
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

/** The Gumbel (maximum) distribution with location μ and scale β > 0 (scipy's gumbel_r). */
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
 * The generalised Pareto distribution with shape ξ, location μ and scale σ > 0 (scipy's genpareto(ξ, μ, σ)): the law of
 * exceedances over a high threshold (Pickands, 1975; Balkema and de Haan, 1974). With z = (x − μ)/σ the survival
 * function is (1 + ξz)^{−1/ξ}, and e^{−z} at ξ = 0 (the exponential). ξ > 0 gives a Pareto-like heavy tail, ξ < 0 a
 * bounded support up to μ − σ/ξ. The mean exists for ξ < 1, the variance for ξ < ½ (∞ beyond).
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

/** ∫_{−π}^{z} e^{κ(cos t − 1)} dt by composite Simpson's rule, with enough panels to resolve the peak's width 1/√κ. */
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

function vonMisesCdf(x: number, loc: number, kappa: number): number {
  const z = x - loc
  if (z <= -Math.PI) return 0
  if (z >= Math.PI) return 1
  return vonMisesIntegral(z, kappa) / vonMisesIntegral(Math.PI, kappa)
}

/**
 * Best and Fisher (1979), "Efficient simulation of the von Mises distribution", Applied Statistics 28(2): a wrapped
 * Cauchy envelope with acceptance rate above 65% for every κ. For κ below 1e-6 the distribution is uniform to double
 * precision (and the envelope's constants cancel), so draws are uniform.
 */
/** One uniform in [0, 1) (2 words of the stream). */
const unit = (s: Stream): number => units(s, 1)[0]

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
 * The von Mises distribution on the circle with mean direction `loc` and concentration κ ≥ 0 (scipy's
 * vonmises(κ, loc)); the density e^{κ cos(x − μ)} / (2π I₀(κ)) is periodic, and the cdf runs over [μ − π, μ + π]
 * (0 below, 1 above). `variance()` is the circular variance 1 − I₁(κ)/I₀(κ). The cdf (by quadrature) and quantile
 * (by bisection) are not differentiable; the density is, in x, μ and κ. No `rsample` (the draw is a rejection sampler).
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

/** The quantile of a truncated standard normal on [α, β], reflected into the lower tail when α > 0 for accuracy. */
function truncatedStandardQuantile(p: number, a: number, b: number): number {
  if (!(p >= 0 && p <= 1)) return NaN
  if (a > 0) return -truncatedStandardQuantile(1 - p, -b, -a)
  const fa = unwrap(normalCdf(a)) as number
  const fb = unwrap(normalCdf(b)) as number
  return Math.min(b, Math.max(a, unwrap(normalQuantile(fa + p * (fb - fa))) as number))
}

/**
 * The normal distribution N(loc, scale²) truncated to [low, high] (either may be infinite; scipy's
 * truncnorm((low − loc)/scale, (high − loc)/scale, loc, scale)). The normaliser Φ(β) − Φ(α) is computed in log space
 * without cancellation, so far-tail truncations (the TrueSkill and probit cases) stay accurate. The quantile (and the
 * sampler, by inversion) reflects into the lower tail; it is not differentiable, so there is no `rsample`.
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
