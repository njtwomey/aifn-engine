/**
 * Target transforms (plan §5.4): fit a regressor on g(y), invert its point predictions, and push its predictive
 * distributions forward through g⁻¹ so the result is still a proper distribution on y (a Gaussian on log y becomes a
 * log-normal on y), after scikit-learn's `TransformedTargetRegressor` (Buitinck et al., 2013). Power maps follow Box and
 * Cox (1964), "An analysis of transformations", JRSS B 26, and Yeo and Johnson (2000), "A new family of power
 * transformations to improve normality or symmetry", Biometrika 87.
 */

import type { AnyUnivariate, Distribution, Raw, Support } from 'aifn-compute/foundation/contracts'
import type { Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, isTensor, map, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import {
  asTensor,
  expectation,
  isUnivariate,
  type Decides,
  type Estimator,
  type Expects,
  type Features,
  type FitOptions,
  type Fitted,
  type Predicts,
  type Samples,
  type Supervised,
} from 'aifn-compute/learning/estimators'
import { LogNormal, type Univariate } from 'aifn-compute/probability/distributions'
import { boxCoxLambda, yeoJohnsonLambda } from 'aifn-compute/probability/stats'
import { DomainError } from 'aifn-compute/foundation/errors'

const values = (t: Tensor): Float64Array => dense.data(t)

/**
 * A fitted, strictly monotone map of targets z = g(y), applied elementwise. `logDerivative(y)` is log |g′(y)|, the
 * log-Jacobian that turns a density on z into one on y.
 */
export interface TargetMap {
  /** A readable name: `log`, `log1p`, `affine`, `standard`, `power(box-cox, λ=…)`. */
  readonly name: string
  /** z = g(y). NaN outside g's domain. */
  apply(y: number): number
  /** y = g⁻¹(z). */
  invert(z: number): number
  /** log |g′(y)|. */
  logDerivative(y: number): number
  /** True when g is increasing (false: decreasing). */
  readonly increasing: boolean
}

/** A target map fitted on the training targets, e.g. a standardisation or a power transform. */
export type TargetMapEstimator = { readonly name: string; fit(y: Tensor): TargetMap }

/** z = log y (y > 0). A Gaussian predictive on z becomes a log-normal on y. */
export function logTarget(): TargetMap {
  return { name: 'log', apply: Math.log, invert: Math.exp, logDerivative: (y) => -Math.log(y), increasing: true }
}

/** z = log(1 + y) (y > −1), for counts and other non-negative targets with zeros. */
export function log1pTarget(): TargetMap {
  return {
    name: 'log1p',
    apply: Math.log1p,
    invert: Math.expm1,
    logDerivative: (y) => -Math.log1p(y),
    increasing: true,
  }
}

/** z = (y − shift) / scale, with scale ≠ 0 (decreasing when negative). */
export function affineTarget(shift: number, scale: number): TargetMap {
  if (scale === 0) throw new DomainError('affineTarget', 'affineTarget: scale must be non-zero')
  return {
    name: 'affine',
    apply: (y) => (y - shift) / scale,
    invert: (z) => z * scale + shift,
    logDerivative: () => -Math.log(Math.abs(scale)),
    increasing: scale > 0,
  }
}

/** Standardise the targets with their training mean and population standard deviation. */
export function standardTarget(): TargetMapEstimator {
  return {
    name: 'standard-target',
    fit(y) {
      const v = values(y)
      let m = 0
      for (const a of v) m += a / v.length
      let s = 0
      for (const a of v) s += (a - m) ** 2 / v.length
      return { ...affineTarget(m, Math.sqrt(s) || 1), name: 'standard' }
    },
  }
}

/**
 * A Box–Cox or Yeo–Johnson power transform of the targets with λ by maximum likelihood (from `aifn-compute/probability/stats`),
 * followed by standardisation when `standardize` (default true), as scikit-learn's `PowerTransformer`.
 */
export function powerTarget({
  method = 'box-cox',
  standardize = true,
}: { method?: 'box-cox' | 'yeo-johnson'; standardize?: boolean } = {}): TargetMapEstimator & { method: string } {
  return {
    name: `power-target(${method})`,
    method,
    fit(y) {
      const v = values(y)
      const lambda = (method === 'box-cox' ? boxCoxLambda(v) : yeoJohnsonLambda(v)).lambda
      const tiny = Number.EPSILON
      const raw =
        method === 'box-cox'
          ? {
              apply: (a: number) => (Math.abs(lambda) < tiny ? Math.log(a) : (a ** lambda - 1) / lambda),
              invert: (z: number) => (Math.abs(lambda) < tiny ? Math.exp(z) : (lambda * z + 1) ** (1 / lambda)),
              // d/dy (y^λ − 1)/λ = y^{λ−1}.
              logDerivative: (a: number) => (lambda - 1) * Math.log(a),
            }
          : {
              apply: (a: number) =>
                a >= 0
                  ? Math.abs(lambda) < tiny
                    ? Math.log1p(a)
                    : ((a + 1) ** lambda - 1) / lambda
                  : Math.abs(lambda - 2) < tiny
                    ? -Math.log1p(-a)
                    : -((1 - a) ** (2 - lambda) - 1) / (2 - lambda),
              invert: (z: number) =>
                z >= 0
                  ? Math.abs(lambda) < tiny
                    ? Math.expm1(z)
                    : (z * lambda + 1) ** (1 / lambda) - 1
                  : Math.abs(lambda - 2) < tiny
                    ? -Math.expm1(-z)
                    : 1 - (-(2 - lambda) * z + 1) ** (1 / (2 - lambda)),
              // d/dy = (1 + |y|)^{sign(y)(λ−1)}: (y + 1)^{λ−1} for y ≥ 0 and (1 − y)^{1−λ} for y < 0.
              logDerivative: (a: number) => (a >= 0 ? lambda - 1 : 1 - lambda) * Math.log1p(Math.abs(a)),
            }
      let m = 0
      let s = 1
      if (standardize) {
        const t = Float64Array.from(v, raw.apply)
        m = t.reduce((a, b) => a + b, 0) / t.length
        s = Math.sqrt(t.reduce((a, b) => a + (b - m) ** 2, 0) / t.length) || 1
      }
      return {
        name: `power(${method}, λ=${lambda.toPrecision(4)})`,
        lambda,
        apply: (a) => (raw.apply(a) - m) / s,
        invert: (z) => raw.invert(z * s + m),
        logDerivative: (a) => raw.logDerivative(a) - Math.log(s),
        increasing: true,
      } as TargetMap & { lambda: number }
    },
  }
}

// ── Pushed-forward distributions ─────────────────────────────────────────────────────────────────────────────────────

/**
 * A univariate distribution of y = g⁻¹(z) for z from `base`: the pushforward of `base` through the inverse map, a
 * contract `Distribution` whose functions return tensors. The entropy and the mode have no general closed form and
 * throw.
 */
export interface TransformedPredictive extends AnyUnivariate {
  /** The distribution of z = g(y). */
  readonly base: AnyUnivariate
  /** The fitted target map g. */
  readonly map: TargetMap
  logProb(y: Value): Tensor
  prob(y: Value): Tensor
  cdf(y: Value): Tensor
  logcdf(y: Value): Tensor
  survival(y: Value): Tensor
  logSurvival(y: Value): Tensor
  quantile(p: Value): Tensor
  isf(q: Value): Tensor
  /** E[y], by 32-point Gauss–Hermite quadrature on normal scores unless closed-form (log-normal). */
  mean(): Tensor
  variance(): Tensor
  stddev(): Tensor
  /** g⁻¹ of the base's median. */
  median(): Tensor
}

/** The image of the base's support under g⁻¹ (ends swapped for a decreasing g). */
function imageSupport(base: AnyUnivariate, g: TargetMap): Support {
  const s = base.support
  const [lo, hi] = s.type === 'interval' || s.type === 'integers' ? [s.lower, s.upper] : [-Infinity, Infinity]
  if (typeof lo !== 'number' || typeof hi !== 'number') return { type: 'real' }
  const a = g.invert(lo)
  const b = g.invert(hi)
  const [lower, upper] = g.increasing ? [a, b] : [b, a]
  return { type: 'interval', lower, upper, lowerOpen: !Number.isFinite(lo), upperOpen: !Number.isFinite(hi) }
}

/** A value as a tensor (numbers become scalar tensors; traced values throw, as in `asTensor`). */
const tensorOf = (v: Value): Tensor => asTensor(v)

/** A raw draw mapped elementwise. */
function mapRaw(x: Raw, f: (v: number) => number): Raw {
  return isTensor(x) ? map(x, f) : f(x)
}

/**
 * The distribution of y = g⁻¹(z) with z ~ `base` (univariate): log p(y) = log p_z(g(y)) + log |g′(y)|, the CDF and
 * quantiles follow through g (reversed for a decreasing g), draws are g⁻¹ of the base's draws, and the mean and
 * variance come from quadrature (see `expectation` in `aifn-compute/learning/estimators`). This is the general case of
 * `transformTarget`'s pushforward. Unlike `aifn-compute/probability/distributions`' `Transformed` (a differentiable bijector,
 * no moments), it takes a scalar `TargetMap` and gives the mean and variance by quadrature.
 */
export function transformedPredictive(base: AnyUnivariate, g: TargetMap): TransformedPredictive {
  const shape = base.batchShape
  const baseAt = (y: Value) => map(tensorOf(y), (v) => g.apply(v))
  const lower = (y: Value) => asTensor(g.increasing ? base.cdf(baseAt(y)) : base.survival(baseAt(y)))
  const d: TransformedPredictive = {
    kind: 'distribution',
    name: 'Transformed',
    params: {},
    base,
    map: g,
    batchShape: shape,
    eventShape: [],
    support: imageSupport(base, g),
    discrete: base.discrete,
    logProb(y) {
      const lp = values(asTensor(base.logProb(baseAt(y))))
      const yv = values(tensorOf(y))
      return fromData(
        Float64Array.from(lp, (l, i) => {
          const yi = yv.length === 1 ? yv[0] : yv[i]
          const z = g.apply(yi)
          return Number.isNaN(z) ? -Infinity : l + g.logDerivative(yi)
        }),
        lp.length === 1 && shape.length === 0 ? [] : shape,
      )
    },
    prob: (y) => map(d.logProb(y), Math.exp),
    cdf: lower,
    logcdf: (y) => asTensor(g.increasing ? base.logcdf(baseAt(y)) : base.logSurvival(baseAt(y))),
    survival: (y) => asTensor(g.increasing ? base.survival(baseAt(y)) : base.cdf(baseAt(y))),
    logSurvival: (y) => asTensor(g.increasing ? base.logSurvival(baseAt(y)) : base.logcdf(baseAt(y))),
    // A decreasing g swaps the tails: the p-quantile of y is g⁻¹ of the base's inverse survival function at p.
    quantile: (p) => map(asTensor(g.increasing ? base.quantile(p) : base.isf(p)), (v) => g.invert(v)),
    isf: (q) => map(asTensor(g.increasing ? base.isf(q) : base.quantile(q)), (v) => g.invert(v)),
    sample: (s: Stream, options) => mapRaw(base.sample(s, options), (v) => g.invert(v)),
    mean: () => expectation(d, (y) => y),
    variance() {
      const m = values(expectation(d, (y) => y))
      const m2 = values(expectation(d, (y) => y * y))
      return fromData(
        Float64Array.from(m, (a, i) => m2[i] - a * a),
        shape,
      )
    },
    stddev: () => map(d.variance(), Math.sqrt),
    entropy() {
      throw new Error('transformedPredictive: the entropy has no closed form')
    },
    mode() {
      throw new Error('transformedPredictive: the mode has no closed form')
    },
    median: () => map(asTensor(base.quantile(0.5)), (v) => g.invert(v)),
  }
  return d
}

/** A batch of log-normal laws: y = exp(z), z ~ N(μ, σ²), backed by `aifn-compute/probability/distributions`' `LogNormal`. */
export interface LogNormalPredictive extends TransformedPredictive {
  readonly name: 'LogNormal'
  /** The `LogNormal(μ, σ)` object itself, for everything the pushforward interface does not list. */
  readonly distribution: Univariate<Tensor>
  /** exp(μ − σ²). */
  mode(): Tensor
  entropy(): Tensor
}

/**
 * The log-normal law of y = exp(z) for a Gaussian base N(μ, σ²) (`base.params.loc`, `base.params.scale`), with closed
 * forms from `aifn-compute/probability/distributions`' `LogNormal`: E[y] = exp(μ + σ²/2), Var[y] = (exp(σ²) − 1) exp(2μ + σ²).
 */
export function logNormalPredictive(base: AnyUnivariate): LogNormalPredictive {
  const { loc, scale } = base.params
  if (loc === undefined || scale === undefined)
    throw new DomainError('logNormalPredictive', 'logNormalPredictive: the base needs loc and scale')
  const ln = LogNormal(asTensor(loc), asTensor(scale))
  return {
    ...transformedPredictive(base, logTarget()),
    name: 'LogNormal',
    params: ln.params,
    distribution: ln,
    logProb: (y) => asTensor(ln.logProb(y)),
    prob: (y) => asTensor(ln.prob(y)),
    cdf: (y) => asTensor(ln.cdf(y)),
    logcdf: (y) => asTensor(ln.logcdf(y)),
    survival: (y) => asTensor(ln.survival(y)),
    logSurvival: (y) => asTensor(ln.logSurvival(y)),
    quantile: (p) => asTensor(ln.quantile(p)),
    isf: (q) => asTensor(ln.isf(q)),
    sample: (s: Stream, options) => ln.sample(s, options),
    mean: () => asTensor(ln.mean()),
    variance: () => asTensor(ln.variance()),
    stddev: () => asTensor(ln.stddev()),
    entropy: () => asTensor(ln.entropy()),
    mode: () => asTensor(ln.mode()),
  }
}

/**
 * Push a predictive distribution on z forward to y = g⁻¹(z): a log-normal for a Gaussian under `logTarget`, the general
 * transformed distribution for any other univariate law.
 */
export function pushForward(d: Distribution, g: TargetMap): TransformedPredictive {
  if (!isUnivariate(d))
    throw new DomainError('pushForward', `pushForward: ${d.name} is not univariate with a quantile function`)
  if (g.name === 'log' && d.name === 'Normal' && d.params.loc !== undefined && d.params.scale !== undefined) {
    return logNormalPredictive(d)
  }
  return transformedPredictive(d, g)
}

// ── transformTarget ──────────────────────────────────────────────────────────────────────────────────────────────────

/** The fitted target-transformed model: the regressor and the map, and the regressor's capabilities on y's scale. */
export type TargetModel<M> = {
  readonly kind: 'model'
  /** Which composition made the model. */
  readonly composition: 'transform-target'
  /** The regressor fitted on z = g(y). */
  readonly regressor: M
  /** The fitted map g. */
  readonly map: TargetMap
} & (M extends Fitted<infer X, infer H> ? Fitted<X, H> : unknown) &
  (M extends Decides<infer X, Tensor> ? Decides<X, Tensor> : unknown) &
  (M extends Predicts<infer X, Distribution>
    ? Predicts<X, TransformedPredictive> & Expects<X> & Samples<X, Tensor>
    : unknown)

/**
 * Fit `regressor` on transformed targets z = g(y) and map its outputs back, as scikit-learn's
 * `TransformedTargetRegressor` for point predictions: `decide(x)` is g⁻¹ of the regressor's point prediction (for a
 * Gaussian on log y, the median of y, not its mean). A predictive distribution is pushed forward through g⁻¹
 * (`pushForward`), so `predictive(x)` is a proper distribution on y, `expect(x)` is E[y | x] under it and `sample`
 * draws y. `forward` stays on the transformed scale. `transform` is a fixed `TargetMap` or an estimator fitted on y.
 *
 * @example transformTarget(linearRegression(), logTarget()).fit(dataset(x, y)).predictive(x) // log-normal
 */
export function transformTarget<X extends Features, M extends object>(
  regressor: Estimator<Supervised<X, Tensor>, M>,
  transform: TargetMap | TargetMapEstimator,
): Estimator<Supervised<X, Tensor>, TargetModel<M>> {
  return {
    name: `transform-target(${regressor.name})`,
    fit(data, options: FitOptions = {}) {
      const g = 'fit' in transform ? transform.fit(data.y) : transform
      const y = values(data.y)
      const z = Float64Array.from(y, (v) => g.apply(v))
      const bad = z.findIndex((v) => !Number.isFinite(v))
      if (bad >= 0)
        throw new DomainError(
          'transformTarget',
          `transformTarget: target ${y[bad]} is outside the domain of the ${g.name} map`,
        )
      const inner = regressor.fit({ ...data, y: fromData(z, data.y.shape) }, options) as M & Record<string, unknown>
      const out: Record<string, unknown> = { kind: 'model', composition: 'transform-target', regressor: inner, map: g }
      if (typeof inner.forward === 'function') out.forward = (x: X) => (inner.forward as (x: X) => Tensor)(x)
      if (typeof inner.decide === 'function') {
        out.decide = (x: X) => map((inner.decide as (x: X) => Tensor)(x), (v) => g.invert(v))
      }
      if (typeof inner.predictive === 'function') {
        const predictive = (x: X) => pushForward((inner.predictive as (x: X) => Distribution)(x), g)
        out.predictive = predictive
        out.expect = (x: X, f?: (y: number) => number) => expectation(predictive(x), f)
        out.sample = (s: Stream, x: X, n?: number) =>
          asTensor(predictive(x).sample(s, { shape: n === undefined ? [] : [n] }))
      }
      return out as TargetModel<M>
    },
  }
}
