/**
 * Target transforms (plan §5.4): fit a regressor on $g(y)$, invert its point predictions, and push its predictive
 * distributions forward through $g^{-1}$ so the result is still a proper distribution on $y$ (a Gaussian on $\log y$
 * becomes a log-normal on $y$), after scikit-learn's `TransformedTargetRegressor` (Buitinck et al., 2013). Power maps
 * follow Box and Cox (1964), "An analysis of transformations", JRSS B 26, and Yeo and Johnson (2000), "A new family of
 * power transformations to improve normality or symmetry", Biometrika 87.
 *
 * A map is a `TargetMap`: a strictly monotone scalar function $z = g(y)$ with its inverse and its log-derivative, so
 * that a density on $z$ becomes one on $y$ by the change of variables
 * $\log p_y(y) = \log p_z(g(y)) + \log \lvert g'(y) \rvert$. Fixed maps (`logTarget`, `log1pTarget`, `affineTarget`)
 * are used as they are; map estimators (`standardTarget`, `powerTarget`) are fitted on the training targets.
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

/**
 * A tensor's values as a dense float64 array, in row-major order.
 *
 * @param t The tensor; not modified.
 * @returns Its values (possibly shared with the tensor; not to be written).
 */
const values = (t: Tensor): Float64Array => dense.data(t)

/**
 * A fitted, strictly monotone map of targets $z = g(y)$, applied elementwise. `logDerivative(y)` is
 * $\log \lvert g'(y) \rvert$, the log-Jacobian that turns a density on $z$ into one on $y$.
 */
export interface TargetMap {
  /**
   * A readable name: `log`, `log1p`, `affine`, `standard`, or `power(<method>, …)` with the fitted $\lambda$. A
   * map named `log` is recognised by `pushForward`.
   */
  readonly name: string
  /** $z = g(y)$. NaN (or infinite) outside the domain of $g$. */
  apply(y: number): number
  /** $y = g^{-1}(z)$. */
  invert(z: number): number
  /** $\log \lvert g'(y) \rvert$. */
  logDerivative(y: number): number
  /** True when $g$ is increasing (false: decreasing). */
  readonly increasing: boolean
}

/**
 * A target map fitted on the training targets, e.g. a standardisation or a power transform: a readable `name`, and
 * `fit(y)` returning the `TargetMap` fitted on the target vector `y`.
 */
export type TargetMapEstimator = { readonly name: string; fit(y: Tensor): TargetMap }

/**
 * $z = \log y$ for $y > 0$. A Gaussian predictive on $z$ becomes a log-normal on $y$ (see `pushForward`).
 *
 * @returns The map, with $\log \lvert g'(y) \rvert = -\log y$.
 *
 * @example The log map and its inverse
 * const g = logTarget()
 * print('g(e) =', g.apply(Math.E))
 * print('g^-1(0) =', g.invert(0))
 * print('log|g\'(2)| = -log 2 =', g.logDerivative(2))
 * print('g(-1) =', g.apply(-1))
 */
export function logTarget(): TargetMap {
  return { name: 'log', apply: Math.log, invert: Math.exp, logDerivative: (y) => -Math.log(y), increasing: true }
}

/**
 * $z = \log(1 + y)$ for $y > -1$, for counts and other non-negative targets with zeros.
 *
 * @returns The map, with $\log \lvert g'(y) \rvert = -\log(1 + y)$.
 *
 * @example Zero maps to zero, so counts with zeros are fine
 * const g = log1pTarget()
 * print('g(0), g(1), g(9) =', [0, 1, 9].map((y) => g.apply(y)))
 * print('g^-1(log 10) =', g.invert(Math.log(10)))
 */
export function log1pTarget(): TargetMap {
  return {
    name: 'log1p',
    apply: Math.log1p,
    invert: Math.expm1,
    logDerivative: (y) => -Math.log1p(y),
    increasing: true,
  }
}

/**
 * $z = (y - a) / b$ for shift $a$ and scale $b \ne 0$ (decreasing when $b < 0$). Throws `DomainError` for $b = 0$.
 *
 * @param shift The shift $a$ subtracted from $y$.
 * @param scale The scale $b$ the shifted $y$ is divided by; not zero.
 * @returns The map, with $\log \lvert g'(y) \rvert = -\log \lvert b \rvert$.
 *
 * @example A decreasing affine map
 * const g = affineTarget(10, -2)
 * print('g(14) =', g.apply(14))
 * print('g^-1(-2) =', g.invert(-2))
 * print('increasing:', g.increasing)
 * print('log|g\'| = -log 2 =', g.logDerivative(14))
 */
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

/**
 * Standardise the targets with their training mean $m$ and population standard deviation $s$ (divided by $n$; 1 when
 * the targets are all equal): $z = (y - m)/s$, an `affineTarget` named `standard`.
 *
 * @returns A map estimator whose `fit(y)` returns the fitted map.
 *
 * @example Standardise three targets
 * const g = standardTarget().fit(tensor([2, 4, 6]))
 * print('z =', [2, 4, 6].map((y) => g.apply(y)))
 * print('sqrt(8/3) =', Math.sqrt(8 / 3))
 * print('g^-1(1) =', g.invert(1))
 */
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
 * A Box–Cox or Yeo–Johnson power transform of the targets with $\lambda$ by maximum likelihood (`boxCoxLambda` and
 * `yeoJohnsonLambda` of `aifn-compute/probability/stats`), followed by standardisation of the transformed training
 * targets when `standardize`, as scikit-learn's `PowerTransformer`. Box–Cox is
 * $(y^\lambda - 1)/\lambda$ ($\log y$ at $\lambda = 0$) and needs $y > 0$; Yeo–Johnson extends it to every real $y$.
 * The fitted map carries its `lambda` beside the `TargetMap` fields.
 *
 * @param options The transform and whether to standardise.
 * @param options.method `'box-cox'` (positive targets) or `'yeo-johnson'` (any real targets).
 * @param options.standardize Standardise the transformed targets to mean 0 and population standard deviation 1.
 * @returns A map estimator whose `fit(y)` estimates $\lambda$ from `y` and returns the fitted map.
 *
 * @example Box–Cox on doubling targets is close to a log
 * const g = powerTarget({ standardize: false }).fit(tensor([1, 2, 4, 8, 16]))
 * print('name:', g.name)
 * print('lambda =', g.lambda)
 * print('z =', [1, 2, 4, 8, 16].map((y) => g.apply(y)))
 * print('round trip of 8:', g.invert(g.apply(8)))
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
 * A univariate distribution of $y = g^{-1}(z)$ for $z$ from `base`: the pushforward of `base` through the inverse
 * map, a contract `Distribution` whose functions return tensors. The entropy and the mode have no general closed form
 * and throw.
 */
export interface TransformedPredictive extends AnyUnivariate {
  /** The distribution of $z = g(y)$. */
  readonly base: AnyUnivariate
  /** The fitted target map $g$. */
  readonly map: TargetMap
  /** $\log p_y(y) = \log p_z(g(y)) + \log \lvert g'(y) \rvert$; $-\infty$ outside the domain of $g$. */
  logProb(y: Value): Tensor
  /** The density $p_y(y)$. */
  prob(y: Value): Tensor
  /** $\Pr(Y \le y)$, the base's CDF at $g(y)$ (its survival function for a decreasing $g$). */
  cdf(y: Value): Tensor
  /** The log of `cdf`. */
  logcdf(y: Value): Tensor
  /** $\Pr(Y > y)$. */
  survival(y: Value): Tensor
  /** The log of `survival`. */
  logSurvival(y: Value): Tensor
  /**
   * The $p$-quantile of $y$: $g^{-1}$ of the base's $p$-quantile (of its inverse survival function at $p$ when $g$
   * decreases).
   */
  quantile(p: Value): Tensor
  /** The inverse survival function: the $y$ with $\Pr(Y > y) = q$. */
  isf(q: Value): Tensor
  /** $\expect[y]$, by 32-point Gauss–Hermite quadrature on normal scores unless closed-form (log-normal). */
  mean(): Tensor
  /** $\var[y] = \expect[y^2] - \expect[y]^2$, by the same quadrature unless closed-form. */
  variance(): Tensor
  /** The square root of `variance`. */
  stddev(): Tensor
  /** $g^{-1}$ of the base's median. */
  median(): Tensor
}

/**
 * The image of the base's support under $g^{-1}$ (ends swapped for a decreasing $g$): an interval, open at an end
 * that comes from an infinite end of the base's support. A support that is not an interval or a range of integers
 * gives the whole real line.
 *
 * @param base The distribution of $z$.
 * @param g The target map.
 * @returns The support of $y$.
 */
function imageSupport(base: AnyUnivariate, g: TargetMap): Support {
  const s = base.support
  const [lo, hi] = s.type === 'interval' || s.type === 'integers' ? [s.lower, s.upper] : [-Infinity, Infinity]
  if (typeof lo !== 'number' || typeof hi !== 'number') return { type: 'real' }
  const a = g.invert(lo)
  const b = g.invert(hi)
  const [lower, upper] = g.increasing ? [a, b] : [b, a]
  return { type: 'interval', lower, upper, lowerOpen: !Number.isFinite(lo), upperOpen: !Number.isFinite(hi) }
}

/**
 * A value as a tensor (numbers become scalar tensors; traced values throw, as in `asTensor`).
 *
 * @param v A number or an untraced tensor.
 * @returns The value as a tensor.
 */
const tensorOf = (v: Value): Tensor => asTensor(v)

/**
 * A raw draw mapped elementwise.
 *
 * @param x A draw: a number or a tensor.
 * @param f The function applied to each value.
 * @returns The mapped draw, of the same kind and shape as `x`.
 */
function mapRaw(x: Raw, f: (v: number) => number): Raw {
  return isTensor(x) ? map(x, f) : f(x)
}

/**
 * The distribution of $y = g^{-1}(z)$ with $z$ drawn from `base` (univariate):
 * $\log p_y(y) = \log p_z(g(y)) + \log \lvert g'(y) \rvert$, the CDF and quantiles follow through $g$ (reversed for a
 * decreasing $g$), draws are $g^{-1}$ of the base's draws, and the mean and variance come from quadrature (see
 * `expectation` in `aifn-compute/learning/estimators`). This is the general case of `transformTarget`'s pushforward.
 * Unlike `aifn-compute/probability/distributions`' `Transformed` (a differentiable bijector, no moments), it takes a
 * scalar `TargetMap` and gives the mean and variance by quadrature. The entropy and the mode throw.
 *
 * @param base The distribution of $z$: univariate, with a CDF and a quantile function (and an inverse survival
 *   function when $g$ decreases). Its batch shape is the result's.
 * @param g The map $z = g(y)$.
 * @returns The distribution of $y$; its functions call the base's as they are used.
 *
 * @example The exponential of a uniform variable, checked by hand
 * // The uniform is written out by hand: its CDF and quantile are the identity and its log density is 0.
 * const uniform01 = {
 *   kind: 'distribution', name: 'Uniform', params: {}, batchShape: [], eventShape: [],
 *   support: { type: 'interval', lower: 0, upper: 1 }, discrete: false,
 *   logProb: (z) => mul(z, 0), cdf: (z) => z, quantile: (p) => p,
 * }
 * const y = transformedPredictive(uniform01, logTarget())
 * print('support:', y.support)
 * print('density at 2 (1/2):', y.prob(2))
 * print('median (e^0.5):', y.median(), Math.exp(0.5))
 * print('cdf at e^0.5:', y.cdf(Math.exp(0.5)))
 * print('mean (e - 1):', y.mean(), Math.E - 1)
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

/**
 * A batch of log-normal laws: $y = e^z$ with $z \sim \Gauss(\mu, \sigma^2)$, backed by
 * `aifn-compute/probability/distributions`' `LogNormal`.
 */
export interface LogNormalPredictive extends TransformedPredictive {
  /** Always `'LogNormal'`. */
  readonly name: 'LogNormal'
  /** The `LogNormal` object itself, for everything the pushforward interface does not list. */
  readonly distribution: Univariate<Tensor>
  /** The mode $e^{\mu - \sigma^2}$. */
  mode(): Tensor
  /** The entropy $\mu + \tfrac12 \log(2\pi e \sigma^2)$. */
  entropy(): Tensor
}

/**
 * The log-normal law of $y = e^z$ for a Gaussian base $\Gauss(\mu, \sigma^2)$ (`base.params.loc`,
 * `base.params.scale`), with closed forms from `aifn-compute/probability/distributions`' `LogNormal`:
 * $\expect[y] = e^{\mu + \sigma^2/2}$ and $\var[y] = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$. Only the base's
 * `loc` and `scale` are read (and its shape and support, by the general pushforward this overrides); the median is
 * still the base's median mapped through $e^z$. Throws `DomainError` when the base has no `loc` or `scale`.
 *
 * @param base A Gaussian distribution of $z = \log y$, with parameters `loc` ($\mu$) and `scale` ($\sigma$).
 * @returns The log-normal distribution of $y$.
 *
 * @example The moments of a standard log-normal
 * // Only loc and scale are read, so a standard normal can be given as plain data.
 * const z = {
 *   kind: 'distribution', name: 'Normal', params: { loc: 0, scale: 1 }, batchShape: [], eventShape: [],
 *   support: { type: 'real' }, discrete: false,
 * }
 * const y = logNormalPredictive(z)
 * print('mean (e^0.5):', y.mean(), Math.exp(0.5))
 * print('variance ((e - 1) e):', y.variance(), (Math.E - 1) * Math.E)
 * print('mode (e^-1):', y.mode(), Math.exp(-1))
 * print('cdf at 1:', y.cdf(1))
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
 * Push a predictive distribution on $z$ forward to $y = g^{-1}(z)$: a log-normal (`logNormalPredictive`) for a
 * `Normal` with `loc` and `scale` under a map named `log`, and the general transformed distribution
 * (`transformedPredictive`) for any other univariate law. Throws `DomainError` when `d` is not univariate with a CDF
 * and a quantile function.
 *
 * @param d The predictive distribution of $z$.
 * @param g The fitted map $z = g(y)$.
 * @returns The distribution of $y$.
 *
 * @example An affine map of a uniform, checked by hand
 * // z uniform on (0, 1), written out by hand; z = (y - 10) / 2 makes y uniform on (10, 12).
 * const uniform01 = {
 *   kind: 'distribution', name: 'Uniform', params: {}, batchShape: [], eventShape: [],
 *   support: { type: 'interval', lower: 0, upper: 1 }, discrete: false,
 *   logProb: (z) => mul(z, 0), cdf: (z) => z, quantile: (p) => p,
 * }
 * const y = pushForward(uniform01, affineTarget(10, 2))
 * print('name:', y.name)
 * print('support:', y.support)
 * print('density at 11 (1/2):', y.prob(11))
 * print('quartiles:', y.quantile(0.25), y.quantile(0.75))
 * print('mean (11):', y.mean())
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
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** Which composition made the model. */
  readonly composition: 'transform-target'
  /** The regressor fitted on $z = g(y)$. */
  readonly regressor: M
  /** The fitted map $g$. */
  readonly map: TargetMap
} & (M extends Fitted<infer X, infer H> ? Fitted<X, H> : unknown) &
  (M extends Decides<infer X, Tensor> ? Decides<X, Tensor> : unknown) &
  (M extends Predicts<infer X, Distribution>
    ? Predicts<X, TransformedPredictive> & Expects<X> & Samples<X, Tensor>
    : unknown)

/**
 * Fit `regressor` on transformed targets $z = g(y)$ and map its outputs back, as scikit-learn's
 * `TransformedTargetRegressor` for point predictions: `decide(x)` is $g^{-1}$ of the regressor's point prediction (for
 * a Gaussian on $\log y$, the median of $y$, not its mean). A predictive distribution is pushed forward through
 * $g^{-1}$ (`pushForward`), so `predictive(x)` is a proper distribution on $y$, `expect(x)` is
 * $\expect[y \mid x]$ under it and `sample` draws $y$. `forward` stays on the transformed scale. Fitting throws
 * `DomainError` when a target is outside the map's domain (its $g(y)$ is not finite).
 *
 * @param regressor The estimator fitted on the transformed targets; the fitting options are passed to it.
 * @param transform The map: a fixed `TargetMap`, or a `TargetMapEstimator` fitted on the training targets.
 * @returns An estimator whose fitted model has the regressor's `forward`, `decide` and `predictive` (with `expect` and
 *   `sample`) on the scale of $y$, and the fitted `regressor` and `map`.
 *
 * @example A line on log y is exponential growth on y
 * const line = {
 *   name: 'line',
 *   fit: (d) => {
 *     const xbar = mean(d.x)
 *     const zbar = mean(d.y)
 *     const xc = sub(d.x, xbar)
 *     const slope = sum(mul(xc, d.y)) / sum(mul(xc, xc))
 *     return { decide: (x) => add(mul(sub(x, xbar), slope), zbar) }
 *   },
 * }
 * const model = transformTarget(line, logTarget()).fit({ x: tensor([0, 1, 2]), y: tensor([1, 10, 100]) })
 * print('predictions at x = 3, 4:', model.decide(tensor([3, 4])))
 * print('on the log scale:', model.regressor.decide(tensor([3, 4])))
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
