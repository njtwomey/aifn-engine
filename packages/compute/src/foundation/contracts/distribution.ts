/**
 * Distributions, log-densities and bijectors (design S §2.5).
 *
 * Distributions are plain objects whose parameters are public fields and whose derived quantities are methods.
 * Parameters and arguments may be numbers, tensors or traced values. A result is a number when every input is a
 * number, a tensor when any is a tensor, and traced when any is traced (so log-densities are differentiable in both the
 * parameters and the value); `Kind<P>` computes that kind from the union of input types.
 *
 * `Distribution` is what every family has; `AnyUnivariate` and `AnyMultivariate` add the univariate functions and the
 * covariance, and their typed forms give result kinds. `aifn-compute/probability/distributions` implements them.
 */

import type { Kinded } from './kinds'
import type { Info } from './registry'
import type { Space } from './space'
import type { Raw, Scalar, Shape, Size, Tensor, Traced, Value, Vector, VectorLike } from './numbers'
import type { SampleOptions, Stream } from './random'

// ── Result kinds ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The kind of a result computed from inputs of types `P` (a union): `Value` when that is only known at run time,
 * `Traced` when any input is traced, a number when all are numbers, otherwise a tensor.
 */
export type Kind<P> = Value extends P
  ? Value
  : [Extract<P, Traced>] extends [never]
    ? [P] extends [number]
      ? number
      : Tensor
    : Traced

/**
 * The kind of a result that reduces an event axis (multivariate log-densities): a number for one event or a tensor for
 * a batch (known only at run time, so `Raw`), or traced when any input is traced.
 */
export type EventKind<P> = Value extends P ? Value : [Extract<P, Traced>] extends [never] ? Raw : Traced

/** The kind of a draw: a number when every parameter is a number, otherwise a tensor. */
export type SampleKind<P> = [P] extends [number] ? number : Tensor

// ── Support and structure ────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Where a distribution puts its mass. `real` is the real line; `interval` is $[\text{lower}, \text{upper}]$ (either
 * end may be infinite; `lower` and `upper` may be tensors for a batch); `integers` are the integers in
 * $[\text{lower}, \text{upper}]$; `simplex`, `real-vector`, `positive-definite` and `count-vector` (non-negative
 * integer vectors summing to `total`) are event spaces of multivariate distributions; `circle` is an interval of
 * length $2\pi$ on which the density is periodic.
 */
export type Support =
  /** The real line. */
  | { type: 'real' }
  /**
   * The interval from `lower` to `upper`. Finite ends are closed unless marked open by `lowerOpen` or `upperOpen` (as
   * `Transformed` marks the ends of an image, e.g. $(0, 1)$).
   */
  | { type: 'interval'; lower: Value; upper: Value; lowerOpen?: boolean; upperOpen?: boolean }
  /** The integers from `lower` to `upper` inclusive (either may be infinite). */
  | { type: 'integers'; lower: Value; upper: Value }
  /** The circle as the interval from `lower` to `upper`, of length $2\pi$, whose ends are identified. */
  | { type: 'circle'; lower: Value; upper: Value }
  /** Vectors of non-negative entries summing to 1. */
  | { type: 'simplex' }
  /** Real vectors. */
  | { type: 'real-vector' }
  /** Symmetric positive-definite matrices. */
  | { type: 'positive-definite' }
  /** Vectors of non-negative integers summing to `total`. */
  | { type: 'count-vector'; total: Value }

/**
 * Exponential-family structure: $\log p(x) = \sum_i \eta_i \cdot T_i(x) - A(\eta) + \log h(x)$, where each
 * $\eta_i \cdot T_i(x)$ is summed over the event axes. Used by `pgm` and `ep` for conjugate updates and message
 * passing.
 */
export interface ExponentialFamily {
  /** The natural parameters $\eta$, one entry per sufficient statistic, each with the batch (and event) shape. */
  naturalParams(): Value[]
  /** The sufficient statistics $T(x)$, in the order of `naturalParams`. */
  sufficientStats(x: Value): Value[]
  /** The log-partition function $A(\eta)$, with the batch shape. */
  logPartition(): Value
  /** $\log h(x)$, the base measure. */
  logBaseMeasure(x: Value): Value
}

// ── The protocol ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What every distribution has (`kind: 'distribution'`). Draws come from a plain-data `Stream`; `rsample`, a pathwise
 * draw written with primitives (differentiable in the parameters), exists where one does: location–scale families,
 * transformed distributions, the multivariate normal through its Cholesky factor. Moments are methods.
 */
export interface Distribution extends Kinded<'distribution'> {
  /** The family's name, e.g. `Normal`; the key of its registry entry, and what `kl` dispatches on. */
  readonly name: string
  /** The parameters as given (after conversion to the documented parameterisation), by name. */
  readonly params: Readonly<Record<string, Value>>
  /** The shape of a batch of independent distributions (the parameters' broadcast shape). */
  readonly batchShape: Shape
  /** The shape of one draw: `[]` for univariate distributions, `[d]` for vectors, `[d, d]` for matrices. */
  readonly eventShape: Shape
  /** Where the distribution puts its mass. */
  readonly support: Support
  /** True for distributions over integers (mass functions). */
  readonly discrete: boolean
  /** Exponential-family structure, for the members of that family. */
  readonly expFamily?: ExponentialFamily
  /**
   * Draws, from the stream only, of shape `[...shape, ...batchShape, ...eventShape]` (a number for an unbatched
   * univariate distribution without `shape`). Not differentiable: parameters must not be traced (use `rsample`).
   */
  sample(s: Stream, options?: SampleOptions): Raw
  /** Reparameterised draws of the same shape, differentiable in the parameters; absent when no pathwise draw exists. */
  rsample?(s: Stream, options?: SampleOptions): Value
  /** $\log p(x)$, reducing the event axes, broadcast against the batch. */
  logProb(x: Value): Value
  /** The density (continuous) or the mass (discrete). */
  prob(x: Value): Value
  /** The mean, with the batch and event shape (NaN where it is undefined). */
  mean(): Value
  /** The variance, elementwise, with the batch and event shape (NaN where it is undefined). */
  variance(): Value
  /** The standard deviation, the square root of `variance`. */
  stddev(): Value
  /** Differential entropy (continuous) or entropy (discrete), in nats. */
  entropy(): Value
  /** The most probable value, with the batch and event shape. */
  mode(): Value
}

/**
 * A univariate distribution (event shape `[]`) of any parameter kind. Every function of a value broadcasts the value
 * against the batch. Methods that have no closed form for a family (the entropy of a mixture) throw, and undefined
 * moments (the mean of a Cauchy) are NaN.
 */
export interface AnyUnivariate extends Distribution {
  /** A scalar event: the shape of one draw is `[]`. */
  readonly eventShape: readonly []
  /** $P(X \le x)$. */
  cdf(x: Value): Value
  /** $\log P(X \le x)$, accurate far in the lower tail. */
  logcdf(x: Value): Value
  /** $P(X > x)$, computed directly where that is more accurate than $1 - $ `cdf`. */
  survival(x: Value): Value
  /** $\log P(X > x)$, accurate far in the upper tail. */
  logSurvival(x: Value): Value
  /** The inverse cdf: the smallest $x$ with $F(x) \ge p$. */
  quantile(p: Value): Value
  /**
   * The inverse survival function: the $x$ with $S(x) = q$ (the smallest $x$ with $S(x) \le q$ when discrete).
   * Equal to `quantile(1 - q)`, but accurate for tiny $q$, where $1 - q$ rounds.
   */
  isf(q: Value): Value
}

/**
 * A univariate distribution whose parameters have types `P`: results have the kinds `Kind` computes. Each method is
 * that of `AnyUnivariate`; a function of a value is a number when the parameters and the value are numbers, a tensor
 * when any is a tensor, and traced when any is traced.
 */
export interface TypedUnivariate<P extends Value> extends AnyUnivariate {
  /** $\log p(x)$, of the kind of the parameters and `x`. */
  logProb<X extends Value>(x: X): Kind<P | X>
  /** The density or mass at `x`, of the kind of the parameters and `x`. */
  prob<X extends Value>(x: X): Kind<P | X>
  /** $P(X \le x)$, of the kind of the parameters and `x`. */
  cdf<X extends Value>(x: X): Kind<P | X>
  /** $\log P(X \le x)$, of the kind of the parameters and `x`. */
  logcdf<X extends Value>(x: X): Kind<P | X>
  /** $P(X > x)$, of the kind of the parameters and `x`. */
  survival<X extends Value>(x: X): Kind<P | X>
  /** $\log P(X > x)$, of the kind of the parameters and `x`. */
  logSurvival<X extends Value>(x: X): Kind<P | X>
  /** The quantile at probability `p`, of the kind of the parameters and `p`. */
  quantile<X extends Value>(p: X): Kind<P | X>
  /** The inverse survival function at `q`, of the kind of the parameters and `q`. */
  isf<X extends Value>(q: X): Kind<P | X>
  /** The mean, of the kind of the parameters. */
  mean(): Kind<P>
  /** The variance, of the kind of the parameters. */
  variance(): Kind<P>
  /** The standard deviation, of the kind of the parameters. */
  stddev(): Kind<P>
  /** The entropy in nats, of the kind of the parameters. */
  entropy(): Kind<P>
  /** The mode, of the kind of the parameters. */
  mode(): Kind<P>
  /** One draw per batch member: a number when every parameter is a number. */
  sample(s: Stream): SampleKind<P>
  /** Draws of shape `[...shape, ...batchShape]`, always a tensor. */
  sample(s: Stream, options: SampleOptions & { shape: Shape }): Tensor
  /** Draws, a number or a tensor as the options decide. */
  sample(s: Stream, options?: SampleOptions): Raw
}

/**
 * A univariate distribution. `Univariate` alone accepts any (its methods return `Value`); `Univariate<number>`, as
 * returned by `Normal(0, 1)`, returns numbers for numbers, tensors for tensors and traced values for traced ones.
 */
export type Univariate<P extends Value = Value> = Value extends P ? AnyUnivariate : TypedUnivariate<P>

/** A distribution over vectors or matrices, of any parameter kind. `logProb` reduces the event axes. */
export interface AnyMultivariate extends Distribution {
  /** The mean, with shape `[...batchShape, ...eventShape]`. */
  mean(): Value
  /** The elementwise variance, with shape `[...batchShape, ...eventShape]`. */
  variance(): Value
  /** The covariance matrix of a vector-valued distribution, `[...batchShape, d, d]`. */
  covariance(): Value
}

/**
 * A multivariate distribution whose parameters have types `P`. Each method is that of `AnyMultivariate`; moments are
 * tensors (traced when a parameter is), and functions of a value reduce the event axes (`EventKind`).
 */
export interface TypedMultivariate<P extends Value> extends AnyMultivariate {
  /** $\log p(x)$ per event: a number for one event, a tensor for a batch, traced when any input is. */
  logProb<X extends Value>(x: X): EventKind<P | X>
  /** The density or mass per event, of the kind `logProb` gives. */
  prob<X extends Value>(x: X): EventKind<P | X>
  /** The mean, a tensor of shape `[...batchShape, ...eventShape]` (traced when a parameter is). */
  mean(): Kind<P | Tensor>
  /** The elementwise variance, shaped as `mean`. */
  variance(): Kind<P | Tensor>
  /** The elementwise standard deviation, shaped as `mean`. */
  stddev(): Kind<P | Tensor>
  /** The covariance matrix, `[...batchShape, d, d]`. */
  covariance(): Kind<P | Tensor>
  /** The entropy in nats, one per batch member. */
  entropy(): EventKind<P>
  /** The mode, shaped as `mean`. */
  mode(): Kind<P | Tensor>
  /** Draws of shape `[...shape, ...batchShape, ...eventShape]`, always a tensor. */
  sample(s: Stream, options?: SampleOptions): Tensor
}

/** A multivariate distribution; `Multivariate` alone accepts any. */
export type Multivariate<P extends Value = Value> = Value extends P ? AnyMultivariate : TypedMultivariate<P>

/** A distribution on the wire: its registry family and parameters, rebuilt by `fromSpec`. */
export interface DistributionSpec {
  /** The registry key of the family, e.g. `Normal`. */
  readonly family: string
  /** The constructor's parameters by name, as numbers or flat arrays of numbers. */
  readonly params: Readonly<Record<string, number | readonly number[]>>
}

// ── Log-densities ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A log-density known up to a constant (`kind: 'log-density'`): what samplers, variational inference, diffusion toy
 * targets and mixtures produce or consume. `logDensity(theta)` is $\log \pi(\thetavec) + c$ for $\thetavec$ of
 * length `dim` ($-\infty$ outside the support), written with primitives so that `grad` works. `truth` gives samplers a
 * known answer for their diagnostics.
 */
export interface LogDensity extends Kinded<'log-density'> {
  /** A readable name, for display. */
  readonly name?: string
  /** The length of $\thetavec$. */
  readonly dim: Size
  /** $\log \pi(\thetavec) + c$. */
  logDensity(theta: Value): Value
  /** $\nabla \log \pi(\thetavec)$ in closed form, where one is cheaper than differentiating `logDensity`. */
  grad?(theta: Vector): VectorLike
  /** Where the density is positive, when it is not the whole of $\reals^d$. */
  readonly support?: Support
  /** True when the constant is zero ($\pi$ integrates to 1). */
  readonly normalised: boolean
  /** The known answer: the exact `mean` and covariance `cov`, or reference `samples`. */
  readonly truth?: { readonly mean?: Tensor; readonly cov?: Tensor; readonly samples?: Tensor }
}

// ── Bijectors ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * An interval of the real line, from `lower` to `upper`, each end open or closed as `lowerOpen` and `upperOpen` say.
 * Infinite ends are always open.
 */
export type Interval = { lower: Scalar; upper: Scalar; lowerOpen: boolean; upperOpen: boolean }

/**
 * A monotone bijection $y = f(x)$ from `domain` onto `codomain`, with its inverse and $\log \lvert f'(x) \rvert$. All
 * three are compositions of primitives, so a transformed log-density stays differentiable.
 */
export type Bijector = {
  /** A readable name, e.g. `exp`. */
  name: string
  /** $f(x)$, elementwise. */
  forward(x: Value): Value
  /** $f^{-1}(y)$, elementwise. */
  inverse(y: Value): Value
  /** $\log \lvert dy/dx \rvert$ at $x$. */
  logAbsDetJacobian(x: Value): Value
  /** True when $f$ is increasing (then cdfs map directly; otherwise the survival function is used). */
  increasing: boolean
  /** Where $f$ is defined. */
  domain: Interval
  /** The image of `domain` under $f$. */
  codomain: Interval
}

/**
 * Registry metadata of a named log density (a test density for samplers: banana, funnel, mixtures): its parameters,
 * its dimension (null when the dimension is a parameter) and whether its exact moments are given (`truth`).
 */
export interface LogDensityInfo extends Info {
  /** The entry kind of a named log density. */
  readonly kind: 'log-density'
  /** Its parameters, with their defaults. */
  readonly params: Space
  /** The length of $\thetavec$, or null when it is a parameter. */
  readonly dim: Size | null
  /** True when its log densities carry exact moments (`truth`). */
  readonly truth: boolean
}
