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
 * Where a distribution puts its mass. `real` is the real line; `interval` is [lower, upper] (either end may be
 * infinite; `lower` and `upper` may be tensors for a batch); `integers` are the integers in [lower, upper]; `simplex`,
 * `real-vector`, `positive-definite` and `count-vector` (non-negative integer vectors summing to n) are event spaces
 * of multivariate distributions; `circle` is an interval of length 2π on which the density is periodic.
 */
export type Support =
  | { type: 'real' }
  /** Finite ends are closed unless marked open (as `Transformed` marks the ends of an image, e.g. (0, 1)). */
  | { type: 'interval'; lower: Value; upper: Value; lowerOpen?: boolean; upperOpen?: boolean }
  | { type: 'integers'; lower: Value; upper: Value }
  | { type: 'circle'; lower: Value; upper: Value }
  | { type: 'simplex' }
  | { type: 'real-vector' }
  | { type: 'positive-definite' }
  | { type: 'count-vector'; total: Value }

/**
 * Exponential-family structure: log p(x) = Σᵢ ηᵢ · Tᵢ(x) − A(η) + log h(x), where each ηᵢ · Tᵢ(x) is summed over
 * the event axes. Used by `pgm` and `ep` for conjugate updates and message passing.
 */
export interface ExponentialFamily {
  /** The natural parameters η, one entry per sufficient statistic, each with the batch (and event) shape. */
  naturalParams(): Value[]
  /** The sufficient statistics T(x), in the order of `naturalParams`. */
  sufficientStats(x: Value): Value[]
  /** The log-partition function A(η), with the batch shape. */
  logPartition(): Value
  /** log h(x), the base measure. */
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
  /** log p(x), reducing the event axes, broadcast against the batch. */
  logProb(x: Value): Value
  /** The density (continuous) or the mass (discrete). */
  prob(x: Value): Value
  mean(): Value
  variance(): Value
  stddev(): Value
  /** Differential entropy (continuous) or entropy (discrete), in nats. */
  entropy(): Value
  mode(): Value
}

/**
 * A univariate distribution (event shape `[]`) of any parameter kind. Every function of a value broadcasts the value
 * against the batch. Methods that have no closed form for a family (the entropy of a mixture) throw, and undefined
 * moments (the mean of a Cauchy) are NaN.
 */
export interface AnyUnivariate extends Distribution {
  readonly eventShape: readonly []
  /** P(X ≤ x). */
  cdf(x: Value): Value
  logcdf(x: Value): Value
  /** P(X > x), computed directly where that is more accurate than 1 − cdf. */
  survival(x: Value): Value
  logSurvival(x: Value): Value
  /** The inverse cdf: the smallest x with cdf(x) ≥ p. */
  quantile(p: Value): Value
  /**
   * The inverse survival function: the x with survival(x) = q (the smallest x with survival(x) ≤ q when discrete).
   * Equal to quantile(1 − q), but accurate for tiny q, where 1 − q rounds.
   */
  isf(q: Value): Value
}

/** A univariate distribution whose parameters have types `P`: results have the kinds `Kind` computes. */
export interface TypedUnivariate<P extends Value> extends AnyUnivariate {
  logProb<X extends Value>(x: X): Kind<P | X>
  prob<X extends Value>(x: X): Kind<P | X>
  cdf<X extends Value>(x: X): Kind<P | X>
  logcdf<X extends Value>(x: X): Kind<P | X>
  survival<X extends Value>(x: X): Kind<P | X>
  logSurvival<X extends Value>(x: X): Kind<P | X>
  quantile<X extends Value>(p: X): Kind<P | X>
  isf<X extends Value>(q: X): Kind<P | X>
  mean(): Kind<P>
  variance(): Kind<P>
  stddev(): Kind<P>
  entropy(): Kind<P>
  mode(): Kind<P>
  sample(s: Stream): SampleKind<P>
  sample(s: Stream, options: SampleOptions & { shape: Shape }): Tensor
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

/** A multivariate distribution whose parameters have types `P`. */
export interface TypedMultivariate<P extends Value> extends AnyMultivariate {
  logProb<X extends Value>(x: X): EventKind<P | X>
  prob<X extends Value>(x: X): EventKind<P | X>
  mean(): Kind<P | Tensor>
  variance(): Kind<P | Tensor>
  stddev(): Kind<P | Tensor>
  covariance(): Kind<P | Tensor>
  entropy(): EventKind<P>
  mode(): Kind<P | Tensor>
  sample(s: Stream, options?: SampleOptions): Tensor
}

/** A multivariate distribution; `Multivariate` alone accepts any. */
export type Multivariate<P extends Value = Value> = Value extends P ? AnyMultivariate : TypedMultivariate<P>

/** A distribution on the wire: its registry family and parameters, rebuilt by `fromSpec`. */
export interface DistributionSpec {
  readonly family: string
  readonly params: Readonly<Record<string, number | readonly number[]>>
}

// ── Log-densities ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A log-density known up to a constant (`kind: 'log-density'`): what samplers, variational inference, diffusion toy
 * targets and mixtures produce or consume. `logDensity(θ)` is log π(θ) + const for θ of length `dim` (−Infinity outside
 * the support), written with primitives so that `grad` works. `truth` gives samplers a known answer for their
 * diagnostics.
 */
export interface LogDensity extends Kinded<'log-density'> {
  readonly name?: string
  /** The length of θ. */
  readonly dim: Size
  /** log π(θ) + const. */
  logDensity(theta: Value): Value
  /** ∇ log π(θ) in closed form, where one is cheaper than differentiating `logDensity`. */
  grad?(theta: Vector): VectorLike
  readonly support?: Support
  /** True when the constant is zero (π integrates to 1). */
  readonly normalised: boolean
  readonly truth?: { readonly mean?: Tensor; readonly cov?: Tensor; readonly samples?: Tensor }
}

// ── Bijectors ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** An interval of the real line. Infinite ends are always open. */
export type Interval = { lower: Scalar; upper: Scalar; lowerOpen: boolean; upperOpen: boolean }

/**
 * A monotone bijection y = f(x) from `domain` onto `codomain`, with its inverse and log |f′(x)|. All three are
 * compositions of primitives, so a transformed log-density stays differentiable.
 */
export type Bijector = {
  name: string
  forward(x: Value): Value
  inverse(y: Value): Value
  /** log |dy/dx| at x. */
  logAbsDetJacobian(x: Value): Value
  /** True when f is increasing (then cdfs map directly; otherwise the survival function is used). */
  increasing: boolean
  /** Where f is defined. */
  domain: Interval
  /** f(domain). */
  codomain: Interval
}

/**
 * Registry metadata of a named log density (a test density for samplers: banana, funnel, mixtures): its parameters,
 * its dimension (null when the dimension is a parameter) and whether its exact moments are given (`truth`).
 */
export interface LogDensityInfo extends Info {
  readonly kind: 'log-density'
  readonly params: Space
  readonly dim: Size | null
  readonly truth: boolean
}
