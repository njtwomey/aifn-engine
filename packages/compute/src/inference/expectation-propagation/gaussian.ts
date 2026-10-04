/**
 * Gaussian and exponential-family message algebra for EP and message passing (Minka 2001, "A family of algorithms
 * for approximate Bayesian inference", PhD thesis, §3.1; Bishop 2006, §10.7). In natural parameters, multiplying two
 * members of a family adds their parameters, dividing subtracts them, raising to a power scales them, and damping is
 * a convex combination (a weighted geometric mean of the densities).
 *
 * A univariate Gaussian here is `{ precision: τ, shift: ν }` with τ = 1/σ² and ν = μ/σ², so N(x; μ, σ²) ∝
 * exp(−½ τ x² + ν x). Parameters may be numbers or tensors (a batch, elementwise with broadcasting). τ = ν = 0 is the
 * uniform "message" 1. A message may have negative precision (an improper site); only its product with others need
 * be proper.
 */

import type { Distribution } from 'aifn-compute/probability/distributions'
import {
  Beta,
  Bernoulli,
  Categorical,
  Dirichlet,
  Exponential,
  Gamma,
  Normal,
  Poisson,
} from 'aifn-compute/probability/distributions'
import { inverse } from 'aifn-compute/numerics/linalg'
import {
  add,
  div,
  exp,
  matmul,
  mul,
  neg,
  sqrt,
  sub,
  type Matrix,
  type Tensor,
  type Value,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A Gaussian in natural parameters: precision τ = 1/σ² and shift ν = μ/σ² (numbers, or tensors for a batch). */
export interface NaturalGaussian<T extends Value = number> {
  precision: T
  shift: T
}

/** Mean and variance of a Gaussian. */
export interface GaussianMoments<T extends Value = number> {
  mean: T
  variance: T
}

/** The uniform message: τ = ν = 0. */
export const UNIFORM_GAUSSIAN: NaturalGaussian = { precision: 0, shift: 0 }

/** Natural parameters of N(mean, variance): τ = 1/variance, ν = mean/variance. */
export function naturalGaussian<T extends Value>(mean: T, variance: T): NaturalGaussian<T> {
  return { precision: div(1, variance) as T, shift: div(mean, variance) as T }
}

/** Mean ν/τ and variance 1/τ of a Gaussian in natural parameters (NaN or negative for improper messages). */
export function gaussianMoments<T extends Value>(g: NaturalGaussian<T>): GaussianMoments<T> {
  return { mean: div(g.shift, g.precision) as T, variance: div(1, g.precision) as T }
}

/** The product of two Gaussian densities (up to a constant): parameters add. */
export function multiplyGaussians<T extends Value>(a: NaturalGaussian<T>, b: NaturalGaussian<T>): NaturalGaussian<T> {
  return { precision: add(a.precision, b.precision) as T, shift: add(a.shift, b.shift) as T }
}

/** The ratio a / b of two Gaussian densities (up to a constant), e.g. a cavity q / site: parameters subtract. */
export function divideGaussians<T extends Value>(a: NaturalGaussian<T>, b: NaturalGaussian<T>): NaturalGaussian<T> {
  return { precision: sub(a.precision, b.precision) as T, shift: sub(a.shift, b.shift) as T }
}

/** g^p: parameters scale by p (power EP's fractional sites). */
export function powerGaussian<T extends Value>(g: NaturalGaussian<T>, p: number): NaturalGaussian<T> {
  return { precision: mul(g.precision, p) as T, shift: mul(g.shift, p) as T }
}

/**
 * A damped update: (1 − λ) next + λ old in natural parameters, with λ ∈ [0, 1) the weight of the old message
 * (λ = 0: no damping).
 */
export function dampGaussian<T extends Value>(
  next: NaturalGaussian<T>,
  old: NaturalGaussian<T>,
  damping: number,
): NaturalGaussian<T> {
  if (damping === 0) return next
  return {
    precision: add(mul(next.precision, 1 - damping), mul(old.precision, damping)) as T,
    shift: add(mul(next.shift, 1 - damping), mul(old.shift, damping)) as T,
  }
}

/** A proper Gaussian as an `aifn-compute/probability/distributions` Normal (by mean and standard deviation). */
export function gaussianToNormal(g: NaturalGaussian<Value>): Distribution {
  const m = gaussianMoments(g)
  return Normal(m.mean, sqrt(m.variance))
}

/** Natural parameters of a Normal distribution object, from its exponential-family form (η₁ = ν, η₂ = −τ/2). */
export function normalToGaussian(d: Distribution): NaturalGaussian<Value> {
  if (d.name !== 'Normal' || !d.expFamily) throw new DomainError('normalToGaussian', 'normalToGaussian: needs a Normal')
  const [eta1, eta2] = d.expFamily.naturalParams()
  return { precision: mul(-2, eta2), shift: eta1 }
}

// ── Multivariate ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** A multivariate Gaussian in natural parameters: precision matrix Λ = Σ⁻¹ (d × d) and shift η = Λμ (d). */
export interface NaturalMvGaussian {
  precision: Matrix
  shift: Vector
}

/** Natural parameters of N(mean, covariance). */
export function naturalMvGaussian(mean: Vector, covariance: Matrix): NaturalMvGaussian {
  const precision = inverse(covariance)
  return { precision, shift: matmul(precision, mean) as Vector }
}

/** Mean Λ⁻¹η and covariance Λ⁻¹ of a multivariate Gaussian in natural parameters. */
export function mvGaussianMoments(g: NaturalMvGaussian): { mean: Vector; covariance: Matrix } {
  const covariance = inverse(g.precision)
  return { mean: matmul(covariance, g.shift) as Vector, covariance }
}

/** Product of multivariate Gaussians: Λ and η add. */
export function multiplyMvGaussians(a: NaturalMvGaussian, b: NaturalMvGaussian): NaturalMvGaussian {
  return { precision: add(a.precision, b.precision), shift: add(a.shift, b.shift) }
}

/** Ratio of multivariate Gaussians: Λ and η subtract. */
export function divideMvGaussians(a: NaturalMvGaussian, b: NaturalMvGaussian): NaturalMvGaussian {
  return { precision: sub(a.precision, b.precision), shift: sub(a.shift, b.shift) }
}

// ── Exponential-family messages ─────────────────────────────────────────────────────────────────────────────────────

/** A message in an exponential family: the family's name and its natural parameters (`aifn-compute/probability/distributions` order). */
export interface ExpFamilyMessage {
  family: string
  natural: Value[]
}

/** The natural parameters of an exponential-family distribution object. */
export function messageOf(d: Distribution): ExpFamilyMessage {
  if (!d.expFamily) throw new DomainError('messageOf', `messageOf: ${d.name} is not an exponential family`)
  return { family: d.name, natural: d.expFamily.naturalParams() }
}

function sameFamily(a: ExpFamilyMessage, b: ExpFamilyMessage): void {
  if (a.family !== b.family || a.natural.length !== b.natural.length)
    throw new DomainError('messages', `messages: ${a.family} and ${b.family} are not the same family`)
}

/** The product of two messages of one family: natural parameters add (the base measure is counted once). */
export function multiplyMessages(a: ExpFamilyMessage, b: ExpFamilyMessage): ExpFamilyMessage {
  sameFamily(a, b)
  return { family: a.family, natural: a.natural.map((x, i) => add(x, b.natural[i])) }
}

/** The ratio of two messages of one family: natural parameters subtract. */
export function divideMessages(a: ExpFamilyMessage, b: ExpFamilyMessage): ExpFamilyMessage {
  sameFamily(a, b)
  return { family: a.family, natural: a.natural.map((x, i) => sub(x, b.natural[i])) }
}

/** (1 − λ) next + λ old in natural parameters. */
export function dampMessages(next: ExpFamilyMessage, old: ExpFamilyMessage, damping: number): ExpFamilyMessage {
  sameFamily(next, old)
  return {
    family: next.family,
    natural: next.natural.map((x, i) => add(mul(x, 1 - damping), mul(old.natural[i], damping))),
  }
}

/** A message raised to a power p: natural parameters scale by p. */
export function powerMessage(m: ExpFamilyMessage, p: number): ExpFamilyMessage {
  return { family: m.family, natural: m.natural.map((x) => mul(x, p)) }
}

/**
 * The distribution object with these natural parameters, for Normal (η = (μ/σ², −1/(2σ²))), Gamma
 * ((shape − 1, −rate)), Beta ((a − 1, b − 1)), Dirichlet (α − 1), Bernoulli and Categorical (logits), Poisson
 * (log rate) and Exponential (−rate). Throws when the parameters are not a proper member (e.g. a negative variance).
 */
export function messageToDistribution(m: ExpFamilyMessage): Distribution {
  const [a, b] = m.natural
  switch (m.family) {
    case 'Normal': {
      const precision = mul(-2, b)
      return Normal(div(a, precision), sqrt(div(1, precision)))
    }
    case 'Gamma':
      return Gamma(add(a, 1), neg(b))
    case 'Beta':
      return Beta(add(a, 1), add(b, 1))
    case 'Dirichlet':
      return Dirichlet(add(a, 1) as Tensor)
    case 'Bernoulli':
      return Bernoulli({ logits: a })
    case 'Categorical':
      return Categorical({ logits: a })
    case 'Poisson':
      return Poisson(exp(a))
    case 'Exponential':
      return Exponential(neg(a))
    default:
      throw new DomainError('messageToDistribution', `messageToDistribution: ${m.family} is not supported`)
  }
}
