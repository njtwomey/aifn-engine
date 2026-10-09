/**
 * Gaussian and exponential-family message algebra for EP and message passing (Minka 2001, "A family of algorithms
 * for approximate Bayesian inference", PhD thesis, §3.1; Bishop 2006, §10.7). In natural parameters, multiplying two
 * members of a family adds their parameters, dividing subtracts them, raising to a power scales them, and damping is
 * a convex combination (a weighted geometric mean of the densities).
 *
 * A univariate Gaussian here is `{ precision, shift }`: $\tau = 1/\sigma^2$ and $\nu = \mu/\sigma^2$, so
 * $\Gauss(x; \mu, \sigma^2) \propto \exp(-\tfrac{1}{2} \tau x^2 + \nu x)$. Parameters may be numbers or tensors (a
 * batch, elementwise with broadcasting). $\tau = \nu = 0$ is the uniform "message" 1. A message may have negative
 * precision (an improper site); only its product with others need be proper. A multivariate Gaussian is
 * $\Lambdamat = \Sigmamat^{-1}$ and $\etavec = \Lambdamat\muvec$. Other exponential families are carried as an
 * `ExpFamilyMessage`, in the natural parameters of `aifn-compute/probability/distributions`.
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

/**
 * A Gaussian in natural parameters: precision $\tau = 1/\sigma^2$ and shift $\nu = \mu/\sigma^2$ (numbers, or tensors
 * for a batch).
 */
export interface NaturalGaussian<T extends Value = number> {
  /** The precision $\tau$; zero for a flat message, negative for an improper one. */
  precision: T
  /** The shift $\nu$, the precision times the mean. */
  shift: T
}

/** Mean and variance of a Gaussian. */
export interface GaussianMoments<T extends Value = number> {
  /** The mean $\mu$. */
  mean: T
  /** The variance $\sigma^2$. */
  variance: T
}

/** The uniform message: $\tau = \nu = 0$. */
export const UNIFORM_GAUSSIAN: NaturalGaussian = { precision: 0, shift: 0 }

/**
 * Natural parameters of $\Gauss(\mu, \sigma^2)$: $\tau = 1/\sigma^2$, $\nu = \mu/\sigma^2$. Differentiable, and
 * elementwise for tensors.
 *
 * @param mean The mean $\mu$ (a number, or a tensor for a batch).
 * @param variance The variance $\sigma^2$ (positive, not checked; broadcast against `mean`).
 * @returns The precision and shift.
 *
 * @example Round trip through the moments
 * const g = naturalGaussian(2, 4)
 * print('natural =', g)
 * print('moments =', gaussianMoments(g))
 *
 * @example A batch of Gaussians
 * print(naturalGaussian(tensor([0, 1, 2]), tensor([1, 0.5, 0.25])))
 */
export function naturalGaussian<T extends Value>(mean: T, variance: T): NaturalGaussian<T> {
  return { precision: div(1, variance) as T, shift: div(mean, variance) as T }
}

/**
 * Mean $\nu/\tau$ and variance $1/\tau$ of a Gaussian in natural parameters (infinite, NaN or negative for flat and
 * improper messages). Differentiable, and elementwise for tensors.
 *
 * @param g The Gaussian in natural parameters.
 * @returns Its mean and variance.
 *
 * @example A proper Gaussian and an improper site
 * print('proper:', gaussianMoments({ precision: 2, shift: 1 }))
 * print('improper:', gaussianMoments({ precision: -0.5, shift: 1 }))
 */
export function gaussianMoments<T extends Value>(g: NaturalGaussian<T>): GaussianMoments<T> {
  return { mean: div(g.shift, g.precision) as T, variance: div(1, g.precision) as T }
}

/**
 * The product of two Gaussian densities (up to a constant): parameters add. Differentiable, and elementwise for
 * tensors.
 *
 * @param a The first Gaussian, in natural parameters.
 * @param b The second Gaussian, in natural parameters.
 * @returns Their product, $\tau_a + \tau_b$ and $\nu_a + \nu_b$.
 *
 * @example A prior times a likelihood
 * // A prior N(0, 1) times the likelihood N(2; θ, 1) of one observation.
 * const posterior = multiplyGaussians(naturalGaussian(0, 1), naturalGaussian(2, 1))
 * print(gaussianMoments(posterior))
 */
export function multiplyGaussians<T extends Value>(a: NaturalGaussian<T>, b: NaturalGaussian<T>): NaturalGaussian<T> {
  return { precision: add(a.precision, b.precision) as T, shift: add(a.shift, b.shift) as T }
}

/**
 * The ratio $a / b$ of two Gaussian densities (up to a constant), e.g. a cavity $q / \tilde{f}$: parameters
 * subtract. The result may be improper. Differentiable, and elementwise for tensors.
 *
 * @param a The numerator, in natural parameters (in EP, the current approximation $q$).
 * @param b The denominator, in natural parameters (in EP, a site $\tilde{f}$).
 * @returns Their ratio, $\tau_a - \tau_b$ and $\nu_a - \nu_b$.
 *
 * @example The cavity of an EP site
 * const q = naturalGaussian(1, 0.5)
 * const site = naturalGaussian(2, 1)
 * const cavity = divideGaussians(q, site)
 * print('cavity =', gaussianMoments(cavity))
 * print('cavity times site =', gaussianMoments(multiplyGaussians(cavity, site)))
 */
export function divideGaussians<T extends Value>(a: NaturalGaussian<T>, b: NaturalGaussian<T>): NaturalGaussian<T> {
  return { precision: sub(a.precision, b.precision) as T, shift: sub(a.shift, b.shift) as T }
}

/**
 * $g^p$: parameters scale by $p$ (power EP's fractional sites). The mean is unchanged and the variance divided by
 * $p$. Differentiable, and elementwise for tensors.
 *
 * @param g The Gaussian, in natural parameters.
 * @param p The power (any real; 0 gives the uniform message).
 * @returns $g^p$.
 *
 * @example A square root widens a Gaussian
 * print(gaussianMoments(powerGaussian(naturalGaussian(3, 1), 0.5)))
 */
export function powerGaussian<T extends Value>(g: NaturalGaussian<T>, p: number): NaturalGaussian<T> {
  return { precision: mul(g.precision, p) as T, shift: mul(g.shift, p) as T }
}

/**
 * A damped update: $(1 - \lambda)\,\text{next} + \lambda\,\text{old}$ in natural parameters, with
 * $\lambda \in [0, 1)$ the weight of the old message ($\lambda = 0$: no damping, and `next` itself is returned).
 * Differentiable, and elementwise for tensors.
 *
 * @param next The newly computed message.
 * @param old The message it replaces.
 * @param damping The weight $\lambda$ of the old message (not checked).
 * @returns The damped message.
 *
 * @example Half-way between two messages
 * print(dampGaussian({ precision: 4, shift: 2 }, { precision: 2, shift: 0 }, 0.5))
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

/**
 * A proper Gaussian as an `aifn-compute/probability/distributions` Normal (by mean and standard deviation). Throws
 * `DomainError` from `Normal` when the precision is not positive.
 *
 * @param g The Gaussian in natural parameters, with a positive precision.
 * @returns The Normal distribution object.
 *
 * @example From natural parameters to a distribution
 * const d = gaussianToNormal({ precision: 4, shift: 4 })
 * print(d.name, d.params)
 * print('log p(1) =', d.logProb(1))
 */
export function gaussianToNormal(g: NaturalGaussian<Value>): Distribution {
  const m = gaussianMoments(g)
  return Normal(m.mean, sqrt(m.variance))
}

/**
 * Natural parameters of a Normal distribution object, from its exponential-family form ($\eta_1 = \nu$,
 * $\eta_2 = -\tau/2$). Throws `DomainError` for any other distribution.
 *
 * @param d A Normal distribution object.
 * @returns Its precision and shift.
 *
 * @example The round trip
 * print(normalToGaussian(gaussianToNormal(naturalGaussian(1, 0.25))))
 */
export function normalToGaussian(d: Distribution): NaturalGaussian<Value> {
  if (d.name !== 'Normal' || !d.expFamily) throw new DomainError('normalToGaussian', 'normalToGaussian: needs a Normal')
  const [eta1, eta2] = d.expFamily.naturalParams()
  return { precision: mul(-2, eta2), shift: eta1 }
}

// ── Multivariate ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A multivariate Gaussian in natural parameters: precision matrix $\Lambdamat = \Sigmamat^{-1}$ ($d \times d$) and
 * shift $\etavec = \Lambdamat\muvec$ (length $d$).
 */
export interface NaturalMvGaussian {
  /** The precision matrix $\Lambdamat$ ($d \times d$). */
  precision: Matrix
  /** The shift $\etavec = \Lambdamat\muvec$ (length $d$). */
  shift: Vector
}

/**
 * Natural parameters of $\Gauss(\muvec, \Sigmamat)$: $\Lambdamat = \Sigmamat^{-1}$ and $\etavec = \Lambdamat\muvec$.
 * Throws `LinAlgError` when the covariance is singular.
 *
 * @param mean The mean $\muvec$ (length $d$).
 * @param covariance The covariance $\Sigmamat$ ($d \times d$, invertible).
 * @returns The precision matrix and shift.
 *
 * @example A diagonal covariance
 * print(naturalMvGaussian(tensor([1, 2]), tensor([[2, 0], [0, 4]])))
 */
export function naturalMvGaussian(mean: Vector, covariance: Matrix): NaturalMvGaussian {
  const precision = inverse(covariance)
  return { precision, shift: matmul(precision, mean) as Vector }
}

/**
 * Mean $\Lambdamat^{-1}\etavec$ and covariance $\Lambdamat^{-1}$ of a multivariate Gaussian in natural parameters.
 * Throws `LinAlgError` when the precision is singular.
 *
 * @param g The Gaussian in natural parameters, with an invertible precision.
 * @returns Its mean and covariance.
 *
 * @example Round trip with naturalMvGaussian
 * const g = naturalMvGaussian(tensor([1, -1]), tensor([[2, 1], [1, 2]]))
 * print(mvGaussianMoments(g))
 */
export function mvGaussianMoments(g: NaturalMvGaussian): { mean: Vector; covariance: Matrix } {
  const covariance = inverse(g.precision)
  return { mean: matmul(covariance, g.shift) as Vector, covariance }
}

/**
 * Product of multivariate Gaussians: $\Lambdamat$ and $\etavec$ add.
 *
 * @param a The first Gaussian, in natural parameters.
 * @param b The second Gaussian, of the same dimension.
 * @returns Their product (up to a constant).
 *
 * @example Two independent pieces of evidence about a pair
 * // One Gaussian knows only the first coordinate (mean 1), the other only the second (mean 2).
 * const a = { precision: tensor([[1, 0], [0, 0]]), shift: tensor([1, 0]) }
 * const b = { precision: tensor([[0, 0], [0, 4]]), shift: tensor([0, 8]) }
 * print(mvGaussianMoments(multiplyMvGaussians(a, b)))
 */
export function multiplyMvGaussians(a: NaturalMvGaussian, b: NaturalMvGaussian): NaturalMvGaussian {
  return { precision: add(a.precision, b.precision), shift: add(a.shift, b.shift) }
}

/**
 * Ratio of multivariate Gaussians: $\Lambdamat$ and $\etavec$ subtract. The result may be improper.
 *
 * @param a The numerator, in natural parameters.
 * @param b The denominator, of the same dimension.
 * @returns Their ratio (up to a constant).
 *
 * @example Dividing out a factor recovers the other
 * const a = naturalMvGaussian(tensor([0, 0]), tensor([[1, 0], [0, 1]]))
 * const b = naturalMvGaussian(tensor([1, 1]), tensor([[2, 0], [0, 2]]))
 * print(divideMvGaussians(multiplyMvGaussians(a, b), b))
 */
export function divideMvGaussians(a: NaturalMvGaussian, b: NaturalMvGaussian): NaturalMvGaussian {
  return { precision: sub(a.precision, b.precision), shift: sub(a.shift, b.shift) }
}

// ── Exponential-family messages ─────────────────────────────────────────────────────────────────────────────────────

/**
 * A message in an exponential family: the family's name and its natural parameters (in the order of
 * `aifn-compute/probability/distributions`).
 */
export interface ExpFamilyMessage {
  /** The family's name, as the distribution objects give it (`'Normal'`, `'Gamma'`, `'Beta'`, ...). */
  family: string
  /** The natural parameters, as the family's `expFamily.naturalParams()` returns them. */
  natural: Value[]
}

/**
 * The natural parameters of an exponential-family distribution object. Throws `DomainError` when the distribution is
 * not an exponential family.
 *
 * @param d The distribution object.
 * @returns Its family name and natural parameters.
 *
 * @example A Beta(2, 3) as a message
 * const beta = messageToDistribution({ family: 'Beta', natural: [1, 2] })
 * print(beta.name, beta.params)
 * print(messageOf(beta))
 */
export function messageOf(d: Distribution): ExpFamilyMessage {
  if (!d.expFamily) throw new DomainError('messageOf', `messageOf: ${d.name} is not an exponential family`)
  return { family: d.name, natural: d.expFamily.naturalParams() }
}

/**
 * Throws `DomainError` unless two messages are of the same family and have as many natural parameters.
 *
 * @param a The first message.
 * @param b The second message.
 */
function sameFamily(a: ExpFamilyMessage, b: ExpFamilyMessage): void {
  if (a.family !== b.family || a.natural.length !== b.natural.length)
    throw new DomainError('messages', `messages: ${a.family} and ${b.family} are not the same family`)
}

/**
 * The product of two messages of one family: natural parameters add (the base measure is counted once). Throws
 * `DomainError` when the families differ.
 *
 * @param a The first message.
 * @param b The second message, of the same family.
 * @returns Their product (up to a constant).
 *
 * @example Conjugate updating of a Beta
 * // A Beta(1, 1) prior (natural parameters 0, 0) times the likelihood of 3 successes and 1 failure (3, 1).
 * const prior = { family: 'Beta', natural: [0, 0] }
 * const likelihood = { family: 'Beta', natural: [3, 1] }
 * const posterior = messageToDistribution(multiplyMessages(prior, likelihood))
 * print(posterior.name, posterior.params)
 */
export function multiplyMessages(a: ExpFamilyMessage, b: ExpFamilyMessage): ExpFamilyMessage {
  sameFamily(a, b)
  return { family: a.family, natural: a.natural.map((x, i) => add(x, b.natural[i])) }
}

/**
 * The ratio of two messages of one family: natural parameters subtract. Throws `DomainError` when the families
 * differ.
 *
 * @param a The numerator.
 * @param b The denominator, of the same family.
 * @returns Their ratio (up to a constant); possibly not a proper member.
 *
 * @example A Gamma cavity
 * const q = { family: 'Gamma', natural: [4, -2] }
 * const site = { family: 'Gamma', natural: [1, -0.5] }
 * const cavity = messageToDistribution(divideMessages(q, site))
 * print(cavity.name, cavity.params)
 */
export function divideMessages(a: ExpFamilyMessage, b: ExpFamilyMessage): ExpFamilyMessage {
  sameFamily(a, b)
  return { family: a.family, natural: a.natural.map((x, i) => sub(x, b.natural[i])) }
}

/**
 * A damped update $(1 - \lambda)\,\text{next} + \lambda\,\text{old}$ in natural parameters. Throws `DomainError`
 * when the families differ.
 *
 * @param next The newly computed message.
 * @param old The message it replaces, of the same family.
 * @param damping The weight $\lambda$ of the old message, in $[0, 1)$ (not checked).
 * @returns The damped message.
 *
 * @example A quarter of the way back
 * print(dampMessages({ family: 'Poisson', natural: [2] }, { family: 'Poisson', natural: [0] }, 0.25))
 */
export function dampMessages(next: ExpFamilyMessage, old: ExpFamilyMessage, damping: number): ExpFamilyMessage {
  sameFamily(next, old)
  return {
    family: next.family,
    natural: next.natural.map((x, i) => add(mul(x, 1 - damping), mul(old.natural[i], damping))),
  }
}

/**
 * A message raised to a power $p$: natural parameters scale by $p$.
 *
 * @param m The message.
 * @param p The power.
 * @returns $m^p$.
 *
 * @example Half of a Beta likelihood
 * const half = powerMessage({ family: 'Beta', natural: [4, 2] }, 0.5)
 * print(half)
 * print(messageToDistribution(half).params)
 */
export function powerMessage(m: ExpFamilyMessage, p: number): ExpFamilyMessage {
  return { family: m.family, natural: m.natural.map((x) => mul(x, p)) }
}

/**
 * The distribution object with these natural parameters, for Normal ($\etavec = (\mu/\sigma^2, -1/(2\sigma^2))$),
 * Gamma ($(\text{shape} - 1, -\text{rate})$), Beta ($(a - 1, b - 1)$), Dirichlet ($\alphavec - 1$), Bernoulli and
 * Categorical (logits), Poisson (log rate) and Exponential ($-\text{rate}$). Throws `DomainError` when the parameters
 * are not a proper member (e.g. a negative variance) or the family is not one of these.
 *
 * @param m The message: a supported family and its natural parameters.
 * @returns The distribution object.
 *
 * @example A Normal and a Poisson from natural parameters
 * const n = messageToDistribution({ family: 'Normal', natural: [2, -0.5] })
 * print(n.name, 'mean', n.mean(), 'variance', n.variance())
 * print(messageToDistribution({ family: 'Poisson', natural: [Math.log(3)] }).params)
 *
 * @example An improper message is refused
 * try {
 *   messageToDistribution({ family: 'Normal', natural: [0, 0.5] })
 * } catch (e) {
 *   print(e.message)
 * }
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
