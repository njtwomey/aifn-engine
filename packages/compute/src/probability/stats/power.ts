/**
 * Box–Cox and Yeo–Johnson power transforms, with $\lambda$ chosen by maximum likelihood, as scipy's
 * `boxcox`/`yeojohnson` and `boxcox_normmax`. The column transformer built on them (scikit-learn's `PowerTransformer`)
 * is an application: `powerTransform` in `aifn-methods/learning`.
 *
 * The transforms act elementwise on a number or a tensor and switch to their logarithmic limits within machine epsilon
 * of $\lambda = 0$ (and $\lambda = 2$ for Yeo–Johnson's negative branch). The searches for $\lambda$ maximise a profile
 * log-likelihood by Brent's method from the bracket $(-2, 2)$.
 */

import type { Scalar } from 'aifn-compute/foundation/contracts'
import { EPS, map, type Tensor } from 'aifn-compute/foundation/tensor'
import { allValues, type Data } from './input'
import { varianceOf } from './descriptive'
import { minimizeScalar } from 'aifn-compute/numerics/roots'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Below this $\lvert \lambda \rvert$ (or $\lvert \lambda - 2 \rvert$ for Yeo–Johnson's negative branch) the
 * transforms switch to their logarithmic $\lambda \to 0$ (or $\lambda \to 2$) limits, as scikit-learn does.
 */
const TINY = EPS

/**
 * The Box–Cox transform of one value.
 *
 * @param x The value, positive.
 * @param lambda The power $\lambda$.
 * @returns $(x^\lambda - 1)/\lambda$, or $\log x$ near $\lambda = 0$.
 */
function boxCoxScalar(x: number, lambda: number): number {
  return Math.abs(lambda) < TINY ? Math.log(x) : (Math.pow(x, lambda) - 1) / lambda
}

/**
 * The inverse Box–Cox transform of one value.
 *
 * @param z The transformed value.
 * @param lambda The power $\lambda$ it was transformed with.
 * @returns $(\lambda z + 1)^{1/\lambda}$, or $e^z$ near $\lambda = 0$.
 */
function boxCoxInverseScalar(z: number, lambda: number): number {
  return Math.abs(lambda) < TINY ? Math.exp(z) : Math.pow(lambda * z + 1, 1 / lambda)
}

/**
 * The Yeo–Johnson transform of one value, by the branch for its sign.
 *
 * @param x The value, any real.
 * @param lambda The power $\lambda$.
 * @returns The transformed value.
 */
function yeoJohnsonScalar(x: number, lambda: number): number {
  if (x >= 0) return Math.abs(lambda) < TINY ? Math.log1p(x) : (Math.pow(x + 1, lambda) - 1) / lambda
  return Math.abs(lambda - 2) < TINY ? -Math.log1p(-x) : -(Math.pow(1 - x, 2 - lambda) - 1) / (2 - lambda)
}

/**
 * The inverse Yeo–Johnson transform of one value; the sign of $z$ is that of the original value, and picks the branch.
 *
 * @param z The transformed value.
 * @param lambda The power $\lambda$ it was transformed with.
 * @returns The original value.
 */
function yeoJohnsonInverseScalar(z: number, lambda: number): number {
  if (z >= 0) return Math.abs(lambda) < TINY ? Math.expm1(z) : Math.pow(z * lambda + 1, 1 / lambda) - 1
  return Math.abs(lambda - 2) < TINY ? -Math.expm1(-z) : 1 - Math.pow(-(2 - lambda) * z + 1, 1 / (2 - lambda))
}

/**
 * The Box–Cox transform $(x^\lambda - 1)/\lambda$ ($\log x$ at $\lambda = 0$) of positive $x$, elementwise (Box and
 * Cox, 1964, JRSS B 26). The data are not checked: a value that is not positive gives NaN or $-\infty$. As
 * `scipy.stats.boxcox(x, lmbda)`.
 *
 * @param x The value, or a tensor of values, all positive.
 * @param lambda The power $\lambda$.
 * @returns The transformed value, or a tensor of the shape of `x`.
 *
 * @example Three powers: a log, a square root and a shift
 * const x = tensor([1, 2, 4])
 * print('lambda = 0:', boxCox(x, 0))
 * print('lambda = 0.5:', boxCox(x, 0.5))
 * print('lambda = 1:', boxCox(x, 1))
 */
export function boxCox(x: number, lambda: number): number
export function boxCox(x: Tensor, lambda: number): Tensor
export function boxCox(x: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof x === 'number' ? boxCoxScalar(x, lambda) : map(x, (v) => boxCoxScalar(v, lambda))
}

/**
 * The inverse Box–Cox transform $(\lambda z + 1)^{1/\lambda}$ ($e^z$ at $\lambda = 0$), elementwise. Defined where
 * $\lambda z + 1 > 0$, which every value `boxCox` returns satisfies.
 *
 * @param z The transformed value, or a tensor of them.
 * @param lambda The power $\lambda$ they were transformed with.
 * @returns The original value, or a tensor of the shape of `z`.
 *
 * @example A round trip
 * const z = boxCox(tensor([1, 2, 4]), 0.5)
 * print('z =', z)
 * print('back =', boxCoxInverse(z, 0.5))
 */
export function boxCoxInverse(z: number, lambda: number): number
export function boxCoxInverse(z: Tensor, lambda: number): Tensor
export function boxCoxInverse(z: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof z === 'number' ? boxCoxInverseScalar(z, lambda) : map(z, (v) => boxCoxInverseScalar(v, lambda))
}

/**
 * The Yeo–Johnson transform of any real $x$, elementwise (Yeo and Johnson, 2000, "A new family of power transformations
 * to improve normality or symmetry", Biometrika 87): $((x + 1)^\lambda - 1)/\lambda$ for $x \ge 0$ and
 * $-((1 - x)^{2-\lambda} - 1)/(2 - \lambda)$ for $x < 0$, with logarithmic limits at $\lambda = 0$ and
 * $\lambda = 2$. As `scipy.stats.yeojohnson(x, lmbda)`.
 *
 * @param x The value, or a tensor of values, of any sign.
 * @param lambda The power $\lambda$.
 * @returns The transformed value, or a tensor of the shape of `x`.
 *
 * @example Negative, zero and positive values
 * print('lambda = 0.5:', yeoJohnson(tensor([-2, 0, 3]), 0.5))
 * print('lambda = 1 is the identity:', yeoJohnson(tensor([-2, 0, 3]), 1))
 */
export function yeoJohnson(x: number, lambda: number): number
export function yeoJohnson(x: Tensor, lambda: number): Tensor
export function yeoJohnson(x: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof x === 'number' ? yeoJohnsonScalar(x, lambda) : map(x, (v) => yeoJohnsonScalar(v, lambda))
}

/**
 * The inverse Yeo–Johnson transform, elementwise: the branch is chosen by the sign of $z$, which is that of the
 * original value.
 *
 * @param z The transformed value, or a tensor of them.
 * @param lambda The power $\lambda$ they were transformed with.
 * @returns The original value, or a tensor of the shape of `z`.
 *
 * @example A round trip
 * const z = yeoJohnson(tensor([-2, 0, 3]), 0.5)
 * print('z =', z)
 * print('back =', yeoJohnsonInverse(z, 0.5))
 */
export function yeoJohnsonInverse(z: number, lambda: number): number
export function yeoJohnsonInverse(z: Tensor, lambda: number): Tensor
export function yeoJohnsonInverse(z: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof z === 'number' ? yeoJohnsonInverseScalar(z, lambda) : map(z, (v) => yeoJohnsonInverseScalar(v, lambda))
}

// ── Maximum likelihood for λ ─────────────────────────────────────────────────────────────────────────────────────────

/** The result of a $\lambda$ search. */
export type PowerLambda = {
  /** The maximising $\lambda$. */
  lambda: number
  /** The profile log-likelihood at $\lambda$ (up to a constant). */
  logLikelihood: number
  /** The iterations Brent's method took. */
  iterations: number
  /** Whether Brent's method met its tolerance. */
  converged: boolean
}

/**
 * The Box–Cox $\lambda$ maximising the profile log-likelihood
 * $(\lambda - 1) \sum_i \log x_i - (n/2) \log \hat{\sigma}^2(\lambda)$, $\hat{\sigma}^2$ the population variance of
 * the transformed values (Box and Cox, 1964), by Brent's method from the bracket $(-2, 2)$ as scipy's
 * `boxcox_normmax(method='mle')`. Throws `DomainError` unless every value is positive, and for no values.
 *
 * @param x The data, all positive: an array, or a tensor of any rank (every element).
 * @returns The maximising `lambda`, the `logLikelihood` there (as `scipy.stats.boxcox_llf`), and Brent's `iterations`
 *   and `converged`.
 *
 * @example Right-skewed data want a log-like transform
 * const fit = boxCoxLambda([1, 2, 3, 4, 5, 10, 20, 50])
 * print('lambda =', fit.lambda)
 * print('log-likelihood =', fit.logLikelihood)
 * print('transformed =', boxCox(tensor([1, 2, 3, 4, 5, 10, 20, 50]), fit.lambda))
 */
export function boxCoxLambda(x: Data): PowerLambda {
  const v = Float64Array.from(allValues(x))
  if (v.some((a) => !(a > 0))) throw new DomainError('boxCoxLambda', 'boxCoxLambda: Box–Cox needs positive data')
  let sumLog = 0
  for (const a of v) sumLog += Math.log(a)
  const n = v.length
  const nll = (lambda: number) => {
    const s2 = varianceOf(
      Float64Array.from(v, (a) => boxCoxScalar(a, lambda)),
      {},
    )
    return -((lambda - 1) * sumLog - (n / 2) * Math.log(s2))
  }
  const r = minimizeScalar(nll, { bracket: [-2, 2] })
  return { lambda: r.x, logLikelihood: -r.value, iterations: r.steps, converged: r.converged }
}

/**
 * The Yeo–Johnson $\lambda$ maximising
 * $-(n/2) \log \hat{\sigma}^2(\lambda) + (\lambda - 1) \sum_i \sgn(x_i) \log(1 + \lvert x_i \rvert)$, by Brent's
 * method from the bracket $(-2, 2)$, as scikit-learn's `PowerTransformer`. Throws `DomainError` for no values.
 *
 * @param x The data, of any sign: an array, or a tensor of any rank (every element).
 * @returns The maximising `lambda`, the `logLikelihood` there (as `scipy.stats.yeojohnson_llf`), and Brent's
 *   `iterations` and `converged`.
 *
 * @example Data with negative values and a long right tail
 * const fit = yeoJohnsonLambda([-3, -1, 0, 1, 2, 5, 10, 30])
 * print('lambda =', fit.lambda)
 * print('log-likelihood =', fit.logLikelihood)
 */
export function yeoJohnsonLambda(x: Data): PowerLambda {
  const v = Float64Array.from(allValues(x))
  let jacobian = 0
  for (const a of v) jacobian += Math.sign(a) * Math.log1p(Math.abs(a))
  const n = v.length
  const nll = (lambda: number) => {
    const s2 = varianceOf(
      Float64Array.from(v, (a) => yeoJohnsonScalar(a, lambda)),
      {},
    )
    return s2 === 0 ? Infinity : (n / 2) * Math.log(s2) - (lambda - 1) * jacobian
  }
  const r = minimizeScalar(nll, { bracket: [-2, 2] })
  return { lambda: r.x, logLikelihood: -r.value, iterations: r.steps, converged: r.converged }
}
