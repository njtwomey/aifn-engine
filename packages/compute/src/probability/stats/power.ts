/**
 * Box–Cox and Yeo–Johnson power transforms, with λ chosen by maximum likelihood, as scipy's `boxcox`/`yeojohnson` and
 * `boxcox_normmax`. The column transformer built on them (scikit-learn's `PowerTransformer`) is an application:
 * `powerTransform` in `aifn-methods/learning`.
 */

import type { Scalar } from 'aifn-compute/foundation/contracts'
import { EPS, map, type Tensor } from 'aifn-compute/foundation/tensor'
import { allValues, type Data } from './input'
import { varianceOf } from './descriptive'
import { minimizeScalar } from 'aifn-compute/numerics/roots'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Below this |λ| (or |λ − 2| for Yeo–Johnson's negative branch) the logarithmic limit is used, as scikit-learn does. */
// Below this |λ| the transforms switch to their λ → 0 (or λ → 2) limits.
const TINY = EPS

function boxCoxScalar(x: number, lambda: number): number {
  return Math.abs(lambda) < TINY ? Math.log(x) : (Math.pow(x, lambda) - 1) / lambda
}

function boxCoxInverseScalar(z: number, lambda: number): number {
  return Math.abs(lambda) < TINY ? Math.exp(z) : Math.pow(lambda * z + 1, 1 / lambda)
}

function yeoJohnsonScalar(x: number, lambda: number): number {
  if (x >= 0) return Math.abs(lambda) < TINY ? Math.log1p(x) : (Math.pow(x + 1, lambda) - 1) / lambda
  return Math.abs(lambda - 2) < TINY ? -Math.log1p(-x) : -(Math.pow(1 - x, 2 - lambda) - 1) / (2 - lambda)
}

function yeoJohnsonInverseScalar(z: number, lambda: number): number {
  if (z >= 0) return Math.abs(lambda) < TINY ? Math.expm1(z) : Math.pow(z * lambda + 1, 1 / lambda) - 1
  return Math.abs(lambda - 2) < TINY ? -Math.expm1(-z) : 1 - Math.pow(-(2 - lambda) * z + 1, 1 / (2 - lambda))
}

/** The Box–Cox transform (x^λ − 1)/λ (log x at λ = 0) of positive x, elementwise (Box and Cox, 1964, JRSS B 26). */
export function boxCox(x: number, lambda: number): number
export function boxCox(x: Tensor, lambda: number): Tensor
export function boxCox(x: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof x === 'number' ? boxCoxScalar(x, lambda) : map(x, (v) => boxCoxScalar(v, lambda))
}

/** The inverse Box–Cox transform (λz + 1)^{1/λ} (exp z at λ = 0), elementwise. */
export function boxCoxInverse(z: number, lambda: number): number
export function boxCoxInverse(z: Tensor, lambda: number): Tensor
export function boxCoxInverse(z: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof z === 'number' ? boxCoxInverseScalar(z, lambda) : map(z, (v) => boxCoxInverseScalar(v, lambda))
}

/**
 * The Yeo–Johnson transform of any real x, elementwise (Yeo and Johnson, 2000, "A new family of power transformations
 * to improve normality or symmetry", Biometrika 87): ((x + 1)^λ − 1)/λ for x ≥ 0 and −((1 − x)^{2−λ} − 1)/(2 − λ) for
 * x < 0, with logarithmic limits at λ = 0 and λ = 2.
 */
export function yeoJohnson(x: number, lambda: number): number
export function yeoJohnson(x: Tensor, lambda: number): Tensor
export function yeoJohnson(x: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof x === 'number' ? yeoJohnsonScalar(x, lambda) : map(x, (v) => yeoJohnsonScalar(v, lambda))
}

/** The inverse Yeo–Johnson transform, elementwise. */
export function yeoJohnsonInverse(z: number, lambda: number): number
export function yeoJohnsonInverse(z: Tensor, lambda: number): Tensor
export function yeoJohnsonInverse(z: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof z === 'number' ? yeoJohnsonInverseScalar(z, lambda) : map(z, (v) => yeoJohnsonInverseScalar(v, lambda))
}

// ── Maximum likelihood for λ ─────────────────────────────────────────────────────────────────────────────────────────

/** The result of a λ search. */
export type PowerLambda = {
  lambda: number
  /** The profile log-likelihood at λ (up to a constant). */
  logLikelihood: number
  iterations: number
  converged: boolean
}

/**
 * The Box–Cox λ maximising the profile log-likelihood (λ − 1) Σ log xᵢ − (n/2) log σ̂²(λ), σ̂² the population
 * variance of the transformed values (Box and Cox, 1964), by Brent's method from the bracket (−2, 2) as scipy's
 * `boxcox_normmax(method='mle')`. x must be positive.
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
 * The Yeo–Johnson λ maximising −(n/2) log σ̂²(λ) + (λ − 1) Σ sign(xᵢ) log(1 + |xᵢ|), by Brent's method from the bracket
 * (−2, 2), as scikit-learn's `PowerTransformer`.
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
