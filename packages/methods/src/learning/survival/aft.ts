/**
 * Parametric accelerated-failure-time (AFT) models: log T = μ + xᵀβ + σ ε, so a covariate multiplies the time scale by
 * exp(βₖ) per unit. The error ε is standard extreme-value (minimum Gumbel) for the Weibull model and standard normal for
 * the log-normal one. With z = (log t − μ − xᵀβ)/σ, an observed event contributes log f(t) = log f_ε(z) − log σ − log t
 * and a right-censored time contributes log S(t) = log S_ε(z), where
 *
 * - Weibull: log f_ε(z) = z − eᶻ and log S_ε(z) = −eᶻ (shape k = 1/σ, scale λ = exp(μ + xᵀβ));
 * - log-normal: log f_ε(z) = log φ(z) and log S_ε(z) = log Φ(−z).
 *
 * The log-likelihood is written with tensor primitives, differentiated by autodiff and maximised by L-BFGS over
 * (μ, β, log σ). Standard errors come from the inverse of the autodiff Hessian.
 */

import { hessian, valueAndGrad } from 'aifn-compute/foundation/autodiff'
import {
  add,
  dense,
  exp,
  fromData,
  matmul,
  mul,
  neg,
  reshape,
  slice,
  sub,
  sum,
  toFlat,
  unwrap,
  type MatrixLike,
  type Tensor,
  type Value,
  type VectorLike,
} from 'aifn-compute/foundation/tensor'
import { inverse } from 'aifn-compute/numerics/linalg'
import { normalCdf, normalLogCdf } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The error law of an AFT model. */
export type AftFamily = 'weibull' | 'log-normal'

/** A fitted AFT model. */
export interface AftFit {
  readonly kind: 'aft-model'
  readonly family: AftFamily
  /** The intercept μ. */
  readonly intercept: number
  /** β: log time ratios per unit of each covariate. */
  readonly coefficients: Float64Array
  /** σ (the Weibull shape is 1/σ). */
  readonly scale: number
  /** Standard errors of (μ, β, log σ), from the inverse Hessian of the log-likelihood. */
  readonly standardErrors: Float64Array
  readonly logLikelihood: number
  readonly converged: boolean
}

const scalar = (v: Value): number => {
  const u = unwrap(v)
  return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
}

/** Fit a Weibull or log-normal AFT model to covariates x [n, p], times [n] > 0 and event flags [n]. */
export function aftModel(
  x: MatrixLike,
  time: VectorLike,
  event: VectorLike,
  options: { family?: AftFamily; maxSteps?: number } = {},
): AftFit {
  const { family = 'weibull', maxSteps = 500 } = options
  const X = dense.toMatrixF64(x, 'aftModel')
  const t = dense.toF64(time, 'aftModel')
  const e = dense.toF64(event, 'aftModel')
  const { m: n, n: p } = X
  if (t.length !== n || e.length !== n)
    throw new ShapeError('aftModel', 'aftModel: x, time and event must have the same rows')
  for (const v of t) if (!(v > 0)) throw new DomainError('aftModel', 'aftModel: times must be positive')
  const Xt = fromData(X.data, [n, p])
  const logT = fromData(Float64Array.from(t, Math.log), [n])
  const E = fromData(e, [n])
  const C = fromData(
    Float64Array.from(e, (v) => 1 - v),
    [n],
  )
  let sumLogT = 0
  for (let i = 0; i < n; i++) if (e[i] === 1) sumLogT += Math.log(t[i])

  const logLikelihood = (theta: Value): Value => {
    const mu = slice(theta, 0)
    const beta = reshape(slice(theta, [1, 1 + p]), [p, 1])
    const logSigma = slice(theta, 1 + p)
    const eta = add(mu, reshape(matmul(Xt, beta), [n]))
    const z = mul(sub(logT, eta), exp(neg(logSigma)))
    let logF: Value
    let logS: Value
    if (family === 'weibull') {
      const ez = exp(z)
      logF = sub(z, ez)
      logS = neg(ez)
    } else {
      logF = mul(-0.5, add(mul(z, z), Math.log(2 * Math.PI)))
      logS = normalLogCdf(neg(z))
    }
    const events = sum(mul(E, sub(logF, logSigma)))
    return sub(add(events, sum(mul(C, logS))), sumLogT)
  }
  const negative = (theta: Value) => neg(logLikelihood(theta))
  const vg = valueAndGrad(negative)
  // Start at the log-times' mean and spread, β = 0.
  const start = new Float64Array(p + 2)
  let m = 0
  for (let i = 0; i < n; i++) m += Math.log(t[i]) / n
  let v = 0
  for (let i = 0; i < n; i++) v += (Math.log(t[i]) - m) ** 2 / n
  start[0] = m
  start[p + 1] = 0.5 * Math.log(Math.max(v, 1e-6))
  const result = minimize(
    (theta) => {
      const { value, grad } = vg(theta)
      return { value: scalar(value), grad: unwrap(grad as Value) as Tensor }
    },
    start,
    { method: 'lbfgs', maxSteps },
  )
  const theta = Float64Array.from(toFlat(result.x))
  const H = hessian(negative)(fromData(theta, [p + 2])) as Tensor
  const cov = toFlat(inverse(H) as Tensor)
  return {
    kind: 'aft-model',
    family,
    intercept: theta[0],
    coefficients: theta.slice(1, 1 + p),
    scale: Math.exp(theta[p + 1]),
    standardErrors: Float64Array.from({ length: p + 2 }, (_, k) => Math.sqrt(cov[k * (p + 2) + k])),
    logLikelihood: -result.value,
    converged: result.converged,
  }
}

/** S(t | x) of a fitted AFT model at the given times. */
export function aftSurvival(fit: AftFit, x: ArrayLike<number>, times: ArrayLike<number>): Float64Array {
  let eta = fit.intercept
  for (let k = 0; k < fit.coefficients.length; k++) eta += fit.coefficients[k] * x[k]
  return Float64Array.from(times, (t) => {
    const z = (Math.log(t) - eta) / fit.scale
    return fit.family === 'weibull' ? Math.exp(-Math.exp(z)) : (normalCdf(-z) as number)
  })
}

/** The time ratio exp(βₖ) per unit of each covariate: >1 lengthens survival. */
export function timeRatios(fit: AftFit): Float64Array {
  return Float64Array.from(fit.coefficients, Math.exp)
}
