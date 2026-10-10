/**
 * Parametric accelerated-failure-time (AFT) models (Kalbfleisch and Prentice, 2002):
 * $\log T = \mu + \xvec^\top\betavec + \sigma\varepsilon$, so a covariate multiplies the time scale by
 * $e^{\beta_k}$ per unit. The error $\varepsilon$ is standard extreme-value (minimum Gumbel) for the Weibull model
 * and standard normal for the log-normal one. With $z = (\log t - \mu - \xvec^\top\betavec)/\sigma$, an observed
 * event contributes $\log f(t) = \log f_\varepsilon(z) - \log \sigma - \log t$ and a right-censored time
 * contributes $\log S(t) = \log S_\varepsilon(z)$, where
 *
 * - Weibull: $\log f_\varepsilon(z) = z - e^z$ and $\log S_\varepsilon(z) = -e^z$ (shape $k = 1/\sigma$, scale
 *   $\lambda = e^{\mu + \xvec^\top\betavec}$), as lifelines' `WeibullAFTFitter`;
 * - log-normal: $\log f_\varepsilon(z) = \log \phi(z)$ and $\log S_\varepsilon(z) = \log \Phi(-z)$.
 *
 * The log-likelihood is written with tensor primitives, differentiated by autodiff and maximised by L-BFGS over
 * $(\mu, \betavec, \log \sigma)$. Standard errors come from the inverse of the autodiff Hessian of the negative
 * log-likelihood (the observed information).
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
  /** Always `'aft-model'`. */
  readonly kind: 'aft-model'
  /** The error law that was fitted. */
  readonly family: AftFamily
  /** The intercept $\mu$. */
  readonly intercept: number
  /** $\betavec$: the log time ratio per unit of each covariate, $p$ values. */
  readonly coefficients: Float64Array
  /** $\sigma$ (the Weibull shape is $1/\sigma$). */
  readonly scale: number
  /**
   * Standard errors of $(\mu, \betavec, \log \sigma)$, $p + 2$ values, from the inverse Hessian of the negative
   * log-likelihood.
   */
  readonly standardErrors: Float64Array
  /** The maximised log-likelihood, including the $-\log t$ of each event. */
  readonly logLikelihood: number
  /** Whether L-BFGS met its convergence test within `maxSteps`. */
  readonly converged: boolean
}

/**
 * A scalar value as a number (the first entry of a tensor; a traced value is read as its current value).
 *
 * @param v The value.
 * @returns Its number.
 */
const scalar = (v: Value): number => {
  const u = unwrap(v)
  return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
}

/**
 * Fit a Weibull or log-normal AFT model by maximum likelihood with right censoring, from $\beta = 0$ and the mean and
 * spread of the log-times. Throws `ShapeError` when `x`, `time` and `event` differ in rows, and `DomainError` for a
 * time that is not positive. Not converging is reported in `converged`, not thrown.
 *
 * @param x The covariates, $n \times p$ (one row per subject).
 * @param time The observed times, $n$ positive values: the event time, or the censoring time.
 * @param event The event flags, $n$ values: 1 for an observed event, 0 for a right-censored time.
 * @param options `family`: `'weibull'` (default) or `'log-normal'`. `maxSteps`: the most L-BFGS steps (default 500).
 * @returns The fit: $\mu$, $\betavec$, $\sigma$, their standard errors and the log-likelihood.
 *
 * @example A Weibull fit recovers the simulated parameters
 * // 200 Weibull times with log T = 1 + 0.5 x + 0.5 e (e minimum-Gumbel), x alternating 0 and 1, censored at 6.
 * const n = 200
 * const x = Array.from({ length: n }, (_, i) => [i % 2])
 * const e = toArray(log(neg(log(uniform(stream(0), 0, 1, { shape: [n] })))))
 * const t = x.map(([xi], i) => Math.exp(1 + 0.5 * xi + 0.5 * e[i]))
 * const time = t.map((ti) => Math.min(ti, 6))
 * const event = t.map((ti) => (ti < 6 ? 1 : 0))
 * const fit = aftModel(x, time, event)
 * print('intercept =', fit.intercept, ' coefficient =', fit.coefficients, ' scale =', fit.scale)
 * print('standard errors =', fit.standardErrors)
 */
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

/**
 * The survival function $S(t \mid \xvec)$ of a fitted AFT model at the given times: $\exp(-e^z)$ (Weibull) or
 * $\Phi(-z)$ (log-normal), $z = (\log t - \mu - \xvec^\top\betavec)/\sigma$.
 *
 * @param fit The fitted model.
 * @param x The subject's covariates, $p$ values.
 * @param times The times $t > 0$ to evaluate at.
 * @returns $S(t \mid \xvec)$ at each time.
 *
 * @example The second group survives longer
 * // 200 Weibull times with log T = 1 + 0.5 x + 0.5 e (e minimum-Gumbel), x alternating 0 and 1, censored at 6.
 * const n = 200
 * const x = Array.from({ length: n }, (_, i) => [i % 2])
 * const e = toArray(log(neg(log(uniform(stream(0), 0, 1, { shape: [n] })))))
 * const t = x.map(([xi], i) => Math.exp(1 + 0.5 * xi + 0.5 * e[i]))
 * const time = t.map((ti) => Math.min(ti, 6))
 * const event = t.map((ti) => (ti < 6 ? 1 : 0))
 * const fit = aftModel(x, time, event)
 * print('S(1), S(2), S(4) at x = 0:', aftSurvival(fit, [0], [1, 2, 4]))
 * print('S(1), S(2), S(4) at x = 1:', aftSurvival(fit, [1], [1, 2, 4]))
 */
export function aftSurvival(fit: AftFit, x: ArrayLike<number>, times: ArrayLike<number>): Float64Array {
  let eta = fit.intercept
  for (let k = 0; k < fit.coefficients.length; k++) eta += fit.coefficients[k] * x[k]
  return Float64Array.from(times, (t) => {
    const z = (Math.log(t) - eta) / fit.scale
    return fit.family === 'weibull' ? Math.exp(-Math.exp(z)) : (normalCdf(-z) as number)
  })
}

/**
 * The time ratio $e^{\beta_k}$ per unit of each covariate: above 1 it lengthens survival, below 1 it shortens it.
 *
 * @param fit The fitted model.
 * @returns One ratio per covariate.
 *
 * @example A time ratio near the simulated one
 * // 200 Weibull times with log T = 1 + 0.5 x + 0.5 e (e minimum-Gumbel), x alternating 0 and 1, censored at 6.
 * const n = 200
 * const x = Array.from({ length: n }, (_, i) => [i % 2])
 * const e = toArray(log(neg(log(uniform(stream(0), 0, 1, { shape: [n] })))))
 * const t = x.map(([xi], i) => Math.exp(1 + 0.5 * xi + 0.5 * e[i]))
 * const time = t.map((ti) => Math.min(ti, 6))
 * const event = t.map((ti) => (ti < 6 ? 1 : 0))
 * const fit = aftModel(x, time, event)
 * print('time ratio =', timeRatios(fit), ' simulated =', Math.exp(0.5))
 */
export function timeRatios(fit: AftFit): Float64Array {
  return Float64Array.from(fit.coefficients, Math.exp)
}
