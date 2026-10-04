/**
 * AdPredictor (Graepel et al., 2010): Bayesian online probit regression for click-through prediction on sparse binary
 * features. Each feature value i has a weight wᵢ ~ N(μᵢ, σᵢ²); an impression with active features A is clicked
 * (y = +1) or not (y = −1) with P(y | w) = Φ(y Σ_{i∈A} wᵢ / β). Assumed-density filtering keeps the posterior Gaussian
 * and factorised: with Σ² = β² + Σ_{i∈A} σᵢ² and t = y Σ_{i∈A} μᵢ / Σ,
 *
 *   μᵢ ← μᵢ + y (σᵢ²/Σ) v(t),   σᵢ² ← σᵢ² (1 − (σᵢ²/Σ²) w(t)),
 *
 * where v(t) = φ(t)/Φ(t) and w(t) = v(t)(v(t) + t) are the mean and variance corrections of a truncated Gaussian (the
 * same as TrueSkill's). The predictive click probability is Φ(Σ μᵢ / Σ).
 */

import { normalCdf, truncatedNormalV, truncatedNormalW } from 'aifn-compute/numerics/special'

/** An AdPredictor model: per-feature Gaussian weights, plain data. */
export interface AdPredictor {
  readonly beta: number
  readonly mean: Float64Array
  readonly variance: Float64Array
}

/** A new model over `features` feature values with prior N(0, priorVariance) on every weight (default 1, β = 1). */
export function adPredictor(features: number, options: { beta?: number; priorVariance?: number } = {}): AdPredictor {
  const { beta = 1, priorVariance = 1 } = options
  return { beta, mean: new Float64Array(features), variance: new Float64Array(features).fill(priorVariance) }
}

const spread = (m: AdPredictor, active: readonly number[]) =>
  Math.sqrt(m.beta * m.beta + active.reduce((s, i) => s + m.variance[i], 0))

/** The predictive click probability Φ(Σ μ / Σ) of an impression with active features. */
export function adPredictorProbability(m: AdPredictor, active: readonly number[]): number {
  return normalCdf(active.reduce((s, i) => s + m.mean[i], 0) / spread(m, active)) as number
}

/** One ADF update on an impression (module docs); returns a new model. */
export function adPredictorUpdate(m: AdPredictor, active: readonly number[], clicked: boolean): AdPredictor {
  const y = clicked ? 1 : -1
  const S = spread(m, active)
  const t = (y * active.reduce((s, i) => s + m.mean[i], 0)) / S
  const v = truncatedNormalV(t)
  const w = truncatedNormalW(t)
  const mean = Float64Array.from(m.mean)
  const variance = Float64Array.from(m.variance)
  for (const i of active) {
    mean[i] += ((y * m.variance[i]) / S) * v
    variance[i] *= 1 - (m.variance[i] / (S * S)) * w
  }
  return { beta: m.beta, mean, variance }
}
