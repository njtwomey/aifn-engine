/**
 * AdPredictor (Graepel et al., 2010): Bayesian online probit regression for click-through prediction on sparse binary
 * features.
 *
 * Each feature value $i$ has a weight $w_i \sim \Gauss(\mu_i, \sigma_i^2)$; an impression with active features
 * $\Acal$ is clicked ($y = +1$) or not ($y = -1$) with $P(y \mid \wvec) = \Phi(y \sum_{i \in \Acal} w_i / \beta)$.
 * Assumed-density filtering keeps the posterior Gaussian and factorised: with
 * $s^2 = \beta^2 + \sum_{i \in \Acal} \sigma_i^2$ and $t = y \sum_{i \in \Acal} \mu_i / s$, each active weight is
 * updated by
 *
 * $\mu_i \leftarrow \mu_i + y (\sigma_i^2 / s)\, v(t)$ and
 * $\sigma_i^2 \leftarrow \sigma_i^2 (1 - (\sigma_i^2 / s^2)\, w(t))$,
 *
 * where $v(t) = \phi(t) / \Phi(t)$ and $w(t) = v(t)(v(t) + t)$ are the mean and variance corrections of a truncated
 * Gaussian (the same as TrueSkill's). The predictive click probability is $\Phi(\sum_{i \in \Acal} \mu_i / s)$. A model
 * is plain data, and an update returns a new one.
 */

import { normalCdf, truncatedNormalV, truncatedNormalW } from 'aifn-compute/numerics/special'

/** An AdPredictor model: per-feature Gaussian weights, plain data. */
export interface AdPredictor {
  /** The scale $\beta$ of the probit likelihood. */
  readonly beta: number
  /** The posterior means $\mu_i$, one per feature value. */
  readonly mean: Float64Array
  /** The posterior variances $\sigma_i^2$, one per feature value. */
  readonly variance: Float64Array
}

/**
 * A new model over `features` feature values with the prior $\Gauss(0, \sigma_0^2)$ on every weight.
 *
 * @param features The number of feature values; an impression names its active ones by index.
 * @param options `beta`, the probit scale $\beta$ (default 1), and `priorVariance`, the prior variance $\sigma_0^2$
 *   of every weight (default 1).
 * @returns The model, with zero means.
 *
 * @example Three feature values
 * print(adPredictor(3, { beta: 0.5 }))
 */
export function adPredictor(features: number, options: { beta?: number; priorVariance?: number } = {}): AdPredictor {
  const { beta = 1, priorVariance = 1 } = options
  return { beta, mean: new Float64Array(features), variance: new Float64Array(features).fill(priorVariance) }
}

/**
 * The predictive spread $s = \sqrt{\beta^2 + \sum_{i \in \Acal} \sigma_i^2}$ of an impression.
 *
 * @param m The model.
 * @param active The indices of the active feature values $\Acal$.
 * @returns $s$.
 */
const spread = (m: AdPredictor, active: readonly number[]) =>
  Math.sqrt(m.beta * m.beta + active.reduce((s, i) => s + m.variance[i], 0))

/**
 * The predictive click probability $\Phi(\sum_{i \in \Acal} \mu_i / s)$ of an impression with active features.
 *
 * @param m The model.
 * @param active The indices of the active feature values $\Acal$.
 * @returns The probability of a click.
 *
 * @example A click raises the probability for the same features
 * const m = adPredictor(2)
 * print('before:', adPredictorProbability(m, [0, 1]))
 * print('after a click:', adPredictorProbability(adPredictorUpdate(m, [0, 1], true), [0, 1]))
 */
export function adPredictorProbability(m: AdPredictor, active: readonly number[]): number {
  return normalCdf(active.reduce((s, i) => s + m.mean[i], 0) / spread(m, active)) as number
}

/**
 * One assumed-density filtering update on an impression (see the file comment): only the active weights change.
 *
 * @param m The model; not modified.
 * @param active The indices of the active feature values $\Acal$.
 * @param clicked Whether the impression was clicked ($y = +1$) or not ($y = -1$).
 * @returns A new model.
 *
 * @example Ad 0 is always clicked, ad 1 never; both appear on site 2
 * let m = adPredictor(3)
 * for (let k = 0; k < 5; k++) {
 *   m = adPredictorUpdate(m, [0, 2], true)
 *   m = adPredictorUpdate(m, [1, 2], false)
 * }
 * print('means:', m.mean)
 * print('variances:', m.variance)
 * print('P(click) ad 0:', adPredictorProbability(m, [0, 2]), 'ad 1:', adPredictorProbability(m, [1, 2]))
 */
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
