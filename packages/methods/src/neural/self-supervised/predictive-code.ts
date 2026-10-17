/**
 * The linear one-dimensional code of a stationary Gaussian series that is most informative about the value $k$ steps
 * ahead: contrastive predictive coding's optimum for a linear context and a Gaussian series.
 *
 * The code is $c_t = \wvec^\top (x_t, x_{t-1}, \dots, x_{t-L+1})$, a linear map of a window of $L$ past values. For
 * jointly Gaussian variables $I(c_t; x_{t+k}) = -\tfrac12 \log(1 - \rho^2)$, with $\rho$ the correlation of $c_t$ and
 * $x_{t+k}$, so the most informative code maximises $\rho$. That is the least-squares predictor of $x_{t+k}$ from the
 * window, found from the autocovariance $\gamma$ by the normal equations $\Gammamat \wvec = \gammavec_k$ with
 * $\Gamma_{ij} = \gamma(\lvert i - j \rvert)$ and $(\gammavec_k)_i = \gamma(k + i)$ (for $k = 1$, the AR($L$)
 * coefficients of `levinsonDurbin`). It is the optimum that contrastive predictive coding's InfoNCE objective
 * approaches for a linear context and a Gaussian series (van den Oord et al., 2018, §2): the critic's optimum is the
 * density ratio, whose logarithm is linear in $c_t$ for Gaussian data. `codeCorrelation` measures what the code keeps
 * of one additive component of the series.
 *
 * Exact from the autocovariances; no simulation. The normal equations are solved by `solveDense`
 * (`aifn-compute/numerics/linalg`).
 */

import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { solveDense } from 'aifn-compute/numerics/linalg'

/** The most informative linear code of a window about the value `horizon` steps ahead. */
export type PredictiveCode = {
  /** The weights $w_0, \dots, w_{L-1}$ on $x_t, \dots, x_{t-L+1}$ (scaled so that the code predicts $x_{t+k}$). */
  weights: Float64Array
  /** The code's variance $\wvec^\top \Gammamat \wvec$. */
  variance: number
  /** The squared correlation $\rho^2$ of the code with $x_{t+k}$. */
  r2: number
  /** $I(c_t; x_{t+k}) = -\tfrac12 \log(1 - \rho^2)$, in nats. */
  information: number
}

/**
 * The least-squares predictor of $x_{t+k}$ from $(x_t, \dots, x_{t-L+1})$, from the series' autocovariances. With
 * `target`, the predicted quantity is instead an additive component $s$ of the series $k$ steps ahead (the other terms
 * of $x$ independent of $s$), whose cross-covariance with the window is $\gamma_s(k + i)$; $\rho^2$ and the
 * information are then about $s_{t+k}$. $\rho^2$ is clamped to $[0, 1 - 10^{-12}]$ so the information stays finite.
 * Throws `DomainError` for a window or horizon below 1, `ShapeError` for too few autocovariances, and
 * `NumericalError` (`singular`) when $\Gammamat$ is singular (a deterministic series).
 *
 * @param acov The autocovariances $\gamma(0), \gamma(1), \dots$ of the series, at least $L + k$ of them.
 * @param window The window length $L \ge 1$.
 * @param horizon The horizon $k \ge 1$.
 * @param target The autocovariances $\gamma_s$ of the component to predict (default the series itself).
 * @returns The weights, the code's variance, $\rho^2$ and the mutual information.
 *
 * @example An AR(1) series with $\phi = 0.8$: $\rho^2 = \phi^{2k}$
 * const acov = Array.from({ length: 8 }, (_, h) => 0.8 ** h / (1 - 0.64))
 * for (const k of [1, 2, 3]) {
 *   const code = linearPredictiveCode(acov, 2, k)
 *   print('k =', k, ' weights:', code.weights, ' rho² =', code.r2, ' nats:', code.information)
 * }
 */
export function linearPredictiveCode(
  acov: ArrayLike<number>,
  window: number,
  horizon: number,
  target: ArrayLike<number> = acov,
): PredictiveCode {
  const L = window
  const k = horizon
  if (!(L >= 1) || !(k >= 1))
    throw new DomainError('linearPredictiveCode', 'linearPredictiveCode: window and horizon must be at least 1')
  if (acov.length < L + k || target.length < L + k)
    throw new ShapeError('linearPredictiveCode', 'linearPredictiveCode: need at least window + horizon autocovariances')
  const G = new Float64Array(L * L)
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) G[i * L + j] = acov[Math.abs(i - j)]
  const rhs = Float64Array.from({ length: L }, (_, i) => target[k + i])
  const { x } = solveDense(G, rhs, L)
  if (!x)
    throw new NumericalError(
      'linearPredictiveCode',
      'linearPredictiveCode: the autocovariance matrix of the window is singular',
      'singular',
    )
  const weights = Float64Array.from(x)
  let variance = 0
  for (let i = 0; i < L; i++) variance += weights[i] * rhs[i]
  const r2 = Math.min(1 - 1e-12, Math.max(0, variance / target[0]))
  return { weights, variance, r2, information: -0.5 * Math.log(1 - r2) }
}

/**
 * The correlation of the code $c_t = \wvec^\top (x_t, \dots, x_{t-L+1})$ with an additive component $s$ of the series
 * at time $t + j$, when $x = s + (\text{terms independent of } s)$:
 * $\operatorname{cov}(c_t, s_{t+j}) = \sum_i w_i \gamma_s(i + j)$, divided by the two standard deviations. 0 when
 * either variance is 0.
 *
 * @param code The code, from `linearPredictiveCode`.
 * @param componentAcov The autocovariances $\gamma_s$ of the component, at least $L + j$ of them.
 * @param lead The lead $j \ge 0$ (0 for the component now, $k$ for the component at the predicted time).
 * @returns The correlation, in $[-1, 1]$.
 *
 * @example How much of a slow component a one-step code keeps
 * // The series is a slow AR(1) component (phi 0.95, variance 1) plus white noise of variance 1.
 * const slow = Array.from({ length: 10 }, (_, h) => 0.95 ** h)
 * const acov = slow.map((v, h) => (h === 0 ? v + 1 : v))
 * const code = linearPredictiveCode(acov, 4, 1, slow)
 * print('correlation with the slow component now:', codeCorrelation(code, slow))
 */
export function codeCorrelation(code: PredictiveCode, componentAcov: ArrayLike<number>, lead = 0): number {
  const w = code.weights
  let cov = 0
  for (let i = 0; i < w.length; i++) cov += w[i] * componentAcov[i + lead]
  const denom = Math.sqrt(code.variance * componentAcov[0])
  return denom > 0 ? cov / denom : 0
}

/**
 * The code applied to a series: $c_t = \sum_i w_i x_{t-i}$ for $t \ge L - 1$, NaN before (the window is not full).
 *
 * @param code The code.
 * @param x The series.
 * @returns The code at each time, one value per entry of `x`.
 *
 * @example A two-tap code on a short series
 * const code = { weights: Float64Array.from([0.5, 0.25]), variance: 1, r2: 0.5, information: Math.log(2) / 2 }
 * print(applyPredictiveCode(code, [1, 2, 3, 4]))
 */
export function applyPredictiveCode(code: PredictiveCode, x: ArrayLike<number>): Float64Array {
  const w = code.weights
  const out = new Float64Array(x.length).fill(NaN)
  for (let t = w.length - 1; t < x.length; t++) {
    let c = 0
    for (let i = 0; i < w.length; i++) c += w[i] * x[t - i]
    out[t] = c
  }
  return out
}
