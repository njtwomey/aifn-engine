/**
 * Metrics of real-valued point predictions: squared and absolute errors, R² and explained variance, robust and
 * distributional losses (Huber, log-cosh, squared log error, pinball, Tweedie deviances), percentage errors and scaled
 * errors for forecasting (MAPE, sMAPE, WMAPE, MASE, RMSSE).
 */

import { median } from 'aifn-compute/probability/stats'
import { caseWeights, defineMetric, divide, nonEmpty, sameLength, values, weightedMeanOf, type Data } from './core'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options shared by the regression metrics. */
export type RegressionOptions = {
  /** A weight per case; means become weighted means. */
  sampleWeight?: Data
}

/** True values and predictions as arrays, checked. */
function pair(yTrue: Data, yPred: Data, what: string) {
  const y = values(yTrue)
  const p = values(yPred)
  sameLength(y, p, what)
  nonEmpty(y.length, what)
  return { y, p }
}

/** The (weighted) mean of f(yᵢ, ŷᵢ). */
function meanLoss(yTrue: Data, yPred: Data, o: RegressionOptions, what: string, f: (y: number, p: number) => number) {
  const { y, p } = pair(yTrue, yPred, what)
  return weightedMeanOf(
    Float64Array.from(y, (v, i) => f(v, p[i])),
    caseWeights(o.sampleWeight, y.length, what),
  )
}

const errorInfo = (key: string, name: string, note: string) =>
  ({
    key,
    stability: 'stable',
    name,
    inputs: 'values',
    direction: 'lower',
    range: [0, Infinity],
    notes: [note],
    capability: 'decide',
  }) as const

/** Mean squared error, (1/n) Σ (yᵢ − ŷᵢ)², minimised by the conditional mean (squared-and-absolute-error-metrics). */
export const meanSquaredError = defineMetric(
  errorInfo('meanSquaredError', 'Mean squared error', 'squared-and-absolute-error-metrics'),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'meanSquaredError', (y, p) => (y - p) ** 2),
)

/** Root mean squared error, √MSE, in the units of y. */
export const rootMeanSquaredError = defineMetric(
  errorInfo('rootMeanSquaredError', 'Root mean squared error', 'squared-and-absolute-error-metrics'),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    Math.sqrt(meanLoss(yTrue, yPred, options, 'rootMeanSquaredError', (y, p) => (y - p) ** 2)),
)

/** Mean absolute error, (1/n) Σ |yᵢ − ŷᵢ|, minimised by the conditional median. */
export const meanAbsoluteError = defineMetric(
  errorInfo('meanAbsoluteError', 'Mean absolute error', 'squared-and-absolute-error-metrics'),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'meanAbsoluteError', (y, p) => Math.abs(y - p)),
)

/** Median absolute error, the median of |yᵢ − ŷᵢ|: up to half the predictions can be arbitrarily wrong. */
export const medianAbsoluteError = defineMetric(
  errorInfo('medianAbsoluteError', 'Median absolute error', 'squared-and-absolute-error-metrics'),
  (yTrue: Data, yPred: Data): number => {
    const { y, p } = pair(yTrue, yPred, 'medianAbsoluteError')
    return median(Float64Array.from(y, (v, i) => Math.abs(v - p[i])))
  },
)

/** Maximum error, the worst single miss maxᵢ |yᵢ − ŷᵢ|. */
export const maxError = defineMetric(
  errorInfo('maxError', 'Maximum error', 'squared-and-absolute-error-metrics'),
  (yTrue: Data, yPred: Data): number => {
    const { y, p } = pair(yTrue, yPred, 'maxError')
    let m = 0
    for (let i = 0; i < y.length; i++) m = Math.max(m, Math.abs(y[i] - p[i]))
    return m
  },
)

/**
 * Normalised RMSE: RMSE divided by the range, the mean or the population standard deviation of y (default `std`,
 * which equals √(1 − R²)).
 */
export const normalisedRootMeanSquaredError = defineMetric(
  errorInfo('normalisedRootMeanSquaredError', 'Normalised RMSE', 'squared-and-absolute-error-metrics'),
  (yTrue: Data, yPred: Data, options: { by?: 'range' | 'mean' | 'std' } = {}): number => {
    const { y, p } = pair(yTrue, yPred, 'normalisedRootMeanSquaredError')
    const rmse = Math.sqrt(weightedMeanOf(Float64Array.from(y, (v, i) => (v - p[i]) ** 2)))
    const m = weightedMeanOf(y)
    const by = options.by ?? 'std'
    if (by === 'mean') return divide(rmse, m)
    // A loop, not Math.max(...y): spreading more than ~1e5 arguments overflows the call stack.
    if (by === 'range')
      return divide(rmse, y.reduce((a, v) => Math.max(a, v), -Infinity) - y.reduce((a, v) => Math.min(a, v), Infinity))
    return divide(rmse, Math.sqrt(weightedMeanOf(Float64Array.from(y, (v) => (v - m) ** 2))))
  },
)

/** SSE and SST, and the error mean and variance, with optional weights. */
function sums(yTrue: Data, yPred: Data, o: RegressionOptions, what: string) {
  const { y, p } = pair(yTrue, yPred, what)
  const w = caseWeights(o.sampleWeight, y.length, what)
  const e = Float64Array.from(y, (v, i) => v - p[i])
  const yBar = weightedMeanOf(y, w)
  const eBar = weightedMeanOf(e, w)
  let sse = 0
  let sst = 0
  let see = 0
  for (let i = 0; i < y.length; i++) {
    const wi = w ? w[i] : 1
    sse += wi * e[i] ** 2
    sst += wi * (y[i] - yBar) ** 2
    see += wi * (e[i] - eBar) ** 2
  }
  return { sse, sst, see, n: y.length }
}

/**
 * The coefficient of determination R² = 1 − SSE/SST (r-squared-and-explained-variance): 0 for predicting ȳ, 1 for
 * perfect predictions, negative for worse than ȳ. NaN when y is constant (scikit-learn substitutes 1 or 0).
 */
export const r2Score = defineMetric(
  {
    key: 'r2Score',
    stability: 'stable',
    name: 'Coefficient of determination R²',
    inputs: 'values',
    direction: 'higher',
    range: [-Infinity, 1],
    notes: ['r-squared-and-explained-variance'],
    capability: 'decide',
  },
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number => {
    const s = sums(yTrue, yPred, options, 'r2Score')
    return 1 - divide(s.sse, s.sst)
  },
)

/** The explained variance score 1 − var(y − ŷ)/var(y), which ignores a constant offset: EV − R² = ē²/var(y) ≥ 0. */
export const explainedVariance = defineMetric(
  {
    key: 'explainedVariance',
    stability: 'stable',
    name: 'Explained variance',
    inputs: 'values',
    direction: 'higher',
    range: [-Infinity, 1],
    notes: ['r-squared-and-explained-variance'],
    capability: 'decide',
  },
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number => {
    const s = sums(yTrue, yPred, options, 'explainedVariance')
    return 1 - divide(s.see, s.sst)
  },
)

/** Adjusted R² = 1 − (1 − R²)(n − 1)/(n − p − 1) for a model with `predictors` p. */
export const adjustedR2Score = defineMetric(
  {
    key: 'adjustedR2Score',
    stability: 'stable',
    name: 'Adjusted R²',
    inputs: 'values',
    direction: 'higher',
    range: [-Infinity, 1],
    notes: ['r-squared-and-explained-variance'],
    capability: 'decide',
  },
  (yTrue: Data, yPred: Data, options: { predictors: number }): number => {
    const s = sums(yTrue, yPred, {}, 'adjustedR2Score')
    const r2 = 1 - divide(s.sse, s.sst)
    return 1 - (1 - r2) * divide(s.n - 1, s.n - options.predictors - 1)
  },
)

// ── Robust and distributional losses ─────────────────────────────────────────────────────────────────────────────────

const robustNote = 'robust-and-distributional-regression-losses'

/** Log-cosh loss (1/n) Σ log cosh(eᵢ): ≈ e²/2 for small errors and |e| − log 2 for large ones. Stable for large e. */
export const logCoshError = defineMetric(
  errorInfo('logCoshError', 'Log-cosh', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'logCoshError', (y, p) => {
      const a = Math.abs(y - p)
      // log cosh a = a + log1p(e^{−2a}) − log 2, which does not overflow.
      return a + Math.log1p(Math.exp(-2 * a)) - Math.LN2
    }),
)

/** The Huber loss with threshold δ (default 1): ½e² for |e| ≤ δ, δ(|e| − ½δ) beyond (Huber 1964). */
export const huberLoss = defineMetric(
  errorInfo('huberLoss', 'Huber loss', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions & { delta?: number } = {}): number => {
    const d = options.delta ?? 1
    return meanLoss(yTrue, yPred, options, 'huberLoss', (y, p) => {
      const a = Math.abs(y - p)
      return a <= d ? 0.5 * a * a : d * (a - 0.5 * d)
    })
  },
)

/** Mean squared log error (1/n) Σ (log(1 + yᵢ) − log(1 + ŷᵢ))², for y, ŷ > −1 (a relative error). */
export const meanSquaredLogError = defineMetric(
  errorInfo('meanSquaredLogError', 'Mean squared log error', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'meanSquaredLogError', (y, p) => (Math.log1p(y) - Math.log1p(p)) ** 2),
)

/**
 * The pinball (quantile) loss at level τ (default 0.5): τ(y − q) when y ≥ q and (1 − τ)(q − y) otherwise, minimised by
 * the τ-quantile. At τ = 0.5 it is half the absolute error.
 */
export const pinballLoss = defineMetric(
  errorInfo('pinballLoss', 'Pinball (quantile) loss', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions & { tau?: number } = {}): number => {
    const tau = options.tau ?? 0.5
    return meanLoss(yTrue, yPred, options, 'pinballLoss', (y, q) => (y >= q ? tau * (y - q) : (1 - tau) * (q - y)))
  },
)

/**
 * Unit Tweedie deviance d_p(y, μ) (Jørgensen 1987): squared error at p = 0, the Poisson deviance
 * 2(y log(y/μ) − y + μ) at p = 1, the gamma deviance 2(log(μ/y) + y/μ − 1) at p = 2, and otherwise
 * 2(max(y, 0)^{2−p}/((1 − p)(2 − p)) − yμ^{1−p}/(1 − p) + μ^{2−p}/(2 − p)).
 */
export function tweedieUnitDeviance(y: number, mu: number, power: number): number {
  if (power === 0) return (y - mu) ** 2
  if (power === 1) return 2 * ((y > 0 ? y * Math.log(y / mu) : 0) - y + mu)
  if (power === 2) return 2 * (Math.log(mu / y) + y / mu - 1)
  const p = power
  return 2 * (Math.max(y, 0) ** (2 - p) / ((1 - p) * (2 - p)) - (y * mu ** (1 - p)) / (1 - p) + mu ** (2 - p) / (2 - p))
}

/** Mean Tweedie deviance with the given power (default 0); the mean is the best prediction for every power. */
export const tweedieDeviance = defineMetric(
  errorInfo('tweedieDeviance', 'Mean Tweedie deviance', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions & { power?: number } = {}): number =>
    meanLoss(yTrue, yPred, options, 'tweedieDeviance', (y, m) => tweedieUnitDeviance(y, m, options.power ?? 0)),
)

/** Mean Poisson deviance (Tweedie power 1), for counts; needs ŷ > 0. */
export const poissonDeviance = defineMetric(
  errorInfo('poissonDeviance', 'Mean Poisson deviance', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'poissonDeviance', (y, m) => tweedieUnitDeviance(y, m, 1)),
)

/** Mean gamma deviance (Tweedie power 2), a relative error; needs y, ŷ > 0. */
export const gammaDeviance = defineMetric(
  errorInfo('gammaDeviance', 'Mean gamma deviance', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'gammaDeviance', (y, m) => tweedieUnitDeviance(y, m, 2)),
)

// ── Forecasting ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Mean absolute percentage error (1/n) Σ |yₜ − ŷₜ|/|yₜ| (percentage-errors), as a fraction (0.2 means 20%), like
 * scikit-learn. Infinite when an actual is 0.
 */
export const meanAbsolutePercentageError = defineMetric(
  errorInfo('meanAbsolutePercentageError', 'Mean absolute percentage error', 'percentage-errors'),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'meanAbsolutePercentageError', (y, p) => Math.abs(y - p) / Math.abs(y)),
)

/**
 * Symmetric MAPE (1/n) Σ 2|yₜ − ŷₜ|/(|yₜ| + |ŷₜ|) as a fraction in [0, 2] (Makridakis 1993). A term with yₜ = ŷₜ = 0
 * is 0/0 and makes the result NaN.
 */
export const symmetricMeanAbsolutePercentageError = defineMetric(
  { ...errorInfo('symmetricMeanAbsolutePercentageError', 'Symmetric MAPE', 'percentage-errors'), range: [0, 2] },
  (yTrue: Data, yPred: Data): number =>
    meanLoss(
      yTrue,
      yPred,
      {},
      'symmetricMeanAbsolutePercentageError',
      (y, p) => (2 * Math.abs(y - p)) / (Math.abs(y) + Math.abs(p)),
    ),
)

/** Weighted MAPE Σ|yₜ − ŷₜ| / Σ|yₜ|: total absolute error as a fraction of total volume. */
export const weightedMeanAbsolutePercentageError = defineMetric(
  errorInfo('weightedMeanAbsolutePercentageError', 'Weighted MAPE', 'percentage-errors'),
  (yTrue: Data, yPred: Data): number => {
    const { y, p } = pair(yTrue, yPred, 'weightedMeanAbsolutePercentageError')
    let err = 0
    let vol = 0
    for (let i = 0; i < y.length; i++) {
      err += Math.abs(y[i] - p[i])
      vol += Math.abs(y[i])
    }
    return divide(err, vol)
  },
)

/** The in-sample seasonal-naive scale Σ_{t>m} |yₜ − yₜ₋ₘ|^power / (T − m). */
function naiveScale(train: Data, season: number, power: 1 | 2): number {
  const y = values(train)
  if (y.length <= season)
    throw new DomainError('metrics', `metrics: the training series needs more than ${season} values`)
  let s = 0
  for (let t = season; t < y.length; t++) s += Math.abs(y[t] - y[t - season]) ** power
  return s / (y.length - season)
}

/** Options of the scaled errors: the training series and its seasonal period. */
export type ScaledErrorOptions = {
  /** The in-sample (training) series y₁ … y_T that sets the scale. */
  train: Data
  /** Seasonal period m (default 1, the non-seasonal naive forecast). */
  season?: number
}

/**
 * Mean absolute scaled error (Hyndman and Koehler 2006; scaled-errors-and-mean-absolute-scaled-error): the MAE of the
 * forecasts divided by Q = (1/(T − m)) Σ |yₜ − yₜ₋ₘ|, the in-sample MAE of the seasonal naive forecast. NaN when the
 * training series repeats exactly with period m.
 */
export const meanAbsoluteScaledError = defineMetric(
  errorInfo('meanAbsoluteScaledError', 'Mean absolute scaled error', 'scaled-errors-and-mean-absolute-scaled-error'),
  (yTrue: Data, yPred: Data, options: ScaledErrorOptions): number => {
    const { y, p } = pair(yTrue, yPred, 'meanAbsoluteScaledError')
    let mae = 0
    for (let i = 0; i < y.length; i++) mae += Math.abs(y[i] - p[i])
    return divide(mae / y.length, naiveScale(options.train, options.season ?? 1, 1))
  },
)

/**
 * Root mean squared scaled error: √(MSE of the forecasts / in-sample MSE of the seasonal naive forecast), the M5
 * competition's scale-free error (Makridakis et al. 2022).
 */
export const rootMeanSquaredScaledError = defineMetric(
  errorInfo(
    'rootMeanSquaredScaledError',
    'Root mean squared scaled error',
    'scaled-errors-and-mean-absolute-scaled-error',
  ),
  (yTrue: Data, yPred: Data, options: ScaledErrorOptions): number => {
    const { y, p } = pair(yTrue, yPred, 'rootMeanSquaredScaledError')
    let mse = 0
    for (let i = 0; i < y.length; i++) mse += (y[i] - p[i]) ** 2
    return Math.sqrt(divide(mse / y.length, naiveScale(options.train, options.season ?? 1, 2)))
  },
)
