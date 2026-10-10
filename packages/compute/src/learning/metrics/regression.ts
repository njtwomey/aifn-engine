/**
 * Metrics of real-valued point predictions: squared and absolute errors, $R^2$ and explained variance, robust and
 * distributional losses (Huber, log-cosh, squared log error, pinball, Tweedie deviances), percentage errors and scaled
 * errors for forecasting (MAPE, sMAPE, WMAPE, MASE, RMSSE).
 *
 * Every metric takes the true values $y_i$ and the predictions $\hat y_i$ as arrays or tensors of the same length
 * (flattened row-major), throws on a length mismatch or an empty input, and returns a number. They follow
 * sklearn.metrics: those whose options include `RegressionOptions` take an optional `sampleWeight` per case, and an
 * undefined ratio (a constant $y$ in $R^2$, a zero denominator) is NaN rather than a substituted value.
 */

import { median } from 'aifn-compute/probability/stats'
import { caseWeights, defineMetric, divide, nonEmpty, sameLength, values, weightedMeanOf, type Data } from './core'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options shared by the regression metrics. */
export type RegressionOptions = {
  /** A weight per case, as many as there are cases; means become weighted means $\sum_i w_i e_i / \sum_i w_i$. */
  sampleWeight?: Data
}

/**
 * True values and predictions as new arrays, checked to have the same, non-zero length.
 *
 * @param yTrue The true values (an array or a tensor, read flattened).
 * @param yPred The predictions, one per true value.
 * @param what The caller's name for error messages.
 * @returns `y`, the true values, and `p`, the predictions, as Float64Arrays. Throws `ShapeError` when the lengths
 *   differ and `DomainError` when they are empty.
 */
function pair(yTrue: Data, yPred: Data, what: string) {
  const y = values(yTrue)
  const p = values(yPred)
  sameLength(y, p, what)
  nonEmpty(y.length, what)
  return { y, p }
}

/**
 * The (weighted) mean of $f(y_i, \hat y_i)$ over the cases.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param o The metric's options; only `sampleWeight` is read, and without it the mean is unweighted.
 * @param what The caller's name for error messages.
 * @param f The per-case loss, given a true value and its prediction.
 * @returns The mean loss.
 */
function meanLoss(yTrue: Data, yPred: Data, o: RegressionOptions, what: string, f: (y: number, p: number) => number) {
  const { y, p } = pair(yTrue, yPred, what)
  return weightedMeanOf(
    Float64Array.from(y, (v, i) => f(v, p[i])),
    caseWeights(o.sampleWeight, y.length, what),
  )
}

/**
 * The registry metadata of an error metric: stable, read from `values`, lower is better, range $[0, \infty)$.
 *
 * @param key The metric's registry key (its export name).
 * @param name The metric's display name.
 * @param note The key of the note that explains it.
 * @returns The metadata, with its literal fields kept.
 */
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

/**
 * Mean squared error, $\frac{1}{n} \sum_i (y_i - \hat y_i)^2$, minimised by the conditional mean
 * (squared-and-absolute-error-metrics). As sklearn's `mean_squared_error`.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param options `sampleWeight`, a weight per case.
 * @returns The mean squared error, in the squared units of $y$.
 *
 * @example Errors of 0.5, 0.5, 0 and 1
 * print('MSE', meanSquaredError([3, -0.5, 2, 7], [2.5, 0, 2, 8]))
 * print('weighted', meanSquaredError([3, -0.5, 2, 7], [2.5, 0, 2, 8], { sampleWeight: [1, 1, 1, 5] }))
 */
export const meanSquaredError = defineMetric(
  errorInfo('meanSquaredError', 'Mean squared error', 'squared-and-absolute-error-metrics'),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'meanSquaredError', (y, p) => (y - p) ** 2),
)

/**
 * Root mean squared error, $\sqrt{\mathrm{MSE}}$, in the units of $y$.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param options `sampleWeight`, a weight per case.
 * @returns The square root of the (weighted) mean squared error.
 *
 * @example The square root of the MSE
 * const y = [3, -0.5, 2, 7]
 * const p = [2.5, 0, 2, 8]
 * print('RMSE', rootMeanSquaredError(y, p))
 * print('sqrt(MSE)', Math.sqrt(meanSquaredError(y, p)))
 */
export const rootMeanSquaredError = defineMetric(
  errorInfo('rootMeanSquaredError', 'Root mean squared error', 'squared-and-absolute-error-metrics'),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    Math.sqrt(meanLoss(yTrue, yPred, options, 'rootMeanSquaredError', (y, p) => (y - p) ** 2)),
)

/**
 * Mean absolute error, $\frac{1}{n} \sum_i \lvert y_i - \hat y_i \rvert$, minimised by the conditional median. As
 * sklearn's `mean_absolute_error`.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param options `sampleWeight`, a weight per case.
 * @returns The mean absolute error, in the units of $y$.
 *
 * @example Errors of 0.5, 0.5, 0 and 1
 * print('MAE', meanAbsoluteError([3, -0.5, 2, 7], [2.5, 0, 2, 8]))
 */
export const meanAbsoluteError = defineMetric(
  errorInfo('meanAbsoluteError', 'Mean absolute error', 'squared-and-absolute-error-metrics'),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'meanAbsoluteError', (y, p) => Math.abs(y - p)),
)

/**
 * Median absolute error, the median of $\lvert y_i - \hat y_i \rvert$: robust to up to half the predictions being
 * arbitrarily wrong. Unweighted.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @returns The median absolute error (the mean of the two middle errors for an even count).
 *
 * @example One wild prediction does not move it
 * print('median AE', medianAbsoluteError([1, 2, 3, 4, 5], [1.5, 2, 3.5, 4, 50]))
 * print('MAE', meanAbsoluteError([1, 2, 3, 4, 5], [1.5, 2, 3.5, 4, 50]))
 */
export const medianAbsoluteError = defineMetric(
  errorInfo('medianAbsoluteError', 'Median absolute error', 'squared-and-absolute-error-metrics'),
  (yTrue: Data, yPred: Data): number => {
    const { y, p } = pair(yTrue, yPred, 'medianAbsoluteError')
    return median(Float64Array.from(y, (v, i) => Math.abs(v - p[i])))
  },
)

/**
 * Maximum error, the worst single miss $\max_i \lvert y_i - \hat y_i \rvert$. As sklearn's `max_error`.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @returns The largest absolute error.
 *
 * @example The worst of the misses
 * print('max error', maxError([3, 2, 7, 1], [4, 2, 7, 1]))
 */
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
 * Normalised RMSE: the RMSE divided by the range, the mean or the population standard deviation of $y$ (default
 * `std`, which makes it $\sqrt{1 - R^2}$). Unweighted; NaN when the divisor is 0.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param options `by`, the scale of $y$ to divide by: `'range'` ($\max_i y_i - \min_i y_i$), `'mean'` ($\bar y$) or
 *   `'std'` (the population standard deviation, the default).
 * @returns The RMSE as a fraction of the chosen scale.
 *
 * @example By the standard deviation, it is the square root of the unexplained fraction
 * const y = [3, -0.5, 2, 7]
 * const p = [2.5, 0, 2, 8]
 * print('by std', normalisedRootMeanSquaredError(y, p))
 * print('sqrt(1 - R2)', Math.sqrt(1 - r2Score(y, p)))
 * print('by range', normalisedRootMeanSquaredError(y, p, { by: 'range' }))
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

/**
 * The (weighted) sums of squares behind $R^2$ and the explained variance, with $e_i = y_i - \hat y_i$:
 * $\mathrm{SSE} = \sum_i w_i e_i^2$, $\mathrm{SST} = \sum_i w_i (y_i - \bar y)^2$ and
 * $\sum_i w_i (e_i - \bar e)^2$, the means $\bar y$ and $\bar e$ weighted too ($w_i = 1$ without weights).
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param o The metric's options; only `sampleWeight` is read.
 * @param what The caller's name for error messages.
 * @returns `sse`, `sst`, `see` (the spread of the errors about their mean) and `n`, the number of cases.
 */
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
 * The coefficient of determination $R^2 = 1 - \mathrm{SSE}/\mathrm{SST}$ (r-squared-and-explained-variance): 0 for
 * predicting $\bar y$, 1 for perfect predictions, negative for worse than $\bar y$. NaN when $y$ is constant
 * (scikit-learn substitutes 1 or 0).
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param options `sampleWeight`, a weight per case (in both sums and in $\bar y$).
 * @returns $R^2$, at most 1.
 *
 * @example Good predictions, and the mean as a prediction
 * const y = [3, -0.5, 2, 7]
 * print('R2', r2Score(y, [2.5, 0, 2, 8]))
 * print('predicting the mean', r2Score(y, [2.875, 2.875, 2.875, 2.875]))
 * print('constant y', r2Score([1, 1, 1], [1, 1, 2]))
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

/**
 * The explained variance score $1 - \var(y - \hat y)/\var(y)$, which ignores a constant offset:
 * $\mathrm{EV} - R^2 = \bar e^2/\var(y) \ge 0$, with $\bar e$ the mean error. NaN when $y$ is constant. As sklearn's
 * `explained_variance_score`.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param options `sampleWeight`, a weight per case.
 * @returns The explained variance, at most 1.
 *
 * @example An offset costs R2 but not the explained variance
 * const y = [1, 2, 3, 4]
 * const shifted = [2, 3, 4, 5]
 * print('EV', explainedVariance(y, shifted))
 * print('R2', r2Score(y, shifted))
 */
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

/**
 * Adjusted $R^2 = 1 - (1 - R^2)(n - 1)/(n - p - 1)$ for a model with $p$ = `predictors` predictors, which penalises
 * each predictor added. Unweighted; NaN when $n = p + 1$.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param options `predictors`, the number $p$ of predictors in the model (not counting the intercept).
 * @returns The adjusted $R^2$.
 *
 * @example The same fit with more predictors scores lower
 * const y = [3, -0.5, 2, 7, 4, 1]
 * const p = [2.5, 0, 2, 8, 4.5, 1]
 * print('R2', r2Score(y, p))
 * print('adjusted, 1 predictor', adjustedR2Score(y, p, { predictors: 1 }))
 * print('adjusted, 3 predictors', adjustedR2Score(y, p, { predictors: 3 }))
 */
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

/**
 * Log-cosh loss $\frac{1}{n} \sum_i \log\cosh e_i$ with $e_i = y_i - \hat y_i$: about $e^2/2$ for small errors and
 * $\lvert e \rvert - \log 2$ for large ones. Computed without overflow for large $e$.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param options `sampleWeight`, a weight per case.
 * @returns The mean log-cosh of the errors.
 *
 * @example Quadratic for a small error, linear for a large one
 * print('e = 0.1', logCoshError([0], [0.1]), 'vs e^2/2 =', 0.005)
 * print('e = 1000', logCoshError([0], [1000]), 'vs |e| - log 2 =', 1000 - Math.LN2)
 */
export const logCoshError = defineMetric(
  errorInfo('logCoshError', 'Log-cosh', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'logCoshError', (y, p) => {
      const a = Math.abs(y - p)
      // log cosh a = a + log1p(e^{−2a}) − log 2, which does not overflow.
      return a + Math.log1p(Math.exp(-2 * a)) - Math.LN2
    }),
)

/**
 * The mean Huber loss with threshold $\delta$ (default 1): $\tfrac{1}{2} e^2$ for $\lvert e \rvert \le \delta$ and
 * $\delta(\lvert e \rvert - \tfrac{1}{2}\delta)$ beyond, with $e = y - \hat y$ (Huber 1964).
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param options `sampleWeight`, a weight per case, and `delta`, the threshold $\delta$ where the loss turns from
 *   quadratic to linear (default 1).
 * @returns The mean Huber loss.
 *
 * @example Errors of 0.5, 1 and 3: 0.125, 0.5 and 2.5
 * print('Huber', huberLoss([0, 0, 0], [0.5, 1, 3]))
 * print('delta = 3, half the MSE', huberLoss([0, 0, 0], [0.5, 1, 3], { delta: 3 }))
 */
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

/**
 * Mean squared log error $\frac{1}{n} \sum_i (\log(1 + y_i) - \log(1 + \hat y_i))^2$, for $y, \hat y > -1$: a
 * relative error. Infinite for a value of $-1$ and NaN below it, where scikit-learn throws. As sklearn's
 * `mean_squared_log_error`.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predictions $\hat y_i$, one per true value.
 * @param options `sampleWeight`, a weight per case.
 * @returns The mean squared difference of the logs.
 *
 * @example Small for proportional misses
 * print('MSLE', meanSquaredLogError([3, 5, 2.5, 7], [2.5, 5, 4, 8]))
 */
export const meanSquaredLogError = defineMetric(
  errorInfo('meanSquaredLogError', 'Mean squared log error', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'meanSquaredLogError', (y, p) => (Math.log1p(y) - Math.log1p(p)) ** 2),
)

/**
 * The mean pinball (quantile) loss at level $\tau$ (default 0.5): $\tau(y - q)$ when $y \ge q$ and
 * $(1 - \tau)(q - y)$ otherwise, for a predicted quantile $q$; minimised by the $\tau$-quantile. At $\tau = 0.5$ it is
 * half the absolute error. As sklearn's `mean_pinball_loss` with `alpha` $= \tau$.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predicted $\tau$-quantiles $q_i$, one per true value.
 * @param options `sampleWeight`, a weight per case, and `tau`, the quantile level $\tau$ in $(0, 1)$ (default 0.5).
 * @returns The mean pinball loss.
 *
 * @example Under-prediction costs more at a high level
 * const y = [1, 2, 3]
 * print('tau = 0.5', pinballLoss(y, [0, 2, 3]))
 * print('tau = 0.9, under', pinballLoss(y, [0, 2, 3], { tau: 0.9 }))
 * print('tau = 0.9, over', pinballLoss(y, [2, 2, 3], { tau: 0.9 }))
 */
export const pinballLoss = defineMetric(
  errorInfo('pinballLoss', 'Pinball (quantile) loss', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions & { tau?: number } = {}): number => {
    const tau = options.tau ?? 0.5
    return meanLoss(yTrue, yPred, options, 'pinballLoss', (y, q) => (y >= q ? tau * (y - q) : (1 - tau) * (q - y)))
  },
)

/**
 * Unit Tweedie deviance $d_p(y, \mu)$ (Jørgensen 1987): the squared error $(y - \mu)^2$ at $p = 0$, the Poisson
 * deviance $2(y \log(y/\mu) - y + \mu)$ at $p = 1$ (with $y \log y = 0$ at $y = 0$), the gamma deviance
 * $2(\log(\mu/y) + y/\mu - 1)$ at $p = 2$, and otherwise
 * $2\big(\max(y, 0)^{2-p}/((1 - p)(2 - p)) - y\mu^{1-p}/(1 - p) + \mu^{2-p}/(2 - p)\big)$. The domain is not checked:
 * $\mu \le 0$, or $y \le 0$ at $p = 2$, gives NaN or an infinity.
 *
 * @param y The observed value.
 * @param mu The predicted mean $\mu$.
 * @param power The Tweedie power $p$.
 * @returns The deviance $d_p(y, \mu)$, 0 when $\mu = y$.
 *
 * @example The deviance of a count of 2 predicted as 0.5, at three powers
 * print('p = 0 (squared error)', tweedieUnitDeviance(2, 0.5, 0))
 * print('p = 1 (Poisson)', tweedieUnitDeviance(2, 0.5, 1))
 * print('p = 1.5 (compound Poisson-gamma)', tweedieUnitDeviance(2, 0.5, 1.5))
 */
export function tweedieUnitDeviance(y: number, mu: number, power: number): number {
  if (power === 0) return (y - mu) ** 2
  if (power === 1) return 2 * ((y > 0 ? y * Math.log(y / mu) : 0) - y + mu)
  if (power === 2) return 2 * (Math.log(mu / y) + y / mu - 1)
  const p = power
  return 2 * (Math.max(y, 0) ** (2 - p) / ((1 - p) * (2 - p)) - (y * mu ** (1 - p)) / (1 - p) + mu ** (2 - p) / (2 - p))
}

/**
 * Mean Tweedie deviance $\frac{1}{n} \sum_i d_p(y_i, \hat y_i)$ with the given power (default 0, the MSE); the mean
 * is the best prediction for every power. As sklearn's `mean_tweedie_deviance`, without its domain checks.
 *
 * @param yTrue The true values $y_i$.
 * @param yPred The predicted means $\hat y_i$, one per true value.
 * @param options `sampleWeight`, a weight per case, and `power`, the Tweedie power $p$ (default 0).
 * @returns The mean unit deviance, as `tweedieUnitDeviance` gives it.
 *
 * @example Power 0 is the MSE, power 1 the Poisson deviance
 * const y = [2, 0, 1, 4]
 * const mu = [0.5, 0.5, 2, 2]
 * print('p = 0', tweedieDeviance(y, mu), 'MSE', meanSquaredError(y, mu))
 * print('p = 1', tweedieDeviance(y, mu, { power: 1 }), 'Poisson', poissonDeviance(y, mu))
 */
export const tweedieDeviance = defineMetric(
  errorInfo('tweedieDeviance', 'Mean Tweedie deviance', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions & { power?: number } = {}): number =>
    meanLoss(yTrue, yPred, options, 'tweedieDeviance', (y, m) => tweedieUnitDeviance(y, m, options.power ?? 0)),
)

/**
 * Mean Poisson deviance (Tweedie power 1), for counts; needs $\hat y > 0$. As sklearn's `mean_poisson_deviance`.
 *
 * @param yTrue The observed counts $y_i \ge 0$.
 * @param yPred The predicted means $\hat y_i > 0$, one per count.
 * @param options `sampleWeight`, a weight per case.
 * @returns The mean Poisson deviance.
 *
 * @example Counts against predicted rates
 * print('Poisson deviance', poissonDeviance([2, 0, 1, 4], [0.5, 0.5, 2, 2]))
 */
export const poissonDeviance = defineMetric(
  errorInfo('poissonDeviance', 'Mean Poisson deviance', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'poissonDeviance', (y, m) => tweedieUnitDeviance(y, m, 1)),
)

/**
 * Mean gamma deviance (Tweedie power 2), a relative error; needs $y, \hat y > 0$. As sklearn's
 * `mean_gamma_deviance`.
 *
 * @param yTrue The true values $y_i > 0$.
 * @param yPred The predicted means $\hat y_i > 0$, one per true value.
 * @param options `sampleWeight`, a weight per case.
 * @returns The mean gamma deviance.
 *
 * @example It depends only on the ratio of prediction to truth
 * print('small values', gammaDeviance([2, 0.5, 1, 4], [0.5, 0.5, 2, 2]))
 * print('scaled by 100', gammaDeviance([200, 50, 100, 400], [50, 50, 200, 200]))
 */
export const gammaDeviance = defineMetric(
  errorInfo('gammaDeviance', 'Mean gamma deviance', robustNote),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'gammaDeviance', (y, m) => tweedieUnitDeviance(y, m, 2)),
)

// ── Forecasting ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Mean absolute percentage error $\frac{1}{n} \sum_t \lvert y_t - \hat y_t \rvert / \lvert y_t \rvert$
 * (percentage-errors), as a fraction (0.2 means 20%), like scikit-learn. Infinite when an actual is 0 (scikit-learn
 * divides by a small $\varepsilon$ instead).
 *
 * @param yTrue The actual values $y_t$.
 * @param yPred The forecasts $\hat y_t$, one per actual.
 * @param options `sampleWeight`, a weight per case.
 * @returns The mean absolute percentage error, as a fraction.
 *
 * @example Misses of 10% and 20%
 * print('MAPE', meanAbsolutePercentageError([100, 200], [110, 160]))
 */
export const meanAbsolutePercentageError = defineMetric(
  errorInfo('meanAbsolutePercentageError', 'Mean absolute percentage error', 'percentage-errors'),
  (yTrue: Data, yPred: Data, options: RegressionOptions = {}): number =>
    meanLoss(yTrue, yPred, options, 'meanAbsolutePercentageError', (y, p) => Math.abs(y - p) / Math.abs(y)),
)

/**
 * Symmetric MAPE $\frac{1}{n} \sum_t 2\lvert y_t - \hat y_t \rvert / (\lvert y_t \rvert + \lvert \hat y_t \rvert)$ as
 * a fraction in $[0, 2]$ (Makridakis 1993). A term with $y_t = \hat y_t = 0$ is $0/0$ and makes the result NaN.
 * Unweighted.
 *
 * @param yTrue The actual values $y_t$.
 * @param yPred The forecasts $\hat y_t$, one per actual.
 * @returns The symmetric MAPE, as a fraction.
 *
 * @example Over- and under-forecasting by the same amount
 * print('over by 10', symmetricMeanAbsolutePercentageError([100], [110]))
 * print('under by 10', symmetricMeanAbsolutePercentageError([100], [90]))
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

/**
 * Weighted MAPE $\sum_t \lvert y_t - \hat y_t \rvert / \sum_t \lvert y_t \rvert$: the total absolute error as a
 * fraction of the total volume. NaN when every actual is 0.
 *
 * @param yTrue The actual values $y_t$.
 * @param yPred The forecasts $\hat y_t$, one per actual.
 * @returns The weighted MAPE, as a fraction.
 *
 * @example A miss on a large value counts for more than on a small one
 * print('WMAPE', weightedMeanAbsolutePercentageError([100, 200], [110, 160]))
 * print('MAPE', meanAbsolutePercentageError([100, 200], [110, 160]))
 */
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

/**
 * The in-sample seasonal-naive scale $\frac{1}{T - m} \sum_{t > m} \lvert y_t - y_{t-m} \rvert^r$ with $r$ =
 * `power`: the mean error of forecasting each training value by the one a season earlier. Throws `DomainError` when
 * the series has no more than $m$ values.
 *
 * @param train The training series $y_1, \dots, y_T$.
 * @param season The seasonal period $m$ (1 for the non-seasonal naive forecast).
 * @param power The power $r$ of the errors: 1 for MASE, 2 for RMSSE.
 * @returns The scale.
 */
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
  /** The in-sample (training) series $y_1, \dots, y_T$ that sets the scale; it needs more than $m$ values. */
  train: Data
  /** Seasonal period $m$ (default 1, the non-seasonal naive forecast). */
  season?: number
}

/**
 * Mean absolute scaled error (Hyndman and Koehler 2006; scaled-errors-and-mean-absolute-scaled-error): the MAE of the
 * forecasts divided by $Q = \frac{1}{T - m} \sum_{t > m} \lvert y_t - y_{t-m} \rvert$, the in-sample MAE of the
 * seasonal naive forecast. Below 1 beats that naive forecast. NaN when the training series repeats exactly with
 * period $m$; throws `DomainError` when it has no more than $m$ values. Unweighted.
 *
 * @param yTrue The actual values over the forecast horizon.
 * @param yPred The forecasts, one per actual.
 * @param options `train`, the training series that sets the scale, and `season`, its period $m$ (default 1).
 * @returns The MASE.
 *
 * @example A training series whose naive forecast is off by 1 each step
 * print('MASE', meanAbsoluteScaledError([6, 7], [6.5, 6], { train: [1, 2, 3, 4, 5] }))
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
 * Root mean squared scaled error: the square root of the MSE of the forecasts over the in-sample MSE of the seasonal
 * naive forecast, the M5 competition's scale-free error (Makridakis et al. 2022). Throws `DomainError` when the
 * training series has no more than $m$ values. Unweighted.
 *
 * @param yTrue The actual values over the forecast horizon.
 * @param yPred The forecasts, one per actual.
 * @param options `train`, the training series that sets the scale, and `season`, its period $m$ (default 1).
 * @returns The RMSSE.
 *
 * @example A seasonal series of period 2
 * const train = [10, 20, 11, 21, 12, 22]
 * print('RMSSE, m = 2', rootMeanSquaredScaledError([13, 23], [13.5, 22], { train, season: 2 }))
 * print('RMSSE, m = 1', rootMeanSquaredScaledError([13, 23], [13.5, 22], { train }))
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
