/**
 * Exponential smoothing in the component form of Hyndman & Athanasopoulos (2021), "Forecasting: Principles and
 * Practice", 3rd ed., §8: level $\ell$, optionally damped trend $b$ and additive or multiplicative seasonal $s$ of
 * period $m$. Simple exponential smoothing, Holt's linear (and damped) trend and Holt–Winters are special cases of one
 * recursion. The prediction intervals are those of the matching additive-error ETS model (Hyndman et al., 2008), with
 * $\hat\sigma^2$ the mean squared one-step error; the parameters are fitted by least squares, as statsmodels'
 * `ExponentialSmoothing` does.
 */

import { normalQuantile } from 'aifn-compute/numerics/special'
import { mean as meanOf, tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { logit, sigmoid } from 'aifn-compute/numerics/special'
import { simplexFit, type FitState } from './fit'
import { toVec, type VectorLike } from './inputs'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The structure of an exponential-smoothing model: which components it has. */
export type SmoothingStructure = {
  /** `none` (simple), `additive` (Holt) or `damped` (Gardner & McKenzie, 1985). Default `none`. */
  trend?: 'none' | 'additive' | 'damped'
  /** `none`, `additive` or `multiplicative` (Holt–Winters, Winters, 1960). Default `none`. */
  seasonal?: 'none' | 'additive' | 'multiplicative'
  /** The seasonal period $m$, at least 2 (required with a seasonal component). */
  period?: number
}

/** An exponential-smoothing model: its structure and smoothing parameters, each in $[0, 1]$ (not checked). */
export type SmoothingSpec = SmoothingStructure & {
  /** Level smoothing $\alpha$. */
  alpha: number
  /** Trend smoothing $\beta^*$ (default 0.1; ignored without a trend). */
  beta?: number
  /** Seasonal smoothing $\gamma$ (default 0.1; ignored without a season). */
  gamma?: number
  /** Damping $\phi$ (only with `trend: 'damped'`; 1 gives Holt's trend). Default 0.98. */
  phi?: number
  /**
   * Initial states $\ell_0$, $b_0$ and the $m$ seasonal states $s_{1-m}, \dots, s_0$ (oldest first); each one left out
   * comes from the heuristic on the first two seasons (or the first two values).
   */
  initial?: { level?: number; trend?: number; seasonal?: VectorLike }
}

/** An exponential-smoothing fit and its forecasts. */
export type SmoothingResult = {
  /** One-step-ahead forecasts $\hat{y}_{t \mid t-1}$, aligned with $y$. */
  fitted: Vector
  /** The one-step errors $y_t - \hat{y}_{t \mid t-1}$. */
  residuals: Vector
  /** The level $\ell_t$ after each observation. */
  level: Vector
  /** The trend $b_t$ after each observation (zeros without a trend). */
  trend: Vector
  /** The seasonal state $s_t$ after each observation (zeros without a season). */
  season: Vector
  /** The sum of squared residuals, $\sum_t (y_t - \hat{y}_{t \mid t-1})^2$. */
  sse: number
  /** The residual variance $\hat\sigma^2 = \mathrm{SSE}/n$. */
  sigma2: number
  /** Forecasts $\hat{y}_{T+h \mid T}$, $h = 1, \dots, \mathrm{horizon}$. */
  forecast: Vector
  /** The lower ends of the prediction intervals at `level`, one per forecast. */
  lower: Vector
  /** The upper ends of the prediction intervals at `level`, one per forecast. */
  upper: Vector
  /**
   * True when the intervals are the additive-error formula applied to a multiplicative-seasonal model, which is only
   * an approximation (Hyndman et al., 2008, §6.4 give no closed form there).
   */
  approximateIntervals: boolean
}

/**
 * A `SmoothingSpec` with its defaults filled in: no trend gives $\beta^* = 0$ and $\phi = 1$, no season $\gamma = 0$
 * and period 1. `initial` is kept as given.
 */
type Parsed = Required<Omit<SmoothingSpec, 'initial'>> & { initial: SmoothingSpec['initial'] }

/**
 * Fill the defaults of a `SmoothingSpec`. Throws `DomainError` for a seasonal model without a period of at least 2.
 *
 * @param spec The model as given by the caller.
 * @returns The model with every parameter set.
 */
function parse(spec: SmoothingSpec): Parsed {
  const trend = spec.trend ?? 'none'
  const seasonal = spec.seasonal ?? 'none'
  if (seasonal !== 'none' && !(spec.period && spec.period >= 2))
    throw new DomainError('exponentialSmoothing', 'exponentialSmoothing: a seasonal model needs period ≥ 2')
  return {
    alpha: spec.alpha,
    beta: trend === 'none' ? 0 : (spec.beta ?? 0.1),
    gamma: seasonal === 'none' ? 0 : (spec.gamma ?? 0.1),
    phi: trend === 'damped' ? (spec.phi ?? 0.98) : 1,
    trend,
    seasonal,
    period: seasonal === 'none' ? 1 : spec.period!,
    initial: spec.initial,
  }
}

/**
 * Initial states from the first two seasons (the first two values without a season): level the mean of the first
 * season, trend the change in seasonal means per step, and seasonal states the first season's values relative to the
 * straight line through its centre; the level is then moved from the season's centre to just before $t = 1$. Values
 * given in `p.initial` replace the heuristic ones. Throws `DomainError` for fewer than $2m$ (or 2) observations.
 *
 * @param y The series.
 * @param p The parsed model: its structure, period and `initial`.
 * @returns The initial level `l`, trend `b` and the $m$ seasonal states `s`, oldest first (empty without a season).
 */
function initialStates(y: number[], p: Parsed) {
  const seasonal = p.seasonal !== 'none'
  const m = p.period
  const need = seasonal ? 2 * m : 2
  if (y.length < need)
    throw new DomainError('exponentialSmoothing', `exponentialSmoothing: need at least ${need} observations`)
  const first = meanOf(tensor(y.slice(0, m)))
  const second = meanOf(tensor(y.slice(m, 2 * m)))
  let b = p.trend !== 'none' ? (second - first) / m : 0
  let s = seasonal
    ? y.slice(0, m).map((v, j) => {
        const line = first + b * (j - (m - 1) / 2)
        return p.seasonal === 'additive' ? v - line : v / line
      })
    : []
  let l = first - b * ((m - 1) / 2 + 1)
  if (p.initial?.level !== undefined) l = p.initial.level
  if (p.initial?.trend !== undefined) b = p.initial.trend
  if (p.initial?.seasonal !== undefined) s = toVec(p.initial.seasonal, 'exponentialSmoothing')
  return { l, b, s }
}

/**
 * Run the smoothing recursion over $y$ and forecast `horizon` steps. With $\ell^- = \ell_{t-1} + \phi b_{t-1}$:
 * $\hat{y}_{t \mid t-1} = \ell^- + s_{t-m}$ (or $\ell^- s_{t-m}$),
 * $\ell_t = \alpha(y_t - s_{t-m}) + (1 - \alpha)\ell^-$,
 * $b_t = \beta^*(\ell_t - \ell_{t-1}) + (1 - \beta^*)\phi b_{t-1}$ and
 * $s_t = \gamma(y_t - \ell^-) + (1 - \gamma)s_{t-m}$ (for multiplicative seasonality, $y_t / s_{t-m}$ and
 * $y_t / \ell^-$ in place of the differences). Forecasts
 * $\hat{y}_{T+h \mid T} = \ell_T + \phi_h b_T + s_{T+h-m(k+1)}$, $k = \lfloor (h - 1)/m \rfloor$ (the seasonal term a
 * factor for multiplicative), with $\phi_h = \phi + \dots + \phi^h$. Interval variances
 * $\hat\sigma^2 (1 + \sum_{j=1}^{h-1} c_j^2)$, $c_j = \alpha + \alpha\beta^* \phi_j + \gamma [j \bmod m = 0]$
 * (Hyndman et al., 2008, Table 6.1, class 1), which are only approximate for multiplicative seasonality.
 *
 * @param y The series, at least $2m$ values with a season and 2 without.
 * @param spec The model: its structure, smoothing parameters and optional initial states.
 * @param options Forecast options.
 * @param options.horizon The number of steps to forecast past the end of $y$ (default 0, no forecasts).
 * @param options.level The coverage of the prediction intervals (default 0.95).
 * @returns The fitted values and states over $y$, the error summaries, and the forecasts with their intervals.
 *
 * @example Simple exponential smoothing of a short series
 * const r = exponentialSmoothing([3, 5, 4, 6, 5, 7], { alpha: 0.5 }, { horizon: 3 })
 * print('fitted:', r.fitted)
 * print('level:', r.level)
 * print('forecast:', r.forecast)
 * print('95% interval:', r.lower, r.upper)
 *
 * @example Holt–Winters follows a trend and a season exactly
 * const y = Array.from({ length: 16 }, (_, t) => 10 + 0.5 * t + [2, 0, -2, 0][t % 4])
 * const spec = { trend: 'additive', seasonal: 'additive', period: 4, alpha: 0.3, beta: 0.1, gamma: 0.1 }
 * print('forecast:', exponentialSmoothing(y, spec, { horizon: 4 }).forecast)
 * print('truth:   ', [16, 17, 18, 19].map((t) => 10 + 0.5 * t + [2, 0, -2, 0][t % 4]))
 */
export function exponentialSmoothing(
  y: VectorLike,
  spec: SmoothingSpec,
  { horizon = 0, level = 0.95 }: { horizon?: number; level?: number } = {},
): SmoothingResult {
  const ys = toVec(y, 'exponentialSmoothing')
  const p = parse(spec)
  const { alpha, beta, gamma, phi, period: m } = p
  const seasonal = p.seasonal !== 'none'
  const add = p.seasonal !== 'multiplicative'
  let { l, b, s } = initialStates(ys, p)
  const fitted: number[] = []
  const lv: number[] = []
  const tr: number[] = []
  const se: number[] = []
  for (const v of ys) {
    const base = l + phi * b
    const sOld = seasonal ? s[0] : add ? 0 : 1
    fitted.push(add ? base + sOld : base * sOld)
    const lNew = add ? alpha * (v - sOld) + (1 - alpha) * base : alpha * (v / sOld) + (1 - alpha) * base
    const bNew = p.trend !== 'none' ? beta * (lNew - l) + (1 - beta) * phi * b : 0
    let sNew = sOld
    if (seasonal) {
      sNew = add ? gamma * (v - base) + (1 - gamma) * sOld : gamma * (v / base) + (1 - gamma) * sOld
      s = [...s.slice(1), sNew]
    }
    l = lNew
    b = bNew
    lv.push(l)
    tr.push(b)
    se.push(sNew)
  }
  const residuals = ys.map((v, i) => v - fitted[i])
  const sse = residuals.reduce((a, r) => a + r * r, 0)
  const sigma2 = sse / ys.length
  const forecast: number[] = []
  const sd: number[] = []
  let damp = 0
  let acc = 1
  for (let h = 1; h <= horizon; h++) {
    damp += phi ** h
    const baseH = l + (p.trend !== 'none' ? damp * b : 0)
    const sh = seasonal ? s[(h - 1) % m] : add ? 0 : 1
    forecast.push(add ? baseH + sh : baseH * sh)
    sd.push(Math.sqrt(sigma2 * acc))
    // c_h for the next horizon: φ_h = φ + … + φ^h is `damp` now.
    const c = alpha + (p.trend !== 'none' ? alpha * beta * damp : 0) + (seasonal && h % m === 0 ? gamma : 0)
    acc += c * c
  }
  const z = normalQuantile(0.5 + level / 2) as number
  return {
    fitted: tensor(fitted),
    residuals: tensor(residuals),
    level: tensor(lv),
    trend: tensor(tr),
    season: tensor(se),
    sse,
    sigma2,
    forecast: tensor(forecast),
    lower: tensor(forecast.map((f, i) => f - z * sd[i])),
    upper: tensor(forecast.map((f, i) => f + z * sd[i])),
    approximateIntervals: p.seasonal === 'multiplicative',
  }
}

/**
 * A least-squares fitter of the smoothing parameters as a traceable algorithm: Nelder–Mead on the SSE of the one-step
 * forecasts over logit coordinates, so $\alpha$, $\beta^*$, $\gamma$ stay in $(0, 1)$ and $\phi$ in $(0.8, 0.995)$ (the
 * range of Hyndman et al., 2008, §2.2). It starts from $\alpha = 0.3$, $\beta^* = \gamma = 0.1$ and $\phi = 0.95$.
 * Initial states follow the heuristic and are not fitted. Each state's `params` is the spec at the best vertex.
 * `init` takes no start (`run(alg, undefined, steps)`).
 *
 * @param y The series.
 * @param structure Which components the model has (trend, season, period); only their parameters are fitted.
 * @returns The algorithm; its state is a `FitState` whose `params` is a `SmoothingSpec`.
 *
 * @example Fit Holt's linear trend to a noisy line
 * const { x } = simulateArma(stream(1), { ar: [0.5] }, 60)
 * const y = toFlat(x).map((v, t) => 10 + 0.1 * t + v)
 * const s = run(exponentialSmoothingFitSteps(y, { trend: 'additive' }), undefined, 500)
 * print('α =', s.params.alpha, 'β* =', s.params.beta)
 * print('SSE =', s.objective, 'converged:', s.converged, 'after', s.t, 'steps')
 */
export function exponentialSmoothingFitSteps(
  y: VectorLike,
  structure: SmoothingStructure,
): Algorithm<void, FitState<SmoothingSpec>> {
  const ys = toVec(y, 'exponentialSmoothingFitSteps')
  const trend = structure.trend ?? 'none'
  const seasonal = structure.seasonal ?? 'none'
  const names: ('alpha' | 'beta' | 'gamma' | 'phi')[] = ['alpha']
  if (trend !== 'none') names.push('beta')
  if (seasonal !== 'none') names.push('gamma')
  if (trend === 'damped') names.push('phi')
  const decode = (u: number[]): SmoothingSpec => {
    const spec: SmoothingSpec = { ...structure, alpha: 0 }
    names.forEach((k, i) => (spec[k] = k === 'phi' ? 0.8 + 0.195 * sigmoid(u[i]) : sigmoid(u[i])))
    return spec
  }
  const start = names.map((k) => (k === 'alpha' ? logit(0.3) : k === 'phi' ? logit((0.95 - 0.8) / 0.195) : logit(0.1)))
  return simplexFit('exponential-smoothing', (u) => exponentialSmoothing(ys, decode(u)).sse, decode, start, 1e-10)
}

export type { FitState }
