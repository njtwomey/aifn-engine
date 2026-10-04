/**
 * Exponential smoothing in the component form of Hyndman & Athanasopoulos (2021), "Forecasting: Principles and
 * Practice", 3rd ed., §8: level ℓ, optionally damped trend b and additive or multiplicative seasonal s of period m.
 * Simple exponential smoothing, Holt's linear (and damped) trend and Holt–Winters are special cases of one recursion.
 */

import { normalQuantile } from 'aifn-compute/numerics/special'
import { mean as meanOf, tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { logit, sigmoid } from 'aifn-compute/numerics/special'
import { simplexFit, type FitState } from './fit'
import { toVec, type VectorLike } from './inputs'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The structure of an exponential-smoothing model. */
export type SmoothingStructure = {
  /** `none` (simple), `additive` (Holt) or `damped` (Gardner & McKenzie, 1985). Default `none`. */
  trend?: 'none' | 'additive' | 'damped'
  /** `none`, `additive` or `multiplicative` (Holt–Winters, Winters, 1960). Default `none`. */
  seasonal?: 'none' | 'additive' | 'multiplicative'
  /** The seasonal period m (required with a seasonal component). */
  period?: number
}

/** An exponential-smoothing model: its structure and smoothing parameters, each in [0, 1]. */
export type SmoothingSpec = SmoothingStructure & {
  /** Level smoothing α. */
  alpha: number
  /** Trend smoothing β* (ignored without a trend). */
  beta?: number
  /** Seasonal smoothing γ (ignored without a season). */
  gamma?: number
  /** Damping φ (only with `trend: 'damped'`; 1 gives Holt's trend). Default 0.98. */
  phi?: number
  /** Initial states; by default a heuristic from the first two seasons (or the first two values). */
  initial?: { level?: number; trend?: number; seasonal?: VectorLike }
}

/** An exponential-smoothing fit and its forecasts. */
export type SmoothingResult = {
  /** One-step-ahead forecasts ŷ_{t|t−1}, aligned with y. */
  fitted: Vector
  /** y_t − ŷ_{t|t−1}. */
  residuals: Vector
  /** The level ℓ_t after each observation. */
  level: Vector
  /** The trend b_t after each observation (zeros without a trend). */
  trend: Vector
  /** The seasonal state s_t after each observation (zeros, or ones for multiplicative, without a season). */
  season: Vector
  /** Σ residuals². */
  sse: number
  /** The residual variance σ̂² = SSE/n. */
  sigma2: number
  /** Forecasts ŷ_{T+h|T}, h = 1 … horizon. */
  forecast: Vector
  /** Prediction interval at `level`. */
  lower: Vector
  upper: Vector
  /**
   * True when the intervals are the additive-error formula applied to a multiplicative-seasonal model, which is only
   * an approximation (Hyndman et al., 2008, §6.4 give no closed form there).
   */
  approximateIntervals: boolean
}

type Parsed = Required<Omit<SmoothingSpec, 'initial'>> & { initial: SmoothingSpec['initial'] }

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
 * Initial states from the first two seasons: level the mean of the first season, trend the change in seasonal means
 * per step, and seasonal states the first season's values relative to the straight line through its centre; the
 * level is then moved from the season's centre to just before t = 1.
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
 * Run the smoothing recursion over y and forecast `horizon` steps. With ℓ⁻ = ℓ_{t−1} + φ b_{t−1}:
 * ŷ = ℓ⁻ + s_{t−m} (or ℓ⁻ s_{t−m}); ℓ_t = α(y_t − s_{t−m}) + (1 − α)ℓ⁻; b_t = β*(ℓ_t − ℓ_{t−1}) + (1 − β*)φ b_{t−1};
 * s_t = γ(y_t − ℓ⁻) + (1 − γ)s_{t−m} (divisions for multiplicative). Forecasts ŷ_{T+h} = ℓ_T + φ_h b_T + s, with
 * φ_h = φ + … + φ^h. Interval variances σ²(1 + Σ_{j<h} c_j²), c_j = α + αβ* φ_j + γ·[j mod m = 0] (Hyndman et al.,
 * 2008, Table 6.1, class 1), at `level` (default 0.95).
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
 * forecasts over logit coordinates, so α, β*, γ stay in (0, 1) and φ in (0.8, 0.995) (the range of Hyndman et al.,
 * 2008, §2.2). Initial states follow the heuristic. Each state's `params` is the spec at the best vertex.
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
