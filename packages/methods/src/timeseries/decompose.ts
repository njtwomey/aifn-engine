/**
 * Differencing and its inverse, and seasonal decomposition: the classical moving-average method and a small STL
 * (seasonal-trend decomposition by loess).
 */

import { median } from 'aifn-compute/probability/stats'
import type { Decomposition } from 'aifn-compute/foundation/contracts'
import { tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { toVec, type VectorLike } from './inputs'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The lag-`lag` difference applied `order` times: ∇_s x_t = x_t − x_{t−s}. Each pass shortens the series by `lag`.
 * `difference(x, { lag: 12 })` is a seasonal difference; `{ order: 2 }` the second difference.
 */
export function difference(x: VectorLike, { lag = 1, order = 1 }: { lag?: number; order?: number } = {}): Vector {
  let xs = toVec(x, 'difference')
  for (let k = 0; k < order; k++) xs = xs.slice(lag).map((v, t) => v - xs[t])
  return tensor(xs)
}

/**
 * The inverse of one lag-`lag` difference: given the differences d and the `lag` values that preceded them,
 * x_t = d_t + x_{t−lag}. Returns the full series, the initial values first.
 */
export function undifference(d: VectorLike, initial: VectorLike, { lag = 1 }: { lag?: number } = {}): Vector {
  const ds = toVec(d, 'undifference')
  const x = toVec(initial, 'undifference')
  if (x.length !== lag)
    throw new ShapeError('undifference', `undifference: need ${lag} initial values, got ${x.length}`)
  for (const v of ds) x.push(v + x[x.length - lag])
  return tensor(x)
}

/**
 * A seasonal decomposition y = trend + seasonal + remainder (or × for multiplicative): the contract's `Decomposition`
 * (`components` trend and seasonal, `residual` the remainder, `original` y, `axis` t = 0 … n − 1), with the parts also
 * as named fields.
 */
export interface SeasonalDecomposition extends Decomposition {
  readonly trend: Vector
  readonly seasonal: Vector
  readonly remainder: Vector
  /** The seasonal pattern over one period, s₀ … s_{m−1} (indexed by t mod m). */
  readonly pattern: Vector
}

function seasonalDecomposition(
  method: string,
  y: number[],
  trend: number[],
  season: number[],
  remainder: number[],
  pattern: number[],
): SeasonalDecomposition {
  const parts = {
    trend: tensor(trend),
    seasonal: tensor(season),
    remainder: tensor(remainder),
    pattern: tensor(pattern),
  }
  return {
    kind: 'decomposition',
    method,
    axis: tensor(y.map((_, t) => t)),
    components: [
      { name: 'trend', values: parts.trend },
      { name: 'seasonal', values: parts.seasonal },
    ],
    residual: parts.remainder,
    original: tensor(y),
    ...parts,
  }
}

/** The centred moving average of length m (a 2×m average for even m); NaN within half a window of either end. */
function centredMovingAverage(x: number[], m: number): number[] {
  const n = x.length
  const out = new Array<number>(n).fill(NaN)
  const h = Math.floor(m / 2)
  for (let t = h; t < n - h; t++) {
    let s = 0
    if (m % 2 === 1) for (let k = -h; k <= h; k++) s += x[t + k] / m
    else {
      for (let k = -h + 1; k < h; k++) s += x[t + k] / m
      s += (x[t - h] + x[t + h]) / (2 * m)
    }
    out[t] = s
  }
  return out
}

/**
 * Classical decomposition (Kendall & Stuart; Hyndman & Athanasopoulos, 2021, §3.4): the trend is the centred moving
 * average of length m (NaN at the ends), the seasonal pattern the average detrended value at each position t mod m,
 * normalised to sum to zero (additive) or average one (multiplicative), and the remainder what is left.
 */
export function classicalDecomposition(
  y: VectorLike,
  period: number,
  { model = 'additive' }: { model?: 'additive' | 'multiplicative' } = {},
): SeasonalDecomposition {
  const ys = toVec(y, 'classicalDecomposition')
  if (ys.length < 2 * period)
    throw new DomainError('classicalDecomposition', 'classicalDecomposition: need at least two full periods')
  const add = model === 'additive'
  const trend = centredMovingAverage(ys, period)
  const sums = new Array<number>(period).fill(0)
  const counts = new Array<number>(period).fill(0)
  ys.forEach((v, t) => {
    if (Number.isNaN(trend[t])) return
    sums[t % period] += add ? v - trend[t] : v / trend[t]
    counts[t % period]++
  })
  let pattern = sums.map((s, k) => s / counts[k])
  const centre = pattern.reduce((a, b) => a + b, 0) / period
  pattern = pattern.map((p) => (add ? p - centre : p / centre))
  const seasonal = ys.map((_, t) => pattern[t % period])
  const remainder = ys.map((v, t) => (add ? v - trend[t] - seasonal[t] : v / (trend[t] * seasonal[t])))
  const method = model === 'additive' ? 'classical' : 'classical-multiplicative'
  return seasonalDecomposition(method, ys, trend, seasonal, remainder, pattern)
}

/**
 * Local linear regression (loess, Cleveland, 1979) of values v at positions 0 … n−1 with tricube weights over the q
 * nearest positions (times robustness weights w), evaluated at position `at` (which may lie outside the data). When
 * every weight in the window is zero it returns the nearest value.
 */
function loessAt(v: number[], w: number[], q: number, at: number): number {
  const n = v.length
  const k = Math.min(q, n)
  // The k nearest positions form a window [lo, lo + k).
  let lo = Math.round(at) - Math.floor(k / 2)
  lo = Math.max(0, Math.min(n - k, lo))
  const dmax = Math.max(Math.abs(at - lo), Math.abs(at - (lo + k - 1))) * (q > n ? q / n : 1) + 1e-12
  let sw = 0
  let sx = 0
  let sy = 0
  let sxx = 0
  let sxy = 0
  for (let i = lo; i < lo + k; i++) {
    const u = Math.abs(i - at) / dmax
    const wi = (u < 1 ? (1 - u ** 3) ** 3 : 0) * w[i]
    sw += wi
    sx += wi * i
    sy += wi * v[i]
    sxx += wi * i * i
    sxy += wi * i * v[i]
  }
  // Every weight in the window is zero (robustness weights can zero a whole window once the fit is nearly exact):
  // keep the nearest value, as the Fortran stless does, rather than return NaN.
  if (!(sw > 0)) return v[Math.max(0, Math.min(n - 1, Math.round(at)))]
  const mx = sx / sw
  const my = sy / sw
  const vxx = sxx / sw - mx * mx
  const slope = vxx > 1e-12 ? (sxy / sw - mx * my) / vxx : 0
  return my + slope * (at - mx)
}

const movingAverage = (x: number[], m: number): number[] =>
  x.slice(m - 1).map((_, t) => {
    let s = 0
    for (let k = 0; k < m; k++) s += x[t + k]
    return s / m
  })

/** Options of `stl`. */
export type StlOptions = {
  /** Seasonal loess span (odd, ≥ 7); default 7. Larger gives a more stable seasonal pattern. */
  seasonalSpan?: number
  /** Trend loess span; default the smallest odd integer ≥ 1.5m/(1 − 1.5/seasonalSpan) (Cleveland et al., 1990). */
  trendSpan?: number
  /** Inner-loop passes; default 2. */
  inner?: number
  /** Outer robustness passes with bisquare weights on the remainder; default 0 (no robustness). */
  robust?: number
}

/**
 * STL, reduced to its core (Cleveland, Cleveland, McRae & Terpenning, 1990, "STL: a seasonal-trend decomposition
 * procedure based on loess", J. Official Statistics 6): each inner pass detrends y, smooths every cycle-subseries (the
 * values at one position of the period) by loess, extended one period at each end; removes the low-frequency part of
 * that by moving averages of lengths m, m, 3 and a loess; and fits the trend by loess to the deseasonalised series.
 * Robust passes reweight by the bisquare of remainder / (6 · median |remainder|). Additive only; no jumps or degree
 * choices.
 */
export function stl(y: VectorLike, period: number, options: StlOptions = {}): SeasonalDecomposition {
  const ys = toVec(y, 'stl')
  const n = ys.length
  const m = period
  if (n < 2 * m) throw new DomainError('stl', 'stl: need at least two full periods')
  const ns = options.seasonalSpan ?? 7
  let nt = options.trendSpan ?? Math.ceil((1.5 * m) / (1 - 1.5 / ns))
  if (nt % 2 === 0) nt++
  let nl = m + 1
  if (nl % 2 === 0) nl++
  const inner = options.inner ?? 2
  const outer = options.robust ?? 0
  const scale = ys.reduce((a, v) => Math.max(a, Math.abs(v)), 0)
  let trend = new Array<number>(n).fill(0)
  let seasonal = new Array<number>(n).fill(0)
  let weights = new Array<number>(n).fill(1)
  for (let o = 0; o <= outer; o++) {
    for (let it = 0; it < inner; it++) {
      const detrended = ys.map((v, t) => v - trend[t])
      // Cycle-subseries smoothing, each extended by one value before and after: C has length n + 2m.
      const C = new Array<number>(n + 2 * m).fill(0)
      for (let k = 0; k < m; k++) {
        const idx: number[] = []
        for (let t = k; t < n; t += m) idx.push(t)
        const v = idx.map((t) => detrended[t])
        const w = idx.map((t) => weights[t])
        for (let j = -1; j <= v.length; j++) C[k + (j + 1) * m] = loessAt(v, w, ns, j)
      }
      // Low-pass filter of C: MA(m), MA(m), MA(3), then loess of span nl; its length is n.
      const low = movingAverage(movingAverage(movingAverage(C, m), m), 3)
      const ones = new Array<number>(low.length).fill(1)
      const L = low.map((_, t) => loessAt(low, ones, nl, t))
      seasonal = ys.map((_, t) => C[t + m] - L[t])
      const deseasonal = ys.map((v, t) => v - seasonal[t])
      trend = ys.map((_, t) => loessAt(deseasonal, weights, nt, t))
    }
    if (o < outer) {
      const r = ys.map((v, t) => Math.abs(v - trend[t] - seasonal[t]))
      // h = 6 · median |remainder|, floored at a rounding-level fraction of the data's scale: once the fit is exact up
      // to rounding, residuals of 1e-15 would otherwise set the scale and zero the weights of ordinary points.
      const h = Math.max(6 * median(r), 1e-9 * scale)
      // R's stl thresholds (Cleveland et al., 1990, Fortran stlrwt): weight 1 within 0.001·h, bisquare up to 0.999·h,
      // 0 beyond. A perfect fit (h = 0) keeps every weight at 1.
      weights = r.map((v) => {
        const u = h > 0 ? v / h : 0
        if (u <= 0.001) return 1
        return u <= 0.999 ? (1 - u * u) ** 2 : 0
      })
    }
  }
  const pattern = Array.from({ length: m }, (_, k) => {
    let s = 0
    let c = 0
    for (let t = k; t < n; t += m) {
      s += seasonal[t]
      c++
    }
    return s / c
  })
  const remainder = ys.map((v, t) => v - trend[t] - seasonal[t])
  return seasonalDecomposition('stl', ys, trend, seasonal, remainder, pattern)
}
