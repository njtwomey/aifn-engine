/**
 * Differencing and its inverse, and seasonal decomposition: the classical moving-average method and a small STL
 * (seasonal-trend decomposition by loess).
 *
 * A decomposition splits $y_t = T_t + S_t + R_t$ (or $y_t = T_t S_t R_t$ for the multiplicative classical method) into
 * a trend $T_t$, a seasonal part $S_t$ of period $m$ and a remainder $R_t$, and returns them as the contract's
 * `Decomposition` with the parts also as named fields. The period is a whole number of observations, and both methods
 * need at least two full periods.
 */

import { median } from 'aifn-compute/probability/stats'
import type { Decomposition } from 'aifn-compute/foundation/contracts'
import { tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { toVec, type VectorLike } from './inputs'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The lag-$s$ difference $\nabla_s x_t = x_t - x_{t-s}$ applied `order` times, with $s$ = `lag`. Each pass shortens
 * the series by `lag`. `difference(x, { lag: 12 })` is a seasonal difference; `{ order: 2 }` the second difference.
 *
 * @param x The series.
 * @param options Differencing options.
 * @param options.lag The lag $s$ of each difference (default 1).
 * @param options.order How many times the difference is applied (default 1; 0 returns `x` unchanged).
 * @returns The differenced series, $\mathrm{order} \cdot s$ values shorter than `x`.
 *
 * @example Differences of the squares
 * const x = [1, 4, 9, 16, 25]
 * print('first difference:', difference(x))
 * print('second difference:', difference(x, { order: 2 }))
 * print('lag-2 difference:', difference(x, { lag: 2 }))
 */
export function difference(x: VectorLike, { lag = 1, order = 1 }: { lag?: number; order?: number } = {}): Vector {
  let xs = toVec(x, 'difference')
  for (let k = 0; k < order; k++) xs = xs.slice(lag).map((v, t) => v - xs[t])
  return tensor(xs)
}

/**
 * The inverse of one lag-$s$ difference ($s$ = `lag`): given the differences $d_t$ and the $s$ values that preceded
 * them, $x_t = d_t + x_{t-s}$. Throws `ShapeError` unless exactly $s$ initial values are given.
 *
 * @param d The differences $d_t$.
 * @param initial The first $s$ values of the series, which the differences continue.
 * @param options Options.
 * @param options.lag The lag $s$ of the difference being undone (default 1).
 * @returns The full series, the initial values first: $s$ values longer than `d`.
 *
 * @example Undo a first and a lag-2 difference
 * const x = [1, 4, 9, 16, 25]
 * print(undifference(difference(x), [1]))
 * print(undifference(difference(x, { lag: 2 }), [1, 4], { lag: 2 }))
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
 * A seasonal decomposition $y = \mathrm{trend} + \mathrm{seasonal} + \mathrm{remainder}$ (or a product, for
 * multiplicative): the contract's `Decomposition` (`components` trend and seasonal, `residual` the remainder,
 * `original` $y$, `axis` $t = 0, \dots, n - 1$), with the parts also as named fields.
 */
export interface SeasonalDecomposition extends Decomposition {
  /** The trend, one value per observation (NaN where the method has none, as at the ends of a moving average). */
  readonly trend: Vector
  /** The seasonal part, one value per observation: the pattern repeated (classical) or varying slowly (STL). */
  readonly seasonal: Vector
  /** What is left: $y$ minus (or divided by) trend and seasonal part. */
  readonly remainder: Vector
  /** The seasonal pattern over one period, $s_0, \dots, s_{m-1}$ (indexed by $t \bmod m$). */
  readonly pattern: Vector
}

/**
 * Assemble a `SeasonalDecomposition` from its parts.
 *
 * @param method The method's name, as the contract's `method` field (`'classical'`, `'stl'`, ...).
 * @param y The original series, $n$ values.
 * @param trend The trend, $n$ values.
 * @param season The seasonal part, $n$ values.
 * @param remainder The remainder, $n$ values.
 * @param pattern The seasonal pattern over one period, $m$ values.
 * @returns The decomposition, with the parts as tensors.
 */
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

/**
 * The centred moving average of length $m$ (a $2 \times m$ average for even $m$, the two end values weighted
 * $\frac{1}{2m}$); NaN within half a window of either end.
 *
 * @param x The series.
 * @param m The window length.
 * @returns The averages, aligned with `x`: the first and last $\lfloor m/2 \rfloor$ are NaN.
 */
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
 * average of length $m$ (NaN within $\lfloor m/2 \rfloor$ of each end), the seasonal pattern the average detrended
 * value ($y_t - T_t$, or $y_t / T_t$) at each position $t \bmod m$, normalised to sum to zero (additive) or average
 * one (multiplicative), and the remainder what is left (NaN where the trend is). Throws `DomainError` for fewer than
 * two full periods. As statsmodels' `seasonal_decompose`.
 *
 * @param y The series.
 * @param period The season length $m$, in observations.
 * @param options Options.
 * @param options.model `'additive'` (default), $y = T + S + R$, or `'multiplicative'`, $y = T \cdot S \cdot R$.
 * @returns The decomposition, with `method` `'classical'` or `'classical-multiplicative'`.
 *
 * @example A sine of period 4 on a straight line
 * const y = Array.from({ length: 16 }, (_, t) => 0.5 * t + Math.sin((Math.PI * t) / 2))
 * const dec = classicalDecomposition(y, 4)
 * print('pattern:', dec.pattern)
 * print('trend:', dec.trend)
 *
 * @example Multiplicative seasonal factors
 * const y = Array.from({ length: 12 }, (_, t) => (10 + t) * [1.2, 1, 0.8, 1][t % 4])
 * print('pattern:', classicalDecomposition(y, 4, { model: 'multiplicative' }).pattern)
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
 * Local linear regression (loess, Cleveland, 1979) of values $v$ at positions $0, \dots, n - 1$ with tricube weights
 * over the $q$ nearest positions (times robustness weights $w$), evaluated at position `at` (which may lie outside the
 * data). When $q > n$ every position is in the window and the tricube bandwidth is widened by the factor $q / n$. When
 * every weight in the window is zero it returns the nearest value.
 *
 * @param v The values, one per position.
 * @param w The robustness weights, one per position (all 1 for a plain fit).
 * @param q The span: the number of nearest positions in the window.
 * @param at The position the fit is evaluated at; a fraction or a value outside $[0, n - 1]$ is allowed.
 * @returns The fitted value at `at`.
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

/**
 * The trailing moving average of length $m$, $\frac{1}{m} \sum_{k=0}^{m-1} x_{t+k}$, over every full window.
 *
 * @param x The series.
 * @param m The window length.
 * @returns The $n - m + 1$ averages, the first over $x_0, \dots, x_{m-1}$.
 */
const movingAverage = (x: number[], m: number): number[] =>
  x.slice(m - 1).map((_, t) => {
    let s = 0
    for (let k = 0; k < m; k++) s += x[t + k]
    return s / m
  })

/** Options of `stl`. */
export type StlOptions = {
  /**
   * Seasonal loess span, in cycles; default 7. Larger gives a more stable seasonal pattern. Cleveland et al. (1990) ask
   * for an odd value of at least 7; this is not checked.
   */
  seasonalSpan?: number
  /**
   * Trend loess span; default the smallest odd integer $\ge 1.5m / (1 - 1.5 / n_s)$, with $n_s$ the seasonal span
   * (Cleveland et al., 1990). An even value, given or computed, is raised by one.
   */
  trendSpan?: number
  /** Inner-loop passes; default 2. */
  inner?: number
  /** Outer robustness passes with bisquare weights on the remainder; default 0 (no robustness). */
  robust?: number
}

/**
 * STL, reduced to its core (Cleveland, Cleveland, McRae & Terpenning, 1990, "STL: a seasonal-trend decomposition
 * procedure based on loess", J. Official Statistics 6): each inner pass detrends $y$, smooths every cycle-subseries
 * (the values at one position of the period) by loess, extended one period at each end; removes the low-frequency
 * part of that by moving averages of lengths $m$, $m$, 3 and a loess of span the smallest odd integer $\ge m + 1$;
 * and fits the trend by loess to the deseasonalised series. Robust passes reweight by the bisquare of
 * $R_t / h$, $h = 6 \operatorname{median} \lvert R_t \rvert$ (weight 1 below $0.001h$ and 0 above $0.999h$, as R's
 * `stl`). Additive only, with local-linear fits throughout; no jumps or degree choices. The returned `pattern` is the
 * mean seasonal value at each position. Throws `DomainError` for fewer than two full periods.
 *
 * @param y The series.
 * @param period The season length $m$, in observations.
 * @param options The loess spans and the numbers of inner and robustness passes (see `StlOptions`).
 * @returns The decomposition, with `method` `'stl'`.
 *
 * @example A sine of period 4 on a straight line
 * const y = Array.from({ length: 24 }, (_, t) => 0.5 * t + Math.sin((Math.PI * t) / 2))
 * const dec = stl(y, 4)
 * print('pattern:', dec.pattern)
 * print('trend:', dec.trend)
 * print('largest |remainder|:', Math.max(...toFlat(dec.remainder).map(Math.abs)))
 *
 * @example Robustness passes keep an outlier out of the seasonal pattern
 * const y = Array.from({ length: 24 }, (_, t) => 0.5 * t + Math.sin((Math.PI * t) / 2) + (t === 10 ? 8 : 0))
 * print('plain pattern:', stl(y, 4).pattern)
 * const robust = stl(y, 4, { robust: 5 })
 * print('robust pattern:', robust.pattern)
 * print('robust remainder at the outlier:', toFlat(robust.remainder)[10])
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
