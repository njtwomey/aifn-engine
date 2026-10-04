/** Numbers the probability layers draw from aifn objects: a distribution's default range, its support, a signed area. */
import type { Univariate } from 'aifn-compute/probability/distributions'
import { toFlat, unwrap, type Value } from 'aifn-compute/foundation/tensor'
import type { Range } from '../viewport'
import type { AxisInterval } from './axis'

const numbers = (v: Value): number[] => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}

/** A univariate distribution's support as an interval of the real line (a circle's too), or undefined. */
export function supportOf(d: Univariate): AxisInterval | undefined {
  const s = d.support
  if (s.type === 'real') return { lower: -Infinity, upper: Infinity, lowerOpen: true, upperOpen: true }
  if (s.type === 'interval' || s.type === 'integers' || s.type === 'circle') {
    const lower = Math.min(...numbers(s.lower))
    const upper = Math.max(...numbers(s.upper))
    const open = s.type === 'interval' ? s : { lowerOpen: false, upperOpen: false }
    return {
      lower,
      upper,
      lowerOpen: !!open.lowerOpen || !Number.isFinite(lower),
      upperOpen: !!open.upperOpen || !Number.isFinite(upper),
    }
  }
  return undefined
}

/**
 * The x range a view of a univariate distribution shows by default: its 0.002–0.998 quantiles (or its moments), padded
 * by 5% for a density and to whole integers for a mass function, and never past a bounded end of its support.
 */
export function distributionRange(d: Univariate): Range {
  const ends = ((): Range => {
    try {
      const lo = Math.min(...numbers(d.quantile(0.002)))
      const hi = Math.max(...numbers(d.quantile(0.998)))
      if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) return [lo, hi]
    } catch {
      // No quantile (or it failed): fall back to the moments.
    }
    try {
      const m = numbers(d.mean())
      const s = numbers(d.stddev())
      const lo = Math.min(...m.map((v, i) => v - 5 * s[i]))
      const hi = Math.max(...m.map((v, i) => v + 5 * s[i]))
      if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) return [lo, hi]
    } catch {
      // No moments either.
    }
    return [-5, 5]
  })()
  const s = supportOf(d)
  let [lo, hi]: Range = d.discrete
    ? [Math.floor(ends[0]) - 1, Math.ceil(ends[1]) + 1]
    : [ends[0] - 0.05 * (ends[1] - ends[0]), ends[1] + 0.05 * (ends[1] - ends[0])]
  if (s && Number.isFinite(s.lower)) lo = Math.max(lo, d.discrete ? s.lower - 1 : s.lower)
  if (s && Number.isFinite(s.upper)) hi = Math.min(hi, d.discrete ? s.upper + 1 : s.upper)
  return [lo, hi]
}

/** The density (or mass) of `d` at each of `x`, with non-finite values (an infinite density at an end) as NaN. */
export function evaluate(d: Univariate, x: readonly number[]): number[] {
  return x.map((v) => {
    const p = unwrap(d.prob(v))
    const n = typeof p === 'number' ? p : toFlat(p)[0]
    return Number.isFinite(n) ? n : NaN
  })
}

export type SignedAreaResult = { net: number; positive: number; negative: number }

/**
 * The area between a sampled curve and zero by the trapezoid rule, split where the curve crosses zero (the crossing
 * found by linear interpolation), so `positive` and `negative` are each exact for the piecewise-linear curve.
 */
export function signedArea(x: ArrayLike<number>, y: ArrayLike<number>): SignedAreaResult {
  let positive = 0
  let negative = 0
  for (let i = 0; i + 1 < x.length; i++) {
    const [x0, x1, y0, y1] = [x[i], x[i + 1], y[i], y[i + 1]]
    if (![x0, x1, y0, y1].every(Number.isFinite)) continue
    const add = (a: number, b: number, w: number) => {
      const area = ((a + b) / 2) * w
      if (area >= 0) positive += area
      else negative -= area
    }
    if (y0 * y1 < 0) {
      const t = y0 / (y0 - y1)
      add(y0, 0, t * (x1 - x0))
      add(0, y1, (1 - t) * (x1 - x0))
    } else add(y0, y1, x1 - x0)
  }
  return { net: positive - negative, positive, negative }
}

/** The curve split at zero into its positive and negative parts, each closed along y = 0, for filling. */
export function signedParts(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
): { positive: number[][]; negative: number[][] } {
  const positive: number[][] = []
  const negative: number[][] = []
  for (let i = 0; i < x.length; i++) {
    const [xi, yi] = [x[i], y[i]]
    if (!Number.isFinite(xi) || !Number.isFinite(yi)) continue
    positive.push([xi, Math.max(yi, 0)])
    negative.push([xi, Math.min(yi, 0)])
    if (i + 1 < x.length && yi * y[i + 1] < 0) {
      const xc = xi + (yi / (yi - y[i + 1])) * (x[i + 1] - xi)
      positive.push([xc, 0])
      negative.push([xc, 0])
    }
  }
  return { positive, negative }
}
