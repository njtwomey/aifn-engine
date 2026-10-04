/**
 * Adaptive quadrature with error estimates: adaptive Simpson (local, depth-first) and globally adaptive Gauss–Kronrod
 * 7–15 (bisect the interval with the largest error estimate, as QUADPACK's QAG; Piessens et al., 1983, "QUADPACK"),
 * both traceable, plus `integrate`, which handles infinite limits by a change of variables (QUADPACK's QAGI).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { EPS, TINY } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { romberg, type Integrand } from './rules'

/** One piece of the subdivision: the interval, its estimate and its error estimate. */
export type Interval = {
  /** Left endpoint of interval. */
  a: number
  /** Right endpoint of interval. */
  b: number
  /** Estimated integral value over $[a, b]$. */
  value: number
  /** Estimated absolute error over $[a, b]$. */
  error: number
}

// ---------------------------------------------------------------------------------------------------------------------
// Adaptive Simpson.

type Pending = {
  a: number
  b: number
  fa: number
  fm: number
  fb: number
  whole: number
  tolerance: number
  depth: number
}

/** The state of `adaptiveSimpson`. */
export type AdaptiveSimpsonState = Status & {
  /** Current step count. */
  t: Size
  /** Intervals accepted so far, in the order they were accepted. */
  accepted: Interval[]
  /** Intervals still to examine (a stack; the last is examined next). */
  pending: Pending[]
  /** The sum of the accepted estimates (the integral once `pending` is empty). */
  value: number
  /** The sum of the accepted error estimates. */
  error: number
  /** The interval examined on the last step and whether it was split. */
  examined: { a: number; b: number; split: boolean } | null
  /** Cumulative count of integrand evaluations. */
  evaluations: number
  /** True once every interval met its tolerance (none was accepted only because of the depth limit). */
  converged: boolean
  /** Intervals accepted at the depth limit without meeting their tolerance. */
  unresolved: number
  /** True once the accumulated estimate is not finite. */
  diverged: boolean
}

/**
 * Adaptive Simpson quadrature (Kuncir, 1962; Lyness, 1969): on each interval compare Simpson's rule $S$ with the sum
 * of Simpson's rule on its halves $S_2$; if $|S_2 - S| \le 15 \cdot \text{tol}$ accept $S_2 + (S_2 - S)/15$ (Richardson)
 * with error estimate $|S_2 - S|/15$, else split with $\text{tol}$ halved. Each step examines one interval (depth
 * first). `init` takes `{ a, b }`.
 *
 * @param f The univariate real integrand function.
 * @param options Options controlling error tolerance and maximum recursion depth.
 * @param options.tolerance Error tolerance target (default 1e-10).
 * @param options.maxDepth Maximum recursion subdivision depth (default 50).
 * @returns An `Algorithm` stepping through adaptive Simpson subdivision.
 *
 * @example Adaptively integrate a function with a sharp feature
 * const alg = adaptiveSimpson((x) => 1 / (1 + 100 * x * x))
 * const state = run(alg, { a: -1, b: 1 }, 50)
 * print('integral =', state.value)
 */
export function adaptiveSimpson(
  f: Integrand,
  { tolerance = 1e-10, maxDepth = 50 }: { tolerance?: number; maxDepth?: number } = {},
): Algorithm<{ a: number; b: number }, AdaptiveSimpsonState> {
  const simpson = (a: number, fa: number, fm: number, b: number, fb: number) => ((b - a) / 6) * (fa + 4 * fm + fb)
  return {
    name: 'adaptive-simpson',
    init: ({ a, b }) => {
      const fa = f(a)
      const fb = f(b)
      const fm = f((a + b) / 2)
      return {
        t: 0,
        accepted: [],
        pending: [{ a, b, fa, fm, fb, whole: simpson(a, fa, fm, b, fb), tolerance, depth: 0 }],
        value: 0,
        error: 0,
        examined: null,
        evaluations: 3,
        converged: false,
        unresolved: 0,
        diverged: false,
      }
    },
    step: (s) => {
      const pending = s.pending.slice(0, -1)
      const p = s.pending[s.pending.length - 1]
      const m = (p.a + p.b) / 2
      const flm = f((p.a + m) / 2)
      const frm = f((m + p.b) / 2)
      const left = simpson(p.a, p.fa, flm, m, p.fm)
      const right = simpson(m, p.fm, frm, p.b, p.fb)
      const delta = left + right - p.whole
      const ok = Math.abs(delta) <= 15 * p.tolerance
      const atLimit = p.depth >= maxDepth
      let { accepted, value, error, unresolved } = s
      if (ok || atLimit || !Number.isFinite(delta)) {
        const estimate = left + right + delta / 15
        accepted = [...accepted, { a: p.a, b: p.b, value: estimate, error: Math.abs(delta) / 15 }]
        value += estimate
        error += Math.abs(delta) / 15
        if (!ok) unresolved++
      } else {
        const tol = p.tolerance / 2
        // Push the right half first so the left half is examined next.
        pending.push({ a: m, b: p.b, fa: p.fm, fm: frm, fb: p.fb, whole: right, tolerance: tol, depth: p.depth + 1 })
        pending.push({ a: p.a, b: m, fa: p.fa, fm: flm, fb: p.fm, whole: left, tolerance: tol, depth: p.depth + 1 })
      }
      return {
        t: s.t + 1,
        accepted,
        pending,
        value,
        error,
        examined: { a: p.a, b: p.b, split: !(ok || atLimit || !Number.isFinite(delta)) },
        evaluations: s.evaluations + 2,
        converged: pending.length === 0 && unresolved === 0,
        unresolved,
        diverged: !Number.isFinite(value),
      }
    },
    // Finished (every interval accepted) even when some were accepted only at the depth limit.
    done: (s) => s.pending.length === 0,
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Gauss–Kronrod 7–15.

// QUADPACK qk15 (Piessens et al., 1983): Kronrod abscissae (the odd entries are the 7-point Gauss abscissae) and weights.
const XGK = [
  0.9914553711208126, 0.9491079123427585, 0.8648644233597691, 0.7415311855993945, 0.5860872354676911,
  0.4058451513773972, 0.20778495500789848, 0,
]
const WGK = [
  0.022935322010529224, 0.06309209262997856, 0.10479001032225019, 0.14065325971552592, 0.1690047266392679,
  0.19035057806478542, 0.20443294007529889, 0.20948214108472782,
]
const WG = [0.1294849661688697, 0.27970539148927664, 0.3818300505051189, 0.4179591836734694]
const EPMACH = EPS
const UFLOW = TINY

/**
 * The 15-point Kronrod estimate on $[a, b]$ and QUADPACK's error estimate from its difference with the embedded 7-point
 * Gauss estimate, scaled as in `qk15` (the $(200 \cdot |K - G|/\text{resasc})^{1.5}$ heuristic).
 *
 * @param f The univariate real integrand function.
 * @param a The lower bound of the subinterval.
 * @param b The upper bound of the subinterval.
 * @returns An `Interval` holding endpoints, 15-point Kronrod estimate, and estimated error.
 *
 * @example Evaluate 15-point Gauss-Kronrod estimate on [0, 1]
 * const res = kronrod15((x) => x * x, 0, 1)
 * print('value =', res.value)
 */
export function kronrod15(f: Integrand, a: number, b: number): Interval {
  const centre = 0.5 * (a + b)
  const half = 0.5 * (b - a)
  const absHalf = Math.abs(half)
  const fc = f(centre)
  let resg = fc * WG[3]
  let resk = fc * WGK[7]
  let resabs = Math.abs(resk)
  const fv1 = new Array<number>(7)
  const fv2 = new Array<number>(7)
  for (let j = 0; j < 7; j++) {
    const dx = half * XGK[j]
    const f1 = f(centre - dx)
    const f2 = f(centre + dx)
    fv1[j] = f1
    fv2[j] = f2
    resk += WGK[j] * (f1 + f2)
    resabs += WGK[j] * (Math.abs(f1) + Math.abs(f2))
    if (j % 2 === 1) resg += WG[(j - 1) / 2] * (f1 + f2)
  }
  const mean = resk * 0.5
  let resasc = WGK[7] * Math.abs(fc - mean)
  for (let j = 0; j < 7; j++) resasc += WGK[j] * (Math.abs(fv1[j] - mean) + Math.abs(fv2[j] - mean))
  const value = resk * half
  resabs *= absHalf
  resasc *= absHalf
  let error = Math.abs((resk - resg) * half)
  if (resasc !== 0 && error !== 0) error = resasc * Math.min(1, ((200 * error) / resasc) ** 1.5)
  if (resabs > UFLOW / (50 * EPMACH)) error = Math.max(EPMACH * 50 * resabs, error)
  return { a, b, value, error }
}

/** The state of `gaussKronrod`. */
export type GaussKronrodState = Status & {
  /** The step index $t$. */
  t: Size
  /** The current subdivision, in order of $a$. */
  intervals: Interval[]
  /** The sum of the interval estimates. */
  value: number
  /** The sum of the interval error estimates. */
  error: number
  /** The interval bisected on the last step (null at $t = 0$). */
  split: { a: number; b: number } | null
  /** Cumulative count of integrand evaluations. */
  evaluations: number
  /** True when total error is at or below tolerance. */
  converged: boolean
  /** True once the estimate is not finite. */
  diverged: boolean
  /** True when the worst interval can no longer be bisected in floating point. */
  stalled: boolean
}

/**
 * Globally adaptive Gauss–Kronrod 7–15 quadrature (QUADPACK's QAG with key 1; Piessens et al., 1983): start with
 * $[a, b]$ (split at `points` and into `panels` equal pieces when given) and, each step, bisect the interval with the
 * largest error estimate, until the total error is at most $\max(\text{atol}, \text{rtol} \cdot |\text{value}|)$
 * (defaults 1.49e-8, as scipy's `quad`). `init` takes finite `{ a, b, points?, panels? }`; use `integrate` for
 * infinite limits.
 *
 * @param f The univariate real integrand function.
 * @param options Convergence options controlling absolute and relative tolerances.
 * @param options.atol Absolute error tolerance (default 1.49e-8).
 * @param options.rtol Relative error tolerance (default 1.49e-8).
 * @returns An `Algorithm` stepping through globally adaptive Gauss-Kronrod subdivision.
 *
 * @example Globally adaptive Gauss-Kronrod integration of a smooth curve
 * const alg = gaussKronrod(Math.sin)
 * const state = run(alg, { a: 0, b: Math.PI }, 20)
 * print('integral =', state.value)
 */
export function gaussKronrod(
  f: Integrand,
  { atol = 1.49e-8, rtol = 1.49e-8 }: { atol?: number; rtol?: number } = {},
): Algorithm<{ a: number; b: number; points?: readonly number[]; panels?: Size }, GaussKronrodState> {
  const summarise = (intervals: Interval[]) => {
    let value = 0
    let error = 0
    for (const i of intervals) {
      value += i.value
      error += i.error
    }
    const stalled = intervals.some((i) => {
      const m = 0.5 * (i.a + i.b)
      return i.error > 0 && (m <= Math.min(i.a, i.b) || m >= Math.max(i.a, i.b))
    })
    const converged = error <= Math.max(atol, rtol * Math.abs(value))
    return { value, error, converged, diverged: !Number.isFinite(value), stalled }
  }
  return {
    name: 'gauss-kronrod-15',
    init: ({ a, b, points = [], panels = 1 }) => {
      const cuts = startingCuts(a, b, points, panels)
      const intervals = cuts.slice(1).map((hi, k) => kronrod15(f, cuts[k], hi))
      return { t: 0, intervals, ...summarise(intervals), split: null, evaluations: 15 * intervals.length }
    },
    step: (s) => {
      let worst = 0
      for (let i = 1; i < s.intervals.length; i++) if (s.intervals[i].error > s.intervals[worst].error) worst = i
      const { a, b } = s.intervals[worst]
      const m = 0.5 * (a + b)
      const intervals = [
        ...s.intervals.slice(0, worst),
        kronrod15(f, a, m),
        kronrod15(f, m, b),
        ...s.intervals.slice(worst + 1),
      ]
      return { t: s.t + 1, intervals, ...summarise(intervals), split: { a, b }, evaluations: s.evaluations + 30 }
    },
    // Stop also when the worst interval can no longer be bisected in floating point.
    done: (s) => s.stalled,
  }
}

/**
 * The ends of the starting intervals of $[a, b]$: each of `panels` equal pieces, further split at the `points` that lie
 * strictly inside, sorted and without duplicates.
 *
 * @param a Start of integration interval.
 * @param b End of integration interval.
 * @param points Array of predefined internal split points.
 * @param panels Number of initial equal subintervals ($panels \ge 1$).
 * @returns Sorted array of interval cut points.
 */
function startingCuts(a: number, b: number, points: readonly number[], panels: Size): number[] {
  if (!(Number.isInteger(panels) && panels >= 1))
    throw new DomainError('integrate', `integrate: panels must be a positive integer, got ${panels}`)
  const lo = Math.min(a, b)
  const hi = Math.max(a, b)
  const inner = [
    ...Array.from({ length: panels - 1 }, (_, k) => lo + ((k + 1) * (hi - lo)) / panels),
    ...points.filter((p) => p > lo && p < hi),
  ]
  const sorted = [...new Set(inner)].sort((x, y) => x - y)
  return a <= b ? [lo, ...sorted, hi] : [hi, ...sorted.reverse(), lo]
}

/** The result of `integrate`. */
export type IntegrationResult = {
  /** Approximated integral value. */
  value: number
  /** The estimated absolute error. */
  error: number
  /** Total number of integrand evaluations performed. */
  evaluations: number
  /** Subintervals used (in the transformed variable for infinite limits; $2^k$ panels for Romberg). */
  intervals: Size
  /** Whether the error target was satisfied. */
  converged: boolean
}

/** Options of `integrate`. */
export type IntegrateOptions = {
  /** `'gauss-kronrod'` (default, globally adaptive 7–15) or `'romberg'` (for smooth f on a finite range). */
  method?: 'gauss-kronrod' | 'romberg'
  /** Absolute and relative error targets (defaults 1.49e-8 for Gauss–Kronrod, as scipy's `quad`; 1e-12 for Romberg). */
  atol?: number
  rtol?: number
  /** Gauss–Kronrod: at most this many subintervals (default 200). */
  maxIntervals?: Size
  /**
   * Gauss–Kronrod: points where f has a narrow feature, a kink or a jump, at which the range is split before the
   * first estimate (scipy's `points`). A feature narrower than the spacing of the first rule's 15 nodes can otherwise be
   * missed entirely: both estimates agree that f is 0 and the run stops at once.
   */
  points?: readonly number[]
  /** Gauss–Kronrod: equal panels to start from (default 1). */
  panels?: Size
  /** Romberg: at most this many halvings (default 20). */
  maxLevels?: Size
}

/**
 * $\int_a^b f(x)\,dx$ by globally adaptive Gauss–Kronrod 7–15 (QUADPACK's QAG, as scipy's `quad` without its extrapolation),
 * at most `maxIntervals` subintervals (default 200), or by Romberg integration (`method: 'romberg'`, at most
 * `maxLevels` halvings). Infinite limits are mapped to $(0, 1]$ as in QUADPACK's QAGI: $x = a + (1 - t)/t$ for $[a, \infty)$,
 * $x = b - (1 - t)/t$ for $(-\infty, b]$, and $f(x) + f(-x)$ on $[0, \infty)$ for $(-\infty, \infty)$. The Kronrod nodes
 * never touch $t = 0$. Pass `points` for a narrow peak far from the first rule's nodes (see `IntegrateOptions`).
 *
 * There is no Wynn $\varepsilon$ extrapolation (QAGS), so an integrable endpoint singularity converges slowly: $\int_0^1 x^{-0.9}\,dx$
 * stops at 200 intervals with `converged: false` where `quad` needs a few dozen. Remove the singularity by a
 * substitution first ($x = u^k$ with $k$ large enough that the integrand is bounded), or raise `maxIntervals`.
 *
 * @param f Univariate real integrand function.
 * @param a Lower integration limit (may be $-\infty$).
 * @param b Upper integration limit (may be $\infty$).
 * @param options Integration options controlling algorithm method, tolerances, and interval caps.
 * @returns An `IntegrationResult` holding the computed value, error estimate, evaluations, intervals, and convergence status.
 *
 * @example Integrate Gaussian density from -infinity to infinity
 * const res = integrate((x) => Math.exp(-x * x), -Infinity, Infinity)
 * print('integral =', res.value)
 */
export function integrate(f: Integrand, a: number, b: number, options: IntegrateOptions = {}): IntegrationResult {
  if (a === b) return { value: 0, error: 0, evaluations: 0, intervals: 0, converged: true }
  if (a > b) {
    const r = integrate(f, b, a, options)
    return { ...r, value: -r.value }
  }
  let g: Integrand = f
  let lo = a
  let hi = b
  if (a === -Infinity && b === Infinity) {
    g = (t) => {
      const x = (1 - t) / t
      return (f(x) + f(-x)) / (t * t)
    }
    ;[lo, hi] = [0, 1]
  } else if (b === Infinity) {
    g = (t) => f(a + (1 - t) / t) / (t * t)
    ;[lo, hi] = [0, 1]
  } else if (a === -Infinity) {
    g = (t) => f(b - (1 - t) / t) / (t * t)
    ;[lo, hi] = [0, 1]
  }
  if (options.method === 'romberg') {
    const r = run(romberg(g, options), { a: lo, b: hi }, options.maxLevels ?? 20)
    return { value: r.value, error: r.error, evaluations: r.evaluations, intervals: 2 ** r.t, converged: r.converged }
  }
  // Breakpoints move with the variable: t = 1/(1 + |x − end|) on a half-line, t = 1/(1 + |x|) for the folded real line.
  const toT = (x: number) =>
    a === -Infinity && b === Infinity
      ? 1 / (1 + Math.abs(x))
      : b === Infinity
        ? 1 / (1 + x - a)
        : a === -Infinity
          ? 1 / (1 + b - x)
          : x
  const points = (options.points ?? []).filter((x) => x > a && x < b).map(toT)
  const panels = options.panels ?? 1
  const start = { a: lo, b: hi, points, panels }
  const s = run(gaussKronrod(g, options), start, Math.max(0, (options.maxIntervals ?? 200) - points.length - panels))
  return {
    value: s.value,
    error: s.error,
    evaluations: s.evaluations,
    intervals: s.intervals.length,
    converged: s.converged,
  }
}
