/**
 * Minimisation of a function of one variable without derivatives: Brent's method (parabolic interpolation safeguarded
 * by golden-section steps) and plain golden-section search, on a bracket found by downhill expansion or on fixed
 * bounds. Brent (1973), "Algorithms for Minimization without Derivatives", ch. 5; Press et al. (2007), "Numerical
 * Recipes", 3rd ed., §10.1–10.3; the options follow `scipy.optimize.minimize_scalar`.
 */

import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `minimizeScalar`. */
export type MinimizeScalarOptions = {
  /**
   * `brent` (default): parabolic steps with golden-section fallback, superlinear near a smooth minimum. `golden`:
   * golden-section search only, linear convergence by the factor 0.618 per step, robust to non-smooth functions.
   */
  method?: 'brent' | 'golden'
  /**
   * Two starting points (a, b): a bracket a < b < c with f(b) below f(a) and f(c) is found by expanding downhill from
   * them, so the minimum found may lie outside [a, b]. Default (0, 1). Ignored when `bounds` is given.
   */
  bracket?: readonly [number, number]
  /** Search only within [lo, hi] (no expansion); the result may be an endpoint when f is monotone there. */
  bounds?: readonly [number, number]
  /** Relative tolerance on x: stop when the interval is within about tolerance·|x| + 1e-11 of x. Default 1.48e-8. */
  tolerance?: number
  /** Largest number of Brent or golden-section steps (after bracketing). Default 500. */
  maxSteps?: number
}

/** The result of `minimizeScalar`. */
export type MinimizeScalarResult = {
  /** The point found and f there. */
  x: number
  value: number
  /** Brent or golden-section steps taken, and calls of f in total (bracketing included). */
  steps: number
  evaluations: number
  /** True when the tolerance was met within `maxSteps` (and, without `bounds`, a bracket was found). */
  converged: boolean
  /** False when downhill expansion found no bracket in 100 expansions (f may decrease without bound). */
  bracketed: boolean
}

const GOLD = 1.618034
const CGOLD = 0.381966

type Bracket = { a: number; b: number; c: number; fb: number; found: boolean }

/**
 * Expand downhill from (a, b) until f(b) < f(c): golden-ratio steps with parabolic extrapolation limited to 100 times
 * the step (Press et al., 2007, `mnbrak`; as scipy's `bracket`).
 */
function bracketMinimum(f: (x: number) => number, a0: number, b0: number): Bracket {
  let [a, b] = [a0, b0]
  let [fa, fb] = [f(a), f(b)]
  if (fb > fa) [a, b, fa, fb] = [b, a, fb, fa]
  let c = b + GOLD * (b - a)
  let fc = f(c)
  for (let grow = 0; fb > fc; grow++) {
    if (grow >= 100) return { a, b, c, fb, found: false }
    const r = (b - a) * (fb - fc)
    const q = (b - c) * (fb - fa)
    const denom = 2 * Math.sign(q - r) * Math.max(Math.abs(q - r), 1e-21)
    let u = b - ((b - c) * q - (b - a) * r) / denom
    const ulim = b + 100 * (c - b)
    let fu: number
    if ((b - u) * (u - c) > 0) {
      fu = f(u)
      if (fu < fc) return { a: b, b: u, c, fb: fu, found: true }
      if (fu > fb) return { a, b, c: u, fb, found: true }
      u = c + GOLD * (c - b)
      fu = f(u)
    } else if ((c - u) * (u - ulim) > 0) {
      fu = f(u)
      if (fu < fc) {
        ;[b, c, u] = [c, u, u + GOLD * (u - c)]
        ;[fb, fc, fu] = [fc, fu, f(u)]
      }
    } else if ((u - ulim) * (ulim - c) >= 0) {
      u = ulim
      fu = f(u)
    } else {
      u = c + GOLD * (c - b)
      fu = f(u)
    }
    ;[a, b, c, fa, fb, fc] = [b, c, u, fb, fc, fu]
  }
  return { a, b, c, fb, found: true }
}

type Search = { x: number; value: number; steps: number; converged: boolean }

/** Brent's method on [lo, hi] from the point x (with f(x) = fx) inside it. */
function brent(
  f: (x: number) => number,
  lo: number,
  hi: number,
  x0: number,
  fx0: number,
  tol: number,
  maxSteps: number,
): Search {
  let [x, w, v] = [x0, x0, x0]
  let [fx, fw, fv] = [fx0, fx0, fx0]
  let d = 0
  let e = 0
  for (let it = 0; it < maxSteps; it++) {
    const xm = 0.5 * (lo + hi)
    const tol1 = tol * Math.abs(x) + 1e-11
    const tol2 = 2 * tol1
    if (Math.abs(x - xm) <= tol2 - 0.5 * (hi - lo)) return { x, value: fx, steps: it, converged: true }
    if (Math.abs(e) > tol1) {
      // Try a parabola through (v, w, x); accept it only if it falls inside the interval and moves less than half the
      // step before last, else take a golden-section step into the larger part.
      let r = (x - w) * (fx - fv)
      let q = (x - v) * (fx - fw)
      let p = (x - v) * q - (x - w) * r
      q = 2 * (q - r)
      if (q > 0) p = -p
      q = Math.abs(q)
      const eTemp = e
      e = d
      if (Math.abs(p) >= Math.abs(0.5 * q * eTemp) || p <= q * (lo - x) || p >= q * (hi - x)) {
        e = x >= xm ? lo - x : hi - x
        d = CGOLD * e
      } else {
        d = p / q
        const u = x + d
        if (u - lo < tol2 || hi - u < tol2) d = Math.sign(xm - x) * tol1 || tol1
        r = 0
      }
    } else {
      e = x >= xm ? lo - x : hi - x
      d = CGOLD * e
    }
    const u = Math.abs(d) >= tol1 ? x + d : x + (Math.sign(d) || 1) * tol1
    const fu = f(u)
    if (fu <= fx) {
      if (u >= x) lo = x
      else hi = x
      ;[v, w, x] = [w, x, u]
      ;[fv, fw, fx] = [fw, fx, fu]
    } else {
      if (u < x) lo = u
      else hi = u
      if (fu <= fw || w === x) {
        ;[v, w, fv, fw] = [w, u, fw, fu]
      } else if (fu <= fv || v === x || v === w) {
        ;[v, fv] = [u, fu]
      }
    }
  }
  return { x, value: fx, steps: maxSteps, converged: false }
}

/** Golden-section search on [lo, hi]: each step keeps the part holding the lower of two interior points. */
function golden(f: (x: number) => number, lo: number, hi: number, tol: number, maxSteps: number): Search {
  const g = (Math.sqrt(5) - 1) / 2
  let c = hi - g * (hi - lo)
  let d = lo + g * (hi - lo)
  let fc = f(c)
  let fd = f(d)
  for (let it = 0; it < maxSteps; it++) {
    const x = fc < fd ? c : d
    if (hi - lo <= 2 * (tol * Math.abs(x) + 1e-11)) return { x, value: Math.min(fc, fd), steps: it, converged: true }
    if (fc < fd) {
      ;[hi, d, fd] = [d, c, fc]
      c = hi - g * (hi - lo)
      fc = f(c)
    } else {
      ;[lo, c, fc] = [c, d, fd]
      d = lo + g * (hi - lo)
      fd = f(d)
    }
  }
  return fc < fd
    ? { x: c, value: fc, steps: maxSteps, converged: false }
    : { x: d, value: fd, steps: maxSteps, converged: false }
}

/**
 * A local minimum of f: ℝ → ℝ without derivatives. With `bounds`, the search stays in [lo, hi]; otherwise a bracket is
 * first found by expanding downhill from `bracket` (default (0, 1)), as `scipy.optimize.minimize_scalar`. NaN values of
 * f compare as larger than any number, so they steer the search away.
 *
 * @example minimizeScalar((x) => (x - 2) ** 2).x // 2
 */
export function minimizeScalar(f: (x: number) => number, options: MinimizeScalarOptions = {}): MinimizeScalarResult {
  const method = options.method ?? 'brent'
  const tol = options.tolerance ?? 1.48e-8
  const maxSteps = options.maxSteps ?? 500
  let evaluations = 0
  const g = (x: number) => {
    evaluations++
    const y = f(x)
    return Number.isNaN(y) ? Infinity : y
  }
  let lo: number
  let hi: number
  let start: { x: number; fx: number } | undefined
  let bracketed = true
  if (options.bounds) {
    ;[lo, hi] = options.bounds
    if (!(lo <= hi))
      throw new DomainError('minimizeScalar', `minimizeScalar: bounds must satisfy lo ≤ hi, got [${lo}, ${hi}]`)
  } else {
    const [a, b] = options.bracket ?? [0, 1]
    if (a === b) throw new DomainError('minimizeScalar', 'minimizeScalar: the two bracket points must differ')
    const br = bracketMinimum(g, a, b)
    bracketed = br.found
    lo = Math.min(br.a, br.c)
    hi = Math.max(br.a, br.c)
    start = { x: br.b, fx: br.fb }
  }
  let search: Search
  if (method === 'golden') search = golden(g, lo, hi, tol, maxSteps)
  else {
    const x0 = start?.x ?? lo + CGOLD * (hi - lo)
    search = brent(g, lo, hi, x0, start?.fx ?? g(x0), tol, maxSteps)
  }
  return {
    x: search.x,
    value: search.value,
    steps: search.steps,
    evaluations,
    converged: search.converged && bracketed,
    bracketed,
  }
}
