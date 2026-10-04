/**
 * Minimisation of a function of one variable without derivatives: Brent's method (parabolic interpolation safeguarded
 * by golden-section steps) and plain golden-section search, on a bracket found by downhill expansion or on fixed
 * bounds. Brent (1973), "Algorithms for Minimization without Derivatives", ch. 5; Press et al. (2007), "Numerical
 * Recipes", 3rd ed., §10.1–10.3; the options follow `scipy.optimize.minimize_scalar`.
 */

import { DomainError } from 'aifn-compute/foundation/errors'

/** Options configuring scalar function minimisation in `minimizeScalar`. */
export type MinimizeScalarOptions = {
  /**
   * Minimisation algorithm: `'brent'` (default) using parabolic interpolation with golden-section safeguards,
   * or `'golden'` for pure golden-section search.
   */
  method?: 'brent' | 'golden'
  /**
   * Two initial search points $[a, b]$ for downhill bracket expansion (default $[0, 1]$).
   * Ignored when `bounds` is provided.
   */
  bracket?: readonly [number, number]
  /** Constrained interval $[lo, hi]$ to search within without bracket expansion. */
  bounds?: readonly [number, number]
  /** Relative stopping tolerance on $x$: stops when interval width is within $\text{tolerance} \cdot |x| + 10^{-11}$. */
  tolerance?: number
  /** Maximum number of minimisation iterations allowed (default 500). */
  maxSteps?: number
}

/** Result returned by `minimizeScalar`. */
export type MinimizeScalarResult = {
  /** Location of the estimated local minimum $x$. */
  x: number
  /** Function value $f(x)$ at the estimated minimiser. */
  value: number
  /** Number of minimisation steps taken after initial bracketing. */
  steps: number
  /** Total number of objective function calls performed (including bracketing). */
  evaluations: number
  /** True when convergence tolerance was achieved within the step budget. */
  converged: boolean
  /** False if downhill bracket expansion failed to find a local minimum bracket. */
  bracketed: boolean
}

const GOLD = 1.618034
const CGOLD = 0.381966

type Bracket = { a: number; b: number; c: number; fb: number; found: boolean }

/**
 * Expand downhill from initial points $(a_0, b_0)$ until a local minimum is bracketed.
 *
 * Uses golden-ratio extrapolation and parabolic steps limited to 100 times the step length
 * (Press et al., 2007, `mnbrak`).
 *
 * @param f Univariate scalar objective function.
 * @param a0 First initial point.
 * @param b0 Second initial point defining search direction.
 * @returns Bracket structure containing endpoints $a, b, c$ where $f(b) < \min(f(a), f(c))$.
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

/**
 * Brent's 1D minimisation method combining parabolic interpolation with golden-section steps.
 *
 * @param f Univariate scalar objective function.
 * @param lo Lower bracket interval endpoint.
 * @param hi Upper bracket interval endpoint.
 * @param x0 Initial interior evaluation point.
 * @param fx0 Precomputed objective value $f(x_0)$.
 * @param tol Relative convergence tolerance.
 * @param maxSteps Maximum iteration budget.
 * @returns Search outcome with minimiser location, objective value, and convergence flag.
 */
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

/**
 * Golden-section search for finding a local minimum on $[lo, hi]$.
 *
 * @param f Univariate scalar objective function.
 * @param lo Lower bracket interval endpoint.
 * @param hi Upper bracket interval endpoint.
 * @param tol Relative convergence tolerance.
 * @param maxSteps Maximum iteration budget.
 * @returns Search outcome with minimiser location, objective value, and convergence flag.
 */
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
 * Find a local minimum of a scalar function $f: \mathbb{R} \to \mathbb{R}$ without derivatives.
 *
 * When `bounds` is supplied, searches strictly within $[lo, hi]$. Otherwise, an initial bracket
 * is discovered by expanding downhill from `bracket` (default $[0, 1]$), matching SciPy's
 * `scipy.optimize.minimize_scalar`. Objective evaluations returning NaN are treated as $+\infty$
 * to steer the search towards feasible regions.
 *
 * @param f Univariate scalar objective function.
 * @param options Configuration for method, bracketing, bounds, and tolerances.
 * @returns Minimisation result with estimated minimiser, function value, and diagnostics.
 *
 * @example Minimize quadratic function
 * const res = minimizeScalar(x => (x - 2) ** 2)
 * print('minimum at =', res.x)
 * print('value =', res.value)
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
