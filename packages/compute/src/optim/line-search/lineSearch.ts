/**
 * Line searches along a direction $\pvec$ from a point $\xvec$: backtracking to the Armijo (sufficient decrease)
 * condition, and a search for a step satisfying the strong Wolfe conditions. Both record every trial point they
 * evaluate.
 *
 * Along the line the objective is $\phi(\alpha) = f(\xvec + \alpha\pvec)$, with slope
 * $\phi'(\alpha) = \nabla f(\xvec + \alpha\pvec)^\top\pvec$. Sufficient decrease is
 * $\phi(\alpha) \le \phi(0) + c_1\alpha\phi'(0)$ and strong curvature is
 * $\lvert\phi'(\alpha)\rvert \le c_2\lvert\phi'(0)\rvert$, with $0 < c_1 < c_2 < 1$. A search needs a descent
 * direction, $\phi'(0) < 0$; otherwise it fails at once. Failure is reported (`converged: false`) and never thrown, and
 * the point returned is then the best trial below $f(\xvec)$, or $\xvec$ itself. The vector functions (`backtracking`,
 * `strongWolfe`) wrap the working-array ones (`backtrackingSearch`, `strongWolfeSearch`) that the optimisers call in
 * their inner loops.
 *
 * Nocedal & Wright (2006), "Numerical Optimization", 2nd ed.: Algorithm 3.1 (backtracking), Algorithms 3.5 and 3.6
 * (strong Wolfe search and zoom), equation 3.59 (cubic interpolation).
 */

import type { Vector } from 'aifn-compute/foundation/tensor'
import type { ObjectiveFn, VectorLike } from 'aifn-compute/foundation/contracts'
import { evaluate } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

const { axpy, dot, toF64, vec } = dense
type F64 = dense.F64

/**
 * One evaluated step length `alpha` ($\alpha$): the point `x` ($\xvec + \alpha\pvec$), the objective `value` there,
 * and the `slope` $\nabla f(\xvec + \alpha\pvec)^\top\pvec$ along the direction.
 */
export type LineSearchTrial = { alpha: number; x: Vector; value: number; slope: number }

/** The outcome of a line search. */
export type LineSearchResult = {
  /** Which search produced the result. */
  method: 'backtracking' | 'strong-wolfe'
  /**
   * The step length $\alpha$ returned: the accepted one, or after a failure the trial with the lowest value below
   * $f(\xvec)$, or 0 when no trial lowered $f$.
   */
  alpha: number
  /** The point returned, $\xvec + \alpha\pvec$. */
  x: Vector
  /** $f$ at the point returned. */
  value: number
  /** $\nabla f$ at the point returned. */
  grad: Vector
  /** Every trial evaluated, in order; the point returned is among them unless `alpha` is 0. */
  trials: LineSearchTrial[]
  /** Evaluations of $f$ made by the search: one per trial (an evaluation at $\xvec$ itself is not counted). */
  evaluations: number
  /**
   * The slope at $\alpha = 0$, $\nabla f(\xvec)^\top\pvec$. The search fails at once unless it is negative
   * ($\pvec$ must be a descent direction).
   */
  initialSlope: number
  /**
   * Sufficient decrease at `alpha`: $f(\xvec + \alpha\pvec) \le f(\xvec) + c_1\alpha\nabla f(\xvec)^\top\pvec$
   * (false when `alpha` is 0).
   */
  armijo: boolean
  /**
   * Strong curvature at `alpha`:
   * $\lvert\nabla f(\xvec + \alpha\pvec)^\top\pvec\rvert \le c_2\lvert\nabla f(\xvec)^\top\pvec\rvert$ (false
   * when `alpha` is 0); null for backtracking, which does not test it.
   */
  curvature: boolean | null
  /** True when the conditions the search enforces hold at `alpha`. */
  converged: boolean
}

/** Options for `backtracking`. */
export type BacktrackingOptions = {
  /** First step length tried. Default 1. */
  alpha0?: number
  /** Factor $\rho$ applied to $\alpha$ after each failed trial, in $(0, 1)$. Default 0.5. */
  shrink?: number
  /** Sufficient-decrease constant $c_1$ in $(0, 1)$. Default 1e-4. */
  c1?: number
  /** Most trials. Default 50. */
  maxTrials?: number
}

/** Options for `strongWolfe`. */
export type StrongWolfeOptions = {
  /** First step length tried (capped at `alphaMax`). Default 1. */
  alpha0?: number
  /** Sufficient-decrease constant $c_1$. Default 1e-4. */
  c1?: number
  /** Curvature constant $c_2$ in $(c_1, 1)$. Default 0.9 (quasi-Newton); use 0.1 for nonlinear conjugate gradients. */
  c2?: number
  /** Largest step length. The bracketing phase doubles $\alpha$ up to it. Default 1e10. */
  alphaMax?: number
  /** Most trials (bracketing and zoom together). Default 30. */
  maxTrials?: number
}

/**
 * A search's outcome on working arrays: the public `result`, plus the point returned (`x`), its gradient (`grad`) and
 * its `value` as float64 arrays and a number, for the optimiser to carry on from.
 */
export type SearchOutcome = { result: LineSearchResult; x: F64; grad: F64; value: number }

/** A trial in working arrays: the step length, the point, its value and gradient, and the slope along $\pvec$. */
type Point = { alpha: number; x: F64; value: number; grad: F64; slope: number }

/**
 * Packs the accepted point, or, when the search failed, the trial with the lowest value if it is below $f(\xvec)$,
 * else $\alpha = 0$ ($\xvec$ unchanged). Failure is reported by `converged: false`, never hidden.
 *
 * @param method Which search ran, recorded in the result.
 * @param x0 The start point $\xvec$, returned when no trial is chosen.
 * @param value0 $f(\xvec)$, the value a fallback trial must beat.
 * @param grad0 $\nabla f(\xvec)$, returned with `x0`.
 * @param slope0 The initial slope $\nabla f(\xvec)^\top\pvec$, used to test the conditions at the point returned.
 * @param c1 The sufficient-decrease constant $c_1$.
 * @param c2 The curvature constant $c_2$, or null when the curvature condition is not tested (backtracking).
 * @param points Every trial evaluated, in order.
 * @param accepted The trial that met the search's conditions, or null when none did.
 * @returns The outcome, with `converged` true exactly when `accepted` was given.
 */
function finish(
  method: LineSearchResult['method'],
  x0: F64,
  value0: number,
  grad0: F64,
  slope0: number,
  c1: number,
  c2: number | null,
  points: Point[],
  accepted: Point | null,
): SearchOutcome {
  let chosen = accepted
  if (!chosen) {
    for (const p of points) if (p.value < value0 && (!chosen || p.value < chosen.value)) chosen = p
  }
  const trials = points.map((p) => ({ alpha: p.alpha, x: vec(p.x), value: p.value, slope: p.slope }))
  const at = chosen ?? { alpha: 0, x: x0, value: value0, grad: grad0, slope: slope0 }
  const armijo = at.alpha > 0 && at.value <= value0 + c1 * at.alpha * slope0
  const curvature = c2 === null ? null : at.alpha > 0 && Math.abs(at.slope) <= c2 * Math.abs(slope0)
  return {
    result: {
      method,
      alpha: at.alpha,
      x: vec(at.x),
      value: at.value,
      grad: vec(at.grad),
      trials,
      evaluations: points.length,
      initialSlope: slope0,
      armijo,
      curvature,
      converged: accepted !== null,
    },
    x: at.x,
    grad: at.grad,
    value: at.value,
  }
}

/**
 * Evaluates the objective at $\xvec + \alpha\pvec$.
 *
 * @param f The objective, returning value and gradient.
 * @param x The start point $\xvec$ (not modified).
 * @param p The search direction $\pvec$.
 * @param alpha The step length $\alpha$.
 * @param where The caller's name, for error messages.
 * @returns The trial: its point, value, gradient and slope $\nabla f^\top\pvec$.
 */
function probe(f: ObjectiveFn, x: F64, p: F64, alpha: number, where: string): Point {
  const xa = axpy(alpha, p, x)
  const { value, grad } = evaluate(f, xa, where)
  return { alpha, x: xa, value, grad, slope: dot(grad, p) }
}

/**
 * Backtracking on float64 working arrays (see `backtracking`): $\xvec$, $f(\xvec)$, $\nabla f(\xvec)$ and the
 * direction $\pvec$ are given, and the point returned comes back as arrays. For optimisers' inner loops; the arrays
 * are not mutated. A non-finite trial value counts as a failed trial, so the step shrinks past it.
 *
 * @param f The objective, returning value and gradient.
 * @param x The start point $\xvec$, $n$ values.
 * @param value $f(\xvec)$, already known to the caller.
 * @param grad $\nabla f(\xvec)$, $n$ values.
 * @param p The search direction $\pvec$, $n$ values; it must be a descent direction ($\nabla f^\top\pvec < 0$).
 * @param options The first step length, the shrink factor $\rho$, $c_1$ and the trial budget.
 * @returns The public result, with the point returned, its gradient and its value as working arrays.
 *
 * @example Backtrack on working arrays, as an optimiser's inner loop does
 * // f(x) = x₁² + 10x₂² from (1, 1), whose value is 11 and gradient (2, 20).
 * const f = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * const x = new Float64Array([1, 1])
 * const out = backtrackingSearch(f, x, 11, new Float64Array([2, 20]), new Float64Array([-2, -20]))
 * print('alpha =', out.result.alpha)
 * print('new x =', out.x, ' f =', out.value)
 * print('start x, unchanged =', x)
 */
export function backtrackingSearch(
  f: ObjectiveFn,
  x: F64,
  value: number,
  grad: F64,
  p: F64,
  options: BacktrackingOptions = {},
): SearchOutcome {
  const { alpha0 = 1, shrink = 0.5, c1 = 1e-4, maxTrials = 50 } = options
  const slope0 = dot(grad, p)
  const points: Point[] = []
  if (!(slope0 < 0)) return finish('backtracking', x, value, grad, slope0, c1, null, points, null)
  let alpha = alpha0
  for (let k = 0; k < maxTrials; k++) {
    const point = probe(f, x, p, alpha, 'backtracking')
    points.push(point)
    // A non-finite value fails the comparison, so the step shrinks.
    if (point.value <= value + c1 * alpha * slope0)
      return finish('backtracking', x, value, grad, slope0, c1, null, points, point)
    alpha *= shrink
  }
  return finish('backtracking', x, value, grad, slope0, c1, null, points, null)
}

/**
 * The minimiser of the cubic interpolating $\phi$ and $\phi'$ at $a$ and $b$ (Nocedal & Wright eq. 3.59), or NaN when
 * the cubic has no minimiser there.
 *
 * @param a The first step length.
 * @param fa $\phi(a)$.
 * @param da $\phi'(a)$.
 * @param b The second step length.
 * @param fb $\phi(b)$.
 * @param db $\phi'(b)$.
 * @returns The step length minimising the interpolating cubic (not clamped to the interval), or NaN.
 */
function cubicMinimiser(a: number, fa: number, da: number, b: number, fb: number, db: number): number {
  const d1 = da + db - 3 * ((fa - fb) / (a - b))
  const disc = d1 * d1 - da * db
  if (!(disc >= 0)) return NaN
  const d2 = Math.sign(b - a) * Math.sqrt(disc)
  return b - (b - a) * ((db + d2 - d1) / (db - da + 2 * d2))
}

/**
 * The strong Wolfe search on float64 working arrays (see `strongWolfe` and `backtrackingSearch`). For optimisers'
 * inner loops; the arrays are not mutated.
 *
 * @param f The objective, returning value and gradient.
 * @param x The start point $\xvec$, $n$ values.
 * @param value $f(\xvec)$, already known to the caller.
 * @param grad $\nabla f(\xvec)$, $n$ values.
 * @param p The search direction $\pvec$, $n$ values; it must be a descent direction ($\nabla f^\top\pvec < 0$).
 * @param options The first and largest step lengths, $c_1$, $c_2$ and the trial budget.
 * @returns The public result, with the point returned, its gradient and its value as working arrays.
 *
 * @example One strong Wolfe step on working arrays
 * // f(x) = x₁² + 10x₂² from (1, 1) along −∇f: the exact minimiser along the line is 404 / 8008.
 * const f = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * const g = new Float64Array([2, 20])
 * const out = strongWolfeSearch(f, new Float64Array([1, 1]), 11, g, new Float64Array([-2, -20]), { c2: 0.1 })
 * print('alpha =', out.result.alpha, ' exact:', 404 / 8008)
 * print('x =', out.x, ' grad =', out.grad)
 */
export function strongWolfeSearch(
  f: ObjectiveFn,
  x: F64,
  value: number,
  grad: F64,
  p: F64,
  options: StrongWolfeOptions = {},
): SearchOutcome {
  const { alpha0 = 1, c1 = 1e-4, c2 = 0.9, alphaMax = 1e10, maxTrials = 30 } = options
  const slope0 = dot(grad, p)
  const points: Point[] = []
  const done = (accepted: Point | null) => finish('strong-wolfe', x, value, grad, slope0, c1, c2, points, accepted)
  if (!(slope0 < 0)) return done(null)

  const sufficient = (q: Point) => q.value <= value + c1 * q.alpha * slope0
  const flat = (q: Point) => Math.abs(q.slope) <= -c2 * slope0
  const origin: Point = { alpha: 0, x, value, grad, slope: slope0 }

  // Zoom (Algorithm 3.6): lo satisfies sufficient decrease with the lowest value so far; the interval between lo and
  // hi contains a step satisfying the strong Wolfe conditions.
  const zoom = (lo: Point, hi: Point): Point | null => {
    while (points.length < maxTrials) {
      const [left, right] = lo.alpha < hi.alpha ? [lo.alpha, hi.alpha] : [hi.alpha, lo.alpha]
      let alpha = cubicMinimiser(lo.alpha, lo.value, lo.slope, hi.alpha, hi.value, hi.slope)
      // Safeguard: stay well inside the interval, else bisect.
      const margin = 0.1 * (right - left)
      if (!(alpha > left + margin && alpha < right - margin)) alpha = 0.5 * (lo.alpha + hi.alpha)
      if (right - left < 1e-16 * Math.max(1, right)) return null
      const q = probe(f, x, p, alpha, 'strongWolfe')
      points.push(q)
      if (!sufficient(q) || q.value >= lo.value) hi = q
      else {
        if (flat(q)) return q
        if (q.slope * (hi.alpha - lo.alpha) >= 0) hi = lo
        lo = q
      }
    }
    return null
  }

  // Bracketing (Algorithm 3.5).
  let previous = origin
  let alpha = Math.min(alpha0, alphaMax)
  for (let i = 1; points.length < maxTrials; i++) {
    const q = probe(f, x, p, alpha, 'strongWolfe')
    points.push(q)
    if (!sufficient(q) || (i > 1 && q.value >= previous.value)) return done(zoom(previous, q))
    if (flat(q)) return done(q)
    if (q.slope >= 0) return done(zoom(q, previous))
    if (alpha >= alphaMax) break
    previous = q
    alpha = Math.min(2 * alpha, alphaMax)
  }
  return done(null)
}

/**
 * Converts the arguments of `backtracking` and `strongWolfe` to working arrays, evaluating the objective at $\xvec$
 * for whichever of the value and gradient the caller did not supply.
 *
 * @param f The objective, returning value and gradient.
 * @param x The start point $\xvec$.
 * @param direction The search direction $\pvec$, of the same length as `x` (a `ShapeError` otherwise).
 * @param at The caller's $f(\xvec)$ (`value`) and $\nabla f(\xvec)$ (`grad`), either of which may be left out.
 * @param where The caller's name, for error messages.
 * @returns The start point, direction, value and gradient as working arrays and a number.
 */
function start(
  f: ObjectiveFn,
  x: VectorLike,
  direction: VectorLike,
  at: { value?: number; grad?: VectorLike },
  where: string,
) {
  const x0 = toF64(x, where)
  const p = toF64(direction, where)
  if (p.length !== x0.length) throw new ShapeError(where, `${where}: direction and x differ in length`)
  let value = at.value
  let grad = at.grad === undefined ? undefined : toF64(at.grad, where)
  if (value === undefined || grad === undefined) {
    const e = evaluate(f, x0, where)
    value ??= e.value
    grad ??= e.grad
  }
  return { x0, p, value, grad }
}

/**
 * Backtracking line search: tries $\alpha = \alpha_0, \alpha_0\rho, \alpha_0\rho^2, \dots$ and accepts the first
 * $\alpha$ with sufficient decrease $f(\xvec + \alpha\pvec) \le f(\xvec) + c_1\alpha\nabla f(\xvec)^\top\pvec$
 * (Nocedal & Wright, Algorithm 3.1).
 *
 * Every trial is recorded. When $\pvec$ is not a descent direction or no trial passes within `maxTrials`, `converged`
 * is false and the best trial below $f(\xvec)$ (or $\alpha = 0$) is returned. Direction and point of different
 * lengths throw `ShapeError`.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param x The start point $\xvec$, a vector of length $n$.
 * @param direction The search direction $\pvec$, of length $n$.
 * @param options The search's options, and optionally `value` and `grad`, the caller's $f(\xvec)$ and
 *   $\nabla f(\xvec)$. Whichever is left out is computed by evaluating $f$ at $\xvec$, which `evaluations` does not
 *   count.
 * @returns The step length and point returned, with every trial and which conditions hold there.
 *
 * @example Halve the step until the decrease is sufficient
 * // f(x) = x₁² + 10x₂² from (1, 1), along the steepest-descent direction −∇f = (−2, −20).
 * const f = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * const r = backtracking(f, [1, 1], [-2, -20])
 * print('step lengths tried =', r.trials.map((t) => t.alpha))
 * print('alpha =', r.alpha)
 * print('x =', r.x, ' f(x) =', r.value)
 * print('converged =', r.converged)
 *
 * @example An uphill direction is reported, not searched
 * const f = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * const r = backtracking(f, [1, 1], [2, 20])
 * print('initial slope =', r.initialSlope)
 * print('alpha =', r.alpha, ' converged =', r.converged)
 */
export function backtracking(
  f: ObjectiveFn,
  x: VectorLike,
  direction: VectorLike,
  options: BacktrackingOptions & { value?: number; grad?: VectorLike } = {},
): LineSearchResult {
  const { x0, p, value, grad } = start(f, x, direction, options, 'backtracking')
  return backtrackingSearch(f, x0, value, grad, p, options).result
}

/**
 * Line search for a step satisfying the strong Wolfe conditions, sufficient decrease
 * $f(\xvec + \alpha\pvec) \le f(\xvec) + c_1\alpha\nabla f(\xvec)^\top\pvec$ and curvature
 * $\lvert\nabla f(\xvec + \alpha\pvec)^\top\pvec\rvert \le c_2\lvert\nabla f(\xvec)^\top\pvec\rvert$, by
 * bracketing (doubling $\alpha$ up to `alphaMax`) then zooming with safeguarded cubic interpolation (Nocedal & Wright,
 * Algorithms 3.5 and 3.6).
 *
 * Every trial is recorded; failure is reported by `converged: false`, as for `backtracking`.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param x The start point $\xvec$, a vector of length $n$.
 * @param direction The search direction $\pvec$, of length $n$.
 * @param options The search's options, and optionally `value` and `grad`, the caller's $f(\xvec)$ and
 *   $\nabla f(\xvec)$. Whichever is left out is computed by evaluating $f$ at $\xvec$, which `evaluations` does not
 *   count.
 * @returns The step length and point returned, with every trial and which conditions hold there.
 *
 * @example A step close to the exact minimiser along the line
 * // f(x) = x₁² + 10x₂² from (1, 1) along −∇f: along the line, f is least at 404 / 8008.
 * const f = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + 10 * b * b, grad: [2 * a, 20 * b] }
 * }
 * const r = strongWolfe(f, [1, 1], [-2, -20], { c2: 0.1 })
 * print('alpha =', r.alpha, ' exact:', 404 / 8008)
 * print('x =', r.x, ' f(x) =', r.value)
 * print('armijo =', r.armijo, ' curvature =', r.curvature)
 * print('evaluations =', r.evaluations)
 */
export function strongWolfe(
  f: ObjectiveFn,
  x: VectorLike,
  direction: VectorLike,
  options: StrongWolfeOptions & { value?: number; grad?: VectorLike } = {},
): LineSearchResult {
  const { x0, p, value, grad } = start(f, x, direction, options, 'strongWolfe')
  return strongWolfeSearch(f, x0, value, grad, p, options).result
}
