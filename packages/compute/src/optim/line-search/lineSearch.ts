/**
 * Line searches along a direction p from a point x: backtracking to the Armijo (sufficient decrease) condition, and a
 * search for a step satisfying the strong Wolfe conditions. Both record every trial point they evaluate.
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

/** One evaluated step length: the point x + αp, f there, and the slope ∇f(x + αp)ᵀp. */
export type LineSearchTrial = { alpha: number; x: Vector; value: number; slope: number }

/** The outcome of a line search. */
export type LineSearchResult = {
  method: 'backtracking' | 'strong-wolfe'
  /** The accepted step length (0 when no trial lowered f). */
  alpha: number
  /** The accepted point x + αp. */
  x: Vector
  /** f at the accepted point. */
  value: number
  /** ∇f at the accepted point. */
  grad: Vector
  /** Every trial evaluated, in order; the accepted one is among them unless `alpha` is 0. */
  trials: LineSearchTrial[]
  evaluations: number
  /** The slope at α = 0, ∇f(x)ᵀp. The search fails at once unless it is negative (p must be a descent direction). */
  initialSlope: number
  /** Sufficient decrease at `alpha`: f(x + αp) ≤ f(x) + c₁α∇f(x)ᵀp. */
  armijo: boolean
  /** Strong curvature at `alpha`: |∇f(x + αp)ᵀp| ≤ c₂|∇f(x)ᵀp|; null for backtracking, which does not test it. */
  curvature: boolean | null
  /** True when the conditions the search enforces hold at `alpha`. */
  converged: boolean
}

/** Options for `backtracking`. */
export type BacktrackingOptions = {
  /** First step length tried. Default 1. */
  alpha0?: number
  /** Factor applied to α after each failed trial, in (0, 1). Default 0.5. */
  shrink?: number
  /** Sufficient-decrease constant c₁ in (0, 1). Default 1e-4. */
  c1?: number
  /** Most trials. Default 50. */
  maxTrials?: number
}

/** Options for `strongWolfe`. */
export type StrongWolfeOptions = {
  alpha0?: number
  /** Sufficient-decrease constant c₁. Default 1e-4. */
  c1?: number
  /** Curvature constant c₂ in (c₁, 1). Default 0.9 (quasi-Newton); use 0.1 for nonlinear conjugate gradients. */
  c2?: number
  /** Largest step length. Default 1e10. */
  alphaMax?: number
  /** Most trials (bracketing and zoom together). Default 30. */
  maxTrials?: number
}

/** A search's outcome on working arrays: the public result plus the accepted point, gradient and value. */
export type SearchOutcome = { result: LineSearchResult; x: F64; grad: F64; value: number }

type Point = { alpha: number; x: F64; value: number; grad: F64; slope: number }

/**
 * Packs the accepted point, or, when the search failed, the trial with the lowest value if it is below f(x), else
 * α = 0 (x unchanged). Failure is reported by `converged: false`, never hidden.
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

function probe(f: ObjectiveFn, x: F64, p: F64, alpha: number, where: string): Point {
  const xa = axpy(alpha, p, x)
  const { value, grad } = evaluate(f, xa, where)
  return { alpha, x: xa, value, grad, slope: dot(grad, p) }
}

/**
 * Backtracking on float64 working arrays (see `backtracking`): x, f(x), ∇f(x) and the direction p are given, and the
 * accepted point comes back as arrays. For optimisers' inner loops; the arrays are not mutated.
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
 * The minimiser of the cubic interpolating φ and φ′ at a and b (Nocedal & Wright eq. 3.59), or NaN when the cubic
 * has no minimiser there.
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
 * Backtracking line search: tries α = α₀, α₀ρ, α₀ρ², … and accepts the first α with sufficient decrease
 * f(x + αp) ≤ f(x) + c₁α∇f(x)ᵀp (Nocedal & Wright, Algorithm 3.1).
 *
 * `x` and `direction` are vectors of length n; `at` may supply f(x) and ∇f(x) (otherwise f is evaluated at x, which
 * `evaluations` does not count). Every trial is recorded. When p is not a descent direction or no trial passes within
 * `maxTrials`, `converged` is false and the best trial below f(x) (or α = 0) is returned.
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
 * f(x + αp) ≤ f(x) + c₁α∇f(x)ᵀp and curvature |∇f(x + αp)ᵀp| ≤ c₂|∇f(x)ᵀp|, by bracketing then zooming with
 * safeguarded cubic interpolation (Nocedal & Wright, Algorithms 3.5 and 3.6).
 *
 * Arguments as for `backtracking`. Every trial is recorded; failure is reported by `converged: false`.
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
