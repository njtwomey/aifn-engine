/**
 * Second-order methods with a supplied Hessian: damped Newton (a Newton direction, made a descent direction by adding
 * a multiple of the identity when the Hessian is not positive definite, and a line search), and a trust-region method
 * with the dogleg step.
 *
 * Both minimise the local quadratic model
 * $m(\pvec) = f(\xvec) + \nabla f(\xvec)^\top\pvec + \tfrac12\pvec^\top\nabla^2 f(\xvec)\pvec$: Newton's method
 * steps to its minimiser and then searches along the step, the trust region trusts it only within a radius $\Delta$
 * that it adapts. Each takes the objective (value and gradient) and a `hessian` function
 * returning $\nabla^2 f(\xvec)$ as an $n \times n$ matrix, evaluated once at each new iterate. Both converge
 * quadratically near a minimiser with a positive definite Hessian (Nocedal & Wright, 2006, "Numerical Optimization",
 * 2nd ed., chapters 3 and 4). Failure is reported in the state (`diverged`, `stalled`), not thrown.
 */

import { cholesky, choleskySolve } from 'aifn-compute/numerics/linalg'
import type { Matrix, Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { StartOptions } from '../options'
import {
  backtrackingSearch,
  strongWolfeSearch,
  type BacktrackingOptions,
  type LineSearchResult,
} from 'aifn-compute/optim/line-search'
import type { Hessian, IterateState, ObjectiveFn, StoppingOptions } from 'aifn-compute/foundation/contracts'
import { DEFAULT_DIVERGE, DEFAULT_TOLERANCE, divergedAt, evaluate } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'

const { allFinite, axpy, data, dot, mat, matVec, norm, scale, sub, toF64, toMatrixF64, vec } = dense
type F64 = dense.F64

/**
 * Evaluates the Hessian as a working array, checked for shape.
 *
 * @param h The Hessian function.
 * @param x The point, $n$ values.
 * @param where The caller's name, for error messages.
 * @returns $\nabla^2 f(\xvec)$ as a row-major array of $n^2$ values.
 */
function hessianAt(h: Hessian, x: F64, where: string): F64 {
  return toMatrixF64(h(vec(x)), where, x.length, x.length).data
}

/**
 * Solves $(\Hmat + \tau\Imat)\pvec = -\gvec$ with the first $\tau$ on a doubling ladder that makes
 * $\Hmat + \tau\Imat$ positive definite (it factors by Cholesky): Nocedal & Wright (2006), Algorithm 3.3, "Cholesky
 * with added multiple of the identity". The ladder starts at $\tau = 0$ when every $H_{ii} > 0$, else at
 * $\beta - \min_i H_{ii}$, and continues $\tau \leftarrow \max(2\tau, \beta)$, with
 * $\beta = 10^{-3} \max_i \lvert H_{ii} \rvert$ (or $10^{-3}$ for a zero diagonal). At most 80 shifts are tried.
 *
 * @param H The Hessian $\Hmat$, a row-major array of $n^2$ values (symmetric; only its lower triangle is factored).
 * @param g The gradient $\gvec$, $n$ values.
 * @returns The `direction` $\pvec$ and the `shift` $\tau$ used, or null when $\Hmat$ has a non-finite entry or no
 *   shift tried made it factor.
 */
export function shiftedNewtonDirection(H: F64, g: F64): { direction: F64; shift: number } | null {
  const n = g.length
  if (!allFinite(H)) return null
  let minDiagonal = Infinity
  let maxDiagonal = 0
  for (let i = 0; i < n; i++) {
    minDiagonal = Math.min(minDiagonal, H[i * n + i])
    maxDiagonal = Math.max(maxDiagonal, Math.abs(H[i * n + i]))
  }
  const beta = 1e-3 * (maxDiagonal > 0 ? maxDiagonal : 1)
  let tau = minDiagonal > 0 ? 0 : beta - minDiagonal
  for (let attempt = 0; attempt < 80; attempt++) {
    const shifted = Float64Array.from(H)
    for (let i = 0; i < n; i++) shifted[i * n + i] += tau
    const c = cholesky(mat(shifted, n, n), { jitter: false })
    if (!c.failed) {
      const solved = data(choleskySolve(c.L, vec(scale(-1, g))))
      return { direction: Float64Array.from(solved), shift: tau }
    }
    tau = Math.max(2 * tau, beta)
  }
  return null
}

/** The state of `newton`. */
export type NewtonState = IterateState & {
  /** $\nabla f(\xvec)$. */
  grad: Vector
  /** $\lVert \nabla f(\xvec) \rVert_2$, compared with `tolerance`. */
  gradNorm: number
  /** $\nabla^2 f(\xvec)$. */
  hessian: Matrix
  /** The last Newton direction $\pvec = -(\Hmat + \tau\Imat)^{-1}\nabla f$ (zeros at $t = 0$). */
  direction: Vector
  /** The shift $\tau$ added to the Hessian's diagonal on the last step (0 when it was positive definite). */
  shift: number
  /** The accepted step length on the last step (1 is a full Newton step); NaN at $t = 0$. */
  stepSize: number
  /**
   * The Newton decrement squared, $-\nabla f^\top\pvec = \nabla f^\top(\Hmat + \tau\Imat)^{-1}\nabla f$, on the last
   * step: twice the predicted decrease. NaN at $t = 0$.
   */
  decrement: number
  /** The last line search with its trials; null for a pure Newton step or at $t = 0$. */
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower $f$ ($\xvec$ unchanged); the run stops. */
  stalled: boolean
}

/** Options for `newton`. */
export type NewtonOptions = StoppingOptions & {
  /** The Hessian $\nabla^2 f(\xvec)$, required. */
  hessian: Hessian
  /**
   * `'backtracking'` (default) damps the step with an Armijo backtracking search from $\alpha = 1$; `'strong-wolfe'`
   * uses a strong Wolfe search; `'none'` takes the full Newton step (pure Newton's method).
   */
  lineSearch?: 'backtracking' | 'strong-wolfe' | 'none'
  /** Options of the backtracking search. The strong Wolfe search always runs with its defaults. */
  lineSearchOptions?: BacktrackingOptions
}

/**
 * Damped Newton's method: $\pvec = -(\Hmat + \tau\Imat)^{-1}\nabla f(\xvec)$, with $\tau = 0$ when the Hessian
 * $\Hmat$ is positive definite and otherwise a shift that makes it so (Nocedal & Wright, Algorithm 3.3, see
 * `shiftedNewtonDirection`), then $\xvec \leftarrow \xvec + \alpha\pvec$ with $\alpha$ from a backtracking line search
 * ($\alpha = 1$ near a minimiser, where convergence is quadratic). The shift keeps $\pvec$ a descent direction, so
 * the method heads for a minimum and not a saddle point.
 *
 * If no shift makes $\Hmat + \tau\Imat$ factorable (a non-finite Hessian), the state is flagged `diverged`. A line
 * search that cannot lower $f$ sets `stalled`; both end the run.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The `hessian` (required), the line search and its options, and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Rosenbrock's function from the standard start
 * // (1 − a)² + 100(b − a²)², least at (1, 1), with its exact Hessian.
 * const rosenbrock = (x) => {
 *   const [a, b] = x.data
 *   return {
 *     value: (1 - a) ** 2 + 100 * (b - a * a) ** 2,
 *     grad: [-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)],
 *   }
 * }
 * const hessian = (x) => {
 *   const [a, b] = x.data
 *   return [
 *     [2 - 400 * (b - a * a) + 800 * a * a, -400 * a],
 *     [-400 * a, 200],
 *   ]
 * }
 * for (const steps of [1, 2, 3]) {
 *   const s = run(newton(rosenbrock, { hessian }), { x0: [-1.2, 1] }, steps)
 *   print(`after ${steps} steps: x =`, s.x, ' step length =', s.stepSize)
 * }
 * const s = run(newton(rosenbrock, { hessian }), { x0: [-1.2, 1] }, 100)
 * print('converged after', s.t, 'steps: x =', s.x, ' f =', s.value)
 *
 * @example A shift turns an indefinite Hessian into a descent direction
 * // f(x) = a² − b² + b⁴ has a saddle at (0, 0) and minima at (0, ±1/√2). At (1, 0.1) its Hessian is indefinite.
 * const f = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a - b * b + b ** 4, grad: [2 * a, -2 * b + 4 * b ** 3] }
 * }
 * const hessian = (x) => [
 *   [2, 0],
 *   [0, -2 + 12 * x.data[1] ** 2],
 * ]
 * const first = run(newton(f, { hessian }), { x0: [1, 0.1] }, 1)
 * print('shift on the first step =', first.shift)
 * const s = run(newton(f, { hessian }), { x0: [1, 0.1] }, 100)
 * print('x =', s.x, ' f =', s.value)
 * print('1 / sqrt(2) =', Math.SQRT1_2)
 */
export function newton(f: ObjectiveFn, options: NewtonOptions): Algorithm<StartOptions, NewtonState> {
  const {
    hessian,
    lineSearch = 'backtracking',
    tolerance = DEFAULT_TOLERANCE,
    divergeAbove = DEFAULT_DIVERGE,
  } = options
  const name = lineSearch === 'none' ? 'newton' : 'damped-newton'
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const { value, grad } = evaluate(f, x, name)
      const H = hessianAt(hessian, x, name)
      const gradNorm = norm(grad)
      return {
        t: 0,
        x: vec(x),
        value,
        grad: vec(grad),
        gradNorm,
        hessian: mat(H, x.length, x.length),
        direction: vec(new Float64Array(x.length)),
        shift: 0,
        stepSize: NaN,
        decrement: NaN,
        lineSearch: null,
        stalled: false,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (s) => {
      const n = s.x.shape[0]
      const x = data(s.x)
      const g = data(s.grad)
      const solved = shiftedNewtonDirection(data(s.hessian), g)
      if (!solved) return { ...s, t: s.t + 1, diverged: true }
      const { direction: p, shift } = solved
      let next: F64
      let value: number
      let grad: F64
      let alpha = 1
      let search: LineSearchResult | null = null
      let evaluations = 0
      if (lineSearch === 'none') {
        next = axpy(1, p, x)
        ;({ value, grad } = evaluate(f, next, name))
        evaluations = 1
      } else {
        const found =
          lineSearch === 'backtracking'
            ? backtrackingSearch(f, x, s.value, g, p, options.lineSearchOptions)
            : strongWolfeSearch(f, x, s.value, g, p)
        ;({ x: next, value, grad } = found)
        search = found.result
        alpha = found.result.alpha
        evaluations = found.result.evaluations
      }
      const gradNorm = norm(grad)
      return {
        t: s.t + 1,
        x: vec(next),
        value,
        grad: vec(grad),
        gradNorm,
        hessian: mat(hessianAt(hessian, next, name), n, n),
        direction: vec(p),
        shift,
        stepSize: alpha,
        decrement: -dot(g, p),
        lineSearch: search,
        stalled: search !== null && alpha === 0,
        evaluations: s.evaluations + evaluations,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, next, divergeAbove),
      }
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Trust region with the dogleg step.

/**
 * Which step the dogleg took: `'newton'` (the full step fits), `'dogleg'` (the path meets the boundary), `'cauchy'`
 * (steepest descent, cut at the boundary or at the model's minimum along it), or `'none'` (a zero gradient, and at
 * $t = 0$).
 */
export type DoglegKind = 'newton' | 'dogleg' | 'cauchy' | 'none'

/** The state of `trustRegion`. */
export type TrustRegionState = IterateState & {
  /** $\nabla f(\xvec)$. */
  grad: Vector
  /** $\lVert \nabla f(\xvec) \rVert_2$, compared with `tolerance`. */
  gradNorm: number
  /** $\nabla^2 f(\xvec)$. */
  hessian: Matrix
  /** The trust-region radius $\Delta$ for the next step. */
  radius: number
  /** The step $\pvec$ proposed on the last step (whether or not it was accepted). */
  step: Vector
  /** `newton` (the full step fits), `dogleg` (the path meets the boundary), `cauchy` (steepest descent). */
  stepKind: DoglegKind
  /** Actual reduction $f(\xvec) - f(\xvec + \pvec)$ on the last step. */
  actual: number
  /** Predicted reduction of the quadratic model, $-(\gvec^\top\pvec + \tfrac12\pvec^\top\Hmat\pvec)$. */
  predicted: number
  /** $\rho$ = actual / predicted: near 1 when the model is good. */
  ratio: number
  /** Whether the last proposed step was accepted (x moved). */
  accepted: boolean
}

/** Options for `trustRegion`. */
export type TrustRegionOptions = StoppingOptions & {
  /** The Hessian $\nabla^2 f(\xvec)$, required. */
  hessian: Hessian
  /** Initial radius $\Delta_0$. Default 1. */
  radius?: number
  /** Largest radius. Default 1e3. */
  maxRadius?: number
  /** Accept a step when $\rho > \eta$. Default 0.15 (Nocedal & Wright suggest $\eta \in [0, \tfrac14)$). */
  eta?: number
}

/**
 * The dogleg step for the model $m(\pvec) = \gvec^\top\pvec + \tfrac12\pvec^\top\Bmat\pvec$ within
 * $\lVert \pvec \rVert \le \Delta$ (Nocedal & Wright, §4.1): the full Newton step if $\Bmat$ is positive definite and
 * the step fits; otherwise the path from the origin to the minimiser along $-\gvec$ and on towards the Newton step,
 * cut at the boundary. When $\Bmat$ is not positive definite the Cauchy point is used (eq. 4.11–4.12).
 *
 * @param g The gradient $\gvec$, $n$ values.
 * @param B The model's Hessian $\Bmat$, a row-major array of $n^2$ values.
 * @param radius The trust-region radius $\Delta$.
 * @returns The step `p` and which kind it is; a zero step of kind `'none'` when $\gvec = \zeros$.
 */
function dogleg(g: F64, B: F64, radius: number): { p: F64; kind: DoglegKind } {
  const n = g.length
  const gNorm = norm(g)
  if (gNorm === 0) return { p: new Float64Array(n), kind: 'none' }
  const Bg = matVec(B, g, n, n)
  const gBg = dot(g, Bg)
  // Cauchy point: minimise the model along −g within the region.
  const tauC = gBg <= 0 ? 1 : Math.min(1, gNorm ** 3 / (radius * gBg))
  const cauchy = scale((-tauC * radius) / gNorm, g)
  const c = cholesky(mat(B, n, n), { jitter: false })
  if (c.failed) return { p: cauchy, kind: 'cauchy' }
  const newtonStep = Float64Array.from(data(choleskySolve(c.L, vec(scale(-1, g)))))
  if (norm(newtonStep) <= radius) return { p: newtonStep, kind: 'newton' }
  // The unconstrained minimiser along −g.
  const pu = scale(-dot(g, g) / gBg, g)
  if (norm(pu) >= radius) return { p: scale(-radius / gNorm, g), kind: 'cauchy' }
  // Solve ‖pu + s(pb − pu)‖ = Δ for s in [0, 1].
  const d = sub(newtonStep, pu)
  const a = dot(d, d)
  const b = 2 * dot(pu, d)
  const cc = dot(pu, pu) - radius * radius
  const sStar = (-b + Math.sqrt(b * b - 4 * a * cc)) / (2 * a)
  return { p: axpy(sStar, d, pu), kind: 'dogleg' }
}

/**
 * A trust-region method with the dogleg step (Nocedal & Wright, Algorithm 4.1): each step minimises the quadratic
 * model within radius $\Delta$, compares the actual and predicted reductions ($\rho$), accepts the step when
 * $\rho > \eta$, and doubles $\Delta$ (up to `maxRadius`, when $\rho > \tfrac34$ and the step reached the boundary) or
 * shrinks it to $\tfrac14\lVert \pvec \rVert$ (when $\rho < \tfrac14$, or $f$ was not finite at the trial point).
 * Rejected steps leave $\xvec$ unchanged. A radius that collapses below $10^{-14}$ counts as converged, since no step
 * can make progress; a non-finite Hessian flags `diverged`.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The `hessian` (required), the initial and largest radii, the acceptance threshold $\eta$, and the
 *   stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`, and each step evaluates $f$ once.
 *
 * @example Rosenbrock's function from the standard start
 * // The second step overshoots (ratio below 0) and is rejected; the radius shrinks and the next one is accepted.
 * const rosenbrock = (x) => {
 *   const [a, b] = x.data
 *   return {
 *     value: (1 - a) ** 2 + 100 * (b - a * a) ** 2,
 *     grad: [-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)],
 *   }
 * }
 * const hessian = (x) => {
 *   const [a, b] = x.data
 *   return [
 *     [2 - 400 * (b - a * a) + 800 * a * a, -400 * a],
 *     [-400 * a, 200],
 *   ]
 * }
 * for (const steps of [1, 2, 3]) {
 *   const s = run(trustRegion(rosenbrock, { hessian }), { x0: [-1.2, 1] }, steps)
 *   print(`step ${steps}:`, s.stepKind, ' ratio =', s.ratio, ' accepted =', s.accepted, ' radius =', s.radius)
 * }
 * const s = run(trustRegion(rosenbrock, { hessian }), { x0: [-1.2, 1] }, 200)
 * print('converged after', s.t, 'steps: x =', s.x, ' f =', s.value)
 */
export function trustRegion(f: ObjectiveFn, options: TrustRegionOptions): Algorithm<StartOptions, TrustRegionState> {
  const {
    hessian,
    radius: radius0 = 1,
    maxRadius = 1e3,
    eta = 0.15,
    tolerance = DEFAULT_TOLERANCE,
    divergeAbove = DEFAULT_DIVERGE,
  } = options
  const name = 'trust-region-dogleg'
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const { value, grad } = evaluate(f, x, name)
      const gradNorm = norm(grad)
      return {
        t: 0,
        x: vec(x),
        value,
        grad: vec(grad),
        gradNorm,
        hessian: mat(hessianAt(hessian, x, name), x.length, x.length),
        radius: radius0,
        step: vec(new Float64Array(x.length)),
        stepKind: 'none',
        actual: NaN,
        predicted: NaN,
        ratio: NaN,
        accepted: false,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (s) => {
      const n = s.x.shape[0]
      const x = data(s.x)
      const g = data(s.grad)
      const B = data(s.hessian)
      if (!allFinite(B)) return { ...s, t: s.t + 1, diverged: true }
      const { p, kind } = dogleg(g, B, s.radius)
      const predicted = -(dot(g, p) + 0.5 * dot(p, matVec(B, p, n, n)))
      const trial = axpy(1, p, x)
      const e = evaluate(f, trial, name)
      const actual = s.value - e.value
      // A non-finite trial value gives ρ = NaN, which fails every test below: the step is rejected and Δ shrinks.
      const ratio = predicted > 0 ? actual / predicted : actual >= 0 ? 1 : -1
      const pNorm = norm(p)
      let radius = s.radius
      if (!(ratio >= 0.25)) radius = 0.25 * pNorm
      else if (ratio > 0.75 && Math.abs(pNorm - s.radius) <= 1e-12 * s.radius)
        radius = Math.min(2 * s.radius, maxRadius)
      const accepted = ratio > eta
      const common = { t: s.t + 1, radius, step: vec(p), stepKind: kind, actual, predicted, ratio, accepted }
      if (!accepted) {
        // Δ collapsing to zero means no step can make progress at x (e.g. rounding at the minimum).
        return { ...s, ...common, evaluations: s.evaluations + 1, converged: s.converged || radius < 1e-14 }
      }
      const gradNorm = norm(e.grad)
      return {
        ...common,
        x: vec(trial),
        value: e.value,
        grad: vec(e.grad),
        gradNorm,
        hessian: mat(hessianAt(hessian, trial, name), n, n),
        evaluations: s.evaluations + 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(e.value, trial, divergeAbove),
      }
    },
    done: (s) => s.converged || s.diverged,
  }
}
