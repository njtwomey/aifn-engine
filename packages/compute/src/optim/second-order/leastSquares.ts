/**
 * Nonlinear least squares, minimising $f(\xvec) = \tfrac12\lVert \rvec(\xvec) \rVert^2$ for residuals
 * $\rvec: \reals^n \to \reals^m$ with Jacobian $\Jmat$ ($m \times n$): Gauss–Newton, which solves the linearised
 * problem $\min_{\pvec} \lVert \Jmat\pvec + \rvec \rVert$ at each step, and Levenberg–Marquardt, which damps it.
 *
 * Both use $\Jmat^\top\Jmat$ in place of the Hessian of $f$, which drops the second derivatives of the residuals, so
 * they need only the residuals and their Jacobian, and converge fast when the residuals at the solution are small.
 * The gradient is $\nabla f = \Jmat^\top\rvec$. Non-convergence and divergence (a non-finite value, iterate or
 * Jacobian) are reported in the state, not thrown.
 */

import { cholesky, choleskySolve, lstsq } from 'aifn-compute/numerics/linalg'
import type { Matrix, Vector } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { RunOptions, StartOptions } from '../options'
import { backtrackingSearch, type BacktrackingOptions, type LineSearchResult } from 'aifn-compute/optim/line-search'
import type {
  IterateState,
  MatrixLike,
  ObjectiveFn,
  StoppingOptions,
  VectorLike,
} from 'aifn-compute/foundation/contracts'
import { DEFAULT_DIVERGE, DEFAULT_TOLERANCE, divergedAt } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'

const { allFinite, axpy, data, dot, gram, mat, matTVec, matVec, norm, scale, toF64, toMatrixF64, vec } = dense
type F64 = dense.F64

/**
 * Residuals and their Jacobian at a point: `residuals` $\rvec$ (length $m$) and `jacobian` $\Jmat$
 * ($m \times n$, $J_{ij} = \partial r_i / \partial x_j$).
 */
export type ResidualFunction = (x: Vector) => { residuals: VectorLike; jacobian: MatrixLike }

/**
 * The residuals at a point as working arrays: $\rvec$, $\Jmat$ (row-major), $m$, and the value
 * $\tfrac12\lVert \rvec \rVert^2$ and gradient $\Jmat^\top\rvec$ of the objective.
 */
type Residuals = { r: F64; J: F64; m: number; value: number; grad: F64 }

/**
 * Evaluates the residual function at a point and derives the least-squares value and gradient from it.
 *
 * @param fn The residual function.
 * @param x The point, $n$ values.
 * @param where The caller's name, for error messages (a Jacobian that is not $m \times n$ throws).
 * @returns The residuals, the Jacobian, $m$, $\tfrac12\lVert \rvec \rVert^2$ and $\Jmat^\top\rvec$.
 */
function residualsAt(fn: ResidualFunction, x: F64, where: string): Residuals {
  const out = fn(vec(x))
  const r = toF64(out.residuals, where)
  const m = r.length
  const { data: J } = toMatrixF64(out.jacobian, where, m, x.length)
  return { r, J, m, value: 0.5 * dot(r, r), grad: matTVec(J, r, m, x.length) }
}

/**
 * $\tfrac12\lVert \rvec(\xvec) \rVert^2$ and its gradient $\Jmat^\top\rvec$ as an objective, for line searches.
 *
 * @param fn The residual function.
 * @param where The caller's name, for error messages.
 * @returns The objective, returning `{ value, grad }`.
 */
function asObjective(fn: ResidualFunction, where: string): ObjectiveFn {
  return (x) => {
    const { value, grad } = residualsAt(fn, data(x), where)
    return { value, grad }
  }
}

/** The state of `gaussNewton` and `levenbergMarquardt`. */
export type LeastSquaresState = IterateState & {
  /** $\rvec(\xvec)$. */
  residuals: Vector
  /** $\Jmat(\xvec)$, $m \times n$. */
  jacobian: Matrix
  /** $\nabla f = \Jmat^\top\rvec$. */
  grad: Vector
  /** $\lVert \Jmat^\top\rvec \rVert_2$, compared with `tolerance`. */
  gradNorm: number
  /** The step proposed on the last step (zeros at $t = 0$). */
  step: Vector
}

/** The state of `gaussNewton`. */
export type GaussNewtonState = LeastSquaresState & {
  /**
   * Numerical rank of $\Jmat$ at the last step (from the SVD); below $n$ means the step is the minimum-norm one. NaN
   * at $t = 0$.
   */
  rank: number
  /** The step length $\alpha$ taken along the last step (1 without a line search); NaN at $t = 0$. */
  stepSize: number
  /** The last backtracking search, with its trial points; null without a line search or at $t = 0$. */
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower $f$ ($\xvec$ unchanged); the run stops. */
  stalled: boolean
}

/** Options for `gaussNewton`. */
export type GaussNewtonOptions = StoppingOptions & {
  /** `'backtracking'` (default, damped Gauss–Newton) or `'none'` (full steps). */
  lineSearch?: 'backtracking' | 'none'
  /** Options of the backtracking search. */
  lineSearchOptions?: BacktrackingOptions
}

/**
 * The state shared by both methods at the start point: the residuals, Jacobian, value and gradient there.
 *
 * @param fn The residual function.
 * @param x0 The start point $\xvec_0$.
 * @param name The method's name, for error messages.
 * @param tolerance The gradient-norm tolerance for `converged`.
 * @param divergeAbove The value above which the state is flagged `diverged` (as is a non-finite Jacobian).
 * @returns The state at $t = 0$, with one evaluation counted.
 */
function initial(fn: ResidualFunction, x0: VectorLike, name: string, tolerance: number, divergeAbove: number) {
  const x = toF64(x0, name)
  const e = residualsAt(fn, x, name)
  const gradNorm = norm(e.grad)
  const state: LeastSquaresState = {
    t: 0,
    x: vec(x),
    value: e.value,
    residuals: vec(e.r),
    jacobian: mat(e.J, e.m, x.length),
    grad: vec(e.grad),
    gradNorm,
    step: vec(new Float64Array(x.length)),
    evaluations: 1,
    converged: gradNorm <= tolerance,
    diverged: divergedAt(e.value, x, divergeAbove) || !allFinite(e.J),
  }
  return state
}

/**
 * Gauss–Newton (Nocedal & Wright, §10.3): the step $\pvec$ minimises $\lVert \Jmat\pvec + \rvec \rVert$ (solved by
 * the SVD, so a rank-deficient $\Jmat$ gives the minimum-norm step), then $\xvec \leftarrow \xvec + \alpha\pvec$ with
 * $\alpha$ from Armijo backtracking on $\tfrac12\lVert \rvec \rVert^2$ (or $\alpha = 1$ with `lineSearch: 'none'`).
 * Convergence is fast when the residuals at the solution are small. A line search that cannot lower $f$ sets
 * `stalled` and ends the run.
 *
 * @param residuals The residual function, returning $\rvec(\xvec)$ and $\Jmat(\xvec)$.
 * @param options The line search and its options, and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Fit an exponential decay
 * // Fit y = a·exp(bt) to four exact points of 2·exp(−t/2), from (a, b) = (1, 0).
 * const t = [0, 1, 2, 3]
 * const y = t.map((ti) => 2 * Math.exp(-0.5 * ti))
 * const residuals = (x) => {
 *   const [a, b] = x.data
 *   return {
 *     residuals: t.map((ti, i) => a * Math.exp(b * ti) - y[i]),
 *     jacobian: t.map((ti) => [Math.exp(b * ti), a * ti * Math.exp(b * ti)]),
 *   }
 * }
 * for (const steps of [1, 2, 4]) {
 *   const s = run(gaussNewton(residuals), { x0: [1, 0] }, steps)
 *   print(`after ${steps} steps: (a, b) =`, s.x, ' f =', s.value)
 * }
 */
export function gaussNewton(
  residuals: ResidualFunction,
  options: GaussNewtonOptions = {},
): Algorithm<StartOptions, GaussNewtonState> {
  const { lineSearch = 'backtracking', tolerance = DEFAULT_TOLERANCE, divergeAbove = DEFAULT_DIVERGE } = options
  const name = 'gauss-newton'
  const objective = asObjective(residuals, name)
  return {
    name,
    init: ({ x0 }) => ({
      ...initial(residuals, x0, name, tolerance, divergeAbove),
      rank: NaN,
      stepSize: NaN,
      lineSearch: null,
      stalled: false,
    }),
    step: (s) => {
      const x = data(s.x)
      const r = data(s.residuals)
      const solution = lstsq(s.jacobian, vec(scale(-1, r)))
      const p = Float64Array.from(data(solution.x))
      let next: F64
      let search: LineSearchResult | null = null
      let evaluations = 1
      let alpha = 1
      if (lineSearch === 'backtracking') {
        const found = backtrackingSearch(objective, x, s.value, data(s.grad), p, options.lineSearchOptions)
        next = found.x
        search = found.result
        alpha = found.result.alpha
        evaluations = found.result.evaluations + 1
      } else next = axpy(1, p, x)
      const e = residualsAt(residuals, next, name)
      const gradNorm = norm(e.grad)
      return {
        t: s.t + 1,
        x: vec(next),
        value: e.value,
        residuals: vec(e.r),
        jacobian: mat(e.J, e.m, x.length),
        grad: vec(e.grad),
        gradNorm,
        step: vec(p),
        rank: solution.rank,
        stepSize: alpha,
        lineSearch: search,
        stalled: search !== null && alpha === 0,
        evaluations: s.evaluations + evaluations,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(e.value, next, divergeAbove) || !allFinite(e.J),
      }
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}

/** The state of `levenbergMarquardt`. */
export type LevenbergMarquardtState = LeastSquaresState & {
  /** The damping $\lambda$ for the next step. */
  lambda: number
  /** Nielsen's growth factor $\nu$ for $\lambda$ after a rejected step (2 after an accepted one). */
  nu: number
  /** Gain ratio $\rho$ = actual / predicted reduction of the last step (NaN at $t = 0$). */
  ratio: number
  /** Whether the last proposed step was accepted (x moved). */
  accepted: boolean
}

/** Options for `levenbergMarquardt`. */
export type LevenbergMarquardtOptions = StoppingOptions & {
  /** $\tau$ in the initial damping $\lambda = \tau \max_i (\Jmat^\top\Jmat)_{ii}$. Default 1e-3. */
  tau?: number
  /**
   * `'marquardt'` damps with $\lambda\diag(\Jmat^\top\Jmat)$ (Marquardt, 1963; each entry at least $10^{-12}$),
   * making the step invariant to rescaling $\xvec$; `'identity'` (default) damps with $\lambda\Imat$ (Levenberg, 1944).
   */
  scaling?: 'identity' | 'marquardt'
}

/**
 * Levenberg–Marquardt (Levenberg, 1944; Marquardt, 1963) with Nielsen's (1999) damping update: solve
 * $(\Jmat^\top\Jmat + \lambda\Dmat)\pvec = -\Jmat^\top\rvec$, accept when the gain ratio $\rho$ is positive, then
 * $\lambda \leftarrow \lambda\max(\tfrac13, 1 - (2\rho - 1)^3)$ and $\nu \leftarrow 2$ on success,
 * $\lambda \leftarrow \lambda\nu$ and $\nu \leftarrow 2\nu$ on failure (Madsen, Nielsen & Tingleff, 2004, "Methods
 * for Non-Linear Least Squares Problems", Algorithm 3.16). Large $\lambda$ gives short gradient-descent steps, small
 * $\lambda$ Gauss–Newton steps. $\Dmat$ is $\Imat$ or $\diag(\Jmat^\top\Jmat)$ (see `scaling`). A rejected step leaves
 * $\xvec$ unchanged; a step too small to change $\xvec$ counts as converged.
 *
 * @param residuals The residual function, returning $\rvec(\xvec)$ and $\Jmat(\xvec)$.
 * @param options The initial damping factor $\tau$, the damping `scaling`, and the stopping options.
 * @returns The algorithm; `init` takes `{ x0 }`, and each step evaluates the residuals once.
 *
 * @example Fit an exponential decay, watching the damping fall
 * // Fit y = a·exp(bt) to four exact points of 2·exp(−t/2), from (a, b) = (1, 0).
 * const t = [0, 1, 2, 3]
 * const y = t.map((ti) => 2 * Math.exp(-0.5 * ti))
 * const residuals = (x) => {
 *   const [a, b] = x.data
 *   return {
 *     residuals: t.map((ti, i) => a * Math.exp(b * ti) - y[i]),
 *     jacobian: t.map((ti) => [Math.exp(b * ti), a * ti * Math.exp(b * ti)]),
 *   }
 * }
 * for (const steps of [1, 2, 4]) {
 *   const s = run(levenbergMarquardt(residuals), { x0: [1, 0] }, steps)
 *   print(`after ${steps} steps: (a, b) =`, s.x, ' lambda =', s.lambda, ' accepted =', s.accepted)
 * }
 */
export function levenbergMarquardt(
  residuals: ResidualFunction,
  options: LevenbergMarquardtOptions = {},
): Algorithm<StartOptions, LevenbergMarquardtState> {
  const { tau = 1e-3, scaling = 'identity', tolerance = DEFAULT_TOLERANCE, divergeAbove = DEFAULT_DIVERGE } = options
  const name = 'levenberg-marquardt'
  return {
    name,
    init: ({ x0 }) => {
      const base = initial(residuals, x0, name, tolerance, divergeAbove)
      const n = base.x.shape[0]
      const A = gram(data(base.jacobian), base.residuals.shape[0], n)
      let maxDiagonal = 0
      for (let i = 0; i < n; i++) maxDiagonal = Math.max(maxDiagonal, A[i * n + i])
      return { ...base, lambda: tau * (maxDiagonal > 0 ? maxDiagonal : 1), nu: 2, ratio: NaN, accepted: false }
    },
    step: (s) => {
      const n = s.x.shape[0]
      const m = s.residuals.shape[0]
      const x = data(s.x)
      const g = data(s.grad)
      const A = gram(data(s.jacobian), m, n)
      const damped = Float64Array.from(A)
      for (let i = 0; i < n; i++)
        damped[i * n + i] += s.lambda * (scaling === 'marquardt' ? Math.max(A[i * n + i], 1e-12) : 1)
      const c = cholesky(mat(damped, n, n), { jitter: false })
      if (c.failed) return { ...s, t: s.t + 1, lambda: s.lambda * s.nu, nu: 2 * s.nu, accepted: false, ratio: NaN }
      const p = Float64Array.from(data(choleskySolve(c.L, vec(scale(-1, g)))))
      const trial = axpy(1, p, x)
      const e = residualsAt(residuals, trial, name)
      // Predicted reduction of the linear model: L(0) − L(p) = −pᵀJᵀr − ½‖Jp‖².
      const Jp = matVec(data(s.jacobian), p, m, n)
      const predicted = -dot(p, g) - 0.5 * dot(Jp, Jp)
      const ratio = (s.value - e.value) / predicted
      const common = { t: s.t + 1, step: vec(p), ratio, evaluations: s.evaluations + 1 }
      if (ratio > 0 && Number.isFinite(e.value)) {
        const gradNorm = norm(e.grad)
        return {
          ...common,
          x: vec(trial),
          value: e.value,
          residuals: vec(e.r),
          jacobian: mat(e.J, m, n),
          grad: vec(e.grad),
          gradNorm,
          lambda: s.lambda * Math.max(1 / 3, 1 - (2 * ratio - 1) ** 3),
          nu: 2,
          accepted: true,
          converged: gradNorm <= tolerance,
          diverged: divergedAt(e.value, trial, divergeAbove) || !allFinite(e.J),
        }
      }
      // A step too small to change x in floating point means λ has grown past any useful value.
      const stuck = norm(p) <= 1e-15 * (norm(x) + 1e-15)
      return {
        ...s,
        ...common,
        lambda: s.lambda * s.nu,
        nu: 2 * s.nu,
        accepted: false,
        converged: s.converged || stuck,
      }
    },
    done: (s) => s.converged || s.diverged,
  }
}

/** The result of `leastSquares`. */
export type LeastSquaresResult = {
  /** The method that ran. */
  method: 'gauss-newton' | 'levenberg-marquardt'
  /** The final iterate. */
  x: Vector
  /** $\tfrac12\lVert \rvec(\xvec) \rVert^2$. */
  value: number
  /** $\rvec(\xvec)$ at the final iterate. */
  residuals: Vector
  /** $\lVert \Jmat^\top\rvec \rVert_2$ at the final iterate. */
  gradNorm: number
  /** Steps taken. */
  steps: number
  /** Evaluations of the residual function, the initial one included. */
  evaluations: number
  /** The method's stopping test passed. */
  converged: boolean
  /** The value, iterate or Jacobian became non-finite, or the value passed `divergeAbove`. */
  diverged: boolean
}

/**
 * Minimises $\tfrac12\lVert \rvec(\xvec) \rVert^2$ from `x0` by Levenberg–Marquardt (default) or Gauss–Newton, for
 * at most `maxSteps` steps (default 200). A `run` wrapper over `levenbergMarquardt` / `gaussNewton`; failure to
 * converge is reported in the result.
 *
 * @param residuals The residual function, returning $\rvec(\xvec)$ and $\Jmat(\xvec)$.
 * @param x0 The start point $\xvec_0$.
 * @param options `method` and `maxSteps`, and the chosen method's options.
 * @returns The final iterate, its value, residuals and gradient norm, the steps and evaluations used, and whether it
 *   converged or diverged.
 *
 * @example Fit an exponential decay in one call
 * // Fit y = a·exp(bt) to four exact points of 2·exp(−t/2), from (a, b) = (1, 0).
 * const t = [0, 1, 2, 3]
 * const y = t.map((ti) => 2 * Math.exp(-0.5 * ti))
 * const residuals = (x) => {
 *   const [a, b] = x.data
 *   return {
 *     residuals: t.map((ti, i) => a * Math.exp(b * ti) - y[i]),
 *     jacobian: t.map((ti) => [Math.exp(b * ti), a * ti * Math.exp(b * ti)]),
 *   }
 * }
 * const fit = leastSquares(residuals, [1, 0])
 * print('(a, b) =', fit.x, ' f =', fit.value)
 * print('steps =', fit.steps, ' converged =', fit.converged)
 */
export function leastSquares(
  residuals: ResidualFunction,
  x0: VectorLike,
  options: { method?: 'gauss-newton' | 'levenberg-marquardt' } & Pick<RunOptions, 'maxSteps'> &
    LevenbergMarquardtOptions &
    GaussNewtonOptions = {},
): LeastSquaresResult {
  const { method = 'levenberg-marquardt', maxSteps = 200 } = options
  const s: LeastSquaresState =
    method === 'gauss-newton'
      ? run(gaussNewton(residuals, options), { x0 }, maxSteps)
      : run(levenbergMarquardt(residuals, options), { x0 }, maxSteps)
  return {
    method,
    x: s.x,
    value: s.value,
    residuals: s.residuals,
    gradNorm: s.gradNorm,
    steps: s.t,
    evaluations: s.evaluations,
    converged: s.converged,
    diverged: s.diverged,
  }
}
