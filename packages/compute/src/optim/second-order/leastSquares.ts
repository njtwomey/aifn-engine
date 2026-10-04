/**
 * Nonlinear least squares, minimising f(x) = ½‖r(x)‖² for residuals r: ℝⁿ → ℝᵐ with Jacobian J (m×n): Gauss–Newton,
 * which solves the linearised problem min ‖Jp + r‖ at each step, and Levenberg–Marquardt, which damps it.
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

/** Residuals and their Jacobian at a point: r (length m) and J (m×n, J_ij = ∂r_i/∂x_j). */
export type ResidualFunction = (x: Vector) => { residuals: VectorLike; jacobian: MatrixLike }

type Residuals = { r: F64; J: F64; m: number; value: number; grad: F64 }

function residualsAt(fn: ResidualFunction, x: F64, where: string): Residuals {
  const out = fn(vec(x))
  const r = toF64(out.residuals, where)
  const m = r.length
  const { data: J } = toMatrixF64(out.jacobian, where, m, x.length)
  return { r, J, m, value: 0.5 * dot(r, r), grad: matTVec(J, r, m, x.length) }
}

/** ½‖r(x)‖² and its gradient Jᵀr as an objective, for line searches. */
function asObjective(fn: ResidualFunction, where: string): ObjectiveFn {
  return (x) => {
    const { value, grad } = residualsAt(fn, data(x), where)
    return { value, grad }
  }
}

/** The state of `gaussNewton` and `levenbergMarquardt`. */
export type LeastSquaresState = IterateState & {
  /** r(x). */
  residuals: Vector
  /** J(x), m×n. */
  jacobian: Matrix
  /** ∇f = Jᵀr. */
  grad: Vector
  gradNorm: number
  /** The step proposed on the last step (zeros at t = 0). */
  step: Vector
}

/** The state of `gaussNewton`. */
export type GaussNewtonState = LeastSquaresState & {
  /** Numerical rank of J at the last step (from the SVD); below n means the step is the minimum-norm one. */
  rank: number
  stepSize: number
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower f (x unchanged); the run stops. */
  stalled: boolean
}

/** Options for `gaussNewton`. */
export type GaussNewtonOptions = StoppingOptions & {
  /** `'backtracking'` (default, damped Gauss–Newton) or `'none'` (full steps). */
  lineSearch?: 'backtracking' | 'none'
  lineSearchOptions?: BacktrackingOptions
}

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
 * Gauss–Newton (Nocedal & Wright, §10.3): the step p minimises ‖J p + r‖ (solved by the SVD, so a rank-deficient J
 * gives the minimum-norm step), then x ← x + αp with α from Armijo backtracking on ½‖r‖² (or α = 1 with
 * `lineSearch: 'none'`). Convergence is fast when the residuals at the solution are small. `init` takes `{ x0 }`.
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
  /** The damping λ for the next step. */
  lambda: number
  /** Nielsen's growth factor ν for λ after a rejected step. */
  nu: number
  /** Gain ratio ρ = actual / predicted reduction of the last step. */
  ratio: number
  /** Whether the last proposed step was accepted (x moved). */
  accepted: boolean
}

/** Options for `levenbergMarquardt`. */
export type LevenbergMarquardtOptions = StoppingOptions & {
  /** Initial λ = τ · max diag(JᵀJ). Default τ = 1e-3. */
  tau?: number
  /**
   * `'marquardt'` damps with λ·diag(JᵀJ) (Marquardt, 1963), making the step invariant to rescaling x; `'identity'`
   * (default) damps with λI (Levenberg, 1944).
   */
  scaling?: 'identity' | 'marquardt'
}

/**
 * Levenberg–Marquardt (Levenberg, 1944; Marquardt, 1963) with Nielsen's (1999) damping update: solve
 * (JᵀJ + λD)p = −Jᵀr, accept when the gain ratio ρ is positive, then λ ← λ·max(⅓, 1 − (2ρ − 1)³) and ν ← 2 on
 * success, λ ← λν and ν ← 2ν on failure (Madsen, Nielsen & Tingleff, 2004, "Methods for Non-Linear Least Squares
 * Problems", Algorithm 3.16). Large λ gives short gradient-descent steps, small λ Gauss–Newton steps. `init` takes
 * `{ x0 }`.
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
  method: 'gauss-newton' | 'levenberg-marquardt'
  x: Vector
  /** ½‖r(x)‖². */
  value: number
  residuals: Vector
  gradNorm: number
  steps: number
  evaluations: number
  converged: boolean
  diverged: boolean
}

/**
 * Minimises ½‖r(x)‖² from `x0` by Levenberg–Marquardt (default) or Gauss–Newton, for at most `maxSteps` steps
 * (default 200). A `run` wrapper over `levenbergMarquardt` / `gaussNewton`.
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
