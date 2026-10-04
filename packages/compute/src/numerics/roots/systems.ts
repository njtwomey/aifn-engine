/**
 * Systems of nonlinear equations F(x) = 0 with F: ℝⁿ → ℝⁿ: Newton's method (optionally damped by a backtracking line
 * search on ½‖F‖²), Broyden's quasi-Newton method, fixed-point iteration with convergence diagnostics, and natural-
 * parameter continuation along a homotopy.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import { solveDense } from 'aifn-compute/numerics/linalg'
import { dense, type Matrix, SQRT_EPS, type Vector } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { flagged } from './status'

type F64 = dense.F64
const { allFinite, axpy, data, dot, mat, matVec, norm, scale, sub, toF64, toMatrixF64, vec } = dense

/** A system with its Jacobian: F(x) (length n) and J(x) (n×n, J_ij = ∂F_i/∂x_j). */
export type SystemWithJacobian = (x: Vector) => { value: VectorLike; jacobian: MatrixLike }

/** A system without derivatives: F(x) (length n). */
export type SystemFunction = (x: Vector) => VectorLike

/** Fields every system-solver state carries: the runner's `Status` (set from `failure`) and the iterate. */
export type SystemState = Status & {
  t: Size
  x: Vector
  /** F(x). */
  residual: Vector
  /** ‖F(x)‖₂. */
  residualNorm: number
  /** The step taken last, x_t − x_{t−1} (zeros at t = 0). */
  step: Vector
  evaluations: number
  converged: boolean
  /**
   * Why the method cannot continue, or null: `'singular jacobian'`, `'line search failed'` (both set `terminated`),
   * `'not finite'` or `'diverging'` (both set `diverged`).
   */
  failure: string | null
}

/** Stopping test for systems. */
export type SystemTolerance = {
  /** Stop when ‖F(x)‖₂ ≤ ftol. Default 1e-12. */
  ftol?: number
  /** Or when the step satisfies ‖Δx‖ ≤ xtol·(1 + ‖x‖). Default 1e-14. */
  xtol?: number
}

const tolerances = (o: SystemTolerance) => ({ ftol: o.ftol ?? 1e-12, xtol: o.xtol ?? 1e-14 })

/** Solves A p = b by LU with partial pivoting; null when A is singular. */
function linearSolve(A: F64, b: F64, n: number): F64 | null {
  if (!allFinite(b)) return null
  return solveDense(A, b, n).x as F64 | null
}

function evaluateSystem(F: SystemWithJacobian, x: F64, where: string) {
  const out = F(vec(x))
  const r = toF64(out.value, where)
  if (r.length !== x.length)
    throw new ShapeError(where, `${where}: F returned ${r.length} values for ${x.length} unknowns`)
  return { r, J: toMatrixF64(out.jacobian, where, x.length, x.length).data }
}

/** The state of `newtonSystem`. */
export type NewtonSystemState = SystemState & {
  /** J(x). */
  jacobian: Matrix
  /** The full Newton step −J⁻¹F computed on the last step. */
  newtonStep: Vector
  /** The fraction α of the Newton step taken (1 unless damped). */
  damping: number
  /** The step lengths tried by the backtracking search on the last step (damped only). */
  trials: number[]
}

/**
 * Newton's method for F(x) = 0: solve J(x)p = −F(x) and set x ← x + αp. Without damping α = 1 (quadratic convergence
 * near a root with nonsingular Jacobian); with `damped`, α is halved until the merit ½‖F‖² satisfies
 * ½‖F(x + αp)‖² ≤ (1 − 2cα)·½‖F(x)‖² with c = 1e-4 (Nocedal & Wright, 2006, §11.2, Algorithm 11.4 with a backtracking
 * line search). `init` takes `{ x0 }`.
 */
export function newtonSystem(
  F: SystemWithJacobian,
  options: SystemTolerance & { damped?: boolean } = {},
): Algorithm<{ x0: VectorLike }, NewtonSystemState> {
  const { ftol, xtol } = tolerances(options)
  const name = options.damped ? 'damped-newton-system' : 'newton-system'
  return flagged<{ x0: VectorLike }, NewtonSystemState>({
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const { r, J } = evaluateSystem(F, x, name)
      const residualNorm = norm(r)
      return {
        t: 0,
        x: vec(x),
        residual: vec(r),
        residualNorm,
        jacobian: mat(J, x.length, x.length),
        step: vec(new Float64Array(x.length)),
        newtonStep: vec(new Float64Array(x.length)),
        damping: 1,
        trials: [],
        evaluations: 1,
        converged: residualNorm <= ftol,
        failure: allFinite(r) && allFinite(J) ? null : 'not finite',
      }
    },
    step: (s) => {
      const n = s.x.shape[0]
      const x = data(s.x)
      const r = data(s.residual)
      const p = linearSolve(data(s.jacobian), scale(-1, r), n)
      if (!p) return { ...s, t: s.t + 1, failure: 'singular jacobian' }
      let alpha = 1
      let next = axpy(1, p, x)
      let e = evaluateSystem(F, next, name)
      let evaluations = 1
      const trials = [1]
      if (options.damped) {
        const merit = 0.5 * dot(r, r)
        while (!(0.5 * dot(e.r, e.r) <= (1 - 2e-4 * alpha) * merit) && trials.length < 40) {
          alpha /= 2
          next = axpy(alpha, p, x)
          e = evaluateSystem(F, next, name)
          evaluations++
          trials.push(alpha)
        }
      }
      const residualNorm = norm(e.r)
      const step = sub(next, x)
      return {
        t: s.t + 1,
        x: vec(next),
        residual: vec(e.r),
        residualNorm,
        jacobian: mat(e.J, n, n),
        step: vec(step),
        newtonStep: vec(p),
        damping: alpha,
        trials: options.damped ? trials : [],
        evaluations: s.evaluations + evaluations,
        converged: residualNorm <= ftol || norm(step) <= xtol * (1 + norm(next)),
        failure: allFinite(e.r) && allFinite(e.J) ? null : 'not finite',
      }
    },
  })
}

/** The state of `broyden`. */
export type BroydenState = SystemState & {
  /** The Jacobian approximation B (n×n), updated by rank one each step. */
  jacobian: Matrix
  /** The last change in F, y = F(x_t) − F(x_{t−1}). */
  residualChange: Vector
}

/** Forward-difference Jacobian, column j ≈ (F(x + h e_j) − F(x)) / h with h = √ε·max(1, |x_j|). */
function finiteDifferenceJacobian(F: SystemFunction, x: F64, r: F64, where: string): F64 {
  const n = x.length
  const J = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    const h = SQRT_EPS * Math.max(1, Math.abs(x[j]))
    const xh = Float64Array.from(x)
    xh[j] += h
    const rh = toF64(F(vec(xh)), where)
    for (let i = 0; i < n; i++) J[i * n + j] = (rh[i] - r[i]) / h
  }
  return J
}

/**
 * Broyden's ("good") method (Broyden, 1965, "A class of methods for solving nonlinear simultaneous equations", Math.
 * Comp. 19): solve Bp = −F, x ← x + p, then B ← B + (y − Bs)sᵀ/(sᵀs) with s = Δx and y = ΔF, the smallest change to
 * B (in the Frobenius norm) that satisfies the secant condition Bs = y. `F` returns F(x) only; the initial B is
 * `jacobian0` or a forward-difference Jacobian (n extra evaluations). Converges superlinearly. `init` takes `{ x0 }`.
 */
export function broyden(
  F: SystemFunction,
  options: SystemTolerance & { jacobian0?: MatrixLike } = {},
): Algorithm<{ x0: VectorLike }, BroydenState> {
  const { ftol, xtol } = tolerances(options)
  const name = 'broyden'
  return flagged<{ x0: VectorLike }, BroydenState>({
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const n = x.length
      const r = toF64(F(vec(x)), name)
      const J = options.jacobian0
        ? toMatrixF64(options.jacobian0, name, n, n).data
        : finiteDifferenceJacobian(F, x, r, name)
      const residualNorm = norm(r)
      return {
        t: 0,
        x: vec(x),
        residual: vec(r),
        residualNorm,
        jacobian: mat(J, n, n),
        residualChange: vec(new Float64Array(n)),
        step: vec(new Float64Array(n)),
        evaluations: options.jacobian0 ? 1 : 1 + n,
        converged: residualNorm <= ftol,
        failure: allFinite(r) ? null : 'not finite',
      }
    },
    step: (st) => {
      const n = st.x.shape[0]
      const x = data(st.x)
      const r = data(st.residual)
      const B = data(st.jacobian)
      const s = linearSolve(B, scale(-1, r), n)
      if (!s) return { ...st, t: st.t + 1, failure: 'singular jacobian' }
      const next = axpy(1, s, x)
      const r1 = toF64(F(vec(next)), name)
      const y = sub(r1, r)
      const ss = dot(s, s)
      const correction = sub(y, matVec(B, s, n, n))
      const B1 = Float64Array.from(B)
      if (ss > 0) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B1[i * n + j] += (correction[i] * s[j]) / ss
      const residualNorm = norm(r1)
      return {
        t: st.t + 1,
        x: vec(next),
        residual: vec(r1),
        residualNorm,
        jacobian: mat(B1, n, n),
        residualChange: vec(y),
        step: vec(s),
        evaluations: st.evaluations + 1,
        converged: residualNorm <= ftol || Math.sqrt(ss) <= xtol * (1 + norm(next)),
        failure: allFinite(r1) ? null : 'not finite',
      }
    },
  })
}

/** The state of `fixedPoint`. `residual` is g(x) − x, zero at a fixed point. */
export type FixedPointState = SystemState & {
  /** g(x). */
  image: Vector
  /**
   * The observed contraction factor ‖x_{t+1} − x_t‖ / ‖x_t − x_{t−1}‖ on the last step (NaN until two steps exist):
   * an estimate of the Lipschitz constant L of g near the fixed point. Below 1 means linear convergence at rate L.
   */
  contraction: number
  /**
   * The a-posteriori error bound ‖x_t − x*‖ ≤ L/(1 − L)·‖x_t − x_{t−1}‖ with L the observed contraction (Infinity when
   * L ≥ 1 or unknown). A bound only if g really contracts with constant L.
   */
  errorBound: number
  /** Consecutive steps whose contraction factor was ≥ 1: a run of them suggests the iteration is not converging. */
  expandingSteps: number
}

/**
 * Fixed-point iteration x ← (1 − ω)x + ω·g(x) (ω = `relaxation`, default 1): converges linearly to the fixed point of a
 * contraction (Banach). The state reports the residual ‖g(x) − x‖, the observed contraction factor and an a-posteriori
 * error bound. Stops when ‖g(x) − x‖ ≤ ftol; flags `'diverging'` as a failure after `patience` (default 20)
 * consecutive expanding steps. `g` maps a vector of length n to one of length n; `init` takes `{ x0 }`.
 */
export function fixedPoint(
  g: SystemFunction,
  options: SystemTolerance & { relaxation?: number; patience?: number } = {},
): Algorithm<{ x0: VectorLike }, FixedPointState> {
  const { ftol } = tolerances(options)
  const omega = options.relaxation ?? 1
  const patience = options.patience ?? 20
  const name = 'fixed-point'
  const image = (x: F64) => {
    const gx = toF64(g(vec(x)), name)
    if (gx.length !== x.length) throw new ShapeError(name, `${name}: g returned ${gx.length} values for ${x.length}`)
    return gx
  }
  return flagged<{ x0: VectorLike }, FixedPointState>({
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const gx = image(x)
      const r = sub(gx, x)
      const residualNorm = norm(r)
      return {
        t: 0,
        x: vec(x),
        image: vec(gx),
        residual: vec(r),
        residualNorm,
        step: vec(new Float64Array(x.length)),
        contraction: NaN,
        errorBound: Infinity,
        expandingSteps: 0,
        evaluations: 1,
        converged: residualNorm <= ftol,
        failure: allFinite(gx) ? null : 'not finite',
      }
    },
    step: (s) => {
      const x = data(s.x)
      const next = axpy(omega, data(s.residual), x)
      const gx = image(next)
      const r = sub(gx, next)
      const step = sub(next, x)
      const stepNorm = norm(step)
      const previousStep = norm(data(s.step))
      const contraction = s.t === 0 || previousStep === 0 ? NaN : stepNorm / previousStep
      const expandingSteps = contraction >= 1 ? s.expandingSteps + 1 : 0
      const residualNorm = norm(r)
      let failure: string | null = allFinite(gx) ? null : 'not finite'
      if (!failure && expandingSteps >= patience) failure = 'diverging'
      return {
        t: s.t + 1,
        x: vec(next),
        image: vec(gx),
        residual: vec(r),
        residualNorm,
        step: vec(step),
        contraction,
        errorBound: contraction < 1 ? (contraction / (1 - contraction)) * stepNorm : Infinity,
        expandingSteps,
        evaluations: s.evaluations + 1,
        converged: residualNorm <= ftol,
        failure,
      }
    },
  })
}

// ---------------------------------------------------------------------------------------------------------------------
// Continuation.

/** A homotopy H(x, λ) with its Jacobian in x, for λ from 0 (easy) to 1 (the target). */
export type Homotopy = (x: Vector, lambda: number) => { value: VectorLike; jacobian: MatrixLike }

/**
 * The Newton homotopy H(x, λ) = F(x) − (1 − λ)F(x₀), which x₀ solves at λ = 0 and whose λ = 1 solutions are the roots
 * of F. Its Jacobian in x is J_F(x).
 */
export function newtonHomotopy(F: SystemWithJacobian, x0: VectorLike): Homotopy {
  const start = toF64(F(vec(toF64(x0, 'newtonHomotopy'))).value, 'newtonHomotopy')
  return (x, lambda) => {
    const { value, jacobian } = F(x)
    return { value: vec(axpy(-(1 - lambda), start, toF64(value, 'newtonHomotopy'))), jacobian }
  }
}

/** The state of `continuation`. */
export type ContinuationState = Status & {
  t: Size
  /** The continuation parameter reached, in [0, 1]. */
  lambda: number
  /** The solution of H(x, λ) = 0 at `lambda`. */
  x: Vector
  /** The next increment Δλ to try. */
  dLambda: number
  /** The previous accepted point, used by the secant predictor (null before the first step). */
  previous: { lambda: number; x: Vector } | null
  /** The predictor's guess on the last step. */
  predicted: Vector
  /** Newton corrector iterations and final ‖H‖ on the last step. */
  correctorSteps: number
  correctorResidual: number
  /** Whether the last step's corrector converged (λ advanced) or the step was halved. */
  accepted: boolean
  evaluations: number
  /** True once λ = 1 is reached. */
  converged: boolean
  /** `'step too small'` when Δλ falls below `minStep` (e.g. at a turning point of the solution path), or null. */
  failure: string | null
}

/** Options for `continuation`. */
export type ContinuationOptions = {
  /** First Δλ. Default 0.1. */
  dLambda?: number
  /** Largest Δλ. Default 0.25. */
  maxStep?: number
  /** Give up when Δλ falls below this. Default 1e-8. */
  minStep?: number
  /** Newton corrector iterations per step. Default 8. */
  correctorIterations?: number
  /** Corrector tolerance on ‖H(x, λ)‖. Default 1e-10. */
  tolerance?: number
}

/**
 * Natural-parameter continuation (Allgower & Georg, 2003, "Introduction to Numerical Continuation Methods", ch. 1–2):
 * follow the solution x(λ) of H(x, λ) = 0 from λ = 0, where x₀ solves it, to λ = 1. Each step predicts x at λ + Δλ by
 * secant extrapolation of the last two points, corrects with Newton's method on H(·, λ + Δλ), grows Δλ by 1.5 after a
 * fast success and halves it after a failed correction. `init` takes `{ x0 }`.
 */
export function continuation(
  H: Homotopy,
  options: ContinuationOptions = {},
): Algorithm<{ x0: VectorLike }, ContinuationState> {
  const { dLambda = 0.1, maxStep = 0.25, minStep = 1e-8, correctorIterations = 8, tolerance = 1e-10 } = options
  const name = 'continuation'
  return flagged<{ x0: VectorLike }, ContinuationState>({
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      return {
        t: 0,
        lambda: 0,
        x: vec(x),
        dLambda,
        previous: null,
        predicted: vec(x),
        correctorSteps: 0,
        correctorResidual: NaN,
        accepted: true,
        evaluations: 0,
        converged: false,
        failure: null,
      }
    },
    step: (s) => {
      const n = s.x.shape[0]
      const x = data(s.x)
      const step = Math.min(s.dLambda, 1 - s.lambda)
      const target = s.lambda + step
      // Secant predictor: extrapolate along the last accepted segment of the path.
      let guess = Float64Array.from(x)
      if (s.previous) {
        const span = s.lambda - s.previous.lambda
        if (span > 0) guess = axpy(step / span, sub(x, data(s.previous.x)), x)
      }
      let y = guess
      let residual = Infinity
      let iterations = 0
      let ok = false
      for (; iterations < correctorIterations;) {
        const out = H(vec(y), target)
        const r = toF64(out.value, name)
        residual = norm(r)
        if (residual <= tolerance) {
          ok = true
          break
        }
        const J = toMatrixF64(out.jacobian, name, n, n).data
        const p = linearSolve(J, scale(-1, r), n)
        iterations++
        if (!p) break
        y = axpy(1, p, y)
      }
      if (!ok && iterations === correctorIterations) {
        const r = toF64(H(vec(y), target).value, name)
        residual = norm(r)
        ok = residual <= tolerance
      }
      const evaluations = s.evaluations + iterations + 1
      if (!ok) {
        const halved = s.dLambda / 2
        return {
          ...s,
          t: s.t + 1,
          dLambda: halved,
          predicted: vec(guess),
          correctorSteps: iterations,
          correctorResidual: residual,
          accepted: false,
          evaluations,
          failure: halved < minStep ? 'step too small' : null,
        }
      }
      const lambda = target >= 1 - 1e-15 ? 1 : target
      return {
        t: s.t + 1,
        lambda,
        x: vec(y),
        dLambda: iterations <= 3 ? Math.min(1.5 * s.dLambda, maxStep) : s.dLambda,
        previous: { lambda: s.lambda, x: s.x },
        predicted: vec(guess),
        correctorSteps: iterations,
        correctorResidual: residual,
        accepted: true,
        evaluations,
        converged: lambda === 1,
        failure: null,
      }
    },
  })
}
