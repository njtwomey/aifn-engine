/**
 * Systems of nonlinear equations $F(\xvec) = \mathbf{0}$ with $F: \mathbb{R}^n \to \mathbb{R}^n$: Newton's method
 * (optionally damped by a backtracking line search on $\frac{1}{2}\|F\|_2^2$), Broyden's quasi-Newton method,
 * fixed-point iteration with convergence diagnostics, and natural-parameter continuation along a homotopy.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import { solveDense } from 'aifn-compute/numerics/linalg'
import { dense, type Matrix, SQRT_EPS, type Vector } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { flagged } from './status'

type F64 = dense.F64
const { allFinite, axpy, data, dot, mat, matVec, norm, scale, sub, toF64, toMatrixF64, vec } = dense

/** A system with its Jacobian: $F(\xvec)$ (length $n$) and $J(\xvec)$ ($n \times n$, $J_{ij} = \partial F_i / \partial x_j$). */
export type SystemWithJacobian = (x: Vector) => { value: VectorLike; jacobian: MatrixLike }

/** A system without derivatives: $\mathbf{F}(\mathbf{x})$ of length $n$. */
export type SystemFunction = (x: Vector) => VectorLike

/** Fields carried by every multidimensional root-finder state. */
export type SystemState = Status & {
  /** Iteration step counter ($0$ in the initial state). */
  t: Size
  /** Current iterate vector $\xvec_t$. */
  x: Vector
  /** Residual vector $F(\xvec_t)$. */
  residual: Vector
  /** Euclidean norm of the residual $\|F(\xvec_t)\|_2$. */
  residualNorm: number
  /** Step displacement vector $\xvec_t - \xvec_{t-1}$ (zeros at $t = 0$). */
  step: Vector
  /** Total number of system evaluations performed so far. */
  evaluations: number
  /** True once convergence criteria are satisfied. */
  converged: boolean
  /** Failure diagnosis string, or `null` if the solver is healthy. */
  failure: string | null
}

/** Stopping test tolerances for multidimensional systems. */
export type SystemTolerance = {
  /** Residual tolerance stopping when $\|F(\xvec)\|_2 \le \text{ftol}$ (default $10^{-12}$). */
  ftol?: number
  /** Step tolerance stopping when $\|\Delta\xvec\| \le \text{xtol} \cdot (1 + \|\xvec\|)$ (default $10^{-14}$). */
  xtol?: number
}

/**
 * Fill omitted system-solving tolerances with default thresholds.
 *
 * @param o User-specified tolerance options.
 * @returns Complete tolerances with `ftol` and `xtol`.
 */
const tolerances = (o: SystemTolerance) => ({ ftol: o.ftol ?? 1e-12, xtol: o.xtol ?? 1e-14 })

/**
 * Solve dense linear system $\mathbf{A} \mathbf{p} = \mathbf{b}$ using LU factorisation with partial pivoting.
 *
 * @param A Flattened $n \times n$ coefficient matrix in row-major order.
 * @param b Right-hand side vector of length $n$.
 * @param n Dimension of the square system.
 * @returns Solution vector $\mathbf{p}$, or `null` if $\mathbf{A}$ is singular or $\mathbf{b}$ contains non-finite entries.
 */
function linearSolve(A: F64, b: F64, n: number): F64 | null {
  if (!allFinite(b)) return null
  return solveDense(A, b, n).x as F64 | null
}

/**
 * Evaluate system residual and Jacobian at $x$, validating output dimensions.
 *
 * @param F Nonlinear system mapping vector $x$ to residual and Jacobian.
 * @param x Input coordinate vector as a Float64Array.
 * @param where Calling function name for descriptive error reporting.
 * @returns Object containing evaluated residual Float64Array `r` and flattened Jacobian `J`.
 */
function evaluateSystem(F: SystemWithJacobian, x: F64, where: string) {
  const out = F(vec(x))
  const r = toF64(out.value, where)
  if (r.length !== x.length)
    throw new ShapeError(where, `${where}: F returned ${r.length} values for ${x.length} unknowns`)
  return { r, J: toMatrixF64(out.jacobian, where, x.length, x.length).data }
}

/** The state of `newtonSystem`. */
export type NewtonSystemState = SystemState & {
  /** Jacobian matrix $J(\xvec)$ at the current iterate. */
  jacobian: Matrix
  /** Full undamped Newton step direction $\pvec = -J^{-1}F$. */
  newtonStep: Vector
  /** Damping fraction $\alpha \in (0, 1]$ applied to the Newton step. */
  damping: number
  /** Step length fractions evaluated by backtracking line search on the last step. */
  trials: number[]
}

/**
 * Newton's method for solving multivariate systems $F(\xvec) = \mathbf{0}$ with $F: \mathbb{R}^n \to \mathbb{R}^n$.
 *
 * Computes the search direction by solving $J(\xvec) \pvec = -F(\xvec)$ and updates $\xvec \leftarrow \xvec + \alpha \pvec$.
 * Without damping, $\alpha = 1$, yielding quadratic local convergence. When `damped` is enabled, $\alpha$ is
 * backtracked until the merit function $\frac{1}{2}\|F(\xvec)\|_2^2$ achieves sufficient decrease
 * (Nocedal & Wright, 2006, Algorithm 11.4).
 *
 * @param F System function returning residual vector and Jacobian matrix at $\xvec$.
 * @param options Convergence tolerances and damping toggle.
 * @returns A traceable `Algorithm` executing multidimensional Newton iterations.
 *
 * @example Solve 2D nonlinear system
 * const F = x => ({
 *   value: [x.data[0] + x.data[1] - 3, x.data[0] ** 2 + x.data[1] ** 2 - 5],
 *   jacobian: [[1, 1], [2 * x.data[0], 2 * x.data[1]]],
 * })
 * const alg = newtonSystem(F)
 * const state = run(alg, { x0: [2, 0] }, 20)
 * print('converged =', state.converged)
 * print('solution =', state.x)
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
  /** Approximate Jacobian matrix $B$ updated by rank-one corrections. */
  jacobian: Matrix
  /** Change in residual vector $\yvec = F(\xvec_t) - F(\xvec_{t-1})$. */
  residualChange: Vector
}

/**
 * Approximate the $n \times n$ Jacobian matrix using forward finite differences.
 *
 * Computes column $j$ as $(F(x + h e_j) - F(x)) / h$ with step $h = \sqrt{\varepsilon}\max(1, |x_j|)$.
 *
 * @param F Nonlinear vector function without analytic derivatives.
 * @param x Current evaluation point.
 * @param r Pre-computed residual vector $F(x)$.
 * @param where Caller context string for error attribution.
 * @returns Flattened $n \times n$ column-assembled Jacobian matrix.
 */
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
 * Broyden's "good" quasi-Newton method for solving systems $F(\xvec) = \mathbf{0}$ without analytic Jacobians.
 *
 * Solves $B \pvec = -F(\xvec)$, updates $\xvec \leftarrow \xvec + \pvec$, and performs a rank-one update:
 * $B \leftarrow B + (\yvec - B\svec)\svec^\top / (\svec^\top \svec)$,
 * where $\svec = \Delta\xvec$ and $\yvec = \Delta F$. This produces the minimal Frobenius-norm change
 * satisfying the secant condition $B\svec = \yvec$ (Broyden, 1965).
 * Converges superlinearly without evaluating Jacobians on each step.
 *
 * @param F Nonlinear vector function mapping $\xvec \in \mathbb{R}^n$ to $F(\xvec) \in \mathbb{R}^n$.
 * @param options Convergence tolerances and optional initial Jacobian approximation.
 * @returns A traceable `Algorithm` executing Broyden quasi-Newton steps.
 *
 * @example Solve nonlinear system without derivatives
 * const F = x => [x.data[0] + x.data[1] - 3, x.data[0] ** 2 + x.data[1] ** 2 - 5]
 * const alg = broyden(F)
 * const state = run(alg, { x0: [2, 0] }, 30)
 * print('converged =', state.converged)
 * print('solution =', state.x)
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

/** The state of `fixedPoint`. The `residual` is $g(\xvec) - \xvec$, zero at a fixed point. */
export type FixedPointState = SystemState & {
  /** Function value $g(\xvec)$ at the current iterate. */
  image: Vector
  /** Observed contraction factor $\|\Delta\xvec_t\| / \|\Delta\xvec_{t-1}\|$ (NaN before two steps). */
  contraction: number
  /** A posteriori error bound $\frac{L}{1 - L}\|\Delta\xvec_t\|$ based on observed contraction $L$. */
  errorBound: number
  /** Count of consecutive iterations where contraction factor was $\ge 1$. */
  expandingSteps: number
}

/**
 * Traceable multivariate fixed-point iteration $\xvec_{t+1} = (1 - \omega)\xvec_t + \omega g(\xvec_t)$.
 *
 * Converges linearly to the fixed point $g(\xvec^*) = \xvec^*$ for contractive mappings under the Banach
 * fixed-point theorem. Tracks the empirical contraction factor $L \approx \|\Delta\xvec_{t+1}\| / \|\Delta\xvec_t\|$
 * and provides an a posteriori error bound $\|\xvec_t - \xvec^*\| \le \frac{L}{1 - L}\|\Delta\xvec_t\|$.
 *
 * @param g Vector-valued mapping $g: \mathbb{R}^n \to \mathbb{R}^n$.
 * @param options Convergence tolerances, relaxation parameter $\omega$, and divergence patience.
 * @returns A traceable `Algorithm` stepping through fixed-point updates.
 *
 * @example Solve fixed point equation
 * const g = x => [Math.cos(x.data[1]), Math.sin(x.data[0])]
 * const alg = fixedPoint(g)
 * const state = run(alg, { x0: [0.5, 0.5] }, 50)
 * print('converged =', state.converged)
 * print('fixed point =', state.x)
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

/** A homotopy $H(\xvec, \lambda)$ with its Jacobian in $\xvec$, for $\lambda$ from 0 to 1. */
export type Homotopy = (x: Vector, lambda: number) => { value: VectorLike; jacobian: MatrixLike }

/**
 * Construct the standard Newton homotopy $H(\xvec, \lambda) = F(\xvec) - (1 - \lambda)F(\xvec_0)$.
 *
 * At $\lambda = 0$, $\xvec_0$ is an exact solution $H(\xvec_0, 0) = \mathbf{0}$. At $\lambda = 1$,
 * solutions of $H(\xvec, 1) = \mathbf{0}$ coincide with roots of $F(\xvec) = \mathbf{0}$.
 * The Jacobian with respect to $\xvec$ equals $J_F(\xvec)$ for all $\lambda \in [0, 1]$.
 *
 * @param F Nonlinear system with Jacobian evaluations.
 * @param x0 Starting point satisfying $H(\xvec_0, 0) = \mathbf{0}$.
 * @returns A `Homotopy` function mapping $(\xvec, \lambda)$ to residual and Jacobian.
 *
 * @example Form Newton homotopy
 * const F = x => ({
 *   value: [x.data[0] ** 2 - 2],
 *   jacobian: [[2 * x.data[0]]],
 * })
 * const H = newtonHomotopy(F, [1])
 * print('H at lambda=0 residual =', H(tensor([1]), 0).value)
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
  /** Continuation iteration step counter. */
  t: Size
  /** Current continuation parameter value $\lambda \in [0, 1]$. */
  lambda: number
  /** Solution vector $\xvec(\lambda)$ at the current continuation parameter. */
  x: Vector
  /** Parameter increment $\Delta\lambda$ to attempt on the subsequent step. */
  dLambda: number
  /** Previously accepted path point $(\lambda_{t-1}, \xvec_{t-1})$ for secant extrapolation. */
  previous: { lambda: number; x: Vector } | null
  /** Predicted solution vector before Newton correction. */
  predicted: Vector
  /** Number of Newton corrector iterations executed on the last step. */
  correctorSteps: number
  /** Final residual norm $\|H(\xvec, \lambda)\|_2$ after corrector iterations. */
  correctorResidual: number
  /** Whether the last corrector step converged and advanced $\lambda$. */
  accepted: boolean
  /** Total number of homotopy evaluations performed so far. */
  evaluations: number
  /** True once $\lambda = 1$ is successfully reached. */
  converged: boolean
  /** Failure diagnosis string, or `null` if continuation is healthy. */
  failure: string | null
}

/** Options configuring natural-parameter homotopy continuation. */
export type ContinuationOptions = {
  /** Initial continuation parameter step size $\Delta\lambda$ (default 0.1). */
  dLambda?: number
  /** Maximum allowable continuation step size (default 0.25). */
  maxStep?: number
  /** Minimum allowable continuation step size before terminating (default $10^{-8}$). */
  minStep?: number
  /** Maximum Newton corrector iterations per step (default 8). */
  correctorIterations?: number
  /** Residual tolerance on $\|H(\xvec, \lambda)\|_2$ for corrector acceptance (default $10^{-10}$). */
  tolerance?: number
}

/**
 * Natural-parameter homotopy continuation for solving $H(\xvec, \lambda) = \mathbf{0}$ from $\lambda = 0$ to $\lambda = 1$.
 *
 * Follows the solution path $\xvec(\lambda)$ starting from known solution $\xvec_0$ at $\lambda = 0$ up to the
 * target solution at $\lambda = 1$ (Allgower & Georg, 2003). At each step, a secant predictor estimates $\xvec$
 * at $\lambda + \Delta\lambda$, followed by a Newton corrector on $H(\cdot, \lambda + \Delta\lambda)$. The step size
 * $\Delta\lambda$ adapts dynamically based on corrector convergence speed.
 *
 * @param H Homotopy function mapping $(\xvec, \lambda)$ to residual and Jacobian.
 * @param options Continuation step size bounds, corrector iterations, and tolerance.
 * @returns A traceable `Algorithm` tracking the continuation solution curve.
 *
 * @example Natural-parameter continuation along Newton homotopy
 * const F = x => ({
 *   value: [x.data[0] ** 2 - 2],
 *   jacobian: [[2 * x.data[0]]],
 * })
 * const H = newtonHomotopy(F, [1])
 * const alg = continuation(H, { dLambda: 0.5 })
 * const state = run(alg, { x0: [1] }, 20)
 * print('converged =', state.converged)
 * print('root =', state.x)
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
