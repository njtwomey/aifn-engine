/**
 * The variable-order, variable-step backward differentiation formulae (orders 1–5) for stiff systems, as scipy's
 * `solve_ivp(method='BDF')`: the quasi-constant step-size implementation of Shampine & Reichelt (1997, "The MATLAB ODE
 * Suite", SIAM J. Sci. Comput. 18), with the numerical differentiation formulae (NDF) of Klopfenstein and Shampine,
 * modified Newton iterations with a reused Jacobian and LU factor, and order selection from the error estimates of the
 * neighbouring orders (Byrne & Hindmarsh, 1975, "A polyalgorithm for the numerical solution of ordinary differential
 * equations", ACM TOMS 1; Hairer & Wanner, 1996, "Solving Ordinary Differential Equations II", §III.1).
 */

import { dense, fromData, type Matrix } from 'aifn-compute/foundation/tensor'
import { factorDense, solveFactored, type DenseFactor } from 'aifn-compute/numerics/linalg'
import type { Algorithm, Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { startingStep, type StepAttempt } from './adaptive'
import { evaluate, initialState } from './explicit'
import { jacobianOf } from './implicit'
import type { InitialValue, JacobianOption, OdeState, Rhs } from './types'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const { allFinite, toF64 } = dense
type F64 = dense.F64

const MAX_ORDER = 5
const NEWTON_MAX = 4
const MIN_FACTOR = 0.2
const MAX_FACTOR = 10

/** Options for `adaptiveBdf`. */
export type AdaptiveBdfOptions = {
  /** The end time (required: it fixes the direction and bounds the last step). */
  tEnd: Scalar
  /** Relative tolerance. Default 1e-3 (scipy's default). */
  rtol?: Scalar
  /** Absolute tolerance, one value or one per component. Default 1e-6. */
  atol?: Scalar | VectorLike
  /** The first step size; default chosen by Hairer's algorithm for order 1. */
  initialStepSize?: Scalar
  /** The largest step size allowed (magnitude). Default ∞. */
  maxStepSize?: Scalar
  /** How to compute ∂f/∂x. Default `'autodiff'`. */
  jacobian?: JacobianOption
  /**
   * `'ndf'` (default, as scipy and MATLAB's ode15s) uses the numerical differentiation formulae, BDF with the
   * correction κ·γ_k (y_{n+1} − y⁽⁰⁾_{n+1}) that lowers the error constant; `'bdf'` sets κ = 0 (the classical BDF).
   */
  variant?: 'ndf' | 'bdf'
}

/** The state of `adaptiveBdf`. */
export interface AdaptiveBdfState extends OdeState {
  /** The order (1–5) the next step will use. */
  order: Size
  /** The step size proposed for the next step (signed). */
  nextStepSize: Scalar
  /**
   * The backward differences of the interpolating polynomial, scaled by the step: row j is ∇ʲy_n·(h/h)ʲ for the current
   * equal-step grid ((MAX_ORDER + 3) × n; rows above order + 2 unused).
   */
  differences: Matrix
  /** Steps taken since the step size or order last changed. */
  equalSteps: Size
  /** The attempts made to take the last step, in order; the last one was accepted. */
  attempts: StepAttempt[]
  /** Newton iterations of the last accepted step. */
  newtonSteps: Size
  /** LU factorisations of I − c·J so far. */
  factorisations: Size
  /** The Jacobian ∂f/∂x currently used by Newton (row-major n × n; refreshed only when Newton fails). */
  jacobianValue: Matrix
  /** The LU factor of I − c·J in use, or null when it must be rebuilt (internal to the solver). */
  factor: DenseFactor | null
}

/** The RMS norm of v/scale. */
function rms(v: ArrayLike<number>, scale: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += (v[i] / scale[i]) ** 2
  return Math.sqrt(s / Math.max(1, v.length))
}

/** R(order, factor): the matrix that maps the differences on a step h to those on factor·h (Shampine & Reichelt). */
function changeMatrix(order: number, factor: number): F64 {
  const m = order + 1
  const M = new Float64Array(m * m)
  for (let j = 0; j < m; j++) M[j] = 1
  for (let i = 1; i < m; i++) for (let j = 1; j < m; j++) M[i * m + j] = (i - 1 - factor * j) / i
  for (let i = 1; i < m; i++) for (let j = 0; j < m; j++) M[i * m + j] *= M[(i - 1) * m + j]
  return M
}

/** Rescale the differences in place for a step size changed by `factor`: D[0..order] ← (R·U)ᵀ D[0..order]. */
function changeDifferences(D: F64, n: number, order: number, factor: number): void {
  const m = order + 1
  const R = changeMatrix(order, factor)
  const U = changeMatrix(order, 1)
  const RU = dense.matMul(R, U, m, m, m)
  const next = dense.matMul(dense.transpose(RU, m, m), D.subarray(0, m * n), m, m, n)
  D.set(next, 0)
}

/** The smallest step scipy allows at time t: ten units in the last place, towards the direction of integration. */
function minimumStep(t: number, dir: number): number {
  const buf = new Float64Array([t])
  const bits = new BigInt64Array(buf.buffer)
  if (t === 0) return 10 * Number.MIN_VALUE
  bits[0] += Math.sign(t) === dir ? 1n : -1n
  return 10 * Math.abs(buf[0] - t)
}

/**
 * The adaptive BDF solver for stiff x′ = f(t, x) on [t₀, tEnd], with the order chosen automatically from 1 to 5, as
 * scipy's `BDF`. Each step predicts y⁽⁰⁾ = Σ_{j≤k} D_j from the backward differences D, then solves the order-k
 * NDF equation (y − y⁽⁰⁾) − (h/α_k)(f(t+h, y)) + ψ = 0 by at most four modified Newton iterations with the matrix
 * I − (h/α_k)J, where α_k = (1 − κ_k)γ_k, γ_k = Σ_{j≤k} 1/j and ψ = Σ_{j≤k} γ_j D_j / α_k. J is refreshed only when
 * Newton fails to converge with the old one, and a step whose Newton iteration still fails is halved.
 *
 * The local error is (κ_kγ_k + 1/(k+1))·d with d = y − y⁽⁰⁾, in the weighted RMS norm with scale atol + rtol·|y|; a
 * step with error norm above 1 is retried with h·max(0.2, s·err^{−1/(k+1)}), where the safety s = 0.9·9/(8 + Newton
 * iterations). After k + 1 equal steps the solver compares the error estimates of orders k − 1, k and k + 1 and moves to
 * the order that allows the largest step, changing h by at most a factor of 10. `step` is one accepted step; `attempts`
 * lists the tries.
 */
export function adaptiveBdf(f: Rhs, options: AdaptiveBdfOptions): Algorithm<InitialValue, AdaptiveBdfState> {
  const {
    tEnd,
    rtol = 1e-3,
    atol: atolOption = 1e-6,
    maxStepSize: hMax = Infinity,
    jacobian = 'autodiff',
    variant = 'ndf',
  } = options
  const name = 'adaptiveBdf'
  if (!Number.isFinite(tEnd)) throw new DomainError(name, `${name}: tEnd must be finite`)
  if (!(rtol > 0)) throw new DomainError(name, `${name}: rtol must be positive`)
  // scipy's coefficients: κ (NDF), γ_k = Σ 1/j, α_k = (1 − κ_k)γ_k, error constants κ_kγ_k + 1/(k+1).
  const kappa = variant === 'ndf' ? [0, -0.185, -1 / 9, -0.0823, -0.0415, 0] : [0, 0, 0, 0, 0, 0]
  const gamma = [0]
  for (let k = 1; k <= MAX_ORDER; k++) gamma.push(gamma[k - 1] + 1 / k)
  const alpha = gamma.map((g, k) => (1 - kappa[k]) * g)
  const errorConst = gamma.map((g, k) => kappa[k] * g + 1 / (k + 1))
  const newtonTol = Math.max((10 * Number.EPSILON) / rtol, Math.min(0.03, Math.sqrt(rtol)))

  return {
    name,
    init: ({ x0, t0 = 0 }) => {
      const y = toF64(x0, name)
      const n = y.length
      const atol = typeof atolOption === 'number' ? atolOption : toF64(atolOption, name)
      if (typeof atol !== 'number' && atol.length !== n)
        throw new ShapeError(name, `${name}: atol has ${atol.length} values for a state of length ${n}`)
      const base = initialState(y, t0)
      const dir = Math.sign(tEnd - t0) || 1
      const f0 = evaluate(f, t0, y, name)
      let evaluations = 1
      let hAbs = options.initialStepSize
      if (hAbs === undefined) {
        hAbs = startingStep(f, t0, y, f0, dir, rtol, atol, 1, name, Math.abs(tEnd - t0), hMax)
        evaluations++
      }
      hAbs = Math.abs(hAbs)
      const { J, evaluations: extra } = jacobianOf(f, jacobian, t0, y, name)
      const D = new Float64Array((MAX_ORDER + 3) * n)
      D.set(y, 0)
      for (let i = 0; i < n; i++) D[n + i] = f0[i] * hAbs * dir
      return {
        ...base,
        evaluations: evaluations + extra,
        jacobianEvaluations: 1,
        order: 1,
        nextStepSize: dir * hAbs,
        differences: fromData(D, [MAX_ORDER + 3, n]),
        equalSteps: 0,
        attempts: [],
        newtonSteps: 0,
        factorisations: 0,
        jacobianValue: fromData(J, [n, n]),
        factor: null,
      }
    },
    step: (s) => {
      if (s.failure !== null) return s
      const n = s.x.shape[0]
      const atol = typeof atolOption === 'number' ? atolOption : toF64(atolOption, name)
      const at = (i: number) => (typeof atol === 'number' ? atol : atol[i])
      const t = s.time
      const dir = Math.sign(s.nextStepSize) || 1
      const D = Float64Array.from(dense.data(s.differences))
      let J = dense.data(s.jacobianValue)
      let LU = s.factor
      let order = s.order
      let equalSteps = s.equalSteps
      let evaluations = 0
      let jacobianEvaluations = 0
      let factorisations = 0
      const attempts: StepAttempt[] = []
      const minStep = minimumStep(t, dir)
      let hAbs = Math.abs(s.nextStepSize)
      if (hAbs > hMax) {
        changeDifferences(D, n, order, hMax / hAbs)
        hAbs = hMax
        equalSteps = 0
      } else if (hAbs < minStep) {
        changeDifferences(D, n, order, minStep / hAbs)
        hAbs = minStep
        equalSteps = 0
      }
      // The Jacobian is refreshed at most once per step, when Newton fails to converge with the old one.
      let currentJacobian = false
      const fail = (failure: string): AdaptiveBdfState => ({
        ...s,
        attempts,
        evaluations: s.evaluations + evaluations,
        jacobianEvaluations: s.jacobianEvaluations + jacobianEvaluations,
        factorisations: s.factorisations + factorisations,
        diverged: true,
        failure,
      })

      let tNew = t
      let yNew: F64 = new Float64Array(n)
      let d: F64 = new Float64Array(n)
      let errNorm = NaN
      let iterations = 0
      let scale = new Float64Array(n)
      let safety = 0.9
      for (;;) {
        if (hAbs < minStep) return fail('step size underflow')
        tNew = t + dir * hAbs
        if (dir * (tNew - tEnd) > 0) {
          tNew = tEnd
          changeDifferences(D, n, order, Math.abs(tNew - t) / hAbs)
          equalSteps = 0
          LU = null
        }
        const h = tNew - t
        hAbs = Math.abs(h)
        const predict = new Float64Array(n)
        for (let j = 0; j <= order; j++) for (let i = 0; i < n; i++) predict[i] += D[j * n + i]
        for (let i = 0; i < n; i++) scale[i] = at(i) + rtol * Math.abs(predict[i])
        const psi = new Float64Array(n)
        for (let j = 1; j <= order; j++) for (let i = 0; i < n; i++) psi[i] += (D[j * n + i] * gamma[j]) / alpha[order]
        const c = h / alpha[order]
        let converged = false
        for (;;) {
          if (LU === null) {
            const M = new Float64Array(n * n)
            for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i * n + j] = (i === j ? 1 : 0) - c * J[i * n + j]
            LU = factorDense(M, n)
            factorisations++
          }
          // Modified Newton on (y − y⁽⁰⁾) = c f(t_new, y) − ψ, stopping early when the contraction rate predicts that
          // the tolerance cannot be met within NEWTON_MAX iterations (scipy's `solve_bdf_system`).
          const y = Float64Array.from(predict)
          const dd = new Float64Array(n)
          let normOld = NaN
          converged = false
          let k = 0
          for (; k < NEWTON_MAX; k++) {
            const fy = evaluate(f, tNew, y, name)
            evaluations++
            if (!allFinite(fy)) break
            const rhs = Float64Array.from(fy, (v, i) => c * v - psi[i] - dd[i])
            const dy = solveFactored(LU, rhs)
            if (dy === null || !allFinite(dy)) break
            const dyNorm = rms(dy, scale)
            const rate = Number.isNaN(normOld) ? NaN : dyNorm / normOld
            if (!Number.isNaN(rate) && (rate >= 1 || (rate ** (NEWTON_MAX - k) / (1 - rate)) * dyNorm > newtonTol))
              break
            for (let i = 0; i < n; i++) {
              y[i] += dy[i]
              dd[i] += dy[i]
            }
            if (dyNorm === 0 || (!Number.isNaN(rate) && (rate / (1 - rate)) * dyNorm < newtonTol)) {
              converged = true
              break
            }
            normOld = dyNorm
          }
          iterations = Math.min(k + 1, NEWTON_MAX)
          yNew = y
          d = dd
          if (converged || currentJacobian) break
          const fresh = jacobianOf(f, jacobian, tNew, predict, name)
          J = fresh.J
          evaluations += fresh.evaluations
          jacobianEvaluations++
          LU = null
          currentJacobian = true
        }
        if (!converged) {
          attempts.push({ stepSize: dir * hAbs, error: NaN, accepted: false })
          hAbs *= 0.5
          changeDifferences(D, n, order, 0.5)
          equalSteps = 0
          LU = null
          continue
        }
        safety = (0.9 * (2 * NEWTON_MAX + 1)) / (2 * NEWTON_MAX + iterations)
        scale = Float64Array.from(yNew, (v, i) => at(i) + rtol * Math.abs(v))
        errNorm = rms(
          Float64Array.from(d, (v) => errorConst[order] * v),
          scale,
        )
        if (errNorm > 1) {
          attempts.push({ stepSize: dir * hAbs, error: errNorm, accepted: false })
          const factor = Math.max(MIN_FACTOR, safety * errNorm ** (-1 / (order + 1)))
          hAbs *= factor
          changeDifferences(D, n, order, factor)
          equalSteps = 0
          // Newton converged, so the factor is kept (scipy keeps it too, as a modified-Newton matrix).
          continue
        }
        attempts.push({ stepSize: dir * hAbs, error: errNorm, accepted: true })
        break
      }
      equalSteps++
      const stepTaken = tNew - t
      // Update the differences: with d = ∇^{k+1}y_{n+1}, ∇^{j+1}y_{n+1} = ∇^j y_{n+1} − ∇^j y_n.
      for (let i = 0; i < n; i++) {
        D[(order + 2) * n + i] = d[i] - D[(order + 1) * n + i]
        D[(order + 1) * n + i] = d[i]
      }
      for (let j = order; j >= 0; j--) for (let i = 0; i < n; i++) D[j * n + i] += D[(j + 1) * n + i]

      let nextAbs = hAbs
      if (equalSteps >= order + 1) {
        const errM =
          order > 1 ? rms(dense.scale(errorConst[order - 1], D.subarray(order * n, (order + 1) * n)), scale) : Infinity
        const errP =
          order < MAX_ORDER
            ? rms(dense.scale(errorConst[order + 1], D.subarray((order + 2) * n, (order + 3) * n)), scale)
            : Infinity
        const norms = [errM, errNorm, errP]
        const factors = norms.map((e, i) => e ** (-1 / (order + i)))
        let best = 0
        for (let i = 1; i < 3; i++) if (factors[i] > factors[best]) best = i
        order += best - 1
        const factor = Math.min(MAX_FACTOR, safety * factors[best])
        nextAbs *= factor
        changeDifferences(D, n, order, factor)
        equalSteps = 0
        LU = null
      }
      return {
        t: s.t + 1,
        time: tNew,
        x: fromData(yNew, [n]),
        stepSize: stepTaken,
        error: errNorm,
        evaluations: s.evaluations + evaluations,
        jacobianEvaluations: s.jacobianEvaluations + jacobianEvaluations,
        rejected: s.rejected + attempts.length - 1,
        diverged: false,
        failure: null,
        order,
        nextStepSize: dir * nextAbs,
        differences: fromData(D, [MAX_ORDER + 3, n]),
        equalSteps,
        attempts,
        newtonSteps: iterations,
        factorisations: s.factorisations + factorisations,
        jacobianValue: fromData(Float64Array.from(J), [n, n]),
        factor: LU,
      }
    },
    done: (s) => s.failure !== null || s.time === tEnd,
  }
}
