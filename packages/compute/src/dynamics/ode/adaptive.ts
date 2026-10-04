/**
 * Adaptive Runge–Kutta: the Dormand–Prince 5(4) pair with error control, the first-same-as-last property, Hairer's
 * starting step size and a record of every attempted step (Dormand & Prince, 1980, "A family of embedded Runge–Kutta
 * formulae", J. Comput. Appl. Math. 6; Hairer, Nørsett & Wanner, 1993, §II.4, as in scipy's `RK45`).
 */

import { dense, unwrap, type Tensor, type Value, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Scalar } from 'aifn-compute/foundation/contracts'
import { combine, evaluate, evaluateValue, initialValue, stages, type ButcherTableau } from './explicit'
import type { InitialValue, OdeState, Rhs } from './types'
import { DomainError } from 'aifn-compute/foundation/errors'

const { allFinite } = dense
type F64 = dense.F64

/** The Dormand–Prince 5(4) tableau: b gives the fifth-order solution, bHat the embedded fourth-order one. */
export const DORMAND_PRINCE: ButcherTableau = {
  name: 'dormand-prince',
  order: 5,
  c: [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1],
  a: [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
  ],
  b: [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
  bHat: [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40],
}

/** One attempted step of an adaptive solver. */
export type StepAttempt = { stepSize: Scalar; error: Scalar; accepted: boolean }

/** The state of `dormandPrince`. */
export interface AdaptiveState extends OdeState {
  /** The step size proposed for the next step. */
  nextStepSize: Scalar
  /** f(t, x) at the current point (reused as the first stage of the next step: first same as last). */
  derivative: Vector
  /** The attempts made to take the last step, in order; the last one was accepted. */
  attempts: StepAttempt[]
}

/** Options for `dormandPrince`. */
export type AdaptiveOptions = {
  /** The end time (required: it fixes the direction and bounds the last step). */
  tEnd: Scalar
  /** Relative tolerance. Default 1e-3 (scipy's default). */
  rtol?: Scalar
  /** Absolute tolerance. Default 1e-6. */
  atol?: Scalar
  /** The first step size; default chosen by Hairer's algorithm. */
  initialStepSize?: Scalar
  /** The largest step size allowed (magnitude). Default ∞. */
  maxStepSize?: Scalar
  /** The smallest step size (magnitude) before the run fails with `'step size underflow'`. Default 1e-12·|time|. */
  minStepSize?: Scalar
}

const SAFETY = 0.9
const MIN_FACTOR = 0.2
const MAX_FACTOR = 10

/** The weighted RMS norm ‖e‖ = √(mean((e_i / (atol + rtol·max(|x_i|, |y_i|)))²)). */
function errorNorm(e: F64, x: F64, y: F64, rtol: number, atol: number): number {
  let s = 0
  for (let i = 0; i < e.length; i++) {
    const sc = atol + rtol * Math.max(Math.abs(x[i]), Math.abs(y[i]))
    s += (e[i] / sc) ** 2
  }
  return Math.sqrt(s / Math.max(1, e.length))
}

/**
 * The starting step size of Hairer, Nørsett & Wanner (1993, §II.4, "Starting step size"), as scipy's
 * `select_initial_step`: balance h so that an Euler step's change and the estimated second derivative are both small
 * relative to the tolerance, for a method whose local error is O(h^{order+1}). `atol` is a scalar or one per
 * component. The result is at most the interval length and `maxStep`. Costs one evaluation. Internal to the adaptive
 * solvers.
 */
export function startingStep(
  f: Rhs,
  t0: number,
  x0: F64,
  f0: F64,
  dir: number,
  rtol: number,
  atol: number | ArrayLike<number>,
  order: number,
  where: string,
  interval = Infinity,
  maxStep = Infinity,
): number {
  if (interval === 0) return 0
  const at = (i: number) => (typeof atol === 'number' ? atol : atol[i])
  const scaled = (v: F64) => {
    let s = 0
    for (let i = 0; i < v.length; i++) s += (v[i] / (at(i) + Math.abs(x0[i]) * rtol)) ** 2
    return Math.sqrt(s / Math.max(1, v.length))
  }
  const d0 = scaled(x0)
  const d1 = scaled(f0)
  const h0 = Math.min(d0 < 1e-5 || d1 < 1e-5 ? 1e-6 : (0.01 * d0) / d1, interval)
  const x1 = Float64Array.from(x0, (v, i) => v + dir * h0 * f0[i])
  const f1 = evaluate(f, t0 + dir * h0, x1, where)
  const d2 = scaled(Float64Array.from(f1, (v, i) => v - f0[i])) / h0
  const h1 = Math.max(d1, d2) <= 1e-15 ? Math.max(1e-6, h0 * 1e-3) : (0.01 / Math.max(d1, d2)) ** (1 / (order + 1))
  return Math.min(100 * h0, h1, interval, maxStep)
}

/**
 * The Dormand–Prince 5(4) adaptive solver for x′ = f(t, x) on [t₀, tEnd] (tEnd < t₀ integrates backwards). Each
 * step estimates its local error as the difference between the fifth- and fourth-order solutions, measured in the
 * weighted RMS norm with tolerances `rtol` and `atol`; a step with error norm above 1 is rejected and retried smaller.
 * The next step size is h·min(10, max(0.2, 0.9·err^{−1/5})) and the solution advances with the fifth-order result
 * (local extrapolation). Six evaluations per attempt, thanks to first-same-as-last. Each `step` of the algorithm is
 * one accepted step; its state lists the attempts it took (`attempts`), so a trace records the step-size history.
 */
export function dormandPrince(f: Rhs, options: AdaptiveOptions): Algorithm<InitialValue, AdaptiveState> {
  const { tEnd, rtol = 1e-3, atol = 1e-6, maxStepSize: hMax = Infinity } = options
  if (!Number.isFinite(tEnd)) throw new DomainError('dormandPrince', 'dormandPrince: tEnd must be finite')
  const tab = DORMAND_PRINCE
  const name = tab.name
  const errW = tab.b.map((b, i) => b - tab.bHat![i])
  return {
    name,
    init: ({ x0, t0 = 0 }) => {
      // A traced x₀ stays the state (and f(t₀, x₀) its traced first stage), so `unrolled` differentiates through the
      // accepted steps; step sizes are chosen on primal values and are constants of the discrete solution.
      const base = initialValue(x0, t0, name)
      const x = dense.data(unwrap(base.x as Value) as Tensor)
      const dir = Math.sign(tEnd - t0) || 1
      const k0 = evaluateValue(f, t0, base.x, name)
      const f0 = dense.data(unwrap(k0) as Tensor)
      let evaluations = 1
      let h = options.initialStepSize
      if (h === undefined) {
        h = startingStep(f, t0, x, f0, dir, rtol, atol, 4, name, Math.abs(tEnd - t0), hMax)
        evaluations++
      }
      h = dir * Math.min(Math.abs(h), hMax)
      return { ...base, evaluations, nextStepSize: h, derivative: k0 as Vector, attempts: [] }
    },
    step: (s) => {
      const x = dense.data(unwrap(s.x as Value) as Tensor)
      const dir = Math.sign(s.nextStepSize) || 1
      const hMin = options.minStepSize ?? 1e-12 * Math.max(1, Math.abs(s.time))
      let h = s.nextStepSize
      let evaluations = 0
      const attempts: StepAttempt[] = []
      for (;;) {
        // Do not step past tEnd.
        if (dir * (s.time + h - tEnd) > 0) h = tEnd - s.time
        if (Math.abs(h) < hMin) {
          return {
            ...s,
            attempts,
            evaluations: s.evaluations + evaluations,
            diverged: true,
            failure: 'step size underflow',
          }
        }
        const k = stages(f, tab, s.time, s.x, h, name, s.derivative)
        evaluations += tab.b.length - 1
        const yValue = combine(s.x, h, tab.b, k)
        const y = dense.data(unwrap(yValue) as Tensor)
        const e = dense.data(unwrap(combine(0, h, errW, k)) as Tensor)
        const err = errorNorm(e, x, y, rtol, atol)
        if (!allFinite(y) || !Number.isFinite(err)) {
          attempts.push({ stepSize: h, error: err, accepted: false })
          h *= MIN_FACTOR
          continue
        }
        const factor = err === 0 ? MAX_FACTOR : Math.min(MAX_FACTOR, Math.max(MIN_FACTOR, SAFETY * err ** (-1 / 5)))
        if (err <= 1) {
          attempts.push({ stepSize: h, error: err, accepted: true })
          // After a rejection in this step, do not grow the step again at once (Hairer et al., §II.4).
          const grow = attempts.length > 1 ? Math.min(1, factor) : factor
          const hNext = dir * Math.min(Math.abs(h * grow), hMax)
          // The last stage is evaluated at (t + h, y): first same as last.
          const fy = k[k.length - 1] as Vector
          return {
            t: s.t + 1,
            time: s.time + h,
            x: yValue as Vector,
            stepSize: h,
            error: err,
            evaluations: s.evaluations + evaluations,
            jacobianEvaluations: 0,
            rejected: s.rejected + attempts.length - 1,
            diverged: false,
            failure: null,
            nextStepSize: hNext,
            derivative: fy,
            attempts,
          }
        }
        attempts.push({ stepSize: h, error: err, accepted: false })
        h *= factor
      }
    },
    done: (s) => s.failure !== null || Math.abs(tEnd - s.time) <= 1e-12 * Math.max(1, Math.abs(tEnd)),
  }
}
