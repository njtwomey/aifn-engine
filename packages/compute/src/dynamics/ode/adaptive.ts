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

/** The Dormand–Prince 5(4) tableau: `b` gives the fifth-order solution, `bHat` the embedded fourth-order one. */
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

/**
 * One attempted step of an adaptive solver: `stepSize` is the step tried, `error` its error norm (above 1, or not
 * finite, for a rejected step) and `accepted` whether it was taken.
 */
export type StepAttempt = { stepSize: Scalar; error: Scalar; accepted: boolean }

/** The state of `dormandPrince`. */
export interface AdaptiveState extends OdeState {
  /** The step size proposed for the next step. */
  nextStepSize: Scalar
  /** $f(t, \xvec)$ at the current point (reused as the first stage of the next step: first same as last). */
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
  /** The largest step size allowed (magnitude). Default $\infty$. */
  maxStepSize?: Scalar
  /**
   * The smallest step size (magnitude) before the run fails with `'step size underflow'`. Default
   * $10^{-12} \max(1, \lvert t \rvert)$ at the current time $t$.
   */
  minStepSize?: Scalar
}

const SAFETY = 0.9
const MIN_FACTOR = 0.2
const MAX_FACTOR = 10

/**
 * The weighted RMS norm of an error estimate, $\lVert \evec \rVert = \sqrt{\frac{1}{n} \sum_i (e_i / s_i)^2}$ with
 * scale $s_i = a + r \max(\lvert x_i \rvert, \lvert y_i \rvert)$, $a$ = `atol` and $r$ = `rtol` (Hairer, Nørsett &
 * Wanner, 1993, §II.4). A step is accepted when it is at most 1.
 *
 * @param e The local error estimate $\evec$ ($n$ values).
 * @param x The state at the start of the step ($n$ values).
 * @param y The proposed state at the end of the step ($n$ values).
 * @param rtol The relative tolerance $r$.
 * @param atol The absolute tolerance $a$.
 * @returns The norm; 0 for an empty state.
 */
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
 * `select_initial_step`: balance $h$ so that an Euler step's change and the estimated second derivative are both small
 * relative to the tolerance, for a method whose local error is $O(h^{p+1})$. The result is at most 100 times the trial
 * Euler step, the interval length and `maxStep`. Costs one evaluation. Internal to the adaptive solvers.
 *
 * @param f The right-hand side.
 * @param t0 The initial time $t_0$.
 * @param x0 The initial state $\xvec_0$ ($n$ values; not modified).
 * @param f0 The derivative $f(t_0, \xvec_0)$, already evaluated ($n$ values).
 * @param dir The direction of integration: $+1$ forwards, $-1$ backwards.
 * @param rtol The relative tolerance.
 * @param atol The absolute tolerance: one number, or one per component.
 * @param order The order $p$ of the method's local error estimate (4 for Dormand–Prince's embedded solution).
 * @param where The caller's name for error messages.
 * @param interval The length $\lvert t_\text{end} - t_0 \rvert$ of the interval; 0 returns a step of 0.
 * @param maxStep The largest step allowed (magnitude).
 * @returns The magnitude of the first step; the caller applies `dir`.
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
 * The Dormand–Prince 5(4) adaptive solver for $\xvec' = f(t, \xvec)$ on $[t_0, t_\text{end}]$ ($t_\text{end} < t_0$
 * integrates backwards). Each step estimates its local error as the difference between the fifth- and fourth-order
 * solutions, measured in the weighted RMS norm with tolerances `rtol` and `atol`; a step with error norm $e$ above 1
 * is rejected and retried smaller. The next step size is $h \min(10, \max(0.2, 0.9 e^{-1/5}))$ (not grown after a
 * rejection within the same step, and at most `maxStepSize`), and the solution advances with the fifth-order result
 * (local extrapolation). An attempt that gives a non-finite state is retried with $0.2h$. Six evaluations per attempt,
 * thanks to first-same-as-last, and one or two at `init`. Each `step` of the algorithm is one accepted step; its state
 * lists the attempts it took (`attempts`), so a trace records the step-size history. A step that would have to be
 * smaller than `minStepSize` ends the run with `diverged` and `failure` `'step size underflow'`. A non-finite `tEnd`
 * throws `DomainError`.
 *
 * @param f The right-hand side $f(t, \xvec)$.
 * @param options The end time, the tolerances and the bounds on the step size; `tEnd` is required.
 * @returns The solver, an `Algorithm` to run with `run(alg, { x0, t0 }, steps)`; it is `done` on reaching `tEnd`.
 *
 * @example Exponential decay to the default tolerances
 * // x′ = −x from x(0) = 1, so x(2) = e^{−2}.
 * const s = run(dormandPrince((t, x) => neg(x), { tEnd: 2 }), { x0: [1] }, 1000)
 * print('time =', s.time)
 * print('x =', s.x)
 * print('e^{-2} =', Math.exp(-2))
 * print('accepted steps =', s.t)
 * print('rejected attempts =', s.rejected)
 * print('evaluations of f =', s.evaluations)
 *
 * @example A tighter tolerance takes more, smaller steps
 * // A harmonic oscillator, q′ = p, p′ = −q, over one period: back to (1, 0).
 * const f = (t, x) => stack([get(x, 1), neg(get(x, 0))])
 * for (const rtol of [1e-3, 1e-6, 1e-9]) {
 *   const s = run(dormandPrince(f, { tEnd: 2 * Math.PI, rtol, atol: rtol * 1e-3 }), { x0: [1, 0] }, 10000)
 *   print(`rtol ${rtol}: ${s.t} steps, x =`, s.x)
 * }
 *
 * @example The step-size history of a trace
 * const tr = trace(dormandPrince((t, x) => neg(x), { tEnd: 5 }), { x0: [1] }, 1000)
 * // The first step is chosen by Hairer's rule; error control then settles near 0.88 until the last step lands on 5.
 * print('step sizes =', tr.steps.slice(1).map((s) => s.stepSize))
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
