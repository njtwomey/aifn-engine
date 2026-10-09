/**
 * `solveIvp`: one call that runs any solver over an interval and returns the solution as tensors, as scipy's
 * `solve_ivp` (Virtanen et al., 2020, "SciPy 1.0", Nature Methods 17; Hairer, Nørsett & Wanner, 1993, §II).
 */

import { fromData, toFlat, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import type { Algorithm, Scalar, Size, StopReason, Trace, VectorLike } from 'aifn-compute/foundation/contracts'
import { dormandPrince } from './adaptive'
import { withEvents, type EventHit, type OdeEvent } from './events'
import { rungeKutta } from './explicit'
import { adaptiveBdf } from './variable-bdf'
import { bdf, implicitEuler, implicitTrapezoid } from './implicit'
import type { InitialValue, JacobianOption, OdeState, Rhs } from './types'

/** The solvers `solveIvp` can run by name. */
export type OdeMethod =
  | 'euler'
  | 'heun'
  | 'midpoint'
  | 'rk4'
  | 'dormand-prince'
  | 'bdf'
  | 'implicit-euler'
  | 'implicit-trapezoid'
  | 'bdf1'
  | 'bdf2'
  | 'bdf3'

/** Options for `solveIvp`. */
export type SolveIvpOptions = {
  /**
   * The solver to run (default `'dormand-prince'`): an explicit tableau (`'euler'`, `'heun'`, `'midpoint'`, `'rk4'`),
   * the adaptive `'dormand-prince'` or `'bdf'` (`adaptiveBdf`), or a fixed-step implicit method.
   */
  method?: OdeMethod
  /** The step size of a fixed-step method. Default $(t_1 - t_0)/100$. Ignored by the adaptive methods. */
  stepSize?: Scalar
  /** Relative tolerance of the adaptive methods (`'dormand-prince'`, `'bdf'`); default 1e-3. */
  rtol?: Scalar
  /**
   * Absolute tolerance of the adaptive methods; default 1e-6. One number here (`adaptiveBdf` itself also takes one
   * per component).
   */
  atol?: Scalar
  /** The Jacobian option of the implicit methods (`'bdf'`, `'implicit-euler'`, ...); ignored by the explicit ones. */
  jacobian?: JacobianOption
  /** Events to locate (a terminal event stops the run). */
  events?: readonly OdeEvent[]
  /** The most steps to take. Default 100 000. */
  maxSteps?: Size
}

/** The solution returned by `solveIvp`. */
export type OdeSolution = {
  /** The times of the accepted steps, $t_0$ first (length $m$). */
  time: Vector
  /** The states at those times ($m \times n$): row $k$ is the state at `time[k]`. */
  x: Matrix
  /** The step sizes (length $m$; 0 at $t_0$). */
  stepSize: Vector
  /** The local error estimates (length $m$; NaN where the method has none). */
  error: Vector
  /** Evaluations of $f$. */
  evaluations: Size
  /** Evaluations of the Jacobian (implicit methods). */
  jacobianEvaluations: Size
  /** Step attempts rejected by error control (adaptive methods). */
  rejected: Size
  /** The event crossings located, in time order (empty without `events`). */
  events: EventHit[]
  /** `done` on reaching $t_1$ (or a terminal event), `limit` after `maxSteps`, `diverged` on failure. */
  stopped: StopReason
  /** Why the solver stopped early, or null. */
  failure: string | null
  /** The full trace, for figures that want every state. */
  trace: Trace<OdeState>
}

/**
 * The named solver for $[t_0, t_1]$, built with the options that apply to it (`tEnd` is $t_1$).
 *
 * @param f The right-hand side.
 * @param t1 The end time $t_1$.
 * @param t0 The initial time $t_0$, used only for the default fixed step $(t_1 - t_0)/100$.
 * @param options The method and its settings, as `solveIvp` takes them; `events` and `maxSteps` are not read here.
 * @returns The solver, to be started with `{ x0, t0 }`.
 */
export function solverFor(f: Rhs, t1: Scalar, t0: Scalar, options: SolveIvpOptions): Algorithm<InitialValue, OdeState> {
  const { method = 'dormand-prince', stepSize: h = (t1 - t0) / 100, rtol, atol, jacobian } = options
  switch (method) {
    case 'dormand-prince':
      return dormandPrince(f, { tEnd: t1, rtol, atol })
    case 'bdf':
      return adaptiveBdf(f, { tEnd: t1, rtol, atol, jacobian })
    case 'implicit-euler':
      return implicitEuler(f, { stepSize: h, tEnd: t1, jacobian })
    case 'implicit-trapezoid':
      return implicitTrapezoid(f, { stepSize: h, tEnd: t1, jacobian })
    case 'bdf1':
    case 'bdf2':
    case 'bdf3':
      return bdf(f, Number(method[3]) as 1 | 2 | 3, { stepSize: h, tEnd: t1, jacobian })
    default:
      return rungeKutta(f, method, { stepSize: h, tEnd: t1 })
  }
}

/**
 * Solves $\xvec' = f(t, \xvec)$, $\xvec(t_0) = \xvec_0$ on $[t_0, t_1]$ with the named method (default Dormand–Prince
 * with rtol 1e-3 and atol 1e-6, as scipy's `solve_ivp`) and returns every accepted step, the step sizes, error
 * estimates, work counts and located events. Failure is reported, not thrown: a run that cannot continue stops with
 * `stopped` `'diverged'` and the reason in `failure`, and one that runs out of `maxSteps` stops with `'limit'`.
 *
 * @param f The right-hand side $f(t, \xvec)$.
 * @param interval The interval $[t_0, t_1]$ as a pair; $t_1 < t_0$ integrates backwards.
 * @param x0 The initial state $\xvec_0$ (length $n$).
 * @param options The method, its step size or tolerances, events to locate and the step budget.
 * @returns The solution at every accepted step, with the work done, the events found and the full trace.
 *
 * @example Exponential decay with the default method
 * // x′ = −x from x(0) = 1: x(2) = e^{−2}.
 * const sol = solveIvp((t, x) => neg(x), [0, 2], [1])
 * print('times =', sol.time)
 * print('x(2) =', toFlat(sol.x).at(-1))
 * print('e^{-2} =', Math.exp(-2))
 * print('stopped:', sol.stopped)
 *
 * @example A stiff problem: the implicit BDF takes far fewer steps
 * // x′ = −1000 (x − cos t): x stays close to cos t, but an explicit method must keep h below about 0.003.
 * const f = (t, x) => mul(-1000, sub(x, Math.cos(t)))
 * for (const method of ['dormand-prince', 'bdf']) {
 *   const sol = solveIvp(f, [0, 1], [0], { method })
 *   print(`${method}: ${sol.time.shape[0] - 1} steps, x(1) =`, toFlat(sol.x).at(-1))
 * }
 * print('exact x(1) =', (1e6 * Math.cos(1) + 1e3 * Math.sin(1) - 1e6 * Math.exp(-1000)) / (1e6 + 1))
 *
 * @example A terminal event stops the run
 * // A harmonic oscillator from (1, 0): q first reaches 0 at t = π/2.
 * const f = (t, x) => stack([get(x, 1), neg(get(x, 0))])
 * const events = [{ name: 'q = 0', g: (t, x) => get(x, 0), terminal: true }]
 * const sol = solveIvp(f, [0, 10], [1, 0], { rtol: 1e-8, atol: 1e-10, events })
 * print('event at t =', sol.events[0].time)
 * print('pi / 2 =', Math.PI / 2)
 * print('stopped at t =', toFlat(sol.time).at(-1))
 */
export function solveIvp(
  f: Rhs,
  [t0, t1]: readonly [Scalar, Scalar],
  x0: VectorLike,
  options: SolveIvpOptions = {},
): OdeSolution {
  const base = solverFor(f, t1, t0, options)
  const alg = (options.events?.length ? withEvents(base, f, options.events) : base) as Algorithm<InitialValue, OdeState>
  const tr = trace(alg, { x0, t0 }, options.maxSteps ?? 100_000, { stopOnNonFinite: false })
  const last = tr.final
  const n = tr.steps[0].x.shape[0]
  const X = new Float64Array(tr.steps.length * n)
  tr.steps.forEach((s, k) => X.set(toFlat(s.x), k * n))
  return {
    time: fromData(
      Float64Array.from(tr.steps, (s) => s.time),
      [tr.steps.length],
    ),
    x: fromData(X, [tr.steps.length, n]),
    stepSize: fromData(
      Float64Array.from(tr.steps, (s) => s.stepSize),
      [tr.steps.length],
    ),
    error: fromData(
      Float64Array.from(tr.steps, (s) => s.error),
      [tr.steps.length],
    ),
    evaluations: last.evaluations,
    jacobianEvaluations: last.jacobianEvaluations,
    rejected: last.rejected,
    events: (last as OdeState & { events?: EventHit[] }).events ?? [],
    stopped: tr.meta.stopped,
    failure: last.failure,
    trace: tr,
  }
}
