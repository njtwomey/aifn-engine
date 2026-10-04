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
  /** Default `'dormand-prince'`. */
  method?: OdeMethod
  /** The step size of a fixed-step method. Default (t₁ − t₀)/100. */
  stepSize?: Scalar
  /** Relative tolerance of the adaptive methods (`'dormand-prince'`, `'bdf'`). */
  rtol?: Scalar
  /** Absolute tolerance of the adaptive methods (one per component for `'bdf'`). */
  atol?: Scalar
  /** The Jacobian option of the implicit methods. */
  jacobian?: JacobianOption
  /** Events to locate (a terminal event stops the run). */
  events?: readonly OdeEvent[]
  /** The most steps to take. Default 100 000. */
  maxSteps?: Size
}

/** The solution returned by `solveIvp`. */
export type OdeSolution = {
  /** The times of the accepted steps, t₀ first (length m). */
  time: Vector
  /** The states at those times (m × n). */
  x: Matrix
  /** The step sizes (length m; 0 at t₀). */
  stepSize: Vector
  /** The local error estimates (length m; NaN where the method has none). */
  error: Vector
  /** Evaluations of f. */
  evaluations: Size
  /** Evaluations of the Jacobian (implicit methods). */
  jacobianEvaluations: Size
  /** Step attempts rejected by error control (adaptive methods). */
  rejected: Size
  events: EventHit[]
  /** `done` on reaching t₁ (or a terminal event), `limit` after `maxSteps`, `diverged` on failure. */
  stopped: StopReason
  /** Why the solver stopped early, or null. */
  failure: string | null
  /** The full trace, for figures that want every state. */
  trace: Trace<OdeState>
}

/** The named solver for [t₀, t₁]. */
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
 * Solves x′ = f(t, x), x(t₀) = x₀ on [t₀, t₁] with the named method (default Dormand–Prince with rtol 1e-3 and atol
 * 1e-6, as scipy's `solve_ivp`) and returns every accepted step, the step sizes, error estimates, work counts and
 * located events.
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
