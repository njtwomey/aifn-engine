/**
 * `aifn-compute/dynamics/ode`: solvers for initial-value problems $\xvec' = f(t, \xvec)$, $\xvec(t_0) = \xvec_0$, with
 * their stability, events and gradients.
 *
 * - One call: `solveIvp` runs any solver by name over $[t_0, t_1]$ and returns every accepted step as tensors, with the
 *   work done and the events found, as scipy's `solve_ivp`.
 * - Explicit, for non-stiff problems: `rungeKutta` steps with a fixed $h$ from a Butcher tableau (`EULER`, `HEUN`,
 *   `MIDPOINT`, `RK4`, by name through `TABLEAUX`, or one of your own); `dormandPrince` chooses its steps by error
 *   control (`DORMAND_PRINCE`, the 5(4) pair), and `dormandPrinceRows` runs it on many problems at once, each row
 *   with its own steps.
 * - Implicit, for stiff problems: `implicitEuler`, `implicitTrapezoid` and `bdf` (orders 1 to 3) with a fixed $h$,
 *   and `adaptiveBdf` (variable order 1 to 5 and variable step, as scipy's `'BDF'`).
 * - Hamiltonian systems: `symplectic` (symplectic Euler, leapfrog, velocity Verlet) keeps the energy error of a
 *   separable $H(\qvec, \pvec) = T(\pvec) + V(\qvec)$ bounded; `hamiltonianSystem` gives the same system to any
 *   other solver, with its energy.
 * - Linear systems: `linearFlow` evaluates $\xvec(t) = e^{\Amat t} \xvec_0$ exactly, at any times.
 * - Events: `withEvents` wraps any solver to locate the zeros of functions $g(t, \xvec)$ along the solution, and stops
 *   the run at a terminal one.
 * - Stability: `amplification` at a point $z = h\lambda$, `stabilityRegion` on a grid, and `boundaryLocus` for BDF.
 * - Gradients and neural ODEs: `odeAdjoint` (the adjoint method), `odeFlow` (states at many times, differentiated
 *   through the steps or by the adjoint), and `jacobianTrace`, `traceProbe` and `augmentedDynamics` for continuous
 *   normalising flows.
 * - `odeAlgorithms` and `odeFunctions` register the solvers and functions for generic views and workers.
 *
 * Every solver is an `Algorithm` started with `{ x0, t0 }` (`{ q0, p0, t0 }` for `symplectic`), to be run with `run`
 * or `trace`: each step is one accepted step, the state's `t` counts steps and `time` is the solution's time, and a
 * run given `tEnd` is `done` there. A solver that cannot continue reports it in its state (`diverged`, with the reason
 * in `failure`) rather than throwing; invalid options throw when the solver is built or started. The steps of
 * `rungeKutta` and `dormandPrince` are written with primitives, so a solution can be differentiated through them with
 * respect to $\xvec_0$ and the parameters $f$ closes over.
 */

export type { FixedStepOptions, InitialValue, JacobianOption, OdeState, Rhs } from './types'
export { EULER, HEUN, MIDPOINT, RK4, TABLEAUX, rungeKutta, type ButcherTableau } from './explicit'
export { DORMAND_PRINCE, dormandPrince, type AdaptiveOptions, type AdaptiveState, type StepAttempt } from './adaptive'
export { dormandPrinceRows, type RowsRhs, type RowsSolution, type RowsSolveOptions } from './rows'
export { adaptiveBdf, type AdaptiveBdfOptions, type AdaptiveBdfState } from './variable-bdf'
export { bdf, implicitEuler, implicitTrapezoid, type ImplicitOptions, type ImplicitState } from './implicit'
export {
  hamiltonianSystem,
  symplectic,
  type PhaseInitial,
  type SeparableHamiltonian,
  type SymplecticMethod,
  type SymplecticState,
} from './symplectic'
export { withEvents, type EventHit, type EventState, type OdeEvent } from './events'
export { linearFlow } from './linear'
export { amplification, boundaryLocus, stabilityRegion, type StabilityMethod, type StabilityRegion } from './stability'
export { solveIvp, type OdeMethod, type OdeSolution, type SolveIvpOptions } from './solve'
export { odeAdjoint, type OdeAdjointOptions, type OdeSolveInfo, type ParametricRhs } from './adjoint'
export { odeAlgorithms, odeFunctions } from './registry'
export {
  augmentedDynamics,
  jacobianTrace,
  odeFlow,
  traceProbe,
  type AugmentedDynamics,
  type AugmentedDynamicsOptions,
  type AugmentedParts,
  type JacobianTrace,
  type JacobianTraceOptions,
  type OdeFlowMethod,
  type OdeFlowOptions,
  type OdeGradient,
  type ProbeKind,
  type ShapedRhs,
  type TraceEstimator,
} from './neural'
