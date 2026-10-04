/**
 * `aifn-compute/dynamics/ode`: initial-value problems x′ = f(t, x): explicit Runge–Kutta tableaux, adaptive Dormand–Prince,
 * implicit methods (implicit Euler, implicit trapezoid, fixed-step BDF, and the adaptive variable-order BDF), events, symplectic integrators, stability regions, `solve`, and
 * the flow of a linear system (`linearFlow`), and gradients of a solution by the adjoint method (`odeAdjoint`).
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
