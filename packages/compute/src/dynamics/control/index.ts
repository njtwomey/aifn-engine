/**
 * `aifn-compute/dynamics/control`: state feedback for linear plants x′ = Ax + Bu: the linear-quadratic regulator (`lqr`,
 * `dlqr`, on the Riccati solvers of `aifn-compute/numerics/linalg`) and pole placement by Ackermann's formula (`ackermann`); linear model predictive control (`mpcController`, the condensed
 * QP on `aifn-compute/optim/programming`, and `recedingHorizon`, the loop as a step algorithm); and LQG output feedback (`lqg`:
 * the LQR gain on the steady-state Kalman estimate, the Kalman gain as the dual LQR; `lqgSimulation`).
 */

export { closedLoopPoles, dlqr, lqr, type LqrResult, type StateFeedbackPlant } from './lqr'
export { ackermann, type PolePlacement } from './ackermann'
export {
  mpcController,
  recedingHorizon,
  type MpcController,
  type MpcPlan,
  type MpcProblem,
  type RecedingHorizonOptions,
  type RecedingHorizonState,
} from './mpc'
export { lqg, lqgSimulation, type LqgDesign, type LqgPlant, type LqgSimulationState, type LqgWeights } from './lqg'
export { controlAlgorithms, controlFunctions } from './registry'
