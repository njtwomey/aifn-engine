/**
 * `aifn-compute/dynamics/control`: feedback control of linear plants $\xvec' = \Amat\xvec + \Bmat\uvec$ (or
 * $\xvec_{k+1} = \Amat\xvec_k + \Bmat\uvec_k$).
 *
 * - Optimal state feedback $\uvec = -\Kmat\xvec$: `lqr` (continuous time) and `dlqr` (discrete time), on the Riccati
 *   solvers of `aifn-compute/numerics/linalg`; `closedLoopPoles` gives the eigenvalues of $\Amat - \Bmat\Kmat$ for
 *   any gain.
 * - Pole placement: `ackermann`, the unique gain of a single-input plant that puts the closed-loop poles where asked,
 *   with the conditioning of the controllability matrix and the poles obtained.
 * - Linear model predictive control: `mpcController` condenses the finite-horizon problem with input and state bounds
 *   into one QP on `aifn-compute/optim/programming`, solved from each state by `plan`; `recedingHorizon` runs the
 *   loop (plan, apply the first input, repeat) as a step algorithm.
 * - Output feedback: `lqg`, the LQR gain on the steady-state Kalman estimate, with the Kalman gain computed as the
 *   dual LQR; `lqgSimulation` runs the noisy discrete loop as a step algorithm, against full-information LQR if asked.
 *
 * Solvers report rather than throw where they can: an LQR whose Riccati solver stops short says so in `converged` and
 * `failure`, an uncontrollable pair gives `ackermann` a null gain, and an infeasible state bound is dropped from an
 * MPC plan with its status saying so. `controlFunctions` and `controlAlgorithms` register the module's functions and
 * algorithms with the notes they serve.
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
