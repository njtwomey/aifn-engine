/**
 * `aifn-methods/neural/ode`: the neural ODE family on small problems (`odeModel`: NODE, augmented, second-order, the
 * ResNet it discretises; `odeRun` with `reflectionData`), a continuous normalising flow (`cnf`, `cnfRun`) and a latent
 * ODE on irregular trajectories (`trajectories`, `latentOde`, `latentOdeRun`). The differentiable solves, trace
 * estimators and regularisers are compute (`aifn-compute/dynamics/ode`, `aifn-compute/nn/layers` `OdeBlock`).
 */

export { odeModel, type OdeModel, type OdeModelKind, type OdeModelOptions, type OdeModelParams } from './models'
export {
  discInRing,
  odeRun,
  parameterCount,
  reflectionData,
  type GradientComparison,
  type OdeCheckpoint,
  type OdeRun,
  type OdeRunData,
  type OdeRunOptions,
} from './run'
export { cnf, cnfRun, type Cnf, type CnfCheckpoint, type CnfOptions, type CnfRun, type CnfRunOptions } from './cnf'
export {
  latentOde,
  latentOdeRun,
  trajectories,
  type LatentOdeCheckpoint,
  type LatentOdeOptions,
  type LatentOdeParams,
  type LatentOdeRun,
  type LatentOdeRunOptions,
  type TrajectoryKind,
  type TrajectoryOptions,
  type TrajectorySet,
} from './latent'
export { neuralOdeFunctions } from './registry'
