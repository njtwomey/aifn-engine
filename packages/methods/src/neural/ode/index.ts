/**
 * `aifn-methods/neural/ode`: neural ordinary differential equations on problems small enough to watch, from the
 * classifier and its augmented and second-order relatives to a continuous normalising flow and a latent ODE.
 *
 * - The family: `odeModel` builds a neural ODE (`'node'`), the augmented NODE (`'anode'`, zero padding so trajectories
 *   can pass), the second-order NODE (`'sonode'`, $\xvec'' = f(\xvec, \xvec')$) or the ResNet it discretises
 *   (`'resnet'`, $N$ Euler steps of size $1/N$ with untied weights), with a classifier or regressor readout;
 *   `parameterCount` sizes its parameters.
 * - Its data and training: `reflectionData` ($g(x) = -x$, which no 1-d neural ODE fits) and `discInRing` (which no flow
 *   of the plane separates); `odeRun` streams training with the trajectories, the field, the work per iteration and
 *   the adjoint's gradient against backprop's.
 * - Densities: `cnf`, a continuous normalising flow with an exact or Hutchinson trace, and `cnfRun` to train it by
 *   maximum likelihood with the RNODE regularisers.
 * - Irregular time series: `trajectories` (sines or spirals observed at random in a window), `latentOde` (a GRU
 *   encoder, a latent neural ODE and a linear decoder) and `latentOdeRun`, which maximises the ELBO and extrapolates.
 *
 * The differentiable solves, trace estimators and regularisers are compute's (`aifn-compute/dynamics/ode`, and
 * `OdeBlock` in `aifn-compute/nn/layers`): gradients go through the solver by backprop or by the adjoint, as the
 * `solver` options choose. The runs are generators of plain-data snapshots for a worker, deterministic in their seed
 * (wall times aside), and report an error in their last snapshot rather than throwing.
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
