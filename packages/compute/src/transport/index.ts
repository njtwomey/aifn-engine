/**
 * `aifn-compute/transport`: optimal transport, as POT: Sinkhorn, exact transport through `aifn-compute/optim/programming`,
 * one-dimensional and sliced Wasserstein distances, and Gromov–Wasserstein.
 */

export {
  costMatrix,
  exactTransport,
  sinkhorn,
  sinkhornSteps,
  uniformWeights,
  type CostInput,
  type PointsInput,
  type SinkhornOptions,
  type SinkhornStart,
  type SinkhornState,
  type TransportPlan,
  type WeightsInput,
} from './discrete'
export {
  barycenter1d,
  monotonePlan,
  slicedWasserstein,
  wasserstein1d,
  type Barycenter1d,
  type MonotonePlan,
  type SlicedWasserstein,
} from './oneD'
export {
  gromovWasserstein,
  gromovWassersteinSteps,
  type GromovOptions,
  type GromovProblem,
  type GromovState,
} from './gromov'
export { transportAlgorithms, transportFunctions } from './registry'
