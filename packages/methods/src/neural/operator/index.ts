/**
 * `aifn-methods/neural/operator`: operator learning, maps between functions learned from input–output pairs, with the
 * linear baselines that show what structure buys.
 *
 * - Linear baselines in closed form: `fitDenseOperator` (any linear map on grid values), `fitFourierMultiplier` with
 *   `applyFourierMultiplier` (translation-equivariant, one complex multiplier per mode, on any grid), and
 *   `fitPhysicsInformedMultiplier` with `applyModeMultiplier` (the heat equation's space-time multiplier from data and
 *   the equation's residual).
 * - Networks: `deepOnet` (branch and trunk networks, trained on `deepOnetLoss`) and `FourierNeuralOperator1d` (Fourier
 *   layers built on `SpectralConv1d` of `aifn-compute/nn/layers`).
 * - Message passing: `messageWeights` and `meshHeatRollout`, the heat equation on an irregular mesh as a graph
 *   operator would compute it.
 *
 * The training pairs are generated in `aifn-methods/data/synthetic` (`heatOperatorData`, `burgersOperatorData`,
 * `antiderivativeData`) from the periodic solutions of `aifn-methods/dynamics/pde`; errors are measured by
 * `relativeL2Error` of `aifn-compute/learning/metrics`.
 */

export {
  applyFourierMultiplier,
  applyModeMultiplier,
  fitDenseOperator,
  fitFourierMultiplier,
  fitPhysicsInformedMultiplier,
  type FourierMultiplier,
  type PhysicsInformedMultiplierOptions,
} from './linear'
export {
  deepOnet,
  deepOnetLoss,
  type AlignedBatch,
  type DeepOnet,
  type DeepOnetOptions,
  type DeepOnetParams,
} from './deeponet'
export { FourierNeuralOperator1d, type FnoOptions, type FnoParams } from './fno'
export { meshHeatRollout, messageWeights, type MeshHeatOptions, type MessageWeights } from './mesh'
export { operatorFunctions } from './registry'
