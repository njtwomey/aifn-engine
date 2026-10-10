/**
 * `aifn-methods/neural/full-batch`: a small MLP trained by full-batch L-BFGS against gradient descent, Adam and SGD
 * from the same initial weights.
 *
 * - The run: `fullBatchComparison`, a generator of snapshots with each optimiser's loss and gradient norm against
 *   iterations and full-data gradient evaluations, and L-BFGS's step lengths, line-search evaluations and curvature
 *   pairs. L-BFGS and gradient descent run through compute's `fullBatchTraining`, Adam and SGD through `trainingLoop`
 *   on minibatches. `COMPARISON_OPTIMISERS` lists the four in their fixed order.
 * - The model: `comparisonModel`, the MLP of a width, depth and activation with one output, and the map from a flat
 *   parameter vector $\thetavec$ to its parameters.
 *
 * Every optimiser minimises the same objective, the mean loss on the whole training set (binary cross-entropy or mean
 * squared error) plus an optional L2 penalty on the weights, and runs are deterministic from their seed.
 */

export {
  COMPARISON_OPTIMISERS,
  comparisonModel,
  fullBatchComparison,
  type ComparisonNetwork,
  type ComparisonOptimiser,
  type ComparisonOptions,
  type ComparisonRun,
  type ComparisonSnapshot,
  type ComparisonTask,
} from './comparison'
export { fullBatchFunctions } from './registry'
