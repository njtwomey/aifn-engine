/**
 * `aifn-methods/neural/full-batch`: a small MLP trained by full-batch L-BFGS (compute `fullBatchTraining`) against
 * gradient descent, Adam and SGD from the same initial weights (`fullBatchComparison`, a generator of snapshots with
 * the loss against iterations and full-data gradient evaluations, and L-BFGS's step lengths, line-search evaluations
 * and curvature pairs).
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
