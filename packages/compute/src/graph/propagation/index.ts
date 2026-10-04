/**
 * `aifn-compute/graph/propagation`: differentiable message passing over a graph's edges (gather → edge function → sum, mean or
 * max aggregation), the building block of graph neural networks; and label propagation, the harmonic solution (Zhu,
 * Ghahramani and Lafferty 2003) and label spreading (Zhou et al. 2004), in closed form and as steps.
 */

export {
  aggregateEdges,
  edgeSoftmax,
  messageEdges,
  propagate,
  type Aggregation,
  type EdgeFunction,
  type PropagateOptions,
} from './propagation'
export {
  graphAffinity,
  harmonicLabels,
  labelMatrix,
  labelPropagationSteps,
  labelSpreading,
  labelSpreadingSteps,
  normaliseScores,
  pointAffinity,
  randomWalkMatrix,
  spreadingMatrix,
  spreadingResolvent,
  type Affinity,
  type AffinityKernel,
  type LabelOptions,
  type LabelPropagationState,
  type LabelScores,
  type LabelStepsOptions,
  type PointAffinityOptions,
  type SpreadingOptions,
} from './labels'
export { propagationAlgorithms, propagationFunctions } from './registry'
