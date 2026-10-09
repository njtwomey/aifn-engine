/**
 * `aifn-compute/graph/propagation`: differentiable message passing over a graph's edges, the building block of graph
 * neural networks; and label propagation, the harmonic solution (Zhu, Ghahramani and Lafferty 2003) and label
 * spreading (Zhou et al. 2004), in closed form and as steps.
 *
 * - Message passing: `propagate` gathers each edge's source features, applies an edge function and aggregates at the
 *   destination by sum, mean or max; `messageEdges` lists the directed edges it uses, `aggregateEdges` is the
 *   aggregation on its own and `edgeSoftmax` normalises edge scores over each node's incoming edges (attention).
 *   Built from differentiable primitives, so gradients reach the features and the edge function's parameters.
 * - Affinities: `graphAffinity` (a graph's weights, its connectivity or a heat kernel of its distances) and
 *   `pointAffinity` (a Gaussian kernel on points, optionally on a $k$-NN graph); `spreadingMatrix`
 *   ($\Dmat^{-1/2}\Wmat\Dmat^{-1/2}$), `randomWalkMatrix` ($\Dmat^{-1}\Wmat$) and `spreadingResolvent`
 *   ($(1 - \alpha)(\Imat - \alpha\Smat)^{-1}$, the map from labels to the limit of spreading).
 * - Labels in closed form: `harmonicLabels` (labelled nodes fixed) and `labelSpreading` (labels may be revised, by
 *   $\alpha$); `labelMatrix` builds the one-hot $\Ymat$ and `normaliseScores` the final scores and classes.
 * - Labels as steps, for `run` and `trace`: `labelPropagationSteps` and `labelSpreadingSteps`, stopping when the change
 *   falls below a tolerance, as scikit-learn.
 *
 * Labels are integers $0, \dots, C - 1$, with $-1$ for an unlabelled node. Affinities are a graph or a non-negative
 * $n \times n$ matrix. The label functions are not differentiable: they return plain tensors.
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
