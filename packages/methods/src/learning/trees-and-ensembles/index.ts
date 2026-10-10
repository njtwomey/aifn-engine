/**
 * `aifn-methods/learning/trees-and-ensembles`: decision trees by CART and their ensembles, as scikit-learn's `tree`
 * and `ensemble`.
 *
 * - The estimators: `decisionTree` (classification, Gini or entropy) and `regressionTree` (squared error), each with
 *   optional cost-complexity pruning (`pruneAlpha`) and its growth traced node by node.
 * - Growth: `growTree` grows a tree to completion; `treeGrowthSteps` is the same growth as a traceable algorithm, one
 *   node per step, depth-, breadth- or best-first; `splitSearch` and `splitCurve` expose the search at a node.
 * - Reading a tree: `predictTree` (class shares or means), `decideTree`, `applyTree` (the leaf reached),
 *   `decisionPath`, `nodeRegion` (a node's box), `nodePrediction`, `nodeLabel`, `treeSize` and `featureImportances`.
 * - Pruning: `costComplexityPath` (the weakest-link $\alpha$ sequence), `pruneTree` (prune at an $\alpha$) and
 *   `keptNodes` (read a pruned tree in the full tree's ids).
 * - Ensembles, in the child modules: `bagging` (`randomForest`) and `boosting` (`adaBoost`, `gradientBoosting`).
 *
 * A tree is plain data, an `aifn-compute/graph` binary tree whose nodes record their split, rows, statistics and the
 * search made there. Splits are $x_f \le t$ with $t$ halfway between consecutive distinct values, and nodes are
 * numbered in creation order, as scikit-learn numbers them.
 */

export {
  applyTree,
  costComplexityPath,
  decideTree,
  decisionPath,
  decisionTree,
  featureImportances,
  growTree,
  keptNodes,
  nodeLabel,
  nodePrediction,
  nodeRegion,
  predictTree,
  pruneTree,
  regressionTree,
  splitCurve,
  splitSearch,
  treeGrowthSteps,
  treeSize,
  type Criterion,
  type DecisionNode,
  type DecisionPath,
  type DecisionTest,
  type DescentOptions,
  type FeatureBest,
  type GrowthOrder,
  type LeafReason,
  type NodeEvaluation,
  type PendingNode,
  type DecisionTree,
  type DecisionTreeModel,
  type FeatureSplits,
  type RegressionTreeModel,
  type SplitData,
  type SplitSearch,
  type TreeGrowthState,
  type TreeParams,
  type TreeProblem,
  type WeightedData,
} from './tree'
export { randomForest } from './bagging'
export { adaBoost, gradientBoosting } from './boosting'
export { treesAlgorithms, treesFunctions } from './registry'
