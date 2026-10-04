/**
 * `aifn-methods/learning/trees-and-ensembles`: decision trees and their ensembles. The shared layer holds CART
 * (growth, split search, cost-complexity pruning, importances) and the `decisionTree` and `regressionTree` estimators;
 * children: bagging, boosting.
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
