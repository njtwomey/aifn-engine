/**
 * The registry of the `aifn-methods/learning/trees-and-ensembles` group's shared layer: tree growth as a traceable
 * algorithm and the tree functions (prediction, paths, regions, pruning, importances). Bagging and boosting register
 * in their own modules.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as tree from './tree'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/trees-and-ensembles')
const fn = definer<FunctionInfo>('function', 'learning/trees-and-ensembles')
const TREE = ['decision-tree']

algorithm(
  {
    key: 'treeGrowthSteps',
    name: 'Decision-tree growth',
    summary: 'Split one node per step by the best impurity decrease, in depth-, breadth- or best-first order.',
    problem: 'objective',
    state: { iterate: 'tree', flags: [] },
    notes: TREE,
    cite: ['breiman1984', 'quinlan1986'],
  },
  tree.treeGrowthSteps,
)
fn({ key: 'growTree', name: 'Grow a decision tree', role: 'fit', notes: TREE, cite: ['breiman1984'] }, tree.growTree)
fn({ key: 'splitSearch', name: 'Best split search', role: 'solver', notes: TREE }, tree.splitSearch)
fn({ key: 'splitCurve', name: 'Impurity decrease along a feature', role: 'estimator', notes: TREE }, tree.splitCurve)
fn({ key: 'predictTree', name: 'Tree prediction', role: 'inference', notes: TREE }, tree.predictTree)
fn({ key: 'decideTree', name: 'Tree decision', role: 'inference', notes: TREE }, tree.decideTree)
fn({ key: 'applyTree', name: 'Leaf of each point', role: 'inference', notes: TREE }, tree.applyTree)
fn({ key: 'decisionPath', name: 'Decision path', role: 'inference', notes: TREE }, tree.decisionPath)
fn({ key: 'nodeRegion', name: 'Region of a node', role: 'property', notes: TREE }, tree.nodeRegion)
fn({ key: 'nodePrediction', name: 'Prediction of a node', role: 'property', notes: TREE }, tree.nodePrediction)
fn({ key: 'nodeLabel', name: 'Label of a node', role: 'property', notes: TREE }, tree.nodeLabel)
fn({ key: 'treeSize', name: 'Tree size', role: 'property', notes: TREE }, tree.treeSize)
fn(
  {
    key: 'featureImportances',
    name: 'Impurity-based feature importances',
    role: 'estimator',
    notes: [...TREE, 'random-forest'],
  },
  tree.featureImportances,
)
fn(
  {
    key: 'costComplexityPath',
    name: 'Cost-complexity pruning path',
    role: 'estimator',
    notes: ['tree-pruning'],
    cite: ['breiman1984'],
  },
  tree.costComplexityPath,
)
fn(
  { key: 'pruneTree', name: 'Prune a tree', role: 'transform', notes: ['tree-pruning'], cite: ['breiman1984'] },
  tree.pruneTree,
)
fn({ key: 'keptNodes', name: 'Nodes kept by a prune', role: 'property', notes: ['tree-pruning'] }, tree.keptNodes)

/** The algorithms of the group, keyed by factory name. */
export const treesAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>('algorithm', tree) as Table<AlgorithmInfo>
/** The functions of the group, keyed by name. */
export const treesFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', tree) as Table<FunctionInfo>
