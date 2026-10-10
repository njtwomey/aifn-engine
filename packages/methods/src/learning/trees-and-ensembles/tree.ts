/**
 * Decision trees by CART (Breiman, Friedman, Olshen and Stone, 1984, "Classification and Regression Trees"): greedy
 * binary splits $x_f \le t$ chosen to maximise the decrease in weighted impurity (Gini, entropy or squared error),
 * grown depth first by default (breadth-first and best-first on request), and minimal cost-complexity pruning (ibid.,
 * ch. 3; Hastie, Tibshirani and Friedman, 2009, §9.2.2).
 *
 * A split of node $t$ into children $L$ and $R$ decreases the weighted impurity by
 * $(w_t i(t) - w_L i(L) - w_R i(R)) / W$, with $w$ a node's total sample weight, $i$ its impurity and $W$ the total
 * training weight; these decreases sum, per feature, to the impurity importances. The conventions follow scikit-learn's
 * `DecisionTreeClassifier` and `DecisionTreeRegressor`: thresholds halfway between consecutive distinct values,
 * entropy in bits, nodes numbered in the order they are created (preorder, left first, when depth first), and the same
 * cost-complexity path. A tree is plain data: `{ nodes, root }`, each node `{ id, parent, children, ... }` with its
 * `SplitData`. Growth is a traceable algorithm (`treeGrowthSteps`), so a tree can be watched growing node by node.
 */

import type { Tree, TreeNode } from 'aifn-compute/graph'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type {
  AnyUnivariate,
  Decides,
  Estimator,
  Expects,
  FitOptions,
  Fitted,
  Predicts,
  Scores,
  Supervised,
  Trained,
} from 'aifn-compute/learning/estimators'
import type { Status } from 'aifn-compute/foundation/contracts'
import { type Stream, child, integers } from 'aifn-compute/foundation/random'
import { run, trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { classLabels, inputs, matrix, probabilityModel, targets, values } from '../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The impurity a split decreases: Gini or entropy (classification), squared error (regression). */
export type Criterion = 'gini' | 'entropy' | 'squared'

/** What a decision-tree node records, on top of the `aifn-compute/graph` tree structure. */
export interface SplitData {
  /** The node's depth (the root at 0). */
  depth: number
  /** The split feature ($-1$ at a leaf). */
  feature: number
  /** The split threshold $t$ (NaN at a leaf); the left child (slot 0) takes the rows with $x_f \le t$. */
  threshold: number
  /** The number of training rows reaching the node. */
  count: number
  /** The total sample weight of the training rows reaching the node. */
  weight: number
  /** The node's impurity (Gini, entropy in bits, or variance). */
  impurity: number
  /** Classification: the weighted class shares [K]. Regression: the weighted mean target, one entry. */
  value: number[]
  /** The weighted impurity decrease of the split, $(w_t i(t) - w_L i(L) - w_R i(R)) / W$ (0 at a leaf). */
  decrease: number
  /** The training rows that reached the node (indices into the data the tree was grown on). */
  rows: Int32Array
  /**
   * Classification: the weighted class totals, $K$ values. Regression: $[\sum_i w_i y_i, \sum_i w_i y_i^2]$ over the
   * node's rows.
   */
  counts: number[]
  /**
   * The split search the fitter made here: each searched feature's best threshold and decrease (empty when the node
   * was a leaf before any search). The split taken is the largest; `splitCurve` gives a feature's every candidate.
   */
  splits: FeatureBest[]
  /** Why the node is a leaf (null at a split). */
  stop: LeafReason | null
}

/**
 * A feature's best split at a node: its threshold and weighted impurity decrease (NaN and $-\infty$ when it has
 * none).
 */
export interface FeatureBest {
  /** The feature searched. */
  feature: number
  /** Its best threshold, or NaN when no threshold of it is valid. */
  threshold: number
  /** The weighted impurity decrease at that threshold, or $-\infty$. */
  decrease: number
}

/**
 * A node of a decision tree: an `aifn-compute/graph` `TreeNode` (id, parent, children, slot, label) with `SplitData`.
 */
export type DecisionNode = TreeNode<SplitData>

/**
 * A decision tree: an `aifn-compute/graph` binary `Tree` (arity 2; `children` are `[left, right]` at a split, `[]` at
 * a leaf) whose nodes carry `SplitData` and a label (the split $x_j \le t$ as TeX, or the leaf's prediction).
 */
export interface DecisionTree extends Tree<SplitData> {
  /** What the tree predicts: class shares, or a real value. */
  task: 'classification' | 'regression'
  /** The impurity its splits decrease. */
  criterion: Criterion
  /** The number of classes $K$ (1 for regression). */
  classes: number
  /** The number of features $d$ of its inputs. */
  features: number
  /** The growth parameters, resolved (unlimited values as Infinity), and so the order used. */
  params: Required<Omit<TreeParams, 'maxFeatures'>> & { maxFeatures: number }
}

/**
 * The node label: the split as TeX (`$x_{f} \le t$`, the threshold to three significant digits), or the leaf's class
 * (the first of the largest shares) or mean.
 *
 * @param node The node's split data.
 * @param task The tree's task, which decides how a leaf is labelled.
 * @returns The label, as growth stores it in each node's `label`.
 *
 * @example The labels of a small tree
 * const x = tensor([[1, 2], [2, 7], [3, 1], [6, 8], [7, 3], [8, 6]])
 * const tree = growTree(undefined, { x, y: tensor([0, 0, 0, 1, 1, 1]), task: 'classification' })
 * print('root:', nodeLabel(tree.nodes[0], 'classification'))
 * print('leaves:', tree.nodes.slice(1).map((node) => nodeLabel(node, 'classification')))
 */
export function nodeLabel(node: SplitData, task: 'classification' | 'regression'): string {
  const fmt = (v: number) => Number(v.toPrecision(3)).toString()
  if (node.feature >= 0) return `$x_{${node.feature}} \\le ${fmt(node.threshold)}$`
  if (task === 'regression') return `$${fmt(node.value[0])}$`
  let best = 0
  node.value.forEach((v, c) => {
    if (v > node.value[best]) best = c
  })
  return `class ${best}`
}

/**
 * A tree from nodes (parents before children), with its edge list and arity.
 *
 * @param nodes The nodes, indexed by id, each parent before its children; used as given (not copied).
 * @param p The prepared problem, whose task, criterion, sizes and resolved parameters the tree records.
 * @returns The decision tree, rooted at node 0.
 */
function asTree(nodes: DecisionNode[], p: Prepared): DecisionTree {
  return {
    kind: 'tree',
    nodes,
    root: 0,
    edges: nodes.map((node) => (node.parent === null ? null : {})),
    arity: 2,
    task: p.task,
    criterion: p.criterion,
    classes: p.K,
    features: p.d,
    params: {
      criterion: p.criterion,
      maxDepth: p.maxDepth,
      minSamplesSplit: p.minSamplesSplit,
      minSamplesLeaf: p.minSamplesLeaf,
      minImpurityDecrease: p.minImpurityDecrease,
      maxFeatures: p.maxFeatures,
      maxLeaves: p.maxLeaves,
      order: p.order,
    },
  }
}

/** Hyperparameters of tree growth. */
export interface TreeParams {
  /** The impurity: `'gini'` (default) or `'entropy'` for classification, `'squared'` (the only one) for regression. */
  criterion?: Criterion
  /** Maximum depth (root at 0; default unlimited). */
  maxDepth?: number
  /** A node with fewer rows is not split (default 2). */
  minSamplesSplit?: number
  /** Every child keeps at least this many rows (default 1). */
  minSamplesLeaf?: number
  /**
   * A split must decrease the weighted impurity (divided by the total training weight) by at least this (default 0).
   */
  minImpurityDecrease?: number
  /**
   * Features searched at each node: all (default), a count (1 or more), a fraction (below 1) of the $d$ features, or
   * $\lfloor\sqrt{d}\rfloor$ (`'sqrt'`) or $\lfloor\log_2 d\rfloor$ (`'log2'`), at least 1. Subsets are drawn from the
   * stream given to growth (random forests).
   */
  maxFeatures?: number | 'sqrt' | 'log2'
  /**
   * At most this many leaves (default unlimited). With a budget the expansion order matters; scikit-learn grows
   * best-first whenever `max_leaf_nodes` is set.
   */
  maxLeaves?: number
  /**
   * Which waiting node is expanded next (default `depth-first`): the last pushed (depth-first, preorder numbering),
   * the first pushed (breadth-first, level by level) or the one whose best split decreases impurity most (best-first).
   * Without a leaf budget every order grows the same tree, numbered differently, since each split depends only on its
   * own node's rows.
   */
  order?: GrowthOrder
}

/** The order in which tree growth expands waiting nodes. */
export type GrowthOrder = 'depth-first' | 'breadth-first' | 'best-first'

/** Why a node became a leaf. */
export type LeafReason = 'pure' | 'max-depth' | 'min-samples' | 'no-split' | 'min-decrease' | 'max-leaves' | 'pruned'

/**
 * The data a tree is grown on: inputs ($n \times d$), targets (labels $0, \dots, K - 1$, or real values) and optional
 * sample weights.
 */
export interface TreeProblem {
  /** The inputs, $n \times d$. */
  x: Tensor
  /** The targets, $n$ values: integer labels from 0 (classification) or real values (regression). */
  y: Tensor
  /** A non-negative weight per row (default all 1), such as a bootstrap sample's counts. */
  weights?: Tensor
  /** Whether to grow a classification or a regression tree. */
  task: 'classification' | 'regression'
  /** The number of classes (classification; default one more than the largest label, and at least 2). */
  classes?: number
  /** The growth hyperparameters (default: grown until pure, Gini or squared error). */
  params?: TreeParams
}

// ── Impurity ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Impurity of a node from its weighted class totals (classification) or weighted sums (regression): Gini
 * $1 - \sum_k p_k^2$, entropy $-\sum_k p_k \log_2 p_k$, or the weighted variance, with $p_k$ the class shares.
 *
 * @param criterion Which impurity.
 * @param stats The class totals $\sum_{i : y_i = k} w_i$ ($K$ values), or $[\sum_i w_i y_i, \sum_i w_i y_i^2]$.
 * @param w The node's total weight $\sum_i w_i$.
 * @returns The impurity, 0 for an empty (zero-weight) node.
 */
function impurityOf(criterion: Criterion, stats: Float64Array, w: number): number {
  if (w <= 0) return 0
  if (criterion === 'squared') {
    // stats = [Σw y, Σw y²]
    const m = stats[0] / w
    return Math.max(stats[1] / w - m * m, 0)
  }
  let s = 0
  if (criterion === 'gini') {
    for (const c of stats) s += (c / w) ** 2
    return 1 - s
  }
  for (const c of stats) if (c > 0) s -= (c / w) * Math.log2(c / w)
  return s
}

/** Candidate splits of one feature at one node: every threshold considered and its weighted impurity decrease. */
export interface FeatureSplits {
  /** The feature. */
  feature: number
  /**
   * The valid thresholds, ascending: midpoints between consecutive distinct values that leave each child enough rows.
   */
  thresholds: Tensor
  /**
   * The decrease at each threshold, $(w_t i(t) - w_L i(L) - w_R i(R)) / W$ with $W$ the total training weight (so the
   * decreases of the splits taken sum to the importance).
   */
  decreases: Tensor
}

/** The best split at a node, and every candidate that was considered (the split search exposed). */
export interface SplitSearch {
  /** The best split's feature, or $-1$ when no valid split exists. */
  feature: number
  /** The best split's threshold (NaN when there is none). */
  threshold: number
  /** The best split's weighted impurity decrease, divided by the total training weight (0 when there is none). */
  decrease: number
  /** The rows going left, in the node's row order (empty when there is no split). */
  left: Int32Array
  /** The rows going right, in the node's row order. */
  right: Int32Array
  /** Every searched feature's candidate thresholds and decreases, in the order searched. */
  candidates: FeatureSplits[]
}

/** A tree problem read for growth, with its parameters resolved (an unlimited value as `Infinity`). */
interface Prepared {
  /** The number of rows. */
  n: number
  /** The number of features. */
  d: number
  /** The inputs, row-major, $n d$ values. */
  x: Float64Array
  /** The targets, $n$ values. */
  y: Float64Array
  /** The sample weights, $n$ values. */
  w: Float64Array
  /** The total sample weight $W$. */
  totalWeight: number
  /** The task. */
  task: 'classification' | 'regression'
  /** The number of classes (1 for regression). */
  K: number
  /** The impurity. */
  criterion: Criterion
  /** The maximum depth. */
  maxDepth: number
  /** The fewest rows a node needs to be split. */
  minSamplesSplit: number
  /** The fewest rows each child must keep. */
  minSamplesLeaf: number
  /** The smallest weighted impurity decrease a split must make. */
  minImpurityDecrease: number
  /** The number of features searched at each node. */
  maxFeatures: number
  /** The leaf budget. */
  maxLeaves: number
  /** The order waiting nodes are expanded in. */
  order: GrowthOrder
}

/**
 * Read a tree problem and resolve its parameters. Throws `ShapeError` when `x` is not a matrix or `y` has another
 * number of rows, and `DomainError` when the criterion does not fit the task or a class label is not a whole number
 * $\ge 0$.
 *
 * @param problem The data, task and parameters.
 * @param where The caller's name, for error messages.
 * @returns The problem as flat arrays with every parameter resolved.
 */
function prepare(problem: TreeProblem, where: string): Prepared {
  if (problem.x.shape.length !== 2) throw new ShapeError(where, `${where}: expected x of shape [n, d]`)
  const [n, d] = problem.x.shape
  const x = values(problem.x)
  const y = values(problem.y)
  if (y.length !== n) throw new ShapeError(where, `${where}: ${n} rows but ${y.length} targets`)
  const w = problem.weights ? values(problem.weights) : new Float64Array(n).fill(1)
  if (w.length !== n) throw new ShapeError(where, `${where}: ${n} rows but ${w.length} weights`)
  let totalWeight = 0
  for (const u of w) {
    if (!Number.isFinite(u) || u < 0)
      throw new DomainError(where, `${where}: weights must be non-negative finite numbers`)
    totalWeight += u
  }
  const p = problem.params ?? {}
  const criterion = p.criterion ?? (problem.task === 'classification' ? 'gini' : 'squared')
  if ((criterion === 'squared') !== (problem.task === 'regression')) {
    throw new DomainError(where, `${where}: criterion ${criterion} does not fit a ${problem.task} tree`)
  }
  let K = 1
  if (problem.task === 'classification') {
    let top = 0
    for (const v of y) {
      if (!(Number.isInteger(v) && v >= 0)) throw new DomainError(where, `${where}: labels must be integers 0 … K−1`)
      top = Math.max(top, v + 1)
    }
    K = Math.max(problem.classes ?? 0, top, 2)
  }
  const mf = p.maxFeatures
  const maxFeatures =
    mf === undefined
      ? d
      : mf === 'sqrt'
        ? Math.max(1, Math.floor(Math.sqrt(d)))
        : mf === 'log2'
          ? Math.max(1, Math.floor(Math.log2(d)))
          : mf < 1
            ? Math.max(1, Math.floor(mf * d))
            : Math.min(d, Math.floor(mf))
  return {
    n,
    d,
    x,
    y,
    w,
    totalWeight,
    task: problem.task,
    K,
    criterion,
    maxDepth: p.maxDepth ?? Infinity,
    minSamplesSplit: p.minSamplesSplit ?? 2,
    minSamplesLeaf: p.minSamplesLeaf ?? 1,
    minImpurityDecrease: p.minImpurityDecrease ?? 0,
    maxFeatures,
    maxLeaves: p.maxLeaves ?? Infinity,
    order: p.order ?? 'depth-first',
  }
}

/**
 * Weighted statistics of a set of rows: class totals ($K$ values), or $[\sum_i w_i y_i, \sum_i w_i y_i^2]$.
 *
 * @param p The prepared problem.
 * @param rows The row indices.
 * @returns The statistics and the rows' total weight `w`.
 */
function statsOf(p: Prepared, rows: ArrayLike<number>): { stats: Float64Array; w: number } {
  const stats = new Float64Array(p.task === 'classification' ? p.K : 2)
  let w = 0
  for (let r = 0; r < rows.length; r++) {
    const i = rows[r]
    w += p.w[i]
    if (p.task === 'classification') stats[p.y[i]] += p.w[i]
    else {
      stats[0] += p.w[i] * p.y[i]
      stats[1] += p.w[i] * p.y[i] * p.y[i]
    }
  }
  return { stats, w }
}

/** scikit-learn skips thresholds between values closer than this. */
const FEATURE_THRESHOLD = 1e-7

/**
 * The best split of `rows` over `features`, with every candidate threshold and its decrease. Thresholds are midpoints
 * between consecutive distinct values (values closer than `FEATURE_THRESHOLD` count as equal) that leave each child at
 * least `minSamplesLeaf` rows; the first best (by feature order, then threshold) wins ties.
 *
 * @param p The prepared problem.
 * @param rows The rows at the node.
 * @param features The features to search, in order.
 * @returns The best split (feature $-1$ when none is valid), its children's rows and every candidate.
 */
function search(p: Prepared, rows: Int32Array, features: readonly number[]): SplitSearch {
  const { stats: total, w: W } = statsOf(p, rows)
  const parent = W * impurityOf(p.criterion, total, W)
  const S = total.length
  let best = { feature: -1, threshold: NaN, decrease: -Infinity, position: -1, order: new Int32Array(0) }
  const candidates: FeatureSplits[] = []
  const left = new Float64Array(S)
  const right = new Float64Array(S)
  for (const f of features) {
    const order = Int32Array.from(rows).sort((a, b) => p.x[a * p.d + f] - p.x[b * p.d + f] || a - b)
    left.fill(0)
    let wl = 0
    const thresholds: number[] = []
    const decreases: number[] = []
    for (let k = 0; k < order.length - 1; k++) {
      const i = order[k]
      wl += p.w[i]
      if (p.task === 'classification') left[p.y[i]] += p.w[i]
      else {
        left[0] += p.w[i] * p.y[i]
        left[1] += p.w[i] * p.y[i] * p.y[i]
      }
      const here = p.x[i * p.d + f]
      const next = p.x[order[k + 1] * p.d + f]
      if (next <= here + FEATURE_THRESHOLD) continue
      const nl = k + 1
      if (nl < p.minSamplesLeaf || order.length - nl < p.minSamplesLeaf) continue
      for (let s = 0; s < S; s++) right[s] = total[s] - left[s]
      const wr = W - wl
      const decrease = parent - wl * impurityOf(p.criterion, left, wl) - wr * impurityOf(p.criterion, right, wr)
      const threshold = here / 2 + next / 2
      thresholds.push(threshold)
      decreases.push(decrease / p.totalWeight)
      if (decrease > best.decrease) best = { feature: f, threshold, decrease, position: nl, order }
    }
    candidates.push({
      feature: f,
      thresholds: fromData(Float64Array.from(thresholds), [thresholds.length]),
      decreases: fromData(Float64Array.from(decreases), [decreases.length]),
    })
  }
  if (best.feature < 0) {
    return { feature: -1, threshold: NaN, decrease: 0, left: new Int32Array(0), right: new Int32Array(0), candidates }
  }
  // Children keep the node's row order.
  const f = best.feature
  const l: number[] = []
  const r: number[] = []
  for (const i of rows) (p.x[i * p.d + f] <= best.threshold ? l : r).push(i)
  return {
    feature: f,
    threshold: best.threshold,
    decrease: best.decrease / p.totalWeight,
    left: Int32Array.from(l),
    right: Int32Array.from(r),
    candidates,
  }
}

/**
 * The split search at one node: every candidate threshold of every feature with its weighted impurity decrease, and
 * the best one. This is exactly the search tree growth performs at a node (over every feature).
 *
 * @param problem The data, task and parameters (`minSamplesLeaf` limits the thresholds).
 * @param rows The rows at the node, as indices into the data (default: every row, the root).
 * @returns The best split, the rows going each way, and every feature's candidates.
 *
 * @example The first split of a toy dataset
 * const x = tensor([[1, 2], [2, 7], [3, 1], [6, 8], [7, 3], [8, 6]])
 * const found = splitSearch({ x, y: tensor([0, 0, 0, 1, 1, 1]), task: 'classification' })
 * print('feature', found.feature, 'at', found.threshold, ' decrease =', found.decrease)
 * print('left rows:', found.left, ' right rows:', found.right)
 * print('feature 1 thresholds:', found.candidates[1].thresholds)
 * print('feature 1 decreases:', found.candidates[1].decreases)
 */
export function splitSearch(problem: TreeProblem, rows?: ArrayLike<number>): SplitSearch {
  const p = prepare(problem, 'splitSearch')
  const idx = rows ? Int32Array.from(rows) : Int32Array.from({ length: p.n }, (_, i) => i)
  return search(
    p,
    idx,
    Array.from({ length: p.d }, (_, j) => j),
  )
}

// ── Growth ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** A node waiting to be created: its rows, depth and parent ($-1$ for the root) and which child it is. */
export interface PendingNode {
  /** The training rows that reach it (int32 indices). */
  rows: Tensor
  /** Its depth. */
  depth: number
  /** Its parent's id, or $-1$ for the root. */
  parent: number
  /** Which child of the parent it is: 0 left, 1 right. */
  side: 0 | 1
  /** Best-first only: the node evaluated when it was pushed, whose split decrease is its priority. */
  evaluation?: NodeEvaluation
}

/** What growth decides at a node: its statistics, why it is a leaf (null when it splits) and its split search. */
export interface NodeEvaluation {
  /** The node's weighted statistics: class totals, or $[\sum_i w_i y_i, \sum_i w_i y_i^2]$. */
  stats: Float64Array
  /** The node's total weight. */
  w: number
  /** The node's impurity. */
  impurity: number
  /** Why the node is a leaf, or null when it splits (before the leaf budget is applied). */
  leaf: LeafReason | null
  /** Null when the node is a leaf before any search (pure, depth, size). */
  search: SplitSearch | null
}

/** One state of tree growth: the nodes so far and the nodes still to create. */
export interface TreeGrowthState extends Status {
  /** The number of steps taken, one node created per step. */
  t: number
  /** The nodes created so far, as a (partial) tree: every created child is linked to its parent. */
  tree: DecisionTree
  /** The nodes waiting to be created, in push order. */
  pending: PendingNode[]
  /** The node created in this step ($-1$ at the start). */
  current: number
  /** The rows that reached that node (int32 indices). */
  rows: Tensor
  /** The split search made at that node (null at a leaf that was not searched, and at the start). */
  search: SplitSearch | null
  /** Why the node created in this step is a leaf (null when it splits, and at the start). */
  stop: LeafReason | null
  /** True once no node is waiting. */
  done: boolean
}

/**
 * A node's value: its weighted class shares, or its weighted mean target.
 *
 * @param p The prepared problem.
 * @param stats The node's weighted statistics.
 * @param w The node's total weight.
 * @returns $K$ class shares, or one mean (zeros for a zero-weight node).
 */
function nodeValue(p: Prepared, stats: Float64Array, w: number): number[] {
  if (p.task === 'regression') return [w > 0 ? stats[0] / w : 0]
  return Array.from(stats, (c) => (w > 0 ? c / w : 0))
}

/**
 * The order in which a node draws its features: the identity when every feature is searched, otherwise a random
 * permutation (Fisher–Yates) from the stream of the step that creates the node. The first `maxFeatures` are the node's
 * subset; the rest are drawn in turn only when the subset has no valid split.
 *
 * @param p The prepared problem (`maxFeatures` and `d`).
 * @param s The stream of the step; its child `'features'` draws the permutation.
 * @returns The features $0, \dots, d - 1$ in the order the node draws them.
 */
function featureOrder(p: Prepared, s: Stream): number[] {
  const all = Array.from({ length: p.d }, (_, j) => j)
  if (p.maxFeatures >= p.d) return all
  const sub = child(s, 'features')
  for (let a = 0; a < p.d - 1; a++) {
    const b = a + integers(sub, p.d - a)
    ;[all[a], all[b]] = [all[b], all[a]]
  }
  return all
}

/**
 * The split search at a node over a random feature subset: the first `maxFeatures` features of `featureOrder`, then,
 * while no valid split has been found, one more feature at a time, as scikit-learn does ("the search for a split does
 * not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect
 * more than max_features features"). A node becomes a leaf for want of a split only when no feature has one.
 *
 * @param p The prepared problem.
 * @param rows The rows at the node.
 * @param s The stream that draws the feature order.
 * @returns The best split over the features searched, with their candidates.
 */
function searchSubset(p: Prepared, rows: Int32Array, s: Stream): SplitSearch {
  const order = featureOrder(p, s)
  let k = Math.min(p.maxFeatures, p.d)
  const subset = (m: number) => order.slice(0, m).sort((a, b) => a - b)
  let found = search(p, rows, subset(k))
  while (found.feature < 0 && k < p.d) found = search(p, rows, subset(++k))
  return found
}

/**
 * Each searched feature's best threshold and decrease, from the candidates of a split search.
 *
 * @param found The split search.
 * @returns One `FeatureBest` per searched feature, in the order searched.
 */
function featureBests(found: SplitSearch): FeatureBest[] {
  return found.candidates.map((c) => {
    const t = c.thresholds.data as Float64Array
    const d = c.decreases.data as Float64Array
    let k = -1
    for (let i = 0; i < d.length; i++) if (k < 0 || d[i] > d[k]) k = i
    return { feature: c.feature, threshold: k < 0 ? NaN : t[k], decrease: k < 0 ? -Infinity : d[k] }
  })
}

/**
 * Statistics, leaf reason and split search of a node with these rows at this depth. A node is a leaf before any search
 * when it is pure (impurity at most machine epsilon), at `maxDepth`, or has too few rows to split; after the search,
 * when no split is valid or the best decreases the impurity by less than `minImpurityDecrease`.
 *
 * @param p The prepared problem.
 * @param rows The rows at the node.
 * @param depth The node's depth.
 * @param s The stream that draws the node's feature subset.
 * @returns The node's statistics, impurity, leaf reason (null when it splits) and search (null when not searched).
 */
function evaluate(p: Prepared, rows: Int32Array, depth: number, s: Stream): NodeEvaluation {
  const { stats, w } = statsOf(p, rows)
  const impurity = impurityOf(p.criterion, stats, w)
  const early: LeafReason | null =
    impurity <= Number.EPSILON
      ? 'pure'
      : depth >= p.maxDepth
        ? 'max-depth'
        : rows.length < p.minSamplesSplit || rows.length < 2 * p.minSamplesLeaf
          ? 'min-samples'
          : null
  if (early) return { stats, w, impurity, leaf: early, search: null }
  const search = searchSubset(p, rows, s)
  const leaf =
    search.feature < 0 ? 'no-split' : search.decrease + Number.EPSILON < p.minImpurityDecrease ? 'min-decrease' : null
  return { stats, w, impurity, leaf, search }
}

/**
 * The waiting node expanded next: the last pushed, the first pushed, or the one with the largest split decrease (a
 * node not yet evaluated first, a leaf last, the first pushed on ties).
 *
 * @param order The growth order.
 * @param pending The waiting nodes, in push order (not empty).
 * @returns The index in `pending` of the node to expand.
 */
function nextPending(order: GrowthOrder, pending: readonly PendingNode[]): number {
  if (order === 'depth-first') return pending.length - 1
  if (order === 'breadth-first') return 0
  const priority = (q: PendingNode) =>
    !q.evaluation ? Infinity : q.evaluation.leaf ? -Infinity : q.evaluation.search!.decrease
  let best = 0
  for (let i = 1; i < pending.length; i++) if (priority(pending[i]) > priority(pending[best])) best = i
  return best
}

/**
 * CART growth as a traceable algorithm: each step takes one waiting node (by `order`: depth-first pops the last
 * pushed, so nodes are numbered in preorder; breadth-first takes the first; best-first the one whose best split
 * decreases impurity most), decides whether it is a leaf (purity, depth, size, no split worth making, or the leaf
 * budget `maxLeaves`) and otherwise splits it and queues both children. Nodes are numbered in the order they are
 * created. When `maxFeatures` is below the number of features, the step that searches a node draws its feature subset
 * from the step's stream (best-first searches a child when it is queued, from the step's stream keyed by side), and
 * draws further features when the subset has no valid split (`searchSubset`). No start. A tree of $n$ rows has at most
 * $2n - 1$ nodes, so it is complete within $2n - 1$ steps.
 *
 * @param problem The data, task and parameters; read once, when the algorithm is made.
 * @returns The algorithm, whose state holds the partial tree, the waiting nodes and what the last step decided.
 *
 * @example Watch the root split and its children wait
 * const x = tensor([[1, 2], [2, 7], [3, 1], [6, 8], [7, 3], [8, 6]])
 * const alg = treeGrowthSteps({ x, y: tensor([0, 0, 0, 1, 1, 1]), task: 'classification' })
 * const first = run(alg, undefined, 1)
 * print('created node', first.current, first.tree.nodes[0].label, ' waiting:', first.pending.length)
 * const second = run(alg, undefined, 2)
 * print('created node', second.current, second.tree.nodes[1].label, ' leaf because', second.stop)
 * print('done after 4 steps:', run(alg, undefined, 4).done)
 */
export function treeGrowthSteps(problem: TreeProblem): Algorithm<void, TreeGrowthState> {
  const p = prepare(problem, 'treeGrowthSteps')
  return {
    name: 'cart-growth',
    init: () => {
      const active = Array.from({ length: p.n }, (_, i) => i).filter((i) => p.w[i] > 0)
      const rootRows = active.length > 0 ? active : Array.from({ length: p.n }, (_, i) => i)
      return {
        t: 0,
        tree: asTree([], p),
        pending: [
          {
            rows: fromData(Int32Array.from(rootRows), [rootRows.length]),
            depth: 0,
            parent: -1,
            side: 0,
          },
        ],
        current: -1,
        rows: fromData(new Int32Array(0), [0]),
        search: null,
        stop: null,
        done: false,
      }
    },
    step: (state, ctx) => {
      if (state.pending.length === 0) return { ...state, t: state.t + 1, done: true }
      const pending = state.pending.slice()
      const [top] = pending.splice(nextPending(p.order, pending), 1)
      const rows = Int32Array.from(top.rows.data as Int32Array)
      const id = state.tree.nodes.length
      const ev = top.evaluation ?? evaluate(p, rows, top.depth, ctx.stream)
      let leaf = ev.leaf
      // A split turns one leaf into two: it must fit the budget with the leaves made and the nodes still waiting.
      if (!leaf && Number.isFinite(p.maxLeaves)) {
        const made = state.tree.nodes.reduce((a, node) => a + (node.feature < 0 ? 1 : 0), 0)
        if (made + pending.length + 2 > p.maxLeaves) leaf = 'max-leaves'
      }
      const found = ev.search
      const node: DecisionNode = {
        id,
        parent: top.parent < 0 ? null : top.parent,
        children: [],
        ...(top.parent >= 0 && { slot: top.side }),
        label: '',
        depth: top.depth,
        feature: leaf ? -1 : found!.feature,
        threshold: leaf ? NaN : found!.threshold,
        count: rows.length,
        weight: ev.w,
        impurity: ev.impurity,
        value: nodeValue(p, ev.stats, ev.w),
        decrease: leaf ? 0 : found!.decrease,
        rows,
        counts: Array.from(ev.stats),
        splits: found ? featureBests(found) : [],
        stop: leaf,
      }
      node.label = nodeLabel(node, p.task)
      const nodes = state.tree.nodes.slice()
      nodes.push(node)
      if (top.parent >= 0) {
        const parent = { ...nodes[top.parent], children: nodes[top.parent].children.slice() }
        parent.children[top.side] = id
        nodes[top.parent] = parent
      }
      if (!leaf) {
        const kids = ([found!.left, found!.right] as const).map((r, side): PendingNode => ({
          rows: fromData(r, [r.length]),
          depth: top.depth + 1,
          parent: id,
          side: side as 0 | 1,
          ...(p.order === 'best-first' && {
            evaluation: evaluate(p, r, top.depth + 1, child(ctx.stream, side ? 'right' : 'left')),
          }),
        }))
        // Depth-first pops the left child next; the other orders take children left first.
        if (p.order === 'depth-first') pending.push(kids[1], kids[0])
        else pending.push(kids[0], kids[1])
      }
      return {
        t: state.t + 1,
        tree: asTree(nodes, p),
        pending,
        current: id,
        rows: fromData(rows, [rows.length]),
        search: found,
        stop: leaf,
        done: pending.length === 0,
      }
    },
    done: (state) => state.done,
  }
}

/**
 * Grow a tree to completion (see `treeGrowthSteps`); `s` draws the feature subsets when `maxFeatures` is below the
 * number of features (a run of `treeGrowthSteps` on that root stream).
 *
 * @param s The root stream of the run (default `stream(0)`); used only when `maxFeatures` is below the number of
 *   features.
 * @param problem The data, task and parameters.
 * @returns The grown tree.
 *
 * @example A tree on three classes of clusters
 * const x = tensor([[0, 0], [1, 1], [0, 5], [1, 6], [5, 0], [6, 1], [5, 5], [6, 6]])
 * const y = tensor([0, 0, 1, 1, 2, 2, 2, 2])
 * const tree = growTree(undefined, { x, y, task: 'classification' })
 * print('labels:', tree.nodes.map((node) => node.label))
 * print('size:', treeSize(tree))
 */
export function growTree(s: Stream | undefined, problem: TreeProblem): DecisionTree {
  const n = problem.x.shape[0]
  return run(treeGrowthSteps(problem), undefined, 2 * n + 1, { stream: s }).tree
}

// ── Prediction ───────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Options of prediction and paths. `within` marks the nodes present (default all): descent stops at a node whose next
 * child is absent, which then acts as a leaf. It reads a tree part-way through growth (the nodes created so far) or
 * pruned in place (the nodes `keptNodes` leaves), keeping the full tree's node ids.
 */
export interface DescentOptions {
  /** Whether the node with this id is present (default: every node is). */
  within?: (id: number) => boolean
}

/**
 * The node a point (row `i` of the flat matrix `v` with `d` columns) stops at, and the tests on the way.
 *
 * @param tree The tree.
 * @param v The points' values, row-major.
 * @param i The row of the point to descend.
 * @param d The number of columns of `v`.
 * @param within Whether a node is present; descent stops before an absent child (default: every node is).
 * @returns The node ids from the root, the test at each split passed, and the node it stops at.
 */
function descend(tree: DecisionTree, v: ArrayLike<number>, i: number, d: number, within?: (id: number) => boolean) {
  const tests: DecisionTest[] = []
  const nodes = [tree.root]
  let node = tree.nodes[tree.root]
  while (node.children.length) {
    const value = v[i * d + node.feature]
    const left = value <= node.threshold
    const next = node.children[left ? 0 : 1]
    if (next === undefined || (within && !within(next))) break
    tests.push({ node: node.id, feature: node.feature, threshold: node.threshold, value, left })
    node = tree.nodes[next]
    nodes.push(node.id)
  }
  return { nodes, tests, leaf: node.id }
}

/**
 * The leaf each row of `x` reaches, as scikit-learn's `apply`. Throws `ShapeError` unless `x` is $m \times d$ with the
 * tree's $d$ features.
 *
 * @param tree The tree.
 * @param x The points, $m \times d$.
 * @param options Which nodes are present (to read a partly grown or pruned tree in its full ids).
 * @returns The id of the node each row stops at, $m$ int32 values.
 *
 * @example Which leaf each point reaches
 * const x = tensor([[1, 2], [2, 7], [3, 1], [6, 8], [7, 3], [8, 6]])
 * const tree = growTree(undefined, { x, y: tensor([0, 0, 0, 1, 1, 1]), task: 'classification' })
 * print(applyTree(tree, tensor([[0, 0], [5, 0], [10, 10]])))
 */
export function applyTree(tree: DecisionTree, x: Tensor, options: DescentOptions = {}): Tensor {
  if (x.shape.length !== 2 || x.shape[1] !== tree.features) {
    throw new ShapeError('applyTree', `applyTree: expected x of shape [m, ${tree.features}]`)
  }
  const [m, d] = x.shape
  const v = values(x)
  const out = new Int32Array(m)
  for (let i = 0; i < m; i++) out[i] = descend(tree, v, i, d, options.within).leaf
  return fromData(out, [m])
}

/** One test on a decision path: the node, its rule $x_f \le t$, the point's value and the outcome. */
export interface DecisionTest {
  /** The id of the node making the test. */
  node: number
  /** The feature $f$ tested. */
  feature: number
  /** The threshold $t$. */
  threshold: number
  /** The point's value of the feature. */
  value: number
  /** True when the test holds and the point goes left. */
  left: boolean
}

/** A point's way through a tree: the node ids from the root to its leaf, and the test passed at each split. */
export interface DecisionPath {
  /** The node ids from the root to the leaf, both included. */
  nodes: number[]
  /** The test made at each split on the way. */
  tests: DecisionTest[]
  /** The id of the leaf reached. */
  leaf: number
}

/**
 * The decision path of one point from the root to the leaf it reaches, as scikit-learn's `decision_path` for one row.
 * Throws `ShapeError` unless the point has the tree's number of features.
 *
 * @param tree The tree.
 * @param point The point's $d$ feature values.
 * @param options Which nodes are present (to read a partly grown or pruned tree in its full ids).
 * @returns The nodes visited, the tests made and the leaf.
 *
 * @example The tests a point passes on its way down
 * const x = tensor([[0, 0], [1, 1], [0, 5], [1, 6], [5, 0], [6, 1], [5, 5], [6, 6]])
 * const tree = growTree(undefined, { x, y: tensor([0, 0, 1, 1, 2, 2, 2, 2]), task: 'classification' })
 * const path = decisionPath(tree, [0.5, 4])
 * print('nodes:', path.nodes, ' leaf:', path.leaf)
 * for (const t of path.tests) print(`x${t.feature} = ${t.value} <= ${t.threshold}:`, t.left)
 */
export function decisionPath(tree: DecisionTree, point: ArrayLike<number>, options: DescentOptions = {}): DecisionPath {
  if (point.length !== tree.features)
    throw new ShapeError('decisionPath', `decisionPath: expected ${tree.features} features`)
  return descend(tree, point, 0, tree.features, options.within)
}

/**
 * The leaf values of the rows of `x`: weighted class shares (classification) or weighted mean targets (regression).
 * Throws `ShapeError` unless `x` has the tree's number of features.
 *
 * @param tree The tree.
 * @param x The points, $m \times d$.
 * @param options Which nodes are present (to read a partly grown or pruned tree in its full ids).
 * @returns The class shares, $m \times K$, or the predictions, $m$ values.
 *
 * @example Class shares at the leaves of a depth-1 tree
 * const x = tensor([[1], [2], [3], [4], [5], [6]])
 * const y = tensor([0, 0, 1, 0, 1, 1])
 * const tree = growTree(undefined, { x, y, task: 'classification', params: { maxDepth: 1 } })
 * print('split:', tree.nodes[0].label)
 * print(predictTree(tree, tensor([[0], [10]])))
 */
export function predictTree(tree: DecisionTree, x: Tensor, options: DescentOptions = {}): Tensor {
  const leaves = applyTree(tree, x, options).data as Int32Array
  const m = leaves.length
  if (tree.task === 'regression')
    return fromData(
      Float64Array.from(leaves, (id) => tree.nodes[id].value[0]),
      [m],
    )
  const K = tree.classes
  const out = new Float64Array(m * K)
  for (let i = 0; i < m; i++) for (let c = 0; c < K; c++) out[i * K + c] = tree.nodes[leaves[i]].value[c]
  return fromData(out, [m, K])
}

/**
 * The class a node predicts: its weighted majority (the first on ties), or its mean for regression. It is defined at
 * every node, not only at leaves.
 *
 * @param tree The tree.
 * @param id The node's id.
 * @returns The class, or the mean target for regression.
 *
 * @example Every node's prediction
 * const x = tensor([[1, 2], [2, 7], [3, 1], [6, 8], [7, 3], [8, 6]])
 * const tree = growTree(undefined, { x, y: tensor([0, 0, 0, 1, 1, 1]), task: 'classification' })
 * print(tree.nodes.map((node) => nodePrediction(tree, node.id)))
 */
export function nodePrediction(tree: DecisionTree, id: number): number {
  const v = tree.nodes[id].value
  if (tree.task === 'regression') return v[0]
  let best = 0
  for (let c = 1; c < v.length; c++) if (v[c] > v[best]) best = c
  return best
}

/**
 * Decisions for the rows of `x`: the predicted class or, for regression, the prediction. Throws `ShapeError` unless
 * `x` has the tree's number of features.
 *
 * @param tree The tree.
 * @param x The points, $m \times d$.
 * @param options Which nodes are present (to read a partly grown or pruned tree in its full ids).
 * @returns The class of each row ($m$ int32 values), or for regression the predictions ($m$ values).
 *
 * @example Classify new points
 * const x = tensor([[1, 2], [2, 7], [3, 1], [6, 8], [7, 3], [8, 6]])
 * const tree = growTree(undefined, { x, y: tensor([0, 0, 0, 1, 1, 1]), task: 'classification' })
 * print(decideTree(tree, tensor([[0, 9], [4, 0], [5, 5]])))
 */
export function decideTree(tree: DecisionTree, x: Tensor, options: DescentOptions = {}): Tensor {
  const leaves = applyTree(tree, x, options).data as Int32Array
  if (tree.task === 'regression') return predictTree(tree, x, options)
  return fromData(
    Int32Array.from(leaves, (id) => nodePrediction(tree, id)),
    [leaves.length],
  )
}

/**
 * The axis-aligned box of the inputs that reach a node: the intersection of its ancestors' half-spaces ($x_f \le t$
 * going left, $x_f > t$ going right) within `bounds`. A row reaches the node exactly when it lies in the box, lower
 * ends open and upper ends closed.
 *
 * @param tree The tree.
 * @param id The node's id.
 * @param bounds The box to start from: `lower` and `upper` ends of each of the $d$ features (use $\pm\infty$ for
 *   none); not modified.
 * @returns The node's box, its `lower` and `upper` ends per feature.
 *
 * @example The box of each leaf of a two-level tree
 * const x = tensor([[0, 0], [1, 1], [0, 5], [1, 6], [5, 0], [6, 1], [5, 5], [6, 6]])
 * const tree = growTree(undefined, { x, y: tensor([0, 0, 1, 1, 2, 2, 2, 2]), task: 'classification' })
 * for (const node of tree.nodes.filter((node) => node.children.length === 0))
 *   print(node.id, node.label, nodeRegion(tree, node.id, { lower: [0, 0], upper: [10, 10] }))
 */
export function nodeRegion(
  tree: DecisionTree,
  id: number,
  bounds: { lower: ArrayLike<number>; upper: ArrayLike<number> },
): { lower: number[]; upper: number[] } {
  const lower = Array.from(bounds.lower)
  const upper = Array.from(bounds.upper)
  let node = tree.nodes[id]
  while (node.parent !== null) {
    const parent = tree.nodes[node.parent]
    const f = parent.feature
    if (parent.children[0] === node.id) upper[f] = Math.min(upper[f], parent.threshold)
    else lower[f] = Math.max(lower[f], parent.threshold)
    node = parent
  }
  return { lower, upper }
}

/**
 * Every candidate threshold of one feature at a node and its weighted impurity decrease: the fitter's own split search
 * (the function growth calls), re-run on the node's stored rows with the tree's parameters, since the tree keeps only
 * each feature's best. `data` is the data the tree was grown on.
 *
 * @param tree The tree.
 * @param id The node's id.
 * @param feature The feature whose thresholds are wanted (searched even if the node's random subset left it out).
 * @param data The data the tree was grown on: its inputs, targets and weights (the node's rows index into them).
 * @returns The feature's valid thresholds and the decrease at each.
 *
 * @example The impurity decrease along each feature at the root
 * const x = tensor([[1, 2], [2, 7], [3, 1], [6, 8], [7, 3], [8, 6]])
 * const data = { x, y: tensor([0, 0, 0, 1, 1, 1]) }
 * const tree = growTree(undefined, { ...data, task: 'classification' })
 * for (const f of [0, 1]) {
 *   const curve = splitCurve(tree, 0, f, data)
 *   print(`x${f}:`, curve.thresholds, curve.decreases)
 * }
 */
export function splitCurve(
  tree: DecisionTree,
  id: number,
  feature: number,
  data: { x: Tensor; y: Tensor; weights?: Tensor },
): FeatureSplits {
  const p = prepare({ ...data, task: tree.task, classes: tree.classes, params: tree.params }, 'splitCurve')
  return search(p, tree.nodes[id].rows, [feature]).candidates[0]
}

/**
 * Impurity-based feature importances (mean decrease in impurity, as scikit-learn's `feature_importances_`): each
 * feature's total weighted impurity decrease over the splits on it, normalised to sum to 1 (all 0 for a tree with no
 * split).
 *
 * @param tree The tree.
 * @returns The importance of each of the $d$ features.
 *
 * @example Only the informative feature is used
 * const x = tensor([[1, 2], [2, 7], [3, 1], [6, 8], [7, 3], [8, 6]])
 * const tree = growTree(undefined, { x, y: tensor([0, 0, 0, 1, 1, 1]), task: 'classification' })
 * print(featureImportances(tree))
 */
export function featureImportances(tree: DecisionTree): Tensor {
  const out = new Float64Array(tree.features)
  for (const node of tree.nodes) if (node.children.length) out[node.feature] += node.decrease
  let s = 0
  for (const v of out) s += v
  if (s > 0) for (let j = 0; j < out.length; j++) out[j] /= s
  return fromData(out, [tree.features])
}

/**
 * The number of leaves, depth and number of nodes of a tree.
 *
 * @param tree The tree.
 * @returns The `leaves`, the `depth` (largest node depth, the root at 0) and the number of `nodes`.
 *
 * @example The size of a fully grown tree
 * const x = tensor([[0, 0], [1, 1], [0, 5], [1, 6], [5, 0], [6, 1], [5, 5], [6, 6]])
 * print(treeSize(growTree(undefined, { x, y: tensor([0, 0, 1, 1, 2, 2, 2, 2]), task: 'classification' })))
 */
export function treeSize(tree: DecisionTree): { leaves: number; depth: number; nodes: number } {
  let leaves = 0
  let depth = 0
  for (const node of tree.nodes) {
    if (!node.children.length) leaves++
    depth = Math.max(depth, node.depth)
  }
  return { leaves, depth, nodes: tree.nodes.length }
}

// ── Cost-complexity pruning ──────────────────────────────────────────────────────────────────────────────────────

/**
 * The weakest link of a (partly pruned) tree: the internal node with the smallest
 * $g(t) = (R(t) - R(T_t)) / (\lvert \operatorname{leaves}(T_t) \rvert - 1)$, with $R(t) = (w_t / W) i(t)$ and $T_t$
 * the branch below $t$ (the lowest id on ties).
 *
 * @param tree The full tree.
 * @param cut The nodes already collapsed into leaves.
 * @returns The weakest node (`best`, $-1$ when only the root is left) and its $g$ (`alpha`), with the pruned tree's
 *   total leaf impurity and its number of leaves.
 */
function weakestLinks(tree: DecisionTree, cut: Set<number>) {
  const total = tree.nodes[tree.root].weight
  const R = (id: number) => (tree.nodes[id].weight / total) * tree.nodes[id].impurity
  const branch = new Map<number, { r: number; leaves: number }>()
  const visit = (id: number): { r: number; leaves: number } => {
    const node = tree.nodes[id]
    if (!node.children.length || cut.has(id)) return { r: R(id), leaves: 1 }
    const a = visit(node.children[0])
    const b = visit(node.children[1])
    const out = { r: a.r + b.r, leaves: a.leaves + b.leaves }
    branch.set(id, out)
    return out
  }
  const root = visit(tree.root)
  let best = -1
  let bestAlpha = Infinity
  for (const [id, b] of [...branch.entries()].sort((u, v) => u[0] - v[0])) {
    const g = (R(id) - b.r) / (b.leaves - 1)
    if (g < bestAlpha) {
      bestAlpha = g
      best = id
    }
  }
  return { best, alpha: bestAlpha, impurity: root.r, leaves: root.leaves }
}

/**
 * The minimal cost-complexity pruning path (Breiman et al., 1984, ch. 3): the effective $\alpha$ at which each
 * successive weakest link is pruned, and the total leaf impurity $\sum_t (w_t / W) i(t)$ and leaf count of the pruned
 * tree at each $\alpha$ (starting with $\alpha = 0$ and the full tree, ending with the root alone), as scikit-learn's
 * `cost_complexity_pruning_path`. `cuts[k]` is the node (an id of `tree`) whose subtree collapses into a leaf at step
 * $k + 1$, at $\alpha$ = `alphas[k + 1]`.
 *
 * @param tree The tree to prune; not modified.
 * @returns The `alphas`, the `impurities` and `leaves` of the tree pruned at each, and the `cuts` made.
 *
 * @example Pruning a tree that fits one noisy label
 * const x = tensor([[1], [2], [3], [4], [5], [6], [7], [8]])
 * const tree = growTree(undefined, { x, y: tensor([0, 0, 0, 1, 0, 1, 1, 1]), task: 'classification' })
 * const path = costComplexityPath(tree)
 * print('alphas:', path.alphas)
 * print('impurities:', path.impurities)
 * print('leaves:', path.leaves, ' cuts:', path.cuts)
 */
export function costComplexityPath(tree: DecisionTree): {
  alphas: Tensor
  impurities: Tensor
  leaves: Tensor
  cuts: Tensor
} {
  const cut = new Set<number>()
  let w = weakestLinks(tree, cut)
  const alphas = [0]
  const impurities = [w.impurity]
  const leaves = [w.leaves]
  const cuts: number[] = []
  while (w.best >= 0) {
    cut.add(w.best)
    cuts.push(w.best)
    const alpha = w.alpha
    w = weakestLinks(tree, cut)
    alphas.push(alpha)
    impurities.push(w.impurity)
    leaves.push(w.leaves)
  }
  return {
    alphas: fromData(Float64Array.from(alphas), [alphas.length]),
    impurities: fromData(Float64Array.from(impurities), [impurities.length]),
    leaves: fromData(Int32Array.from(leaves), [leaves.length]),
    cuts: fromData(Int32Array.from(cuts), [cuts.length]),
  }
}

/**
 * The nodes a tree keeps after collapsing the subtrees at `cuts` into leaves, as a mask over its node ids (1 kept):
 * every node not strictly below a cut. With `DescentOptions.within` it reads the pruned tree in the full tree's ids.
 *
 * @param tree The full tree.
 * @param cuts The ids of the nodes whose subtrees collapse, such as a prefix of `costComplexityPath`'s `cuts`.
 * @returns One entry per node id: 1 when the node is kept, 0 when it is below a cut.
 *
 * @example Predict with the first cut of the pruning path, in the full tree's ids
 * const x = tensor([[1], [2], [3], [4], [5], [6], [7], [8]])
 * const tree = growTree(undefined, { x, y: tensor([0, 0, 0, 1, 0, 1, 1, 1]), task: 'classification' })
 * const kept = keptNodes(tree, toFlat(costComplexityPath(tree).cuts).slice(0, 1))
 * print('kept:', kept)
 * print('leaves:', applyTree(tree, x, { within: (id) => kept[id] === 1 }))
 */
export function keptNodes(tree: DecisionTree, cuts: Iterable<number>): Uint8Array {
  const out = new Uint8Array(tree.nodes.length)
  const cut = new Set(cuts)
  const visit = (id: number) => {
    out[id] = 1
    if (!cut.has(id)) for (const c of tree.nodes[id].children) visit(c)
  }
  visit(tree.root)
  return out
}

/**
 * The tree pruned at complexity $\alpha$: weakest links are cut while their $g(t) \le \alpha$ (see
 * `costComplexityPath`). Nodes are renumbered in preorder, so the result is a tree of the same shape as one grown to
 * that size; a collapsed node's `stop` is `'pruned'`.
 *
 * @param tree The grown tree; not modified.
 * @param alpha The complexity $\alpha \ge 0$: the price of each leaf, in total leaf impurity.
 * @returns The pruned tree, a new tree with its own node ids.
 *
 * @example Pruning removes the splits around a noisy label
 * const x = tensor([[1], [2], [3], [4], [5], [6], [7], [8]])
 * const tree = growTree(undefined, { x, y: tensor([0, 0, 0, 1, 0, 1, 1, 1]), task: 'classification' })
 * print('grown:', tree.nodes.map((node) => node.label))
 * print('pruned at 0.1:', pruneTree(tree, 0.1).nodes.map((node) => node.label))
 */
export function pruneTree(tree: DecisionTree, alpha: number): DecisionTree {
  const cut = new Set<number>()
  for (;;) {
    const w = weakestLinks(tree, cut)
    if (w.best < 0 || w.alpha > alpha) break
    cut.add(w.best)
  }
  const nodes: DecisionNode[] = []
  const copy = (id: number, parent: number, side: number) => {
    const src = tree.nodes[id]
    const leaf = !src.children.length || cut.has(id)
    const nid = nodes.length
    const { slot: _slot, ...rest } = src
    void _slot
    nodes.push({
      ...rest,
      id: nid,
      parent: parent < 0 ? null : parent,
      children: [],
      ...(parent >= 0 && { slot: side }),
      feature: leaf ? -1 : src.feature,
      threshold: leaf ? NaN : src.threshold,
      decrease: leaf ? 0 : src.decrease,
      stop: src.children.length && leaf ? 'pruned' : src.stop,
      value: src.value.slice(),
    })
    nodes[nid].label = nodeLabel(nodes[nid], tree.task)
    if (parent >= 0) nodes[parent].children[side] = nid
    if (!leaf) {
      copy(src.children[0], nid, 0)
      copy(src.children[1], nid, 1)
    }
  }
  copy(tree.root, -1, 0)
  return { ...tree, nodes, root: 0, edges: nodes.map((node) => (node.parent === null ? null : {})) }
}

/** Data with optional per-row sample weights: inputs `x`, targets `y` and `weights` ($n$ values, default all 1). */
export type WeightedData = Supervised<Tensor, Tensor> & { weights?: Tensor }

/** A fitted classification tree. */
export interface DecisionTreeModel
  extends
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Trained<TreeGrowthState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'decision-tree'
  /** The fitted tree (pruned when `pruneAlpha` is above 0). */
  readonly tree: DecisionTree
  /** The tree before pruning (the same as `tree` when `pruneAlpha` is 0). */
  readonly grown: DecisionTree
  /** The number of classes $K$. */
  readonly classes: number
  /** The impurity importances of the fitted tree, $d$ values summing to 1. */
  readonly featureImportances: Tensor
  /** The leaf (a node id of `tree`) each row reaches. */
  apply(x: Tensor): Tensor
}

/**
 * A CART classification tree for labels $0, \dots, K - 1$ (Gini by default), as scikit-learn's
 * `DecisionTreeClassifier`. `forward` and `score` give the leaf's weighted class shares, $m \times K$, and
 * `predictive` the class law they define; `decide` the largest share (the lowest label on ties). A `pruneAlpha` above
 * 0 prunes the grown tree at that complexity. Growth is traced node by node in `training` (recording the node count).
 *
 * @param params The growth hyperparameters (`TreeParams`), and `pruneAlpha`, the cost-complexity $\alpha$ the grown
 *   tree is pruned at (default 0, no pruning).
 * @returns An estimator whose `fit({ x, y, weights? }, { stream? })` returns the fitted `DecisionTreeModel`; the stream
 *   draws the feature subsets when `maxFeatures` is below the number of features.
 *
 * @example The first split of a toy dataset
 * const x = tensor([[1, 2], [2, 7], [3, 1], [6, 8], [7, 3], [8, 6]])
 * const y = tensor([0, 0, 0, 1, 1, 1])
 * const model = decisionTree().fit({ x, y })
 * print('root split:', model.tree.nodes[0].label, ' Gini decrease:', model.tree.nodes[0].decrease)
 * print('importances:', model.featureImportances)
 * print('decisions:', model.decide(tensor([[0, 9], [9, 0]])))
 *
 * @example Pruning a tree grown on noisy labels
 * // The label is x > 0.5, flipped on every eighth point.
 * const x = uniform(stream(1), 0, 1, { shape: [40, 1] })
 * const y = tensor(Array.from(toFlat(x), (v, i) => (v > 0.5) !== (i % 8 === 0) ? 1 : 0))
 * const model = decisionTree({ pruneAlpha: 0.05 }).fit({ x, y })
 * print('grown leaves:', treeSize(model.grown).leaves, ' pruned leaves:', treeSize(model.tree).leaves)
 * print('pruned tree:', model.tree.nodes.map((node) => node.label))
 */
export function decisionTree(
  params: TreeParams & { pruneAlpha?: number } = {},
): Estimator<WeightedData, DecisionTreeModel> {
  const { pruneAlpha = 0, ...treeParams } = params
  return {
    name: 'decision-tree',
    params,
    fit({ x, y, weights }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'decisionTree')
      const { k: K } = classLabels(y, n, 'decisionTree')
      const problem = { x, y, weights, task: 'classification' as const, classes: K, params: treeParams }
      const training: Trace<TreeGrowthState> = trace(treeGrowthSteps(problem), undefined, 2 * n + 1, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          nodes: (s) => s.tree.nodes.length,
          ...(options.trace?.record as Record<string, (s: TreeGrowthState, t: number) => number> | undefined),
        },
      })
      const grown: DecisionTree = training.final.tree
      const tree = pruneAlpha > 0 ? pruneTree(grown, pruneAlpha) : grown
      const head = (q: Tensor) => {
        inputs(q, d, 'decisionTree')
        return values(predictTree(tree, q)).slice()
      }
      return {
        kind: 'model',
        name: 'decision-tree',
        tree,
        grown,
        classes: K,
        featureImportances: featureImportances(tree),
        training,
        apply: (q: Tensor) => applyTree(tree, q),
        ...probabilityModel(head, (h) => h, K),
      }
    },
  }
}

/** A fitted regression tree. */
export interface RegressionTreeModel
  extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor>, Expects<Tensor>, Trained<TreeGrowthState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'regression-tree'
  /** The fitted tree (pruned when `pruneAlpha` is above 0). */
  readonly tree: DecisionTree
  /** The tree before pruning (the same as `tree` when `pruneAlpha` is 0). */
  readonly grown: DecisionTree
  /** The impurity importances of the fitted tree, $d$ values summing to 1. */
  readonly featureImportances: Tensor
  /** The leaf (a node id of `tree`) each row reaches. */
  apply(x: Tensor): Tensor
}

/**
 * A CART regression tree (squared error), as scikit-learn's `DecisionTreeRegressor`: each leaf predicts the weighted
 * mean of its targets. `expect(x, f)` is the weighted mean of $f$ over the training targets in the leaf each row of `x`
 * reaches (the tree's empirical conditional law of $y$); without `f` it is the prediction. A `pruneAlpha` above 0
 * prunes the grown tree at that complexity.
 *
 * @param params The growth hyperparameters (`TreeParams` but the criterion, which is squared error), and `pruneAlpha`,
 *   the cost-complexity $\alpha$ the grown tree is pruned at (default 0, no pruning).
 * @returns An estimator whose `fit({ x, y, weights? }, { stream? })` returns the fitted `RegressionTreeModel`.
 *
 * @example A step function, and the spread within each leaf
 * const x = tensor([[1], [2], [3], [4], [5], [6]])
 * const y = tensor([1, 1.2, 0.8, 5, 5.4, 4.6])
 * const model = regressionTree({ maxDepth: 1 }).fit({ x, y })
 * print('split:', model.tree.nodes[0].label)
 * const q = tensor([[0], [10]])
 * print('prediction:', model.decide(q))
 * print('E[y^2] in the leaf:', model.expect(q, (v) => v * v))
 */
export function regressionTree(
  params: Omit<TreeParams, 'criterion'> & { pruneAlpha?: number } = {},
): Estimator<WeightedData, RegressionTreeModel> {
  const { pruneAlpha = 0, ...treeParams } = params
  return {
    name: 'regression-tree',
    params,
    fit({ x, y, weights }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'regressionTree')
      targets(y, n, 'regressionTree')
      const problem = {
        x,
        y,
        weights,
        task: 'regression' as const,
        params: { ...treeParams, criterion: 'squared' as const },
      }
      const training = trace(treeGrowthSteps(problem), undefined, 2 * n + 1, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
      })
      const grown: DecisionTree = training.final.tree
      const tree = pruneAlpha > 0 ? pruneTree(grown, pruneAlpha) : grown
      const predict = (q: Tensor) => {
        inputs(q, d, 'regressionTree')
        return predictTree(tree, q)
      }
      // The training targets (and weights) of each leaf, for expectations of functions of y.
      const yv = targets(y, n, 'regressionTree')
      const wv = weights ? values(weights) : null
      const members = new Map<number, number[]>()
      values(applyTree(tree, x)).forEach((leaf, i) => {
        const list = members.get(leaf) ?? []
        list.push(i)
        members.set(leaf, list)
      })
      const expect = (q: Tensor, f?: (y: number) => number) => {
        if (!f) return predict(q)
        const leaves = values(applyTree(tree, q))
        return fromData(
          Float64Array.from(leaves, (leaf) => {
            const rows = members.get(leaf) ?? []
            let total = 0
            let sum = 0
            for (const i of rows) {
              const w = wv ? wv[i] : 1
              total += w
              sum += w * f(yv[i])
            }
            return total > 0 ? sum / total : NaN
          }),
          [leaves.length],
        )
      }
      return {
        kind: 'model',
        name: 'regression-tree',
        tree,
        grown,
        featureImportances: featureImportances(tree),
        training,
        apply: (q: Tensor) => applyTree(tree, q),
        forward: predict,
        decide: predict,
        expect,
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'decisionTree',
    module: 'learning/trees-and-ensembles',
    name: 'Decision tree',
    summary: 'CART classification tree grown greedily, with cost-complexity pruning.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'score'],
    hyper: space({
      criterion: oneOf(['gini', 'entropy']),
      maxDepth: int(1, 32, { default: 32, doc: 'The factory default is unlimited.' }),
      minSamplesSplit: int(2, 100, { default: 2 }),
      minSamplesLeaf: int(1, 100, { default: 1 }),
      minImpurityDecrease: real(0, 1, { default: 0 }),
      maxLeaves: int(2, 1024, { default: 1024, doc: 'The factory default is unlimited.' }),
      order: oneOf(['depth-first', 'breadth-first', 'best-first']),
      pruneAlpha: real(0, 1, { default: 0, label: 'α' }),
    }),
    notes: ['decision-tree', 'tree-pruning'],
    cite: ['breiman1984'],
  },
  decisionTree,
)

defineModel(
  {
    key: 'regressionTree',
    module: 'learning/trees-and-ensembles',
    name: 'Regression tree',
    summary: 'CART regression tree on squared error, with cost-complexity pruning.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'expect'],
    hyper: space({
      maxDepth: int(1, 32, { default: 32, doc: 'The factory default is unlimited.' }),
      minSamplesSplit: int(2, 100, { default: 2 }),
      minSamplesLeaf: int(1, 100, { default: 1 }),
      minImpurityDecrease: real(0, 1, { default: 0 }),
      pruneAlpha: real(0, 1, { default: 0, label: 'α' }),
    }),
    notes: ['decision-tree', 'tree-pruning'],
    cite: ['breiman1984'],
  },
  regressionTree,
)
