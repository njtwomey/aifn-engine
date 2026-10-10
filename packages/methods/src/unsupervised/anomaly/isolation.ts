/**
 * Isolation forest (Liu, Ting and Zhou, 2008): anomalies are few and different, so random axis-aligned splits isolate
 * them in fewer steps than normal points. Each tree is grown on a subsample of $\psi$ points, drawn without
 * replacement, by choosing a feature (among those on which the node's points still differ) and a split value uniformly
 * at random until every point is alone or the depth limit $\lceil \log_2 \psi \rceil$ is reached; a point's score
 * is $s(\xvec) = 2^{-\expect[h(\xvec)] / c(\psi)}$, where $h$ is its path length and $c(n)$ the average path length
 * of an unsuccessful search in a binary search tree of $n$ points. As scikit-learn's `IsolationForest`, whose
 * `score_samples` is $-s(\xvec)$.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { child, choice, stream as makeStream, uniform, integers, type Stream } from 'aifn-compute/foundation/random'
import { dense, toFlat } from 'aifn-compute/foundation/tensor'

/**
 * One node of an isolation tree: an internal split (points with `feature` below `threshold` go `left`, the rest
 * `right`), or a leaf holding `size` training points at `depth` (the root at 0).
 */
export type IsolationNode =
  | { leaf: false; feature: number; threshold: number; left: IsolationNode; right: IsolationNode }
  | { leaf: true; size: Size; depth: Size }

/** A fitted isolation forest. */
export type IsolationForest = {
  /** The roots of the trees. */
  trees: IsolationNode[]
  /** The subsample size $\psi$. */
  sampleSize: Size
  /** The number of features $d$ of the training points. */
  dimension: Size
}

/** Options of `isolationForest`. */
export type IsolationForestOptions = {
  /** Trees $t$ (default 100). */
  trees?: Size
  /** Subsample size $\psi$ per tree (default $\min(256, n)$; a larger one is lowered to $n$). */
  sampleSize?: Size
  /** The stream of the subsamples and splits (default `stream('isolation-forest')`). */
  stream?: Stream
}

/**
 * $c(n) = 2H(n - 1) - 2(n - 1)/n$, the mean path length of an unsuccessful search in a binary search tree of $n$
 * points ($c(1) = 0$, $c(2) = 1$), with the harmonic number $H(i)$ approximated by $\ln i + \gamma$ ($\gamma$
 * Euler's constant), as Liu, Ting and Zhou (2008) and scikit-learn.
 *
 * @param n The number of points; any $n \le 1$ gives 0.
 * @returns The average path length $c(n)$.
 *
 * @example The normaliser of a subsample of 256 points
 * print('c(2) =', averagePathLength(2))
 * print('c(256) =', averagePathLength(256))
 */
export function averagePathLength(n: number): number {
  if (n <= 1) return 0
  if (n === 2) return 1
  const harmonic = Math.log(n - 1) + 0.5772156649015329
  return 2 * harmonic - (2 * (n - 1)) / n
}

/**
 * Grow one isolation tree on some rows: split at random until a node has at most one row, all its rows are equal, or
 * it is at depth `limit`.
 *
 * @param x The training points, row-major, $d$ values per row.
 * @param d The number of features.
 * @param rows The row indices of `x` that reach this node.
 * @param depth The depth of this node (0 at the root).
 * @param limit The depth at which every node is a leaf.
 * @param s The node's stream: the feature is drawn from `child(s, 'feature')`, the split from `child(s, 'split')`,
 *   and the subtrees grow from `child(s, 'left')` and `child(s, 'right')`.
 * @returns The subtree rooted at this node.
 */
function grow(x: Float64Array, d: Size, rows: number[], depth: Size, limit: Size, s: Stream): IsolationNode {
  if (rows.length <= 1 || depth >= limit) return { leaf: true, size: rows.length, depth }
  // Features on which the rows still differ; a node whose rows are identical cannot be split.
  const spread: { f: number; lo: number; hi: number }[] = []
  for (let f = 0; f < d; f++) {
    let lo = Infinity
    let hi = -Infinity
    for (const r of rows) {
      lo = Math.min(lo, x[r * d + f])
      hi = Math.max(hi, x[r * d + f])
    }
    if (hi > lo) spread.push({ f, lo, hi })
  }
  if (spread.length === 0) return { leaf: true, size: rows.length, depth }
  const { f, lo, hi } = spread[integers(child(s, 'feature'), spread.length)]
  const threshold = lo + uniform(child(s, 'split')) * (hi - lo)
  const left = rows.filter((r) => x[r * d + f] < threshold)
  const right = rows.filter((r) => x[r * d + f] >= threshold)
  return {
    leaf: false,
    feature: f,
    threshold,
    left: grow(x, d, left, depth + 1, limit, child(s, 'left')),
    right: grow(x, d, right, depth + 1, limit, child(s, 'right')),
  }
}

/**
 * Grow an isolation forest on the rows of `x`. Tree $t$ draws its subsample and splits from `child(s, t)`, so the
 * same stream gives the same forest.
 *
 * @param x The training points, $n \times d$: nested arrays or a rank-2 tensor.
 * @param options The number of trees, the subsample size and the stream.
 * @returns The forest, to score points with `isolationScore`.
 *
 * @example The outlier scores highest
 * const x = concat([normals(stream(0), [40, 2]), tensor([[5, 5]])])
 * const forest = isolationForest(x, { trees: 50, stream: stream(1) })
 * const s = isolationScore(forest, x)
 * print('score of the outlier', s[40])
 * print('highest of the rest', Math.max(...s.slice(0, 40)))
 */
export function isolationForest(x: MatrixLike, options: IsolationForestOptions = {}): IsolationForest {
  const m = dense.toMatrixF64(x, 'isolationForest')
  const { trees = 100, stream: s = makeStream('isolation-forest') } = options
  const psi = Math.min(options.sampleSize ?? 256, m.m)
  const limit = Math.ceil(Math.log2(Math.max(2, psi)))
  const data = Float64Array.from(m.data)
  const forest: IsolationNode[] = []
  for (let t = 0; t < trees; t++) {
    const ts = child(s, t)
    const rows = Array.from(toFlat(choice(child(ts, 'sample'), m.m, psi, { replace: false })))
    forest.push(grow(data, m.n, rows, 0, limit, child(ts, 'tree')))
  }
  return { trees: forest, sampleSize: psi, dimension: m.n }
}

/**
 * The path length of a point in one tree: the depth of its leaf plus $c(\text{size})$ for the points left unsplit
 * there.
 *
 * @param node The root of the tree.
 * @param x The points' values, row-major; the point read starts at `offset`.
 * @param offset The index in `x` of the point's first feature.
 * @returns The path length $h$.
 */
function pathLength(node: IsolationNode, x: ArrayLike<number>, offset: number): number {
  let n = node
  while (!n.leaf) n = x[offset + n.feature] < n.threshold ? n.left : n.right
  return n.depth + averagePathLength(n.size)
}

/**
 * The anomaly score $s(\xvec) = 2^{-\expect[h(\xvec)] / c(\psi)}$ of each row, the mean path length taken over the
 * trees: near 1 for anomalies, below $\frac{1}{2}$ for normal points. The number of columns is not checked against the
 * forest's `dimension`.
 *
 * @param forest The forest, as `isolationForest` returns it.
 * @param x The points to score, $m \times d$: nested arrays or a rank-2 tensor.
 * @returns The score of each row ($m$ values, in $(0, 1]$).
 *
 * @example Scores of a centre point and a far one
 * const forest = isolationForest(normals(stream(0), [64, 2]), { stream: stream(1) })
 * print('scores', isolationScore(forest, [[0, 0], [4, -4]]))
 */
export function isolationScore(forest: IsolationForest, x: MatrixLike): Float64Array {
  const m = dense.toMatrixF64(x, 'isolationScore')
  const c = averagePathLength(forest.sampleSize)
  const out = new Float64Array(m.m)
  for (let r = 0; r < m.m; r++) {
    let h = 0
    for (const t of forest.trees) h += pathLength(t, m.data, r * m.n)
    out[r] = Math.pow(2, -h / forest.trees.length / c)
  }
  return out
}
