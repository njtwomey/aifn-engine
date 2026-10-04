/**
 * Isolation forest (Liu, Ting and Zhou, 2008): anomalies are few and different, so random axis-aligned splits isolate
 * them in fewer steps than normal points. Each tree is grown on a subsample of ψ points by choosing a feature and a
 * split value uniformly at random until every point is alone or the depth limit ⌈log₂ ψ⌉ is reached; a point's score is
 * 2^{−E[h(x)]/c(ψ)}, where h is its path length and c(n) the average path length of an unsuccessful search in a binary
 * search tree of n points.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { child, choice, stream as makeStream, uniform, integers, type Stream } from 'aifn-compute/foundation/random'
import { dense, toFlat } from 'aifn-compute/foundation/tensor'

/** One node of an isolation tree: an internal split, or a leaf holding `size` training points. */
export type IsolationNode =
  | { leaf: false; feature: number; threshold: number; left: IsolationNode; right: IsolationNode }
  | { leaf: true; size: Size; depth: Size }

/** A fitted isolation forest. */
export type IsolationForest = {
  trees: IsolationNode[]
  /** The subsample size ψ. */
  sampleSize: Size
  dimension: Size
}

/** Options of `isolationForest`. */
export type IsolationForestOptions = {
  /** Trees t (default 100). */
  trees?: Size
  /** Subsample size ψ per tree (default min(256, n)). */
  sampleSize?: Size
  /** The stream of the subsamples and splits (default `stream('isolation-forest')`). */
  stream?: Stream
}

/** c(n) = 2H(n − 1) − 2(n − 1)/n, the mean path length of an unsuccessful BST search (c(1) = 0, c(2) = 1). */
export function averagePathLength(n: number): number {
  if (n <= 1) return 0
  if (n === 2) return 1
  const harmonic = Math.log(n - 1) + 0.5772156649015329
  return 2 * harmonic - (2 * (n - 1)) / n
}

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

/** Grow an isolation forest on the rows of x (n × d). Tree t draws its subsample and splits from `child(s, t)`. */
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

/** The path length of x in one tree: the depth of its leaf plus c(size) for the points left unsplit there. */
function pathLength(node: IsolationNode, x: ArrayLike<number>, offset: number): number {
  let n = node
  while (!n.leaf) n = x[offset + n.feature] < n.threshold ? n.left : n.right
  return n.depth + averagePathLength(n.size)
}

/** The anomaly score s(x) = 2^{−E[h(x)]/c(ψ)} of each row: near 1 for anomalies, below ½ for normal points. */
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
