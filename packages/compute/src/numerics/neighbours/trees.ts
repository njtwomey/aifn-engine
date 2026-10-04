/**
 * Space-partitioning trees for exact nearest-neighbour search: the k-d tree (Bentley 1975, "Multidimensional binary
 * search trees used for associative searching", CACM 18(9); Friedman, Bentley and Finkel 1977, ACM TOMS 3(3)) and the
 * ball tree (Omohundro 1989, "Five balltree construction algorithms", ICSI TR-89-063; Uhlmann 1991), built as
 * scikit-learn's `KDTree` and `BallTree` are: each node splits its points at the median of the coordinate with the
 * largest spread, down to leaves of at most `leafSize` points. The vantage-point tree (Yianilos 1993, "Data structures
 * and algorithms for nearest neighbor search in general metric spaces", SODA) instead splits a node's points at the
 * median distance from one of them, the vantage point.
 *
 * A k-d node keeps the bounding box of its points, a ball node the centroid and the radius that covers them, and a
 * vantage-point node the shell [lo, hi] of distances from its parent's vantage point that holds its points. The search
 * is depth first, nearer child first, and skips a node whose lower bound on the distance to any of its points (the
 * distance from the query to the box, to the ball's surface, or to the shell) already exceeds the k-th best distance found. The
 * answer is exact; what the tree saves is distance evaluations, which `visits` records node by node.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  checkK,
  distanceOf,
  kBest,
  queryOf,
  rowsOf,
  stackResults,
  type Neighbours,
  type NeighbourMetric,
  type QueryResult,
} from './search'

/** The metrics a tree prunes exactly with (the cosine distance breaks the triangle inequality). */
export type TreeMetric = Exclude<NeighbourMetric, 'cosine'>

/** One node of a k-d or ball tree. Its points are `order[start … end − 1]`. */
export interface SpaceTreeNode {
  readonly start: number
  readonly end: number
  /** Child node ids, −1 at a leaf. */
  readonly left: number
  readonly right: number
  readonly depth: number
  /**
   * k-d tree: the split coordinate and value (points with a smaller rank go left); −1 and NaN at a leaf. Vantage-point
   * tree: `dim` is −1 and `split` is the median distance μ from the vantage point at an internal node.
   */
  readonly dim: number
  readonly split: number
  /** k-d tree: the bounding box of the node's points. */
  readonly lower?: readonly number[]
  readonly upper?: readonly number[]
  /** Ball tree: the centroid of the node's points and the largest distance from it to one of them. */
  readonly centre?: readonly number[]
  readonly radius?: number
  /** Vantage-point tree, internal node: the index of its vantage point (its points nearer than `split` go left). */
  readonly vantage?: number
  /**
   * Vantage-point tree, below the root: the parent's vantage point and the least and greatest distance from it to one
   * of this node's points. Every point of the node lies in that shell.
   */
  readonly anchor?: readonly number[]
  readonly shell?: readonly [number, number]
}

/** A built k-d tree or ball tree over n points of width d. */
export interface SpaceTree {
  readonly kind: 'kd-tree' | 'ball-tree' | 'vp-tree'
  readonly n: Size
  readonly d: Size
  /** The points, row-major n × d (a copy). */
  readonly data: Float64Array
  /** The point indices, permuted so that every node's points are contiguous. */
  readonly order: Int32Array
  /** Node 0 is the root; children follow their parent. */
  readonly nodes: readonly SpaceTreeNode[]
  readonly leafSize: Size
  readonly metric: TreeMetric
}

/** Options of {@link kdTree}, {@link ballTree} and {@link vpTree}. */
export interface SpaceTreeOptions {
  /** Most points in a leaf (default 40, as scikit-learn; small values make deep trees for drawing). */
  leafSize?: Size
  /** Default `euclidean`. */
  metric?: TreeMetric
}

function build(kind: SpaceTree['kind'], x: MatrixLike, options: SpaceTreeOptions): SpaceTree {
  const where = kind === 'kd-tree' ? 'kdTree' : kind === 'ball-tree' ? 'ballTree' : 'vpTree'
  const { leafSize = 40, metric = 'euclidean' } = options
  if (!(Number.isInteger(leafSize) && leafSize >= 1))
    throw new DomainError(where, `${where}: leafSize must be a positive integer`)
  if ((metric as NeighbourMetric) === 'cosine')
    throw new DomainError(where, `${where}: the cosine distance is not a metric; normalise and use euclidean`)
  const { n, d, data } = rowsOf(x, where)
  const order = Int32Array.from({ length: n }, (_, i) => i)
  const nodes: SpaceTreeNode[] = []
  const at = (i: number, c: number) => data[order[i] * d + c]
  // The vantage-point tree measures shells in the underlying metric (squared Euclidean is not one).
  const base = metric === 'sqeuclidean' ? 'euclidean' : metric
  const make = (start: number, end: number, depth: number, anchor?: number[], shell?: [number, number]): number => {
    const id = nodes.length
    nodes.push(null as unknown as SpaceTreeNode)
    const lower = new Array<number>(d).fill(Infinity)
    const upper = new Array<number>(d).fill(-Infinity)
    for (let i = start; i < end; i++)
      for (let c = 0; c < d; c++) {
        lower[c] = Math.min(lower[c], at(i, c))
        upper[c] = Math.max(upper[c], at(i, c))
      }
    const shape: {
      lower?: number[]
      upper?: number[]
      centre?: number[]
      radius?: number
      anchor?: number[]
      shell?: [number, number]
    } = {}
    if (kind === 'vp-tree') {
      if (anchor) {
        shape.anchor = anchor
        shape.shell = shell
      }
    } else if (kind === 'kd-tree') {
      shape.lower = lower
      shape.upper = upper
    } else {
      const centre = new Array<number>(d).fill(0)
      for (let i = start; i < end; i++) for (let c = 0; c < d; c++) centre[c] += at(i, c) / (end - start)
      let radius = 0
      for (let i = start; i < end; i++) radius = Math.max(radius, distanceOf(metric, centre, 0, data, order[i], d))
      shape.centre = centre
      shape.radius = radius
    }
    if (end - start <= leafSize) {
      nodes[id] = { start, end, left: -1, right: -1, depth, dim: -1, split: NaN, ...shape }
      return id
    }
    if (kind === 'vp-tree') {
      // Vantage point: the node's point farthest from its centroid (a corner of the cloud, whose distances spread most),
      // ties to the smaller index; the node's points are sorted by distance to it and split at the median.
      const centre = new Array<number>(d).fill(0)
      for (let i = start; i < end; i++) for (let c = 0; c < d; c++) centre[c] += at(i, c) / (end - start)
      let v = order[start]
      let far = -1
      for (let i = start; i < end; i++) {
        const r = distanceOf(base, centre, 0, data, order[i], d)
        if (r > far || (r === far && order[i] < v)) [far, v] = [r, order[i]]
      }
      const vp = Array.from(data.subarray(v * d, (v + 1) * d))
      const dist = new Map<number, number>()
      for (let i = start; i < end; i++) dist.set(order[i], distanceOf(base, vp, 0, data, order[i], d))
      const slice = Array.from(order.subarray(start, end)).sort((a, b) => dist.get(a)! - dist.get(b)! || a - b)
      order.set(slice, start)
      const mid = start + Math.floor((end - start) / 2)
      const mu = dist.get(order[mid])!
      const shellOf = (a: number, b: number): [number, number] => [dist.get(order[a])!, dist.get(order[b - 1])!]
      const left = make(start, mid, depth + 1, vp, shellOf(start, mid))
      const right = make(mid, end, depth + 1, vp, shellOf(mid, end))
      nodes[id] = { start, end, left, right, depth, dim: -1, split: mu, vantage: v, ...shape }
      return id
    }
    let dim = 0
    for (let c = 1; c < d; c++) if (upper[c] - lower[c] > upper[dim] - lower[dim]) dim = c
    // Sort the node's points on the split coordinate (ties by index, so the build is reproducible), split at the median.
    const slice = Array.from(order.subarray(start, end)).sort((a, b) => data[a * d + dim] - data[b * d + dim] || a - b)
    order.set(slice, start)
    const mid = start + Math.floor((end - start) / 2)
    const split = at(mid, dim)
    const left = make(start, mid, depth + 1)
    const right = make(mid, end, depth + 1)
    nodes[id] = { start, end, left, right, depth, dim, split, ...shape }
    return id
  }
  if (n > 0) make(0, n, 0)
  return { kind, n, d, data: Float64Array.from(data), order, nodes, leafSize, metric }
}

/** A k-d tree over the rows of x (n × d): median splits on the coordinate of largest spread, boxes at every node. */
export function kdTree(x: MatrixLike, options: SpaceTreeOptions = {}): SpaceTree {
  return build('kd-tree', x, options)
}

/** A ball tree over the rows of x (n × d): the same splits as the k-d tree, a centroid and radius at every node. */
export function ballTree(x: MatrixLike, options: SpaceTreeOptions = {}): SpaceTree {
  return build('ball-tree', x, options)
}

/**
 * A vantage-point tree over the rows of x (n × d): each node picks the point farthest from its centroid as vantage point
 * and splits its points at the median distance μ from it; a child keeps the shell of distances that holds its points.
 * Only the triangle inequality is used, so any of the tree metrics works.
 */
export function vpTree(x: MatrixLike, options: SpaceTreeOptions = {}): SpaceTree {
  return build('vp-tree', x, options)
}

/**
 * A lower bound on the distance from q to any point of a node: the distance to its box (k-d), to its ball's surface
 * (ball tree), or to the shell of distances from the parent's vantage point that holds its points (vantage-point tree:
 * max(0, lo − s, s − hi) for s the query's distance to that point, by the triangle inequality); 0 when the query is
 * inside, and 0 at a vantage-point tree's root.
 */
export function nodeLowerBound(tree: SpaceTree, node: number, q: ArrayLike<number>): number {
  const nd = tree.nodes[node]
  const { d, metric } = tree
  if (tree.kind === 'vp-tree') {
    if (!nd.anchor || !nd.shell) return 0
    const s = distanceOf(metric === 'sqeuclidean' ? 'euclidean' : metric, q, 0, nd.anchor, 0, d)
    const gap = Math.max(0, nd.shell[0] - s, s - nd.shell[1])
    return metric === 'sqeuclidean' ? gap * gap : gap
  }
  if (tree.kind === 'ball-tree') {
    const base = metric === 'sqeuclidean' ? 'euclidean' : metric
    const gap = Math.max(
      0,
      distanceOf(base, q, 0, nd.centre!, 0, d) - nd.radius! ** (metric === 'sqeuclidean' ? 0.5 : 1),
    )
    return metric === 'sqeuclidean' ? gap * gap : gap
  }
  let s = 0
  for (let c = 0; c < d; c++) {
    const g = Math.max(0, nd.lower![c] - q[c], q[c] - nd.upper![c])
    if (metric === 'manhattan') s += g
    else if (metric === 'chebyshev') s = Math.max(s, g)
    else s += g * g
  }
  return metric === 'euclidean' ? Math.sqrt(s) : s
}

/**
 * One event of a tree search: the node reached, its lower bound, and what the search did: `descend` into its children,
 * `scan` its points (a leaf), or `prune` it because the bound exceeded the k-th best distance `worst` held then.
 */
export interface TreeVisit {
  readonly node: number
  readonly bound: number
  readonly action: 'descend' | 'scan' | 'prune'
  /** The k-th best distance after the event (Infinity until k points are held). */
  readonly worst: number
}

/** The answer to one tree query with its visit order. */
export interface TreeQueryResult extends QueryResult {
  readonly visits: readonly TreeVisit[]
}

/**
 * The exact k nearest points of the tree to the query, by depth-first branch and bound (module notes), with every
 * node visit recorded in order. Ties to the smaller index, as brute force.
 */
export function treeQuery(tree: SpaceTree, query: VectorLike, k: Size): TreeQueryResult {
  const q = queryOf(query, tree.d, 'treeQuery')
  checkK(k, tree.n, 'treeQuery')
  const best = kBest(k)
  const visits: TreeVisit[] = []
  let evaluations = 0
  const visit = (id: number, bound: number) => {
    if (bound > best.worst()) {
      visits.push({ node: id, bound, action: 'prune', worst: best.worst() })
      return
    }
    const nd = tree.nodes[id]
    if (nd.left < 0) {
      for (let i = nd.start; i < nd.end; i++) {
        const j = tree.order[i]
        best.offer(j, distanceOf(tree.metric, q, 0, tree.data, j, tree.d))
        evaluations++
      }
      visits.push({ node: id, bound, action: 'scan', worst: best.worst() })
      return
    }
    visits.push({ node: id, bound, action: 'descend', worst: best.worst() })
    const bl = nodeLowerBound(tree, nd.left, q)
    const br = nodeLowerBound(tree, nd.right, q)
    if (bl <= br) {
      visit(nd.left, bl)
      visit(nd.right, br)
    } else {
      visit(nd.right, br)
      visit(nd.left, bl)
    }
  }
  if (tree.nodes.length > 0) visit(0, nodeLowerBound(tree, 0, q))
  return { indices: best.indices, distances: best.distances, distanceEvaluations: evaluations, visits }
}

/** The exact k nearest points of the tree to each row of `queries` (m × d), as scikit-learn's `tree.query`. */
export function treeSearch(tree: SpaceTree, queries: MatrixLike, k: Size): Neighbours {
  const Q = rowsOf(queries, 'treeSearch')
  const results: QueryResult[] = []
  for (let i = 0; i < Q.n; i++) results.push(treeQuery(tree, Q.data.subarray(i * Q.d, (i + 1) * Q.d), k))
  return stackResults(results, k)
}
