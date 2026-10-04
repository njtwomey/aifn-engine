/**
 * Space-partitioning trees for exact nearest-neighbour search: the $k$-d tree (Bentley 1975, "Multidimensional binary
 * search trees used for associative searching", CACM 18(9); Friedman, Bentley and Finkel 1977, ACM TOMS 3(3)) and the
 * ball tree (Omohundro 1989, "Five balltree construction algorithms", ICSI TR-89-063; Uhlmann 1991), built as
 * scikit-learn's `KDTree` and `BallTree` are: each node splits its points at the median of the coordinate with the
 * largest spread, down to leaves of at most `leafSize` points. The vantage-point tree (Yianilos 1993, "Data structures
 * and algorithms for nearest neighbor search in general metric spaces", SODA) instead splits a node's points at the
 * median distance from one of them, the vantage point.
 *
 * A $k$-d node keeps the bounding box of its points, a ball node the centroid and the radius that covers them, and a
 * vantage-point node the shell $[lo, hi]$ of distances from its parent's vantage point that holds its points. The search
 * is depth first, nearer child first, and skips a node whose lower bound on the distance to any of its points (the
 * distance from the query to the box, to the ball's surface, or to the shell) already exceeds the $k$-th best distance found. The
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

/** One node of a $k$-d or ball tree. Its points are `order[start … end − 1]`. */
export interface SpaceTreeNode {
  /** Index into order array where this node's points begin. */
  readonly start: number
  /** Index into order array where this node's points end. */
  readonly end: number
  /** Left child node ID, or $-1$ at a leaf. */
  readonly left: number
  /** Right child node ID, or $-1$ at a leaf. */
  readonly right: number
  /** Node depth in the tree ($0$ at the root). */
  readonly depth: number
  /**
   * $k$-d tree: split coordinate axis index (points with smaller value go left); $-1$ at a leaf.
   * Vantage-point tree: `dim` is $-1$ and `split` is the median distance $\mu$ from the vantage point.
   */
  readonly dim: number
  /** Split coordinate value or median distance threshold; `NaN` at a leaf. */
  readonly split: number
  /** $k$-d tree: lower corner of bounding box. */
  readonly lower?: readonly number[]
  /** $k$-d tree: upper corner of bounding box. */
  readonly upper?: readonly number[]
  /** Ball tree: centroid coordinate of points in this node. */
  readonly centre?: readonly number[]
  /** Ball tree: enclosing ball radius. */
  readonly radius?: number
  /** Vantage-point tree: index of the vantage point. */
  readonly vantage?: number
  /** Vantage-point tree: anchor point of parent's vantage point. */
  readonly anchor?: readonly number[]
  /** Vantage-point tree: shell distance interval $[lo, hi]$ holding this node's points. */
  readonly shell?: readonly [number, number]
}

/** A built $k$-d tree or ball tree over $n$ points of width $d$. */
export interface SpaceTree {
  /** Discriminator kind of space tree. */
  readonly kind: 'kd-tree' | 'ball-tree' | 'vp-tree'
  /** Number of data points $n$. */
  readonly n: Size
  /** Dimensionality $d$. */
  readonly d: Size
  /** The points, row-major $n \times d$ (a copy). */
  readonly data: Float64Array
  /** The point indices, permuted so that every node's points are contiguous. */
  readonly order: Int32Array
  /** Node array, where node $0$ is the root and children follow their parent. */
  readonly nodes: readonly SpaceTreeNode[]
  /** Maximum number of points in a leaf node. */
  readonly leafSize: Size
  /** Distance metric used. */
  readonly metric: TreeMetric
}

/** Options of {@link kdTree}, {@link ballTree} and {@link vpTree}. */
export interface SpaceTreeOptions {
  /** Most points in a leaf (default 40, as scikit-learn; small values make deep trees for drawing). */
  leafSize?: Size
  /** Default `euclidean`. */
  metric?: TreeMetric
}

/**
 * Build a space-partitioning tree of the given kind.
 *
 * @param kind - Tree structure kind: `'kd-tree'`, `'ball-tree'`, or `'vp-tree'`.
 * @param x - Input data matrix ($n \times d$).
 * @param options - Tree construction options.
 * @param options.leafSize - Maximum points per leaf (default 40).
 * @param options.metric - Metric (default `'euclidean'`).
 * @returns Built space-partitioning tree.
 */
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

/**
 * A $k$-d tree over the rows of $x$ ($n \times d$): median splits on the coordinate of largest spread, boxes at every node.
 *
 * @param x - Input data matrix ($n \times d$).
 * @param options - Tree construction options.
 * @param options.leafSize - Maximum points per leaf (default 40).
 * @param options.metric - Metric (default `'euclidean'`).
 * @returns Built $k$-d tree.
 *
 * @example Build a k-d tree
 * const data = [[0, 0], [1, 2], [2, 1], [3, 3]]
 * const tree = kdTree(data, { leafSize: 2 })
 * print('Nodes built:', tree.nodes.length)
 */
export function kdTree(x: MatrixLike, options: SpaceTreeOptions = {}): SpaceTree {
  return build('kd-tree', x, options)
}

/**
 * A ball tree over the rows of $x$ ($n \times d$): the same splits as the $k$-d tree, a centroid and radius at every node.
 *
 * @param x - Input data matrix ($n \times d$).
 * @param options - Tree construction options.
 * @param options.leafSize - Maximum points per leaf (default 40).
 * @param options.metric - Metric (default `'euclidean'`).
 * @returns Built ball tree.
 *
 * @example Build a ball tree
 * const data = [[0, 0], [1, 2], [2, 1], [3, 3]]
 * const tree = ballTree(data, { leafSize: 2 })
 * print('Nodes built:', tree.nodes.length)
 */
export function ballTree(x: MatrixLike, options: SpaceTreeOptions = {}): SpaceTree {
  return build('ball-tree', x, options)
}

/**
 * A vantage-point tree over the rows of $x$ ($n \times d$): each node picks the point farthest from its centroid as vantage point
 * and splits its points at the median distance $\mu$ from it; a child keeps the shell of distances that holds its points.
 * Only the triangle inequality is used, so any of the tree metrics works.
 *
 * @param x - Input data matrix ($n \times d$).
 * @param options - Tree construction options.
 * @param options.leafSize - Maximum points per leaf (default 40).
 * @param options.metric - Metric (default `'euclidean'`).
 * @returns Built vantage-point tree.
 *
 * @example Build a vantage-point tree
 * const data = [[0, 0], [1, 2], [2, 1], [3, 3]]
 * const tree = vpTree(data, { leafSize: 2 })
 * print('Nodes built:', tree.nodes.length)
 */
export function vpTree(x: MatrixLike, options: SpaceTreeOptions = {}): SpaceTree {
  return build('vp-tree', x, options)
}

/**
 * A lower bound on the distance from $q$ to any point of a node: the distance to its box ($k$-d), to its ball's surface
 * (ball tree), or to the shell of distances from the parent's vantage point that holds its points (vantage-point tree:
 * $\max(0, lo - s, s - hi)$ for $s$ the query's distance to that point, by the triangle inequality); $0$ when the query is
 * inside, and $0$ at a vantage-point tree's root.
 *
 * @param tree - Space tree instance.
 * @param node - Node index in tree.
 * @param q - Query coordinates array.
 * @returns Lower bound on the distance to any point in the node.
 *
 * @example Compute lower bound of root node
 * const data = [[0, 0], [1, 1], [2, 2]]
 * const tree = kdTree(data)
 * const lb = nodeLowerBound(tree, 0, [5, 5])
 * print('Lower bound:', lb)
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
 * `scan` its points (a leaf), or `prune` it because the bound exceeded the $k$-th best distance `worst` held then.
 */
export interface TreeVisit {
  /** Node ID visited. */
  readonly node: number
  /** Lower bound distance to query for this node. */
  readonly bound: number
  /** Action taken: descend into children, scan leaf points, or prune subtree. */
  readonly action: 'descend' | 'scan' | 'prune'
  /** The $k$-th best distance after the event ($\infty$ until $k$ points are held). */
  readonly worst: number
}

/** The answer to one tree query with its visit order. */
export interface TreeQueryResult extends QueryResult {
  /** Trace of all node visits performed during the search. */
  readonly visits: readonly TreeVisit[]
}

/**
 * The exact $k$ nearest points of the tree to the query, by depth-first branch and bound (module notes), with every
 * node visit recorded in order. Ties to the smaller index, as brute force.
 *
 * @param tree - Space tree to search.
 * @param query - Query vector of length $d$.
 * @param k - Number of nearest neighbours $k$ to return.
 * @returns Query result containing $k$ nearest indices, distances, and visit history.
 *
 * @example Query nearest neighbours in a tree
 * const data = [[0, 0], [1, 1], [2, 2], [3, 3]]
 * const tree = kdTree(data)
 * const res = treeQuery(tree, [0.9, 0.9], 2)
 * print('Nearest index:', res.indices[0])
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

/**
 * The exact $k$ nearest points of the tree to each row of `queries` ($m \times d$), as scikit-learn's `tree.query`.
 *
 * @param tree - Space tree to search.
 * @param queries - Query points matrix ($m \times d$).
 * @param k - Number of nearest neighbours $k$ to return.
 * @returns Nearest neighbours results containing indices and distances tensors.
 *
 * @example Search nearest neighbours for multiple queries
 * const data = [[0, 0], [1, 1], [2, 2], [3, 3]]
 * const queries = [[0.1, 0.1], [2.1, 2.1]]
 * const tree = kdTree(data)
 * const res = treeSearch(tree, queries, 2)
 * print('Nearest indices:\n' + res.indices)
 */
export function treeSearch(tree: SpaceTree, queries: MatrixLike, k: Size): Neighbours {
  const Q = rowsOf(queries, 'treeSearch')
  const results: QueryResult[] = []
  for (let i = 0; i < Q.n; i++) results.push(treeQuery(tree, Q.data.subarray(i * Q.d, (i + 1) * Q.d), k))
  return stackResults(results, k)
}
