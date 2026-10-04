/**
 * Standard graphs as data graphs on the base `Graph`: chains, cycles, stars, complete and complete bipartite graphs,
 * balanced trees, lattices (4 or 8 neighbours, optionally periodic), random DAGs and random graphs (Erdős & Rényi
 * 1959, "On random graphs I"; Gilbert 1959; Watts & Strogatz 1998, "Collective dynamics of small-world networks",
 * Nature 393; Barabási & Albert 1999, "Emergence of scaling in random networks", Science 286), and graphs built from
 * points: k-nearest-neighbour and ε-ball graphs (von Luxburg 2007, "A tutorial on spectral clustering", §2.2), on the
 * distances of `aifn-compute/numerics/linalg`. Node and edge orders follow networkx where it defines them.
 */

import type { Index, MatrixLike, Scalar, Size } from 'aifn-compute/foundation/contracts'
import { AifnError } from 'aifn-compute/foundation/errors'
import { child, integers, uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { pairwiseDistances, type PairwiseMetric } from 'aifn-compute/numerics/linalg'
import { fromEdges, type Edge, type Graph } from '../graph'

/** Options of the generated graphs: direction (default undirected) and display labels. */
export interface StructureOptions {
  directed?: boolean
  labels?: readonly string[]
}

const make = (n: Size, edges: Edge[], o: StructureOptions, where: string): Graph => {
  if (!(Number.isInteger(n) && n >= 0)) throw new AifnError(where, `${where}: ${n} is not a node count`)
  return fromEdges(n, edges, { directed: o.directed ?? false, ...(o.labels ? { labels: o.labels } : {}) })
}

/** The chain (path graph) 0 – 1 – … – (n − 1). */
export function chainGraph(n: Size, options: StructureOptions = {}): Graph {
  return make(
    n,
    Array.from({ length: Math.max(n - 1, 0) }, (_, i) => ({ from: i, to: i + 1 })),
    options,
    'chainGraph',
  )
}

/** The cycle 0 – 1 – … – (n − 1) – 0 (n ≥ 3). */
export function cycleGraph(n: Size, options: StructureOptions = {}): Graph {
  if (n < 3) throw new AifnError('cycleGraph', 'cycleGraph: a cycle needs at least 3 nodes')
  return make(
    n,
    Array.from({ length: n }, (_, i) => ({ from: i, to: (i + 1) % n })),
    options,
    'cycleGraph',
  )
}

/** The star: centre 0 joined to leaves 1 … n (n + 1 nodes). */
export function starGraph(n: Size, options: StructureOptions = {}): Graph {
  return make(
    n + 1,
    Array.from({ length: n }, (_, i) => ({ from: 0, to: i + 1 })),
    options,
    'starGraph',
  )
}

/** The complete graph on n nodes: every pair i < j (both directions when directed). */
export function completeGraph(n: Size, options: StructureOptions = {}): Graph {
  const edges: Edge[] = []
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) if (i !== j && (options.directed || i < j)) edges.push({ from: i, to: j })
  return make(n, edges, options, 'completeGraph')
}

/** The complete bipartite graph K_{m,n}: nodes 0 … m − 1 each joined to m … m + n − 1. */
export function completeBipartiteGraph(m: Size, n: Size, options: StructureOptions = {}): Graph {
  const edges: Edge[] = []
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) edges.push({ from: i, to: m + j })
  return make(m + n, edges, options, 'completeBipartiteGraph')
}

/**
 * The complete `arity`-ary tree of the given depth (depth 0 is the root alone), nodes in level order, each joined to
 * its children (directed parent → child when `directed`).
 */
export function balancedTreeGraph(arity: Size, depth: Size, options: StructureOptions = {}): Graph {
  const n = arity === 1 ? depth + 1 : (Math.pow(arity, depth + 1) - 1) / (arity - 1)
  return make(
    n,
    Array.from({ length: Math.max(n - 1, 0) }, (_, i) => ({ from: Math.floor(i / arity), to: i + 1 })),
    options,
    'balancedTreeGraph',
  )
}

/** Options of {@link gridGraph}. */
export interface GridOptions extends StructureOptions {
  /** 4 (rook) or 8 (king) neighbours. Default 4. */
  neighbourhood?: 4 | 8
  /** Wrap both axes (a torus). Default false. */
  periodic?: boolean
}

/**
 * The rows × cols lattice: site (i, j) is node i · cols + j, joined to its right and lower neighbours (and the two
 * lower diagonals with 8 neighbours), wrapping round on a torus with `periodic`. Undirected by default.
 */
export function gridGraph(rows: Size, cols: Size, options: GridOptions = {}): Graph {
  const lags: [number, number][] = [
    [0, 1],
    [1, 0],
  ]
  if ((options.neighbourhood ?? 4) === 8) lags.push([1, 1], [1, -1])
  const seen = new Set<string>()
  const edges: Edge[] = []
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      for (const [di, dj] of lags) {
        let [a, b] = [i + di, j + dj]
        if (options.periodic) [a, b] = [((a % rows) + rows) % rows, ((b % cols) + cols) % cols]
        else if (a < 0 || a >= rows || b < 0 || b >= cols) continue
        const [u, v] = [i * cols + j, a * cols + b]
        const key = u < v ? `${u},${v}` : `${v},${u}`
        // A small torus would link a site to itself or the same pair twice.
        if (u === v || seen.has(key)) continue
        seen.add(key)
        edges.push({ from: u, to: v })
      }
  return make(rows * cols, edges, options, 'gridGraph')
}

/** Uniform draws in [0, 1) from a child stream, as numbers. */
const units = (s: Stream, name: string, n: Size): Float64Array =>
  n === 0 ? new Float64Array(0) : Float64Array.from(toFlat(uniform(child(s, name), 0, 1, { shape: [n] })))

/**
 * The Erdős–Rényi (Gilbert) random graph G(n, p): each of the n(n − 1)/2 pairs (n(n − 1) ordered pairs when
 * directed) is an edge with probability p, independently.
 */
export function erdosRenyiGraph(s: Stream, n: Size, p: Scalar, options: StructureOptions = {}): Graph {
  if (!(p >= 0 && p <= 1)) throw new AifnError('erdosRenyiGraph', 'erdosRenyiGraph: p must be in [0, 1]')
  const pairs: [number, number][] = []
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j && (options.directed || i < j)) pairs.push([i, j])
  const u = units(s, 'edges', pairs.length)
  return make(
    n,
    pairs.filter((_, k) => u[k] < p).map(([from, to]) => ({ from, to })),
    options,
    'erdosRenyiGraph',
  )
}

/**
 * A random DAG: each pair i < j of a random topological order is an edge i → j with probability p. The order is the
 * identity unless `shuffle` is set, so node indices are a topological order by default.
 */
export function randomDag(
  s: Stream,
  n: Size,
  p: Scalar,
  options: { shuffle?: boolean; labels?: readonly string[] } = {},
): Graph {
  const g = erdosRenyiGraph(s, n, p)
  let order = Array.from({ length: n }, (_, i) => i)
  if (options.shuffle) {
    const u = units(s, 'order', n)
    order = order.sort((a, b) => u[a] - u[b])
  }
  return make(
    n,
    g.edges.map((e) => ({ from: order[e.from], to: order[e.to] })),
    { directed: true, ...(options.labels ? { labels: options.labels } : {}) },
    'randomDag',
  )
}

/**
 * The Watts–Strogatz small-world graph: a ring of n nodes each joined to its k nearest neighbours (k even), then each
 * edge (i, i + m) rewired with probability β to (i, w) for a uniformly drawn w that is not i or already joined to i.
 */
export function wattsStrogatzGraph(s: Stream, n: Size, k: Size, beta: Scalar, options: StructureOptions = {}): Graph {
  if (k % 2 !== 0 || k >= n) throw new AifnError('wattsStrogatzGraph', 'wattsStrogatzGraph: k must be even and < n')
  const adj = Array.from({ length: n }, () => new Set<number>())
  const ring: [number, number][] = []
  for (let m = 1; m <= k / 2; m++)
    for (let i = 0; i < n; i++) {
      const j = (i + m) % n
      ring.push([i, j])
      adj[i].add(j)
      adj[j].add(i)
    }
  const u = units(s, 'rewire', ring.length)
  const edges = ring.map(([i, j], e) => {
    if (u[e] >= beta || adj[i].size >= n - 1) return { from: i, to: j }
    let w = integers(child(s, 'target', e), n)
    for (let tries = 1; w === i || adj[i].has(w); tries++) w = integers(child(s, 'target', e, tries), n)
    adj[i].delete(j)
    adj[j].delete(i)
    adj[i].add(w)
    adj[w].add(i)
    return { from: i, to: w }
  })
  return make(n, edges, options, 'wattsStrogatzGraph')
}

/**
 * The Barabási–Albert preferential-attachment graph: start from a star on m + 1 nodes, then join each new node to m
 * distinct existing nodes drawn with probability proportional to their degree.
 */
export function barabasiAlbertGraph(s: Stream, n: Size, m: Size, options: StructureOptions = {}): Graph {
  if (!(m >= 1 && m < n)) throw new AifnError('barabasiAlbertGraph', 'barabasiAlbertGraph: need 1 ≤ m < n')
  const edges: Edge[] = Array.from({ length: m }, (_, i) => ({ from: 0, to: i + 1 }))
  // Each node appears once per edge end, so a uniform pick from `ends` is degree-proportional.
  const ends: number[] = edges.flatMap((e) => [e.from, e.to])
  for (let v = m + 1; v < n; v++) {
    const targets = new Set<number>()
    for (let draw = 0; targets.size < m; draw++) targets.add(ends[integers(child(s, 'attach', v, draw), ends.length)])
    for (const t of targets) {
      edges.push({ from: t, to: v })
      ends.push(t, v)
    }
  }
  return make(n, edges, options, 'barabasiAlbertGraph')
}

// ── Graphs from points ───────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the point-cloud graphs. */
export interface PointGraphOptions {
  /** The distance (see `pairwiseDistances`). Default Euclidean. */
  metric?: PairwiseMetric
  /** Minkowski order. */
  p?: Scalar
}

/** Distances between the rows of X as rows of numbers. */
function distanceRows(x: MatrixLike, o: PointGraphOptions): { n: Size; at: (i: Index, j: Index) => Scalar } {
  const D: Tensor = pairwiseDistances(x, undefined, {
    ...(o.metric ? { metric: o.metric } : {}),
    ...(o.p ? { p: o.p } : {}),
  })
  const n = D.shape[0]
  const flat = toFlat(D)
  return { n, at: (i, j) => flat[i * n + j] }
}

/**
 * The k-nearest-neighbour graph of the rows of X (n × d): an edge i → j, weighted by the distance, for each of the k
 * nearest other points j of i (ties to the smaller index). `mode: 'directed'` keeps the relation as it is;
 * `symmetric` (default) joins i and j when either is among the other's k nearest; `mutual` when both are.
 */
export function kNearestNeighbourGraph(
  x: MatrixLike,
  k: Size,
  options: PointGraphOptions & { mode?: 'directed' | 'symmetric' | 'mutual' } = {},
): Graph {
  const { n, at } = distanceRows(x, options)
  if (!(Number.isInteger(k) && k >= 1 && k < n))
    throw new AifnError('kNearestNeighbourGraph', `kNearestNeighbourGraph: k = ${k} must be in 1 … n − 1`)
  const near = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => at(i, a) - at(i, b) || a - b)
      .slice(0, k),
  )
  const mode = options.mode ?? 'symmetric'
  if (mode === 'directed')
    return fromEdges(
      n,
      near.flatMap((js, i) => js.map((j) => ({ from: i, to: j, weight: at(i, j) }))),
      { directed: true },
    )
  const sets = near.map((js) => new Set(js))
  const edges: Edge[] = []
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const joined = mode === 'mutual' ? sets[i].has(j) && sets[j].has(i) : sets[i].has(j) || sets[j].has(i)
      if (joined) edges.push({ from: i, to: j, weight: at(i, j) })
    }
  return fromEdges(n, edges, { directed: false })
}

/** The ε-ball graph of the rows of X: i and j (i < j) joined, weighted by their distance, when it is at most ε. */
export function epsilonBallGraph(x: MatrixLike, epsilon: Scalar, options: PointGraphOptions = {}): Graph {
  const { n, at } = distanceRows(x, options)
  const edges: Edge[] = []
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) if (at(i, j) <= epsilon) edges.push({ from: i, to: j, weight: at(i, j) })
  return fromEdges(n, edges, { directed: false })
}
