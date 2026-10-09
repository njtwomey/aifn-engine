/**
 * Standard graphs as data graphs on the base `Graph`: chains, cycles, stars, complete and complete bipartite graphs,
 * balanced trees, lattices (4 or 8 neighbours, optionally periodic), random DAGs and random graphs (Erdős & Rényi
 * 1959, "On random graphs I"; Gilbert 1959; Watts & Strogatz 1998, "Collective dynamics of small-world networks",
 * Nature 393; Barabási & Albert 1999, "Emergence of scaling in random networks", Science 286), and graphs built from
 * points: $k$-nearest-neighbour and $\varepsilon$-ball graphs (von Luxburg 2007, "A tutorial on spectral clustering",
 * §2.2), on the distances of `aifn-compute/numerics/linalg`. Node and edge orders follow networkx where it defines
 * them.
 *
 * Every generator returns a plain `Graph` on nodes $0, \dots, n - 1$, undirected unless asked otherwise, and throws
 * `AifnError` on a node count that is not a non-negative integer or a parameter out of range. The random ones take a
 * `Stream` first and draw from named child streams of it, so the same stream gives the same graph. The point-cloud
 * graphs weight each edge by the distance between its ends.
 */

import type { Index, MatrixLike, Scalar, Size } from 'aifn-compute/foundation/contracts'
import { AifnError } from 'aifn-compute/foundation/errors'
import { child, integers, uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { pairwiseDistances, type PairwiseMetric } from 'aifn-compute/numerics/linalg'
import { fromEdges, type Edge, type Graph } from '../graph'

/** Options of the generated graphs: direction (default undirected) and display labels. */
export interface StructureOptions {
  /** True for a directed graph, whose edges run as the generator lists them (default false, undirected). */
  directed?: boolean
  /** A display name per node; there must be one per node or `fromEdges` throws `ShapeError`. */
  labels?: readonly string[]
}

/**
 * Build the graph of a generator from its node count and edges, undirected unless `o.directed` says otherwise.
 *
 * @param n The number of nodes; throws `AifnError` when it is not a non-negative integer.
 * @param edges The edges, in the order the graph keeps them.
 * @param o The caller's options: `directed` (default false) and `labels`.
 * @param where The caller's name, for error messages.
 * @returns The graph, validated by `fromEdges`.
 */
const make = (n: Size, edges: Edge[], o: StructureOptions, where: string): Graph => {
  if (!(Number.isInteger(n) && n >= 0)) throw new AifnError(where, `${where}: ${n} is not a node count`)
  return fromEdges(n, edges, { directed: o.directed ?? false, ...(o.labels ? { labels: o.labels } : {}) })
}

/**
 * The chain (path graph) through nodes $0, 1, \dots, n - 1$ in order: $n - 1$ edges $i$ to $i + 1$.
 *
 * @param n The number of nodes; 0 and 1 give a graph with no edges.
 * @param options Direction (default undirected; directed edges run from $i$ to $i + 1$) and labels.
 * @returns The chain graph.
 *
 * @example A chain of four nodes
 * const g = chainGraph(4)
 * print('nodes:', g.nodes)
 * print('edges:', g.edges.map((e) => [e.from, e.to]))
 */
export function chainGraph(n: Size, options: StructureOptions = {}): Graph {
  return make(
    n,
    Array.from({ length: Math.max(n - 1, 0) }, (_, i) => ({ from: i, to: i + 1 })),
    options,
    'chainGraph',
  )
}

/**
 * The cycle through nodes $0, 1, \dots, n - 1$ and back to 0: $n$ edges $i$ to $(i + 1) \bmod n$. Throws `AifnError`
 * when $n < 3$.
 *
 * @param n The number of nodes, at least 3.
 * @param options Direction (default undirected; directed edges run from $i$ to $(i + 1) \bmod n$) and labels.
 * @returns The cycle graph.
 *
 * @example The last edge closes the cycle
 * const g = cycleGraph(4)
 * print('edges:', g.edges.map((e) => [e.from, e.to]))
 */
export function cycleGraph(n: Size, options: StructureOptions = {}): Graph {
  if (n < 3) throw new AifnError('cycleGraph', 'cycleGraph: a cycle needs at least 3 nodes')
  return make(
    n,
    Array.from({ length: n }, (_, i) => ({ from: i, to: (i + 1) % n })),
    options,
    'cycleGraph',
  )
}

/**
 * The star: centre 0 joined to leaves $1, \dots, n$, so $n + 1$ nodes and $n$ edges.
 *
 * @param n The number of leaves (not of nodes).
 * @param options Direction (default undirected; directed edges run from the centre out) and labels.
 * @returns The star graph.
 *
 * @example Three leaves make four nodes
 * const g = starGraph(3)
 * print('nodes:', g.nodes)
 * print('edges:', g.edges.map((e) => [e.from, e.to]))
 */
export function starGraph(n: Size, options: StructureOptions = {}): Graph {
  return make(
    n + 1,
    Array.from({ length: n }, (_, i) => ({ from: 0, to: i + 1 })),
    options,
    'starGraph',
  )
}

/**
 * The complete graph on $n$ nodes: an edge for every pair $i < j$, so $n(n - 1)/2$ edges, or both directions of every
 * pair ($n(n - 1)$ edges) when directed.
 *
 * @param n The number of nodes.
 * @param options Direction (default undirected) and labels.
 * @returns The complete graph, its edges in row-major order of $(i, j)$.
 *
 * @example Undirected and directed
 * print('undirected:', completeGraph(3).edges.map((e) => [e.from, e.to]))
 * print('directed:', completeGraph(3, { directed: true }).edges.map((e) => [e.from, e.to]))
 */
export function completeGraph(n: Size, options: StructureOptions = {}): Graph {
  const edges: Edge[] = []
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) if (i !== j && (options.directed || i < j)) edges.push({ from: i, to: j })
  return make(n, edges, options, 'completeGraph')
}

/**
 * The complete bipartite graph $K_{m,n}$: each of nodes $0, \dots, m - 1$ joined to each of $m, \dots, m + n - 1$, so
 * $m + n$ nodes and $mn$ edges.
 *
 * @param m The number of nodes on the first side, numbered first.
 * @param n The number of nodes on the second side, numbered after them.
 * @param options Direction (default undirected; directed edges run from the first side to the second) and labels.
 * @returns The complete bipartite graph.
 *
 * @example $K_{2,3}$
 * const g = completeBipartiteGraph(2, 3)
 * print('nodes:', g.nodes)
 * print('edges:', g.edges.map((e) => [e.from, e.to]))
 */
export function completeBipartiteGraph(m: Size, n: Size, options: StructureOptions = {}): Graph {
  const edges: Edge[] = []
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) edges.push({ from: i, to: m + j })
  return make(m + n, edges, options, 'completeBipartiteGraph')
}

/**
 * The complete `arity`-ary tree of the given depth (depth 0 is the root alone), nodes in level order, each joined to
 * its children (directed from parent to child when `directed`). It has $(r^{d + 1} - 1)/(r - 1)$ nodes for arity $r$
 * and depth $d$ ($d + 1$ when $r = 1$), and node $i > 0$ has parent $\lfloor (i - 1)/r \rfloor$.
 *
 * @param arity The number of children $r$ of every internal node.
 * @param depth The number of levels below the root $d$.
 * @param options Direction (default undirected) and labels.
 * @returns The tree as a graph, rooted at node 0.
 *
 * @example A binary tree of depth 2
 * const g = balancedTreeGraph(2, 2)
 * print('nodes:', g.nodes)
 * print('edges:', g.edges.map((e) => [e.from, e.to]))
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

/** Options of {@link gridGraph}: the neighbourhood and wrapping, with the direction and labels of every generator. */
export interface GridOptions extends StructureOptions {
  /** 4 (rook) or 8 (king) neighbours. Default 4. */
  neighbourhood?: 4 | 8
  /** Wrap both axes (a torus). Default false. */
  periodic?: boolean
}

/**
 * The `rows` $\times$ `cols` lattice: site $(i, j)$ is node $i \cdot c + j$ ($c$ = `cols`), joined to its right and
 * lower neighbours (and the two lower diagonals with 8 neighbours), wrapping round on a torus with `periodic`.
 * Undirected by default. On a small torus a wrapped edge that would join a site to itself, or repeat a pair, is left
 * out.
 *
 * @param rows The number of rows of sites.
 * @param cols The number of columns of sites.
 * @param options The neighbourhood (4 or 8), whether to wrap, direction and labels.
 * @returns The lattice graph on `rows * cols` nodes, edges listed site by site in row-major order.
 *
 * @example A 2 by 3 grid, with 4 and 8 neighbours
 * print('4 neighbours:', gridGraph(2, 3).edges.map((e) => [e.from, e.to]))
 * print('8 neighbours:', gridGraph(2, 3, { neighbourhood: 8 }).edges.map((e) => [e.from, e.to]))
 *
 * @example Wrapping a 3 by 3 grid into a torus gives every site four neighbours
 * print('open:', gridGraph(3, 3).edges.length, 'edges')
 * print('torus:', gridGraph(3, 3, { periodic: true }).edges.length, 'edges')
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

/**
 * Uniform draws in $[0, 1)$ from a child stream, as numbers.
 *
 * @param s The parent stream; it is not advanced.
 * @param name The name of the child stream drawn from, so each use of a generator's stream is independent.
 * @param n The number of draws.
 * @returns The $n$ draws.
 */
const units = (s: Stream, name: string, n: Size): Float64Array =>
  n === 0 ? new Float64Array(0) : Float64Array.from(toFlat(uniform(child(s, name), 0, 1, { shape: [n] })))

/**
 * The Erdős–Rényi (Gilbert) random graph $G(n, p)$: each of the $n(n - 1)/2$ pairs ($n(n - 1)$ ordered pairs when
 * directed) is an edge with probability $p$, independently. Throws `AifnError` when $p$ is not in $[0, 1]$.
 *
 * @param s The random stream; the draws come from its child `'edges'`, one per pair.
 * @param n The number of nodes.
 * @param p The probability $p$ that each pair is an edge.
 * @param options Direction (default undirected) and labels.
 * @returns The random graph, edges in row-major order of their pairs.
 *
 * @example The expected number of edges is $p$ times the number of pairs
 * // 20 nodes have 190 pairs, so p = 0.1 gives 19 edges on average.
 * const g = erdosRenyiGraph(stream(0), 20, 0.1)
 * print('edges:', g.edges.length)
 * print('same stream, same graph:', erdosRenyiGraph(stream(0), 20, 0.1).edges.length)
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
 * A random DAG: each pair $i < j$ of a topological order is an edge $i \to j$ with probability $p$ (an Erdős–Rényi
 * graph oriented along the order). The order is the identity unless `shuffle` is set, so node indices are a
 * topological order by default. Always directed. Throws `AifnError` when $p$ is not in $[0, 1]$.
 *
 * @param s The random stream: the edges are drawn as by `erdosRenyiGraph` with it, and the shuffled order from its
 *   child `'order'`, so `shuffle` relabels the same edges.
 * @param n The number of nodes.
 * @param p The probability $p$ that each pair is an edge.
 * @param options Whether to shuffle the topological order, and display labels.
 * @param options.shuffle Relabel the nodes by a random permutation, so that index order is no longer topological
 *   (default false).
 * @param options.labels A display name per node.
 * @returns The directed acyclic graph.
 *
 * @example Every edge runs from a lower index to a higher one
 * const g = randomDag(stream(1), 5, 0.5)
 * print('edges:', g.edges.map((e) => [e.from, e.to]))
 * print('all forward:', g.edges.every((e) => e.from < e.to))
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
 * The Watts–Strogatz small-world graph: a ring of $n$ nodes each joined to its $k$ nearest neighbours ($k$ even), then
 * each edge $(i, i + m)$ rewired with probability $\beta$ to $(i, w)$ for a uniformly drawn $w$ that is not $i$ or
 * already joined to $i$. Edges are visited as networkx does: all edges of lag $m = 1$ round the ring, then $m = 2$,
 * and so on. An edge whose $i$ is already joined to every other node is kept. Throws `AifnError` unless $k$ is even and
 * $k < n$.
 *
 * @param s The random stream: whether each edge is rewired is drawn from its child `'rewire'`, and the new end from
 *   its children `'target'`.
 * @param n The number of nodes.
 * @param k The number of ring neighbours of each node, even (each node is joined to $k/2$ on either side).
 * @param beta The probability $\beta$ of rewiring each edge: 0 keeps the ring lattice, 1 rewires every edge.
 * @param options Direction (default undirected; directed edges run from $i$) and labels.
 * @returns The graph, with $nk/2$ edges.
 *
 * @example With $\beta = 0$ the ring is kept
 * const ring = wattsStrogatzGraph(stream(0), 6, 2, 0)
 * print('ring:', ring.edges.map((e) => [e.from, e.to]))
 * const rewired = wattsStrogatzGraph(stream(1), 6, 2, 0.3)
 * print('beta = 0.3:', rewired.edges.map((e) => [e.from, e.to]))
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
 * The Barabási–Albert preferential-attachment graph: start from a star on $m + 1$ nodes (centre 0), then join each new
 * node to $m$ distinct existing nodes drawn with probability proportional to their degree. Throws `AifnError` unless
 * $1 \le m < n$.
 *
 * @param s The random stream; each draw of a target comes from its own child `'attach'` of it.
 * @param n The number of nodes.
 * @param m The number of edges each new node brings, and the number of leaves of the starting star.
 * @param options Direction (default undirected; directed edges run from the existing node to the new one) and labels.
 * @returns The graph, with $m(n - m)$ edges.
 *
 * @example Early nodes collect the most edges
 * const g = barabasiAlbertGraph(stream(0), 30, 1)
 * const degree = Array(30).fill(0)
 * for (const e of g.edges) { degree[e.from]++; degree[e.to]++ }
 * print('edges:', g.edges.length)
 * print('degrees of nodes 0 to 9:', degree.slice(0, 10))
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
  /** The order $p$ of the Minkowski distance (default 2); used only with `metric: 'minkowski'`. */
  p?: Scalar
}

/**
 * Distances between the rows of $\Xmat$, as a count and an accessor into the flat distance matrix.
 *
 * @param x The points $\Xmat$, one per row ($n \times d$; a rank-1 input is $n$ points on a line).
 * @param o The metric and Minkowski order, passed to `pairwiseDistances`.
 * @returns `n`, the number of points, and `at(i, j)`, the distance between rows $i$ and $j$.
 */
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
 * The $k$-nearest-neighbour graph of the rows of $\Xmat$ ($n \times d$): an edge $i \to j$, weighted by the distance,
 * for each of the $k$ nearest other points $j$ of $i$ (ties to the smaller index). `mode: 'directed'` keeps the
 * relation as it is; `symmetric` (default) joins $i$ and $j$ when either is among the other's $k$ nearest; `mutual`
 * when both are. Throws `AifnError` unless $k$ is an integer with $1 \le k \le n - 1$.
 *
 * @param x The points $\Xmat$, one per row: an $n \times d$ tensor or array of rows (a rank-1 input is $n$ points on
 *   a line).
 * @param k The number of neighbours $k$ of each point.
 * @param options The distance (`metric`, and `p` for Minkowski; default Euclidean) and the `mode`: `'directed'`,
 *   `'symmetric'` (default) or `'mutual'`.
 * @returns The graph on $n$ nodes: directed with $nk$ edges listed point by point, nearest first, for `'directed'`;
 *   otherwise undirected, edges $i < j$ in row-major order.
 *
 * @example Points on a line: symmetric, mutual and directed
 * // 0, 1, 2 and 10: point 3's nearest is 2, but 2's nearest is 1.
 * const x = [[0], [1], [2], [10]]
 * print('symmetric:', kNearestNeighbourGraph(x, 1).edges.map((e) => [e.from, e.to, e.weight]))
 * print('mutual:', kNearestNeighbourGraph(x, 1, { mode: 'mutual' }).edges.map((e) => [e.from, e.to, e.weight]))
 * print('directed:', kNearestNeighbourGraph(x, 1, { mode: 'directed' }).edges.map((e) => [e.from, e.to, e.weight]))
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

/**
 * The $\varepsilon$-ball graph of the rows of $\Xmat$: $i$ and $j$ ($i < j$) joined, weighted by their distance, when
 * it is at most $\varepsilon$. Undirected.
 *
 * @param x The points $\Xmat$, one per row: an $n \times d$ tensor or array of rows (a rank-1 input is $n$ points on
 *   a line).
 * @param epsilon The radius $\varepsilon$: pairs at a distance of at most this are joined.
 * @param options The distance: `metric` (default Euclidean) and `p` for Minkowski.
 * @returns The undirected graph on $n$ nodes, edges $i < j$ in row-major order.
 *
 * @example The corners of a unit square
 * const x = [[0, 0], [1, 0], [0, 1], [1, 1]]
 * print('sides only:', epsilonBallGraph(x, 1).edges.map((e) => [e.from, e.to]))
 * print('with diagonals:', epsilonBallGraph(x, 1.5).edges.map((e) => [e.from, e.to]))
 */
export function epsilonBallGraph(x: MatrixLike, epsilon: Scalar, options: PointGraphOptions = {}): Graph {
  const { n, at } = distanceRows(x, options)
  const edges: Edge[] = []
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) if (at(i, j) <= epsilon) edges.push({ from: i, to: j, weight: at(i, j) })
  return fromEdges(n, edges, { directed: false })
}
