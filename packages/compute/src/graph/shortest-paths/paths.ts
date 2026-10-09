/**
 * Weighted shortest paths: Dijkstra's algorithm with a binary heap (Dijkstra, 1959, "A note on two problems in
 * connexion with graphs", Numer. Math. 1; the heap after Johnson, 1977), A* (Hart, Nilsson and Raphael, 1968, "A formal
 * basis for the heuristic determination of minimum cost paths", IEEE Trans. SSC 4(2)), Bellman–Ford with a
 * negative-cycle witness (Bellman, 1958; Ford, 1956), and Floyd–Warshall (Floyd, 1962, CACM 5(6)). Each is traceable.
 * Distances are Infinity where a node is unreachable and predecessors $-1$ where there is none, as in
 * `scipy.sparse.csgraph` (which uses $-9999$).
 *
 * A path's length is the sum of its edges' weights (1 for an edge without one). An undirected edge can be used in
 * either direction. Dijkstra's algorithm and A* need non-negative weights and throw `DomainError` otherwise;
 * Bellman–Ford and Floyd–Warshall accept negative weights and report a negative cycle instead of distances that would
 * be $-\infty$. A path is read back from the predecessors by `shortestPath`. $V$ is the number of nodes throughout.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { adjacency, floats, floatsOf, ints, intsOf, isDirected, weightOf, type Arc, type Graph } from '../graph'
import { createHeap, heapCopy, heapPop, heapPush, type Heap } from '../heap'

/** An edge as used by Bellman–Ford and Floyd–Warshall: directed, with its index in `graph.edges`. */
export interface DirectedArc {
  /** The node the arc leaves. */
  from: number
  /** The node the arc enters. */
  to: number
  /** The edge's weight (1 when the edge has none). */
  weight: number
  /** The index of the edge in `graph.edges` (shared by both arcs of an undirected edge). */
  edge: number
}

/**
 * The directed arcs of a graph in edge order (an undirected edge gives both directions, one after the other; an
 * undirected self-loop gives one arc). Throws as `adjacency` does for an invalid graph (an endpoint out of range, a NaN
 * weight).
 *
 * @param g The graph; it is not modified.
 * @returns One arc per directed edge and two per undirected edge, `from → to` before `to → from`.
 *
 * @example An undirected edge gives two arcs
 * const edges = [{ from: 0, to: 1, weight: 2 }, { from: 1, to: 2 }]
 * const arcs = directedArcs({ kind: 'graph', nodes: 3, edges, directed: false })
 * print('arcs =', arcs.map((a) => `${a.from} → ${a.to} (w ${a.weight}, edge ${a.edge})`))
 */
export function directedArcs(g: Graph): DirectedArc[] {
  adjacency(g) // validates the graph
  const out: DirectedArc[] = []
  g.edges.forEach((e, k) => {
    const weight = weightOf(e)
    out.push({ from: e.from, to: e.to, weight, edge: k })
    if (!isDirected(g) && e.from !== e.to) out.push({ from: e.to, to: e.from, weight, edge: k })
  })
  return out
}

/**
 * The path from `source` to `target` read from a predecessor array (as returned by the shortest-path functions), as
 * node indices; empty when the target is unreachable. For Floyd–Warshall pass the predecessor row of the source. The
 * walk back from `target` gives up (an empty path) when it meets $-1$ or takes more than $V$ steps, so a predecessor
 * cycle cannot make it loop.
 *
 * @param predecessors The predecessor of each node on a shortest path from `source`, $-1$ for none: a vector of
 *   length $V$ (a strided view, such as a row of a matrix, is read in place).
 * @param source The node the path starts from, where the walk back stops.
 * @param target The node the path ends at.
 * @returns The nodes from `source` to `target` inclusive (int32), just `[source]` when they are the same node, or an
 *   empty vector when `target` is not reached.
 *
 * @example A path from Dijkstra's predecessors
 * const triples = [[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 1], [2, 3, 5]]
 * const edges = triples.map(([from, to, weight]) => ({ from, to, weight }))
 * const { predecessor } = dijkstra({ kind: 'graph', nodes: 5, edges }, 0)
 * print('0 to 3:', shortestPath(predecessor, 0, 3))
 * // Node 4 has no edges.
 * print('0 to 4:', shortestPath(predecessor, 0, 4))
 *
 * @example A row of the Floyd–Warshall predecessors
 * const triples = [[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 1], [2, 3, 5]]
 * const edges = triples.map(([from, to, weight]) => ({ from, to, weight }))
 * const { predecessor } = floydWarshall({ kind: 'graph', nodes: 4, edges })
 * print('2 to 3:', shortestPath(slice(predecessor, 2), 2, 3))
 */
export function shortestPath(predecessors: Tensor, source: number, target: number): Tensor {
  const pred = predecessors.data
  const out: number[] = [target]
  let v = target
  while (v !== source) {
    v = pred[predecessors.offset + v * (predecessors.strides[0] ?? 1)]
    if (v < 0 || out.length > predecessors.shape[0]) return ints([])
    out.push(v)
  }
  return ints(out.reverse())
}

// ---------------------------------------------------------------------------------------------------------------------
// Dijkstra and A*.

/** Options of the single-source shortest-path algorithms. */
export interface ShortestPathOptions {
  /** The node distances are measured from. */
  source: number
  /** Stop once this node is settled (Dijkstra, A*). */
  target?: number
  /**
   * A* only: an estimate of the remaining distance to the target from each node, as a function or one value per node.
   * Admissible (never overestimating) gives shortest paths; consistent also means no node is expanded twice. Default 0.
   */
  heuristic?: ((v: number) => number) | Tensor | readonly number[]
}

/** A single-source shortest-path problem: the graph and the options. */
export type ShortestPathProblem = ShortestPathOptions & { graph: Graph }

/** One state of Dijkstra's algorithm or A*. Per-node tensors have length $V$. */
export interface DijkstraState extends Status {
  /** Each node's arcs, as `adjacency(graph)` gives them: the order edges are relaxed in. */
  adjacency: readonly (readonly Arc[])[]
  /** Best known distance from the source (final once settled); float64. */
  distance: Tensor
  /** Predecessor on the best known path, $-1$ for none; int32. */
  predecessor: Tensor
  /** 1 for settled (expanded) nodes; int32. */
  settled: Tensor
  /** Nodes in the order they were expanded (a node expanded twice by A* appears twice); int32. */
  order: Tensor
  /**
   * The priority queue: entries of a node $v$ with priority $d(v) + h(v)$ (distance when pushed plus heuristic), in
   * heap order; stale entries are skipped when popped.
   */
  queue: Heap<number>
  /** The heuristic's value at each node (0 for Dijkstra); float64. */
  heuristic: Tensor
  /** The node to stop at once settled, $-1$ for none. */
  target: number
  /** Nodes expanded so far. */
  expanded: number
  /** The node expanded by the last step, or $-1$. */
  current: number
  /** Edges (indices into `graph.edges`) that improved a distance in the last step; int32. */
  relaxed: Tensor
  /** True once the target is settled or the queue holds nothing live. */
  done: boolean
}

/**
 * The heuristic of a problem as one value per node: 0 everywhere when there is none, the function evaluated at each
 * node, or the given values. Throws `ShapeError` when an array or tensor does not have one value per node.
 *
 * @param o The problem; only `graph.nodes` and `heuristic` are read.
 * @returns A fresh array of $V$ values.
 */
function readHeuristic(o: ShortestPathProblem): Float64Array {
  const V = o.graph.nodes
  const h = o.heuristic
  if (h === undefined) return new Float64Array(V)
  if (typeof h === 'function') return Float64Array.from({ length: V }, (_, v) => h(v))
  const a = isTensor(h) ? floatsOf(h) : Float64Array.from(h)
  if (a.length !== V) throw new ShapeError('paths', 'shortest paths: the heuristic needs one value per node')
  return a
}

/**
 * Whether the queue holds an entry that is not stale: one whose node is unsettled and whose priority is still that
 * node's distance plus heuristic.
 *
 * @param queue The priority queue; read, not modified.
 * @param d The current distance of each node.
 * @param h The heuristic's value at each node.
 * @param settled 1 for each settled node, else 0.
 * @returns True when a pop could still expand a node.
 */
function live(queue: Heap<number>, d: ArrayLike<number>, h: ArrayLike<number>, settled: ArrayLike<number>): boolean {
  return queue.entries.some((e) => !settled[e.value] && e.priority === d[e.value] + h[e.value])
}

/**
 * Dijkstra's algorithm or A* (by `name`) on one problem; the heuristic is 0 for Dijkstra's. `init` throws
 * `DomainError` for a negative edge weight. Popped entries are checked against the current distances (lazy deletion),
 * and a node whose distance improves after it was settled (only under an inconsistent heuristic) is reopened.
 *
 * @param name The algorithm's name, also used in its error message: `'dijkstra'` or `'a-star'`.
 * @param o The graph, the source, and optionally the target and the heuristic.
 * @returns The algorithm.
 */
function bestFirst(name: string, o: ShortestPathProblem): Algorithm<void, DijkstraState> {
  return {
    name,
    init: () => {
      const adj = adjacency(o.graph)
      if (adj.some((arcs) => arcs.some((a) => a.weight < 0)))
        throw new DomainError(name, `${name}: edge weights must be non-negative`)
      const V = o.graph.nodes
      const h = readHeuristic(o)
      const d = new Float64Array(V).fill(Infinity)
      const queue = createHeap<number>()
      if (V > 0) {
        d[o.source] = 0
        heapPush(queue, o.source, h[o.source])
      }
      return {
        adjacency: adj,
        distance: floats(d),
        predecessor: ints(new Int32Array(V).fill(-1)),
        settled: ints(new Int32Array(V)),
        order: ints([]),
        queue,
        heuristic: floats(h),
        target: o.target ?? -1,
        expanded: 0,
        current: -1,
        relaxed: ints([]),
        t: 0,
        done: V === 0,
      }
    },
    step: (s) => {
      if (s.done) return s
      const t = s.t + 1
      const d = floatsOf(s.distance)
      const h = s.heuristic.data
      const settled = intsOf(s.settled)
      const queue = heapCopy(s.queue)
      let u = -1
      for (let e = heapPop(queue); e; e = heapPop(queue)) {
        // Lazy deletion: an entry is stale when its node was expanded since or has a better distance now.
        if (!settled[e.value] && e.priority === d[e.value] + h[e.value]) {
          u = e.value
          break
        }
      }
      if (u < 0) return { ...s, t, queue, current: -1, relaxed: ints([]), done: true }
      settled[u] = 1
      const base = {
        ...s,
        t,
        settled: ints(settled),
        order: ints([...s.order.data, u]),
        expanded: s.expanded + 1,
        current: u,
      }
      if (u === s.target) return { ...base, queue, relaxed: ints([]), done: true }
      const pred = intsOf(s.predecessor)
      const relaxed: number[] = []
      for (const arc of s.adjacency[u]) {
        const through = d[u] + arc.weight
        if (through < d[arc.to]) {
          d[arc.to] = through
          pred[arc.to] = u
          settled[arc.to] = 0 // reopens a node only under an inconsistent heuristic
          heapPush(queue, arc.to, through + h[arc.to])
          relaxed.push(arc.edge)
        }
      }
      return {
        ...base,
        settled: ints(settled),
        distance: floats(d),
        predecessor: ints(pred),
        queue,
        relaxed: ints(relaxed),
        done: !live(queue, d, h, settled),
      }
    },
    done: (s) => s.done,
  }
}

/**
 * Dijkstra's algorithm (Dijkstra, 1959) as a traceable algorithm (start: none); weights must be non-negative, or
 * `init` throws `DomainError`. Each step pops the queued node of least tentative distance (skipping stale entries),
 * settles it and relaxes its edges in adjacency order, pushing every improved neighbour. Done when the queue holds
 * nothing live, or the target is settled (its edges are then not relaxed). At most $V$ steps.
 *
 * @param graph The graph, with non-negative weights (1 where an edge has none); it is not modified.
 * @param options The `source`, and optionally the `target` to stop at.
 * @returns The algorithm, to run with `run(alg, undefined, steps)` or step through with `trace`.
 *
 * @example Two steps
 * // 0 → 1 (4), 0 → 2 (1), 2 → 1 (2), 1 → 3 (1), 2 → 3 (5).
 * const triples = [[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 1], [2, 3, 5]]
 * const edges = triples.map(([from, to, weight]) => ({ from, to, weight }))
 * const g = { kind: 'graph', nodes: 4, edges }
 * // Step 1 settles 0; step 2 settles 2, the nearest, and improves 1 and 3 through it.
 * const s = run(dijkstraSteps(g, { source: 0 }), undefined, 2)
 * print('settled in order =', s.order)
 * print('distance =', s.distance)
 * print('edges relaxed by step 2 =', s.relaxed)
 */
export function dijkstraSteps(
  graph: Graph,
  options: Omit<ShortestPathOptions, 'heuristic'>,
): Algorithm<void, DijkstraState> {
  return bestFirst('dijkstra', { graph, source: options.source, target: options.target })
}

/**
 * A* search (Hart, Nilsson and Raphael, 1968) as a traceable algorithm (start: none): Dijkstra's algorithm with the
 * queue ordered by $d(v) + h(v)$, the distance so far plus the heuristic's estimate of the distance left. Nodes may be
 * expanded again when the heuristic is admissible but not consistent. `expanded` counts the expansions, to set
 * against Dijkstra's algorithm (a zero heuristic) to see the work saved.
 *
 * @param graph The graph, with non-negative weights (1 where an edge has none); it is not modified.
 * @param options The `source`, the `target` to stop at, and the `heuristic` (default 0, which makes it Dijkstra's).
 * @returns The algorithm, to run with `run(alg, undefined, steps)` or step through with `trace`.
 *
 * @example The queue after two expansions
 * // A line 0 - 1 - 2 - 3 to the target 3, and a branch 0 - 4 - 5 leading away; h is the distance left along the line.
 * const edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 6, edges, directed: false }
 * const s = run(aStarSteps(g, { source: 0, target: 3, heuristic: [3, 2, 1, 0, 4, 5] }), undefined, 2)
 * print('expanded in order =', s.order)
 * print('queue (node, priority) =', s.queue.entries.map((e) => [e.value, e.priority]))
 */
export function aStarSteps(graph: Graph, options: ShortestPathOptions): Algorithm<void, DijkstraState> {
  return bestFirst('a-star', { graph, ...options })
}

/** Shortest-path distances and predecessors from one source. */
export interface ShortestPaths {
  /** The length of a shortest path from the source to each node, Infinity where unreachable; float64, length $V$. */
  distance: Tensor
  /** Each node's previous node on that path, $-1$ for the source and unreachable nodes; int32, length $V$. */
  predecessor: Tensor
}

/**
 * Single-source shortest paths by Dijkstra's algorithm (Dijkstra, 1959; non-negative weights); stops early at
 * `target` if given. Runs `dijkstraSteps` to the end. Throws `DomainError` for a negative edge weight.
 *
 * @param graph The graph, with non-negative weights (1 where an edge has none); it is not modified.
 * @param source The node distances are measured from.
 * @param target A node to stop at once its distance is final. The distances of nodes not settled by then are only
 *   upper bounds (Infinity where not yet reached).
 * @returns The `distance` to each node and each node's `predecessor`; `shortestPath` reads a path from them.
 *
 * @example Distances and a path
 * // 0 → 1 (4), 0 → 2 (1), 2 → 1 (2), 1 → 3 (1), 2 → 3 (5): the best route to 3 is 0 → 2 → 1 → 3.
 * const triples = [[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 1], [2, 3, 5]]
 * const edges = triples.map(([from, to, weight]) => ({ from, to, weight }))
 * const { distance, predecessor } = dijkstra({ kind: 'graph', nodes: 4, edges }, 0)
 * print('distance =', distance)
 * print('predecessor =', predecessor)
 * print('path to 3 =', shortestPath(predecessor, 0, 3))
 *
 * @example Stopping at a target leaves the rest unfinished
 * const triples = [[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 1], [2, 3, 5]]
 * const edges = triples.map(([from, to, weight]) => ({ from, to, weight }))
 * // Node 2 is settled second, before its edges are relaxed: 1 keeps the bound 4, and 3 is not reached.
 * print('distance =', dijkstra({ kind: 'graph', nodes: 4, edges }, 0, 2).distance)
 */
export function dijkstra(graph: Graph, source: number, target?: number): ShortestPaths {
  const s = run(dijkstraSteps(graph, { source, target }), undefined, graph.nodes + 1)
  return { distance: s.distance, predecessor: s.predecessor }
}

/**
 * A shortest path from `source` to `target` by A* (Hart, Nilsson and Raphael, 1968) with the given heuristic: the path
 * (empty when unreachable), its length (Infinity when unreachable) and the number of node expansions. Runs
 * `aStarSteps` to the end. The path is a shortest one when the heuristic is admissible. Throws `DomainError` for a
 * negative edge weight and `ShapeError` for a heuristic without one value per node.
 *
 * @param graph The graph, with non-negative weights (1 where an edge has none); it is not modified.
 * @param source The node the path starts from.
 * @param target The node the path ends at; the search stops once it is settled.
 * @param heuristic An estimate of the distance left to `target` from each node: a function of the node, or one value
 *   per node. Undefined gives 0 everywhere, which is Dijkstra's algorithm.
 * @returns The `path` (int32 node indices), its length `distance`, and the number of nodes `expanded`.
 *
 * @example A good heuristic saves expansions
 * // A line 0 - 1 - 2 - 3 to the target 3, and a branch 0 - 4 - 5 leading away; h is the distance left along the line.
 * const edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 6, edges, directed: false }
 * const informed = aStar(g, 0, 3, [3, 2, 1, 0, 4, 5])
 * print('path =', informed.path)
 * print('distance =', informed.distance)
 * print('expanded with h =', informed.expanded)
 * print('expanded with h = 0 =', aStar(g, 0, 3, [0, 0, 0, 0, 0, 0]).expanded)
 */
export function aStar(
  graph: Graph,
  source: number,
  target: number,
  heuristic: ShortestPathOptions['heuristic'],
): { path: Tensor; distance: number; expanded: number } {
  const s = run(aStarSteps(graph, { source, target, heuristic }), undefined, Infinity)
  return {
    path: shortestPath(s.predecessor, source, target),
    distance: s.distance.data[target],
    expanded: s.expanded,
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Bellman–Ford.

/** One state of the Bellman–Ford algorithm. */
export interface BellmanFordState extends Status {
  /** Best known distance from the source to each node, Infinity where not reached yet; float64, length $V$. */
  distance: Tensor
  /** Predecessor on the best known path, $-1$ for none; int32, length $V$. */
  predecessor: Tensor
  /** Passes over the edges made so far. */
  pass: number
  /** Edges (indices into `graph.edges`) that improved a distance in the last pass; int32. */
  relaxed: Tensor
  /** A negative cycle reachable from the source, as its nodes in order; null when there is none (or not yet found). */
  negativeCycle: Tensor | null
  /** True after a pass that changed nothing, or after the $V$-th pass. */
  done: boolean
  /** The arcs relaxed by each pass, in this order (`directedArcs` of the graph). */
  arcs: readonly DirectedArc[]
  /** The number of nodes $V$, which bounds the passes. */
  nodes: number
}

/**
 * A cycle of the predecessor graph reached by walking back from `start`, or null.
 *
 * @param pred The predecessor of each node, $-1$ for none; read, not modified.
 * @param start The node the walk starts from.
 * @returns The cycle's nodes in edge order (each node's successor follows it, the last leads back to the first), or
 *   null when the walk ends at a node with no predecessor.
 */
function predecessorCycle(pred: Int32Array, start: number): number[] | null {
  const seen = new Map<number, number>()
  const walk: number[] = []
  for (let v = start; v >= 0 && !seen.has(v); v = pred[v]) {
    seen.set(v, walk.length)
    walk.push(v)
  }
  const last = pred[walk[walk.length - 1]]
  if (last < 0 || !seen.has(last)) return null
  // walk runs backwards along predecessors: reverse the cycle so it follows the edges.
  return walk.slice(seen.get(last)).reverse()
}

/** The init and step of `bellmanFordSteps` on the whole problem; `t` is added by the factory. */
const bellmanFordPasses = {
  init: ({ graph, source }: ShortestPathProblem): Omit<BellmanFordState, 't'> => {
    const d = new Float64Array(graph.nodes).fill(Infinity)
    d[source] = 0
    return {
      distance: floats(d),
      predecessor: ints(new Int32Array(graph.nodes).fill(-1)),
      pass: 0,
      relaxed: ints([]),
      negativeCycle: null,
      done: graph.nodes === 0,
      arcs: directedArcs(graph),
      nodes: graph.nodes,
    }
  },
  step: (s: BellmanFordState): Omit<BellmanFordState, 't'> => {
    if (s.done) return s
    const d = floatsOf(s.distance)
    const pred = intsOf(s.predecessor)
    const relaxed: number[] = []
    const updated: number[] = []
    for (const e of s.arcs) {
      if (d[e.from] + e.weight < d[e.to]) {
        d[e.to] = d[e.from] + e.weight
        pred[e.to] = e.from
        relaxed.push(e.edge)
        updated.push(e.to)
      }
    }
    const pass = s.pass + 1
    let negativeCycle: Tensor | null = null
    if (updated.length > 0 && pass >= s.nodes) {
      // A V-th improving pass means a negative cycle: walking back from an updated node reaches a cycle of
      // predecessors, which is negative.
      for (const v of [...updated, ...pred.keys()]) {
        const cycle = predecessorCycle(pred, v)
        if (cycle) {
          negativeCycle = ints(cycle)
          break
        }
      }
    }
    return {
      ...s,
      distance: floats(d),
      predecessor: ints(pred),
      pass,
      relaxed: ints(relaxed),
      negativeCycle,
      done: updated.length === 0 || pass >= s.nodes,
    }
  },
}

/**
 * The Bellman–Ford algorithm (Bellman, 1958; Ford, 1956) as a traceable algorithm (start: none); weights may be
 * negative. Each step is one pass relaxing every arc of `directedArcs` in order. It is done after a pass that changes
 * nothing, or after $V - 1$ passes plus one more: if that pass still improves a distance, a negative cycle is
 * reachable and is reported in `negativeCycle`, found by walking back along the predecessors. An undirected edge of
 * negative weight is a negative cycle by itself (it can be used back and forth).
 *
 * @param graph The graph; weights may be negative. It is not modified.
 * @param options The `source` distances are measured from.
 * @returns The algorithm, to run with `run(alg, undefined, steps)` or step through with `trace`.
 *
 * @example Pass by pass
 * // 0 → 1 (4), 0 → 2 (5), 2 → 1 (−3), 1 → 3 (2): the negative edge makes 0 → 2 → 1 the shorter way to 1.
 * const edges = [[0, 1, 4], [0, 2, 5], [2, 1, -3], [1, 3, 2]].map(([from, to, weight]) => ({ from, to, weight }))
 * const g = { kind: 'graph', nodes: 4, edges }
 * const one = run(bellmanFordSteps(g, { source: 0 }), undefined, 1)
 * print('after pass 1:', one.distance, 'edges improved:', one.relaxed)
 * const two = run(bellmanFordSteps(g, { source: 0 }), undefined, 2)
 * print('after pass 2:', two.distance, 'edges improved:', two.relaxed, 'done:', two.done)
 */
export function bellmanFordSteps(graph: Graph, options: { source: number }): Algorithm<void, BellmanFordState> {
  const problem: ShortestPathProblem = { graph, ...options }
  return {
    name: 'bellman-ford',
    init: () => ({ ...bellmanFordPasses.init(problem), t: 0 }),
    step: (s) => ({ ...bellmanFordPasses.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

/**
 * Single-source shortest paths by Bellman–Ford (Bellman, 1958; Ford, 1956), which allows negative weights;
 * `negativeCycle` is set when one is reachable (distances are then meaningless). Runs `bellmanFordSteps` to the end:
 * at most $V$ passes over the edges.
 *
 * @param graph The graph; weights may be negative, and an undirected edge of negative weight is a negative cycle. It
 *   is not modified.
 * @param source The node distances are measured from.
 * @returns The `distance` to each node, each node's `predecessor`, and a reachable `negativeCycle` (its nodes in
 *   order, int32) or null.
 *
 * @example A negative edge, no negative cycle
 * const edges = [[0, 1, 4], [0, 2, 5], [2, 1, -3], [1, 3, 2]].map(([from, to, weight]) => ({ from, to, weight }))
 * const r = bellmanFord({ kind: 'graph', nodes: 4, edges }, 0)
 * print('distance =', r.distance)
 * print('path to 3 =', shortestPath(r.predecessor, 0, 3))
 * print('negative cycle =', r.negativeCycle)
 *
 * @example A negative cycle is reported
 * // 1 → 2 (−2) and 2 → 1 (1) make a cycle of weight −1, reachable from 0.
 * const edges = [[0, 1, 1], [1, 2, -2], [2, 1, 1]].map(([from, to, weight]) => ({ from, to, weight }))
 * print('negative cycle =', bellmanFord({ kind: 'graph', nodes: 3, edges }, 0).negativeCycle)
 */
export function bellmanFord(graph: Graph, source: number): ShortestPaths & { negativeCycle: Tensor | null } {
  const s = run(bellmanFordSteps(graph, { source }), undefined, graph.nodes + 1)
  return { distance: s.distance, predecessor: s.predecessor, negativeCycle: s.negativeCycle }
}

// ---------------------------------------------------------------------------------------------------------------------
// Floyd–Warshall.

/** One state of the Floyd–Warshall algorithm. */
export interface FloydWarshallState extends Status {
  /**
   * Entry $(i, j)$: the length of a shortest path from $i$ to $j$ whose intermediate nodes are among
   * $0, \dots, k - 1$; Infinity where there is none; float64, $V \times V$.
   */
  distance: Tensor
  /** Entry $(i, j)$: the node before $j$ on the best known path from $i$, $-1$ for none; int32, $V \times V$. */
  predecessor: Tensor
  /** Intermediate nodes allowed so far (the next step allows node `k`). */
  k: number
  /** True when some $d_{ii} < 0$: a negative cycle through $i$. */
  negativeCycle: boolean
  /** True once every node has been allowed as an intermediate ($k = V$). */
  done: boolean
}

/** The init and step of `floydWarshallSteps` on the whole problem; `t` is added by the factory. */
const floydWarshallSweeps = {
  init: (graph: Graph): Omit<FloydWarshallState, 't'> => {
    const V = graph.nodes
    const d = new Float64Array(V * V).fill(Infinity)
    const p = new Int32Array(V * V).fill(-1)
    for (let i = 0; i < V; i++) d[i * V + i] = 0
    for (const e of directedArcs(graph)) {
      if (e.from === e.to && e.weight >= 0) continue
      if (e.weight < d[e.from * V + e.to]) {
        d[e.from * V + e.to] = e.weight
        p[e.from * V + e.to] = e.from
      }
    }
    let negative = false
    for (let i = 0; i < V; i++) if (d[i * V + i] < 0) negative = true
    return { distance: square(d, V), predecessor: squareInts(p, V), k: 0, negativeCycle: negative, done: V === 0 }
  },
  step: (s: FloydWarshallState): Omit<FloydWarshallState, 't'> => {
    if (s.done) return s
    const V = s.distance.shape[0]
    const d = floatsOf(s.distance)
    const p = intsOf(s.predecessor)
    const k = s.k
    for (let i = 0; i < V; i++) {
      const dik = d[i * V + k]
      if (dik === Infinity) continue
      for (let j = 0; j < V; j++) {
        const through = dik + d[k * V + j]
        if (through < d[i * V + j]) {
          d[i * V + j] = through
          p[i * V + j] = p[k * V + j]
        }
      }
    }
    let negative = false
    for (let i = 0; i < V; i++) if (d[i * V + i] < 0) negative = true
    return {
      distance: square(d, V),
      predecessor: squareInts(p, V),
      k: k + 1,
      negativeCycle: negative,
      done: k + 1 >= V,
    }
  },
}

/**
 * The Floyd–Warshall algorithm (Floyd, 1962) as a traceable algorithm (start: none). It starts from the edge weights
 * (the least of parallel edges, 0 on the diagonal; a self-loop counts only when negative). Step $k$ allows node $k$
 * as an intermediate: $d_{ij} \leftarrow \min(d_{ij}, d_{ik} + d_{kj})$, taking $j$'s predecessor from row $k$ when
 * that improves. Done after $V$ steps; `negativeCycle` reports a negative diagonal.
 *
 * @param graph The graph; weights may be negative. It is not modified.
 * @returns The algorithm, to run with `run(alg, undefined, steps)` or step through with `trace`.
 *
 * @example Row 0 as intermediates are allowed
 * // 0 → 1 (4), 0 → 2 (1), 2 → 1 (2), 1 → 3 (1), 2 → 3 (5).
 * const triples = [[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 1], [2, 3, 5]]
 * const edges = triples.map(([from, to, weight]) => ({ from, to, weight }))
 * const g = { kind: 'graph', nodes: 4, edges }
 * const rowZero = (steps) => slice(run(floydWarshallSteps(g), undefined, steps).distance, 0)
 * print('edges only:', rowZero(0))
 * print('via 0 and 1:', rowZero(2))
 * print('via 0, 1 and 2:', rowZero(3))
 */
export function floydWarshallSteps(graph: Graph): Algorithm<void, FloydWarshallState> {
  const problem: Graph = graph
  return {
    name: 'floyd-warshall',
    init: () => ({ ...floydWarshallSweeps.init(problem), t: 0 }),
    step: (s) => ({ ...floydWarshallSweeps.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

/**
 * A row-major array of $V^2$ distances as a $V \times V$ float64 tensor.
 *
 * @param a The entries, row by row; kept as the tensor's data, not copied.
 * @param V The number of rows and columns.
 * @returns The $V \times V$ tensor.
 */
const square = (a: Float64Array, V: number): Tensor => fromData(a, [V, V])
/**
 * A row-major array of $V^2$ predecessors as a $V \times V$ int32 tensor.
 *
 * @param a The entries, row by row; kept as the tensor's data, not copied.
 * @param V The number of rows and columns.
 * @returns The $V \times V$ tensor.
 */
const squareInts = (a: Int32Array, V: number): Tensor => fromData(a, [V, V])

/**
 * All-pairs shortest paths by Floyd–Warshall (Floyd, 1962): $V \times V$ distances and predecessors, and a
 * negative-cycle flag. Runs `floydWarshallSteps` to the end, in $O(V^3)$ time. Negative weights are allowed; when
 * `negativeCycle` is true the distances are meaningless.
 *
 * @param graph The graph; weights may be negative. It is not modified.
 * @returns `distance` ($V \times V$, entry $(i, j)$ the length of a shortest path from $i$ to $j$, Infinity where
 *   there is none), `predecessor` ($V \times V$, entry $(i, j)$ the node before $j$ on it, $-1$ for none; pass row
 *   $i$ to `shortestPath`), and `negativeCycle`.
 *
 * @example All pairs, and one path read back
 * // 0 → 1 (4), 0 → 2 (1), 2 → 1 (2), 1 → 3 (1), 2 → 3 (5).
 * const triples = [[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 1], [2, 3, 5]]
 * const edges = triples.map(([from, to, weight]) => ({ from, to, weight }))
 * const { distance, predecessor, negativeCycle } = floydWarshall({ kind: 'graph', nodes: 4, edges })
 * print('distance =', distance)
 * print('path 0 to 3 =', shortestPath(slice(predecessor, 0), 0, 3))
 * print('negative cycle =', negativeCycle)
 */
export function floydWarshall(graph: Graph): { distance: Tensor; predecessor: Tensor; negativeCycle: boolean } {
  const s = run(floydWarshallSteps(graph), undefined, graph.nodes)
  return { distance: s.distance, predecessor: s.predecessor, negativeCycle: s.negativeCycle }
}
