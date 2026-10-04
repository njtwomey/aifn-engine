/**
 * Weighted shortest paths: Dijkstra's algorithm with a binary heap (Dijkstra, 1959, "A note on two problems in
 * connexion with graphs", Numer. Math. 1; the heap after Johnson, 1977), A* (Hart, Nilsson and Raphael, 1968, "A formal
 * basis for the heuristic determination of minimum cost paths", IEEE Trans. SSC 4(2)), Bellman–Ford with a
 * negative-cycle witness (Bellman, 1958; Ford, 1956), and Floyd–Warshall (Floyd, 1962, CACM 5(6)). Each is traceable.
 * Distances are Infinity where a node is unreachable and predecessors −1 where there is none, as in
 * `scipy.sparse.csgraph` (which uses −9999).
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { adjacency, floats, floatsOf, ints, intsOf, isDirected, weightOf, type Arc, type Graph } from '../graph'
import { createHeap, heapCopy, heapPop, heapPush, type Heap } from '../heap'

/** An edge as used by Bellman–Ford and Floyd–Warshall: directed, with its index in `graph.edges`. */
export interface DirectedArc {
  from: number
  to: number
  weight: number
  edge: number
}

/** The directed arcs of a graph in edge order (an undirected edge gives both directions, one after the other). */
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
 * node indices; empty when the target is unreachable. For Floyd–Warshall pass the predecessor row of the source.
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

/** One state of Dijkstra's algorithm or A*. Per-node tensors have length V. */
export interface DijkstraState extends Status {
  adjacency: readonly (readonly Arc[])[]
  /** Best known distance from the source (final once settled); float64. */
  distance: Tensor
  /** Predecessor on the best known path, −1 for none; int32. */
  predecessor: Tensor
  /** 1 for settled (expanded) nodes; int32. */
  settled: Tensor
  /** Nodes in the order they were expanded (a node expanded twice by A* appears twice); int32. */
  order: Tensor
  /** The priority queue: entries (node, priority = distance + heuristic) in heap order; stale entries are skipped. */
  queue: Heap<number>
  /** The heuristic's value at each node (0 for Dijkstra); float64. */
  heuristic: Tensor
  target: number
  /** Nodes expanded so far. */
  expanded: number
  /** The node expanded by the last step, or −1. */
  current: number
  /** Edges (indices into `graph.edges`) that improved a distance in the last step; int32. */
  relaxed: Tensor
  done: boolean
}

function readHeuristic(o: ShortestPathProblem): Float64Array {
  const V = o.graph.nodes
  const h = o.heuristic
  if (h === undefined) return new Float64Array(V)
  if (typeof h === 'function') return Float64Array.from({ length: V }, (_, v) => h(v))
  const a = isTensor(h) ? floatsOf(h) : Float64Array.from(h)
  if (a.length !== V) throw new ShapeError('paths', 'shortest paths: the heuristic needs one value per node')
  return a
}

/** Whether the queue holds an entry that is not stale. */
function live(queue: Heap<number>, d: ArrayLike<number>, h: ArrayLike<number>, settled: ArrayLike<number>): boolean {
  return queue.entries.some((e) => !settled[e.value] && e.priority === d[e.value] + h[e.value])
}

/** Dijkstra's algorithm or A* (by `name`) on one problem; the heuristic is 0 for Dijkstra's. */
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
 * Dijkstra's algorithm as a traceable algorithm. Options: `{ source, target? }`; weights must be non-negative. Each
 * step pops the queued node of least tentative distance (skipping stale entries), settles it and relaxes its edges in
 * adjacency order, pushing every improved neighbour. Done when the queue holds nothing live, or the target is settled.
 */
export function dijkstraSteps(
  graph: Graph,
  options: Omit<ShortestPathOptions, 'heuristic'>,
): Algorithm<void, DijkstraState> {
  return bestFirst('dijkstra', { graph, source: options.source, target: options.target })
}

/**
 * A* search as a traceable algorithm: Dijkstra's algorithm with the queue ordered by distance + heuristic. Options:
 * `{ source, target, heuristic }`. Nodes may be expanded again when the heuristic is admissible but not consistent.
 * `expanded` counts the work saved against Dijkstra's algorithm (a zero heuristic).
 */
export function aStarSteps(graph: Graph, options: ShortestPathOptions): Algorithm<void, DijkstraState> {
  return bestFirst('a-star', { graph, ...options })
}

/** Shortest-path distances and predecessors from one source. */
export interface ShortestPaths {
  distance: Tensor
  predecessor: Tensor
}

/** Single-source shortest paths by Dijkstra's algorithm (non-negative weights); stops early at `target` if given. */
export function dijkstra(graph: Graph, source: number, target?: number): ShortestPaths {
  const s = run(dijkstraSteps(graph, { source, target }), undefined, graph.nodes + 1)
  return { distance: s.distance, predecessor: s.predecessor }
}

/**
 * A shortest path from `source` to `target` by A* with the given heuristic: the path (empty when unreachable), its
 * length (Infinity when unreachable) and the number of node expansions.
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
  distance: Tensor
  predecessor: Tensor
  /** Passes over the edges made so far. */
  pass: number
  /** Edges (indices into `graph.edges`) that improved a distance in the last pass; int32. */
  relaxed: Tensor
  /** A negative cycle reachable from the source, as its nodes in order; null when there is none (or not yet found). */
  negativeCycle: Tensor | null
  done: boolean
  arcs: readonly DirectedArc[]
  nodes: number
}

/** A cycle of the predecessor graph reached by walking back from `start`, or null. */
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
 * The Bellman–Ford algorithm as a traceable algorithm. Options: `{ source }`; weights may be negative. Each step
 * is one pass relaxing every edge in order. It is done after a pass that changes nothing, or after V − 1 passes plus one
 * more: if that pass still improves a distance, a negative cycle is reachable and is reported in `negativeCycle`.
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
 * Single-source shortest paths by Bellman–Ford; `negativeCycle` is set when one is reachable (distances are then
 * meaningless).
 */
export function bellmanFord(graph: Graph, source: number): ShortestPaths & { negativeCycle: Tensor | null } {
  const s = run(bellmanFordSteps(graph, { source }), undefined, graph.nodes + 1)
  return { distance: s.distance, predecessor: s.predecessor, negativeCycle: s.negativeCycle }
}

// ---------------------------------------------------------------------------------------------------------------------
// Floyd–Warshall.

/** One state of the Floyd–Warshall algorithm. */
export interface FloydWarshallState extends Status {
  /** Shortest distances using intermediate nodes 0…k−1, V × V. */
  distance: Tensor
  /** predecessor[i][j]: the node before j on the best known path from i, −1 for none; int32, V × V. */
  predecessor: Tensor
  /** Intermediate nodes allowed so far (the next step allows node k). */
  k: number
  /** True when some distance[i][i] < 0: a negative cycle through i. */
  negativeCycle: boolean
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
 * The Floyd–Warshall algorithm as a traceable algorithm. Options: a graph. Step k allows node k as an intermediate:
 * d[i][j] ← min(d[i][j], d[i][k] + d[k][j]). Done after V steps; `negativeCycle` reports a negative diagonal.
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

const square = (a: Float64Array, V: number): Tensor => fromData(a, [V, V])
const squareInts = (a: Int32Array, V: number): Tensor => fromData(a, [V, V])

/** All-pairs shortest paths by Floyd–Warshall: V × V distances and predecessors, and a negative-cycle flag. */
export function floydWarshall(graph: Graph): { distance: Tensor; predecessor: Tensor; negativeCycle: boolean } {
  const s = run(floydWarshallSteps(graph), undefined, graph.nodes)
  return { distance: s.distance, predecessor: s.predecessor, negativeCycle: s.negativeCycle }
}
