/**
 * Topological order and cycles: Kahn's algorithm (Kahn, 1962, "Topological sorting of large networks", CACM 5(11)),
 * the depth-first order (reverse postorder; Tarjan, 1976, "Edge-disjoint spanning trees and depth-first search", Acta
 * Informatica 6; CLRS §22.4), and cycle detection from a depth-first back edge.
 *
 * A topological order lists the nodes of a directed graph so that every edge goes from an earlier node to a later one;
 * one exists exactly when the graph has no directed cycle (it is a DAG). When there is none, a cycle is returned as the
 * witness. Ties are broken by node index and adjacency order, so every result is reproducible.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { adjacency, ints, intsOf, isDirected, type Arc, type Graph } from '../graph'
import { depthFirstSearch, stepBound } from './traversal'

/** One state of Kahn's algorithm. */
export interface KahnState extends Status {
  /** Each node's out-arcs, as `adjacency(graph)` gives them: the order out-neighbours are decremented in. */
  adjacency: readonly (readonly Arc[])[]
  /** In-degree of each node counting only edges from nodes not yet output; int32, length $V$. */
  inDegree: Tensor
  /** Nodes whose remaining in-degree is 0, waiting to be output (FIFO, front first); int32. */
  queue: Tensor
  /** Nodes output so far, in topological order; int32. */
  order: Tensor
  /** The node output by the last step, $-1$ for none. */
  current: number
  /** Edges (indices into `graph.edges`) whose targets the last step decremented; int32. */
  decremented: Tensor
  /** Nodes whose in-degree the last step brought to 0 (and queued); int32. */
  released: Tensor
  /** True once the queue is empty; if `order` then misses nodes, they lie on or after a cycle. */
  done: boolean
}

/** The init and step of `kahnSteps` on the whole problem; `t` is added by the factory. */
const kahn = {
  init: (graph: Graph): Omit<KahnState, 't'> => {
    if (!isDirected(graph)) throw new DomainError('kahn', 'kahn: topological order needs a directed graph')
    const inDegree = new Int32Array(graph.nodes)
    for (const e of graph.edges) inDegree[e.to]++
    const queue: number[] = []
    inDegree.forEach((d, v) => d === 0 && queue.push(v))
    return {
      adjacency: adjacency(graph),
      inDegree: ints(inDegree),
      queue: ints(queue),
      order: ints([]),
      current: -1,
      decremented: ints([]),
      released: ints([]),
      done: queue.length === 0,
    }
  },
  step: (s: KahnState): Omit<KahnState, 't'> => {
    if (s.done) return s
    const [v, ...rest] = s.queue.data
    const inDegree = intsOf(s.inDegree)
    const decremented: number[] = []
    const released: number[] = []
    for (const arc of s.adjacency[v]) {
      decremented.push(arc.edge)
      if (--inDegree[arc.to] === 0) released.push(arc.to)
    }
    const queue = [...rest, ...released]
    return {
      ...s,
      inDegree: ints(inDegree),
      queue: ints(queue),
      order: ints([...s.order.data, v]),
      current: v,
      decremented: ints(decremented),
      released: ints(released),
      done: queue.length === 0,
    }
  },
}

/**
 * Kahn's algorithm (Kahn, 1962) as a traceable algorithm (start: none). The queue starts with every node of in-degree
 * 0 in index order; each step outputs the front of the queue and decrements the in-degree of each of its out-neighbours
 * (in adjacency order), queueing those that reach 0 at the back. Parallel edges count separately; a self-loop keeps its
 * node out. It is done when the queue empties: after $V$ steps for a DAG, earlier when a cycle holds nodes back.
 *
 * @param graph A directed graph; it is not modified. An undirected one makes `init` throw `DomainError`, so the error
 *   comes when the algorithm is first run, not from this call.
 * @returns The algorithm, to run with `run(alg, undefined, steps)` or step through with `trace`.
 *
 * @example Two steps of a small DAG
 * // 0 → 2, 1 → 2, 2 → 3 and 1 → 3: nodes 0 and 1 start with in-degree 0.
 * const edges = [[0, 2], [1, 2], [2, 3], [1, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 4, edges }
 * const s = run(kahnSteps(g), undefined, 2)
 * print('output so far =', s.order)
 * print('in-degree =', s.inDegree)
 * print('released by node 1 =', s.released)
 * print('queue =', s.queue)
 *
 * @example A cycle empties the queue early
 * // 0 → 1, 1 → 2 and 2 → 1: once 0 is output, 1 and 2 still wait on each other.
 * const edges = [[0, 1], [1, 2], [2, 1]].map(([from, to]) => ({ from, to }))
 * const s = run(kahnSteps({ kind: 'graph', nodes: 3, edges }), undefined, 10)
 * print('done =', s.done)
 * print('order =', s.order)
 * print('in-degree left =', s.inDegree)
 */
export function kahnSteps(graph: Graph): Algorithm<void, KahnState> {
  const problem: Graph = graph
  return {
    name: 'kahn',
    init: () => ({ ...kahn.init(problem), t: 0 }),
    step: (s) => ({ ...kahn.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

/**
 * A directed cycle as its nodes in order (the last has an edge back to the first), found from the first back edge of
 * a depth-first search over all nodes (in index order); null for an acyclic graph. Undirected graphs: a cycle of the
 * undirected graph (a self-loop or a pair of parallel edges counts). The cycle is the back edge closed by the
 * depth-first tree path from its ancestor end to its other end, so it is simple but not necessarily the shortest.
 *
 * @param graph The graph to search, directed or undirected; it is not modified.
 * @returns The cycle's nodes in order, starting at the ancestor end of the back edge (int32), or null when there is
 *   no cycle.
 *
 * @example A cycle and an acyclic graph
 * // 0 → 1 → 2 → 0, with 2 → 3 leading out of the cycle.
 * const edges = [[0, 1], [1, 2], [2, 0], [2, 3]].map(([from, to]) => ({ from, to }))
 * print('cycle =', findCycle({ kind: 'graph', nodes: 4, edges }))
 * // Without the edge 2 → 0 there is none.
 * const dag = [[0, 1], [1, 2], [2, 3]].map(([from, to]) => ({ from, to }))
 * print('acyclic =', findCycle({ kind: 'graph', nodes: 4, edges: dag }))
 *
 * @example An undirected triangle
 * const edges = [[0, 1], [1, 2], [2, 0]].map(([from, to]) => ({ from, to }))
 * print('cycle =', findCycle({ kind: 'graph', nodes: 3, edges, directed: false }))
 */
export function findCycle(graph: Graph): Tensor | null {
  const r = depthFirstSearch(graph)
  const k = r.edgeClass.indexOf('back')
  if (k < 0) return null
  const e = graph.edges[k]
  // A back edge v → w (w an ancestor of v, or v itself): the cycle is w … v along tree edges. In an undirected graph
  // the edge may have been met from either end; the ancestor is the end discovered first.
  const [v, w] =
    isDirected(graph) || r.discovery.data[e.from] > r.discovery.data[e.to] ? [e.from, e.to] : [e.to, e.from]
  const cycle = [v]
  for (let u = v; u !== w;) {
    u = r.parent.data[u]
    cycle.push(u)
  }
  return ints(cycle.reverse())
}

/** The result of a topological sort. */
export interface TopologicalOrder {
  /**
   * Nodes in topological order (every edge goes forward); int32. With a cycle: by Kahn's algorithm, only the nodes it
   * could output; by depth-first search, every node, in an order that is not topological.
   */
  order: Tensor
  /** A witness cycle when the graph is not a DAG (as `findCycle` gives it), else null. */
  cycle: Tensor | null
}

/**
 * A topological order of a directed graph by Kahn's algorithm (`method: 'kahn'`, the default: a first-in first-out
 * queue that starts with the sources in index order) or by reverse depth-first postorder (`'depth-first'`, roots in
 * index order). A graph with a cycle has none: `cycle` holds one, and `order` is partial (Kahn) or not topological
 * (depth-first). Throws `DomainError` for an undirected graph.
 *
 * @param graph A directed graph; it is not modified.
 * @param options `method`: `'kahn'` (default) or `'depth-first'`. The two can give different, equally valid orders.
 * @returns The `order` and, when the graph is not a DAG, a witness `cycle`.
 *
 * @example Two valid orders of one DAG
 * // 0 → 2, 1 → 2, 2 → 3 and 1 → 3.
 * const edges = [[0, 2], [1, 2], [2, 3], [1, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 4, edges }
 * print('Kahn:', topologicalSort(g).order)
 * print('depth-first:', topologicalSort(g, { method: 'depth-first' }).order)
 *
 * @example A cycle is reported with the partial order
 * // 0 → 1 → 2 → 1: Kahn's algorithm outputs 0 and stops.
 * const edges = [[0, 1], [1, 2], [2, 1]].map(([from, to]) => ({ from, to }))
 * const { order, cycle } = topologicalSort({ kind: 'graph', nodes: 3, edges })
 * print('order =', order)
 * print('cycle =', cycle)
 */
export function topologicalSort(graph: Graph, options: { method?: 'kahn' | 'depth-first' } = {}): TopologicalOrder {
  if (!isDirected(graph)) throw new DomainError('topologicalSort', 'topologicalSort: needs a directed graph')
  if ((options.method ?? 'kahn') === 'kahn') {
    const s = run(kahnSteps(graph), undefined, stepBound(graph))
    return { order: s.order, cycle: s.order.shape[0] < graph.nodes ? findCycle(graph) : null }
  }
  const r = depthFirstSearch(graph)
  const cycle = r.edgeClass.includes('back') ? findCycle(graph) : null
  return { order: ints(Array.from(r.postorder.data).reverse()), cycle }
}

/**
 * True when the graph is directed and has no cycle (a self-loop is a cycle). An undirected graph gives false, whatever
 * its edges.
 *
 * @param graph The graph to test; it is not modified.
 * @returns Whether `graph` is a directed acyclic graph.
 *
 * @example A DAG, the same with one edge reversed, and an undirected edge
 * const dag = [[0, 1], [1, 2], [0, 2]].map(([from, to]) => ({ from, to }))
 * const cyclic = [[0, 1], [1, 2], [2, 0]].map(([from, to]) => ({ from, to }))
 * print('0 → 1 → 2, 0 → 2:', isDag({ kind: 'graph', nodes: 3, edges: dag }))
 * print('0 → 1 → 2 → 0:', isDag({ kind: 'graph', nodes: 3, edges: cyclic }))
 * print('undirected 0 - 1:', isDag({ kind: 'graph', nodes: 2, edges: [{ from: 0, to: 1 }], directed: false }))
 */
export function isDag(graph: Graph): boolean {
  return isDirected(graph) && findCycle(graph) === null
}
