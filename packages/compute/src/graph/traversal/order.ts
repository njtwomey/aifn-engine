/**
 * Topological order and cycles: Kahn's algorithm (Kahn, 1962, "Topological sorting of large networks", CACM 5(11)),
 * the depth-first order (reverse postorder; Tarjan, 1976, "Edge-disjoint spanning trees and depth-first search", Acta
 * Informatica 6; CLRS §22.4), and cycle detection from a depth-first back edge.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { adjacency, ints, intsOf, isDirected, type Arc, type Graph } from '../graph'
import { depthFirstSearch, stepBound } from './traversal'

/** One state of Kahn's algorithm. */
export interface KahnState extends Status {
  adjacency: readonly (readonly Arc[])[]
  /** In-degree of each node counting only edges from nodes not yet output; int32, length V. */
  inDegree: Tensor
  /** Nodes whose remaining in-degree is 0, waiting to be output (FIFO, front first); int32. */
  queue: Tensor
  /** Nodes output so far, in topological order; int32. */
  order: Tensor
  /** The node output by the last step, −1 for none. */
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
 * Kahn's algorithm as a traceable algorithm. Options: a directed graph. The queue starts with every node of in-degree
 * 0 in index order; each step outputs the front of the queue and decrements the in-degree of each of its out-neighbours
 * (in adjacency order), queueing those that reach 0. Parallel edges count separately; a self-loop keeps its node out.
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
 * a depth-first search over all nodes; null for an acyclic graph. Undirected graphs: a cycle of the undirected graph
 * (a self-loop or a pair of parallel edges counts).
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
  /** Nodes in topological order (every edge goes forward); with a cycle, only the nodes Kahn's algorithm could output. */
  order: Tensor
  /** A witness cycle when the graph is not a DAG, else null. */
  cycle: Tensor | null
}

/**
 * A topological order of a directed graph by Kahn's algorithm (`method: 'kahn'`, the default; ties in index order) or
 * by reverse depth-first postorder (`'depth-first'`, roots in index order). A graph with a cycle has none: `cycle`
 * holds one, and `order` is partial (Kahn) or not topological (depth-first).
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

/** True when the directed graph has no cycle (a self-loop is a cycle). */
export function isDag(graph: Graph): boolean {
  return isDirected(graph) && findCycle(graph) === null
}
