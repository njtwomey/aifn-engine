/**
 * Minimum spanning trees (forests, on a disconnected graph): Kruskal's algorithm with union–find (Kruskal, 1956, "On
 * the shortest spanning subtree of a graph and the traveling salesman problem", Proc. AMS 7(1)) and Prim's algorithm
 * with a binary heap (Jarník, 1930; Prim, 1957, "Shortest connection networks and some generalizations", Bell Syst.
 * Tech. J. 36(6)). Edge directions are ignored.
 *
 * Both are step-through algorithms (`kruskalSteps`, `primSteps`) whose states carry the tree so far as edge indices
 * into `graph.edges` and its total weight, with an `event` saying what the last step did; `minimumSpanningTree` runs
 * either to the end. On a graph with $c$ connected components the result is a forest of $c$ trees and $n - c$ edges.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { adjacency, ints, intsOf, weightOf, type Arc, type Graph } from '../graph'
import {
  createHeap,
  heapCopy,
  heapPop,
  heapPush,
  unionFind,
  unionFindCopy,
  unionFindRoot,
  unite,
  type Heap,
  type UnionFind,
} from '../heap'

// ---------------------------------------------------------------------------------------------------------------------
// Kruskal.

/** What the last step of Kruskal's or Prim's algorithm did with an edge. */
export type SpanningEvent = 'start' | 'root' | 'accept' | 'reject' | 'done'

/** One state of Kruskal's algorithm. */
export interface KruskalState extends Status {
  /** Edge indices sorted by weight (ties in edge order); int32. */
  sorted: Tensor
  /** How many of `sorted` have been examined. */
  position: number
  /** The forest so far as disjoint sets of nodes. */
  forest: UnionFind
  /** Edges accepted into the tree, in order; int32. */
  tree: Tensor
  /** Total weight of `tree`. */
  weight: number
  /** The edge examined by the last step, $-1$ for none. */
  edge: number
  /** What the last step did with `edge`: `'start'` before any step, then `'accept'` or `'reject'`. */
  event: SpanningEvent
  /** The graph being spanned, as given. */
  graph: Graph
  /** True once one tree spans every node or every edge has been examined. */
  done: boolean
}

/** The init and step of `kruskalSteps` on the whole problem; `t` is added by the factory. */
const kruskal = {
  init: (graph: Graph): Omit<KruskalState, 't'> => {
    adjacency(graph) // validates
    const order = graph.edges
      .map((_, k) => k)
      .sort((a, b) => weightOf(graph.edges[a]) - weightOf(graph.edges[b]) || a - b)
    const forest = unionFind(graph.nodes)
    return {
      sorted: ints(order),
      position: 0,
      forest,
      tree: ints([]),
      weight: 0,
      edge: -1,
      event: 'start',
      graph,
      done: forest.count <= 1 || order.length === 0,
    }
  },
  step: (s: KruskalState): Omit<KruskalState, 't'> => {
    if (s.done) return s
    const k = s.sorted.data[s.position]
    const e = s.graph.edges[k]
    const forest = unionFindCopy(s.forest)
    const accept = unite(forest, e.from, e.to)
    const position = s.position + 1
    const done = forest.count <= 1 || position === s.sorted.shape[0]
    if (!accept) return { ...s, position, edge: k, event: 'reject', done }
    return {
      ...s,
      position,
      forest,
      tree: ints([...s.tree.data, k]),
      weight: s.weight + weightOf(e),
      edge: k,
      event: 'accept',
      done,
    }
  },
}

/**
 * Kruskal's algorithm as a traceable algorithm. Each step examines the next-lightest edge (ties broken by edge order)
 * and accepts it when its ends lie in different trees of the forest (merging them), else rejects it as closing a
 * cycle. Done when one tree remains or every edge has been examined, so a graph with $E$ edges takes at most $E$ steps.
 * The graph is validated when the algorithm is initialised (an endpoint out of range throws `DomainError`).
 *
 * @param graph The graph to span. Edge directions are ignored, and an unset weight counts as 1.
 * @returns The algorithm, whose `init` ignores its argument (run it with `undefined`): its states hold the accepted
 *   edges in `tree` and their total in `weight`.
 *
 * @example Step by step: the third edge closes a cycle and is rejected
 * // Edges [from, to, weight]: 0–1 and 1–2 of weight 1, 0–2 of weight 2, 2–3 of weight 3.
 * const edges = [[0, 1, 1], [1, 2, 1], [0, 2, 2], [2, 3, 3]].map(([from, to, weight]) => ({ from, to, weight }))
 * const g = { kind: 'graph', nodes: 4, directed: false, edges }
 * for (const k of [1, 2, 3, 4]) {
 *   const s = run(kruskalSteps(g), undefined, k)
 *   print(`step ${k}:`, s.event, 'edge', s.edge, '· tree', s.tree, '· weight', s.weight)
 * }
 */
export function kruskalSteps(graph: Graph): Algorithm<void, KruskalState> {
  const problem: Graph = graph
  return {
    name: 'kruskal',
    init: () => ({ ...kruskal.init(problem), t: 0 }),
    step: (s) => ({ ...kruskal.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Prim.

/** A queued candidate edge of Prim's algorithm: the edge and the node outside the tree it would add. */
export interface PrimCandidate {
  /** The edge's index in `graph.edges`. */
  edge: number
  /** The end of the edge that was outside the tree when it was queued. */
  to: number
}

/** One state of Prim's algorithm. */
export interface PrimState extends Status {
  /** The neighbour lists of the graph taken as undirected, as `adjacency` gives them. */
  adjacency: readonly (readonly Arc[])[]
  /** 1 for nodes in the tree; int32, length $V$. */
  inTree: Tensor
  /** Candidate edges leaving the tree, keyed by weight; entries whose `to` has joined the tree since are stale. */
  queue: Heap<PrimCandidate>
  /** Edges accepted into the tree, in order; int32. */
  tree: Tensor
  /** Total weight of `tree`. */
  weight: number
  /** Roots are tried in index order from here (a new tree starts when the queue runs dry). */
  nextRoot: number
  /** The node added by the last step (a new root or the far end of an accepted edge), $-1$ for none. */
  current: number
  /** The edge accepted or rejected (stale) by the last step, $-1$ for none. */
  edge: number
  /** What the last step did: `'start'`, `'root'`, `'accept'`, `'reject'`, or `'done'` when no node is left. */
  event: SpanningEvent
  /** True once every node is in the forest and the queue is empty. */
  done: boolean
}

/** The init and step of `primSteps` on the whole problem; `t` is added by the factory. */
const prim = {
  init: (graph: Graph): Omit<PrimState, 't'> => ({
    adjacency: adjacency({ ...graph, directed: false }),
    inTree: ints(new Int32Array(graph.nodes)),
    queue: createHeap<PrimCandidate>(),
    tree: ints([]),
    weight: 0,
    nextRoot: 0,
    current: -1,
    edge: -1,
    event: 'start',
    done: graph.nodes === 0,
  }),
  step: (s: PrimState): Omit<PrimState, 't'> => {
    if (s.done) return s
    const inTree = intsOf(s.inTree)
    const queue = heapCopy(s.queue)
    const add = (v: number) => {
      inTree[v] = 1
      for (const a of s.adjacency[v]) if (!inTree[a.to]) heapPush(queue, { edge: a.edge, to: a.to }, a.weight)
    }
    const next = heapPop(queue)
    if (!next) {
      let r = s.nextRoot
      while (r < inTree.length && inTree[r]) r++
      if (r === inTree.length) return { ...s, nextRoot: r, current: -1, edge: -1, event: 'done', done: true }
      add(r)
      return { ...s, inTree: ints(inTree), queue, nextRoot: r + 1, current: r, edge: -1, event: 'root' }
    }
    const { edge, to } = next.value
    if (inTree[to]) return { ...s, queue, current: -1, edge, event: 'reject' }
    add(to)
    return {
      ...s,
      inTree: ints(inTree),
      queue,
      tree: ints([...s.tree.data, edge]),
      weight: s.weight + next.priority,
      current: to,
      edge,
      event: 'accept',
    }
  },
}

/**
 * Prim's algorithm as a traceable algorithm. The first step makes node 0 a root (`root`); after that each step pops
 * the lightest queued edge: if its far end is outside the tree, the edge and the node join and the node's edges to
 * outside nodes are queued (`accept`); otherwise it is stale (`reject`). When the queue is empty the lowest-numbered
 * node outside the tree starts a new tree (`root`), so a disconnected graph gives a minimum spanning forest; the step
 * that finds no such node is `done`.
 *
 * @param graph The graph to span. Edge directions are ignored, and an unset weight counts as 1.
 * @returns The algorithm, whose `init` ignores its argument (run it with `undefined`): its states hold the accepted
 *   edges in `tree` and their total in `weight`.
 *
 * @example Step by step: grow from node 0, skip a stale edge
 * // Edges [from, to, weight]: 0–1 and 1–2 of weight 1, 0–2 of weight 2, 2–3 of weight 3.
 * const edges = [[0, 1, 1], [1, 2, 1], [0, 2, 2], [2, 3, 3]].map(([from, to, weight]) => ({ from, to, weight }))
 * const g = { kind: 'graph', nodes: 4, directed: false, edges }
 * for (const k of [1, 2, 3, 4, 5, 6]) {
 *   const s = run(primSteps(g), undefined, k)
 *   print(`step ${k}:`, s.event, 'node', s.current, 'edge', s.edge, '· tree', s.tree, '· weight', s.weight)
 * }
 */
export function primSteps(graph: Graph): Algorithm<void, PrimState> {
  const problem: Graph = graph
  return {
    name: 'prim',
    init: () => ({ ...prim.init(problem), t: 0 }),
    step: (s) => ({ ...prim.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

/** A minimum spanning forest. */
export interface SpanningTree {
  /** The chosen edges (indices into `graph.edges`), in the order the algorithm took them; int32. */
  edges: Tensor
  /** Their total weight. */
  weight: number
  /** Number of trees in the forest (1 for a connected graph). */
  trees: number
}

/**
 * A minimum spanning tree (a forest, on a disconnected graph) by Kruskal's algorithm (default) or Prim's, run to the
 * end. Edge directions are ignored. The two methods give the same total weight; with tied weights they may choose
 * different edges, and they list them in different orders.
 *
 * @param graph The graph to span; an unset weight counts as 1. An endpoint out of range throws `DomainError`.
 * @param options `method`: `'kruskal'` (default) or `'prim'`.
 * @returns The chosen edges, their total weight and the number of trees in the forest.
 *
 * @example Kruskal and Prim agree on the weight
 * // A square 0–1–2–3 with weights 1, 2, 1, 3 and a diagonal 0–2 of weight 4.
 * const triples = [[0, 1, 1], [1, 2, 2], [2, 3, 1], [3, 0, 3], [0, 2, 4]]
 * const edges = triples.map(([from, to, weight]) => ({ from, to, weight }))
 * const g = { kind: 'graph', nodes: 4, directed: false, edges }
 * print('kruskal:', minimumSpanningTree(g))
 * print('prim:', minimumSpanningTree(g, { method: 'prim' }))
 *
 * @example A disconnected graph gives a forest
 * // A triangle 0, 1, 2 and a separate edge 3–4.
 * const edges = [[0, 1, 2], [1, 2, 1], [0, 2, 5], [3, 4, 7]].map(([from, to, weight]) => ({ from, to, weight }))
 * const g = { kind: 'graph', nodes: 5, directed: false, edges }
 * print(minimumSpanningTree(g))
 */
export function minimumSpanningTree(graph: Graph, options: { method?: 'kruskal' | 'prim' } = {}): SpanningTree {
  const trees = (edges: Tensor) => graph.nodes - edges.shape[0]
  if ((options.method ?? 'kruskal') === 'kruskal') {
    const s = run(kruskalSteps(graph), undefined, graph.edges.length + 1)
    return { edges: s.tree, weight: s.weight, trees: trees(s.tree) }
  }
  const s = run(primSteps(graph), undefined, 2 * graph.edges.length + 2 * graph.nodes + 2)
  return { edges: s.tree, weight: s.weight, trees: trees(s.tree) }
}

/**
 * Whether nodes $a$ and $b$ are in one set of a union–find forest (a convenience over `unionFindRoot`). It mutates
 * `uf` by path halving, which changes `parent` but not the sets.
 *
 * @param uf The union–find forest, such as the `forest` of a Kruskal state.
 * @param a A node, $0 \le a < n$.
 * @param b Another node, $0 \le b < n$.
 * @returns True when $a$ and $b$ have the same root.
 *
 * @example Which nodes Kruskal's forest has joined so far
 * // After two steps on the path 0–1–2–3, nodes 0, 1 and 2 are joined and node 3 is still alone.
 * const edges = [{ from: 0, to: 1 }, { from: 1, to: 2 }, { from: 2, to: 3 }]
 * const g = { kind: 'graph', nodes: 4, directed: false, edges }
 * const { forest } = run(kruskalSteps(g), undefined, 2)
 * print('0 and 2:', sameSet(forest, 0, 2))
 * print('0 and 3:', sameSet(forest, 0, 3))
 */
export function sameSet(uf: UnionFind, a: number, b: number): boolean {
  return unionFindRoot(uf, a) === unionFindRoot(uf, b)
}
