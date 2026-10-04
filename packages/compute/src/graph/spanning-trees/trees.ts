/**
 * Minimum spanning trees (forests, on a disconnected graph): Kruskal's algorithm with union–find (Kruskal, 1956, "On
 * the shortest spanning subtree of a graph and the traveling salesman problem", Proc. AMS 7(1)) and Prim's algorithm
 * with a binary heap (Jarník, 1930; Prim, 1957, "Shortest connection networks and some generalizations", Bell Syst.
 * Tech. J. 36(6)). Edge directions are ignored.
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
  /** The edge examined by the last step, −1 for none. */
  edge: number
  event: SpanningEvent
  graph: Graph
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
 * Kruskal's algorithm as a traceable algorithm. Options: a graph. Each step examines the next-lightest edge and accepts
 * it when its ends lie in different trees of the forest (merging them), else rejects it as closing a cycle. Done when
 * one tree remains or every edge has been examined.
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
  edge: number
  to: number
}

/** One state of Prim's algorithm. */
export interface PrimState extends Status {
  adjacency: readonly (readonly Arc[])[]
  /** 1 for nodes in the tree; int32, length V. */
  inTree: Tensor
  /** Candidate edges leaving the tree, keyed by weight; entries whose `to` has joined the tree since are stale. */
  queue: Heap<PrimCandidate>
  tree: Tensor
  weight: number
  /** Roots are tried in index order from here (a new tree starts when the queue runs dry). */
  nextRoot: number
  /** The node added by the last step, −1 for none. */
  current: number
  /** The edge accepted or rejected (stale) by the last step, −1 for none. */
  edge: number
  event: SpanningEvent
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
 * Prim's algorithm as a traceable algorithm. Options: a graph. Starting from node 0, each step pops the lightest
 * queued edge: if its far end is outside the tree, the edge and the node join and the node's edges to outside nodes are
 * queued (`accept`); otherwise it is stale (`reject`). When the queue is empty the next node outside the tree starts
 * a new tree (`root`), so a disconnected graph gives a minimum spanning forest.
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

/** A minimum spanning tree (forest) by Kruskal's algorithm (default) or Prim's; edge directions are ignored. */
export function minimumSpanningTree(graph: Graph, options: { method?: 'kruskal' | 'prim' } = {}): SpanningTree {
  const trees = (edges: Tensor) => graph.nodes - edges.shape[0]
  if ((options.method ?? 'kruskal') === 'kruskal') {
    const s = run(kruskalSteps(graph), undefined, graph.edges.length + 1)
    return { edges: s.tree, weight: s.weight, trees: trees(s.tree) }
  }
  const s = run(primSteps(graph), undefined, 2 * graph.edges.length + 2 * graph.nodes + 2)
  return { edges: s.tree, weight: s.weight, trees: trees(s.tree) }
}

/** Whether nodes a and b are in one set (a convenience over `unionFindRoot`; mutates by path halving). */
export function sameSet(uf: UnionFind, a: number, b: number): boolean {
  return unionFindRoot(uf, a) === unionFindRoot(uf, b)
}
