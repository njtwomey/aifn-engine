/**
 * Traversals as traceable algorithms: breadth-first search (Moore, 1959, "The shortest path through a maze"),
 * depth-first search with discovery and finish times and edge classification (Tarjan, 1972, "Depth-first search and
 * linear graph algorithms", SIAM J. Comput. 1(2); Cormen et al., "Introduction to Algorithms", 3rd ed., §22.2–22.3),
 * and iterative deepening (Korf, 1985, "Depth-first iterative-deepening", Artificial Intelligence 27).
 *
 * Every step examines one edge, or takes the next node from the frontier, or finishes a node, so a figure can step
 * through the frontier edge by edge. Node colours follow CLRS: 0 white (undiscovered), 1 grey (discovered, not
 * finished), 2 black (finished). Neighbours are visited in `adjacency` order, which the edge list fixes, so every
 * traversal is reproducible. $V$ is the number of nodes and $E$ the number of edges throughout.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { adjacency, floats, ints, intsOf, isDirected, type Arc, type Graph } from '../graph'
import { spanningForestOf, type SpanningTreeEdge, type SpanningTreeNode, type Tree } from '../tree'

/** Options of the traversals: the start node(s). Without `source`, every node is a root in index order. */
export interface TraversalOptions {
  /**
   * A start node, or several tried in order (each one still undiscovered when its turn comes starts a new tree).
   * Default: $0, 1, \dots, V - 1$.
   */
  source?: number | readonly number[]
}

/** A traversal problem: the graph and the options. */
export type TraversalProblem = TraversalOptions & { graph: Graph }

/**
 * The roots a traversal tries, in order: `source` as a list, or every node in index order when it is left out.
 *
 * @param o The traversal problem; only `source` and `graph.nodes` are read.
 * @returns A fresh array of node indices.
 */
const rootsOf = (o: TraversalProblem): number[] =>
  o.source === undefined
    ? Array.from({ length: o.graph.nodes }, (_, v) => v)
    : typeof o.source === 'number'
      ? [o.source]
      : [...o.source]

// ---------------------------------------------------------------------------------------------------------------------
// Breadth-first search.

/** What the last step of breadth-first search did. */
export type BreadthFirstEvent = 'start' | 'root' | 'expand' | 'discover' | 'seen' | 'done'

/** One state of breadth-first search. Per-node tensors have length $V$; int32 unless stated. */
export interface BreadthFirstState extends Status {
  /** Each node's arcs, as `adjacency(graph)` gives them: the order neighbours are examined in. */
  adjacency: readonly (readonly Arc[])[]
  /** The nodes tried as roots, in order (`source`, or every node). */
  roots: readonly number[]
  /** The next entry of `roots` to try once the queue empties. */
  nextRoot: number
  /** The frontier: a FIFO queue, front first. */
  queue: Tensor
  /** 0 white, 1 grey (discovered), 2 black (all neighbours examined). */
  colour: Tensor
  /** Parent in the breadth-first tree, $-1$ for roots and undiscovered nodes. */
  parent: Tensor
  /** The edge (index into `graph.edges`) that discovered each node, $-1$ for none. */
  parentEdge: Tensor
  /** Edges from the root of the node's tree, $-1$ when undiscovered. */
  depth: Tensor
  /** Nodes in discovery order. */
  order: Tensor
  /** The node whose neighbours are being examined, $-1$ for none. */
  current: number
  /** Position in `adjacency[current]` of the next edge to examine. */
  cursor: number
  /** The edge examined by the last step, $-1$ for none. */
  edge: number
  /** What the last step did. */
  event: BreadthFirstEvent
  /** True once every root has been tried and the queue is empty. */
  done: boolean
}

/** The init and step of `breadthFirstSteps` on the whole problem; `t` is added by the factory. */
const breadthFirst = {
  init: (o: TraversalProblem): Omit<BreadthFirstState, 't'> => {
    const V = o.graph.nodes
    return {
      adjacency: adjacency(o.graph),
      roots: rootsOf(o),
      nextRoot: 0,
      queue: ints([]),
      colour: ints(new Int32Array(V)),
      parent: ints(new Int32Array(V).fill(-1)),
      parentEdge: ints(new Int32Array(V).fill(-1)),
      depth: ints(new Int32Array(V).fill(-1)),
      order: ints([]),
      current: -1,
      cursor: 0,
      edge: -1,
      event: 'start',
      done: V === 0,
    }
  },
  step: (s: BreadthFirstState): Omit<BreadthFirstState, 't'> => {
    if (s.done) return s
    const colour = intsOf(s.colour)
    if (s.current >= 0 && s.cursor < s.adjacency[s.current].length) {
      const arc = s.adjacency[s.current][s.cursor]
      if (colour[arc.to] !== 0) return { ...s, cursor: s.cursor + 1, edge: arc.edge, event: 'seen' }
      colour[arc.to] = 1
      const parent = intsOf(s.parent)
      const parentEdge = intsOf(s.parentEdge)
      const depth = intsOf(s.depth)
      parent[arc.to] = s.current
      parentEdge[arc.to] = arc.edge
      depth[arc.to] = depth[s.current] + 1
      return {
        ...s,
        colour: ints(colour),
        parent: ints(parent),
        parentEdge: ints(parentEdge),
        depth: ints(depth),
        order: ints([...s.order.data, arc.to]),
        queue: ints([...s.queue.data, arc.to]),
        cursor: s.cursor + 1,
        edge: arc.edge,
        event: 'discover',
      }
    }
    if (s.current >= 0) colour[s.current] = 2
    if (s.queue.shape[0] > 0) {
      const [next, ...rest] = s.queue.data
      return { ...s, colour: ints(colour), queue: ints(rest), current: next, cursor: 0, edge: -1, event: 'expand' }
    }
    let k = s.nextRoot
    while (k < s.roots.length && colour[s.roots[k]] !== 0) k++
    if (k === s.roots.length)
      return { ...s, colour: ints(colour), nextRoot: k, current: -1, edge: -1, event: 'done', done: true }
    const r = s.roots[k]
    colour[r] = 1
    const depth = intsOf(s.depth)
    depth[r] = 0
    return {
      ...s,
      colour: ints(colour),
      depth: ints(depth),
      order: ints([...s.order.data, r]),
      queue: ints([r]),
      nextRoot: k + 1,
      current: -1,
      edge: -1,
      event: 'root',
    }
  },
}

/**
 * Breadth-first search as a traceable algorithm (start: none). A step either examines the next edge of the current
 * node (discovering its other end if white: event `discover`, else `seen`), or, when the current node has no edges
 * left, blackens it and dequeues the next (`expand`), or, when the queue is empty, starts a tree at the next
 * undiscovered root (`root`), or ends the search (`done`). Neighbours are visited in `adjacency` order. A directed
 * graph is searched along its edges, an undirected one in both directions. At most $2E + 2V + 1$ steps.
 *
 * @param graph The graph to search; it is not modified.
 * @param options The start node(s) (default: every node in index order, giving a breadth-first forest).
 * @returns The algorithm, to run with `run(alg, undefined, steps)` or step through with `trace`.
 *
 * @example The frontier after the first few steps
 * // A square 0-1-3-2-0 with a tail 3-4, undirected.
 * const edges = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges, directed: false }
 * // Step 1 starts the tree at 0, step 2 takes 0 off the queue, steps 3 and 4 discover 1 and 2.
 * const s = run(breadthFirstSteps(g, { source: 0 }), undefined, 4)
 * print('event =', s.event)
 * print('queue =', s.queue)
 * print('discovered =', s.order)
 *
 * @example Run to the end
 * const edges = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges, directed: false }
 * const s = run(breadthFirstSteps(g, { source: 0 }), undefined, 100)
 * print('steps =', s.t)
 * print('depth =', s.depth)
 * print('parent =', s.parent)
 */
export function breadthFirstSteps(graph: Graph, options: TraversalOptions = {}): Algorithm<void, BreadthFirstState> {
  const problem: TraversalProblem = { graph, ...options }
  return {
    name: 'breadth-first-search',
    init: () => ({ ...breadthFirst.init(problem), t: 0 }),
    step: (s) => ({ ...breadthFirst.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

/**
 * Enough steps for any traversal of `g` to finish: two per edge (one per end), three per node, plus two of slack.
 *
 * @param g The graph to be traversed; only its node and edge counts are read.
 * @returns $2E + 3V + 2$.
 */
export const stepBound = (g: Graph): number => 2 * g.edges.length + 3 * g.nodes + 2

/** The result of breadth-first search. */
export interface BreadthFirstResult {
  /** Nodes in discovery (visit) order; int32. */
  order: Tensor
  /** Edges from the root, $-1$ when unreached; int32, length $V$. */
  depth: Tensor
  /** Parent in the breadth-first tree, $-1$ for roots and unreached nodes; int32, length $V$. */
  parent: Tensor
  /** Nodes grouped by depth, in discovery order: `layers[d]` holds the nodes at depth $d$; int32 vectors. */
  layers: Tensor[]
  /**
   * The breadth-first forest as `Tree`s over graph vertices, one per root in visit order, siblings in discovery order
   * (see `spanningTreeOf`). For a partial search use `spanningForestOf(graph, { parent, parentEdge, order })` on a
   * state.
   */
  trees: Tree<SpanningTreeNode, SpanningTreeEdge>[]
}

/**
 * Breadth-first search from `source` (default: every node in turn): visit order, depths, parents, layers and the
 * breadth-first forest. Runs `breadthFirstSteps` to the end. Depths are numbers of edges, so a node's depth is its
 * unweighted distance from the root of its tree.
 *
 * @param graph The graph to search (directed edges are followed forwards only); it is not modified.
 * @param source The start node, or several tried in order, each still undiscovered one starting a new tree. Left out,
 *   every node is tried in index order and every node is reached.
 * @returns The visit `order`, `depth` and `parent` of each node, the `layers` by depth and the `trees`.
 *
 * @example Layers of a small graph
 * // A square 0-1-3-2-0 with a tail 3-4, undirected.
 * const edges = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges, directed: false }
 * const r = breadthFirstSearch(g, 0)
 * print('order =', r.order)
 * print('depth =', r.depth)
 * print('parent =', r.parent)
 * print('layers =', r.layers)
 *
 * @example Directed edges are followed forwards only
 * // 0 → 1 → 2, and 3 → 0: from 0, node 3 is never reached.
 * const edges = [[0, 1], [1, 2], [3, 0]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 4, edges }
 * print('depth from 0 =', breadthFirstSearch(g, 0).depth)
 * print('depth over all roots =', breadthFirstSearch(g).depth)
 */
export function breadthFirstSearch(graph: Graph, source?: number | readonly number[]): BreadthFirstResult {
  const s = run(breadthFirstSteps(graph, { source }), undefined, stepBound(graph))
  const layers: number[][] = []
  for (const v of s.order.data) (layers[s.depth.data[v]] ??= []).push(v)
  const trees = spanningForestOf(graph, { parent: s.parent.data, parentEdge: s.parentEdge.data, order: s.order.data })
  return { order: s.order, depth: s.depth, parent: s.parent, layers: layers.map(ints), trees }
}

/**
 * Unweighted single-source shortest paths by breadth-first search: `distance` (float64, number of edges, Infinity
 * when unreachable) and `predecessor` (int32, $-1$ for none), in the form the weighted solvers of
 * `aifn-compute/graph/shortest-paths` return. Edge weights are ignored.
 *
 * @param graph The graph to search (directed edges are followed forwards only); it is not modified.
 * @param source The node the distances are measured from.
 * @returns `distance`, the number of edges on a shortest path to each node, and `predecessor`, each node's previous
 *   node on one such path (the breadth-first parent); both of length $V$.
 *
 * @example Hop counts on a path with a shortcut
 * // 0 → 1 → 2 → 3 and a shortcut 0 → 2; node 4 has no edges.
 * const edges = [[0, 1], [1, 2], [2, 3], [0, 2]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges }
 * const { distance, predecessor } = unweightedShortestPaths(g, 0)
 * print('distance =', distance)
 * print('predecessor =', predecessor)
 */
export function unweightedShortestPaths(graph: Graph, source: number): { distance: Tensor; predecessor: Tensor } {
  const r = breadthFirstSearch(graph, source)
  return { distance: floats(Array.from(r.depth.data, (d) => (d < 0 ? Infinity : d))), predecessor: r.parent }
}

// ---------------------------------------------------------------------------------------------------------------------
// Depth-first search.

/**
 * The class of an edge in a depth-first forest (CLRS §22.3): `tree` (discovered a node), `back` (to an ancestor still
 * on the stack, including self-loops), `forward` (to a finished descendant), `cross` (anything else). Undirected
 * graphs have only tree and back edges.
 */
export type EdgeClass = 'tree' | 'back' | 'forward' | 'cross'

/** What the last step of depth-first search did: started a tree, classified an edge, re-met an edge, or finished. */
export type DepthFirstEvent = 'start' | 'root' | EdgeClass | 'revisit' | 'finish' | 'done'

/** One state of depth-first search. Per-node tensors have length $V$; int32. */
export interface DepthFirstState extends Status {
  /** Each node's arcs, as `adjacency(graph)` gives them: the order neighbours are examined in. */
  adjacency: readonly (readonly Arc[])[]
  /** Whether the graph is directed (an undirected one has only tree and back edges). */
  directed: boolean
  /** The nodes tried as roots, in order (`source`, or every node). */
  roots: readonly number[]
  /** The next entry of `roots` to try once the stack empties. */
  nextRoot: number
  /** The explicit stack of grey nodes, bottom first (the current path from the root). */
  stack: Tensor
  /** For each stack entry, the position in its adjacency list of the next edge to examine. */
  cursor: Tensor
  /** 0 white, 1 grey (on the stack), 2 black (finished). */
  colour: Tensor
  /** Parent in the depth-first forest, $-1$ for roots and undiscovered nodes. */
  parent: Tensor
  /** The tree edge (index into `graph.edges`) that discovered each node, $-1$ for none. */
  parentEdge: Tensor
  /** Discovery times $1, 2, \dots$ as in CLRS (one clock for discovery and finish); $-1$ until set. */
  discovery: Tensor
  /** Finish times, on the same clock as `discovery`; $-1$ until set. */
  finish: Tensor
  /** The clock: the last time handed out. */
  time: number
  /** Nodes in discovery order. */
  preorder: Tensor
  /** Nodes in finish order. */
  postorder: Tensor
  /** Each edge's class once examined (null before), indexed like `graph.edges`. */
  edgeClass: readonly (EdgeClass | null)[]
  /** The node the last step acted on: the new root, the node whose edge was examined, or the node finished. */
  current: number
  /** The edge examined by the last step, $-1$ for none. */
  edge: number
  /** What the last step did: an edge's class when it classified one. */
  event: DepthFirstEvent
  /** True once every root has been tried and the stack is empty. */
  done: boolean
}

/** @internal Depth-first search's init and step, shared with Kosaraju; callers add `t`. */
export const depthFirst = {
  init: (o: TraversalProblem): Omit<DepthFirstState, 't'> => {
    const V = o.graph.nodes
    const none = () => ints(new Int32Array(V).fill(-1))
    return {
      adjacency: adjacency(o.graph),
      directed: isDirected(o.graph),
      roots: rootsOf(o),
      nextRoot: 0,
      stack: ints([]),
      cursor: ints([]),
      colour: ints(new Int32Array(V)),
      parent: none(),
      parentEdge: none(),
      discovery: none(),
      finish: none(),
      time: 0,
      preorder: ints([]),
      postorder: ints([]),
      edgeClass: new Array<EdgeClass | null>(o.graph.edges.length).fill(null),
      current: -1,
      edge: -1,
      event: 'start',
      done: V === 0,
    }
  },
  step: (s: DepthFirstState): Omit<DepthFirstState, 't'> => {
    if (s.done) return s
    const colour = intsOf(s.colour)
    const discovery = intsOf(s.discovery)
    const n = s.stack.shape[0]
    if (n === 0) {
      let k = s.nextRoot
      while (k < s.roots.length && colour[s.roots[k]] !== 0) k++
      if (k === s.roots.length) return { ...s, nextRoot: k, current: -1, edge: -1, event: 'done', done: true }
      const r = s.roots[k]
      colour[r] = 1
      discovery[r] = s.time + 1
      return {
        ...s,
        nextRoot: k + 1,
        stack: ints([r]),
        cursor: ints([0]),
        colour: ints(colour),
        discovery: ints(discovery),
        time: s.time + 1,
        preorder: ints([...s.preorder.data, r]),
        current: r,
        edge: -1,
        event: 'root',
      }
    }
    const v = s.stack.data[n - 1]
    const cursor = intsOf(s.cursor)
    const arcs = s.adjacency[v]
    if (cursor[n - 1] < arcs.length) {
      const arc = arcs[cursor[n - 1]++]
      const w = arc.to
      const base = { ...s, cursor: ints(cursor), current: v, edge: arc.edge }
      if (s.edgeClass[arc.edge] !== null || (!s.directed && arc.edge === s.parentEdge.data[v]))
        return { ...base, event: 'revisit' }
      const edgeClass = [...s.edgeClass]
      if (colour[w] === 0) {
        edgeClass[arc.edge] = 'tree'
        colour[w] = 1
        discovery[w] = s.time + 1
        const parent = intsOf(s.parent)
        const parentEdge = intsOf(s.parentEdge)
        parent[w] = v
        parentEdge[w] = arc.edge
        return {
          ...base,
          stack: ints([...s.stack.data, w]),
          cursor: ints([...cursor, 0]),
          colour: ints(colour),
          discovery: ints(discovery),
          parent: ints(parent),
          parentEdge: ints(parentEdge),
          time: s.time + 1,
          preorder: ints([...s.preorder.data, w]),
          edgeClass,
          event: 'tree',
        }
      }
      const cls: EdgeClass = colour[w] === 1 ? 'back' : discovery[v] < discovery[w] ? 'forward' : 'cross'
      edgeClass[arc.edge] = cls
      return { ...base, edgeClass, event: cls }
    }
    colour[v] = 2
    const finish = intsOf(s.finish)
    finish[v] = s.time + 1
    return {
      ...s,
      stack: ints(s.stack.data.slice(0, n - 1)),
      cursor: ints(cursor.slice(0, n - 1)),
      colour: ints(colour),
      finish: ints(finish),
      time: s.time + 1,
      postorder: ints([...s.postorder.data, v]),
      current: v,
      edge: -1,
      event: 'finish',
    }
  },
}

/**
 * Depth-first search as a traceable algorithm (start: none), iterative with an explicit stack (so it visits nodes
 * exactly as the recursive version does). A step examines the next edge of the node on top of the stack and
 * classifies it (pushing its other end on a tree edge), or finishes that node when it has no edges left, or starts a
 * tree at the next white root, or ends the search (`done`). In an undirected graph the edge back to the parent, and an
 * edge already classified from its other end, are `revisit`s.
 *
 * @param graph The graph to search; it is not modified.
 * @param options The start node(s) (default: every node in index order, giving a depth-first forest).
 * @returns The algorithm, to run with `run(alg, undefined, steps)` or step through with `trace`.
 *
 * @example Stop at the first back edge
 * // 0 → 1 → 2 → 0 is a cycle; 0 → 2 and 3 → 1 are extra edges.
 * const edges = [[0, 1], [1, 2], [2, 0], [0, 2], [3, 1]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 4, edges }
 * // Step 1 starts at 0, steps 2 and 3 push 1 and 2, step 4 meets edge 2 → 0.
 * const s = run(depthFirstSteps(g), undefined, 4)
 * print('event =', s.event)
 * print('edge =', s.edge)
 * print('stack =', s.stack)
 * print('discovery =', s.discovery)
 */
export function depthFirstSteps(graph: Graph, options: TraversalOptions = {}): Algorithm<void, DepthFirstState> {
  const problem: TraversalProblem = { graph, ...options }
  return {
    name: 'depth-first-search',
    init: () => ({ ...depthFirst.init(problem), t: 0 }),
    step: (s) => ({ ...depthFirst.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

/** The result of depth-first search. */
export interface DepthFirstResult {
  /** The reached nodes in discovery order; int32. */
  preorder: Tensor
  /** The reached nodes in finish order; int32. */
  postorder: Tensor
  /**
   * Discovery times (on one clock with `finish`, running $1, \dots, 2V$ over the whole forest), $-1$ for nodes not
   * reached; int32, length $V$.
   */
  discovery: Tensor
  /** Finish times, on the clock of `discovery`, $-1$ for nodes not reached; int32, length $V$. */
  finish: Tensor
  /** Parent in the depth-first forest, $-1$ for roots and unreached nodes; int32, length $V$. */
  parent: Tensor
  /** Each edge's class, null for edges never examined (outside the reached part); indexed like `graph.edges`. */
  edgeClass: (EdgeClass | null)[]
  /** The depth-first forest as `Tree`s over graph vertices, one per root, siblings in discovery order. */
  trees: Tree<SpanningTreeNode, SpanningTreeEdge>[]
}

/**
 * Depth-first search from `source` (default: every node in index order, giving a depth-first forest): preorder,
 * postorder, discovery and finish times, parents, edge classes (CLRS §22.3) and the depth-first forest. Runs
 * `depthFirstSteps` to the end.
 *
 * @param graph The graph to search (directed edges are followed forwards only); it is not modified.
 * @param source The start node, or several tried in order, each still undiscovered one starting a new tree. Left out,
 *   every node is tried in index order and every node is reached.
 * @returns The orders, times, parents, edge classes and trees; see `DepthFirstResult`.
 *
 * @example Times and edge classes of a directed graph
 * // 0 → 1 → 2 → 0 is a cycle; 0 → 2 and 3 → 1 are extra edges.
 * const edges = [[0, 1], [1, 2], [2, 0], [0, 2], [3, 1]].map(([from, to]) => ({ from, to }))
 * const r = depthFirstSearch({ kind: 'graph', nodes: 4, edges })
 * print('preorder =', r.preorder)
 * print('postorder =', r.postorder)
 * print('discovery =', r.discovery)
 * print('finish =', r.finish)
 * print('edge classes =', r.edgeClass)
 *
 * @example An undirected graph has only tree and back edges
 * // A triangle 0-1-2 with a pendant node 3 on 2.
 * const edges = [[0, 1], [1, 2], [2, 0], [2, 3]].map(([from, to]) => ({ from, to }))
 * const r = depthFirstSearch({ kind: 'graph', nodes: 4, edges, directed: false }, 0)
 * print('parent =', r.parent)
 * print('edge classes =', r.edgeClass)
 */
export function depthFirstSearch(graph: Graph, source?: number | readonly number[]): DepthFirstResult {
  const s = run(depthFirstSteps(graph, { source }), undefined, stepBound(graph))
  return {
    preorder: s.preorder,
    postorder: s.postorder,
    discovery: s.discovery,
    finish: s.finish,
    parent: s.parent,
    edgeClass: [...s.edgeClass],
    trees: spanningForestOf(graph, { parent: s.parent.data, parentEdge: s.parentEdge.data, order: s.preorder.data }),
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Iterative deepening.

/** Options of iterative deepening. */
export interface IterativeDeepeningOptions {
  /** The node every pass starts from. */
  source: number
  /** Stop when this node is reached. Without a target, deepen until no node is cut off (all reachable depths known). */
  target?: number
  /** The largest depth limit tried (default $V - 1$, beyond which no simple path reaches). */
  maxDepth?: number
}

/**
 * What the last step of iterative deepening did: `deepen` (started a pass with a limit one larger), `visit` (entered a
 * node), `skip` (passed over an edge to a node already reached no deeper in this pass), `cutoff` (left a node at the
 * limit that has edges), `backtrack` (left a node with no edges left), `found` (entered the target), `done`.
 */
export type IterativeDeepeningEvent = 'start' | 'deepen' | 'visit' | 'skip' | 'cutoff' | 'backtrack' | 'found' | 'done'

/** One state of iterative deepening. */
export interface IterativeDeepeningState extends Status {
  /** Each node's arcs, as `adjacency(graph)` gives them: the order neighbours are examined in. */
  adjacency: readonly (readonly Arc[])[]
  /** The node every pass starts from. */
  source: number
  /** The node searched for, $-1$ for none. */
  target: number
  /** The largest depth limit that will be tried. */
  maxDepth: number
  /** The current depth limit ($0, 1, 2, \dots$). */
  limit: number
  /** The current path from the source (a stack, bottom first); a node's depth is its position. */
  stack: Tensor
  /** For each stack entry, the position in its adjacency list of the next edge to examine. */
  cursor: Tensor
  /** Least depth at which each node has been reached in this pass, $-1$ if not yet; int32, length $V$. */
  best: Tensor
  /**
   * True when some node at the limit in this pass has at least one edge, wherever it leads (so a deeper pass may
   * reach more).
   */
  cutoff: boolean
  /** Node visits over all passes, counting repeats and the source at the start of each pass. */
  expanded: number
  /** True once the target has been reached; `stack` is then the path to it. */
  found: boolean
  /** The node the last step acted on: the node entered, the node whose edge was skipped, or the node left. */
  current: number
  /** The edge the last step followed or skipped, $-1$ for none. */
  edge: number
  /** What the last step did. */
  event: IterativeDeepeningEvent
  /** True once the target is found, or a pass ends with no cutoff, or the pass at `maxDepth` ends. */
  done: boolean
}

/** The init and step of `iterativeDeepeningSteps` on the whole problem; `t` is added by the factory. */
const iterativeDeepeningSearch = {
  init: (o: IterativeDeepeningOptions & { graph: Graph }): Omit<IterativeDeepeningState, 't'> => {
    const V = o.graph.nodes
    const best = new Int32Array(V).fill(-1)
    best[o.source] = 0
    const found = o.target === o.source
    return {
      adjacency: adjacency(o.graph),
      source: o.source,
      target: o.target ?? -1,
      maxDepth: o.maxDepth ?? Math.max(V - 1, 0),
      limit: 0,
      stack: ints([o.source]),
      cursor: ints([0]),
      best: ints(best),
      cutoff: false,
      expanded: 1,
      found,
      current: o.source,
      edge: -1,
      event: found ? 'found' : 'start',
      done: found,
    }
  },
  step: (s: IterativeDeepeningState): Omit<IterativeDeepeningState, 't'> => {
    if (s.done) return s
    const n = s.stack.shape[0]
    if (n === 0) {
      if (!s.cutoff || s.limit >= s.maxDepth) return { ...s, current: -1, edge: -1, event: 'done', done: true }
      const best = new Int32Array(s.best.shape[0]).fill(-1)
      best[s.source] = 0
      return {
        ...s,
        limit: s.limit + 1,
        stack: ints([s.source]),
        cursor: ints([0]),
        best: ints(best),
        cutoff: false,
        expanded: s.expanded + 1,
        current: s.source,
        edge: -1,
        event: 'deepen',
      }
    }
    const v = s.stack.data[n - 1]
    const depth = n - 1
    const arcs = s.adjacency[v]
    const cursor = intsOf(s.cursor)
    if (depth === s.limit || cursor[n - 1] >= arcs.length) {
      const cut = depth === s.limit && arcs.length > 0
      return {
        ...s,
        stack: ints(s.stack.data.slice(0, n - 1)),
        cursor: ints(cursor.slice(0, n - 1)),
        cutoff: s.cutoff || cut,
        current: v,
        edge: -1,
        event: cut ? 'cutoff' : 'backtrack',
      }
    }
    const arc = arcs[cursor[n - 1]++]
    const known = s.best.data[arc.to]
    if (known >= 0 && known <= depth + 1)
      return { ...s, cursor: ints(cursor), current: v, edge: arc.edge, event: 'skip' }
    const best = intsOf(s.best)
    best[arc.to] = depth + 1
    const found = arc.to === s.target
    return {
      ...s,
      stack: ints([...s.stack.data, arc.to]),
      cursor: ints([...cursor, 0]),
      best: ints(best),
      expanded: s.expanded + 1,
      found,
      current: arc.to,
      edge: arc.edge,
      event: found ? 'found' : 'visit',
      done: found,
    }
  },
}

/**
 * Iterative deepening as a traceable algorithm (Korf, 1985; start: none): depth-limited depth-first searches with
 * limits $0, 1, 2, \dots$ until the target is reached, a pass cuts nothing off, or the pass at `maxDepth` ends.
 * Within a pass a node is re-entered only when reached by a shorter path than before in that pass, which keeps each
 * pass polynomial on graphs with cycles and makes the first path found to the target a shortest one. A step visits or
 * skips one edge, cuts off or backtracks from one node, or starts a deeper pass.
 *
 * @param graph The graph to search (directed edges are followed forwards only); it is not modified.
 * @param options The `source`, and optionally the `target` to stop at and the largest depth limit `maxDepth`.
 * @returns The algorithm, to run with `run(alg, undefined, steps)` or step through with `trace`.
 *
 * @example The second pass, part way through
 * // Two routes from 0 to 3: 0-1-2-3 and 0-4-3, undirected.
 * const edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges, directed: false }
 * // Pass 0 cuts off at 0; pass 1 enters 1, cuts it off, and enters 4.
 * const s = run(iterativeDeepeningSteps(g, { source: 0, target: 3 }), undefined, 5)
 * print('limit =', s.limit)
 * print('event =', s.event)
 * print('stack =', s.stack)
 */
export function iterativeDeepeningSteps(
  graph: Graph,
  options: IterativeDeepeningOptions,
): Algorithm<void, IterativeDeepeningState> {
  const problem: IterativeDeepeningOptions & { graph: Graph } = { graph, ...options }
  return {
    name: 'iterative-deepening',
    init: () => ({ ...iterativeDeepeningSearch.init(problem), t: 0 }),
    step: (s) => ({ ...iterativeDeepeningSearch.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

/** The result of iterative deepening. */
export interface IterativeDeepeningResult {
  /** Whether the target was reached (false when no target was given). */
  found: boolean
  /** A shortest path from the source to the target (empty when not found); int32. */
  path: Tensor
  /** The last depth limit tried. */
  limit: number
  /** Node visits over all passes, counting repeats. */
  expanded: number
  /**
   * Depth of each node in the last pass, $-1$ where not reached; with no target, the breadth-first depths. When the
   * target is found the pass stopped early, so nodes it had not reached yet are $-1$.
   */
  depth: Tensor
}

/**
 * Iterative-deepening search from `source`, to `target` if given (Korf, 1985). Runs `iterativeDeepeningSteps` to the
 * end with the default `maxDepth` of $V - 1$. The path found is a shortest one in number of edges, at the cost of
 * revisiting the shallow nodes in every pass, which `expanded` counts.
 *
 * @param graph The graph to search (directed edges are followed forwards only); it is not modified.
 * @param source The node every pass starts from.
 * @param target The node to find. Left out, the search deepens until no node is cut off, and `depth` is then the
 *   breadth-first depth of every node reachable from `source`.
 * @returns Whether the target was found, the path to it, the last limit, the number of visits and the depths.
 *
 * @example A shortest path, found on the second deepening
 * // Two routes from 0 to 3: 0-1-2-3 and 0-4-3, undirected.
 * const edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges, directed: false }
 * const r = iterativeDeepening(g, 0, 3)
 * print('path =', r.path)
 * print('limit =', r.limit)
 * print('expanded =', r.expanded)
 *
 * @example Without a target, the depths of every reachable node
 * const edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges, directed: false }
 * print('depth =', iterativeDeepening(g, 0).depth)
 * print('breadth-first depth =', breadthFirstSearch(g, 0).depth)
 */
export function iterativeDeepening(graph: Graph, source: number, target?: number): IterativeDeepeningResult {
  const bound = (graph.nodes + 1) * (stepBound(graph) + 2) * Math.max(graph.nodes, 1)
  const s = run(iterativeDeepeningSteps(graph, { source, target }), undefined, bound)
  return { found: s.found, path: s.found ? s.stack : ints([]), limit: s.limit, expanded: s.expanded, depth: s.best }
}
