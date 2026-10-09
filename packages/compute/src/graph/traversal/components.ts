/**
 * Components: connected components by union–find, strongly connected components by Tarjan's algorithm (Tarjan, 1972,
 * "Depth-first search and linear graph algorithms", SIAM J. Comput. 1(2)) and by Kosaraju's (Sharir, 1981, "A
 * strong-connectivity algorithm and its applications in data flow analysis", Comput. Math. Appl. 7(1); CLRS §22.5),
 * the condensation DAG, and a bipartiteness check that returns a two-colouring or an odd cycle.
 *
 * A partition is returned as `Components`: a label per node and the members of each component. Tarjan's algorithm
 * numbers the strongly connected components in reverse topological order of the condensation, Kosaraju's in
 * topological order. $V$ is the number of nodes throughout.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { adjacency, ints, intsOf, isDirected, reverse, type Arc, type Edge, type Graph } from '../graph'
import { unionFind, unionFindRoot, unite } from '../heap'
import { breadthFirstSearch, depthFirst, stepBound, type DepthFirstState } from './traversal'

/** A partition of the nodes into components. */
export interface Components {
  /** The number of components. */
  count: number
  /** The component of each node, an index below `count`; int32, length $V$. */
  labels: Tensor
  /** The nodes of each component (int32 vectors), in the order the components are numbered. */
  members: Tensor[]
}

/**
 * A `Components` from one label per node: the members of each component are its nodes in increasing order.
 *
 * @param labels The component of each node, each an integer below `count`; it is copied, not kept.
 * @param count The number of components; a label not used by any node gives an empty member list.
 * @returns The count, the labels as an int32 tensor, and each component's members.
 */
function partition(labels: ArrayLike<number>, count: number): Components {
  const members: number[][] = Array.from({ length: count }, () => [])
  for (let v = 0; v < labels.length; v++) members[labels[v]].push(v)
  return { count, labels: ints(labels), members: members.map(ints) }
}

/**
 * Connected components, ignoring edge direction (weakly connected components of a directed graph), by union–find.
 * Components are numbered in the order of their smallest nodes, so node 0 is in component 0.
 *
 * @param graph The graph; only its node count and the ends of its edges are read.
 * @returns The number of components, each node's component and each component's members.
 *
 * @example Three components, one of them a lone node
 * // 0 - 1, 2 - 3 - 4, and 5 alone.
 * const edges = [[0, 1], [3, 4], [2, 3]].map(([from, to]) => ({ from, to }))
 * const c = connectedComponents({ kind: 'graph', nodes: 6, edges, directed: false })
 * print('count =', c.count)
 * print('labels =', c.labels)
 * print('members =', c.members)
 *
 * @example Direction is ignored
 * // 0 → 1 ← 2: no directed path joins 0 and 2, but they are weakly connected.
 * const edges = [[0, 1], [2, 1]].map(([from, to]) => ({ from, to }))
 * print('labels =', connectedComponents({ kind: 'graph', nodes: 3, edges }).labels)
 */
export function connectedComponents(graph: Graph): Components {
  const uf = unionFind(graph.nodes)
  for (const e of graph.edges) unite(uf, e.from, e.to)
  const id = new Map<number, number>()
  const labels = new Int32Array(graph.nodes)
  for (let v = 0; v < graph.nodes; v++) {
    const r = unionFindRoot(uf, v)
    if (!id.has(r)) id.set(r, id.size)
    labels[v] = id.get(r)!
  }
  return partition(labels, id.size)
}

// ---------------------------------------------------------------------------------------------------------------------
// Tarjan.

/**
 * What the last step of Tarjan's algorithm did: `root` (started a search at an unindexed node), `tree` (followed an
 * edge to an unindexed node), `back` (met an edge to a node on Tarjan's stack, lowering the low-link), `ignore` (met an
 * edge into a finished component), `return` (finished a node that is not a component's root), `component` (finished a
 * component's root and popped the component), `done`.
 */
export type TarjanEvent = 'start' | 'root' | 'tree' | 'back' | 'ignore' | 'return' | 'component' | 'done'

/** One state of Tarjan's algorithm. Per-node tensors have length $V$; int32. */
export interface TarjanState extends Status {
  /** Each node's arcs, as `adjacency(graph)` gives them: the order edges are examined in. */
  adjacency: readonly (readonly Arc[])[]
  /** Roots are tried in index order from here. */
  nextRoot: number
  /** The depth-first path (call stack), bottom first. */
  path: Tensor
  /** For each entry of `path`, the position in its adjacency list of the next edge to examine. */
  cursor: Tensor
  /** Discovery index of each node ($0, 1, \dots$), $-1$ until discovered. */
  index: Tensor
  /**
   * Low-link of each node: the least index of a node on Tarjan's stack that the node's depth-first subtree reaches by
   * one non-tree edge, or its own index if smaller; $-1$ until discovered.
   */
  lowlink: Tensor
  /** Tarjan's stack of nodes not yet assigned a component, bottom first. */
  stack: Tensor
  /** 1 for each node on Tarjan's stack, else 0. */
  onStack: Tensor
  /** The next discovery index. */
  counter: number
  /** Component of each node, $-1$ until assigned. Components are numbered in the order found (reverse topological). */
  component: Tensor
  /** Components found so far. */
  count: number
  /** The nodes of the component completed by the last step (empty otherwise). */
  emitted: Tensor
  /** The node the last step acted on: the new root, the node whose edge was examined, or the node finished. */
  current: number
  /** The edge examined by the last step, $-1$ for none. */
  edge: number
  /** What the last step did. */
  event: TarjanEvent
  /** True once every node has been indexed and the path is empty. */
  done: boolean
}

/** The init and step of `tarjanSteps` on the whole problem; `t` is added by the factory. */
const tarjan = {
  init: (graph: Graph): Omit<TarjanState, 't'> => {
    const V = graph.nodes
    return {
      adjacency: adjacency(graph),
      nextRoot: 0,
      path: ints([]),
      cursor: ints([]),
      index: ints(new Int32Array(V).fill(-1)),
      lowlink: ints(new Int32Array(V).fill(-1)),
      stack: ints([]),
      onStack: ints(new Int32Array(V)),
      counter: 0,
      component: ints(new Int32Array(V).fill(-1)),
      count: 0,
      emitted: ints([]),
      current: -1,
      edge: -1,
      event: 'start',
      done: V === 0,
    }
  },
  step: (s: TarjanState): Omit<TarjanState, 't'> => {
    if (s.done) return s
    const index = intsOf(s.index)
    const lowlink = intsOf(s.lowlink)
    const onStack = intsOf(s.onStack)
    const visit = (w: number) => {
      index[w] = lowlink[w] = s.counter
      onStack[w] = 1
    }
    const n = s.path.shape[0]
    if (n === 0) {
      let r = s.nextRoot
      while (r < index.length && index[r] >= 0) r++
      if (r === index.length)
        return { ...s, nextRoot: r, emitted: ints([]), current: -1, edge: -1, event: 'done', done: true }
      visit(r)
      return {
        ...s,
        nextRoot: r + 1,
        path: ints([r]),
        cursor: ints([0]),
        index: ints(index),
        lowlink: ints(lowlink),
        onStack: ints(onStack),
        stack: ints([...s.stack.data, r]),
        counter: s.counter + 1,
        emitted: ints([]),
        current: r,
        edge: -1,
        event: 'root',
      }
    }
    const v = s.path.data[n - 1]
    const cursor = intsOf(s.cursor)
    const arcs = s.adjacency[v]
    if (cursor[n - 1] < arcs.length) {
      const arc = arcs[cursor[n - 1]++]
      const w = arc.to
      const base = { ...s, cursor: ints(cursor), emitted: ints([]), current: v, edge: arc.edge }
      if (index[w] < 0) {
        visit(w)
        return {
          ...base,
          path: ints([...s.path.data, w]),
          cursor: ints([...cursor, 0]),
          index: ints(index),
          lowlink: ints(lowlink),
          onStack: ints(onStack),
          stack: ints([...s.stack.data, w]),
          counter: s.counter + 1,
          event: 'tree',
        }
      }
      if (onStack[w]) {
        lowlink[v] = Math.min(lowlink[v], index[w])
        return { ...base, lowlink: ints(lowlink), event: 'back' }
      }
      return { ...base, event: 'ignore' }
    }
    // v is finished.
    let stack = Array.from(s.stack.data)
    let component = s.component
    let count = s.count
    let emitted: number[] = []
    if (lowlink[v] === index[v]) {
      const at = stack.lastIndexOf(v)
      emitted = stack.slice(at)
      stack = stack.slice(0, at)
      const labels = intsOf(s.component)
      for (const u of emitted) {
        labels[u] = count
        onStack[u] = 0
      }
      component = ints(labels)
      count++
    }
    if (n > 1) {
      const u = s.path.data[n - 2]
      lowlink[u] = Math.min(lowlink[u], lowlink[v])
    }
    return {
      ...s,
      path: ints(s.path.data.slice(0, n - 1)),
      cursor: ints(cursor.slice(0, n - 1)),
      lowlink: ints(lowlink),
      onStack: ints(onStack),
      stack: ints(stack),
      component,
      count,
      emitted: ints(emitted),
      current: v,
      edge: -1,
      event: emitted.length ? 'component' : 'return',
    }
  },
}

/**
 * Tarjan's strongly connected components (Tarjan, 1972) as a traceable algorithm (start: none), iterative with an
 * explicit call stack. A step starts a search at the next unindexed node, or examines one edge of the node on top of
 * the path (a tree edge pushes the other end; an edge to a node on Tarjan's stack lowers the low-link: `back`; an edge
 * into a finished component is ignored), or finishes the top node and passes its low-link to its parent: if the
 * low-link equals the node's index the node is the root of a component, which is popped off Tarjan's stack
 * (`component`); otherwise the event is `return`.
 *
 * @param graph The graph, directed or undirected (where the components are the connected ones); it is not modified.
 * @returns The algorithm, to run with `run(alg, undefined, steps)` or step through with `trace`.
 *
 * @example Up to the first component
 * // The cycle 0 → 1 → 2 → 0 leads by 2 → 3 into the cycle 3 ⇄ 4.
 * const edges = [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges }
 * // Steps 1 to 7 reach 4 and meet 4 → 3; step 8 finishes 4, step 9 finishes 3, the root of {3, 4}.
 * const s = run(tarjanSteps(g), undefined, 9)
 * print('event =', s.event)
 * print('emitted =', s.emitted)
 * print('index =', s.index)
 * print('lowlink =', s.lowlink)
 * print('still on the stack =', s.stack)
 */
export function tarjanSteps(graph: Graph): Algorithm<void, TarjanState> {
  const problem: Graph = graph
  return {
    name: 'tarjan',
    init: () => ({ ...tarjan.init(problem), t: 0 }),
    step: (s) => ({ ...tarjan.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Kosaraju.

/** One state of Kosaraju's algorithm. */
export interface KosarajuState extends Status {
  /** `forward`: a depth-first search of the graph for finish times; `reverse`: searches of the reversed graph. */
  phase: 'forward' | 'reverse'
  /** The depth-first search of the current phase (see `depthFirstSteps`; its `t` counts that search's steps). */
  search: DepthFirstState
  /** The reversed graph, searched in the second phase. */
  reversed: Graph
  /** Nodes in decreasing finish time of the first phase (the second phase's root order); empty until then; int32. */
  rootOrder: Tensor
  /**
   * Component of each node, $-1$ until assigned; numbered in the order found (topological order of the condensation);
   * int32, length $V$.
   */
  component: Tensor
  /** Components found so far (trees started in the second phase). */
  count: number
  /** True once the second phase's search is done. */
  done: boolean
}

/** The init and step of `kosarajuSteps` on the whole problem; `t` is added by the factory. */
const kosaraju = {
  init: (graph: Graph): Omit<KosarajuState, 't'> => ({
    phase: 'forward',
    search: { ...depthFirst.init({ graph }), t: 0 },
    reversed: reverse(graph),
    rootOrder: ints([]),
    component: ints(new Int32Array(graph.nodes).fill(-1)),
    count: 0,
    done: graph.nodes === 0,
  }),
  step: (s: KosarajuState): Omit<KosarajuState, 't'> => {
    if (s.done) return s
    const search = { ...depthFirst.step(s.search), t: s.search.t + 1 }
    if (s.phase === 'forward') {
      if (!search.done) return { ...s, search }
      const rootOrder = Array.from(search.postorder.data).reverse()
      return {
        ...s,
        phase: 'reverse',
        search: { ...depthFirst.init({ graph: s.reversed, source: rootOrder }), t: 0 },
        rootOrder: ints(rootOrder),
      }
    }
    let { component, count } = s
    if (search.event === 'root' || search.event === 'tree') {
      if (search.event === 'root') count++
      const labels = intsOf(component)
      labels[search.preorder.data[search.preorder.shape[0] - 1]] = count - 1
      component = ints(labels)
    }
    return { ...s, search, component, count, done: search.done }
  },
}

/**
 * Kosaraju's strongly connected components (Sharir, 1981) as a traceable algorithm (start: none). Phase 1 is a
 * depth-first search of the graph over all nodes; phase 2 searches the reversed graph, taking roots in decreasing
 * finish time of phase 1. Each tree of phase 2 is one component. Every step is one step of the current depth-first
 * search; the step that ends phase 1 also sets up phase 2.
 *
 * @param graph The graph, directed or undirected (where the components are the connected ones); it is not modified.
 * @returns The algorithm, to run with `run(alg, undefined, steps)` or step through with `trace`.
 *
 * @example The two phases
 * // The cycle 0 → 1 → 2 → 0 leads by 2 → 3 into the cycle 3 ⇄ 4.
 * const edges = [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges }
 * // Phase 1 takes 13 steps: 5 roots and tree edges, 2 back edges, 5 finishes and the end.
 * const s = run(kosarajuSteps(g), undefined, 13)
 * print('phase =', s.phase)
 * print('root order of phase 2 =', s.rootOrder)
 * const end = run(kosarajuSteps(g), undefined, 100)
 * print('component =', end.component)
 */
export function kosarajuSteps(graph: Graph): Algorithm<void, KosarajuState> {
  const problem: Graph = graph
  return {
    name: 'kosaraju',
    init: () => ({ ...kosaraju.init(problem), t: 0 }),
    step: (s) => ({ ...kosaraju.step(s), t: s.t + 1 }),
    done: (s) => s.done,
  }
}

/**
 * Strongly connected components by Tarjan's algorithm (default; components numbered in reverse topological order of
 * the condensation) or Kosaraju's (numbered in topological order). For an undirected graph they are its connected
 * components. Runs `tarjanSteps` or `kosarajuSteps` to the end.
 *
 * @param graph The graph; it is not modified.
 * @param options `method`: `'tarjan'` (default) or `'kosaraju'`. Both find the same components; only the numbering
 *   differs.
 * @returns The number of components, each node's component and each component's members.
 *
 * @example The two methods number the components in opposite orders
 * // The cycle 0 → 1 → 2 → 0 leads by 2 → 3 into the cycle 3 ⇄ 4.
 * const edges = [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 3]].map(([from, to]) => ({ from, to }))
 * const g = { kind: 'graph', nodes: 5, edges }
 * print('Tarjan:', stronglyConnectedComponents(g).members)
 * print('Kosaraju:', stronglyConnectedComponents(g, { method: 'kosaraju' }).members)
 */
export function stronglyConnectedComponents(
  graph: Graph,
  options: { method?: 'tarjan' | 'kosaraju' } = {},
): Components {
  const bound = 2 * stepBound(graph) + 2
  const s =
    (options.method ?? 'tarjan') === 'tarjan'
      ? run(tarjanSteps(graph), undefined, bound)
      : run(kosarajuSteps(graph), undefined, bound)
  return partition(s.component.data, s.count)
}

/**
 * The condensation of a directed graph: one node per strongly connected component (numbered as by
 * `stronglyConnectedComponents` with the given method) and one edge between two components for each ordered pair joined
 * by at least one edge, in order of first occurrence. It is a DAG. Its labels list each component's members, joined by
 * commas (by label when the graph has labels). Edge weights are dropped.
 *
 * @param graph The graph to condense; it is not modified. An undirected graph gives its connected components with no
 *   edges between them.
 * @param options `method`: `'tarjan'` (default) or `'kosaraju'`, which fixes the numbering of the components.
 * @returns The condensed `graph`, with the direction of `graph`, and the `components` it was built from.
 *
 * @example Two cycles joined by one edge condense to one edge
 * // The cycle 0 → 1 → 2 → 0 leads by 2 → 3 into the cycle 3 ⇄ 4.
 * const edges = [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 3]].map(([from, to]) => ({ from, to }))
 * const c = condensation({ kind: 'graph', nodes: 5, edges })
 * // Tarjan's numbering finds {3, 4} first, so the edge runs from component 1 to component 0.
 * print('nodes =', c.graph.labels)
 * print('edges =', c.graph.edges)
 */
export function condensation(
  graph: Graph,
  options: { method?: 'tarjan' | 'kosaraju' } = {},
): { graph: Graph; components: Components } {
  const components = stronglyConnectedComponents(graph, options)
  const label = components.labels.data
  const seen = new Set<string>()
  const edges: Edge[] = []
  for (const e of graph.edges) {
    const a = label[e.from]
    const b = label[e.to]
    if (a === b || seen.has(`${a},${b}`)) continue
    seen.add(`${a},${b}`)
    edges.push({ from: a, to: b })
  }
  const labels = components.members.map((m) =>
    Array.from(m.data, (v) => (graph.labels ? graph.labels[v] : String(v))).join(','),
  )
  return { graph: { kind: 'graph', nodes: components.count, edges, directed: isDirected(graph), labels }, components }
}

// ---------------------------------------------------------------------------------------------------------------------
// Bipartiteness.

/**
 * A two-colouring (`colour`: side 0 or 1 of each node, int32 of length $V$), or an odd cycle that rules one out;
 * `bipartite` says which.
 */
export type BipartiteResult =
  | { bipartite: true; /** Side 0 or 1 of each node; int32, length V. */ colour: Tensor }
  | {
      bipartite: false
      /** An odd cycle as its nodes in order (the last joined to the first); int32. */ oddCycle: Tensor
    }

/**
 * Whether the graph (edge directions ignored) is bipartite. Breadth-first search from every node colours each node
 * by the parity of its depth; an edge between two nodes of one colour joins two nodes of equal depth, and their paths
 * up to their lowest common ancestor close an odd cycle with it. A self-loop is an odd cycle of one node.
 *
 * @param graph The graph, directed or not; directions are ignored and it is not modified.
 * @returns `{ bipartite: true, colour }`, the side of each node (each tree's root on side 0), or
 *   `{ bipartite: false, oddCycle }` for the first edge, in edge order, that joins two nodes of one side.
 *
 * @example An even cycle is bipartite, a triangle is not
 * const square = [[0, 1], [1, 2], [2, 3], [3, 0]].map(([from, to]) => ({ from, to }))
 * const triangle = [[0, 1], [1, 2], [2, 0]].map(([from, to]) => ({ from, to }))
 * print('square:', bipartite({ kind: 'graph', nodes: 4, edges: square, directed: false }))
 * print('triangle:', bipartite({ kind: 'graph', nodes: 3, edges: triangle, directed: false }))
 */
export function bipartite(graph: Graph): BipartiteResult {
  const undirected: Graph = { ...graph, directed: false }
  const { depth, parent } = breadthFirstSearch(undirected)
  const colour = Array.from(depth.data, (d) => d % 2)
  for (const e of graph.edges) {
    if (colour[e.from] !== colour[e.to]) continue
    let a = e.from
    let b = e.to
    const left = [a]
    const right = [b]
    while (a !== b) {
      a = parent.data[a]
      b = parent.data[b]
      left.push(a)
      right.push(b)
    }
    // left runs from e.from up to the common ancestor, right from e.to up to it: join them into one cycle.
    right.pop()
    return { bipartite: false, oddCycle: ints([...left.reverse(), ...right]) }
  }
  return { bipartite: true, colour: ints(colour) }
}
