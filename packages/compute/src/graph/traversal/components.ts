/**
 * Components: connected components by union–find, strongly connected components by Tarjan's algorithm (Tarjan, 1972,
 * "Depth-first search and linear graph algorithms", SIAM J. Comput. 1(2)) and by Kosaraju's (Sharir, 1981, "A
 * strong-connectivity algorithm and its applications in data flow analysis", Comput. Math. Appl. 7(1); CLRS §22.5),
 * the condensation DAG, and a bipartiteness check that returns a two-colouring or an odd cycle.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { adjacency, ints, intsOf, isDirected, reverse, type Arc, type Edge, type Graph } from '../graph'
import { unionFind, unionFindRoot, unite } from '../heap'
import { breadthFirstSearch, depthFirst, stepBound, type DepthFirstState } from './traversal'

/** A partition of the nodes into components. */
export interface Components {
  count: number
  /** The component of each node, 0 … count − 1; int32, length V. */
  labels: Tensor
  /** The nodes of each component (int32 vectors), in the order the components are numbered. */
  members: Tensor[]
}

function partition(labels: ArrayLike<number>, count: number): Components {
  const members: number[][] = Array.from({ length: count }, () => [])
  for (let v = 0; v < labels.length; v++) members[labels[v]].push(v)
  return { count, labels: ints(labels), members: members.map(ints) }
}

/**
 * Connected components, ignoring edge direction (weakly connected components of a directed graph), by union–find.
 * Components are numbered by their smallest node.
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

/** What the last step of Tarjan's algorithm did. */
export type TarjanEvent = 'start' | 'root' | 'tree' | 'back' | 'ignore' | 'return' | 'component' | 'done'

/** One state of Tarjan's algorithm. Per-node tensors have length V; int32. */
export interface TarjanState extends Status {
  adjacency: readonly (readonly Arc[])[]
  /** Roots are tried in index order from here. */
  nextRoot: number
  /** The depth-first path (call stack), bottom first, and each entry's next adjacency position. */
  path: Tensor
  cursor: Tensor
  /** Discovery index (0, 1, …) and low-link: the least index reachable through the node's subtree and one back edge. */
  index: Tensor
  lowlink: Tensor
  /** Tarjan's stack of nodes not yet assigned a component, bottom first, and a 0/1 membership flag per node. */
  stack: Tensor
  onStack: Tensor
  /** The next discovery index. */
  counter: number
  /** Component of each node, −1 until assigned. Components are numbered in the order found (reverse topological). */
  component: Tensor
  count: number
  /** The nodes of the component completed by the last step (empty otherwise). */
  emitted: Tensor
  current: number
  edge: number
  event: TarjanEvent
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
 * Tarjan's strongly connected components as a traceable algorithm, iterative with an explicit call stack. Options: a
 * graph. A step starts a search at the next unindexed node, or examines one edge of the node on top of the path (a tree
 * edge pushes the other end; an edge to a node on Tarjan's stack lowers the low-link: `back`; an edge into a finished
 * component is ignored), or finishes the top node: if its low-link equals its index it is the root of a component,
 * which is popped off Tarjan's stack (`component`), and its low-link is passed to its parent (`return`).
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
  /** Component of each node, −1 until assigned; numbered in the order found (topological order of the condensation). */
  component: Tensor
  count: number
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
 * Kosaraju's strongly connected components as a traceable algorithm. Options: a graph. Phase 1 is a depth-first search
 * of the graph over all nodes; phase 2 searches the reversed graph, taking roots in decreasing finish time of phase 1.
 * Each tree of phase 2 is one component. Every step is one step of the current depth-first search.
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
 * components.
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
 * by at least one edge, in order of first occurrence. It is a DAG. Its labels list each component's members (by label
 * when the graph has labels).
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

/** A two-colouring, or an odd cycle that rules one out. */
export type BipartiteResult =
  | { bipartite: true; /** Side 0 or 1 of each node; int32, length V. */ colour: Tensor }
  | {
      bipartite: false
      /** An odd cycle as its nodes in order (the last joined to the first); int32. */ oddCycle: Tensor
    }

/**
 * Whether the graph (edge directions ignored) is bipartite. Breadth-first search from every node colours each node
 * by the parity of its depth; an edge between two nodes of one colour joins two nodes of equal depth, and their paths
 * up to their lowest common ancestor close an odd cycle with it.
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
