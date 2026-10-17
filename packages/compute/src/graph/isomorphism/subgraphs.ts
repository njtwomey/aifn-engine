/**
 * Subgraph selection policies and the DS-WL test of subgraph graph networks (Bevilacqua et al. 2022, "Equivariant
 * subgraph aggregation networks", ICLR, §2.2 for the policies and §3.1 for DS-WL).
 *
 * A policy maps a graph to a bag of subgraphs: `node-deleted` removes every edge at one node (one subgraph per node),
 * `edge-deleted` removes one edge (one per edge), and `ego` keeps the edges inside the depth-$k$ neighbourhood of one
 * root (one per node), the root marked by an initial label when `markRoot` is set (EGO+). Every subgraph keeps all $n$
 * nodes, so the bags stay aligned with the original nodes, as in the paper's implementation. `subgraphWeisfeilerLeman`
 * runs colour refinement on every subgraph of every graph with one shared palette and compares the bags (multisets)
 * of stable colour histograms: the DS-WL test, the upper bound on what a DS-GNN with a message-passing encoder can
 * tell apart.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { kHopNeighbourhood } from 'aifn-compute/graph/traversal'
import { fromEdges, isDirected, type Graph } from '../graph'
import { weisfeilerLeman } from './weisfeiler-leman'

/** A subgraph selection policy: delete a node's edges, delete an edge, or keep a node's ego network. */
export type SubgraphPolicy = 'node-deleted' | 'edge-deleted' | 'ego'

/** One subgraph of a bag. */
export interface Subgraph {
  /** The subgraph, on the same $n$ nodes as the graph. */
  graph: Graph
  /** The node whose edges were deleted, or the ego root; $-1$ for an edge-deleted subgraph. */
  node: number
  /** The deleted edge of an edge-deleted subgraph, as its two ends. */
  edge?: readonly [number, number]
  /** Initial node labels for colour refinement: 1 on a marked root, else 0. */
  initial: number[]
}

/** Options of `subgraphBag` and `subgraphWeisfeilerLeman`. */
export interface SubgraphOptions {
  /** The depth $k$ of the ego networks (default 1). */
  depth?: number
  /** Mark the ego root with label 1 (EGO+; default true). */
  markRoot?: boolean
}

/**
 * The bag of subgraphs a policy selects (see the file comment). Throws `DomainError` for a directed graph.
 *
 * @param g The undirected graph, each edge listed once.
 * @param policy The selection policy.
 * @param options The depth of the ego networks and whether their roots are marked.
 * @returns The subgraphs, in node order (edge order for `edge-deleted`).
 *
 * @example The node-deleted bag of a triangle: three single edges
 * import { cycleGraph } from 'aifn-compute/graph/structures'
 * const bag = subgraphBag(cycleGraph(3), 'node-deleted')
 * print(bag.map((s) => s.graph.edges.map((e) => [e.from, e.to])))
 */
export function subgraphBag(g: Graph, policy: SubgraphPolicy, options: SubgraphOptions = {}): Subgraph[] {
  if (isDirected(g)) throw new DomainError('subgraphBag', 'subgraphBag: the graph must be undirected')
  const n = g.nodes
  const edges = g.edges.map((e) => [e.from, e.to] as const)
  const zeros = () => new Array<number>(n).fill(0)
  const make = (kept: (readonly [number, number])[]) => fromEdges(n, kept, { directed: false })
  if (policy === 'node-deleted')
    return Array.from({ length: n }, (_, v) => ({
      graph: make(edges.filter(([a, b]) => a !== v && b !== v)),
      node: v,
      initial: zeros(),
    }))
  if (policy === 'edge-deleted')
    return edges.map((e, k) => ({ graph: make(edges.filter((_, j) => j !== k)), node: -1, edge: e, initial: zeros() }))
  const depth = options.depth ?? 1
  const markRoot = options.markRoot ?? true
  return Array.from({ length: n }, (_, root) => {
    const hops = toFlat(kHopNeighbourhood(g, root, depth).depth)
    const initial = zeros()
    if (markRoot) initial[root] = 1
    return { graph: make(edges.filter(([a, b]) => hops[a] >= 0 && hops[b] >= 0)), node: root, initial }
  })
}

/** The result of `subgraphWeisfeilerLeman`. */
export interface SubgraphWlResult {
  /** `bags[g][i]`: a key of the stable colour histogram of subgraph $i$ of graph $g$ (equal keys, equal histograms). */
  bags: string[][]
  /** Whether the bags of some two graphs differ as multisets. */
  distinguished: boolean
  /** `subgraphs[g]`: the bag of graph $g$. */
  subgraphs: Subgraph[][]
}

/**
 * The DS-WL test: colour refinement on each subgraph, with one palette for all, then a comparison of the bags of
 * stable histograms (see the file comment). Throws as `subgraphBag` does.
 *
 * @param graphs The undirected graphs to compare.
 * @param policy The selection policy.
 * @param options The depth of the ego networks and whether their roots are marked.
 * @returns Each subgraph's histogram key, whether the bags differ, and the bags of subgraphs.
 *
 * @example Node deletion separates a 6-cycle from two triangles, which 1-WL cannot
 * import { cycleGraph } from 'aifn-compute/graph/structures'
 * const edges = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]].map(([from, to]) => ({ from, to }))
 * const triangles = { kind: 'graph', nodes: 6, edges, directed: false }
 * print('separated:', subgraphWeisfeilerLeman([cycleGraph(6), triangles], 'node-deleted').distinguished)
 */
export function subgraphWeisfeilerLeman(
  graphs: readonly Graph[],
  policy: SubgraphPolicy,
  options: SubgraphOptions = {},
): SubgraphWlResult {
  const subgraphs = graphs.map((g) => subgraphBag(g, policy, options))
  const flat = subgraphs.flat()
  const trace = weisfeilerLeman(
    flat.map((s) => s.graph),
    { initial: flat.map((s) => s.initial) },
  )
  const keys = trace.histograms[trace.histograms.length - 1].map((h) => h.join(','))
  let k = 0
  const bags = subgraphs.map((bag) => bag.map(() => keys[k++]))
  const sorted = bags.map((b) => [...b].sort().join('|'))
  return { bags, distinguished: sorted.some((s) => s !== sorted[0]), subgraphs }
}
