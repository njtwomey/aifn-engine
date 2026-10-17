/**
 * Graph rewiring against oversquashing: changes to a graph's edges that widen its bottlenecks before a message-passing
 * network runs on it. `withVirtualNode` joins one new node to every node (the master node of Gilmer et al. 2017,
 * "Neural message passing for quantum chemistry", ICML), so every two nodes are at most two hops apart. `fosrRewire`
 * adds edges by first-order spectral rewiring (Karhadkar, Banerjee and Montúfar 2023, "FoSR: first-order spectral
 * rewiring for addressing oversquashing in GNNs", ICLR, proposition 5 and algorithm 1): each step adds the non-edge
 * $(i, j)$ that minimises
 *
 * $$\frac{x_i x_j}{\sqrt{(1 + d_i)(1 + d_j)}},$$
 *
 * with $\xvec$ the eigenvector of the second largest eigenvalue of $\Dmat^{-1/2}\Amat\Dmat^{-1/2}$: to first order,
 * the edge that most increases the spectral gap. A negative product means $i$ and $j$ sit on opposite sides of the
 * graph's weakest cut. Unlike the paper, the eigenvector is computed exactly by `eigh` at every step rather than
 * tracked by power iteration, which the paper uses for speed on large graphs.
 *
 * The bottleneck itself is measured by `cheegerConstant` (`aifn-compute/graph/measures`) and the spectral gap by
 * `spectralGap` of `aifn-compute/probability/markov` applied to `randomWalkMatrix` (`aifn-compute/graph/propagation`).
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { adjacencyMatrix } from 'aifn-compute/graph/matrices'
import { eigh } from 'aifn-compute/numerics/linalg'
import { isDirected, type Edge, type Graph } from '../graph'

/**
 * The graph with one extra node, numbered $V$, joined to every original node: in both directions when the graph is
 * directed. Labels, when the graph has them, gain `'virtual'`.
 *
 * @param g A graph with $V$ nodes.
 * @returns The graph on $V + 1$ nodes, the original edges first, then the $V$ edges to the new node (in node order;
 *   $2V$ for a directed graph, each node's pair together).
 *
 * @example A path of three nodes and its virtual node
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * const g = withVirtualNode(chainGraph(3))
 * print('nodes:', g.nodes, ' edges:', g.edges.map((e) => [e.from, e.to]))
 */
export function withVirtualNode(g: Graph): Graph {
  const n = g.nodes
  const extra: Edge[] = []
  for (let v = 0; v < n; v++) {
    extra.push({ from: v, to: n })
    if (isDirected(g)) extra.push({ from: n, to: v })
  }
  return {
    ...g,
    nodes: n + 1,
    edges: [...g.edges, ...extra],
    ...(g.labels ? { labels: [...g.labels, 'virtual'] } : {}),
  }
}

/** The result of `fosrRewire`. */
export interface FosrResult {
  /** The added edges in the order they were added, each as $[i, j]$ with $i < j$. */
  added: [number, number][]
  /** The rewired graph: the original edges, then the added ones. */
  graph: Graph
}

/**
 * First-order spectral rewiring (see the file comment): add up to `count` edges, one at a time, each the non-edge of
 * the current graph with the smallest score, ties to the smaller $(i, j)$. Stops early when the graph is complete.
 * Each step solves an eigenproblem, $O(V^3)$: for graphs of up to a few hundred nodes. Throws `DomainError` for a
 * directed graph or a `count` that is not a non-negative integer.
 *
 * @param g The undirected graph; edge weights count in the degrees and the eigenvector.
 * @param count The number of edges to add.
 * @returns The added edges and the rewired graph.
 *
 * @example On the barbell, FoSR joins the two cliques
 * import { barbellGraph } from 'aifn-compute/graph/structures'
 * print('added:', fosrRewire(barbellGraph(5, 3), 3).added)
 */
export function fosrRewire(g: Graph, count: Size): FosrResult {
  const where = 'fosrRewire'
  if (isDirected(g)) throw new DomainError(where, `${where}: the graph must be undirected`)
  if (!(Number.isInteger(count) && count >= 0)) throw new DomainError(where, `${where}: count must be a count`)
  const n = g.nodes
  let current = g
  const added: [number, number][] = []
  for (let step = 0; step < count; step++) {
    const a = adjacencyMatrix(current).data
    const d = new Float64Array(n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) d[i] += a[i * n + j]
    const m = new Float64Array(n * n)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) m[i * n + j] = d[i] > 0 && d[j] > 0 ? a[i * n + j] / Math.sqrt(d[i] * d[j]) : 0
    // eigh orders the eigenvalues from the largest: column 1 holds the second eigenvector.
    const vectors = toFlat(eigh(fromData(m, [n, n])).vectors)
    let best: [number, number] | null = null
    let bestScore = Infinity
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        if (a[i * n + j] !== 0) continue
        const score = (vectors[i * n + 1] * vectors[j * n + 1]) / Math.sqrt((1 + d[i]) * (1 + d[j]))
        if (score < bestScore - 1e-12) {
          bestScore = score
          best = [i, j]
        }
      }
    if (!best) break
    added.push(best)
    current = { ...current, edges: [...current.edges, { from: best[0], to: best[1] }] }
  }
  return { added, graph: current }
}
