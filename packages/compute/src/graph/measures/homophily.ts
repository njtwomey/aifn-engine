/**
 * How often the edges of a labelled graph join nodes of the same class. Edge homophily is the share of edges whose
 * ends share a class (Zhu et al. 2020, "Beyond homophily in graph neural networks", NeurIPS). Node homophily averages,
 * over nodes with at least one neighbour, the share of a node's neighbours in its own class (Pei et al. 2020,
 * "Geom-GCN", ICLR). Adjusted homophily subtracts the edge homophily expected when edges ignore the classes but keep
 * the degrees (Platonov et al. 2023, "Characterizing graph datasets for node classification", NeurIPS):
 *
 * $$
 * h_{\text{adj}} = \frac{h_{\text{edge}} - \sum_k \bar p_k^2}{1 - \sum_k \bar p_k^2},
 * \qquad \bar p_k = \frac{D_k}{2\lvert\Ecal\rvert},
 * $$
 *
 * with $D_k$ the total degree of class $k$. It is 0 for a graph whose edges ignore the classes, whatever the class
 * sizes, and negative under heterophily.
 *
 * Every edge is read as an unordered pair and counted once as listed; self-loops are ignored. Labels are class indices
 * $0, 1, \dots$ (an int32 tensor or an array). A graph without edges (other than self-loops) gives NaN, reported
 * rather than hidden.
 */

import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { isTensor, toFlat } from 'aifn-compute/foundation/tensor'
import type { Graph } from '../graph'

/**
 * The labels as an array, checked against the graph: throws `ShapeError` unless there is one per node and
 * `DomainError` unless each is a non-negative integer.
 *
 * @param g The graph.
 * @param y The labels, one per node.
 * @param where The caller's name, for error messages.
 * @returns The labels as numbers.
 */
function readLabels(g: Graph, y: VectorLike, where: string): ArrayLike<number> {
  const labels = isTensor(y) ? toFlat(y) : y
  if (labels.length !== g.nodes)
    throw new ShapeError(where, `${where}: ${labels.length} labels for ${g.nodes} nodes`, [[labels.length], [g.nodes]])
  for (let i = 0; i < labels.length; i++)
    if (!(Number.isInteger(labels[i]) && labels[i] >= 0))
      throw new DomainError(where, `${where}: label ${labels[i]} of node ${i} is not a class index`)
  return labels
}

/**
 * The edges of a graph that are not self-loops.
 *
 * @param g The graph.
 * @returns Its edges with distinct ends, in edge order.
 */
const pairs = (g: Graph) => g.edges.filter((e) => e.from !== e.to)

/**
 * Edge homophily: the share of edges whose two ends have the same label (Zhu et al. 2020).
 *
 * @param g The graph, each undirected edge listed once; self-loops are ignored.
 * @param y The class of every node.
 * @returns The share in $[0, 1]$, or NaN for a graph with no edges.
 *
 * @example A 6-cycle labelled A A B B A B
 * import { cycleGraph } from 'aifn-compute/graph/structures'
 * const g = cycleGraph(6)
 * print('edge homophily:', edgeHomophily(g, [0, 0, 1, 1, 0, 1]))
 */
export function edgeHomophily(g: Graph, y: VectorLike): number {
  const labels = readLabels(g, y, 'edgeHomophily')
  const es = pairs(g)
  if (es.length === 0) return NaN
  return es.filter((e) => labels[e.from] === labels[e.to]).length / es.length
}

/**
 * Node homophily: the mean, over nodes with at least one neighbour, of the share of a node's neighbours that have its
 * label (Pei et al. 2020). Nodes with no neighbours are left out of the mean.
 *
 * @param g The graph, read as undirected; self-loops are ignored.
 * @param y The class of every node.
 * @returns The mean share in $[0, 1]$, or NaN when no node has a neighbour.
 *
 * @example The same 6-cycle: every node has one neighbour of its class out of two, or none
 * import { cycleGraph } from 'aifn-compute/graph/structures'
 * const g = cycleGraph(6)
 * print('node homophily:', nodeHomophily(g, [0, 0, 1, 1, 0, 1]))
 */
export function nodeHomophily(g: Graph, y: VectorLike): number {
  const labels = readLabels(g, y, 'nodeHomophily')
  const same = new Float64Array(g.nodes)
  const degree = new Float64Array(g.nodes)
  for (const e of pairs(g)) {
    const s = labels[e.from] === labels[e.to] ? 1 : 0
    same[e.from] += s
    same[e.to] += s
    degree[e.from]++
    degree[e.to]++
  }
  let total = 0
  let count = 0
  for (let v = 0; v < g.nodes; v++)
    if (degree[v] > 0) {
      total += same[v] / degree[v]
      count++
    }
  return count ? total / count : NaN
}

/**
 * Adjusted homophily (Platonov et al. 2023): edge homophily corrected for class sizes and degrees, 0 when edges
 * ignore the labels, 1 when every edge joins a class to itself, negative under heterophily. See the file comment for
 * the formula.
 *
 * @param g The graph, read as undirected; self-loops are ignored.
 * @param y The class of every node, integers from 0.
 * @returns The adjusted homophily, at most 1; NaN for a graph with no edges, and for one whose edges all lie in one
 *   class (where the correction divides by zero).
 *
 * @example The 6-cycle is heterophilous once degrees are accounted for
 * import { cycleGraph } from 'aifn-compute/graph/structures'
 * const g = cycleGraph(6)
 * const y = [0, 0, 1, 1, 0, 1]
 * print('edge:', edgeHomophily(g, y), ' adjusted:', adjustedHomophily(g, y))
 */
export function adjustedHomophily(g: Graph, y: VectorLike): number {
  const labels = readLabels(g, y, 'adjustedHomophily')
  const es = pairs(g)
  if (es.length === 0) return NaN
  let classes = 0
  for (let i = 0; i < labels.length; i++) classes = Math.max(classes, labels[i] + 1)
  const mass = new Float64Array(classes)
  let same = 0
  for (const e of es) {
    mass[labels[e.from]]++
    mass[labels[e.to]]++
    if (labels[e.from] === labels[e.to]) same++
  }
  let expected = 0
  for (const m of mass) expected += (m / (2 * es.length)) ** 2
  return expected === 1 ? NaN : (same / es.length - expected) / (1 - expected)
}
