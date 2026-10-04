/**
 * The matrices of a graph as dense float64 Tensors: adjacency, degree, the Laplacian L = D − A with its symmetric
 * normalisation I − D^{−1/2} A D^{−1/2} and its random-walk normalisation I − D^{−1} A (Chung 1997, "Spectral Graph
 * Theory", ch. 1; von Luxburg 2007, "A tutorial on spectral clustering", §3), and the incidence matrix.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { adjacency, isDirected, weightOf, type Graph } from '../graph'

/** Whether to use edge weights (default) or count edges. */
export interface WeightOptions {
  weighted?: boolean
}

/**
 * The V × V adjacency matrix: entry (i, j) is the summed weight of the edges i → j (their count with
 * `weighted: false`), 0 where there is none; symmetric for an undirected graph (a self-loop counts once).
 */
export function adjacencyMatrix(g: Graph, options: WeightOptions = {}): Tensor {
  adjacency(g) // checks the graph
  const V = g.nodes
  const a = new Float64Array(V * V)
  const weighted = options.weighted ?? true
  for (const e of g.edges) {
    const w = weighted ? weightOf(e) : 1
    a[e.from * V + e.to] += w
    if (!isDirected(g) && e.from !== e.to) a[e.to * V + e.from] += w
  }
  return fromData(a, [V, V])
}

/** Row sums of the adjacency matrix: the (weighted) out-degree of each node, a float64 vector of length V. */
export function degrees(g: Graph, options: WeightOptions = {}): Tensor {
  const A = adjacencyMatrix(g, options).data
  const V = g.nodes
  const d = new Float64Array(V)
  for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) d[i] += A[i * V + j]
  return fromData(d, [V])
}

/** The diagonal V × V degree matrix D = diag(degrees). */
export function degreeMatrix(g: Graph, options: WeightOptions = {}): Tensor {
  const d = degrees(g, options).data
  const V = g.nodes
  const out = new Float64Array(V * V)
  for (let i = 0; i < V; i++) out[i * V + i] = d[i]
  return fromData(out, [V, V])
}

/**
 * The graph Laplacian (V × V): `unnormalised` L = D − A (default), `symmetric` L_sym = I − D^{−1/2} A D^{−1/2}, or
 * `random-walk` L_rw = I − D^{−1} A, with D the (weighted) out-degrees. A node of degree 0 has a zero row in the
 * normalised forms (and a unit diagonal only where its degree is positive). For an undirected graph L and L_sym are
 * symmetric positive semidefinite, with one zero eigenvalue per connected component.
 */
export function laplacian(
  g: Graph,
  options: WeightOptions & { normalisation?: 'unnormalised' | 'symmetric' | 'random-walk' } = {},
): Tensor {
  const A = adjacencyMatrix(g, options).data
  const d = degrees(g, options).data
  const V: Size = g.nodes
  const out = new Float64Array(V * V)
  const mode = options.normalisation ?? 'unnormalised'
  const inv = (x: number) => (x > 0 ? 1 / x : 0)
  for (let i = 0; i < V; i++)
    for (let j = 0; j < V; j++) {
      const a = A[i * V + j]
      const diag = i === j ? (mode === 'unnormalised' ? d[i] : d[i] > 0 ? 1 : 0) : 0
      const scaled =
        mode === 'unnormalised' ? a : mode === 'symmetric' ? a * Math.sqrt(inv(d[i]) * inv(d[j])) : a * inv(d[i])
      out[i * V + j] = diag - scaled
    }
  return fromData(out, [V, V])
}

/**
 * The V × E incidence matrix, column k for edge k. Oriented (default for a directed graph): −w at `from` and +w at
 * `to`; unoriented (default for an undirected graph): +w at both ends. w is the weight (1 with `weighted: false`,
 * the default here). For an oriented incidence matrix B of an undirected graph, B Bᵀ = L (unweighted).
 */
export function incidenceMatrix(g: Graph, options: { oriented?: boolean; weighted?: boolean } = {}): Tensor {
  adjacency(g)
  const V = g.nodes
  const E = g.edges.length
  const oriented = options.oriented ?? isDirected(g)
  const out = new Float64Array(V * E)
  g.edges.forEach((e, k) => {
    const w = options.weighted ? weightOf(e) : 1
    if (e.from === e.to) return // a self-loop has no incidence column entries in either convention
    out[e.from * E + k] += oriented ? -w : w
    out[e.to * E + k] += w
  })
  return fromData(out, [V, E])
}
