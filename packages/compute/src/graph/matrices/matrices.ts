/**
 * The matrices of a graph as dense float64 Tensors: adjacency, degree, the Laplacian $\Lmat = \Dmat - \Amat$ with its
 * symmetric normalisation $\Imat - \Dmat^{-1/2}\Amat\Dmat^{-1/2}$ and its random-walk normalisation
 * $\Imat - \Dmat^{-1}\Amat$ (Chung 1997, "Spectral Graph Theory", ch. 1; von Luxburg 2007, "A tutorial on spectral
 * clustering", §3), the rescaled Laplacian of Chebyshev filters, and the incidence matrices of graphs and
 * hypergraphs.
 *
 * Node $i$ is row $i$; the degrees $\Dmat$ are the row sums of $\Amat$, so out-degrees for a directed graph. Edge
 * weights are used by default (`weighted: false` counts edges instead), except by `incidenceMatrix`, which is
 * unweighted unless asked. Every function checks the graph first and throws on an endpoint out of range.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { adjacency, checkHyperedges, isDirected, weightOf, type Graph, type Hyperedges } from '../graph'

/** Whether to use edge weights (default) or count edges. */
export interface WeightOptions {
  /** True (default): an edge contributes its weight (1 when unset). False: every edge contributes 1. */
  weighted?: boolean
}

/**
 * The $V \times V$ adjacency matrix $\Amat$: $A_{ij}$ is the summed weight of the edges $i \to j$ (their count with
 * `weighted: false`), 0 where there is none, so parallel edges add up. Symmetric for an undirected graph, where each
 * edge fills both $A_{ij}$ and $A_{ji}$ and a self-loop counts once. Throws on an invalid graph.
 *
 * @param g The graph, with $V$ nodes.
 * @param options Whether to sum edge weights (the default) or count edges.
 * @returns A float64 $V \times V$ tensor.
 *
 * @example A path 0 - 1 - 2, undirected
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2, weight: 5 }], directed: false }
 * print('A =', adjacencyMatrix(g))
 * print('counted =', adjacencyMatrix(g, { weighted: false }))
 *
 * @example A directed edge fills one entry only
 * const g = { kind: 'graph', nodes: 2, edges: [{ from: 0, to: 1 }] }
 * print('A =', adjacencyMatrix(g))
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

/**
 * Row sums of the adjacency matrix, $d_i = \sum_j A_{ij}$: the (weighted) out-degree of each node, or its degree when
 * the graph is undirected (an undirected self-loop counts once).
 *
 * @param g The graph, with $V$ nodes.
 * @param options Whether to sum edge weights (the default) or count edges.
 * @returns A float64 vector of length $V$.
 *
 * @example The star with centre 0
 * const edges = [{ from: 0, to: 1 }, { from: 0, to: 2 }, { from: 0, to: 3 }]
 * print('degrees =', degrees({ kind: 'graph', nodes: 4, edges, directed: false }))
 *
 * @example Directed: out-degrees only
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 0, to: 2 }, { from: 1, to: 2 }] }
 * print('out-degrees =', degrees(g))
 */
export function degrees(g: Graph, options: WeightOptions = {}): Tensor {
  const A = adjacencyMatrix(g, options).data
  const V = g.nodes
  const d = new Float64Array(V)
  for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) d[i] += A[i * V + j]
  return fromData(d, [V])
}

/**
 * The diagonal $V \times V$ degree matrix $\Dmat = \diag(\dvec)$, with $\dvec$ the vector of `degrees`.
 *
 * @param g The graph, with $V$ nodes.
 * @param options Whether to sum edge weights (the default) or count edges.
 * @returns A float64 $V \times V$ tensor, zero off the diagonal.
 *
 * @example The path 0 - 1 - 2
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * print('D =', degreeMatrix(g))
 */
export function degreeMatrix(g: Graph, options: WeightOptions = {}): Tensor {
  const d = degrees(g, options).data
  const V = g.nodes
  const out = new Float64Array(V * V)
  for (let i = 0; i < V; i++) out[i * V + i] = d[i]
  return fromData(out, [V, V])
}

/**
 * The $V \times V$ graph Laplacian: `unnormalised` $\Lmat = \Dmat - \Amat$ (default), `symmetric`
 * $\Lmat_{\text{sym}} = \Imat - \Dmat^{-1/2}\Amat\Dmat^{-1/2}$, or `random-walk`
 * $\Lmat_{\text{rw}} = \Imat - \Dmat^{-1}\Amat$, with $\Dmat$ the (weighted) out-degrees. In the normalised forms
 * $d^{-1}$ is taken as 0 for a node of degree 0, so its row is zero (no unit on the diagonal either). For an undirected
 * graph with non-negative weights, $\Lmat$ and $\Lmat_{\text{sym}}$ are symmetric positive semidefinite, with one
 * zero eigenvalue per connected component.
 *
 * The graph may also be given as its affinity (weight) matrix $\Wmat$, which is then used as $\Amat$ as it is (a
 * tensor or rows; `weighted` is ignored), as in graph-based semi-supervised learning, where $\Wmat$ comes from a
 * kernel. The diagonal of $\Amat$ (self-loops) counts in the degrees but cancels in the unnormalised $\Lmat$, whose
 * diagonal is $\sum_{j \ne i} A_{ij}$. Throws `ShapeError` for a matrix that is not square.
 *
 * @param g The graph, with $V$ nodes, or its $V \times V$ affinity matrix $\Wmat$.
 * @param options Whether to sum edge weights (the default) or count edges (for a graph only), and `normalisation`, the
 *   form to return (`'unnormalised'` when left out).
 * @returns A float64 $V \times V$ tensor.
 *
 * @example The path 0 - 1 - 2, whose Laplacian rows sum to zero
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * print('L =', laplacian(g))
 * print('row sums =', sum(laplacian(g), 1))
 *
 * @example The normalised forms of the same path
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * print('L_sym =', laplacian(g, { normalisation: 'symmetric' }))
 * print('L_rw =', laplacian(g, { normalisation: 'random-walk' }))
 *
 * @example From an affinity matrix: a weighted path
 * print('L =', laplacian([[0, 1, 0], [1, 0, 2], [0, 2, 0]]))
 */
export function laplacian(
  g: Graph | MatrixLike,
  options: WeightOptions & { normalisation?: 'unnormalised' | 'symmetric' | 'random-walk' } = {},
): Tensor {
  let A: ArrayLike<number>
  let V: Size
  if (isGraph(g)) {
    A = adjacencyMatrix(g, options).data
    V = g.nodes
  } else {
    const m = dense.toMatrixF64(g, 'laplacian')
    if (m.m !== m.n)
      throw new ShapeError('laplacian', `laplacian: the affinity matrix must be square, got ${m.m} × ${m.n}`)
    A = m.data
    V = m.m
  }
  const d = new Float64Array(V)
  for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) d[i] += A[i * V + j]
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
 * True when the argument is a `Graph` (an object with a node count and an edge list) rather than a matrix.
 *
 * @param g A graph, or a matrix as a tensor or rows.
 * @returns Whether it is a graph.
 */
function isGraph(g: Graph | MatrixLike): g is Graph {
  return !isTensor(g) && 'edges' in g && 'nodes' in g
}

/**
 * The $V \times E$ incidence matrix $\Bmat$, column $k$ for edge $k$. Oriented (default for a directed graph): $-w$ at
 * `from` and $+w$ at `to`; unoriented (default for an undirected graph): $+w$ at both ends. $w$ is the edge's weight
 * with `weighted: true`, and 1 otherwise (unweighted is the default here, unlike the other matrices). A self-loop's
 * column is zero. For the oriented, unweighted incidence matrix of an undirected graph,
 * $\Bmat\Bmat^\top = \Lmat$, the unnormalised Laplacian of the edge counts.
 *
 * @param g The graph, with $V$ nodes and $E$ edges.
 * @param options `oriented` chooses the sign convention (left out: oriented exactly when the graph is directed), and
 *   `weighted` puts the edge weights in place of the ones (default false).
 * @returns A float64 $V \times E$ tensor.
 *
 * @example A directed path 0 -> 1 -> 2
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }] }
 * print('B =', incidenceMatrix(g))
 *
 * @example Oriented incidence of an undirected path gives the Laplacian
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const B = incidenceMatrix(g, { oriented: true })
 * print('B =', B)
 * print('B B^T =', matmul(B, transpose(B)))
 * print('L =', laplacian(g))
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

/** Options of `rescaledLaplacian`. */
export interface RescaleOptions {
  /** The Laplacian: `'symmetric'` (default, as in ChebNet) or `'unnormalised'`. */
  normalisation?: 'symmetric' | 'unnormalised'
  /**
   * The upper end $\lambda_{\max}$ of the spectrum. Default 2 for the symmetric Laplacian (its exact bound) and
   * $2 d_{\max}$ for the unnormalised one; any bound at least the largest eigenvalue keeps the rescaled spectrum in
   * $[-1, 1]$.
   */
  lambdaMax?: number
}

/**
 * The rescaled Laplacian $\tilde\Lmat = 2\Lmat/\lambda_{\max} - \Imat$ of an undirected graph, edge weights ignored,
 * whose eigenvalues lie in $[-1, 1]$ when $\lambda_{\max}$ bounds those of $\Lmat$: the argument of the Chebyshev
 * polynomials of a ChebNet filter (Hammond, Vandergheynst and Gribonval 2011, "Wavelets on graphs via spectral graph
 * theory", ACHA 30(2), §6; Defferrard, Bresson and Vandergheynst 2016, §2.1). Throws `DomainError` for a
 * $\lambda_{\max}$ that is not positive.
 *
 * @param g The undirected graph, with $V$ nodes.
 * @param options The Laplacian to rescale and the bound $\lambda_{\max}$.
 * @returns `lTilde`, the float64 $V \times V$ matrix $\tilde\Lmat$, and the `lambdaMax` used.
 *
 * @example The path 0 - 1 - 2, symmetric and unnormalised
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * print('symmetric:', rescaledLaplacian(g).lTilde)
 * const u = rescaledLaplacian(g, { normalisation: 'unnormalised' })
 * print('unnormalised, lambda max', u.lambdaMax, ':', u.lTilde)
 */
export function rescaledLaplacian(g: Graph, options: RescaleOptions = {}): { lTilde: Tensor; lambdaMax: number } {
  const normalisation = options.normalisation ?? 'symmetric'
  const L = laplacian(g, { weighted: false, normalisation }).data
  let lambdaMax = options.lambdaMax
  if (lambdaMax === undefined) {
    if (normalisation === 'symmetric') lambdaMax = 2
    else {
      const degree = new Float64Array(g.nodes)
      for (const e of g.edges)
        if (e.from !== e.to) {
          degree[e.from]++
          degree[e.to]++
        }
      lambdaMax = 2 * Math.max(1, ...degree)
    }
  }
  if (!(lambdaMax > 0))
    throw new DomainError('rescaledLaplacian', `rescaledLaplacian: lambdaMax = ${lambdaMax} must be positive`)
  const V = g.nodes
  const out = new Float64Array(V * V)
  for (let i = 0; i < V * V; i++) out[i] = (2 / lambdaMax) * L[i]
  for (let i = 0; i < V; i++) out[i * V + i] -= 1
  return { lTilde: fromData(out, [V, V]), lambdaMax }
}

/**
 * The incidence matrix $\Hmat$ of a hypergraph, $n \times m$, with $H_{ve} = 1$ when node $v$ is in hyperedge $e$
 * (Zhou, Huang and Schölkopf 2006, "Learning with hypergraphs", NeurIPS). Unlike `incidenceMatrix`, a column may hold
 * any number of ones. Throws `DomainError` for an empty hyperedge, a repeated member or one outside the nodes.
 *
 * @param nodes The number of nodes $n$.
 * @param hyperedges The $m$ hyperedges, each a list of member nodes.
 * @returns A float64 $n \times m$ tensor of zeros and ones.
 *
 * @example Three papers by six authors
 * print('H =', hypergraphIncidence(6, [[0, 1, 2], [2, 3], [3, 4, 5]]))
 */
export function hypergraphIncidence(nodes: Size, hyperedges: Hyperedges): Tensor {
  checkHyperedges(nodes, hyperedges, 'hypergraphIncidence')
  const m = hyperedges.length
  const out = new Float64Array(nodes * m)
  hyperedges.forEach((e, k) => e.forEach((v) => (out[v * m + k] = 1)))
  return fromData(out, [nodes, m])
}
