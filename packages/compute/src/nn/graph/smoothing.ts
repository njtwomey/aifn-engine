/**
 * What depth does to a graph network: the dense GCN propagation matrix, its spectrum, the Dirichlet energy that
 * measures oversmoothing, PairNorm and DropEdge against it, and the sensitivity of one node to another through $r$
 * layers that measures oversquashing.
 *
 * - `gcnMatrix` is the dense $V \times V$ form of `gcnCoefficients`,
 *   $\hat\Amat = \tilde\Dmat^{-1/2}(\Amat + \Imat)\tilde\Dmat^{-1/2}$ by default (Kipf and Welling 2017, eq. 2), with
 *   the same options (normalisation, self-loops, edge weights). Row $v$, column $u$ holds $\hat A_{vu}$, the weight of
 *   the message $u \to v$.
 * - `adjacencySpectrum` gives its eigenvalues $1 = \mu_1 \ge \dots \ge \mu_V$ and the contraction
 *   $\lambda = \max(\lvert\mu_2\rvert, \lvert\mu_V\rvert)$, which bounds the decay of the energy per layer:
 *   $E(\hat\Amat\Hmat) \le \lambda^2 E(\Hmat)$ (Cai and Wang 2020, "A note on over-smoothing for graph neural
 *   networks", ICML GRL+ workshop).
 * - `dirichletEnergy` is the degree-normalised Dirichlet energy
 *   $E(\Hmat) = \frac12\sum_{i \ne j}\tilde A_{ij}\lVert\hvec_i/\sqrt{\tilde d_i} - \hvec_j/\sqrt{\tilde d_j}\rVert^2$,
 *   zero exactly when every row of $\Hmat$ is $\sqrt{\tilde d_i}$ times one common vector (on a connected graph).
 * - `pairNorm` (Zhao and Akoglu 2020, "PairNorm: tackling oversmoothing in GNNs", ICLR, §4, the PairNorm-SI form)
 *   centres the columns and rescales so that the mean squared row norm is $s^2$.
 * - `dropEdge` (Rong et al. 2020, "DropEdge: towards deep graph convolutional networks on node classification",
 *   ICLR) keeps each edge with probability $1 - p$.
 * - `propagationSensitivity` gives the entries $(\hat\Amat^r)_{vu}$, $r = 0, \dots, R$, which bound the Jacobian
 *   $\lvert\partial \hvec_v^{(r)} / \partial \xvec_u\rvert$ of an $r$-layer network up to a factor set by the weights
 *   (Topping et al. 2022, "Understanding over-squashing and bottlenecks on graphs via curvature", ICLR, lemma 1).
 *
 * The matrices are dense: for graphs of up to a few hundred nodes.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import {
  dense,
  div,
  fromData,
  maximum,
  mean,
  mul,
  shapeOfValue,
  sqrt,
  square,
  sub,
  sum,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { eigh } from 'aifn-compute/numerics/linalg'
import { gcnCoefficients, type GraphConvOptions } from './layers'

/**
 * The dense GCN propagation matrix $\hat\Amat$ (see the file comment): `gcnCoefficients` scattered into a
 * $V \times V$ matrix, row $v$ and column $u$ for the message $u \to v$. A node of degree 0 gets a zero row.
 *
 * @param g The graph.
 * @param options The normalisation (default `symmetric`), self-loops (default true) and edge weights (default false),
 *   as for `graphConv`.
 * @returns The float64 $V \times V$ matrix.
 *
 * @example The path 0 - 1 - 2: $\hat A_{01} = 1/\sqrt{6}$
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * print('A hat =', gcnMatrix(chainGraph(3)))
 * print('1 / sqrt(6) =', 1 / Math.sqrt(6))
 */
export function gcnMatrix(g: Graph, options: GraphConvOptions = {}): Tensor {
  const { edges, coefficients } = gcnCoefficients(g, options)
  const c = toFlat(coefficients)
  const V = g.nodes
  const out = new Float64Array(V * V)
  edges.source.forEach((u, k) => (out[edges.destination[k] * V + u] += c[k]))
  return fromData(out, [V, V])
}

/**
 * The eigenvalues of the symmetric GCN matrix in descending order and the contraction factor
 * $\lambda = \max(\lvert\mu_2\rvert, \lvert\mu_V\rvert)$ (see the file comment); $\lambda = 0$ for a single node.
 * Throws `DomainError` for a directed graph, whose matrix is not symmetric.
 *
 * @param g The undirected graph.
 * @param options Self-loops (default true) and edge weights (default false); the normalisation is symmetric.
 * @returns The eigenvalues (a float64 vector, descending) and $\lambda$.
 *
 * @example The path 0 - 1 - 2: eigenvalues 1, 1/2 and -1/6
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * const { values, contraction } = adjacencySpectrum(chainGraph(3))
 * print('eigenvalues:', values, ' contraction:', contraction)
 */
export function adjacencySpectrum(
  g: Graph,
  options: Omit<GraphConvOptions, 'normalisation'> = {},
): { values: Tensor; contraction: number } {
  if (g.directed !== false)
    throw new DomainError('adjacencySpectrum', 'adjacencySpectrum: the graph must be undirected')
  const values = Float64Array.from(toFlat(eigh(gcnMatrix(g, { ...options, normalisation: 'symmetric' })).values))
  values.sort((x, y) => y - x)
  const n = values.length
  const contraction = n > 1 ? Math.max(Math.abs(values[1]), Math.abs(values[n - 1])) : 0
  return { values: fromData(values), contraction }
}

/**
 * The degree-normalised Dirichlet energy of node features (see the file comment), with $\tilde\Amat = \Amat + \Imat$
 * and its degrees by default; the self-loops count in the degrees only. Throws `ShapeError` unless `h` has one row
 * per node.
 *
 * @param g The undirected graph.
 * @param h The node features $\Hmat$, $V \times F$ (a tensor or rows).
 * @param options Whether self-loops count in the degrees (default true) and whether edge weights are used (default
 *   false), as for `gcnMatrix`.
 * @returns The energy, at least 0.
 *
 * @example Smooth features have low energy, alternating ones high
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * const g = chainGraph(4)
 * print('constant:', dirichletEnergy(g, [[1], [1], [1], [1]]))
 * print('alternating:', dirichletEnergy(g, [[1], [-1], [1], [-1]]))
 */
export function dirichletEnergy(
  g: Graph,
  h: MatrixLike,
  options: Pick<GraphConvOptions, 'selfLoops' | 'weighted'> = {},
): number {
  const { data: x, m, n: F } = dense.toMatrixF64(h, 'dirichletEnergy')
  const V = g.nodes
  if (m !== V) throw new ShapeError('dirichletEnergy', `dirichletEnergy: features must have ${V} rows, got ${m}`)
  // The unnormalised matrix with self-loops: A + I, whose row sums are the degrees.
  const a = toFlat(gcnMatrix(g, { ...options, normalisation: 'none' }))
  const d = new Float64Array(V)
  for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) d[i] += a[i * V + j]
  let e = 0
  for (let i = 0; i < V; i++)
    for (let j = 0; j < V; j++) {
      if (i === j || a[i * V + j] === 0 || d[i] === 0 || d[j] === 0) continue
      const si = 1 / Math.sqrt(d[i])
      const sj = 1 / Math.sqrt(d[j])
      let s = 0
      for (let k = 0; k < F; k++) s += (x[i * F + k] * si - x[j * F + k] * sj) ** 2
      e += 0.5 * a[i * V + j] * s
    }
  return e
}

/**
 * PairNorm: subtract the column means, then rescale every row by one factor so that the mean squared row norm is
 * $s^2$; the total squared distance between all pairs of rows is then $2n^2 s^2$ whatever the input. Rows that are
 * all equal give zeros. Differentiable in `h`. Throws `ShapeError` unless `h` is a matrix.
 *
 * @param h The node features, $V \times F$.
 * @param scale The target root-mean-square row norm $s > 0$.
 * @returns The normalised features, $V \times F$.
 *
 * @example Zero column means and a mean squared row norm of $s^2 = 4$
 * const y = pairNorm(tensor([[1, 2], [3, 4], [5, 9]]), 2)
 * print('PairNorm:', y)
 * print('column means:', mean(y, 0), ' mean squared norm:', div(sum(square(y)), 3))
 */
export function pairNorm(h: Value, scale = 1): Value {
  const shape = shapeOfValue(h)
  if (shape.length !== 2) throw new ShapeError('pairNorm', `pairNorm: expected a matrix, got [${shape.join(', ')}]`)
  const centred = sub(h, mean(h, 0, true))
  const msq = div(sum(square(centred)), shape[0])
  // A floor in place of 0 keeps all-equal rows at zero instead of 0/0.
  return div(mul(centred, scale), sqrt(maximum(msq, Number.MIN_VALUE)))
}

/**
 * DropEdge: a copy of the graph with each edge kept independently with probability $1 - p$, same nodes. Throws
 * `DomainError` for $p$ outside $[0, 1]$.
 *
 * @param s The random stream; one uniform is drawn per edge, in edge order, advancing it (an edge is kept when its
 *   uniform is at least $p$).
 * @param g The graph.
 * @param p The drop probability.
 * @returns The thinned graph.
 *
 * @example Dropping about half of a 6-cycle's edges
 * import { cycleGraph } from 'aifn-compute/graph/structures'
 * print('kept:', dropEdge(stream(0), cycleGraph(6), 0.5).edges.map((e) => [e.from, e.to]))
 */
export function dropEdge(s: Stream, g: Graph, p: number): Graph {
  if (!(p >= 0 && p <= 1)) throw new DomainError('dropEdge', `dropEdge: p = ${p} must lie in [0, 1]`)
  const u = g.edges.length ? toFlat(uniform(s, 0, 1, { shape: [g.edges.length] })) : new Float64Array(0)
  return { ...g, edges: g.edges.filter((_, k) => u[k] >= p) }
}

/**
 * The sensitivity of one target node to every node through $r$ layers: entry $(r, u)$ is $(\hat\Amat^r)_{vu}$ for
 * $r = 0, \dots, R$, with $\hat\Amat$ the symmetric GCN matrix with self-loops (see the file comment). Row 0 is the
 * indicator of $v$; entry $(r, u)$ is 0 when $u$ is more than $r$ hops from $v$. Throws `DomainError` for a target
 * outside the graph or a negative depth.
 *
 * @param g The undirected graph.
 * @param target The node $v$.
 * @param depth The largest power $R$.
 * @returns A float64 $(R + 1) \times V$ tensor.
 *
 * @example A path of five nodes: the far end is reached at the fourth power
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * print(propagationSensitivity(chainGraph(5), 4, 4))
 */
export function propagationSensitivity(g: Graph, target: number, depth: Size): Tensor {
  const where = 'propagationSensitivity'
  const n = g.nodes
  if (!(Number.isInteger(target) && target >= 0 && target < n))
    throw new DomainError(where, `${where}: target ${target} is outside 0…${n - 1}`)
  if (!(Number.isInteger(depth) && depth >= 0)) throw new DomainError(where, `${where}: depth must be a count`)
  const m = toFlat(gcnMatrix(g))
  const out = new Float64Array((depth + 1) * n)
  out[target] = 1
  // Row v of A^r, built as (e_v^T A) A ...; A is symmetric, so this is also column v.
  for (let r = 1; r <= depth; r++)
    for (let k = 0; k < n; k++) {
      const x = out[(r - 1) * n + k]
      if (x !== 0) for (let u = 0; u < n; u++) out[r * n + u] += x * m[k * n + u]
    }
  return fromData(out, [depth + 1, n])
}
