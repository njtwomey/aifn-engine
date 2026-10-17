/**
 * Graphormer's spatial encoding (Ying et al. 2021, "Do transformers really perform bad for graph representation?",
 * NeurIPS, §3.1.2, eq. 6): full self-attention over the nodes of a graph, with a learned scalar bias
 * $b_{\phi(i, j)}$ added to the score of every pair, indexed by the shortest-path distance $\phi(i, j)$,
 *
 * $$s_{ij} = \frac{\qvec_i^\top\kvec_j}{\sqrt{d_k}} + b_{\phi(i, j)},$$
 *
 * each row a softmax over all nodes, so a node attends to every other node and the graph enters only through the bias.
 * The hop counts come from `hopDistances` (`aifn-compute/graph/traversal`); `spatialBias` turns them into the bias
 * matrix from a table indexed by distance (the last entry used for every longer distance, as a clipped table in
 * practice does, and a separate value for unreachable pairs); `graphormerAttention` runs `scaledDotProductAttention`
 * (`aifn-compute/nn/attention`) with that bias. One head; the paper's centrality and edge encodings are left out.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, matmul, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { scaledDotProductAttention } from 'aifn-compute/nn/attention'

/**
 * The bias matrix $B_{ij} = b_{\min(\phi(i, j), K)}$ from a table $b_0, \dots, b_K$, and `unreachable` where no path
 * exists (an infinite distance). Throws `DomainError` for an empty table or a distance that is negative or not an
 * integer.
 *
 * @param distances The $V \times V$ hop counts, e.g. from `hopDistances`; Infinity where there is no path.
 * @param table The bias per distance, $b_0$ first; longer distances use the last entry.
 * @param unreachable The bias of pairs with no path (default $-\infty$: no attention). The diagonal is always
 *   reachable, so every row keeps some weight.
 * @returns The float64 $V \times V$ bias.
 *
 * @example Distances on a path of three nodes, biased by (0, 0.5, -1)
 * print(spatialBias([[0, 1, 2], [1, 0, 1], [2, 1, 0]], [0, 0.5, -1]))
 */
export function spatialBias(distances: MatrixLike, table: VectorLike, unreachable = -Infinity): Tensor {
  const b = dense.toF64(table, 'spatialBias')
  if (b.length === 0) throw new DomainError('spatialBias', 'spatialBias: the table is empty')
  const { data, m, n } = dense.toMatrixF64(distances, 'spatialBias')
  const last = b.length - 1
  const out = new Float64Array(m * n)
  data.forEach((d, k) => {
    if (!Number.isFinite(d)) out[k] = unreachable
    else if (!(Number.isInteger(d) && d >= 0))
      throw new DomainError('spatialBias', `spatialBias: distance ${d} is not a hop count`)
    else out[k] = b[Math.min(d, last)]
  })
  return fromData(out, [m, n])
}

/** The projections of one Graphormer head. */
export interface GraphormerWeights {
  /** $\Wmat_Q$, $F \times d_k$. */
  query: Value
  /** $\Wmat_K$, $F \times d_k$. */
  key: Value
  /** $\Wmat_V$, $F \times d_v$. */
  value: Value
}

/**
 * One head of Graphormer attention, $\operatorname{softmax}(\Qmat\Kmat^\top/\sqrt{d_k} + \Bmat)\Vmat$ with
 * $\Qmat = \Hmat\Wmat_Q$, $\Kmat = \Hmat\Wmat_K$, $\Vmat = \Hmat\Wmat_V$ and $\Bmat$ the spatial bias. Differentiable
 * in `h`, the weights and the bias.
 *
 * @param h The node features $\Hmat$, $V \times F$.
 * @param weights The projections $\Wmat_Q$, $\Wmat_K$ and $\Wmat_V$.
 * @param bias The $V \times V$ spatial bias, e.g. from `spatialBias`.
 * @returns The `output` ($V \times d_v$) and the attention `weights` ($V \times V$, rows summing to one).
 *
 * @example A bias that favours neighbours moves weight onto them
 * import { hopDistances } from 'aifn-compute/graph/traversal'
 * import { fromEdges } from 'aifn-compute/graph'
 * const g = fromEdges(4, [[0, 1], [0, 2], [1, 2], [2, 3]], { directed: false })
 * const w = tensor([[0.5]])
 * const head = { query: w, key: w, value: w }
 * const h = tensor([[1], [2], [3], [4]])
 * const d = hopDistances(g)
 * print('no bias:', toRows(graphormerAttention(h, head, spatialBias(d, [0])).weights)[0])
 * print('biased:', toRows(graphormerAttention(h, head, spatialBias(d, [0, 0.5, -1])).weights)[0])
 */
export function graphormerAttention(
  h: Value,
  weights: GraphormerWeights,
  bias: Value,
): { output: Value; weights: Value } {
  const q = matmul(h, weights.query)
  const k = matmul(h, weights.key)
  const v = matmul(h, weights.value)
  const r = scaledDotProductAttention(q, k, v, { bias })
  return { output: r.output, weights: r.weights }
}
