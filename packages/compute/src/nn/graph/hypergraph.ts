/**
 * Message passing on a hypergraph: $n$ nodes and $m$ hyperedges, each hyperedge a set of nodes, recorded in the
 * incidence matrix $\Hmat \in \{0, 1\}^{n \times m}$ (`hypergraphIncidence` of `aifn-compute/graph/matrices`).
 *
 * - `hypergraphMeanPass` is the two-stage mean, node to hyperedge then hyperedge to node, run as two rounds of
 *   `propagate` on the star expansion (`starExpansion` of `aifn-compute/graph/structures`).
 * - `hgnnConv` is the HGNN layer of Feng, You, Zhang, Ji and Gao (2019), "Hypergraph neural networks", AAAI, eq. 10
 *   and 11, before the activation:
 *
 * $$\Dmat_v^{-1/2}\Hmat\Wmat\Dmat_e^{-1}\Hmat^\top\Dmat_v^{-1/2}\Xmat\Thetamat,$$
 *
 * with vertex degrees $d(v) = \sum_e w_e H_{ve}$ and edge degrees $\delta(e) = \sum_v H_{ve}$; the normalised
 * hypergraph Laplacian behind it is Zhou, Huang and Schölkopf (2006), "Learning with hypergraphs", NeurIPS. The paper
 * learns no hyperedge weights in its experiments; `edgeWeights` is a fixed input here.
 *
 * Hyperedges are validated (no empty hyperedge, no repeated or out-of-range member), so $\delta(e) \ge 1$ and no
 * division by zero can occur; a node in no hyperedge gets zeros. Both are differentiable in the features and the
 * weights.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import {
  concat,
  fromData,
  matmul,
  mul,
  shapeOfValue,
  slice,
  transpose,
  zeros,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Hyperedges } from 'aifn-compute/graph'
import { hypergraphIncidence } from 'aifn-compute/graph/matrices'
import { propagate } from 'aifn-compute/graph/propagation'
import { starExpansion } from 'aifn-compute/graph/structures'

/**
 * The two-stage mean: each hyperedge takes the mean of its members' vectors, then each node the mean of its
 * hyperedges' vectors. A node in no hyperedge gets zeros. Differentiable in `h`. Throws `DomainError` for invalid
 * hyperedges.
 *
 * @param h The node features, $n \times F$.
 * @param hyperedges The $m$ hyperedges, each a list of member nodes.
 * @returns The hyperedge vectors `edges` ($m \times F$) and the new node vectors `nodes` ($n \times F$).
 *
 * @example Three papers by six authors with features 1 to 6
 * const pass = hypergraphMeanPass(tensor([[1], [2], [3], [4], [5], [6]]), [[0, 1, 2], [2, 3], [3, 4, 5]])
 * print('paper means:', pass.edges)
 * print('author means:', pass.nodes)
 */
export function hypergraphMeanPass(h: Value, hyperedges: Hyperedges): { edges: Value; nodes: Value } {
  const [n, f] = shapeOfValue(h)
  const m = hyperedges.length
  const { toEdges, toNodes } = starExpansion(n, hyperedges)
  const padded = concat([h, zeros([m, f])], 0)
  const plain = { message: (source: Value) => source }
  const atEdges = propagate(toEdges, padded, { ...plain, aggregate: 'mean' })
  const atNodes = propagate(toNodes, atEdges, { ...plain, aggregate: 'mean' })
  return { edges: slice(atEdges, [n, n + m]), nodes: slice(atNodes, [0, n]) }
}

/**
 * The HGNN hyperedge convolution before the activation (see the file comment). Differentiable in `x` and `theta`.
 * Throws `DomainError` for invalid hyperedges or edge weights that do not match them.
 *
 * @param x The node features $\Xmat$, $n \times F$.
 * @param hyperedges The $m$ hyperedges.
 * @param theta The weight $\Thetamat$, $F \times G$.
 * @param edgeWeights The hyperedge weights $w_e$ (default 1 each).
 * @returns The new features, $n \times G$.
 *
 * @example With $\Thetamat = 1$ on the six authors
 * print(hgnnConv(tensor([[1], [2], [3], [4], [5], [6]]), [[0, 1, 2], [2, 3], [3, 4, 5]], tensor([[1]])))
 */
export function hgnnConv(x: Value, hyperedges: Hyperedges, theta: Value, edgeWeights?: readonly number[]): Value {
  const n = shapeOfValue(x)[0]
  const m = hyperedges.length
  const w = edgeWeights ?? hyperedges.map(() => 1)
  if (w.length !== m) throw new DomainError('hgnnConv', `hgnnConv: ${w.length} edge weights for ${m} hyperedges`)
  const inc = hypergraphIncidence(n, hyperedges)
  const H = inc.data
  const dv = new Float64Array(n)
  for (let v = 0; v < n; v++) for (let e = 0; e < m; e++) dv[v] += H[v * m + e] * w[e]
  const scaleV = fromData(
    dv.map((d) => (d > 0 ? 1 / Math.sqrt(d) : 0)),
    [n, 1],
  )
  const edgeScale = fromData(
    Float64Array.from(hyperedges, (e, k) => w[k] / e.length),
    [m, 1],
  )
  const xs = mul(scaleV, matmul(x, theta)) // D_v^{-1/2} X Θ
  const atEdges = mul(edgeScale, matmul(transpose(inc), xs)) // W D_e^{-1} Hᵀ (…)
  return mul(scaleV, matmul(inc, atEdges)) // D_v^{-1/2} H (…)
}
