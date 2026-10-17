/**
 * Network embedding as matrix factorisation (NetMF; Qiu, Dong, Ma, Li, Wang and Tang, 2018, "Network embedding as
 * matrix factorization: unifying DeepWalk, LINE, PTE, and node2vec", WSDM, Theorem 2.3 and Algorithm 1).
 *
 * For a connected undirected graph with adjacency $\Amat$, degree matrix $\Dmat$, volume
 * $\operatorname{vol}(G) = \sum_i d_i$ and transition matrix $\Pmat = \Dmat^{-1}\Amat$, DeepWalk with window $T$ and
 * $b$ negative samples implicitly factorises
 * $$\log \Mmat, \qquad \Mmat = \frac{\operatorname{vol}(G)}{bT}\Big(\sum_{r=1}^{T}\Pmat^r\Big)\Dmat^{-1},$$
 * the limit of the shifted pointwise mutual information of walk co-occurrences (Levy and Goldberg, 2014, for the
 * shifted-PMI optimum of skip-gram with negative sampling). `netmfMatrix` returns the truncated logarithm
 * $\log\max(\Mmat, 1)$ that NetMF factorises (zero where a pair co-occurs no more than chance predicts). The same
 * quantity estimated from sampled walks is `ppmi(counts, { shift: b })` of `aifn-compute/text/cooccurrence` on the
 * counts of `walkCooccurrence`, and the NetMF embedding $\Umat_d\Sigmamat_d^{1/2}$ is `wordVectors(m, d)` of the same
 * module. Dense matrices only (Algorithm 2's eigendecomposition approximation for large windows is not implemented).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, matmul, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { adjacencyMatrix, degrees } from 'aifn-compute/graph/matrices'

/** Options of `netmfMatrix`. */
export interface NetmfOptions {
  /** Window $T \ge 1$ (default 5). */
  readonly window?: number
  /** Negative samples $b > 0$ (default 1). */
  readonly negatives?: number
  /** Return $\Mmat$ itself instead of $\log\max(\Mmat, 1)$ (default false). */
  readonly raw?: boolean
}

/**
 * The NetMF matrix of an undirected graph: $\log\max(\Mmat, 1)$ (see the file comment), or $\Mmat$ with `raw`. Edge
 * weights count, and a self-loop stays in $\Pmat = \Dmat^{-1}\Amat$ (unlike `randomWalkMatrix` of
 * `aifn-compute/graph/propagation`, which drops it). Throws `DomainError` for a node of degree zero, where
 * $\Dmat^{-1}$ is undefined, a window that is not a positive integer, or negatives that are not positive.
 *
 * @param g The undirected graph, $n$ nodes.
 * @param options The window $T$, the negatives $b$ and whether to skip the truncated logarithm.
 * @returns The $n \times n$ matrix (symmetric for an undirected graph).
 *
 * @example DeepWalk's implicit matrix on a path of four nodes
 * import { fromEdges } from 'aifn-compute/graph'
 * const g = fromEdges(4, [[0, 1], [1, 2], [2, 3]], { directed: false })
 * print('M (T = 2):', netmfMatrix(g, { window: 2, raw: true }))
 * print('log max(M, 1):', netmfMatrix(g, { window: 2 }))
 */
export function netmfMatrix(g: Graph, options: NetmfOptions = {}): Tensor {
  const { window = 5, negatives = 1, raw = false } = options
  if (!(Number.isInteger(window) && window >= 1))
    throw new DomainError('netmfMatrix', 'netmfMatrix: window must be a positive integer')
  if (!(negatives > 0)) throw new DomainError('netmfMatrix', 'netmfMatrix: negatives must be positive')
  const a = dense.data(adjacencyMatrix(g))
  const d = dense.data(degrees(g))
  const n = g.nodes
  if (d.some((v) => !(v > 0))) throw new DomainError('netmfMatrix', 'netmfMatrix: every node needs at least one edge')
  const vol = d.reduce((s, v) => s + v, 0)
  const p = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) p[i * n + j] = a[i * n + j] / d[i]
  const P = fromData(p, [n, n])
  let power: Tensor = P
  const sum = new Float64Array(n * n)
  for (let r = 1; r <= window; r++) {
    const rows = dense.data(power)
    for (let k = 0; k < n * n; k++) sum[k] += rows[k]
    if (r < window) power = matmul(power, P) as Tensor
  }
  const scale = vol / (negatives * window)
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const m = (scale * sum[i * n + j]) / d[j]
      out[i * n + j] = raw ? m : Math.log(Math.max(m, 1))
    }
  return fromData(out, [n, n])
}
