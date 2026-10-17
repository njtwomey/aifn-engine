/**
 * The out-of-sample extension of graph label propagation (Delalleau, Bengio and Le Roux, 2005, "Efficient
 * non-parametric function induction in semi-supervised learning", AISTATS, §3, eq. 7): class scores for new points
 * from the scores already computed on the graph's nodes, without solving the graph's linear system again.
 *
 * A new point $\xvec$ joins the graph with Gaussian weights $w_j = \exp(-\gamma \lVert \xvec - \xvec_j \rVert^2)$ to
 * the nodes (optionally only to its $k$ nearest, matching a $k$-NN affinity such as `pointAffinity` with
 * `neighbours`), the node scores are held fixed, and the harmonic energy of the new point alone is minimised. The
 * minimiser is the weighted average of the neighbours' score rows,
 *
 * $$\tilde\fvec(\xvec) = \frac{\sum_j w_j \fvec_j}{\sum_j w_j},$$
 *
 * a Nadaraya–Watson (Parzen window) regression on the propagated scores. On an unlabelled node it reproduces the
 * harmonic score. The weights are computed relative to the nearest node, $\exp(-\gamma(d_j^2 - d_{\min}^2))$, which
 * leaves the average unchanged and keeps far queries from underflowing to $0/0$.
 */

import type { MatrixLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData } from 'aifn-compute/foundation/tensor'
import { squaredDistances } from 'aifn-compute/numerics/linalg'
import { bruteForceNeighbours } from 'aifn-compute/numerics/neighbours'
import type { LabelScores } from './labels'

/** Options of {@link harmonicExtension}. */
export interface HarmonicExtensionOptions {
  /** The Gaussian kernel's rate $\gamma > 0$ in $\exp(-\gamma \lVert \xvec - \xvec_j \rVert^2)$. */
  gamma: number
  /** Average over each query's $k$ nearest nodes only (ties to the smaller index); 0 or absent uses every node. */
  neighbours?: number
}

/**
 * Scores for new points as the kernel-weighted average of the node scores; see the file comment. Pure and
 * worker-safe. Throws `DomainError` unless $\gamma > 0$, and `ShapeError` when `x` and `scores` differ in rows.
 *
 * @param x The graph's nodes, $n \times d$.
 * @param scores The node scores, $n \times C$ (from `harmonicLabels` or `labelSpreading`).
 * @param queries The new points, $m \times d$.
 * @param options The kernel rate $\gamma$ and the number of neighbours.
 * @returns The extended scores ($m \times C$, rows summing to one when the node rows do; not renormalised) and the
 *   class of each query's largest score (ties to the lower class; $-1$ for an all-zero row).
 *
 * @example The extension reproduces the harmonic score at a node
 * const x = [[0], [1], [2], [3]]
 * const W = pointAffinity(x, { gamma: 0.5 })
 * const { scores } = harmonicLabels(W, [0, -1, -1, 1])
 * print('harmonic scores:', scores)
 * print('extension at 1 and 1.5:', harmonicExtension(x, scores, [[1], [1.5]], { gamma: 0.5 }).scores)
 */
export function harmonicExtension(
  x: MatrixLike,
  scores: MatrixLike,
  queries: MatrixLike,
  options: HarmonicExtensionOptions,
): LabelScores {
  const { gamma } = options
  if (!(gamma > 0)) throw new DomainError('harmonicExtension', 'harmonicExtension: gamma must be positive')
  const { data: F, m: n, n: C } = dense.toMatrixF64(scores, 'harmonicExtension')
  const d2t = squaredDistances(x, queries)
  if (d2t.shape[0] !== n) throw new ShapeError('harmonicExtension', 'harmonicExtension: x and scores differ in rows')
  const m = d2t.shape[1]
  const d2 = dense.data(d2t)
  const k = options.neighbours && options.neighbours > 0 ? Math.min(options.neighbours, n) : n
  // Each query's nodes, nearest first (all of them, in order, when k = n).
  const near =
    k < n && m > 0
      ? dense.data(bruteForceNeighbours(x, queries, k).indices)
      : Float64Array.from({ length: n }, (_, j) => j)
  const out = new Float64Array(m * C)
  const labels = new Int32Array(m)
  for (let q = 0; q < m; q++) {
    const at = (r: number) => (k < n ? near[q * k + r] : near[r])
    let min = Infinity
    for (let r = 0; r < k; r++) min = Math.min(min, d2[at(r) * m + q])
    let total = 0
    for (let r = 0; r < k; r++) {
      const j = at(r)
      const w = Math.exp(-gamma * (d2[j * m + q] - min))
      total += w
      for (let c = 0; c < C; c++) out[q * C + c] += w * F[j * C + c]
    }
    let best = -1
    for (let c = 0; c < C; c++) {
      out[q * C + c] /= total
      if (out[q * C + c] > 0 && (best < 0 || out[q * C + c] > out[q * C + best])) best = c
    }
    labels[q] = best
  }
  return { scores: fromData(out, [m, C]), labels: fromData(labels, [m]) }
}
