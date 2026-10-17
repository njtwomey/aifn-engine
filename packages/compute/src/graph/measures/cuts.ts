/**
 * The Cheeger constant of an undirected graph, the size of its worst bottleneck:
 *
 * $$
 * h_G = \min_{\emptyset \ne S \subset V}
 *   \frac{\lvert\partial S\rvert}{\min(\operatorname{vol} S, \operatorname{vol} \bar S)},
 * $$
 *
 * with $\lvert\partial S\rvert$ the weight of the edges leaving $S$ and $\operatorname{vol} S$ the sum of the degrees
 * in $S$ (Chung 1997, "Spectral Graph Theory", §2.2). By the Cheeger inequality it is bracketed by the spectral gap
 * $\lambda_1$ of the normalised Laplacian, $\lambda_1 / 2 \le h_G \le \sqrt{2\lambda_1}$, and a small $h_G$ is what
 * squashes messages in a graph network (Topping et al. 2022, "Understanding over-squashing and bottlenecks on graphs
 * via curvature", ICLR, eq. 5 and 6).
 *
 * It is computed exactly, by enumerating the $2^{V - 1} - 1$ sets $S$ that leave out the last node (a set and its
 * complement give the same ratio) in Gray-code order, so that each set differs from the last by one node and the cut
 * and volume are updated in $O(V)$: $O(2^V V)$ in all, for small graphs only.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { isDirected, type Graph } from '../graph'
import { adjacencyMatrix } from 'aifn-compute/graph/matrices'

/** The largest graph `cheegerConstant` enumerates: $2^{21}$ sets of 22 nodes, about a tenth of a second. */
export const CHEEGER_MAX_NODES: Size = 22

/** The result of `cheegerConstant`. */
export interface CheegerResult {
  /** The Cheeger constant $h_G$: 0 for a disconnected graph, Infinity when every set has zero volume. */
  value: number
  /**
   * A minimising set $S$ (int32 node indices, ascending), the one with the smallest bit mask among ties; it never
   * holds the last node.
   */
  set: Tensor
}

/**
 * The exact Cheeger constant $h_G$ of an undirected graph (see the file comment), with a minimising set. Edge weights
 * count in the cut and the degrees. Throws `DomainError` for a directed graph, fewer than two nodes, or more than
 * `CHEEGER_MAX_NODES` nodes. At the limit it takes about a tenth of a second: call it from a worker for large inputs.
 *
 * @param g The undirected graph, with $V$ nodes.
 * @returns The constant and one set attaining it.
 *
 * @example A path of four nodes is cut in the middle
 * import { chainGraph } from 'aifn-compute/graph/structures'
 * const { value, set } = cheegerConstant(chainGraph(4))
 * print('h =', value, ' S =', set)
 *
 * @example The barbell's bottleneck is its path
 * import { barbellGraph } from 'aifn-compute/graph/structures'
 * print('h =', cheegerConstant(barbellGraph(5, 3)).value)
 */
export function cheegerConstant(g: Graph): CheegerResult {
  const where = 'cheegerConstant'
  if (isDirected(g)) throw new DomainError(where, `${where}: the graph must be undirected`)
  const n = g.nodes
  if (n < 2 || n > CHEEGER_MAX_NODES)
    throw new DomainError(where, `${where}: enumeration needs 2 to ${CHEEGER_MAX_NODES} nodes, got ${n}`)
  const a = adjacencyMatrix(g).data
  const d = new Float64Array(n)
  let total = 0
  for (let i = 0; i < n; i++) {
    // A self-loop never crosses a cut, but counts in the degree, as in the adjacency matrix's row sums.
    for (let j = 0; j < n; j++) d[i] += a[i * n + j]
    total += d[i]
  }
  const inS = new Uint8Array(n)
  let cut = 0
  let vol = 0
  let mask = 0
  let best = Infinity
  let bestMask = 0
  for (let k = 1; k < 2 ** (n - 1); k++) {
    // Gray code: step k flips the node of the lowest set bit of k.
    const v = 31 - Math.clz32(k & -k)
    const sign = inS[v] ? -1 : 1
    for (let j = 0; j < n; j++) if (j !== v) cut += sign * (inS[j] ? -a[v * n + j] : a[v * n + j])
    inS[v] ^= 1
    mask ^= 1 << v
    vol += sign * d[v]
    const denom = Math.min(vol, total - vol)
    const h = denom > 0 ? cut / denom : Infinity
    if (h < best || (h === best && mask < bestMask)) {
      best = h
      bestMask = mask
    }
  }
  const set: number[] = []
  for (let i = 0; i < n; i++) if (bestMask & (1 << i)) set.push(i)
  return { value: best, set: fromData(Int32Array.from(set)) }
}
