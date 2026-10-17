/**
 * Node encodings that plain message passing cannot compute, read off the graph's matrices (Dwivedi et al. 2022,
 * "Graph neural networks with learnable structural and positional representations", ICLR, §3.3; Dwivedi et al. 2020,
 * "Benchmarking graph neural networks", on Laplacian positional encodings with random sign flips).
 *
 * `randomWalkEncoding` is the random-walk structural encoding (RWSE): for walk lengths $k = 1, \dots, K$, the
 * probability that a simple random walk started at node $v$ is back at $v$ after $k$ steps, the diagonal of
 * $\Pmat^k$ with $\Pmat = \Dmat^{-1}\Amat$. Self-loops stay in $\Amat$, as in `laplacian` (the propagation matrix
 * `randomWalkMatrix` of `aifn-compute/graph/propagation` drops them), and an isolated node walks nowhere and gets 0
 * at every step. The powers are dense, $O(K V^3)$: for molecules and figure-sized graphs.
 *
 * `laplacianEncoding` is the Laplacian positional encoding (LapPE): the eigenvectors of the symmetric normalised
 * Laplacian $\Imat - \Dmat^{-1/2}\Amat\Dmat^{-1/2}$ for the $k$ smallest eigenvalues after the first, one column
 * each. Signs follow `eigh` (largest-magnitude entry positive); the sign, and at a repeated eigenvalue the basis, are
 * arbitrary, which is why training flips them at random (`flipSigns`). On a disconnected graph only one of the zero
 * eigenvalues is skipped.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { eigh } from 'aifn-compute/numerics/linalg'
import type { Graph } from '../graph'
import { adjacencyMatrix, laplacian } from './matrices'

/**
 * Return probabilities of a simple random walk: entry $(v, k - 1)$ is $(\Pmat^k)_{vv}$, with
 * $\Pmat = \Dmat^{-1}\Amat$ and edge weights used. Throws `DomainError` unless `steps` is a positive integer.
 *
 * @param g The graph (undirected for the usual reading), with $V$ nodes.
 * @param steps The number $K$ of walk lengths, $1$ to $K$.
 * @returns A float64 $V \times K$ tensor.
 *
 * @example A triangle returns in two or three steps; a hexagon only in an even number
 * const ring = (n) => [...Array(n).keys()].map((i) => ({ from: i, to: (i + 1) % n }))
 * const cycle = (n) => ({ kind: 'graph', nodes: n, edges: ring(n), directed: false })
 * print('triangle:', randomWalkEncoding(cycle(3), 4))
 * print('hexagon, node 0:', toRows(randomWalkEncoding(cycle(6), 4))[0])
 */
export function randomWalkEncoding(g: Graph, steps: number): Tensor {
  if (!(Number.isInteger(steps) && steps >= 1))
    throw new DomainError('randomWalkEncoding', `randomWalkEncoding: steps = ${steps} must be a positive integer`)
  const n = g.nodes
  const p = Float64Array.from(adjacencyMatrix(g).data)
  for (let i = 0; i < n; i++) {
    let d = 0
    for (let j = 0; j < n; j++) d += p[i * n + j]
    for (let j = 0; j < n; j++) p[i * n + j] = d > 0 ? p[i * n + j] / d : 0
  }
  const out = new Float64Array(n * steps)
  let power: Float64Array = p
  for (let k = 0; k < steps; k++) {
    for (let v = 0; v < n; v++) out[v * steps + k] = power[v * n + v]
    if (k + 1 < steps) power = dense.matMul(power, p, n, n, n)
  }
  return fromData(out, [n, steps])
}

/** The result of `laplacianEncoding`. */
export interface LaplacianEncoding {
  /** $V \times k$: column $j$ is the eigenvector of the $(j + 2)$-th smallest eigenvalue (zero when there is none). */
  encoding: Tensor
  /** The $k$ eigenvalues used, in increasing order (fewer when the graph has $k$ nodes or fewer), float64. */
  values: Tensor
}

/**
 * The Laplacian positional encoding: the eigenvectors of the symmetric normalised Laplacian (edge weights used) for
 * the $k$ smallest eigenvalues after the smallest (0 on a connected graph). Columns are zero when the graph has fewer
 * than $k + 1$ nodes. Throws `DomainError` unless $k$ is a non-negative integer.
 *
 * @param g The undirected graph, with $V$ nodes.
 * @param k The number of eigenvectors.
 * @returns The $V \times k$ encoding and its eigenvalues.
 *
 * @example The 4-path: eigenvalues $1 - \cos(\pi j / 3)$ after 0
 * const edges = [[0, 1], [1, 2], [2, 3]].map(([from, to]) => ({ from, to }))
 * const path = { kind: 'graph', nodes: 4, edges, directed: false }
 * const { encoding, values } = laplacianEncoding(path, 3)
 * print('eigenvalues:', values)
 * print('encoding:', encoding)
 */
export function laplacianEncoding(g: Graph, k: number): LaplacianEncoding {
  if (!(Number.isInteger(k) && k >= 0))
    throw new DomainError('laplacianEncoding', `laplacianEncoding: k = ${k} must be a count`)
  const { values, vectors } = eigh(laplacian(g, { normalisation: 'symmetric' }))
  const vals = toFlat(values)
  const vecs = toFlat(vectors)
  const n = g.nodes
  // eigh returns the eigenvalues in descending order: take them ascending and skip the first.
  const picked = Array.from(vals, (_, i) => i)
    .sort((i, j) => vals[i] - vals[j])
    .slice(1, 1 + k)
  const out = new Float64Array(n * k)
  for (let v = 0; v < n; v++) picked.forEach((c, j) => (out[v * k + j] = vecs[v * n + c]))
  return { encoding: fromData(out, [n, k]), values: fromData(Float64Array.from(picked, (i) => vals[i])) }
}

/**
 * An encoding with the sign of each column flipped with probability $1/2$, as in training-time sign augmentation.
 *
 * @param s The random stream; one uniform is drawn per column, advancing it.
 * @param encoding A $V \times k$ encoding, e.g. `laplacianEncoding(...).encoding`.
 * @returns The flipped copy and the signs used ($\pm 1$ per column).
 *
 * @example Flips keep the absolute values
 * const edges = [[0, 1], [1, 2], [2, 3]].map(([from, to]) => ({ from, to }))
 * const path = { kind: 'graph', nodes: 4, edges, directed: false }
 * const { encoding } = laplacianEncoding(path, 2)
 * const f = flipSigns(stream(4), encoding)
 * print('signs:', f.signs)
 * print('flipped:', f.encoding)
 */
export function flipSigns(s: Stream, encoding: Tensor): { encoding: Tensor; signs: number[] } {
  const [n, k] = encoding.shape
  const u = toFlat(uniform(s, 0, 1, { shape: [k] }))
  const signs = Array.from(u, (x) => (x < 0.5 ? -1 : 1))
  const data = toFlat(encoding)
  const out = new Float64Array(n * k)
  for (let i = 0; i < n * k; i++) out[i] = data[i] * signs[i % k]
  return { encoding: fromData(out, [n, k]), signs }
}
