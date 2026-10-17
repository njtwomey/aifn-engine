/**
 * Shallow node embeddings in the encoder-decoder view (Hamilton, Ying and Leskovec, 2017, "Representation learning on
 * graphs: methods and applications", §2): an embedding table $\Zmat \in \reals^{n \times d}$ (one row per node) and
 * the dot-product decoder $\zvec_i^\top\zvec_j$, fitted so that $\Zmat\Zmat^\top$ reconstructs a chosen symmetric
 * node-similarity matrix $\Smat$ in squared Frobenius error.
 *
 * `nodeSimilarity` builds three standard choices: the adjacency matrix $\Amat$, the two-hop path counts $\Amat^2$,
 * and the window-averaged random-walk co-occurrence (the NetMF matrix of `netmfMatrix`). `dotProductEmbedding` solves
 * the fit exactly: by the Eckart–Young theorem for symmetric matrices (Eckart and Young, 1936), the best
 * $\Zmat\Zmat^\top$ of rank at most $d$ keeps the $d$ largest positive eigenvalues of $\Smat$,
 * $\Zmat = \Vmat_d\Lambdamat_d^{1/2}$; negative eigenvalues cannot be represented by a dot-product decoder. Ahmed et
 * al. (2013) and Ou et al. (2016, HOPE) factorise adjacency and higher-order proximities in this way.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, matmul, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { adjacencyMatrix } from 'aifn-compute/graph/matrices'
import { eigh } from 'aifn-compute/numerics/linalg'
import { netmfMatrix } from './netmf'

/** The similarity a shallow embedding reconstructs. */
export type SimilarityKind = 'adjacency' | 'two-hop' | 'walk'

/**
 * A symmetric node-similarity matrix: `adjacency` $\Amat$; `two-hop` $\Amat^2$, whose entry $(i, j)$ counts the paths
 * of length two (the diagonal holds the degrees of an unweighted graph); `walk` the NetMF matrix $\log\max(\Mmat, 1)$
 * with window `window` and one negative sample.
 *
 * @param g The undirected graph, $n$ nodes.
 * @param kind Which similarity.
 * @param window The window $T$ of the walk similarity (default 3); unused by the other kinds.
 * @returns The $n \times n$ similarity matrix.
 *
 * @example Three similarities of a path of four nodes
 * import { fromEdges } from 'aifn-compute/graph'
 * const g = fromEdges(4, [[0, 1], [1, 2], [2, 3]], { directed: false })
 * print('adjacency:', nodeSimilarity(g, 'adjacency'))
 * print('two-hop:', nodeSimilarity(g, 'two-hop'))
 * print('walk (T = 3):', nodeSimilarity(g, 'walk'))
 */
export function nodeSimilarity(g: Graph, kind: SimilarityKind, window = 3): Tensor {
  const a = adjacencyMatrix(g)
  if (kind === 'adjacency') return a
  if (kind === 'two-hop') return matmul(a, a) as Tensor
  return netmfMatrix(g, { window })
}

/** The result of `dotProductEmbedding`. */
export interface DotProductEmbedding {
  /** The embedding $\Zmat$, $n \times d$. */
  readonly z: Tensor
  /** The eigenvalues of $\Smat$, descending. */
  readonly eigenvalues: number[]
  /** $\lVert \Smat - \Zmat\Zmat^\top \rVert_F^2 / \lVert \Smat \rVert_F^2$, in $[0, 1]$. */
  readonly relativeError: number
}

/**
 * The rank-$d$ dot-product embedding of a symmetric similarity matrix (see the file comment): the eigenvectors of the
 * $d$ largest eigenvalues, each scaled by the square root of its eigenvalue. A column whose eigenvalue is not positive
 * is left at zero. Throws `DomainError` unless $d$ is an integer from 1 to $n$.
 *
 * @param s The symmetric $n \times n$ similarity $\Smat$ (only its lower triangle is read by the eigensolver).
 * @param dim The dimension $d$, from 1 to $n$.
 * @returns The embedding, the eigenvalues and the relative reconstruction error.
 *
 * @example A rank-two similarity is reconstructed exactly in two dimensions
 * const z0 = tensor([[1, 0], [1, 1], [0, 2]])
 * const S = matmul(z0, transpose(z0))
 * const fit = dotProductEmbedding(S, 2)
 * print('eigenvalues:', fit.eigenvalues)
 * print('relative error d = 2:', fit.relativeError, ' d = 1:', dotProductEmbedding(S, 1).relativeError)
 */
export function dotProductEmbedding(s: MatrixLike, dim: number): DotProductEmbedding {
  const { data: S, m: n, n: cols } = dense.toMatrixF64(s, 'dotProductEmbedding')
  if (cols !== n) throw new ShapeError('dotProductEmbedding', 'dotProductEmbedding: the similarity must be square')
  if (!(Number.isInteger(dim) && dim >= 1 && dim <= n))
    throw new DomainError('dotProductEmbedding', 'dotProductEmbedding: dim must be an integer from 1 to n')
  const { values, vectors } = eigh(fromData(S, [n, n]))
  const lambda = Array.from(dense.data(values as Tensor))
  const V = dense.data(vectors as Tensor)
  const z = new Float64Array(n * dim)
  for (let i = 0; i < n; i++)
    for (let k = 0; k < dim; k++) z[i * dim + k] = lambda[k] > 0 ? V[i * n + k] * Math.sqrt(lambda[k]) : 0
  const zt = fromData(z, [n, dim])
  return { z: zt, eigenvalues: lambda, relativeError: reconstructionError(fromData(S, [n, n]), zt) }
}

/**
 * $\lVert \Smat - \Zmat\Zmat^\top \rVert_F^2 / \lVert \Smat \rVert_F^2$ for any embedding rows $\Zmat$ (e.g. one moved
 * by hand); 0 when $\Smat$ is zero. Throws `ShapeError` when the rows of $\Zmat$ do not match $\Smat$.
 *
 * @param S The $n \times n$ similarity.
 * @param z The $n \times d$ embedding rows.
 * @returns The relative squared error of the dot-product reconstruction.
 *
 * @example The error of an embedding moved by hand
 * const S = tensor([[2, 1], [1, 2]])
 * const a = Math.sqrt(1.5)
 * const b = Math.sqrt(0.5)
 * print('exact:', reconstructionError(S, [[a, b], [a, -b]]))
 * print('top eigenvector only:', reconstructionError(S, [[a], [a]]))
 * print('moved by hand:', reconstructionError(S, [[1], [1]]))
 */
export function reconstructionError(S: MatrixLike, z: MatrixLike): number {
  const { data: s, m: n } = dense.toMatrixF64(S, 'reconstructionError')
  const { data: zd, m: rows, n: d } = dense.toMatrixF64(z, 'reconstructionError')
  if (rows !== n) throw new ShapeError('reconstructionError', 'reconstructionError: z needs one row per node of S')
  let err = 0
  let norm = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let dot = 0
      for (let k = 0; k < d; k++) dot += zd[i * d + k] * zd[j * d + k]
      err += (s[i * n + j] - dot) ** 2
      norm += s[i * n + j] ** 2
    }
  return norm > 0 ? err / norm : 0
}
