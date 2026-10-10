/**
 * Spectral clustering (Ng, Jordan and Weiss, 2002, "On spectral clustering: analysis and an algorithm"; von Luxburg,
 * 2007, "A tutorial on spectral clustering"): an affinity graph $\Wmat$, the top $k$ eigenvectors of the normalised
 * affinity $\Dmat^{-1/2} \Wmat \Dmat^{-1/2}$ ($\Dmat$ the diagonal matrix of degrees), rows scaled to unit length,
 * and k-means on those rows. Everything is dense: the affinity and its eigendecomposition are $n \times n$.
 */

import type { Dataset, Estimator, FitOptions } from 'aifn-compute/learning/estimators'
import { connectedComponents } from 'aifn-compute/graph/traversal'
import { fromEdges } from 'aifn-compute/graph'
import { eigh } from 'aifn-compute/numerics/linalg'
import { dataset } from 'aifn-compute/learning/estimators'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { kmeans } from './centroid'
import { mat, matrix, pairwise, vec } from './util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, space } from 'aifn-compute/foundation/space'

/**
 * How affinities are built: a Gaussian (RBF) kernel of lengthscale $\ell$ (`lengthscale`), or a symmetric
 * $k$-nearest-neighbour graph (`k` neighbours per point, itself excluded).
 */
export type Affinity = { kind: 'rbf'; lengthscale: number } | { kind: 'neighbours'; k: number }

/**
 * The affinity matrix $\Wmat$ (zero diagonal) of the rows of `x`: $W_{ij} = \exp(-\lVert \xvec_i - \xvec_j
 * \rVert^2 / 2\ell^2)$, or the $k$-nearest-neighbour connectivity $\Amat$ ($A_{ij} = 1$ when $j$ is among the $k$
 * nearest of $i$, ties to the lower index) symmetrised as $\frac{1}{2}(\Amat + \Amat^\top)$, as scikit-learn: 1
 * where each point is among the other's $k$ nearest, $\frac{1}{2}$ where only one is.
 *
 * @param x The data, $n \times d$, one point per row.
 * @param affinity The kernel and its lengthscale, or the number of neighbours.
 * @returns $\Wmat$, $n \times n$ and symmetric.
 *
 * @example Nearest-neighbour affinity of three points on a line
 * const x = tensor([[0], [1], [3]])
 * print('1 neighbour', affinityMatrix(x, { kind: 'neighbours', k: 1 }))
 * print('rbf', affinityMatrix(x, { kind: 'rbf', lengthscale: 1 }))
 */
export function affinityMatrix(x: Tensor, affinity: Affinity): Tensor {
  const { n } = matrix(x, 'affinityMatrix')
  const D = pairwise(x)
  const W = new Float64Array(n * n)
  if (affinity.kind === 'rbf') {
    const l2 = affinity.lengthscale ** 2
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) if (i !== j) W[i * n + j] = Math.exp(-(D[i * n + j] ** 2) / (2 * l2))
  } else {
    for (let i = 0; i < n; i++) {
      const order = Array.from({ length: n }, (_, j) => j)
        .filter((j) => j !== i)
        .sort((a, b) => D[i * n + a] - D[i * n + b] || a - b)
      for (const j of order.slice(0, affinity.k)) {
        W[i * n + j] += 0.5
        W[j * n + i] += 0.5
      }
    }
  }
  return mat(W, n, n)
}

/** A fitted spectral clustering. */
export interface SpectralClusteringModel {
  readonly kind: 'model'
  /** Spectral clustering partitions the training rows only: it cannot place new inputs. */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'spectral-clustering'
  /** The cluster of each training row, $n$ values (int32). */
  readonly labels: Tensor
  /** The affinity matrix $\Wmat$, $n \times n$. */
  readonly affinity: Tensor
  /**
   * The rows fed to k-means, $n \times k$: top eigenvectors of $\Dmat^{-1/2} \Wmat \Dmat^{-1/2}$, each row scaled to
   * unit length unless `normaliseRows` is false.
   */
  readonly embedding: Tensor
  /**
   * The $k$ largest eigenvalues of $\Dmat^{-1/2} \Wmat \Dmat^{-1/2}$ (1 has multiplicity equal to the number of
   * components).
   */
  readonly eigenvalues: Tensor
  /**
   * Connected components of the affinity graph (`aifn-compute/graph`): more than $k$ of them means the embedding is
   * degenerate.
   */
  readonly components: number
}

/**
 * Spectral clustering into $k$ groups (Ng, Jordan and Weiss, 2002). k-means on the embedding uses `restarts`
 * k-means++ runs (default 10) from the fit's stream (`stream(0)` when none is given). A point with no affinity to any
 * other (degree 0) gets a zero row in the embedding. The model is transductive: it labels the training rows only.
 *
 * @param params The hyperparameters.
 * @param params.k The number of clusters, which is also the number of eigenvectors kept.
 * @param params.affinity How the affinity graph is built (default an RBF kernel of lengthscale 1).
 * @param params.restarts The k-means++ restarts of k-means on the embedding (default 10).
 * @param params.normaliseRows Whether each row of the embedding is scaled to unit length before k-means (default
 *   true, as Ng, Jordan and Weiss).
 * @returns The estimator; `fit({ x })` takes the data, $n \times d$.
 *
 * @example Two blobs from an RBF affinity
 * const s = stream(0)
 * const x = concat([normals(s, [6, 2], 0, 0.3), normals(s, [6, 2], 4, 0.3)])
 * const model = spectralClustering({ k: 2 }).fit({ x }, { stream: stream(1) })
 * print('labels', model.labels)
 * print('eigenvalues', model.eigenvalues)
 * print('graph components', model.components)
 */
export function spectralClustering(params: {
  k: number
  affinity?: Affinity
  restarts?: number
  normaliseRows?: boolean
}): Estimator<Dataset<Tensor>, SpectralClusteringModel> {
  const { k, affinity = { kind: 'rbf', lengthscale: 1 }, restarts = 10, normaliseRows = true } = params
  return {
    name: 'spectral-clustering',
    params: { k, affinity, restarts, normaliseRows },
    fit({ x }, options: FitOptions = {}) {
      const { n } = matrix(x, 'spectralClustering')
      const A = affinityMatrix(x, affinity)
      const W = A.data as Float64Array
      const edges: [number, number, number][] = []
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (W[i * n + j] > 0) edges.push([i, j, W[i * n + j]])
      const components = connectedComponents(fromEdges(n, edges, { directed: false })).count
      const deg = new Float64Array(n)
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += W[i * n + j]
      const M = new Float64Array(n * n)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          M[i * n + j] = deg[i] > 0 && deg[j] > 0 ? W[i * n + j] / Math.sqrt(deg[i] * deg[j]) : 0
      const e = eigh(fromData(M, [n, n]))
      const V = e.vectors.data as Float64Array
      const U = new Float64Array(n * k)
      for (let i = 0; i < n; i++) {
        let norm = 0
        for (let j = 0; j < k; j++) norm += (U[i * k + j] = V[i * n + j]) ** 2
        norm = Math.sqrt(norm)
        if (normaliseRows && norm > 0) for (let j = 0; j < k; j++) U[i * k + j] /= norm
      }
      const embedding = mat(U, n, k)
      const km = kmeans({ k, restarts }).fit(dataset(embedding), { stream: options.stream })
      return {
        kind: 'model',
        transductive: true,
        name: 'spectral-clustering',
        labels: km.decide(embedding),
        affinity: A,
        embedding,
        eigenvalues: vec(Array.from({ length: k }, (_, j) => (e.values.data as Float64Array)[j])),
        components,
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'spectralClustering',
    module: 'unsupervised/clustering',
    name: 'Spectral clustering',
    summary: 'k-means on the leading eigenvectors of a normalised graph Laplacian.',
    task: 'clustering',
    capabilities: [],
    transductive: true,
    hyper: space({
      k: int(1, 20, { default: 3 }),
      restarts: int(1, 50, { default: 10 }),
      normaliseRows: bool({ default: true }),
    }),
    notes: ['spectral-clustering'],
    cite: ['ng2002b', 'shi2000'],
  },
  spectralClustering,
)
