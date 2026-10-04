/**
 * Spectral clustering (Ng, Jordan and Weiss, 2002, "On spectral clustering: analysis and an algorithm"; von Luxburg,
 * 2007, "A tutorial on spectral clustering"): an affinity graph, the top eigenvectors of the normalised affinity
 * D^(−½) W D^(−½), rows scaled to unit length, and k-means on those rows.
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

/** How affinities are built: a Gaussian (RBF) kernel of lengthscale ℓ, or a symmetric k-nearest-neighbour graph. */
export type Affinity = { kind: 'rbf'; lengthscale: number } | { kind: 'neighbours'; k: number }

/**
 * The affinity matrix W [n, n] (zero diagonal) of the rows of x: exp(−‖xᵢ − xⱼ‖²/2ℓ²), or 1 where either point is
 * among the other's k nearest (the connectivity graph symmetrised as ½(A + Aᵀ), as scikit-learn).
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
  readonly labels: Tensor
  readonly affinity: Tensor
  /** The rows fed to k-means [n, k]: top eigenvectors of D^(−½)WD^(−½), each row scaled to unit length. */
  readonly embedding: Tensor
  /** The k largest eigenvalues of D^(−½)WD^(−½) (1 has multiplicity equal to the number of components). */
  readonly eigenvalues: Tensor
  /** Connected components of the affinity graph (`aifn-compute/graph`): more than k of them means the embedding is degenerate. */
  readonly components: number
}

/**
 * Spectral clustering into k groups (Ng, Jordan and Weiss, 2002). k-means on the embedding uses `restarts` k-means++
 * runs (default 10) from the stream.
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
