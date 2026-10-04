/**
 * Neighbour-graph manifold learning:
 *
 * - `isomap`: geodesic distances along the k-nearest-neighbour graph (shortest paths by Dijkstra, from `aifn-compute/graph`),
 *   then classical MDS (Tenenbaum, de Silva and Langford, 2000).
 * - `laplacianEigenmaps`: the bottom non-trivial eigenvectors of the normalised graph Laplacian (Belkin and Niyogi,
 *   2003), as scikit-learn's `SpectralEmbedding`.
 * - `locallyLinearEmbedding`: LLE (Roweis and Saul, 2000) with scikit-learn's regularised barycentre weights.
 */

import type { Dataset, Estimator } from 'aifn-compute/learning/estimators'
import { connectedComponents } from 'aifn-compute/graph/traversal'
import { dijkstra } from 'aifn-compute/graph/shortest-paths'
import { fromEdges, type Graph } from 'aifn-compute/graph'
import { eigh, solve } from 'aifn-compute/numerics/linalg'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { classicalCore } from '../centring'
import { nearestNeighbours, squaredDistances } from '../neighbourhoods'
import { mat, matrix, values, vec } from '../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, real, space } from 'aifn-compute/foundation/space'

/** The symmetric k-nearest-neighbour graph of the rows of x, edges weighted by Euclidean distance. */
export function neighbourGraph(x: Tensor, k: number): { graph: Graph; distances: Tensor; neighbours: number[][] } {
  const { n, d, v } = matrix(x, 'neighbourGraph')
  const D = Float64Array.from(squaredDistances(v, n, d), Math.sqrt)
  const nb = nearestNeighbours(D, n, k)
  const seen = new Set<number>()
  const edges: [number, number, number][] = []
  for (let i = 0; i < n; i++) {
    for (const j of nb[i]) {
      const key = Math.min(i, j) * n + Math.max(i, j)
      if (seen.has(key)) continue
      seen.add(key)
      edges.push([Math.min(i, j), Math.max(i, j), D[i * n + j]])
    }
  }
  return { graph: fromEdges(n, edges, { directed: false }), distances: mat(D, n, n), neighbours: nb }
}

/** An Isomap embedding. */
export interface IsomapModel {
  readonly kind: 'model'
  /** The embedding covers the training rows only (no out-of-sample transform). */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'isomap'
  readonly embedding: Tensor
  /** Geodesic (graph shortest-path) distances [n, n]; ∞ between components. */
  readonly geodesics: Tensor
  /** Eigenvalues of the double-centred squared geodesics, descending. */
  readonly eigenvalues: Tensor
  readonly graph: Graph
  /** Components of the neighbour graph: more than one leaves infinite geodesics, and the embedding fails. */
  readonly components: number
}

/** Isomap with `neighbours` (default 5) nearest neighbours into `dims` (default 2) dimensions. */
export function isomap(params: { neighbours?: number; dims?: number } = {}): Estimator<Dataset<Tensor>, IsomapModel> {
  const { neighbours: k = 5, dims = 2 } = params
  return {
    name: 'isomap',
    params: { neighbours: k, dims },
    fit({ x }) {
      const { n } = matrix(x, 'isomap')
      const { graph } = neighbourGraph(x, k)
      const components = connectedComponents(graph).count
      const G = new Float64Array(n * n)
      for (let s = 0; s < n; s++) G.set(values(dijkstra(graph, s).distance), s * n)
      if (components > 1) {
        return {
          kind: 'model',
          transductive: true,
          name: 'isomap',
          embedding: mat(new Float64Array(n * dims).fill(NaN), n, dims),
          geodesics: mat(G, n, n),
          eigenvalues: vec([]),
          graph,
          components,
        }
      }
      const { Y, eigenvalues } = classicalCore(
        Float64Array.from(G, (u) => u * u),
        n,
        dims,
      )
      return {
        kind: 'model',
        transductive: true,
        name: 'isomap',
        embedding: mat(Y, n, dims),
        geodesics: mat(G, n, n),
        eigenvalues: vec(eigenvalues),
        graph,
        components,
      }
    },
  }
}

/** A spectral embedding. */
export interface SpectralEmbeddingModel {
  readonly kind: 'model'
  /** The embedding covers the training rows only (no out-of-sample transform). */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'laplacian-eigenmaps'
  readonly embedding: Tensor
  /** Affinity matrix W [n, n]. */
  readonly affinity: Tensor
  /** The smallest eigenvalues of the normalised Laplacian I − D^(−½)WD^(−½), ascending (the first is 0). */
  readonly eigenvalues: Tensor
  readonly components: number
}

/**
 * Laplacian eigenmaps: W from the k-nearest-neighbour graph (connectivity, symmetrised ½(A + Aᵀ), or heat-kernel
 * weights exp(−d²/t)); the embedding is the next `dims` eigenvectors u of the normalised Laplacian after the trivial
 * one, mapped back as u / √degree (the generalised problem L y = λ D y) and signed so that each column's
 * largest-magnitude entry is positive.
 */
export function laplacianEigenmaps(
  params: { neighbours?: number; dims?: number; heat?: number } = {},
): Estimator<Dataset<Tensor>, SpectralEmbeddingModel> {
  const { neighbours: k = 10, dims = 2, heat } = params
  return {
    name: 'laplacian-eigenmaps',
    params: { neighbours: k, dims, heat },
    fit({ x }) {
      const { n } = matrix(x, 'laplacianEigenmaps')
      const { graph, distances, neighbours: nb } = neighbourGraph(x, k)
      const D = values(distances)
      const W = new Float64Array(n * n)
      for (let i = 0; i < n; i++) {
        for (const j of nb[i]) {
          const w = heat === undefined ? 0.5 : 0.5 * Math.exp(-(D[i * n + j] ** 2) / heat)
          W[i * n + j] += w
          W[j * n + i] += w
        }
      }
      const deg = new Float64Array(n)
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += W[i * n + j]
      const L = new Float64Array(n * n)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) L[i * n + j] = (i === j ? 1 : 0) - W[i * n + j] / Math.sqrt(deg[i] * deg[j])
      const e = eigh(fromData(L, [n, n]))
      const lambda = e.values.data as Float64Array
      const V = e.vectors.data as Float64Array
      // Eigenvalues come out descending: the smallest are at the end.
      const Y = new Float64Array(n * dims)
      const eig: number[] = [lambda[n - 1]]
      for (let c = 0; c < dims; c++) {
        const col = n - 2 - c
        eig.push(lambda[col])
        let big = 0
        for (let i = 0; i < n; i++) {
          const u = V[i * n + col] / Math.sqrt(deg[i])
          Y[i * dims + c] = u
          if (Math.abs(u) > Math.abs(big)) big = u
        }
        if (big < 0) for (let i = 0; i < n; i++) Y[i * dims + c] = -Y[i * dims + c]
      }
      return {
        kind: 'model',
        transductive: true,
        name: 'laplacian-eigenmaps',
        embedding: mat(Y, n, dims),
        affinity: mat(W, n, n),
        eigenvalues: vec(eig),
        components: connectedComponents(graph).count,
      }
    },
  }
}

/** An LLE embedding. */
export interface LleModel {
  readonly kind: 'model'
  /** The embedding covers the training rows only (no out-of-sample transform). */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'locally-linear-embedding'
  readonly embedding: Tensor
  /** Reconstruction weights [n, n]: row i holds the weights of xᵢ's neighbours (summing to 1). */
  readonly weights: Tensor
  /** The eigenvalues of M = (I − W)ᵀ(I − W) used, ascending (after the trivial one); their sum is the embedding cost. */
  readonly eigenvalues: Tensor
}

/**
 * Locally linear embedding: each point's weights over its k neighbours minimise ‖xᵢ − Σⱼ wᵢⱼ xⱼ‖² with Σⱼ wᵢⱼ = 1,
 * regularising the local Gram matrix C by `regularisation` × tr C (default 1e-3); the embedding is the bottom
 * eigenvectors of (I − W)ᵀ(I − W) after the constant one (signs are arbitrary).
 */
export function locallyLinearEmbedding(
  params: { neighbours?: number; dims?: number; regularisation?: number } = {},
): Estimator<Dataset<Tensor>, LleModel> {
  const { neighbours: k = 5, dims = 2, regularisation = 1e-3 } = params
  return {
    name: 'locally-linear-embedding',
    params: { neighbours: k, dims, regularisation },
    fit({ x }) {
      const { n, d, v } = matrix(x, 'locallyLinearEmbedding')
      const D = Float64Array.from(squaredDistances(v, n, d), Math.sqrt)
      const nb = nearestNeighbours(D, n, k)
      const W = new Float64Array(n * n)
      for (let i = 0; i < n; i++) {
        const G = new Float64Array(k * d)
        nb[i].forEach((j, a) => {
          for (let c = 0; c < d; c++) G[a * d + c] = v[j * d + c] - v[i * d + c]
        })
        const C = new Float64Array(k * k)
        let tr = 0
        for (let a = 0; a < k; a++) {
          for (let b = 0; b < k; b++) {
            let s = 0
            for (let c = 0; c < d; c++) s += G[a * d + c] * G[b * d + c]
            C[a * k + b] = s
          }
          tr += C[a * k + a]
        }
        const R = tr > 0 ? regularisation * tr : regularisation
        for (let a = 0; a < k; a++) C[a * k + a] += R
        const w = values(solve(fromData(C, [k, k]), fromData(new Float64Array(k).fill(1), [k])) as Tensor)
        const s = w.reduce((p, q) => p + q, 0)
        nb[i].forEach((j, a) => (W[i * n + j] = w[a] / s))
      }
      // M = (I − W)ᵀ(I − W)
      const M = new Float64Array(n * n)
      for (let a = 0; a < n; a++) {
        for (let b = 0; b <= a; b++) {
          let s = 0
          for (let i = 0; i < n; i++) s += ((i === a ? 1 : 0) - W[i * n + a]) * ((i === b ? 1 : 0) - W[i * n + b])
          M[a * n + b] = M[b * n + a] = s
        }
      }
      const e = eigh(fromData(M, [n, n]))
      const lambda = e.values.data as Float64Array
      const V = e.vectors.data as Float64Array
      const Y = new Float64Array(n * dims)
      const eig: number[] = []
      for (let c = 0; c < dims; c++) {
        const col = n - 2 - c
        eig.push(lambda[col])
        for (let i = 0; i < n; i++) Y[i * dims + c] = V[i * n + col]
      }
      return {
        kind: 'model',
        transductive: true,
        name: 'locally-linear-embedding',
        embedding: mat(Y, n, dims),
        weights: mat(W, n, n),
        eigenvalues: vec(eig),
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'isomap',
    module: 'unsupervised/embedding/manifold',
    name: 'Isomap',
    summary: 'Classical MDS of geodesic distances on a nearest-neighbour graph.',
    task: 'embedding',
    capabilities: [],
    transductive: true,
    hyper: space({ neighbours: int(2, 50, { default: 5 }), dims: int(1, 10, { default: 2 }) }),
    notes: ['isomap'],
    cite: ['tenenbaum2000'],
  },
  isomap,
)

defineModel(
  {
    key: 'laplacianEigenmaps',
    module: 'unsupervised/embedding/manifold',
    name: 'Laplacian eigenmaps',
    summary: 'The smallest non-trivial eigenvectors of a nearest-neighbour graph Laplacian.',
    task: 'embedding',
    capabilities: [],
    transductive: true,
    hyper: space({ neighbours: int(2, 50, { default: 10 }), dims: int(1, 10, { default: 2 }) }),
    notes: ['laplacian-eigenmaps'],
    cite: ['belkin2003'],
  },
  laplacianEigenmaps,
)

defineModel(
  {
    key: 'locallyLinearEmbedding',
    module: 'unsupervised/embedding/manifold',
    name: 'Locally linear embedding',
    summary: 'An embedding that keeps each point’s reconstruction weights from its neighbours.',
    task: 'embedding',
    capabilities: [],
    transductive: true,
    hyper: space({
      neighbours: int(2, 50, { default: 5 }),
      dims: int(1, 10, { default: 2 }),
      regularisation: real(1e-6, 1, { default: 1e-3, scale: 'log' }),
    }),
    notes: ['locally-linear-embedding'],
    cite: ['roweis2000'],
  },
  locallyLinearEmbedding,
)
