/**
 * Neighbour-graph manifold learning: Isomap, Laplacian eigenmaps and locally linear embedding.
 *
 * Each starts from the $k$ nearest neighbours of every point (ties to the lower index). Isomap (Tenenbaum, de Silva
 * and Langford, 2000) takes geodesic distances along the neighbour graph (shortest paths by Dijkstra, from
 * `aifn-compute/graph`) and embeds them by classical MDS. Laplacian eigenmaps (Belkin and Niyogi, 2003) take the
 * bottom non-trivial eigenvectors of the normalised graph Laplacian
 * $\Imat - \Dmat^{-1/2}\Wmat\Dmat^{-1/2}$, as scikit-learn's `SpectralEmbedding`. Locally linear embedding
 * (Roweis and Saul, 2000) keeps the regularised barycentre weights that rebuild each point from its neighbours, as
 * scikit-learn's `LocallyLinearEmbedding`.
 *
 * All three are transductive: they place the training rows and have no `transform`.
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

/**
 * The symmetric $k$-nearest-neighbour graph of the rows of `x`: an undirected edge joins two points when either is
 * among the other's $k$ nearest (ties to the lower index), weighted by their Euclidean distance. Throws `ShapeError`
 * when `x` is not a matrix, and `DomainError` unless $1 \le k \le n - 1$.
 *
 * @param x The points ($n \times d$), one per row.
 * @param k The number of nearest neighbours of each point, itself excluded.
 * @returns `graph`, the undirected graph on the $n$ points (each edge once, as `[i, j]` with $i < j$); `distances`,
 *   the full $n \times n$ Euclidean distance matrix; and `neighbours`, each point's $k$ neighbours, nearest first.
 *
 * @example Four points on a line, joined to their nearest neighbour
 * const x = tensor([[0, 0], [1, 0], [3, 0], [6, 0]])
 * const { graph, neighbours } = neighbourGraph(x, 1)
 * print('nearest neighbour of each point:', neighbours)
 * print('edges =', graph.edges.map((e) => [e.from, e.to, e.weight]))
 */
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
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** The embedding covers the training rows only (no out-of-sample transform). */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'isomap'
  /** The coordinates of the training rows ($n \times$ `dims`); all NaN when the graph is disconnected. */
  readonly embedding: Tensor
  /** Geodesic (graph shortest-path) distances ($n \times n$); infinite between components. */
  readonly geodesics: Tensor
  /** All $n$ eigenvalues of the double-centred squared geodesics, descending; empty when the graph is disconnected. */
  readonly eigenvalues: Tensor
  /** The neighbour graph, edges weighted by Euclidean distance. */
  readonly graph: Graph
  /**
   * Components of the neighbour graph: more than one leaves infinite geodesics, and the embedding is NaN (raise
   * `neighbours`).
   */
  readonly components: number
}

/**
 * Isomap (Tenenbaum, de Silva and Langford, 2000): classical MDS of the geodesic distances, the shortest paths along
 * the symmetric $k$-nearest-neighbour graph (`neighbourGraph`), as scikit-learn's `Isomap`. A disconnected graph is
 * reported in `components`, with a NaN embedding, rather than thrown. Throws `DomainError` unless
 * $1 \le k \le n - 1$.
 *
 * @param params The settings of the estimator.
 * @param params.neighbours The number of nearest neighbours $k$ of each point (default 5).
 * @param params.dims The dimension of the embedding (default 2).
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns an `IsomapModel`.
 *
 * @example A half circle unrolls into a line
 * // Ten points on a half circle: the geodesics are chords summed along the arc.
 * const t = linspace(0, Math.PI, 10)
 * const x = stack([cos(t), sin(t)], 1)
 * const model = isomap({ neighbours: 2, dims: 1 }).fit({ x })
 * print('embedding =', model.embedding)
 * print('geodesic end to end =', toArray(model.geodesics)[0][9], 'arc length =', Math.PI)
 * print('top eigenvalues =', toArray(model.eigenvalues).slice(0, 3))
 *
 * @example Too few neighbours split the graph, and the embedding is NaN
 * const x = tensor([[0, 0], [1, 0], [2, 0], [10, 0], [11, 0]])
 * const model = isomap({ neighbours: 1, dims: 1 }).fit({ x })
 * print('components =', model.components)
 * print('embedding =', model.embedding)
 */
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
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** The embedding covers the training rows only (no out-of-sample transform). */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'laplacian-eigenmaps'
  /** The coordinates of the training rows ($n \times$ `dims`). */
  readonly embedding: Tensor
  /** Affinity matrix $\Wmat$ ($n \times n$, symmetric). */
  readonly affinity: Tensor
  /**
   * The `dims` + 1 smallest eigenvalues of the normalised Laplacian $\Imat - \Dmat^{-1/2}\Wmat\Dmat^{-1/2}$,
   * ascending (the first, the trivial one, is 0).
   */
  readonly eigenvalues: Tensor
  /**
   * Components of the neighbour graph. With more than one, the trivial eigenvalue repeats and the embedding mixes
   * components' indicator vectors.
   */
  readonly components: number
}

/**
 * Laplacian eigenmaps (Belkin and Niyogi, 2003): the affinity $\Wmat$ of the $k$-nearest-neighbour graph is the
 * symmetrised connectivity $\tfrac{1}{2}(\Amat + \Amat^\top)$ ($\Amat$ the 0/1 neighbour matrix), or with `heat`
 * the same with heat-kernel weights $\exp(-d_{ij}^2/t)$. The embedding is the `dims` eigenvectors $\uvec$ of the
 * normalised Laplacian after the trivial one, mapped back as $u_i / \sqrt{\text{degree}_i}$ (solutions of
 * $\Lmat\yvec = \lambda\Dmat\yvec$) and signed so that each column's largest-magnitude entry is positive. This is
 * scikit-learn's `SpectralEmbedding(affinity='nearest_neighbors')` with `n_neighbors` = $k + 1$, since scikit-learn
 * counts each point as its own neighbour. Throws `DomainError` unless $1 \le k \le n - 1$.
 *
 * @param params The settings of the estimator.
 * @param params.neighbours The number of nearest neighbours $k$ of each point, itself excluded (default 10).
 * @param params.dims The dimension of the embedding (default 2).
 * @param params.heat The heat-kernel width $t$, in squared distance units; left out, the weights are 0/1
 *   connectivity.
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `SpectralEmbeddingModel`.
 *
 * @example Points along a half circle are ordered by the first coordinate
 * const t = linspace(0, Math.PI, 10)
 * const x = stack([cos(t), sin(t)], 1)
 * const model = laplacianEigenmaps({ neighbours: 4, dims: 1 }).fit({ x })
 * print('embedding =', model.embedding)
 * print('eigenvalues =', model.eigenvalues)
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
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** The embedding covers the training rows only (no out-of-sample transform). */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'locally-linear-embedding'
  /** The coordinates of the training rows ($n \times$ `dims`): unit-length eigenvectors, one per column. */
  readonly embedding: Tensor
  /**
   * Reconstruction weights $\Wmat$ ($n \times n$): row $i$ holds the weights of $\xvec_i$'s neighbours (summing to 1).
   */
  readonly weights: Tensor
  /**
   * The `dims` eigenvalues of $\Mmat = (\Imat - \Wmat)^\top(\Imat - \Wmat)$ used, ascending (after the trivial one);
   * their sum is the embedding cost, scikit-learn's `reconstruction_error_`.
   */
  readonly eigenvalues: Tensor
}

/**
 * Locally linear embedding (Roweis and Saul, 2000): each point's weights over its $k$ neighbours minimise
 * $\lVert \xvec_i - \sum_j w_{ij}\xvec_j \rVert^2$ subject to $\sum_j w_{ij} = 1$, with the local Gram matrix
 * $\Cmat$ regularised by adding `regularisation` $\cdot \trace\Cmat$ to its diagonal (`regularisation` alone when
 * $\trace\Cmat = 0$). The embedding is the bottom eigenvectors of $(\Imat - \Wmat)^\top(\Imat - \Wmat)$ after the
 * constant one, each signed so its largest-magnitude entry is positive. As scikit-learn's
 * `LocallyLinearEmbedding(method='standard')`, up to sign. Throws `DomainError` unless $1 \le k \le n - 1$.
 *
 * @param params The settings of the estimator.
 * @param params.neighbours The number of nearest neighbours $k$ of each point, itself excluded (default 5).
 * @param params.dims The dimension of the embedding (default 2).
 * @param params.regularisation The ridge on the local Gram matrix, relative to its trace (default 1e-3); it keeps
 *   the weights defined when $k > d$.
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns an `LleModel`.
 *
 * @example A half circle is unrolled in order
 * const t = linspace(0, Math.PI, 10)
 * const x = stack([cos(t), sin(t)], 1)
 * const model = locallyLinearEmbedding({ neighbours: 4, dims: 1 }).fit({ x })
 * print('embedding =', model.embedding)
 * print('cost =', model.eigenvalues)
 * print('weights of point 4 =', toArray(model.weights)[4])
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
