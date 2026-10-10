/**
 * Diffusion maps (Coifman and Lafon, 2006): a Gaussian kernel
 * $K_{ij} = \exp(-\lVert \xvec_i - \xvec_j \rVert^2/\varepsilon)$ on the data, the $\alpha$-normalised kernel $\Kmat^{(\alpha)} = \Qmat^{-\alpha}\Kmat\Qmat^{-\alpha}$ ($\Qmat$ the
 * diagonal of kernel degrees; $\alpha = 1$ removes the sampling density, $\alpha = \tfrac{1}{2}$ gives Fokker-Planck
 * diffusion, $\alpha = 0$ the normalised graph Laplacian), and the random walk $\Pmat = \Dmat^{-1}\Kmat^{(\alpha)}$ on
 * it.
 *
 * With the right eigenvectors $\psivec_k$ of $\Pmat$ (eigenvalues $1 = \lambda_0 > \lambda_1 \ge \dots$, normalised so
 * $\sum_i \pi_i \psi_k(i)^2 = 1$ under the stationary law $\pivec$), the map
 * $\Psi_t(i) = (\lambda_k^t \psi_k(i))_{k \ge 1}$ turns the diffusion distance
 * $D_t(i, j)^2 = \sum_y (P^t_{iy} - P^t_{jy})^2 / \pi_y$ into the Euclidean distance (exactly, with every $k$ kept).
 */

import { eigh } from 'aifn-compute/numerics/linalg'
import { median } from 'aifn-compute/probability/stats'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import type { Dataset, Estimator } from 'aifn-compute/learning/estimators'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, real, space } from 'aifn-compute/foundation/space'
import { squaredDistances } from '../neighbourhoods'
import { mat, matrix, values, vec } from '../util'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A fitted diffusion map. */
export interface DiffusionMapModel {
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** The map places the training rows only (no out-of-sample extension). */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'diffusion-map'
  /** Coordinates $\Psi_t$ ($n \times$ `dims`), each column signed so its largest-magnitude entry is positive. */
  readonly embedding: Tensor
  /** The `dims` + 1 largest eigenvalues of $\Pmat$, descending, $\lambda_0 = 1$ first. */
  readonly eigenvalues: Tensor
  /** The random walk $\Pmat$ ($n \times n$, rows sum to 1). */
  readonly transition: Tensor
  /** Its stationary law $\pivec$ ($n$ values), proportional to the degrees of $\Kmat^{(\alpha)}$. */
  readonly stationary: Tensor
  /** The kernel bandwidth $\varepsilon$ used. */
  readonly epsilon: number
}

/**
 * The diffusion map of the training rows (Coifman and Lafon, 2006): the eigenvectors of the random walk on an
 * $\alpha$-normalised Gaussian kernel, found through its symmetric conjugate
 * $\Dmat^{-1/2}\Kmat^{(\alpha)}\Dmat^{-1/2}$ and scaled by $\lambda_k^t$. Each coordinate is signed so that its
 * largest-magnitude entry is positive. Throws `DomainError` unless $1 \le$ `dims` $\le n - 1$, or when the bandwidth
 * is not positive (as the median is when more than half the pairs of rows coincide).
 *
 * @param params The settings of the estimator.
 * @param params.dims The number of coordinates, after the trivial one (default 2).
 * @param params.time The diffusion time $t$, the power of the eigenvalues (default 1); larger times shrink the finer
 *   coordinates.
 * @param params.epsilon The kernel bandwidth $\varepsilon$, in squared distance units (default the median squared
 *   distance between distinct pairs of rows).
 * @param params.alpha The normalisation $\alpha$ of the kernel by its degrees (default 1).
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `DiffusionMapModel`.
 *
 * @example Points along a half circle are ordered by the first coordinate
 * const t = linspace(0, Math.PI, 10)
 * const x = stack([cos(t), sin(t)], 1)
 * const model = diffusionMap({ dims: 1, epsilon: 0.5 }).fit({ x })
 * print('embedding =', model.embedding)
 * print('eigenvalues =', model.eigenvalues)
 *
 * @example With every coordinate kept, map distance is diffusion distance
 * const x = normals(stream(1), [5, 2])
 * const model = diffusionMap({ dims: 4 }).fit({ x })
 * const P = toArray(model.transition)
 * const pi = toArray(model.stationary)
 * const Y = toArray(model.embedding)
 * let diffusion = 0
 * for (let y = 0; y < 5; y++) diffusion += (P[0][y] - P[1][y]) ** 2 / pi[y]
 * let euclidean = 0
 * for (let c = 0; c < 4; c++) euclidean += (Y[0][c] - Y[1][c]) ** 2
 * print('diffusion distance of rows 0 and 1 =', Math.sqrt(diffusion))
 * print('their distance in the map =', Math.sqrt(euclidean))
 */
export function diffusionMap(
  params: { dims?: number; time?: number; epsilon?: number; alpha?: number } = {},
): Estimator<Dataset<Tensor>, DiffusionMapModel> {
  const { dims = 2, time = 1, alpha = 1 } = params
  return {
    name: 'diffusion-map',
    params: { dims, time, alpha, epsilon: params.epsilon },
    fit({ x }) {
      const { n, d, v } = matrix(x, 'diffusionMap')
      if (!(dims >= 1 && dims < n)) throw new DomainError('diffusionMap', `diffusionMap: dims must lie in 1 … ${n - 1}`)
      const D2 = squaredDistances(v, n, d)
      let epsilon = params.epsilon
      if (epsilon === undefined) {
        const off: number[] = []
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off.push(D2[i * n + j])
        epsilon = median(off)
      }
      if (!(epsilon > 0)) throw new DomainError('diffusionMap', 'diffusionMap: epsilon must be positive')
      const K = Float64Array.from(D2, (u) => Math.exp(-u / epsilon))
      const q = new Float64Array(n)
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) q[i] += K[i * n + j]
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] /= (q[i] * q[j]) ** alpha
      const deg = new Float64Array(n)
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += K[i * n + j]
      const total = deg.reduce((a, b) => a + b, 0)
      // The symmetric conjugate A = D^{−½} K D^{−½} shares P's eigenvalues; ψ = D^{−½} u, rescaled by √total.
      const A = new Float64Array(n * n)
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * n + j] = K[i * n + j] / Math.sqrt(deg[i] * deg[j])
      const e = eigh(mat(A, n, n))
      const lambda = values(e.values)
      const U = values(e.vectors)
      const Y = new Float64Array(n * dims)
      for (let c = 0; c < dims; c++) {
        const k = c + 1
        const scale = lambda[k] ** time
        let big = 0
        for (let i = 0; i < n; i++) {
          const psi = (U[i * n + k] * Math.sqrt(total)) / Math.sqrt(deg[i])
          Y[i * dims + c] = scale * psi
          if (Math.abs(Y[i * dims + c]) > Math.abs(big)) big = Y[i * dims + c]
        }
        if (big < 0) for (let i = 0; i < n; i++) Y[i * dims + c] = -Y[i * dims + c]
      }
      const P = new Float64Array(n * n)
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i * n + j] = K[i * n + j] / deg[i]
      return {
        kind: 'model',
        transductive: true,
        name: 'diffusion-map',
        embedding: mat(Y, n, dims),
        eigenvalues: vec(lambda.slice(0, dims + 1)),
        transition: mat(P, n, n),
        stationary: vec(Array.from(deg, (u) => u / total)),
        epsilon,
      }
    },
  }
}

defineModel(
  {
    key: 'diffusionMap',
    module: 'unsupervised/embedding/manifold',
    name: 'Diffusion map',
    summary: 'Eigenvectors of a random walk on a Gaussian kernel, scaled so distances are diffusion distances.',
    task: 'embedding',
    capabilities: [],
    transductive: true,
    hyper: space({
      dims: int(1, 10, { default: 2 }),
      time: int(1, 100, { default: 1 }),
      alpha: real(0, 1, { default: 1 }),
    }),
    notes: ['diffusion-maps'],
    cite: ['coifman2006'],
  },
  diffusionMap,
)
