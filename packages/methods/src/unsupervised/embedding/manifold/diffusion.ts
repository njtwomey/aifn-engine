/**
 * Diffusion maps (Coifman and Lafon, 2006): a Gaussian kernel K_ij = exp(−‖xᵢ − xⱼ‖²/ε) on the data, the α-normalised
 * kernel K⁽ᵅ⁾ = Q^{−α} K Q^{−α} (Q the kernel degrees; α = 1 removes the sampling density, α = ½ gives Fokker–Planck
 * diffusion, α = 0 the normalised graph Laplacian), and the random walk P = D⁻¹K⁽ᵅ⁾ on it. With the right eigenvectors
 * ψ_k of P (eigenvalues 1 = λ₀ > λ₁ ≥ …, normalised so Σᵢ πᵢ ψ_k(i)² = 1 under the stationary law π), the map
 * Ψ_t(i) = (λ_k^t ψ_k(i))_{k ≥ 1} turns the diffusion distance D_t(i, j)² = Σ_y (P^t_{iy} − P^t_{jy})² / π_y into
 * the Euclidean distance (exactly, with every k kept).
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
  readonly kind: 'model'
  /** The map places the training rows only (no out-of-sample extension). */
  readonly transductive: true
  readonly name: 'diffusion-map'
  /** Coordinates Ψ_t [n, dims]. */
  readonly embedding: Tensor
  /** Eigenvalues of P, descending, λ₀ = 1 first [dims + 1]. */
  readonly eigenvalues: Tensor
  /** The random walk P [n, n] (rows sum to 1). */
  readonly transition: Tensor
  /** Its stationary law π [n]. */
  readonly stationary: Tensor
  /** The kernel bandwidth ε used. */
  readonly epsilon: number
}

/**
 * The diffusion map of the rows of x into `dims` dimensions (default 2) at diffusion time `time` (default 1), with
 * bandwidth `epsilon` (default the median squared pairwise distance) and normalisation `alpha` (default 1). Each
 * coordinate is signed so that its largest-magnitude entry is positive.
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
