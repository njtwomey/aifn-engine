/**
 * Gaussian variational families with flat parameter vectors λ, as in ADVI (Kucukelbir et al., 2017, §2.4–2.5):
 *
 * - mean field: λ = (μ, ω) with σ = exp(ω), q = N(μ, diag(σ²)), 2d parameters;
 * - full rank: λ = (μ, ℓ) with ℓ the lower triangle of L row by row, its diagonal stored as log Lᵢᵢ, q = N(μ, LLᵀ),
 *   d + d(d + 1)/2 parameters.
 *
 * Both are reparameterised as x = μ + Lε with ε ~ N(0, I) (L = diag(σ) for mean field), which gives the pathwise
 * gradient, and have closed-form entropies and score functions ∇_λ log q.
 */

import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { VectorLike } from 'aifn-compute/foundation/contracts'

/** A float64 working array (`aifn-compute/foundation/tensor`'s `dense.F64`). */
export type F64 = dense.F64

/** A copy of a vector argument as a working array. */
export const toF64 = (v: VectorLike): F64 => dense.toF64(v, 'variational')
/** A working array wrapped as a rank-1 tensor (`dense.vec`). */
export const { vec } = dense

const LOG_2PI = Math.log(2 * Math.PI)

/** A Gaussian variational family over ℝ^dim with a flat parameter vector of length `size`. */
export type GaussianFamily = {
  kind: 'mean-field' | 'full-rank'
  dim: number
  /** The number of variational parameters. */
  size: number
  /** λ for a given mean and standard deviation (a number, or one per coordinate; full rank: a diagonal L). */
  parameters: (mean: VectorLike, sd?: number | ArrayLike<number>) => Vector
  /** The mean μ of q_λ. */
  mean: (lambda: VectorLike) => Vector
  /** The covariance LLᵀ of q_λ (d×d). */
  covariance: (lambda: VectorLike) => Matrix
  /** The scale factor L (d×d, lower triangular; diagonal for mean field). */
  scale: (lambda: VectorLike) => Matrix
  /** x = μ + Lε. */
  transform: (lambda: VectorLike, eps: VectorLike) => Vector
  /** log q_λ(x). */
  logDensity: (lambda: VectorLike, x: VectorLike) => number
  /** The entropy H[q_λ] = Σ log Lᵢᵢ + (d/2)(1 + log 2π). */
  entropy: (lambda: VectorLike) => number
  /** Internal kernels on working arrays (λ, ε, x); used by the estimators. */
  kernels: FamilyKernels
}

/** Working-array kernels of a family. */
export type FamilyKernels = {
  /** L as a dense row-major d×d array. */
  scaleMatrix: (lambda: F64) => F64
  transform: (lambda: F64, eps: F64) => F64
  /** ε = L⁻¹(x − μ). */
  standardise: (lambda: F64, x: F64) => F64
  logDensity: (lambda: F64, x: F64) => number
  entropy: (lambda: F64) => number
  /** ∇_λ H. */
  entropyGrad: (lambda: F64) => F64
  /** ∇_λ f(μ + Lε) given g = ∇f at x = μ + Lε (the chain rule through the reparameterisation). */
  pathGrad: (lambda: F64, eps: F64, g: F64) => F64
  /** ∇_λ log q_λ(x) at x = μ + Lε. */
  score: (lambda: F64, eps: F64) => F64
}

function wrap(
  kind: GaussianFamily['kind'],
  dim: number,
  size: number,
  k: FamilyKernels,
  params: GaussianFamily['parameters'],
): GaussianFamily {
  const check = (lambda: VectorLike) => {
    const l = toF64(lambda)
    if (l.length !== size)
      throw new ShapeError(`${kind} family`, `${kind} family: λ must have ${size} values, got ${l.length}`)
    return l
  }
  return {
    kind,
    dim,
    size,
    parameters: params,
    mean: (lambda) => vec(check(lambda).slice(0, dim)),
    scale: (lambda) => fromData(k.scaleMatrix(check(lambda)), [dim, dim]),
    covariance: (lambda) => {
      const L = k.scaleMatrix(check(lambda))
      const C = new Float64Array(dim * dim)
      for (let i = 0; i < dim; i++)
        for (let j = 0; j < dim; j++) {
          let s = 0
          for (let r = 0; r < dim; r++) s += L[i * dim + r] * L[j * dim + r]
          C[i * dim + j] = s
        }
      return fromData(C, [dim, dim])
    },
    transform: (lambda, eps) => vec(k.transform(check(lambda), toF64(eps))),
    logDensity: (lambda, x) => k.logDensity(check(lambda), toF64(x)),
    entropy: (lambda) => k.entropy(check(lambda)),
    kernels: k,
  }
}

const perCoordinate = (v: number | ArrayLike<number>, d: number) =>
  typeof v === 'number' ? new Float64Array(d).fill(v) : Float64Array.from(v)

/**
 * The mean-field Gaussian family q(x) = Πᵢ N(xᵢ | μᵢ, σᵢ²) with λ = (μ, log σ) (Kucukelbir et al., 2017, §2.5).
 * Score: ∂ log q/∂μᵢ = εᵢ/σᵢ and ∂ log q/∂ωᵢ = εᵢ² − 1; path: ∂/∂μᵢ = gᵢ and ∂/∂ωᵢ = gᵢσᵢεᵢ.
 */
export function meanFieldGaussian(dim: number): GaussianFamily {
  const d = dim
  const sigma = (l: F64, i: number) => Math.exp(l[d + i])
  const k: FamilyKernels = {
    scaleMatrix: (l) => {
      const L = new Float64Array(d * d)
      for (let i = 0; i < d; i++) L[i * d + i] = sigma(l, i)
      return L
    },
    transform: (l, e) => Float64Array.from({ length: d }, (_, i) => l[i] + sigma(l, i) * e[i]),
    standardise: (l, x) => Float64Array.from({ length: d }, (_, i) => (x[i] - l[i]) / sigma(l, i)),
    logDensity: (l, x) => {
      let s = 0
      for (let i = 0; i < d; i++) s += -0.5 * ((x[i] - l[i]) / sigma(l, i)) ** 2 - l[d + i]
      return s - 0.5 * d * LOG_2PI
    },
    entropy: (l) => {
      let s = 0
      for (let i = 0; i < d; i++) s += l[d + i]
      return s + 0.5 * d * (1 + LOG_2PI)
    },
    entropyGrad: () => {
      const g = new Float64Array(2 * d)
      g.fill(1, d)
      return g
    },
    pathGrad: (l, e, g) => {
      const out = new Float64Array(2 * d)
      for (let i = 0; i < d; i++) {
        out[i] = g[i]
        out[d + i] = g[i] * sigma(l, i) * e[i]
      }
      return out
    },
    score: (l, e) => {
      const out = new Float64Array(2 * d)
      for (let i = 0; i < d; i++) {
        out[i] = e[i] / sigma(l, i)
        out[d + i] = e[i] * e[i] - 1
      }
      return out
    },
  }
  return wrap('mean-field', d, 2 * d, k, (mean, sd = 1) => {
    const m = toF64(mean)
    const s = perCoordinate(sd, d)
    const l = new Float64Array(2 * d)
    l.set(m, 0)
    for (let i = 0; i < d; i++) l[d + i] = Math.log(s[i])
    return vec(l)
  })
}

/**
 * The full-rank Gaussian family q = N(μ, LLᵀ) with λ = (μ, lower triangle of L row by row, diagonal as log Lᵢᵢ)
 * (Kucukelbir et al., 2017, §2.5; Titsias & Lázaro-Gredilla, 2014). With ε = L⁻¹(x − μ) and v = L⁻ᵀε, the score is
 * ∂ log q/∂μ = v and ∂ log q/∂Lᵢⱼ = vᵢεⱼ − δᵢⱼ/Lᵢᵢ; the path gradient is ∂/∂Lᵢⱼ = gᵢεⱼ.
 */
export function fullRankGaussian(dim: number): GaussianFamily {
  const d = dim
  const tri = (d * (d + 1)) / 2
  // Index of Lᵢⱼ (i ≥ j) in λ.
  const at = (i: number, j: number) => d + (i * (i + 1)) / 2 + j
  const scaleMatrix = (l: F64) => {
    const L = new Float64Array(d * d)
    for (let i = 0; i < d; i++)
      for (let j = 0; j <= i; j++) L[i * d + j] = i === j ? Math.exp(l[at(i, i)]) : l[at(i, j)]
    return L
  }
  const k: FamilyKernels = {
    scaleMatrix,
    transform: (l, e) => {
      const L = scaleMatrix(l)
      return Float64Array.from({ length: d }, (_, i) => {
        let s = l[i]
        for (let j = 0; j <= i; j++) s += L[i * d + j] * e[j]
        return s
      })
    },
    standardise: (l, x) => {
      const L = scaleMatrix(l)
      const e = new Float64Array(d)
      for (let i = 0; i < d; i++) {
        let s = x[i] - l[i]
        for (let j = 0; j < i; j++) s -= L[i * d + j] * e[j]
        e[i] = s / L[i * d + i]
      }
      return e
    },
    logDensity: (l, x) => {
      const e = k.standardise(l, x)
      let s = 0
      for (let i = 0; i < d; i++) s += -0.5 * e[i] * e[i] - l[at(i, i)]
      return s - 0.5 * d * LOG_2PI
    },
    entropy: (l) => {
      let s = 0
      for (let i = 0; i < d; i++) s += l[at(i, i)]
      return s + 0.5 * d * (1 + LOG_2PI)
    },
    entropyGrad: () => {
      const g = new Float64Array(d + tri)
      for (let i = 0; i < d; i++) g[at(i, i)] = 1
      return g
    },
    pathGrad: (l, e, g) => {
      const out = new Float64Array(d + tri)
      out.set(g.subarray(0, d), 0)
      for (let i = 0; i < d; i++)
        for (let j = 0; j <= i; j++) out[at(i, j)] = g[i] * e[j] * (i === j ? Math.exp(l[at(i, i)]) : 1)
      return out
    },
    score: (l, e) => {
      const L = scaleMatrix(l)
      // v = L⁻ᵀε by back substitution.
      const v = new Float64Array(d)
      for (let i = d - 1; i >= 0; i--) {
        let s = e[i]
        for (let j = i + 1; j < d; j++) s -= L[j * d + i] * v[j]
        v[i] = s / L[i * d + i]
      }
      const out = new Float64Array(d + tri)
      out.set(v, 0)
      for (let i = 0; i < d; i++)
        for (let j = 0; j <= i; j++) {
          const dL = v[i] * e[j] - (i === j ? 1 / L[i * d + i] : 0)
          out[at(i, j)] = i === j ? dL * L[i * d + i] : dL
        }
      return out
    },
  }
  return wrap('full-rank', d, d + tri, k, (mean, sd = 1) => {
    const m = toF64(mean)
    const s = perCoordinate(sd, d)
    const l = new Float64Array(d + tri)
    l.set(m, 0)
    for (let i = 0; i < d; i++) l[at(i, i)] = Math.log(s[i])
    return vec(l)
  })
}
