/**
 * Gaussian variational families over $\reals^d$ with flat parameter vectors $\lambdavec$, as in ADVI (Kucukelbir et
 * al., 2017, §2.4–2.5):
 *
 * - mean field: $\lambdavec = (\muvec, \omegavec)$ with $\sigmavec = \exp(\omegavec)$,
 *   $q = \Gauss(\muvec, \diag(\sigmavec^2))$, $2d$ parameters;
 * - full rank: $\lambdavec = (\muvec, \boldsymbol{\ell})$ with $\boldsymbol{\ell}$ the lower triangle of $\Lmat$ row
 *   by row, its diagonal stored as $\log L_{ii}$, $q = \Gauss(\muvec, \Lmat\Lmat^\top)$, $d + d(d + 1)/2$
 *   parameters.
 *
 * Both are reparameterised as $\xvec = \muvec + \Lmat\epsilonvec$ with $\epsilonvec \sim \Gauss(\zeros, \Imat)$
 * ($\Lmat = \diag(\sigmavec)$ for mean field), which gives the pathwise gradient, and have closed-form entropies and
 * score functions $\nabla_{\lambdavec} \log q$. The kernels work on plain float64 arrays and are what the estimators
 * of `elbo.ts` call; the tensor-valued methods check the length of $\lambdavec$ and throw `ShapeError`.
 */

import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { VectorLike } from 'aifn-compute/foundation/contracts'

/** A float64 working array (`aifn-compute/foundation/tensor`'s `dense.F64`). */
export type F64 = dense.F64

/**
 * A copy of a vector argument as a working array.
 *
 * @param v The vector: a plain array, typed array or rank-1 tensor.
 * @returns A new float64 array of its values.
 */
export const toF64 = (v: VectorLike): F64 => dense.toF64(v, 'variational')
/** A working array wrapped as a rank-1 tensor (`dense.vec`). */
export const { vec } = dense

const LOG_2PI = Math.log(2 * Math.PI)

/** A Gaussian variational family over $\reals^d$ ($d$ = `dim`) with a flat parameter vector of length `size`. */
export type GaussianFamily = {
  /** Which family: diagonal covariance (`'mean-field'`) or a full lower-triangular scale (`'full-rank'`). */
  kind: 'mean-field' | 'full-rank'
  /** The dimension $d$ of $\xvec$. */
  dim: number
  /** The number of variational parameters: $2d$ for mean field, $d + d(d + 1)/2$ for full rank. */
  size: number
  /**
   * $\lambdavec$ for a given mean and standard deviation (a number, or one per coordinate; default 1). Full rank: a
   * diagonal $\Lmat$.
   */
  parameters: (mean: VectorLike, sd?: number | ArrayLike<number>) => Vector
  /** The mean $\muvec$ of $q_{\lambdavec}$. */
  mean: (lambda: VectorLike) => Vector
  /** The covariance $\Lmat\Lmat^\top$ of $q_{\lambdavec}$ ($d \times d$). */
  covariance: (lambda: VectorLike) => Matrix
  /** The scale factor $\Lmat$ ($d \times d$, lower triangular; diagonal for mean field). */
  scale: (lambda: VectorLike) => Matrix
  /** $\xvec = \muvec + \Lmat\epsilonvec$: a standard normal draw $\epsilonvec$ mapped to a draw of $q_{\lambdavec}$. */
  transform: (lambda: VectorLike, eps: VectorLike) => Vector
  /** $\log q_{\lambdavec}(\xvec)$. */
  logDensity: (lambda: VectorLike, x: VectorLike) => number
  /** The entropy $\entropy[q_{\lambdavec}] = \sum_i \log L_{ii} + (d/2)(1 + \log 2\pi)$. */
  entropy: (lambda: VectorLike) => number
  /** Internal kernels on working arrays ($\lambdavec$, $\epsilonvec$, $\xvec$); used by the estimators. */
  kernels: FamilyKernels
}

/**
 * Working-array kernels of a family. They take $\lambdavec$ as a float64 array of `size` values without checking its
 * length, and return new arrays; gradients are with respect to the stored parameters (so for $\log L_{ii}$, not
 * $L_{ii}$).
 */
export type FamilyKernels = {
  /** $\Lmat$ as a dense row-major $d \times d$ array. */
  scaleMatrix: (lambda: F64) => F64
  /** $\xvec = \muvec + \Lmat\epsilonvec$. */
  transform: (lambda: F64, eps: F64) => F64
  /** $\epsilonvec = \Lmat^{-1}(\xvec - \muvec)$. */
  standardise: (lambda: F64, x: F64) => F64
  /** $\log q_{\lambdavec}(\xvec)$. */
  logDensity: (lambda: F64, x: F64) => number
  /** $\entropy[q_{\lambdavec}]$. */
  entropy: (lambda: F64) => number
  /** $\nabla_{\lambdavec} \entropy$. */
  entropyGrad: (lambda: F64) => F64
  /**
   * $\nabla_{\lambdavec} f(\muvec + \Lmat\epsilonvec)$ given $\gvec = \nabla f$ at
   * $\xvec = \muvec + \Lmat\epsilonvec$ (the chain rule through the reparameterisation).
   */
  pathGrad: (lambda: F64, eps: F64, g: F64) => F64
  /**
   * $\nabla_{\lambdavec} \log q_{\lambdavec}(\xvec)$ at $\xvec = \muvec + \Lmat\epsilonvec$, with $\xvec$ held
   * fixed.
   */
  score: (lambda: F64, eps: F64) => F64
}

/**
 * A family from its kernels: the tensor-valued methods of `GaussianFamily`, each checking that $\lambdavec$ has `size`
 * values (else `ShapeError`).
 *
 * @param kind Which family, also used in error messages.
 * @param dim The dimension $d$ of $\xvec$.
 * @param size The length of $\lambdavec$.
 * @param k The family's working-array kernels.
 * @param params The family's `parameters` method, used as given.
 * @returns The family.
 */
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

/**
 * A value per coordinate: a number repeated $d$ times, or a copy of the values given.
 *
 * @param v One number for every coordinate, or one per coordinate (not checked to have $d$ values).
 * @param d The number of coordinates.
 * @returns A new float64 array.
 */
const perCoordinate = (v: number | ArrayLike<number>, d: number) =>
  typeof v === 'number' ? new Float64Array(d).fill(v) : Float64Array.from(v)

/**
 * The mean-field Gaussian family $q(\xvec) = \prod_i \Gauss(x_i \mid \mu_i, \sigma_i^2)$ with
 * $\lambdavec = (\muvec, \omegavec)$, $\omegavec = \log\sigmavec$ (Kucukelbir et al., 2017, §2.5). Score:
 * $\partial \log q / \partial \mu_i = \epsilon_i / \sigma_i$ and
 * $\partial \log q / \partial \omega_i = \epsilon_i^2 - 1$; path: $\partial / \partial \mu_i = g_i$ and
 * $\partial / \partial \omega_i = g_i \sigma_i \epsilon_i$.
 *
 * @param dim The dimension $d$ of $\xvec$; $\lambdavec$ then has $2d$ values, the means then the log standard
 *   deviations.
 * @returns The family, its methods taking $\lambdavec$.
 *
 * @example Parameters, moments and entropy
 * const q = meanFieldGaussian(2)
 * const lambda = q.parameters([1, -1], [0.5, 2])
 * print('λ =', lambda)
 * print('mean =', q.mean(lambda))
 * print('covariance =', q.covariance(lambda))
 * print('entropy =', q.entropy(lambda), 'check', 1 + Math.log(2 * Math.PI))
 *
 * @example A draw is the mean moved by the scale times a standard normal
 * const q = meanFieldGaussian(2)
 * const lambda = q.parameters([1, -1], [0.5, 2])
 * print('x for ε = (1, 1):', q.transform(lambda, [1, 1]))
 * print('log q at the mean:', q.logDensity(lambda, [1, -1]), 'check', -Math.log(2 * Math.PI))
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
 * The full-rank Gaussian family $q = \Gauss(\muvec, \Lmat\Lmat^\top)$ with $\lambdavec = (\muvec$, the lower
 * triangle of $\Lmat$ row by row, its diagonal as $\log L_{ii})$ (Kucukelbir et al., 2017, §2.5; Titsias &
 * Lázaro-Gredilla, 2014). With $\epsilonvec = \Lmat^{-1}(\xvec - \muvec)$ and $\vvec = \Lmat^{-\top}\epsilonvec$, the
 * score is $\partial \log q / \partial \muvec = \vvec$ and
 * $\partial \log q / \partial L_{ij} = v_i \epsilon_j - \delta_{ij} / L_{ii}$; the path gradient is
 * $\partial / \partial L_{ij} = g_i \epsilon_j$. Both are multiplied by $L_{ii}$ on the diagonal, which is stored as
 * its logarithm.
 *
 * @param dim The dimension $d$ of $\xvec$; $\lambdavec$ then has $d + d(d + 1)/2$ values: the mean, then
 *   $L_{00}$ (as its log), $L_{10}$, $L_{11}$ (as its log), $L_{20}$, and so on.
 * @returns The family, its methods taking $\lambdavec$.
 *
 * @example A correlated Gaussian from its scale factor
 * // L = [[1, 0], [0.5, 2]], with the diagonal stored as logs.
 * const q = fullRankGaussian(2)
 * const lambda = [0, 0, Math.log(1), 0.5, Math.log(2)]
 * print('size =', q.size)
 * print('L =', q.scale(lambda))
 * print('covariance =', q.covariance(lambda))
 * print('log q at the mean:', q.logDensity(lambda, [0, 0]), 'check', -Math.log(2) - Math.log(2 * Math.PI))
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
