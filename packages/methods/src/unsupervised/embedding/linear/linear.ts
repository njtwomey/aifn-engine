/**
 * Linear and kernel projections, and multidimensional scaling:
 *
 * - `pca`: principal component analysis by the SVD of the centred data (Pearson, 1901; Hotelling, 1933), with explained
 *   variance, whitening and reconstruction, as scikit-learn's `PCA` (same sign convention).
 * - `kernelPca`: kernel PCA (Schölkopf, Smola and Müller, 1998) on the double-centred kernel matrix.
 * - `classicalMds`: Torgerson–Gower scaling of a distance matrix.
 * - `smacofSteps`, `metricMds`: metric MDS by SMACOF, iterated Guttman transforms (de Leeuw, 1977; Borg and Groenen,
 *   2005, "Modern Multidimensional Scaling", §8.6).
 */

import { gram, rbf, type Kernel } from 'aifn-compute/learning/kernels'
import type { Dataset, Estimator, FitOptions, Trained, Transforms } from 'aifn-compute/learning/estimators'
import { eigh, svd } from 'aifn-compute/numerics/linalg'
import type { Status } from 'aifn-compute/foundation/contracts'
import { normals } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { classicalCore } from '../centring'
import { squaredDistances } from '../neighbourhoods'
import { mat, matrix, square, values, vec } from '../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, oneOf, space } from 'aifn-compute/foundation/space'
import { ShapeError } from 'aifn-compute/foundation/errors'

// ── PCA ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted PCA. */
export interface PcaModel extends Transforms<Tensor, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'pca'
  /** Principal axes as rows [r, d], by decreasing variance; each signed so its largest-magnitude entry is positive. */
  readonly components: Tensor
  /** Variance along each axis [r] (divided by n − 1). */
  readonly explainedVariance: Tensor
  /** Share of the total variance along each axis [r]. */
  readonly explainedVarianceRatio: Tensor
  readonly singularValues: Tensor
  readonly mean: Tensor
  /** The mean of the discarded variances (0 when none are discarded): the noise level of probabilistic PCA. */
  readonly noiseVariance: number
  readonly whiten: boolean
  /** Scores (x − mean) Vᵣ [m, r], divided by √variance per axis when whitening. */
  transform(x: Tensor): Tensor
  /** Back from scores to the input space [m, d]. */
  inverseTransform(z: Tensor): Tensor
}

/** PCA keeping `components` axes (default all, min(n, d)). */
export function pca(params: { components?: number; whiten?: boolean } = {}): Estimator<Dataset<Tensor>, PcaModel> {
  const { whiten = false } = params
  return {
    name: 'pca',
    params,
    fit({ x }) {
      const { n, d, v } = matrix(x, 'pca')
      const r = Math.min(params.components ?? Math.min(n, d), n, d)
      const mean = new Float64Array(d)
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += v[i * d + j] / n
      const Xc = Float64Array.from(v, (u, t) => u - mean[t % d])
      const { S, V } = svd(fromData(Xc, [n, d]))
      const s = S.data as Float64Array
      const Vd = V.data as Float64Array
      const k = s.length
      const variances = Float64Array.from(s, (u) => (u * u) / (n - 1))
      const total = variances.reduce((a, b) => a + b, 0)
      const comps = new Float64Array(r * d)
      for (let c = 0; c < r; c++) for (let j = 0; j < d; j++) comps[c * d + j] = Vd[j * k + c]
      const kept = variances.slice(0, r)
      let noise = 0
      if (r < Math.min(n, d)) {
        // Discarded variance averaged over the remaining min(n, d) − r directions.
        for (let c = r; c < k; c++) noise += variances[c]
        noise /= Math.min(n, d) - r
      }
      const scale = Float64Array.from(kept, (u) => (whiten ? Math.sqrt(u) : 1))
      return {
        kind: 'model',
        name: 'pca',
        components: mat(comps, r, d),
        explainedVariance: vec(kept),
        explainedVarianceRatio: vec(Array.from(kept, (u) => u / total)),
        singularValues: vec(s.slice(0, r)),
        mean: vec(mean),
        noiseVariance: noise,
        whiten,
        transform: (q: Tensor) => {
          const { n: m, d: dq, v: qv } = matrix(q, 'pca.transform')
          if (dq !== d) throw new ShapeError('pca', `pca: fitted on ${d} features, given ${dq}`)
          const out = new Float64Array(m * r)
          for (let i = 0; i < m; i++) {
            for (let c = 0; c < r; c++) {
              let t = 0
              for (let j = 0; j < d; j++) t += (qv[i * d + j] - mean[j]) * comps[c * d + j]
              out[i * r + c] = t / scale[c]
            }
          }
          return mat(out, m, r)
        },
        inverseTransform: (z: Tensor) => {
          const { n: m, d: rz, v: zv } = matrix(z, 'pca.inverseTransform')
          if (rz !== r) throw new ShapeError('pca', `pca: expected ${r} scores per row, given ${rz}`)
          const out = new Float64Array(m * d)
          for (let i = 0; i < m; i++) {
            for (let j = 0; j < d; j++) {
              let t = mean[j]
              for (let c = 0; c < r; c++) t += zv[i * r + c] * scale[c] * comps[c * d + j]
              out[i * d + j] = t
            }
          }
          return mat(out, m, d)
        },
      }
    },
  }
}

// ── Kernel PCA ───────────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted kernel PCA. */
export interface KernelPcaModel extends Transforms<Tensor, Tensor> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'kernel-pca'
  readonly kernel: Kernel
  /** Eigenvalues of the centred kernel matrix [r], descending. */
  readonly eigenvalues: Tensor
  /** Their eigenvectors as columns [n, r] (unit length). */
  readonly eigenvectors: Tensor
  /** The training rows' coordinates [n, r]: eigenvectors times √eigenvalue. */
  readonly embedding: Tensor
  /** Coordinates of new rows [m, r], from their centred kernel values against the training rows. */
  transform(x: Tensor): Tensor
}

/**
 * Kernel PCA: eigendecomposition of the double-centred kernel matrix K̃ = K − 1K/n − K1/n + 1K1/n², keeping
 * `components` (default 2) axes. The kernel comes from `aifn-compute/learning/kernels` (e.g. `rbf({ lengthscale })`).
 */
export function kernelPca(
  params: { kernel?: Kernel; components?: number } = {},
): Estimator<Dataset<Tensor>, KernelPcaModel> {
  const { kernel = rbf({ lengthscale: 1 }), components: r = 2 } = params
  return {
    name: 'kernel-pca',
    params: { kernel, components: r },
    fit({ x }) {
      const { n, d } = matrix(x, 'kernelPca')
      const K = values(gram(kernel, x))
      const colMean = new Float64Array(n)
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) colMean[j] += K[i * n + j] / n
      const all = colMean.reduce((a, b) => a + b, 0) / n
      const Kc = new Float64Array(n * n)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) Kc[i * n + j] = K[i * n + j] - colMean[i] - colMean[j] + all
      const e = eigh(fromData(Kc, [n, n]))
      const lambda = (e.values.data as Float64Array).slice(0, r)
      const Vfull = e.vectors.data as Float64Array
      const V = new Float64Array(n * r)
      const Y = new Float64Array(n * r)
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < r; c++) {
          V[i * r + c] = Vfull[i * n + c]
          Y[i * r + c] = Vfull[i * n + c] * Math.sqrt(Math.max(lambda[c], 0))
        }
      }
      return {
        kind: 'model',
        name: 'kernel-pca',
        kernel,
        eigenvalues: vec(lambda),
        eigenvectors: mat(V, n, r),
        embedding: mat(Y, n, r),
        transform: (q: Tensor) => {
          const { n: m, d: dq } = matrix(q, 'kernelPca.transform')
          if (dq !== d) throw new ShapeError('kernelPca', `kernelPca: fitted on ${d} features, given ${dq}`)
          const Kq = values(gram(kernel, q, x))
          const out = new Float64Array(m * r)
          for (let i = 0; i < m; i++) {
            let rowMean = 0
            for (let j = 0; j < n; j++) rowMean += Kq[i * n + j] / n
            for (let c = 0; c < r; c++) {
              let t = 0
              for (let j = 0; j < n; j++) t += (Kq[i * n + j] - colMean[j] - rowMean + all) * V[j * r + c]
              out[i * r + c] = lambda[c] > 0 ? t / Math.sqrt(lambda[c]) : 0
            }
          }
          return mat(out, m, r)
        },
      }
    },
  }
}

// ── Multidimensional scaling ─────────────────────────────────────────────────────────────────────────────────────

/** A classical MDS result. */
export interface ClassicalMds {
  /** Coordinates [n, r]. */
  embedding: Tensor
  /** All eigenvalues of B = −½ J D² J, descending (negative ones mean D is not Euclidean). */
  eigenvalues: Tensor
}

/** Classical (Torgerson–Gower) MDS of a distance matrix D [n, n] into `dims` dimensions (default 2). */
export function classicalMds(distances: Tensor, dims = 2): ClassicalMds {
  const { n, v } = square(distances, 'classicalMds')
  const { Y, eigenvalues } = classicalCore(
    Float64Array.from(v, (u) => u * u),
    n,
    dims,
  )
  return { embedding: mat(Y, n, dims), eigenvalues: vec(eigenvalues) }
}

/** Raw stress σ(Y) = Σ_{i<j} (‖yᵢ − yⱼ‖ − δᵢⱼ)² of a configuration Y [n, r] against dissimilarities δ [n, n]. */
export function stress(embedding: Tensor, distances: Tensor): number {
  const { n, d, v } = matrix(embedding, 'stress')
  const { v: delta } = square(distances, 'stress')
  const D2 = squaredDistances(v, n, d)
  let s = 0
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) s += (Math.sqrt(D2[i * n + j]) - delta[i * n + j]) ** 2
  return s
}

/** One SMACOF state. */
export interface SmacofState extends Status {
  /** The configuration [n, r]. */
  embedding: Tensor
  /** Raw stress at this configuration. */
  stress: number
  /** Stress-1, √(σ / Σ δ²), a scale-free measure (Kruskal, 1964). */
  normalisedStress: number
  /** Guttman transforms done. */
  t: number
  converged: boolean
}

/**
 * SMACOF as a traceable algorithm: each step is the Guttman transform Y ← (1/n) B(Y) Y, where B(Y) has off-diagonal
 * entries −δᵢⱼ/dᵢⱼ(Y) (0 where dᵢⱼ = 0) and rows summing to zero; the stress never rises (de Leeuw, 1977; Borg and
 * Groenen, 2005, "Modern Multidimensional Scaling", ch. 8). Converged when the stress falls by less than `tolerance`
 * (default 1e-9) relative to itself. `init` takes a configuration, or starts from classical MDS (`start:
 * 'classical'`, default) or Gaussian noise from the `init` stream (`start: 'random'`).
 */
export function smacofSteps(
  distances: Tensor,
  params: { dims?: number; tolerance?: number } = {},
): Algorithm<{ embedding?: Tensor; start?: 'classical' | 'random' }, SmacofState> {
  const { n, v: delta } = square(distances, 'smacofSteps')
  const { dims = 2, tolerance: tol = 1e-9 } = params
  let total = 0
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) total += delta[i * n + j] ** 2
  const state = (Y: Float64Array, t: number, previous: number): SmacofState => {
    const s = stress(mat(Y, n, dims), distances)
    return {
      embedding: mat(Y, n, dims),
      stress: s,
      normalisedStress: Math.sqrt(s / total),
      t,
      diverged: !Number.isFinite(s),
      converged: Number.isFinite(previous) && previous - s <= tol * Math.max(previous, Number.MIN_VALUE),
    }
  }
  return {
    name: 'smacof',
    init: ({ embedding, start = 'classical' } = {}, s) => {
      if (embedding) return state(Float64Array.from(values(embedding)), 0, NaN)
      if (start === 'random') return state(Float64Array.from(values(normals(s, [n, dims]))), 0, NaN)
      return state(
        classicalCore(
          Float64Array.from(delta, (u) => u * u),
          n,
          dims,
        ).Y,
        0,
        NaN,
      )
    },
    step: (st) => {
      const Y = values(st.embedding)
      const D2 = squaredDistances(Y, n, dims)
      const next = new Float64Array(n * dims)
      for (let i = 0; i < n; i++) {
        let diag = 0
        for (let j = 0; j < n; j++) {
          if (j === i) continue
          const dij = Math.sqrt(D2[i * n + j])
          const b = dij > 0 ? -delta[i * n + j] / dij : 0
          diag -= b
          for (let c = 0; c < dims; c++) next[i * dims + c] += b * Y[j * dims + c]
        }
        for (let c = 0; c < dims; c++) next[i * dims + c] = (next[i * dims + c] + diag * Y[i * dims + c]) / n
      }
      return state(next, st.t + 1, st.stress)
    },
  }
}

/** A fitted metric MDS. */
export interface MetricMdsModel extends Trained<SmacofState> {
  readonly kind: 'model'
  /** Metric MDS places the training rows only: it has no out-of-sample map. */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'metric-mds'
  readonly embedding: Tensor
  readonly stress: number
}

/** Metric MDS of the rows of x (Euclidean dissimilarities) by SMACOF from classical MDS or a random start. */
export function metricMds(
  params: { dims?: number; start?: 'classical' | 'random'; maxSteps?: number; tolerance?: number } = {},
): Estimator<Dataset<Tensor>, MetricMdsModel> {
  const { dims = 2, start = 'classical', maxSteps = 300, tolerance = 1e-9 } = params
  return {
    name: 'metric-mds',
    params: { dims, start, maxSteps, tolerance },
    fit({ x }, options: FitOptions = {}) {
      const { n, d, v } = matrix(x, 'metricMds')
      const D = Float64Array.from(squaredDistances(v, n, d), Math.sqrt)
      const training = trace(smacofSteps(mat(D, n, n), { dims, tolerance }), { start }, maxSteps, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        record: { stress: (s) => s.stress },
      })
      const final = training.final
      return {
        kind: 'model',
        transductive: true,
        name: 'metric-mds',
        embedding: final.embedding,
        stress: final.stress,
        training,
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'pca',
    module: 'unsupervised/embedding/linear',
    name: 'Principal component analysis',
    summary: 'Projection onto the directions of largest variance.',
    task: 'embedding',
    capabilities: ['transform'],
    hyper: space({ components: int(1, 50, { default: 2 }), whiten: bool() }),
    notes: ['principal-component-analysis'],
    cite: ['pearson1901'],
  },
  pca,
)

defineModel(
  {
    key: 'kernelPca',
    module: 'unsupervised/embedding/linear',
    name: 'Kernel PCA',
    summary: 'PCA in the feature space of a kernel (RBF by default).',
    task: 'embedding',
    capabilities: ['transform'],
    hyper: space({ components: int(1, 50, { default: 2 }) }),
    notes: ['kernel-principal-component-analysis'],
    cite: ['scholkopf1998'],
  },
  kernelPca,
)

defineModel(
  {
    key: 'metricMds',
    module: 'unsupervised/embedding/linear',
    name: 'Metric multidimensional scaling',
    summary: 'An embedding whose distances match the data distances, by SMACOF stress majorisation.',
    task: 'embedding',
    capabilities: [],
    transductive: true,
    hyper: space({
      dims: int(1, 10, { default: 2 }),
      start: oneOf(['classical', 'random']),
      maxSteps: int(1, 1000, { default: 300 }),
    }),
    notes: ['multidimensional-scaling'],
    cite: ['kruskal1964'],
  },
  metricMds,
)
