/**
 * Linear and kernel projections, and multidimensional scaling.
 *
 * Principal component analysis is the SVD of the centred data, $\Xmat - \ones\bar{\xvec}^\top = \Umat\Smat\Vmat^\top$
 * (Pearson, 1901; Hotelling, 1933), with explained variance, whitening and reconstruction as scikit-learn's `PCA`
 * (same sign convention). Kernel PCA (Schölkopf, Smola and Müller, 1998) diagonalises the double-centred kernel
 * matrix instead. Classical MDS (Torgerson, 1952; Gower, 1966) embeds a distance matrix through the top eigenpairs of
 * $\Bmat = -\tfrac{1}{2}\Jmat\Dmat^{(2)}\Jmat$, and metric MDS minimises the raw stress
 * $\sigma(\Ymat) = \sum_{i<j} (\lVert \yvec_i - \yvec_j \rVert - \delta_{ij})^2$ by SMACOF, iterated Guttman transforms
 * (de Leeuw, 1977; Borg and Groenen, 2005, "Modern Multidimensional Scaling", §8.6).
 *
 * Data are matrices with one point per row. PCA and kernel PCA map new rows (`transform`); classical and metric MDS
 * place the given points only.
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
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'pca'
  /**
   * Principal axes as rows ($r \times d$), by decreasing variance; each signed so its largest-magnitude entry is
   * positive (the first of equals).
   */
  readonly components: Tensor
  /** Variance along each axis ($r$ values): $s_c^2 / (n - 1)$ for singular value $s_c$. */
  readonly explainedVariance: Tensor
  /** Share of the total variance along each axis ($r$ values), the total taken over all $\min(n, d)$ axes. */
  readonly explainedVarianceRatio: Tensor
  /** The singular values of the centred data along the kept axes ($r$ values, descending). */
  readonly singularValues: Tensor
  /** The mean of the training rows ($d$ values), subtracted before projecting. */
  readonly mean: Tensor
  /**
   * The mean variance of the $\min(n, d) - r$ discarded axes (0 when none are discarded): the noise level of
   * probabilistic PCA.
   */
  readonly noiseVariance: number
  /** Whether scores are divided by the square root of their axis's variance, to unit variance. */
  readonly whiten: boolean
  /**
   * Scores $(\xvec - \bar{\xvec})^\top\Vmat_{r}$ of new rows ($m \times d$ in, $m \times r$ out), divided by
   * $\sqrt{\text{variance}}$ per axis when whitening. Throws `ShapeError` for a different number of features.
   */
  transform(x: Tensor): Tensor
  /** Back from scores ($m \times r$) to the input space ($m \times d$), undoing any whitening. */
  inverseTransform(z: Tensor): Tensor
}

/**
 * Principal component analysis by the SVD of the centred training rows (Pearson, 1901; Hotelling, 1933), as
 * scikit-learn's `PCA` with the full SVD solver: the same variances (divided by $n - 1$), sign convention and noise
 * variance. Throws `ShapeError` when the data are not a matrix.
 *
 * @param params The settings of the estimator.
 * @param params.components The number of axes $r$ to keep (default all, $\min(n, d)$; more is cut to that).
 * @param params.whiten Divide each score by the square root of its axis's variance, so the scores have unit variance
 *   (default false).
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `PcaModel`.
 *
 * @example Points along a line: the first axis is its direction
 * // 20 points along the direction (0.6, 0.8), with a little noise across it.
 * const t = normals(stream(1), [20, 1])
 * const x = add(matmul(t, tensor([[3, 4]])), mul(normals(stream(2), [20, 2]), 0.1))
 * const model = pca({ components: 1 }).fit({ x })
 * print('direction =', model.components)
 * print('explained variance =', model.explainedVariance)
 * print('explained variance ratio =', model.explainedVarianceRatio)
 * print('noise variance =', model.noiseVariance)
 *
 * @example Project onto one axis and back
 * const x = tensor([[0, 0], [1, 1.2], [2, 1.8], [3, 3]])
 * const model = pca({ components: 1 }).fit({ x })
 * const z = model.transform(x)
 * print('scores =', z)
 * print('reconstruction =', model.inverseTransform(z))
 */
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
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'kernel-pca'
  /** The kernel the model was fitted with. */
  readonly kernel: Kernel
  /** The top $r$ eigenvalues of the centred kernel matrix $\tilde{\Kmat}$, descending (not divided by $n$). */
  readonly eigenvalues: Tensor
  /** Their eigenvectors as columns ($n \times r$, unit length). */
  readonly eigenvectors: Tensor
  /**
   * The training rows' coordinates ($n \times r$): each eigenvector times the square root of its eigenvalue (0 for a
   * negative one).
   */
  readonly embedding: Tensor
  /**
   * Coordinates of new rows ($m \times r$), from their kernel values against the training rows, centred with the
   * training means: $\tilde{\kvec}^\top\vvec_c / \sqrt{\lambda_c}$ (0 where $\lambda_c \le 0$). On the training rows
   * this gives `embedding`. Throws `ShapeError` for a different number of features.
   */
  transform(x: Tensor): Tensor
}

/**
 * Kernel PCA (Schölkopf, Smola and Müller, 1998): the eigendecomposition of the double-centred kernel matrix
 * $\tilde{\Kmat} = \Kmat - \ones\ones^\top\Kmat/n - \Kmat\ones\ones^\top/n + \ones\ones^\top\Kmat\ones\ones^\top/n^2$
 * of the training rows. The same eigenvalues as scikit-learn's `KernelPCA`, and the same coordinates up to the sign of
 * each axis. Throws `ShapeError` when the data are not a matrix.
 *
 * @param params The settings of the estimator.
 * @param params.kernel The kernel, from `aifn-compute/learning/kernels` (default `rbf({ lengthscale: 1 })`,
 *   $k(\xvec, \xvec') = \exp(-\lVert \xvec - \xvec' \rVert^2 / 2)$, scikit-learn's `gamma=0.5`).
 * @param params.components The number of axes $r$ to keep (default 2).
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `KernelPcaModel`.
 *
 * @example Two groups of points come apart along the first axis
 * const x = tensor([[0, 0], [1, 0], [0, 1], [3, 3], [4, 3]])
 * const model = kernelPca({ components: 2 }).fit({ x })
 * print('eigenvalues =', model.eigenvalues)
 * print('embedding =', model.embedding)
 * print('a new point near the second group:', model.transform(tensor([[3.5, 3]])))
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
  /** Coordinates ($n \times r$), one point per row; a column whose eigenvalue is negative is zero. */
  embedding: Tensor
  /**
   * All $n$ eigenvalues of $\Bmat = -\tfrac{1}{2}\Jmat\Dmat^{(2)}\Jmat$, descending (negative ones mean the distances
   * are not Euclidean).
   */
  eigenvalues: Tensor
}

/**
 * Classical (Torgerson–Gower) MDS of a distance matrix: the coordinates $\Ymat = \Vmat_{r}\Lambdamat_{r}^{1/2}$ from
 * the top eigenpairs of $\Bmat = -\tfrac{1}{2}\Jmat\Dmat^{(2)}\Jmat$, $\Jmat = \Imat - \ones\ones^\top/n$
 * (Torgerson, 1952; Gower, 1966). Euclidean distances are recovered exactly, up to rotation and reflection, once `dims`
 * covers their dimension. Throws `ShapeError` when `distances` is not square.
 *
 * @param distances The distances $\Dmat$ ($n \times n$, symmetric, zero diagonal); they are squared here.
 * @param dims The number of coordinates $r$ to keep.
 * @returns The coordinates and every eigenvalue of $\Bmat$.
 *
 * @example A 3-4-5 triangle is recovered from its distances
 * const D = tensor([[0, 3, 4], [3, 0, 5], [4, 5, 0]])
 * const { embedding, eigenvalues } = classicalMds(D)
 * print('embedding =', embedding)
 * print('eigenvalues =', eigenvalues)
 * print('stress (0 when every distance is matched) =', stress(embedding, D))
 *
 * @example A negative eigenvalue says the distances are not Euclidean
 * // d(0, 3) = 3 breaks the triangle inequality through point 1: d(0, 1) + d(1, 3) = 2.
 * const D = tensor([[0, 1, 1, 3], [1, 0, 1, 1], [1, 1, 0, 1], [3, 1, 1, 0]])
 * print('eigenvalues =', classicalMds(D).eigenvalues)
 */
export function classicalMds(distances: Tensor, dims = 2): ClassicalMds {
  const { n, v } = square(distances, 'classicalMds')
  const { Y, eigenvalues } = classicalCore(
    Float64Array.from(v, (u) => u * u),
    n,
    dims,
  )
  return { embedding: mat(Y, n, dims), eigenvalues: vec(eigenvalues) }
}

/**
 * Raw stress $\sigma(\Ymat) = \sum_{i<j} (\lVert \yvec_i - \yvec_j \rVert - \delta_{ij})^2$ of a configuration
 * against dissimilarities: what metric MDS minimises (Kruskal, 1964). Throws `ShapeError` for a non-matrix
 * configuration or non-square dissimilarities.
 *
 * @param embedding The configuration $\Ymat$ ($n \times r$), one point per row.
 * @param distances The dissimilarities $\delta_{ij}$ ($n \times n$); only the upper triangle is read.
 * @returns The raw stress, 0 when every distance is matched.
 *
 * @example A right triangle against its own distances, and a squashed one
 * const D = tensor([[0, 3, 4], [3, 0, 5], [4, 5, 0]])
 * print('exact =', stress(tensor([[0, 0], [3, 0], [0, 4]]), D))
 * print('squashed =', stress(tensor([[0, 0], [3, 0], [0, 3]]), D))
 */
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
  /** The configuration ($n \times r$), one point per row. */
  embedding: Tensor
  /** Raw stress $\sigma$ at this configuration. */
  stress: number
  /**
   * $\sqrt{\sigma / \sum_{i<j} \delta_{ij}^2}$, a scale-free measure: Kruskal's (1964) stress-1 at a SMACOF fixed
   * point.
   */
  normalisedStress: number
  /** Guttman transforms done. */
  t: number
  /** The last transform lowered the stress by at most `tolerance` times its previous value. */
  converged: boolean
}

/**
 * SMACOF as a traceable algorithm: each step is the Guttman transform $\Ymat \leftarrow \tfrac{1}{n}\Bmat(\Ymat)\Ymat$,
 * where $\Bmat(\Ymat)$ has off-diagonal entries $-\delta_{ij}/d_{ij}(\Ymat)$ (0 where $d_{ij} = 0$) and rows summing
 * to zero; the stress never rises (de Leeuw, 1977; Borg and Groenen, 2005, "Modern Multidimensional Scaling", ch. 8).
 * Converged when the stress falls by no more than `tolerance` relative to its previous value. `init` takes a
 * configuration (`embedding`), or starts from classical MDS (`start: 'classical'`, default) or standard normal noise
 * from the run's stream (`start: 'random'`). Throws `ShapeError` when `distances` is not square.
 *
 * @param distances The dissimilarities $\delta_{ij}$ to match ($n \times n$, symmetric, zero diagonal).
 * @param params The settings of the algorithm.
 * @param params.dims The dimension $r$ of the configuration (default 2).
 * @param params.tolerance The relative fall in stress below which a step counts as converged (default 1e-9).
 * @returns The algorithm, for `run` or `trace`; its states are `SmacofState`s.
 *
 * @example The stress of a random start falls with each transform
 * const D = tensor([[0, 3, 4], [3, 0, 5], [4, 5, 0]])
 * const smacof = smacofSteps(D)
 * print('stress at the random start =', run(smacof, { start: 'random' }, 0, { stream: stream(1) }).stress)
 * print('after 5 transforms =', run(smacof, { start: 'random' }, 5, { stream: stream(1) }).stress)
 * print('after 20 transforms =', run(smacof, { start: 'random' }, 20, { stream: stream(1) }).stress)
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

/** A fitted metric MDS, with the SMACOF run that placed it (`training`). */
export interface MetricMdsModel extends Trained<SmacofState> {
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** Metric MDS places the training rows only: it has no out-of-sample map. */
  readonly transductive: true
  /** The model's name. */
  readonly name: 'metric-mds'
  /** The coordinates of the training rows ($n \times r$), from the last SMACOF state. */
  readonly embedding: Tensor
  /** The raw stress of `embedding` against the rows' Euclidean distances. */
  readonly stress: number
}

/**
 * Metric MDS of the training rows: their Euclidean distances matched by SMACOF (`smacofSteps`), from classical MDS or a
 * random start, until the stress stops falling or `maxSteps` run out. The run is traced (every step by default, or
 * every `trace.every` of the fit options) and the random start draws from the fit options' `stream`. Transductive:
 * there is no `transform`.
 *
 * @param params The settings of the estimator.
 * @param params.dims The dimension $r$ of the embedding (default 2).
 * @param params.start Where SMACOF starts: `'classical'` (classical MDS, default) or `'random'` (standard normal).
 * @param params.maxSteps The most Guttman transforms to run (default 300).
 * @param params.tolerance The relative fall in stress below which the run stops (default 1e-9).
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `MetricMdsModel`.
 *
 * @example Six points in three dimensions, placed in two
 * const x = normals(stream(3), [6, 3])
 * const model = metricMds({ dims: 2 }).fit({ x })
 * print('stress =', model.stress, 'after', model.training.final.t, 'transforms')
 * print('normalised stress =', model.training.final.normalisedStress)
 */
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
