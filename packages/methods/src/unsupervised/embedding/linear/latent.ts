/**
 * Linear latent-variable models, $\xvec = \Wmat\zvec + \muvec + \epsilonvec$ with
 * $\zvec \sim \Gauss(\zeros, \Imat_{q})$ so that $\xvec \sim \Gauss(\muvec, \Wmat\Wmat^\top + \Psimat)$, and
 * independent component analysis.
 *
 * The loadings $\Wmat$ ($d \times q$) and the noise are fitted by EM (Rubin and Thayer, 1982; Ghahramani and Hinton,
 * 1996), with diagonal noise $\Psimat$ (factor analysis) or isotropic noise $\sigma^2\Imat$ (probabilistic PCA;
 * Tipping and Bishop, 1999), as a step-through algorithm on the sample covariance whose log-likelihood never
 * decreases. Probabilistic PCA also has a closed form, $\Wmat = \Umat_{q}(\Lambdamat_{q} - \sigma^2\Imat)^{1/2}$ with
 * $\sigma^2$ the mean discarded eigenvalue. Scores are posterior means $\expect[\zvec \mid \xvec]$.
 *
 * FastICA (Hyvärinen, 1999; Hyvärinen and Oja, 2000) whitens the data to $\zvec$, then runs the symmetric fixed-point
 * iteration
 *
 * $\Wmat \leftarrow (\Wmat\Wmat^\top)^{-1/2} (\expect[g(\Wmat\zvec)\zvec^\top] - \diag(\expect[g'(\Wmat\zvec)])\Wmat)$
 *
 * with the log-cosh contrast, $g = \tanh$, as scikit-learn's `FastICA(algorithm='parallel', whiten='unit-variance')`.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { normals } from 'aifn-compute/foundation/random'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Dataset, Estimator, FitOptions, Trained, Transforms } from 'aifn-compute/learning/estimators'
import { defineModel } from 'aifn-compute/learning/estimators'
import { eigh, inverse, logDet, pinv, symmetricInverseSqrt } from 'aifn-compute/numerics/linalg'
import { int, oneOf, space } from 'aifn-compute/foundation/space'
import { mat, matrix, values, vec } from '../util'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const { matMul, transpose } = dense
const LOG_2PI = Math.log(2 * Math.PI)

/**
 * Column means and the sample covariance
 * $\Smat = \tfrac{1}{n} \sum_i (\xvec_i - \bar{\xvec})(\xvec_i - \bar{\xvec})^\top$ (the maximum-likelihood one,
 * divided by $n$).
 *
 * @param v The data as a row-major array of $n \times d$ values, one row per point (read, not modified).
 * @param n The number of rows.
 * @param d The number of features.
 * @returns `mean`, the $d$ column means, and `S`, the covariance as a row-major array of $d^2$ values.
 */
function moments(v: Float64Array, n: number, d: number) {
  const mean = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += v[i * d + j] / n
  const Xc = Float64Array.from(v, (u, t) => u - mean[t % d])
  const S = matMul(transpose(Xc, n, d), Xc, d, n, d)
  for (let i = 0; i < S.length; i++) S[i] /= n
  return { mean, S }
}

/**
 * The inverse of a small square matrix, by `inverse` of `aifn-compute/numerics/linalg` (which throws for a singular
 * one).
 *
 * @param A The matrix as a row-major array of $q^2$ values.
 * @param q Its number of rows and columns.
 * @returns $\Amat^{-1}$ as a row-major array of $q^2$ values.
 */
const inv = (A: Float64Array, q: number) => values(inverse(mat(A, q, q)))

/**
 * The model covariance $\Cmat = \Wmat\Wmat^\top + \diag(\psivec)$.
 *
 * @param W The loadings $\Wmat$ as a row-major array of $d \times q$ values.
 * @param psi The noise variances $\psivec$, $d$ values.
 * @param d The number of features.
 * @param q The number of latent factors.
 * @returns $\Cmat$ as a row-major array of $d^2$ values.
 */
function modelCovariance(W: Float64Array, psi: Float64Array, d: number, q: number): Float64Array {
  const C = matMul(W, transpose(W, d, q), d, q, d)
  for (let i = 0; i < d; i++) C[i * d + i] += psi[i]
  return C
}

/**
 * The average log-likelihood per row of data with sample covariance $\Smat$ (about the model's mean) under
 * $\Gauss(\muvec, \Cmat)$: $-\tfrac{1}{2}(d \log 2\pi + \log\det\Cmat + \trace(\Cmat^{-1}\Smat))$, with
 * $\Cmat = \Wmat\Wmat^\top + \diag(\psivec)$.
 *
 * @param S The sample covariance $\Smat$ (divided by $n$) as a row-major array of $d^2$ values.
 * @param W The loadings $\Wmat$ as a row-major array of $d \times q$ values.
 * @param psi The noise variances $\psivec$, $d$ values.
 * @param d The number of features.
 * @param q The number of latent factors.
 * @returns The average log-likelihood per row.
 */
function averageLogLikelihood(S: Float64Array, W: Float64Array, psi: Float64Array, d: number, q: number): number {
  const C = mat(modelCovariance(W, psi, d, q), d, d)
  const Ci = values(inverse(C))
  let tr = 0
  for (let i = 0; i < d * d; i++) tr += Ci[i] * S[i]
  return -0.5 * (d * LOG_2PI + (logDet(C) as number) + tr)
}

/** The noise model: one variance per feature (factor analysis) or one shared variance (PPCA). */
export type LatentNoise = 'diagonal' | 'isotropic'

/** A state of `latentGaussianSteps`. */
export interface LatentGaussianState extends Status {
  /** EM steps done. */
  t: number
  /** Loadings $\Wmat$ ($d \times q$). */
  loadings: Tensor
  /** Noise variances $\psivec$ ($d$ values; all equal for isotropic noise). */
  noise: Tensor
  /** Average log-likelihood per row of the training data. */
  logLikelihood: number
  /** The last step raised the log-likelihood by less than `tolerance`. */
  converged: boolean
}

/**
 * EM for $\xvec = \Wmat\zvec + \muvec + \epsilonvec$ on the rows of `x`, as a step-through algorithm on their sample
 * covariance $\Smat$ (Rubin and Thayer, 1982; Ghahramani and Hinton, 1996). With
 * $\Mmat = \Imat + \Wmat^\top\Psimat^{-1}\Wmat$ and $\betavec = \Mmat^{-1}\Wmat^\top\Psimat^{-1}$ (so
 * $\expect[\zvec \mid \xvec] = \betavec(\xvec - \muvec)$), one step sets
 * $\Wmat \leftarrow \Smat\betavec^\top(\Mmat^{-1} + \betavec\Smat\betavec^\top)^{-1}$ and the noise to
 * $\diag(\Smat - \Wmat\betavec\Smat)$ with the new $\Wmat$ (floored at $10^{-12}$), or its mean for isotropic noise.
 * The log-likelihood never decreases. The initial loadings are normal with standard deviation
 * $\sqrt{\trace(\Smat)/d}/q$, drawn from the run's stream, unless given; the initial noise is $\diag(\Smat)$ (its
 * mean for isotropic noise). Throws `DomainError` unless `latent` is a whole number from 1 to $d - 1$.
 *
 * @param x The data ($n \times d$), one row per point.
 * @param params The settings of the algorithm.
 * @param params.latent The number of latent factors $q$.
 * @param params.noise `'diagonal'` (default): one variance per feature, factor analysis. `'isotropic'`: one shared
 *   variance, probabilistic PCA.
 * @param params.tolerance The rise in average log-likelihood below which a step counts as converged (default 1e-8).
 * @returns The algorithm, for `run` or `trace`; it starts from `{ loadings }` ($d \times q$) or from nothing.
 *
 * @example The log-likelihood rises with each EM step
 * const z = normals(stream(1), [100, 1])
 * const x = add(matmul(z, tensor([[2, 1, 0.5]])), mul(normals(stream(2), [100, 3]), 0.5))
 * const em = latentGaussianSteps(x, { latent: 1 })
 * print('at the start:', run(em, undefined, 0, { stream: stream(3) }).logLikelihood)
 * print('after 1 step:', run(em, undefined, 1, { stream: stream(3) }).logLikelihood)
 * print('after 5 steps:', run(em, undefined, 5, { stream: stream(3) }).logLikelihood)
 * print('after 50 steps:', run(em, undefined, 50, { stream: stream(3) }).logLikelihood)
 */
export function latentGaussianSteps(
  x: Tensor,
  params: { latent: number; noise?: LatentNoise; tolerance?: number },
): Algorithm<{ loadings?: Tensor } | void, LatentGaussianState> {
  const { n, d, v } = matrix(x, 'latentGaussianSteps')
  const { latent: q, noise = 'diagonal', tolerance = 1e-8 } = params
  if (!(Number.isInteger(q) && q >= 1 && q < d))
    throw new DomainError('latentGaussianSteps', `latentGaussianSteps: latent must lie in 1 … ${d - 1}`)
  const { S } = moments(v, n, d)
  const state = (W: Float64Array, psi: Float64Array, t: number, previous: number): LatentGaussianState => {
    const ll = averageLogLikelihood(S, W, psi, d, q)
    return {
      t,
      loadings: mat(W, d, q),
      noise: vec(psi),
      logLikelihood: ll,
      converged: Number.isFinite(previous) && ll - previous < tolerance,
      diverged: !Number.isFinite(ll),
    }
  }
  return {
    name: noise === 'diagonal' ? 'factor-analysis-em' : 'ppca-em',
    init: (input, s) => {
      let W: Float64Array
      if (input && input.loadings) W = Float64Array.from(values(input.loadings))
      else {
        let tr = 0
        for (let i = 0; i < d; i++) tr += S[i * d + i]
        const scale = Math.sqrt(tr / d) / q
        W = Float64Array.from(values(normals(s, [d, q])), (u) => u * scale)
      }
      const diag = Float64Array.from({ length: d }, (_, i) => S[i * d + i])
      const avg = diag.reduce((a, b) => a + b, 0) / d
      return state(W, noise === 'diagonal' ? diag : diag.fill(avg), 0, NaN)
    },
    step: (st) => {
      const W = values(st.loadings)
      const psi = values(st.noise)
      // β = M⁻¹ WᵀΨ⁻¹ [q, d], M = I + WᵀΨ⁻¹W [q, q].
      const WtPi = new Float64Array(q * d)
      for (let a = 0; a < q; a++) for (let j = 0; j < d; j++) WtPi[a * d + j] = W[j * q + a] / psi[j]
      const M = matMul(WtPi, W, q, d, q)
      for (let a = 0; a < q; a++) M[a * q + a] += 1
      const Mi = inv(M, q)
      const beta = matMul(Mi, WtPi, q, q, d)
      const SBt = matMul(S, transpose(beta, q, d), d, d, q) // S βᵀ [d, q]
      const Ezz = matMul(beta, SBt, q, d, q) // β S βᵀ
      for (let i = 0; i < q * q; i++) Ezz[i] += Mi[i]
      const Wn = matMul(SBt, inv(Ezz, q), d, q, q)
      // diag(S − Wn β S) = diag(S) − rowwise Σ_a Wn[j, a] (S βᵀ)[j, a].
      const next = new Float64Array(d)
      for (let j = 0; j < d; j++) {
        let s = S[j * d + j]
        for (let a = 0; a < q; a++) s -= Wn[j * q + a] * SBt[j * q + a]
        next[j] = Math.max(s, 1e-12)
      }
      if (noise === 'isotropic') next.fill(next.reduce((a, b) => a + b, 0) / d)
      return state(Wn, next, st.t + 1, st.logLikelihood)
    },
  }
}

/** A fitted linear-Gaussian latent model (factor analysis or PPCA). */
export interface LatentGaussianModel extends Transforms<Tensor, Tensor> {
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'factor-analysis' | 'probabilistic-pca'
  /** Loadings $\Wmat$ ($d \times q$). */
  readonly loadings: Tensor
  /** Noise variances $\psivec$ ($d$ values). */
  readonly noise: Tensor
  /** The mean $\muvec$ of the training rows ($d$ values). */
  readonly mean: Tensor
  /** Average log-likelihood per training row. */
  readonly logLikelihood: number
  /** The model covariance $\Wmat\Wmat^\top + \Psimat$ ($d \times d$). */
  readonly covariance: Tensor
  /**
   * Posterior means
   * $\expect[\zvec \mid \xvec] = (\Imat + \Wmat^\top\Psimat^{-1}\Wmat)^{-1}\Wmat^\top\Psimat^{-1}(\xvec - \muvec)$
   * of new rows ($m \times d$ in, $m \times q$ out). Throws `ShapeError` for a different number of features.
   */
  transform(x: Tensor): Tensor
  /**
   * The average log-likelihood per row of new rows ($m \times d$) under the fitted Gaussian. Throws `ShapeError` for a
   * different number of features.
   */
  averageLogLikelihood(x: Tensor): number
}

/**
 * The fitted model of factor analysis or PPCA from its parameters: precomputes the posterior-mean map
 * $\betavec = (\Imat + \Wmat^\top\Psimat^{-1}\Wmat)^{-1}\Wmat^\top\Psimat^{-1}$.
 *
 * @param name The model's name, also used in error messages.
 * @param W The loadings $\Wmat$ as a row-major array of $d \times q$ values (kept by the model: not to be modified).
 * @param psi The noise variances $\psivec$, $d$ positive values (kept by the model).
 * @param mean The training mean $\muvec$, $d$ values (kept by the model).
 * @param d The number of features.
 * @param q The number of latent factors.
 * @param logLikelihood The average log-likelihood per training row, reported as is.
 * @returns The fitted model.
 */
function latentModel(
  name: LatentGaussianModel['name'],
  W: Float64Array,
  psi: Float64Array,
  mean: Float64Array,
  d: number,
  q: number,
  logLikelihood: number,
): LatentGaussianModel {
  const WtPi = new Float64Array(q * d)
  for (let a = 0; a < q; a++) for (let j = 0; j < d; j++) WtPi[a * d + j] = W[j * q + a] / psi[j]
  const M = matMul(WtPi, W, q, d, q)
  for (let a = 0; a < q; a++) M[a * q + a] += 1
  const beta = matMul(inv(M, q), WtPi, q, q, d)
  return {
    kind: 'model',
    name,
    loadings: mat(W, d, q),
    noise: vec(psi),
    mean: vec(mean),
    logLikelihood,
    covariance: mat(modelCovariance(W, psi, d, q), d, d),
    transform: (z: Tensor) => {
      const { n, d: dz, v } = matrix(z, `${name}.transform`)
      if (dz !== d) throw new ShapeError(name, `${name}: fitted on ${d} features, given ${dz}`)
      const Xc = Float64Array.from(v, (u, t) => u - mean[t % d])
      return mat(matMul(Xc, transpose(beta, q, d), n, d, q), n, q)
    },
    averageLogLikelihood: (z: Tensor) => {
      const { n, d: dz, v } = matrix(z, `${name}.averageLogLikelihood`)
      if (dz !== d) throw new ShapeError(name, `${name}: fitted on ${d} features, given ${dz}`)
      const Xc = Float64Array.from(v, (u, t) => u - mean[t % d])
      const S = matMul(transpose(Xc, n, d), Xc, d, n, d)
      for (let i = 0; i < S.length; i++) S[i] /= n
      return averageLogLikelihood(S, W, psi, d, q)
    },
  }
}

/** A fitted factor analysis, with its EM trace (`training`). */
export interface FactorAnalysisModel extends LatentGaussianModel, Trained<LatentGaussianState> {
  /** The model's name. */
  readonly name: 'factor-analysis'
}

/**
 * Factor analysis: a Gaussian with covariance $\Wmat\Wmat^\top + \Psimat$ ($\Psimat$ diagonal), fitted by EM
 * (`latentGaussianSteps` with diagonal noise) until the log-likelihood stops rising or `maxSteps` run out. The run is
 * traced (every step by default, or every `trace.every` of the fit options) and starts from random loadings drawn
 * from the fit options' `stream`. The maximum-likelihood fit of scikit-learn's `FactorAnalysis`, which uses another
 * algorithm; loadings agree up to the sign (or rotation) of the factors. Throws `DomainError` unless `latent` is a
 * whole number from 1 to $d - 1$.
 *
 * @param params The settings of the estimator.
 * @param params.latent The number of latent factors $q$ (default 2).
 * @param params.maxSteps The most EM steps to run (default 1000).
 * @param params.tolerance The rise in average log-likelihood below which the run stops (default 1e-8).
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `FactorAnalysisModel`.
 *
 * @example One factor behind four features, each with its own noise
 * const z = normals(stream(1), [200, 1])
 * const noise = mul(normals(stream(2), [200, 4]), tensor([0.3, 0.5, 0.7, 0.9]))
 * const x = add(matmul(z, tensor([[1, 1, 1, 1]])), noise)
 * const model = factorAnalysis({ latent: 1 }).fit({ x }, { stream: stream(3) })
 * print('loadings =', model.loadings)
 * print('noise variances =', model.noise)
 * print('log-likelihood per row =', model.logLikelihood, 'after', model.training.final.t, 'EM steps')
 */
export function factorAnalysis(
  params: { latent?: number; maxSteps?: number; tolerance?: number } = {},
): Estimator<Dataset<Tensor>, FactorAnalysisModel> {
  const { latent = 2, maxSteps = 1000, tolerance = 1e-8 } = params
  return {
    name: 'factor-analysis',
    params: { latent, maxSteps, tolerance },
    fit({ x }, options: FitOptions = {}) {
      const { n, d, v } = matrix(x, 'factorAnalysis')
      const training = trace(latentGaussianSteps(x, { latent, noise: 'diagonal', tolerance }), undefined, maxSteps, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        record: { logLikelihood: (s) => s.logLikelihood },
      })
      const f = training.final
      const model = latentModel(
        'factor-analysis',
        Float64Array.from(values(f.loadings)),
        Float64Array.from(values(f.noise)),
        moments(v, n, d).mean,
        d,
        latent,
        f.logLikelihood,
      )
      return { ...model, name: 'factor-analysis', training }
    },
  }
}

/** A fitted probabilistic PCA. */
export interface ProbabilisticPcaModel extends LatentGaussianModel {
  /** The model's name. */
  readonly name: 'probabilistic-pca'
  /** The shared noise variance $\sigma^2$. */
  readonly noiseVariance: number
}

/**
 * Probabilistic PCA (Tipping and Bishop, 1999): the maximum-likelihood solution in closed form,
 * $\Wmat = \Umat_{q}(\Lambdamat_{q} - \sigma^2\Imat)^{1/2}$ from the eigendecomposition of the sample covariance
 * (divided by $n$) with $\sigma^2$ the mean of the $d - q$ discarded eigenvalues, and each column of $\Wmat$ signed
 * so its largest-magnitude entry is positive; or by EM (`latentGaussianSteps` with isotropic noise, from random
 * loadings drawn from the fit options' `stream`), which reaches the same fit up to the sign or rotation of the loadings. The
 * closed form is scikit-learn's `PCA` read as PPCA, with the variances divided by $n$ rather than $n - 1$. Throws
 * `DomainError` unless $1 \le q \le d - 1$.
 *
 * @param params The settings of the estimator.
 * @param params.latent The number of latent dimensions $q$ (default 2).
 * @param params.method `'closed-form'` (default) or `'em'`.
 * @param params.maxSteps The most EM steps to run (default 1000; EM only).
 * @param params.tolerance The rise in average log-likelihood below which EM stops (default 1e-10; EM only).
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `ProbabilisticPcaModel` (with no
 *   `training`: the EM run is not kept).
 *
 * @example The closed form and EM reach the same fit
 * const z = normals(stream(1), [100, 1])
 * const x = add(matmul(z, tensor([[2, 1, 0.5]])), mul(normals(stream(2), [100, 3]), 0.5))
 * const closed = probabilisticPca({ latent: 1 }).fit({ x })
 * const em = probabilisticPca({ latent: 1, method: 'em' }).fit({ x }, { stream: stream(3) })
 * print('closed form: loadings =', closed.loadings, 'noise variance =', closed.noiseVariance)
 * print('EM: loadings =', em.loadings, 'noise variance =', em.noiseVariance)
 * print('log-likelihood per row:', closed.logLikelihood, em.logLikelihood)
 */
export function probabilisticPca(
  params: { latent?: number; method?: 'closed-form' | 'em'; maxSteps?: number; tolerance?: number } = {},
): Estimator<Dataset<Tensor>, ProbabilisticPcaModel> {
  const { latent: q = 2, method = 'closed-form', maxSteps = 1000, tolerance = 1e-10 } = params
  return {
    name: 'probabilistic-pca',
    params: { latent: q, method, maxSteps, tolerance },
    fit({ x }, options: FitOptions = {}) {
      const { n, d, v } = matrix(x, 'probabilisticPca')
      if (!(q >= 1 && q < d))
        throw new DomainError('probabilisticPca', `probabilisticPca: latent must lie in 1 … ${d - 1}`)
      const { mean, S } = moments(v, n, d)
      let W: Float64Array
      let sigma2: number
      if (method === 'em') {
        const f = trace(latentGaussianSteps(x, { latent: q, noise: 'isotropic', tolerance }), undefined, maxSteps, {
          stream: options.stream,
          every: maxSteps,
        }).final
        W = Float64Array.from(values(f.loadings))
        sigma2 = values(f.noise)[0]
      } else {
        const e = eigh(mat(S, d, d))
        const lambda = values(e.values)
        const U = values(e.vectors)
        sigma2 = 0
        for (let c = q; c < d; c++) sigma2 += lambda[c] / (d - q)
        W = new Float64Array(d * q)
        for (let c = 0; c < q; c++) {
          const scale = Math.sqrt(Math.max(lambda[c] - sigma2, 0))
          for (let j = 0; j < d; j++) W[j * q + c] = U[j * d + c] * scale
        }
      }
      const psi = new Float64Array(d).fill(sigma2)
      const model = latentModel('probabilistic-pca', W, psi, mean, d, q, averageLogLikelihood(S, W, psi, d, q))
      return { ...model, name: 'probabilistic-pca', noiseVariance: sigma2 }
    },
  }
}

// ── FastICA ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The symmetric decorrelation $(\Wmat\Wmat^\top)^{-1/2}\Wmat$, which makes the rows of $\Wmat$ orthonormal while
 * treating them alike.
 *
 * @param W The matrix $\Wmat$ as a row-major array of $r^2$ values (read, not modified).
 * @param r Its number of rows and columns.
 * @returns The decorrelated matrix as a new row-major array of $r^2$ values.
 */
function symmetricDecorrelation(W: Float64Array, r: number): Float64Array {
  const P = values(symmetricInverseSqrt(mat(matMul(W, transpose(W, r, r), r, r, r), r, r), { floor: Number.MIN_VALUE }))
  return matMul(P, W, r, r, r)
}

/**
 * The whitening of FastICA, as scikit-learn's SVD solver: the matrix $\Kmat$ ($r \times d$) whose rows are the top $r$
 * eigenvectors of the sample covariance (divided by $n$), each divided by the square root of its eigenvalue and signed
 * so its first entry is non-negative, so that $\Kmat(\xvec - \bar{\xvec})$ has identity covariance.
 *
 * @param v The data as a row-major array of $n \times d$ values (read, not modified).
 * @param n The number of rows.
 * @param d The number of features.
 * @param r The number of whitened components to keep, at most $d$.
 * @returns `mean` ($d$ values), `K` (row-major, $r \times d$) and `Z`, the whitened data as a row-major $r \times n$
 *   array: one row per component, one column per point.
 */
function whitening(v: Float64Array, n: number, d: number, r: number) {
  const { mean, S } = moments(v, n, d)
  const e = eigh(mat(S, d, d))
  const lambda = values(e.values)
  const U = values(e.vectors)
  const K = new Float64Array(r * d)
  for (let c = 0; c < r; c++) {
    // scikit-learn signs each left singular vector so that its first entry is non-negative.
    const sign = U[c] < 0 ? -1 : 1
    for (let j = 0; j < d; j++) K[c * d + j] = (sign * U[j * d + c]) / Math.sqrt(lambda[c])
  }
  const Xc = Float64Array.from(v, (u, t) => u - mean[t % d])
  // Whitened data [r, n].
  const Z = matMul(K, transpose(Xc, n, d), r, d, n)
  return { mean, K, Z }
}

/** A state of `fastIcaSteps`. */
export interface FastIcaState extends Status {
  /** Fixed-point steps done. */
  t: number
  /** The unmixing matrix $\Wmat$ in whitened coordinates ($r \times r$, orthonormal rows). */
  unmixing: Tensor
  /**
   * $\max_i \lvert \lvert \langle \wvec_i, \wvec_i^{\text{old}} \rangle \rvert - 1 \rvert \in [0, 1]$: 0 when every row
   * has stopped turning (1 before the first step).
   */
  change: number
  /** `change` is below `tolerance`. */
  converged: boolean
}

/**
 * FastICA's symmetric fixed-point iteration (Hyvärinen, 1999): the rows of `x` are whitened to $\zvec$, then each step
 * sets $\Wmat \leftarrow (\Wmat\Wmat^\top)^{-1/2}\Wmat_{+}$ with
 * $\Wmat_{+} = \expect[\tanh(\Wmat\zvec)\zvec^\top] - \diag(\expect[1 - \tanh^2(\Wmat\zvec)])\Wmat$, the
 * expectations being means over the rows. Converged (and the run
 * stops) when `change` is below `tolerance`. The initial $\Wmat$ is the one given (`{ unmixing }`, $r \times r$), or
 * standard normal from the run's stream, decorrelated. Throws `ShapeError` when `x` is not a matrix.
 *
 * @param x The mixed signals ($n \times d$), one row per observation.
 * @param params The settings of the algorithm.
 * @param params.components The number of sources $r$ to recover (default $d$; more is cut to $d$).
 * @param params.tolerance The `change` below which the iteration has converged (default 1e-4).
 * @returns The algorithm, for `run` or `trace`; its states are `FastIcaState`s.
 *
 * @example The rows stop turning within a few steps
 * const t = linspace(0, 8, 200)
 * const x = matmul(stack([sin(mul(t, 2)), sign(sin(mul(t, 3)))], 1), tensor([[1, 0.5], [1, 2]]))
 * const ica = fastIcaSteps(x)
 * print('change after 1 step:', run(ica, undefined, 1, { stream: stream(1) }).change)
 * print('after 2 steps:', run(ica, undefined, 2, { stream: stream(1) }).change)
 * print('after 3 steps:', run(ica, undefined, 3, { stream: stream(1) }).change)
 */
export function fastIcaSteps(
  x: Tensor,
  params: { components?: number; tolerance?: number } = {},
): Algorithm<{ unmixing?: Tensor } | void, FastIcaState> {
  const { n, d, v } = matrix(x, 'fastIcaSteps')
  const r = Math.min(params.components ?? d, d)
  const tolerance = params.tolerance ?? 1e-4
  const { Z } = whitening(v, n, d, r)
  return {
    name: 'fast-ica',
    init: (input, s) => {
      const W0 = input && input.unmixing ? Float64Array.from(values(input.unmixing)) : values(normals(s, [r, r]))
      return { t: 0, unmixing: mat(symmetricDecorrelation(W0, r), r, r), change: 1, converged: false }
    },
    step: (st) => {
      const W = values(st.unmixing)
      const Y = matMul(W, Z, r, r, n)
      const G = new Float64Array(r * n)
      const gPrime = new Float64Array(r)
      for (let i = 0; i < r; i++)
        for (let k = 0; k < n; k++) {
          const t = Math.tanh(Y[i * n + k])
          G[i * n + k] = t
          gPrime[i] += (1 - t * t) / n
        }
      const W1 = matMul(G, transpose(Z, r, n), r, n, r)
      for (let i = 0; i < r; i++)
        for (let j = 0; j < r; j++) W1[i * r + j] = W1[i * r + j] / n - gPrime[i] * W[i * r + j]
      const next = symmetricDecorrelation(W1, r)
      let change = 0
      for (let i = 0; i < r; i++) {
        let dot = 0
        for (let j = 0; j < r; j++) dot += next[i * r + j] * W[i * r + j]
        change = Math.max(change, Math.abs(Math.abs(dot) - 1))
      }
      const diverged = !next.every(Number.isFinite)
      return { t: st.t + 1, unmixing: mat(next, r, r), change, converged: change < tolerance, diverged }
    },
    done: (st) => st.converged,
  }
}

/** A fitted ICA, with the run that found it (`training`). */
export interface FastIcaModel extends Transforms<Tensor, Tensor>, Trained<FastIcaState> {
  /** Marks a fitted model. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'fast-ica'
  /**
   * The unmixing matrix ($r \times d$), whitening included: the sources are
   * $\svec = \text{components}(\xvec - \bar{\xvec})$, each of unit variance on the training rows.
   */
  readonly components: Tensor
  /** The mixing matrix ($d \times r$), the pseudo-inverse of `components`; its columns are the sources' directions. */
  readonly mixing: Tensor
  /** The mean of the training rows ($d$ values). */
  readonly mean: Tensor
  /**
   * Estimated sources of new rows ($m \times d$ in, $m \times r$ out). Throws `ShapeError` for a different number of
   * features.
   */
  transform(x: Tensor): Tensor
  /** Back from sources ($m \times r$) to the input space ($m \times d$): $\text{mixing} \cdot \svec + \bar{\xvec}$. */
  inverseTransform(s: Tensor): Tensor
}

/**
 * Independent component analysis by FastICA (Hyvärinen and Oja, 2000): `fastIcaSteps` to convergence or `maxSteps`,
 * then each source scaled to unit variance, as scikit-learn's `FastICA(algorithm='parallel',
 * whiten='unit-variance')`. The sources come back in an arbitrary order and sign. The run is traced (every step by
 * default, or every `trace.every` of the fit options) and a random start draws from the fit options' `stream`.
 *
 * @param params The settings of the estimator.
 * @param params.components The number of sources $r$ (default all $d$).
 * @param params.maxSteps The most fixed-point steps to run (default 200).
 * @param params.tolerance The `change` below which the run stops (default 1e-4).
 * @param params.init The starting unmixing matrix in whitened coordinates ($r \times r$); left out, a standard normal
 *   one.
 * @returns The estimator: `fit({ x })` on an $n \times d$ matrix returns a `FastIcaModel`.
 *
 * @example A sine and a square wave are unmixed
 * const t = linspace(0, 8, 200)
 * const sources = stack([sin(mul(t, 2)), sign(sin(mul(t, 3)))], 1)
 * const mixing = tensor([[1, 0.5], [1, 2]])
 * const x = matmul(sources, mixing)
 * const model = fastIca().fit({ x }, { stream: stream(1) })
 * print('converged after', model.training.final.t, 'steps')
 * // Unmixing after mixing: one large entry in each row and column.
 * print('mixing then unmixing =', matmul(mixing, transpose(model.components)))
 */
export function fastIca(
  params: { components?: number; maxSteps?: number; tolerance?: number; init?: Tensor } = {},
): Estimator<Dataset<Tensor>, FastIcaModel> {
  const { maxSteps = 200, tolerance = 1e-4 } = params
  return {
    name: 'fast-ica',
    params: { ...params, maxSteps, tolerance },
    fit({ x }, options: FitOptions = {}) {
      const { n, d, v } = matrix(x, 'fastIca')
      const r = Math.min(params.components ?? d, d)
      const start = params.init ? { unmixing: params.init } : undefined
      const training = trace(fastIcaSteps(x, { components: r, tolerance }), start, maxSteps, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        record: { change: (s) => s.change },
      })
      const W = values(training.final.unmixing)
      const { mean, K, Z } = whitening(v, n, d, r)
      const components = matMul(W, K, r, r, d)
      // Unit-variance sources (scikit-learn's whiten='unit-variance'): divide each row by its source's std.
      const Y = matMul(W, Z, r, r, n)
      for (let i = 0; i < r; i++) {
        let m = 0
        for (let k = 0; k < n; k++) m += Y[i * n + k] / n
        let s2 = 0
        for (let k = 0; k < n; k++) s2 += (Y[i * n + k] - m) ** 2 / n
        const sd = Math.sqrt(s2)
        for (let j = 0; j < d; j++) components[i * d + j] /= sd
      }
      const mixing = values(pinv(mat(components, r, d)))
      return {
        kind: 'model',
        name: 'fast-ica',
        components: mat(components, r, d),
        mixing: mat(Float64Array.from(mixing), d, r),
        mean: vec(mean),
        training,
        transform: (q: Tensor) => {
          const { n: m, d: dq, v: qv } = matrix(q, 'fastIca.transform')
          if (dq !== d) throw new ShapeError('fastIca', `fastIca: fitted on ${d} features, given ${dq}`)
          const Xc = Float64Array.from(qv, (u, t) => u - mean[t % d])
          return mat(matMul(Xc, transpose(components, r, d), m, d, r), m, r)
        },
        inverseTransform: (s: Tensor) => {
          const { n: m, v: sv } = matrix(s, 'fastIca.inverseTransform')
          const out = matMul(sv, transpose(mixing, d, r), m, r, d)
          for (let i = 0; i < out.length; i++) out[i] += mean[i % d]
          return mat(out, m, d)
        },
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const MODULE = 'unsupervised/embedding/linear'

defineModel(
  {
    key: 'factorAnalysis',
    module: MODULE,
    name: 'Factor analysis',
    summary: 'A Gaussian with low-rank-plus-diagonal covariance WWᵀ + Ψ, fitted by EM; scores are posterior means.',
    task: 'embedding',
    capabilities: ['transform'],
    hyper: space({ latent: int(1, 20, { default: 2 }), maxSteps: int(1, 5000, { default: 1000 }) }),
    notes: ['factor-analysis'],
    cite: ['rubin1982', 'ghahramani1996'],
  },
  factorAnalysis,
)

defineModel(
  {
    key: 'probabilisticPca',
    module: MODULE,
    name: 'Probabilistic PCA',
    summary:
      'Factor analysis with isotropic noise σ²I: closed-form maximum likelihood from the eigendecomposition, or EM.',
    task: 'embedding',
    capabilities: ['transform'],
    hyper: space({ latent: int(1, 20, { default: 2 }), method: oneOf(['closed-form', 'em']) }),
    notes: ['probabilistic-principal-component-analysis'],
    cite: ['tipping1999'],
  },
  probabilisticPca,
)

defineModel(
  {
    key: 'fastIca',
    module: MODULE,
    name: 'FastICA',
    summary: 'Independent sources by whitening and the symmetric fixed-point iteration on a log-cosh contrast.',
    task: 'embedding',
    capabilities: ['transform'],
    hyper: space({ components: int(1, 50, { default: 2 }), maxSteps: int(1, 2000, { default: 200 }) }),
    notes: ['independent-component-analysis'],
    cite: ['hyvarinen1999', 'hyvarinen2000'],
  },
  fastIca,
)
