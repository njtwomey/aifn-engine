/**
 * Linear latent-variable models: x = Wz + μ + ε with z ~ N(0, I_q), so x ~ N(μ, WWᵀ + Ψ).
 *
 * - `latentGaussianSteps`: EM for W and the noise (Rubin and Thayer, 1982; Ghahramani and Hinton, 1996), with diagonal
 *   noise Ψ (factor analysis) or isotropic noise σ²I (probabilistic PCA; Tipping and Bishop, 1999), as a step-through
 *   algorithm on the sample covariance; the log-likelihood never decreases.
 * - `factorAnalysis`: factor analysis by that EM.
 * - `probabilisticPca`: PPCA in closed form (W = U_q(Λ_q − σ²I)^{1/2}, σ² the mean discarded eigenvalue) or by EM.
 * - `fastIcaSteps`, `fastIca`: independent component analysis by FastICA (Hyvärinen, 1999; Hyvärinen and Oja, 2000):
 *   whitening, then the symmetric fixed-point iteration W ← (WWᵀ)^{−1/2} (E[g(Wx)xᵀ] − diag E[g′(Wx)] W) with the
 *   log-cosh contrast g = tanh, as scikit-learn's `FastICA(algorithm='parallel', whiten='unit-variance')`.
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

/** Column means and the sample covariance S = (1/n) Σ (x − x̄)(x − x̄)ᵀ (the maximum-likelihood one). */
function moments(v: Float64Array, n: number, d: number) {
  const mean = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += v[i * d + j] / n
  const Xc = Float64Array.from(v, (u, t) => u - mean[t % d])
  const S = matMul(transpose(Xc, n, d), Xc, d, n, d)
  for (let i = 0; i < S.length; i++) S[i] /= n
  return { mean, S }
}

const inv = (A: Float64Array, q: number) => values(inverse(mat(A, q, q)))

/** The model covariance C = WWᵀ + diag(ψ) [d, d]. */
function modelCovariance(W: Float64Array, psi: Float64Array, d: number, q: number): Float64Array {
  const C = matMul(W, transpose(W, d, q), d, q, d)
  for (let i = 0; i < d; i++) C[i * d + i] += psi[i]
  return C
}

/** Average log-likelihood per row, −½ (d log 2π + log|C| + tr(C⁻¹S)). */
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
  t: number
  /** Loadings W [d, q]. */
  loadings: Tensor
  /** Noise variances ψ [d] (all equal for isotropic noise). */
  noise: Tensor
  /** Average log-likelihood per row. */
  logLikelihood: number
  converged: boolean
}

/**
 * EM for x = Wz + μ + ε on the rows of x [n, d] with `latent` = q factors. With M = I + WᵀΨ⁻¹W and β = M⁻¹WᵀΨ⁻¹ (so
 * E[z | x] = β(x − μ)), one step sets W ← Sβᵀ(M⁻¹ + βSβᵀ)⁻¹ and the noise to diag(S − WβS) (diagonal) or its mean
 * (isotropic). Converged when the log-likelihood rises by less than `tolerance` (default 1e-8). The initial loadings
 * are Gaussian (scale √(tr S / d) / q) from the `init` stream unless given; the initial noise is diag(S).
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
  readonly kind: 'model'
  readonly name: 'factor-analysis' | 'probabilistic-pca'
  /** Loadings W [d, q]. */
  readonly loadings: Tensor
  /** Noise variances ψ [d]. */
  readonly noise: Tensor
  readonly mean: Tensor
  /** Average log-likelihood per training row. */
  readonly logLikelihood: number
  /** The model covariance WWᵀ + Ψ [d, d]. */
  readonly covariance: Tensor
  /** Posterior means E[z | x] = (I + WᵀΨ⁻¹W)⁻¹WᵀΨ⁻¹(x − μ) [m, q]. */
  transform(x: Tensor): Tensor
  /** The average log-likelihood of new rows under the fitted Gaussian. */
  averageLogLikelihood(x: Tensor): number
}

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

/** A fitted factor analysis, with its EM trace. */
export interface FactorAnalysisModel extends LatentGaussianModel, Trained<LatentGaussianState> {
  readonly name: 'factor-analysis'
}

/** Factor analysis with `latent` factors (default 2) by EM (`latentGaussianSteps`, diagonal noise). */
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
  readonly name: 'probabilistic-pca'
  /** The shared noise variance σ². */
  readonly noiseVariance: number
}

/**
 * Probabilistic PCA with `latent` dimensions (default 2): the maximum-likelihood solution in closed form (default;
 * W = U_q(Λ_q − σ²I)^{1/2} from the eigendecomposition of the sample covariance, σ² the mean of the d − q discarded
 * eigenvalues; loadings signed so each column's largest-magnitude entry is positive), or by EM (isotropic noise).
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

/** (WWᵀ)^{−1/2} W for W [r, r]: the symmetric decorrelation that keeps the rows orthonormal. */
function symmetricDecorrelation(W: Float64Array, r: number): Float64Array {
  const P = values(symmetricInverseSqrt(mat(matMul(W, transpose(W, r, r), r, r, r), r, r), { floor: Number.MIN_VALUE }))
  return matMul(P, W, r, r, r)
}

/** The whitening of FastICA: mean, and K [r, d] with K(x − x̄) of identity covariance (as scikit-learn's SVD solver). */
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
  t: number
  /** The unmixing matrix in whitened coordinates W [r, r] (orthonormal rows). */
  unmixing: Tensor
  /** max_i | |⟨w_i, w_i^old⟩| − 1 | ∈ [0, 1]: 0 when every row has stopped turning (1 before the first step). */
  change: number
  converged: boolean
}

/**
 * FastICA's symmetric fixed-point iteration on the rows of x [n, d], recovering `components` sources (default d): x is
 * whitened to z, then each step sets W ← (WWᵀ)^{−1/2}(E[tanh(Wz) zᵀ] − diag(E[1 − tanh²(Wz)]) W). Converged when
 * `change` < `tolerance` (default 1e-4). The initial W is standard normal from the `init` stream, or given.
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

/** A fitted ICA. */
export interface FastIcaModel extends Transforms<Tensor, Tensor>, Trained<FastIcaState> {
  readonly kind: 'model'
  readonly name: 'fast-ica'
  /** The unmixing matrix [r, d]: sources s = A⁺… = components (x − mean), each source of unit variance. */
  readonly components: Tensor
  /** The mixing matrix [d, r], the pseudo-inverse of `components`. */
  readonly mixing: Tensor
  readonly mean: Tensor
  /** Estimated sources of new rows [m, r]. */
  transform(x: Tensor): Tensor
  /** Back from sources to the input space [m, d]. */
  inverseTransform(s: Tensor): Tensor
}

/**
 * FastICA with `components` sources (default all), by `fastIcaSteps` to convergence or `maxSteps` (default 200), from
 * the unmixing matrix `init` [r, r] or a standard normal one.
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
