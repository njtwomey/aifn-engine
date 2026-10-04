/**
 * Gaussian mixture models fitted by expectation–maximisation (Dempster, Laird and Rubin, 1977; Bishop, 2006, "Pattern
 * Recognition and Machine Learning", §9.2), with full, diagonal or spherical covariances, as scikit-learn's
 * `GaussianMixture` (including its `reg_covar` added to every variance and its mean log-likelihood stopping rule).
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { MultivariateNormal } from 'aifn-compute/probability/distributions'
import {
  bernoulliPredictive,
  categoricalPredictive,
  type AnyUnivariate,
  type Dataset,
  type Decides,
  type Estimator,
  type FitOptions,
  type Fitted,
  type Predicts,
  type Scores,
  type Trained,
} from 'aifn-compute/learning/estimators'
import { cholesky } from 'aifn-compute/numerics/linalg'
import { normals, type Stream, child, uniform } from 'aifn-compute/foundation/random'
import { fromData, logsumexp, type Tensor } from 'aifn-compute/foundation/tensor'
import { softmax } from 'aifn-compute/numerics/special'
import { trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { kmeansPlusPlus } from 'aifn-compute/numerics/neighbours'
import { mat, matrix, nearest, values, vec } from './util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The shape of each component's covariance. */
export type CovarianceType = 'full' | 'diagonal' | 'spherical'

/** Mixture parameters: weights [k], means [k, d], covariances [k, d, d] (always stored full). */
export interface MixtureParameters {
  weights: Tensor
  means: Tensor
  covariances: Tensor
}

/** One EM state: parameters and the responsibilities and log-likelihood they give. */
export interface MixtureState extends MixtureParameters, Status {
  /** EM iterations done. */
  t: number
  /** Responsibilities r_ik = P(component k | xᵢ) at these parameters [n, k]. */
  responsibilities: Tensor
  /** Mean log-likelihood (1/n) Σᵢ log p(xᵢ) at these parameters. */
  logLikelihood: number
  /** Change in `logLikelihood` from the previous state (NaN at the start). */
  change: number
  converged: boolean
  /** Diagonal jitter needed to factor each covariance [k] (0 when positive definite). */
  jitter: Tensor
  diverged: boolean
}

/** Starting parameters, or none (seeded from the stream: k-means++ centres, then one M-step on the hard assignment). */
export interface MixtureInit {
  weights?: Tensor
  means?: Tensor
  covariances?: Tensor
}

/**
 * Per-component log-densities log N(xᵢ | μ_k, Σ_k) [n, k] (each component a `MultivariateNormal`, jitter allowed),
 * plus the jitter used to factor each Σ_k.
 */
function logDensities(v: Float64Array, n: number, d: number, k: number, means: Float64Array, covs: Float64Array) {
  const out = new Float64Array(n * k)
  const jitter = new Float64Array(k)
  const x = fromData(v, [n, d])
  for (let c = 0; c < k; c++) {
    const law = MultivariateNormal(
      fromData(means.slice(c * d, (c + 1) * d), [d]),
      { covariance: fromData(covs.slice(c * d * d, (c + 1) * d * d), [d, d]) },
      { jitter: 'auto' },
    )
    jitter[c] = law.jitter
    const lp = values(law.logProb(x) as Tensor)
    for (let i = 0; i < n; i++) out[i * k + c] = lp[i]
  }
  return { out, jitter }
}

/** E-step: responsibilities and the mean log-likelihood. */
function expectation(
  v: Float64Array,
  n: number,
  d: number,
  k: number,
  weights: Float64Array,
  means: Float64Array,
  covs: Float64Array,
) {
  const { out, jitter } = logDensities(v, n, d, k, means, covs)
  for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) out[i * k + c] += Math.log(weights[c])
  // log p(xᵢ) = logsumexp over components of the joint; the responsibilities are its softmax.
  const joint = fromData(out, [n, k])
  const resp = values(softmax(joint))
  const total = values(logsumexp(joint, 1)).reduce((a, b) => a + b, 0)
  return { resp, logLikelihood: total / n, jitter }
}

/** M-step from responsibilities (scikit-learn's estimators, with `reg` added to the variances). */
function maximisation(
  v: Float64Array,
  n: number,
  d: number,
  k: number,
  resp: Float64Array,
  type: CovarianceType,
  reg: number,
) {
  const nk = new Float64Array(k)
  for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) nk[c] += resp[i * k + c]
  for (let c = 0; c < k; c++) nk[c] += 10 * Number.EPSILON
  const means = new Float64Array(k * d)
  for (let i = 0; i < n; i++)
    for (let c = 0; c < k; c++) for (let j = 0; j < d; j++) means[c * d + j] += resp[i * k + c] * v[i * d + j]
  for (let c = 0; c < k; c++) for (let j = 0; j < d; j++) means[c * d + j] /= nk[c]
  const covs = new Float64Array(k * d * d)
  for (let c = 0; c < k; c++) {
    const C = covs.subarray(c * d * d, (c + 1) * d * d)
    for (let i = 0; i < n; i++) {
      const r = resp[i * k + c]
      for (let a = 0; a < d; a++) {
        const da = v[i * d + a] - means[c * d + a]
        for (let b = 0; b < d; b++) C[a * d + b] += r * da * (v[i * d + b] - means[c * d + b])
      }
    }
    for (let a = 0; a < d * d; a++) C[a] /= nk[c]
    if (type !== 'full') {
      let avg = 0
      for (let a = 0; a < d; a++) avg += C[a * d + a] / d
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) if (a !== b) C[a * d + b] = 0
      if (type === 'spherical') for (let a = 0; a < d; a++) C[a * d + a] = avg
    }
    for (let a = 0; a < d; a++) C[a * d + a] += reg
  }
  const weights = Float64Array.from(nk, (m) => m / n)
  return { weights, means, covs }
}

/**
 * EM for a Gaussian mixture on the rows of x [n, d] as a traceable algorithm. State t holds the parameters after t
 * M-steps and the responsibilities and mean log-likelihood at them; each step is an M-step followed by an E-step. It
 * has converged when the mean log-likelihood changes by less than `tolerance` (default 1e-3). Without starting
 * parameters, `init` seeds from the `init` stream (k-means++ centres, then one M-step on the hard assignment).
 */
export function gaussianMixtureSteps(
  x: Tensor,
  params: { k: number; covariance?: CovarianceType; regularisation?: number; tolerance?: number },
): Algorithm<MixtureInit, MixtureState> {
  const { n, d, v } = matrix(x, 'gaussianMixtureSteps')
  const { k, covariance = 'full', regularisation = 1e-6, tolerance: tol = 1e-3 } = params
  const state = (w: Float64Array, m: Float64Array, c: Float64Array, t: number, previous: number): MixtureState => {
    const e = expectation(v, n, d, k, w, m, c)
    const change = e.logLikelihood - previous
    return {
      weights: vec(w),
      means: mat(m, k, d),
      covariances: fromData(c, [k, d, d]),
      responsibilities: mat(e.resp, n, k),
      logLikelihood: e.logLikelihood,
      change,
      t,
      converged: Math.abs(change) < tol,
      jitter: vec(e.jitter),
      diverged: !Number.isFinite(e.logLikelihood),
    }
  }
  return {
    name: 'gaussian-mixture-em',
    init: ({ weights, means, covariances } = {}, s) => {
      if (weights && means && covariances) {
        return state(
          Float64Array.from(values(weights)),
          Float64Array.from(values(means)),
          Float64Array.from(values(covariances)),
          0,
          NaN,
        )
      }
      const centres = means ? values(means) : values(kmeansPlusPlus(s, x, k).centroids)
      const resp = new Float64Array(n * k)
      for (let i = 0; i < n; i++) resp[i * k + nearest(v, i, centres, k, d)[0]] = 1
      const p = maximisation(v, n, d, k, resp, covariance, regularisation)
      return state(
        weights ? Float64Array.from(values(weights)) : p.weights,
        means ? Float64Array.from(values(means)) : p.means,
        covariances ? Float64Array.from(values(covariances)) : p.covs,
        0,
        NaN,
      )
    },
    step: (s) => {
      const p = maximisation(v, n, d, k, values(s.responsibilities), covariance, regularisation)
      return state(p.weights, p.means, p.covs, s.t + 1, s.logLikelihood)
    },
  }
}

/** A fitted Gaussian mixture. */
export interface GaussianMixtureModel
  extends
    MixtureParameters,
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Trained<MixtureState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'gaussian-mixture'
  readonly covarianceType: CovarianceType
  readonly logLikelihood: number
  readonly converged: boolean
  /** EM iterations taken. */
  readonly steps: number
  /** Responsibilities P(component | x) [m, k]. */
  responsibilities(x: Tensor): Tensor
  /** log p(x) under the mixture [m]. */
  logDensity(x: Tensor): Tensor
  /** n draws from the mixture: rows [n, d] and their components [n]. */
  sampleMixture(s: Stream, n: number): { x: Tensor; components: Tensor }
}

/**
 * A Gaussian mixture fitted by EM (`gaussianMixtureSteps`), from given parameters or a k-means++ seeding. `forward`
 * and `score` give log p(x, component) [m, k]; `predictive` the responsibilities as a class law (Bernoulli for two
 * components, else Categorical); `decide` the most responsible component.
 */
export function gaussianMixture(params: {
  k: number
  covariance?: CovarianceType
  regularisation?: number
  tolerance?: number
  maxSteps?: number
  init?: MixtureInit
}): Estimator<Dataset<Tensor>, GaussianMixtureModel> {
  const { k, covariance = 'full', regularisation = 1e-6, tolerance = 1e-3, maxSteps = 100, init = {} } = params
  return {
    name: 'gaussian-mixture',
    params: { k, covariance, regularisation, tolerance, maxSteps },
    fit({ x }, options: FitOptions = {}) {
      const { d } = matrix(x, 'gaussianMixture')
      const training = trace(gaussianMixtureSteps(x, { k, covariance, regularisation, tolerance }), init, maxSteps, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          logLikelihood: (s) => s.logLikelihood,
          ...(options.trace?.record as Record<string, (s: MixtureState, t: number) => number> | undefined),
        },
      })
      const final = training.final
      const w = values(final.weights)
      const m = values(final.means)
      const c = values(final.covariances)
      const joint = (q: Tensor) => {
        const { n: rows, v, d: dq } = matrix(q, 'gaussianMixture')
        if (dq !== d) throw new ShapeError('gaussianMixture', `gaussianMixture: fitted on ${d} features, given ${dq}`)
        const { out } = logDensities(v, rows, d, k, m, c)
        for (let i = 0; i < rows; i++) for (let j = 0; j < k; j++) out[i * k + j] += Math.log(w[j])
        return { out, rows }
      }
      const resp = (q: Tensor) => {
        const { out, rows } = joint(q)
        const L = fromData(out, [rows, k])
        return { r: values(softmax(L)), logp: values(logsumexp(L, 1)), rows }
      }
      const forward = (q: Tensor) => {
        const { out, rows } = joint(q)
        return mat(out, rows, k)
      }
      return {
        kind: 'model',
        name: 'gaussian-mixture',
        covarianceType: covariance,
        weights: final.weights,
        means: final.means,
        covariances: final.covariances,
        logLikelihood: final.logLikelihood,
        converged: final.converged,
        steps: final.t,
        training,
        forward,
        score: forward,
        responsibilities: (q: Tensor) => {
          const { r, rows } = resp(q)
          return mat(r, rows, k)
        },
        logDensity: (q: Tensor) => vec(resp(q).logp),
        predictive: (q: Tensor) => {
          const { r, rows } = resp(q)
          if (k === 2) return bernoulliPredictive(vec(Float64Array.from({ length: rows }, (_, i) => r[2 * i + 1])))
          return categoricalPredictive(mat(r, rows, k))
        },
        decide: (q: Tensor) => {
          const { r, rows } = resp(q)
          const out = new Int32Array(rows)
          for (let i = 0; i < rows; i++) for (let j = 1; j < k; j++) if (r[i * k + j] > r[i * k + out[i]]) out[i] = j
          return fromData(out, [rows])
        },
        sampleMixture: (s: Stream, count: number) => {
          const comp = new Int32Array(count)
          const out = new Float64Array(count * d)
          const pick = child(s, 'components')
          const z = values(normals(child(s, 'normals'), [count, d]))
          const factors = Array.from(
            { length: k },
            (_, j) => cholesky(fromData(c.slice(j * d * d, (j + 1) * d * d), [d, d])).L.data as Float64Array,
          )
          for (let i = 0; i < count; i++) {
            let u = uniform(pick)
            let j = 0
            while (j < k - 1 && (u -= w[j]) >= 0) j++
            comp[i] = j
            const L = factors[j]
            for (let a = 0; a < d; a++) {
              let t = m[j * d + a]
              for (let b = 0; b <= a; b++) t += L[a * d + b] * z[i * d + b]
              out[i * d + a] = t
            }
          }
          return { x: mat(out, count, d), components: fromData(comp, [count]) }
        },
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'gaussianMixture',
    module: 'unsupervised/clustering',
    name: 'Gaussian mixture',
    summary: 'A mixture of Gaussians fitted by EM; soft assignments are its predictive.',
    task: 'clustering',
    capabilities: ['forward', 'decide', 'predictive', 'score'],
    hyper: space({
      k: int(1, 20, { default: 3 }),
      covariance: oneOf(['full', 'diagonal', 'spherical']),
      regularisation: real(0, 1, { default: 1e-6 }),
      tolerance: real(1e-10, 1, { default: 1e-3, scale: 'log' }),
      maxSteps: int(1, 1000, { default: 100 }),
    }),
    notes: ['gaussian-mixture-model'],
    cite: ['dempster1977'],
  },
  gaussianMixture,
)
