/**
 * A Gaussian mixture as the data distribution, for which every noised marginal is known in closed form: if
 * x₀ ~ Σₖ πₖ N(μₖ, Σₖ) and x = m·x₀ + s·ε, then x ~ Σₖ πₖ N(m·μₖ, m²Σₖ + s²I). So its exact score and noise predictor
 * drive every sampler without training, and samples can be checked against the true marginals (Bishop, 2006, PRML
 * §2.3.3 for the linear-Gaussian marginal; Song et al., 2021, for the score).
 */

import { cholesky, inverse, logDet } from 'aifn-compute/numerics/linalg'
import { categorical, normals, type Stream, child } from 'aifn-compute/foundation/random'
import { fromData, fromRows, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import type { NoisePredictor } from './predictor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A Gaussian mixture in d dimensions. */
export type GaussianMixture = {
  readonly weights: readonly number[]
  readonly means: readonly (readonly number[])[]
  /** Full covariance matrices, d × d each. */
  readonly covariances: readonly (readonly (readonly number[])[])[]
  readonly dimension: number
}

/**
 * A Gaussian mixture from weights (normalised here), means and either covariance matrices or standard deviations
 * (a number per component for isotropic components, or a vector for diagonal ones).
 */
export function gaussianMixtureData(
  weights: readonly number[],
  means: readonly (readonly number[])[],
  spread: readonly (number | readonly number[] | readonly (readonly number[])[])[],
): GaussianMixture {
  const total = weights.reduce((a, b) => a + b, 0)
  if (!(total > 0))
    throw new DomainError('gaussianMixtureData', 'gaussianMixtureData: weights must have a positive sum')
  const d = means[0].length
  const covariances = spread.map((sp) => {
    if (typeof sp === 'number')
      return Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? sp * sp : 0)))
    if (typeof sp[0] === 'number') {
      const v = sp as readonly number[]
      return Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? v[i] * v[i] : 0)))
    }
    return (sp as readonly (readonly number[])[]).map((r) => [...r])
  })
  return { weights: weights.map((w) => w / total), means: means.map((m) => [...m]), covariances, dimension: d }
}

/** n draws from the mixture, shape [n, d]. */
export function sampleMixture(s: Stream, mixture: GaussianMixture, n: number): Tensor {
  const d = mixture.dimension
  const comps = toFlat(categorical(child(s, 'component'), mixture.weights, { shape: [n] }) as Tensor)
  const z = toFlat(normals(child(s, 'noise'), [n, d]))
  const chol = mixture.covariances.map((c) => {
    const f = cholesky(fromRows(c.map((r) => [...r])), { jitter: false })
    if (f.failed) throw new DomainError('sampleMixture', 'sampleMixture: a covariance is not positive definite')
    return toRows(f.L)
  })
  const out = new Float64Array(n * d)
  for (let i = 0; i < n; i++) {
    const k = comps[i]
    for (let a = 0; a < d; a++) {
      let v = mixture.means[k][a]
      for (let b = 0; b <= a; b++) v += chol[k][a][b] * z[i * d + b]
      out[i * d + a] = v
    }
  }
  return fromData(out, [n, d])
}

/** The noised components: means m·μₖ, precisions (m²Σₖ + s²I)⁻¹ and log normalisers. */
function noisedComponents(mixture: GaussianMixture, m: number, s: number) {
  const d = mixture.dimension
  return mixture.covariances.map((cov, k) => {
    const c = cov.map((row, i) => row.map((v, j) => m * m * v + (i === j ? s * s : 0)))
    const C = fromRows(c)
    return {
      mean: mixture.means[k].map((v) => m * v),
      precision: toRows(inverse(C)),
      logNorm: Math.log(mixture.weights[k]) - 0.5 * (d * Math.log(2 * Math.PI) + logDet(C)),
    }
  })
}

/** Per-point log densities of each noised component and their gradients, reduced to log p and ∇ log p. */
function evaluate(mixture: GaussianMixture, x: Tensor, m: number, s: number, withScore: boolean) {
  const d = mixture.dimension
  if (x.shape.length !== 2 || x.shape[1] !== d) throw new ShapeError('mixture', `mixture: points need shape [n, ${d}]`)
  const n = x.shape[0]
  const xs = toFlat(x)
  const comps = noisedComponents(mixture, m, s)
  const K = comps.length
  const logp = new Float64Array(n)
  const score = new Float64Array(withScore ? n * d : 0)
  const lk = new Float64Array(K)
  const grads = withScore ? Array.from({ length: K }, () => new Float64Array(d)) : []
  const diff = new Float64Array(d)
  for (let i = 0; i < n; i++) {
    let top = -Infinity
    for (let k = 0; k < K; k++) {
      const { mean, precision, logNorm } = comps[k]
      for (let a = 0; a < d; a++) diff[a] = xs[i * d + a] - mean[a]
      let quad = 0
      for (let a = 0; a < d; a++) {
        let pa = 0
        for (let b = 0; b < d; b++) pa += precision[a][b] * diff[b]
        quad += diff[a] * pa
        if (withScore) grads[k][a] = -pa
      }
      lk[k] = logNorm - 0.5 * quad
      if (lk[k] > top) top = lk[k]
    }
    let total = 0
    for (let k = 0; k < K; k++) total += Math.exp(lk[k] - top)
    logp[i] = top + Math.log(total)
    if (withScore) {
      // The mixture's score is the responsibility-weighted average of the components' scores.
      for (let k = 0; k < K; k++) {
        const r = Math.exp(lk[k] - logp[i])
        for (let a = 0; a < d; a++) score[i * d + a] += r * grads[k][a]
      }
    }
  }
  return { logp, score }
}

/** log p of the noised mixture x = m·x₀ + s·ε at points x [n, d] (m = 1, s = 0 gives the data density). */
export function mixtureLogDensity(mixture: GaussianMixture, x: Tensor, meanScale = 1, std = 0): Tensor {
  return fromData(evaluate(mixture, x, meanScale, std, false).logp, [x.shape[0]])
}

/** The exact score ∇ₓ log p of the noised mixture at points x [n, d]. */
export function mixtureScore(mixture: GaussianMixture, x: Tensor, meanScale: number, std: number): Tensor {
  return fromData(evaluate(mixture, x, meanScale, std, true).score, [x.shape[0], mixture.dimension])
}

/**
 * The exact noise predictor of the mixture: ε̂(x, ᾱ) = −√(1 − ᾱ)·∇ₓ log p_ᾱ(x), the posterior mean of the noise, which
 * is what a perfectly trained network would output.
 */
export function mixtureNoisePredictor(mixture: GaussianMixture): NoisePredictor {
  return (x, alphaBar) => {
    const s = Math.sqrt(1 - alphaBar)
    const score = evaluate(mixture, x, Math.sqrt(alphaBar), s, true).score
    for (let i = 0; i < score.length; i++) score[i] *= -s
    return fromData(score, [x.shape[0], mixture.dimension])
  }
}

/** The mixture's mean and covariance (for checking samples). */
export function mixtureMoments(mixture: GaussianMixture): { mean: number[]; covariance: number[][] } {
  const d = mixture.dimension
  const mean = Array.from({ length: d }, (_, a) =>
    mixture.weights.reduce((acc, w, k) => acc + w * mixture.means[k][a], 0),
  )
  const covariance = Array.from({ length: d }, (_, a) =>
    Array.from({ length: d }, (_, b) =>
      mixture.weights.reduce(
        (acc, w, k) =>
          acc + w * (mixture.covariances[k][a][b] + (mixture.means[k][a] - mean[a]) * (mixture.means[k][b] - mean[b])),
        0,
      ),
    ),
  )
  return { mean, covariance }
}
