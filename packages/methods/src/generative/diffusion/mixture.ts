/**
 * A Gaussian mixture as the data distribution, for which every noised marginal is known in closed form: if
 * $\xvec_0 \sim \sum_k \pi_k \Gauss(\muvec_k, \Sigmamat_k)$ and $\xvec = m\,\xvec_0 + s\,\epsilonvec$, then
 * $\xvec \sim \sum_k \pi_k \Gauss(m\muvec_k, m^2\Sigmamat_k + s^2\Imat)$. So its exact score and noise predictor drive
 * every sampler without training, and samples can be checked against the true marginals (Bishop, 2006, PRML §2.3.3
 * for the linear-Gaussian marginal; Song et al., 2021, for the score). Points are $[n, d]$ tensors; a mixture's
 * parameters are plain arrays.
 */

import { cholesky, inverse, logDet } from 'aifn-compute/numerics/linalg'
import { categorical, normals, type Stream, child } from 'aifn-compute/foundation/random'
import { fromData, fromRows, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import type { NoisePredictor } from './predictor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A Gaussian mixture in $d$ dimensions, of $K$ components. */
export type GaussianMixture = {
  /** The weights $\pi_k$, summing to 1. */
  readonly weights: readonly number[]
  /** The means $\muvec_k$, $d$ values each. */
  readonly means: readonly (readonly number[])[]
  /** Full covariance matrices $\Sigmamat_k$, $d \times d$ each, as rows. */
  readonly covariances: readonly (readonly (readonly number[])[])[]
  /** The dimension $d$. */
  readonly dimension: number
}

/**
 * A Gaussian mixture from weights (normalised here), means and either covariance matrices or standard deviations
 * (a number per component for isotropic components, or a vector for diagonal ones). The dimension $d$ is the length of
 * the first mean. Throws `DomainError` when the weights do not have a positive sum; nothing else is checked here
 * (`sampleMixture` checks that the covariances are positive definite).
 *
 * @param weights The component weights $\pi_k$, non-negative and in any scale.
 * @param means The component means $\muvec_k$, $d$ values each.
 * @param spread Each component's spread: a number $\sigma$ for $\sigma^2\Imat$, a vector $\sigmavec$ for
 *   $\diag(\sigmavec^2)$, or a $d \times d$ covariance matrix as rows (copied).
 * @returns The mixture.
 *
 * @example An isotropic and a diagonal component, weighted 1 : 3
 * const mixture = gaussianMixtureData([1, 3], [[-2, 0], [2, 0]], [0.5, [1, 0.25]])
 * print('weights', mixture.weights)
 * print('covariances', mixture.covariances)
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

/**
 * $n$ draws from the mixture: a component by its weight, then a Gaussian draw through the component covariance's
 * Cholesky factor. Throws `DomainError` when a covariance is not positive definite.
 *
 * @param s The stream; its children `'component'` and `'noise'` give the components and the Gaussian draws.
 * @param mixture The mixture.
 * @param n The number of draws.
 * @returns The draws, $[n, d]$.
 *
 * @example Draws from two modes at $\pm 2$, and their moments beside the exact ones
 * const mixture = gaussianMixtureData([1, 1], [[-2], [2]], [0.5, 0.5])
 * const x = sampleMixture(stream(1), mixture, 4000)
 * print('first draws', slice(x, [0, 5]))
 * print('sample mean', mean(x), ' variance', variance(x))
 * print('exact', mixtureMoments(mixture))
 */
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

/**
 * The noised components: means $m\muvec_k$, precisions $(m^2\Sigmamat_k + s^2\Imat)^{-1}$ and log normalisers
 * $\log \pi_k - \frac{1}{2}(d \log 2\pi + \log\det(m^2\Sigmamat_k + s^2\Imat))$.
 *
 * @param mixture The mixture.
 * @param m The mean scale $m$.
 * @param s The noise standard deviation $s$.
 * @returns One record per component: `mean` ($d$ values), `precision` ($d \times d$, as rows) and `logNorm`.
 */
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

/**
 * Per-point log densities of each noised component and their gradients, reduced (with log-sum-exp) to $\log p$ and,
 * when asked, to $\nabla \log p$, the responsibility-weighted average of the components' scores. Throws `ShapeError`
 * when the points are not $[n, d]$.
 *
 * @param mixture The mixture.
 * @param x The points, $[n, d]$.
 * @param m The mean scale $m$ of the noised marginal.
 * @param s The noise standard deviation $s$ of the noised marginal.
 * @param withScore Whether to compute the score as well.
 * @returns `logp`, $n$ values, and `score`, row-major $n \times d$ values (empty without `withScore`).
 */
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

/**
 * $\log p(\xvec)$ of the noised mixture $\xvec = m\,\xvec_0 + s\,\epsilonvec$ at points ($m = 1$, $s = 0$, the
 * defaults, give the data density). Throws `ShapeError` when the points are not $[n, d]$.
 *
 * @param mixture The mixture.
 * @param x The points, $[n, d]$.
 * @param meanScale The mean scale $m$.
 * @param std The noise standard deviation $s$.
 * @returns The log densities, $[n]$.
 *
 * @example A standard normal, then the same noised to variance 2
 * const unit = gaussianMixtureData([1], [[0]], [1])
 * print('log p at 0, 1, 2:', mixtureLogDensity(unit, tensor([[0], [1], [2]])))
 * print('noised by s = 1:', mixtureLogDensity(unit, tensor([[0], [1], [2]]), 1, 1))
 */
export function mixtureLogDensity(mixture: GaussianMixture, x: Tensor, meanScale = 1, std = 0): Tensor {
  return fromData(evaluate(mixture, x, meanScale, std, false).logp, [x.shape[0]])
}

/**
 * The exact score $\nabla_{\xvec} \log p(\xvec)$ of the noised mixture $\xvec = m\,\xvec_0 + s\,\epsilonvec$ at
 * points. Throws `ShapeError` when the points are not $[n, d]$.
 *
 * @param mixture The mixture.
 * @param x The points, $[n, d]$.
 * @param meanScale The mean scale $m$.
 * @param std The noise standard deviation $s$.
 * @returns The score at each point, $[n, d]$.
 *
 * @example The score points to the modes, and noise smooths it
 * const mixture = gaussianMixtureData([1, 1], [[-2], [2]], [0.5, 0.5])
 * const x = tensor([[-3], [-2], [0], [2], [3]])
 * print('clean:', mixtureScore(mixture, x, 1, 0))
 * print('noised, m = 0.6, s = 0.8:', mixtureScore(mixture, x, 0.6, 0.8))
 */
export function mixtureScore(mixture: GaussianMixture, x: Tensor, meanScale: number, std: number): Tensor {
  return fromData(evaluate(mixture, x, meanScale, std, true).score, [x.shape[0], mixture.dimension])
}

/**
 * The exact noise predictor of the mixture:
 * $\hat\epsilonvec(\xvec, \bar\alpha) = -\sqrt{1 - \bar\alpha}\,\nabla_{\xvec} \log p_{\bar\alpha}(\xvec)$, the
 * posterior mean of the noise, which is what a perfectly trained network would output.
 *
 * @param mixture The mixture.
 * @returns The predictor, which throws `ShapeError` for points that are not $[n, d]$.
 *
 * @example Near the data the noise points away from the nearer mode; at high noise, away from the mean
 * const predictor = mixtureNoisePredictor(gaussianMixtureData([1, 1], [[-2], [2]], [0.5, 0.5]))
 * const x = tensor([[-1.5], [0], [1.5]])
 * print('noise at alpha bar = 0.9:', predictor(x, 0.9))
 * print('noise at alpha bar = 0.1:', predictor(x, 0.1))
 */
export function mixtureNoisePredictor(mixture: GaussianMixture): NoisePredictor {
  return (x, alphaBar) => {
    const s = Math.sqrt(1 - alphaBar)
    const score = evaluate(mixture, x, Math.sqrt(alphaBar), s, true).score
    for (let i = 0; i < score.length; i++) score[i] *= -s
    return fromData(score, [x.shape[0], mixture.dimension])
  }
}

/**
 * The mixture's mean $\sum_k \pi_k \muvec_k$ and covariance
 * $\sum_k \pi_k (\Sigmamat_k + (\muvec_k - \muvec)(\muvec_k - \muvec)^\top)$, for checking samples.
 *
 * @param mixture The mixture.
 * @returns `mean`, $d$ values, and `covariance`, $d \times d$ as rows.
 *
 * @example Two modes at $\pm 2$ along the first axis
 * print(mixtureMoments(gaussianMixtureData([1, 1], [[-2, 0], [2, 0]], [0.5, 0.5])))
 */
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
