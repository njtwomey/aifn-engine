/**
 * Noise predictors and scores. Every model of the data is used through one interface, a noise predictor
 * $\hat\epsilonvec(\xvec, \bar\alpha)$: given a point
 * $\xvec = \sqrt{\bar\alpha}\,\xvec_0 + \sqrt{1 - \bar\alpha}\,\epsilonvec$ of the variance-preserving process at
 * signal level $\bar\alpha$, it predicts $\epsilonvec$ (Ho et al., 2020, §3.2). The score of the noised density is then
 * $\nabla_{\xvec} \log p(\xvec) = -\hat\epsilonvec / \sqrt{1 - \bar\alpha}$ (Vincent, 2011; Song et al., 2021, eq. 7
 * and appendix), and the same predictor serves any marginal $\xvec = m\,\xvec_0 + s\,\epsilonvec$ (VE, sub-VP) after
 * scaling $\xvec$ by $1/\sqrt{m^2 + s^2}$, which maps it onto the VP process at $\bar\alpha = m^2/(m^2 + s^2)$.
 */

import { div, mul, sub, type Tensor } from 'aifn-compute/foundation/tensor'

/**
 * Predicts the noise $\epsilonvec$ in points `x` ($[n, d]$) of the VP process at signal level `alphaBar`
 * $\bar\alpha \in (0, 1]$; returns $[n, d]$.
 */
export type NoisePredictor = (x: Tensor, alphaBar: number) => Tensor

/**
 * $\hat\epsilonvec$ for points $\xvec = m\,\xvec_0 + s\,\epsilonvec$ of any Gaussian marginal $(m, s)$, through the VP
 * predictor: the points are scaled by $1/r$ with $r = \sqrt{m^2 + s^2}$ and the predictor asked at
 * $\bar\alpha = (m/r)^2$. Scaling leaves the noise unchanged, so its prediction is returned as it is.
 *
 * @param predictor The VP noise predictor.
 * @param x The points $\xvec$, $[n, d]$.
 * @param meanScale The marginal's mean scale $m$.
 * @param std The marginal's standard deviation $s$.
 * @returns $\hat\epsilonvec$, $[n, d]$.
 *
 * @example Data $\Gauss(\zeros, \Imat)$, whose exact prediction is $s\,\xvec/(m^2 + s^2)$
 * const predictor = mixtureNoisePredictor(gaussianMixtureData([1], [[0, 0]], [1]))
 * print('VP, (m, s) = (0.6, 0.8):', predictNoise(predictor, tensor([[1, 2]]), 0.6, 0.8))
 * print('VE, (m, s) = (1, 3):', predictNoise(predictor, tensor([[1, 2]]), 1, 3))
 */
export function predictNoise(predictor: NoisePredictor, x: Tensor, meanScale: number, std: number): Tensor {
  const r = Math.hypot(meanScale, std)
  return predictor(r === 1 ? x : mul(x, 1 / r), (meanScale / r) ** 2)
}

/**
 * The score $\nabla_{\xvec} \log p(\xvec) = -\hat\epsilonvec / s$ of the marginal
 * $\xvec = m\,\xvec_0 + s\,\epsilonvec$, from a noise predictor (through `predictNoise`). Infinite at $s = 0$.
 *
 * @param predictor The VP noise predictor.
 * @param x The points $\xvec$, $[n, d]$.
 * @param meanScale The marginal's mean scale $m$.
 * @param std The marginal's standard deviation $s > 0$.
 * @returns The score at each point, $[n, d]$.
 *
 * @example The score from the exact predictor matches the exact score
 * const unit = gaussianMixtureData([1], [[0, 0]], [1])
 * const x = tensor([[1, 2]])
 * print('from the predictor:', scoreFromPredictor(mixtureNoisePredictor(unit), x, 1, 3))
 * print('exact, -x / 10:', mixtureScore(unit, x, 1, 3))
 */
export function scoreFromPredictor(predictor: NoisePredictor, x: Tensor, meanScale: number, std: number): Tensor {
  return mul(predictNoise(predictor, x, meanScale, std), -1 / std)
}

/**
 * The predicted clean point $\hat\xvec_0 = (\xvec - s\,\hat\epsilonvec)/m$ (Tweedie's formula; Efron, 2011): the
 * posterior mean $\expect[\xvec_0 \mid \xvec]$ when $\hat\epsilonvec$ is the exact one.
 *
 * @param noise The predicted noise $\hat\epsilonvec$, of the shape of `x`.
 * @param x The noised points $\xvec$.
 * @param meanScale The marginal's mean scale $m > 0$.
 * @param std The marginal's standard deviation $s$.
 * @returns $\hat\xvec_0$, of the shape of `x`.
 *
 * @example For data $\Gauss(\zeros, \Imat)$ the posterior mean is $m\,\xvec/(m^2 + s^2) = 0.6\,\xvec$
 * const unit = gaussianMixtureData([1], [[0, 0]], [1])
 * const x = tensor([[1, 2]])
 * const noise = predictNoise(mixtureNoisePredictor(unit), x, 0.6, 0.8)
 * print('predicted clean point:', predictClean(noise, x, 0.6, 0.8))
 */
export function predictClean(noise: Tensor, x: Tensor, meanScale: number, std: number): Tensor {
  return div(sub(x, mul(noise, std)), meanScale)
}
