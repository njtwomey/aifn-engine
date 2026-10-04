/**
 * Noise predictors and scores. Every model of the data is used through one interface, a noise predictor
 * ε̂(x, ᾱ): given a point x = √ᾱ·x₀ + √(1 − ᾱ)·ε of the variance-preserving process at signal level ᾱ, it predicts ε
 * (Ho et al., 2020, §3.2). The score of the noised density is then ∇ₓ log p(x) = −ε̂/√(1 − ᾱ) (Vincent, 2011; Song et
 * al., 2021, eq. 7 and appendix), and the same predictor serves any marginal x = m·x₀ + s·ε (VE, sub-VP) after scaling
 * x by 1/√(m² + s²), which maps it onto the VP process at ᾱ = m²/(m² + s²).
 */

import { div, mul, sub, type Tensor } from 'aifn-compute/foundation/tensor'

/** Predicts the noise ε in points x (shape [n, d]) of the VP process at signal level ᾱ ∈ (0, 1]; returns [n, d]. */
export type NoisePredictor = (x: Tensor, alphaBar: number) => Tensor

/** ε̂ for points x = m·x₀ + s·ε of any Gaussian marginal (m, s), through the VP predictor. */
export function predictNoise(predictor: NoisePredictor, x: Tensor, meanScale: number, std: number): Tensor {
  const r = Math.hypot(meanScale, std)
  return predictor(r === 1 ? x : mul(x, 1 / r), (meanScale / r) ** 2)
}

/** The score ∇ₓ log p(x) = −ε̂/s of the marginal x = m·x₀ + s·ε. */
export function scoreFromPredictor(predictor: NoisePredictor, x: Tensor, meanScale: number, std: number): Tensor {
  return mul(predictNoise(predictor, x, meanScale, std), -1 / std)
}

/** The predicted clean point x̂₀ = (x − s·ε̂)/m (Tweedie's formula; Efron, 2011). */
export function predictClean(noise: Tensor, x: Tensor, meanScale: number, std: number): Tensor {
  return div(sub(x, mul(noise, std)), meanScale)
}
