/**
 * What a figure shows of a GAN on 2-d toy data with a known density: the data's log density on a grid, the coverage
 * of the data's modes by generated points, and the optimal discriminator D*(x) = p_data(x)/(p_data(x) + p_g(x))
 * (Goodfellow et al., 2014, Proposition 1) with p_g estimated by a Gaussian KDE of generated points.
 *
 * The known density is any labelled mixture p(x) = Σⱼ πⱼ p(x | j) with a log density per mode: the `model` of a
 * classification truth of `aifn-methods/data` (a ring or grid of Gaussians, a pinwheel), whose classes are the modes.
 */

import { multivariateKde } from 'aifn-compute/probability/stats'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { mixtureLogDensityOf, modeOf, type LabelledDensity } from '../densities'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Mode coverage of a set of generated points. */
export type ModeCoverage = {
  /** Modes holding at least `minShare` of their expected count of high-quality points. */
  hit: number
  modes: number
  /** The share of generated points that are high quality: log p_data(x) at or above `threshold`. */
  quality: number
  /** High-quality points per mode. */
  perMode: Float64Array
  /** The log-density threshold of quality. */
  threshold: number
}

/** Options of `modeCoverage`. */
export type ModeCoverageOptions = {
  /**
   * The quality threshold on log p_data: a number, or `{ quantile: q }` for the q-quantile of log p_data over the
   * reference points `real` (default q = 0.01, so 99% of real points count as high quality).
   */
  threshold?: number | { quantile: number }
  /** Real points used for the quantile threshold ([n, d]). */
  real?: Tensor
  /** A mode is hit when it holds at least this share of πⱼ·(number of points) high-quality points. Default 0.2. */
  minShare?: number
}

/**
 * Mode coverage (after Srivastava et al., 2017, and Metz et al., 2017, who count modes with high-quality samples near
 * them): each generated point is high quality when its log density under the data is above a threshold, and is
 * assigned to its most probable mode; a mode is hit when it holds at least `minShare` of the high-quality points it
 * would hold under the data.
 */
export function modeCoverage(model: LabelledDensity, samples: Tensor, options: ModeCoverageOptions = {}): ModeCoverage {
  const { minShare = 0.2 } = options
  const spec = options.threshold ?? { quantile: 0.01 }
  let threshold: number
  if (typeof spec === 'number') threshold = spec
  else {
    if (!options.real) throw new DomainError('modeCoverage', 'modeCoverage: a quantile threshold needs the real points')
    const ref = Array.from(mixtureLogDensityOf(model, options.real)).sort((a, b) => a - b)
    threshold = ref[Math.min(ref.length - 1, Math.max(0, Math.floor(spec.quantile * ref.length)))]
  }
  const logp = mixtureLogDensityOf(model, samples)
  const modes = modeOf(model, samples)
  const k = model.classes
  const perMode = new Float64Array(k)
  let good = 0
  for (let i = 0; i < logp.length; i++)
    if (logp[i] >= threshold && modes[i] >= 0) {
      perMode[modes[i]]++
      good++
    }
  const n = logp.length
  let hit = 0
  for (let j = 0; j < k; j++) if (perMode[j] > 0 && perMode[j] >= minShare * model.priors[j] * n) hit++
  return { hit, modes: k, quality: n ? good / n : NaN, perMode, threshold }
}

/** The log density below which D* is left undefined (NaN): neither distribution puts mass there. */
const NEGLIGIBLE = -8

/**
 * The optimal discriminator D*(x) = p_data(x)/(p_data(x) + p_g(x)) at the query points, given log p_data there and
 * generated points: p_g by `multivariateKde` with its bandwidth chosen by leave-one-out likelihood (Scott's rule
 * oversmooths separated narrow modes, which would push D* towards 1 on the data). Computed as σ(log p_data − log p_g).
 * Also returns log p_g. When the generated points lie on a line or a point (a fully collapsed generator), p_g has no
 * density and both are NaN. D* is NaN where both densities are below e⁻⁸.
 */
export function optimalDiscriminator(
  dataLogDensity: ArrayLike<number>,
  samples: Tensor,
  at: Tensor,
): { value: Float64Array; generatorLogDensity: Float64Array } {
  let logPg: Float64Array
  try {
    logPg = Float64Array.from(toFlat(multivariateKde(samples, at, { bandwidth: 'cross-validation' }).logDensity))
  } catch {
    const none = new Float64Array(at.shape[0]).fill(NaN)
    return { value: none, generatorLogDensity: none }
  }
  const value = Float64Array.from(logPg, (lg, i) => {
    const ld = dataLogDensity[i]
    // Where neither density is above e⁻⁸ the ratio is a quotient of two tails: undefined for a figure.
    if (!(Math.max(ld, lg) > NEGLIGIBLE)) return NaN
    const t = ld - lg
    return t >= 0 ? 1 / (1 + Math.exp(-t)) : Math.exp(t) / (1 + Math.exp(t))
  })
  return { value, generatorLogDensity: logPg }
}
