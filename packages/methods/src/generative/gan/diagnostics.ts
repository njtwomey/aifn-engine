/**
 * Diagnostics of a GAN on toy data with a known density: the coverage of the data's modes by generated points, and
 * the optimal discriminator $D^*(\xvec) = p_{\text{data}}(\xvec) / (p_{\text{data}}(\xvec) + p_g(\xvec))$
 * (Goodfellow et al., 2014, Proposition 1) with $p_g$ estimated by a Gaussian KDE of generated points.
 *
 * The known density is any labelled mixture $p(\xvec) = \sum_j \pi_j \, p(\xvec \mid j)$ with a log density per mode:
 * the `model` of a classification truth of `aifn-methods/data` (a ring or grid of Gaussians, a pinwheel), whose
 * classes are the modes.
 */

import { multivariateKde } from 'aifn-compute/probability/stats'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { mixtureLogDensityOf, modeOf, type LabelledDensity } from '../densities'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Mode coverage of a set of generated points. */
export type ModeCoverage = {
  /** The number of modes hit: those holding at least one, and `minShare` times $\pi_j n$, high-quality points. */
  hit: number
  /** The number of modes $k$ of the known density. */
  modes: number
  /**
   * The share of generated points that are high quality: $\log p_{\text{data}}(\xvec)$ at or above `threshold` (NaN
   * for no points).
   */
  quality: number
  /** High-quality points per mode, $k$ counts. */
  perMode: Float64Array
  /** The log-density threshold of quality, as given or as found from the quantile. */
  threshold: number
}

/** Options of `modeCoverage`. */
export type ModeCoverageOptions = {
  /**
   * The quality threshold on $\log p_{\text{data}}$: a number, or `{ quantile: q }` for the $q$-quantile of
   * $\log p_{\text{data}}$ over the reference points `real` (default $q = 0.01$, so 99% of real points count as high
   * quality).
   */
  threshold?: number | { quantile: number }
  /** Real points $[m, d]$ for the quantile threshold; required when the threshold is a quantile. */
  real?: Tensor
  /**
   * A mode is hit when it holds at least this share of $\pi_j n$ high-quality points, for $n$ generated points.
   * Default 0.2.
   */
  minShare?: number
}

/**
 * Mode coverage (after Srivastava et al., 2017, and Metz et al., 2017, who count modes with high-quality samples near
 * them): each generated point is high quality when its log density under the data is at or above a threshold, and is
 * assigned to its most probable mode; a mode $j$ is hit when it holds at least `minShare` times $\pi_j n$
 * high-quality points, the count it would hold of $n$ points drawn from the data. Throws `DomainError` for a quantile
 * threshold without `real`.
 *
 * @param model The known density of the data, a labelled mixture whose classes are the modes.
 * @param samples The generated points $[n, d]$.
 * @param options The quality threshold (with the real points a quantile needs) and `minShare`.
 * @returns The modes hit out of all, the share of high-quality points, their count per mode and the threshold used.
 *
 * @example A collapsed generator covers one of two modes
 * // Two equal unit Gaussians at -2 and 2, in 1-d
 * const logN = (v, m) => -0.5 * (v - m) ** 2 - 0.5 * Math.log(2 * Math.PI)
 * const model = {
 *   classes: 2,
 *   priors: [0.5, 0.5],
 *   logDensity: (x) => tensor(toArray(x).map(([v]) => [logN(v, -2), logN(v, 2)])),
 * }
 * // Every generated point is near the mode at 2
 * const samples = normal(stream(1), 2, 0.3, { shape: [100, 1] })
 * const c = modeCoverage(model, samples, { threshold: -4 })
 * print('modes hit:', c.hit, 'of', c.modes, ' quality:', c.quality, ' per mode:', c.perMode)
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

/** The log density at or below which, for both densities, $D^*$ is left undefined (NaN): neither puts mass there. */
const NEGLIGIBLE = -8

/**
 * The optimal discriminator $D^*(\xvec) = p_{\text{data}}(\xvec) / (p_{\text{data}}(\xvec) + p_g(\xvec))$ at the
 * query points, given $\log p_{\text{data}}$ there and generated points: $p_g$ by `multivariateKde` with its bandwidth
 * chosen by leave-one-out likelihood (Scott's rule oversmooths separated narrow modes, which would push $D^*$ towards 1
 * on the data). Computed stably as $\sigma(\log p_{\text{data}} - \log p_g)$. Also returns $\log p_g$. When the KDE
 * cannot be fitted, as when the generated points lie on a line or a point (a fully collapsed generator), $p_g$ has no
 * density and both are NaN. $D^*$ is NaN where both log densities are at or below $-8$.
 *
 * @param dataLogDensity $\log p_{\text{data}}$ at each query point, in the order of the rows of `at`.
 * @param samples The generated points $[n, d]$, the sample of the KDE.
 * @param at The query points $[m, d]$.
 * @returns `value`, $D^*$ at each query point, and `generatorLogDensity`, $\log p_g$ there ($m$ values each).
 *
 * @example Data from $\Gauss(0, 1)$ against a generator shifted to $\Gauss(0.5, 1)$, in 1-d
 * const logN = (v, m) => -0.5 * (v - m) ** 2 - 0.5 * Math.log(2 * Math.PI)
 * const at = tensor([[-1], [0], [1]])
 * const dataLogDensity = toFlat(at).map((v) => logN(v, 0))
 * const samples = normal(stream(1), 0.5, 1, { shape: [400, 1] })
 * const d = optimalDiscriminator(dataLogDensity, samples, at)
 * print('D* at -1, 0, 1:', d.value)
 * print('log p_g by KDE:', d.generatorLogDensity)
 * print('log p_g exact: ', toFlat(at).map((v) => logN(v, 0.5)))
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
