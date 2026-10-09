/**
 * `aifn-compute/probability/extremes`: tail estimation by peaks over threshold, with a generalised Pareto law for the
 * excesses over a high threshold.
 *
 * - Fitting: `peaksOverThreshold` keeps the observations above a threshold $u$ (given, or an empirical quantile) and
 *   fits their excesses; `fitGeneralisedPareto` is the maximum-likelihood fit itself, for excesses already formed.
 * - Estimates beyond the data: `tailProbability` ($P(X > x)$ for $x \ge u$) and `tailQuantile` (the level-$p$
 *   quantile, the anomaly threshold at a chosen risk).
 * - Choosing the threshold: `meanExcess`, the mean residual life plot, linear in $u$ above a good threshold.
 *
 * Data are plain arrays or tensors read as their flat values; invalid input throws `DomainError`. The results are plain
 * numbers.
 */

export {
  fitGeneralisedPareto,
  meanExcess,
  peaksOverThreshold,
  tailProbability,
  tailQuantile,
  type GeneralisedParetoFit,
  type PeaksOverThreshold,
  type PeaksOverThresholdOptions,
} from './extremes'
export { extremesFunctions } from './registry'
