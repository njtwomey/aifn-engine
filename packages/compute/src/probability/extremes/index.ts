/**
 * `aifn-compute/probability/extremes`: peaks over threshold. The maximum-likelihood generalised Pareto fit of excesses
 * (`fitGeneralisedPareto`), the threshold fit (`peaksOverThreshold`), tail probabilities and quantiles beyond the data
 * (`tailProbability`, `tailQuantile`) and the mean excess function (`meanExcess`).
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
