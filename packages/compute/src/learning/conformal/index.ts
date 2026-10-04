/**
 * `aifn-compute/learning/conformal`: split conformal prediction: the conformal quantile; regression intervals from absolute
 * residuals and conformalised quantile regression; classification sets by the LAC, APS and RAPS scores; Mondrian
 * (group-conditional) quantiles; empirical coverage and set size.
 */

export {
  classificationScores,
  conformalClassification,
  conformalisedQuantileRegression,
  conformalQuantile,
  intervalCoverage,
  mondrianQuantiles,
  setCoverage,
  splitConformalRegression,
  type ClassificationOptions,
  type ClassificationScore,
  type ConformalIntervals,
  type ConformalSets,
  type CoverageSummary,
} from './conformal'
export { conformalFunctions } from './registry'
