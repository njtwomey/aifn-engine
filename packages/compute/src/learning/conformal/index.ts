/**
 * `aifn-compute/learning/conformal`: split conformal prediction, sets and intervals with a finite-sample coverage
 * guarantee around any fitted model.
 *
 * - The quantile: `conformalQuantile`, the $\lceil (n + 1)(1 - \alpha) \rceil$-th smallest of $n$ calibration scores
 *   ($+\infty$ when there are too few), and `mondrianQuantiles`, one per group for coverage within each group.
 * - Regression: `splitConformalRegression` (intervals $\hat y \pm \hat q$ of one width, from absolute residuals) and
 *   `conformalisedQuantileRegression` (a quantile model's own intervals, widened or narrowed by $\hat q$).
 * - Classification: `conformalClassification` (sets by the LAC, APS or RAPS score) and `classificationScores` (the
 *   calibration scores of the true labels).
 * - Checking: `intervalCoverage` and `setCoverage`, the empirical coverage and mean width or size.
 * - `conformalFunctions`: the module's functions as registry entries, keyed by name.
 *
 * Coverage is at least $1 - \alpha$ when calibration and test cases are exchangeable, on average over both, not for
 * each case. The model is fitted elsewhere: every function takes its predictions. Invalid inputs throw `DomainError`.
 * Everything is deterministic except APS and RAPS given a stream.
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
