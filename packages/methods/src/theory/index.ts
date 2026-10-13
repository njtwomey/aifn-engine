/**
 * `aifn-methods/theory`: simulators that demonstrate results of learning theory and probability.
 *
 * - `aifn-methods/theory/bias-variance`: the bias–variance decomposition by resampling training sets, for one model or
 *   across polynomial degrees and $k$-NN neighbour counts.
 * - `aifn-methods/theory/double-descent`: random-features regression across the interpolation threshold, where the
 *   minimum-norm fit's test error peaks and falls again.
 * - `aifn-methods/theory/capacity`: shattering by half-planes, rectangles and intervals (the VC dimension), and the
 *   empirical Rademacher complexity against Massart's bound.
 * - `aifn-methods/theory/concentration`: Chebyshev, Hoeffding, Bernstein, Chernoff and McDiarmid bounds against
 *   simulated tails, and the law of large numbers and central limit theorem.
 * - `aifn-methods/theory/shrinkage`: the James–Stein estimator and Stein's paradox, its risk against the
 *   maximum-likelihood estimate's, simulated and exact.
 * - `aifn-methods/theory/optimism`: the optimism of the training error against the covariance penalty, and the
 *   effective degrees of freedom of least squares, ridge, nearest neighbours and a search over noise features.
 * - `aifn-methods/theory/model-selection`: a penalty tuned by cross-validation over many data sets, with the best
 *   cross-validation score, nested cross-validation and test-set estimates against the true error.
 * - `aifn-methods/theory/adaptive-holdout`: holdouts reused adaptively: the Ladder, Thresholdout, the boosting attack
 *   that fools a full-precision leaderboard, and the expected maximum of $J$ normals.
 * - `aifn-methods/theory/algorithm-comparison`: the Type I error of the tests that compare two learning algorithms,
 *   simulated with two equally good learners after Dietterich (1998).
 *
 * The family exports its shared layer, one-dimensional regression problems and least-squares fits: `targetFunction`
 * (a `TargetName` to its function), `drawTrainingSet`, `unitGrid`, `leastSquaresWeights` (minimum-norm or ridge) and
 * `applyWeights`. Every simulator takes a stream and draws from its children, so a seed fixes the whole study.
 */

export {
  applyWeights,
  drawTrainingSet,
  leastSquaresWeights,
  targetFunction,
  unitGrid,
  type TargetName,
} from './regression'
