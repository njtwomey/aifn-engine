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
