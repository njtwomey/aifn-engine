/**
 * `aifn-compute/probability/stats`: statistics of samples, on plain numeric arrays (`ArrayLike<number>`) and
 * `aifn-compute/foundation/tensor` tensors, as numpy, scipy.stats and statsmodels compute them.
 *
 * - Descriptive: `standardDeviation`, `skewness`, `kurtosis`, `range`, `extent`, `mode`, the weighted `weightedMean`
 *   and `weightedVariance` (population, frequency or reliability weights), `covariance` and `correlation`, and
 *   `zScores` or `standardise` (which returns the centre and scale for new data).
 * - Order statistics: `quantile` by the thirteen methods of numpy (`quantileMethods`), `median`,
 *   `interquartileRange`, `sorted`, `expectile` and `expectiles`.
 * - Ranks: `argsort` (stable), `ranks` with a `TiePolicy`, and the rank correlations `spearman` and `kendallTau`
 *   ($\tau_b$).
 * - Distributions of a sample: `histogram` with an explicit `BinRule`, the empirical CDF `ecdf` (its steps) and
 *   `ecdfAt` (at given points), and Gaussian KDE: `kde` and `kdeBandwidth` in one dimension, `multivariateKde` in $d$.
 * - Sequences: running moments (`emptyMoments`, `momentsPush`, `momentsMerge`, `momentsVariance`), `runningMean` and
 *   `runningVariance`; `autocovariance`, `autocorrelation`, `crossCovariance` and `crossCorrelation` (by FFT for long
 *   sequences: the one autocorrelation in aifn), and the correlogram `sampleAcf` (with white-noise and Bartlett bands)
 *   and `samplePacf`.
 * - Resampling: `resampleIndices`, `shuffled`, `bootstrap` with `bootstrapInterval`, `permutationTest`, and Kish's
 *   `importanceEffectiveSampleSize` (the one ESS of weights in aifn).
 * - Power transforms: `boxCox` and `yeoJohnson`, their inverses, and `boxCoxLambda` and `yeoJohnsonLambda` for
 *   $\lambda$ by maximum likelihood.
 * - Robust location and scatter: `minimumCovarianceDeterminant` (FastMCD) and `squaredMahalanobis`.
 *
 * Hypothesis tests (the Kolmogorov–Smirnov test among them) are `aifn-compute/probability/tests`. The plain reductions
 * `sum`, `mean`, `min`, `max` and `variance` are `aifn-compute/foundation/tensor`'s (one definition per operation);
 * stats adds the statistics tensor does not have, which accept arrays as well as tensors. `statsFunctions` is the
 * registry of the module.
 *
 * Conventions: variances are population (divisor $n$) unless `{ sample: true }`; ranks average ties; histograms count
 * values equal to the last edge in the last bin and report dropped values; randomness comes from an
 * `aifn-compute/foundation/random` stream, first argument. Reductions take `{ axis, keepDims }` to reduce a tensor
 * along one axis (a tensor result); without an axis a tensor of any rank reduces over every element. Sequence and
 * pairwise functions need rank-1 tensors. Array results are rank-1 tensors (float64, or int32 for indices and lags).
 * Invalid input throws `DomainError` or `ShapeError`; nothing here is differentiable.
 */

export {
  correlation,
  covariance,
  extent,
  kurtosis,
  mode,
  range,
  skewness,
  standardDeviation,
  standardise,
  weightedMean,
  weightedVariance,
  zScores,
  type Along,
  type KurtosisOptions,
  type SampleOption,
  type Whole,
} from './descriptive'
export {
  interquartileRange,
  median,
  quantile,
  quantileMethods,
  sorted,
  type QuantileMethod,
  type QuantileOptions,
} from './quantile'
export { expectile, expectiles, type ExpectileOptions } from './expectile'
export { argsort, kendallTau, ranks, spearman, type TiePolicy } from './ranks'
export {
  ecdf,
  ecdfAt,
  histogram,
  kde,
  kdeBandwidth,
  multivariateKde,
  type BandwidthRule,
  type BinRule,
  type Histogram,
  type MultivariateKde,
} from './density'
export {
  autocorrelation,
  autocovariance,
  crossCorrelation,
  crossCovariance,
  emptyMoments,
  momentsMerge,
  momentsPush,
  momentsVariance,
  runningMean,
  runningVariance,
  type LagOptions,
  type Moments,
} from './sequence'
export {
  bootstrap,
  bootstrapInterval,
  importanceEffectiveSampleSize,
  permutationTest,
  resampleIndices,
  shuffled,
  type Bootstrap,
  type PermutationTest,
} from './resampling'
export {
  boxCox,
  boxCoxInverse,
  boxCoxLambda,
  yeoJohnson,
  yeoJohnsonInverse,
  yeoJohnsonLambda,
  type PowerLambda,
} from './power'
export type { AxisOption, Data } from './input'
export { sampleAcf, samplePacf, type SampleAcf } from './acf'
export { minimumCovarianceDeterminant, squaredMahalanobis, type Mcd, type McdOptions } from './robust'
export { statsFunctions } from './registry'
