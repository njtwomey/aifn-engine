/**
 * Statistics on plain numeric arrays (`ArrayLike<number>`) and `aifn-compute/probability/stats` tensors: descriptive statistics,
 * quantiles, expectiles, ranks and rank correlations, histograms with an explicit bin rule, the empirical CDF,
 * Gaussian KDE, running moments, autocovariance and cross-correlation (by FFT for long sequences: the one autocorrelation in aifn),
 * resampling (bootstrap, permutation tests, and Kish's importance ESS, the one ESS of weights in aifn), and the
 * Box–Cox and Yeo–Johnson power transforms with λ by maximum likelihood. Hypothesis tests (the Kolmogorov–Smirnov test
 * among them) are `aifn-compute/probability/tests`.
 *
 * The plain reductions `sum`, `mean`, `min`, `max` and `variance` are `aifn-compute/foundation/tensor`'s (one definition per
 * operation); stats adds the statistics tensor does not have, which accept arrays as well as tensors.
 *
 * Conventions: variances are population (÷ n) unless `{ sample: true }`; ranks average ties; histograms count
 * values equal to the last edge in the last bin and report dropped values; randomness comes from an `aifn-compute/foundation/random`
 * stream, first argument. Reductions take `{ axis, keepDims }` to reduce a tensor along one axis (a tensor result);
 * without an axis a tensor of any rank reduces over every element. Sequence and pairwise functions need rank-1
 * tensors. Array results are rank-1 tensors (float64, or int32 for indices and lags).
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
