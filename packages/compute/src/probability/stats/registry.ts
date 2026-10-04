/**
 * The functions of `aifn-compute/probability/stats`, registered with their role and the notes they serve. The
 * Kolmogorov–Smirnov functions of `goodness.ts` are left to `aifn-compute/probability/tests`, which owns the test protocol.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as acf from './acf'
import * as density from './density'
import * as descriptive from './descriptive'
import * as expectile from './expectile'
import * as power from './power'
import * as quantile from './quantile'
import * as ranks from './ranks'
import * as resampling from './resampling'
import * as robust from './robust'
import * as sequence from './sequence'

const fn = definer<FunctionInfo>('function', 'probability/stats')
const VAR = ['variance-and-covariance']

// ── Descriptive statistics ───────────────────────────────────────────────────────────────────────────────────────────

fn(
  { key: 'standardDeviation', name: 'Standard deviation', role: 'estimator', notes: VAR },
  descriptive.standardDeviation,
)
fn(
  { key: 'weightedMean', name: 'Weighted mean', role: 'estimator', notes: ['expectation', 'importance-sampling'] },
  descriptive.weightedMean,
)
fn({ key: 'weightedVariance', name: 'Weighted variance', role: 'estimator', notes: VAR }, descriptive.weightedVariance)
fn({ key: 'extent', name: 'Extent (minimum and maximum)', role: 'estimator' }, descriptive.extent)
fn({ key: 'range', name: 'Range', role: 'estimator' }, descriptive.range)
fn({ key: 'mode', name: 'Mode', role: 'estimator' }, descriptive.mode)
fn(
  { key: 'skewness', name: 'Skewness', role: 'estimator', notes: ['moment-generating-function'] },
  descriptive.skewness,
)
fn(
  { key: 'kurtosis', name: 'Kurtosis', role: 'estimator', notes: ['moment-generating-function'] },
  descriptive.kurtosis,
)
fn({ key: 'covariance', name: 'Covariance', role: 'estimator', notes: VAR }, descriptive.covariance)
fn(
  { key: 'correlation', name: 'Pearson correlation', role: 'estimator', notes: ['pearson-correlation', ...VAR] },
  descriptive.correlation,
)
fn(
  {
    key: 'zScores',
    name: 'z-scores',
    role: 'transform',
    notes: ['z-score-and-robust-outlier-detection', 'feature-scaling'],
  },
  descriptive.zScores,
)
fn({ key: 'standardise', name: 'Standardise', role: 'transform', notes: ['feature-scaling'] }, descriptive.standardise)

// ── Quantiles, expectiles and ranks ──────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'quantile',
    name: 'Quantile',
    summary: 'Sample quantiles by the nine Hyndman–Fan definitions.',
    role: 'estimator',
    notes: ['quantile-calibration', 'inverse-transform-sampling'],
  },
  quantile.quantile,
)
fn(
  { key: 'median', name: 'Median', role: 'estimator', notes: ['z-score-and-robust-outlier-detection'] },
  quantile.median,
)
fn(
  {
    key: 'interquartileRange',
    name: 'Interquartile range',
    role: 'estimator',
    notes: ['z-score-and-robust-outlier-detection'],
  },
  quantile.interquartileRange,
)
fn({ key: 'sorted', name: 'Sorted values (order statistics)', role: 'transform' }, quantile.sorted)
fn(
  {
    key: 'expectile',
    name: 'Expectile',
    summary: 'The minimiser of the asymmetric squared loss at level τ; the mean at τ = ½.',
    role: 'estimator',
    notes: ['expectile-generalised-additive-models'],
    cite: ['newey1987'],
  },
  expectile.expectile,
)
fn(
  { key: 'expectiles', name: 'Expectiles', role: 'estimator', notes: ['expectile-generalised-additive-models'] },
  expectile.expectiles,
)
fn({ key: 'argsort', name: 'Argsort', role: 'transform' }, ranks.argsort)
fn({ key: 'ranks', name: 'Ranks', role: 'transform', notes: ['rank-correlation'] }, ranks.ranks)
fn({ key: 'spearman', name: "Spearman's ρ", role: 'estimator', notes: ['rank-correlation'] }, ranks.spearman)
fn({ key: 'kendallTau', name: "Kendall's τ", role: 'estimator', notes: ['rank-correlation'] }, ranks.kendallTau)

// ── Densities and the empirical CDF ──────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'histogram',
    name: 'Histogram',
    summary: 'Counts or densities over bins chosen by a rule (Sturges, Scott, Freedman–Diaconis, …) or given edges.',
    role: 'estimator',
    notes: ['histogram-density-estimation'],
  },
  density.histogram,
)
fn({ key: 'ecdf', name: 'Empirical CDF', role: 'estimator', notes: ['inverse-transform-sampling'] }, density.ecdf)
fn({ key: 'ecdfAt', name: 'Empirical CDF at points', role: 'estimator' }, density.ecdfAt)
fn(
  { key: 'kde', name: 'Gaussian kernel density estimate', role: 'estimator', notes: ['kernel-density-estimation'] },
  density.kde,
)
fn(
  {
    key: 'multivariateKde',
    name: 'Multivariate Gaussian kernel density estimate',
    summary: "A d-dimensional Gaussian KDE with the sample covariance scaled by Scott's or Silverman's factor.",
    role: 'estimator',
    notes: ['kernel-density-estimation'],
  },
  density.multivariateKde,
)
fn(
  {
    key: 'kdeBandwidth',
    name: 'KDE bandwidth',
    summary: "Scott's or Silverman's rule-of-thumb bandwidth.",
    role: 'estimator',
    notes: ['kernel-density-estimation'],
  },
  density.kdeBandwidth,
)

// ── Sequences ────────────────────────────────────────────────────────────────────────────────────────────────────────

const ACF = ['autocorrelation-and-partial-autocorrelation', 'autocorrelation-and-wiener-khinchin']
fn({ key: 'autocovariance', name: 'Autocovariance', role: 'estimator', notes: ACF }, sequence.autocovariance)
fn({ key: 'autocorrelation', name: 'Autocorrelation', role: 'estimator', notes: ACF }, sequence.autocorrelation)
fn(
  { key: 'crossCovariance', name: 'Cross-covariance', role: 'estimator', notes: ['coherence-and-cross-spectra'] },
  sequence.crossCovariance,
)
fn(
  {
    key: 'crossCorrelation',
    name: 'Cross-correlation',
    role: 'estimator',
    notes: ['coherence-and-cross-spectra', 'matched-filter'],
  },
  sequence.crossCorrelation,
)
fn(
  {
    key: 'momentsPush',
    name: "Welford's update",
    summary: "Add one value to running moments (count, mean, M₂) by Welford's update.",
    role: 'estimator',
    notes: ['streaming-algorithms-for-learning'],
    cite: ['chan1983'],
  },
  sequence.momentsPush,
)
fn(
  {
    key: 'momentsMerge',
    name: 'Merge running moments',
    summary: "Chan's parallel combination of two sets of running moments.",
    role: 'estimator',
    notes: ['streaming-algorithms-for-learning'],
  },
  sequence.momentsMerge,
)
fn(
  {
    key: 'momentsVariance',
    name: 'Variance of running moments',
    role: 'estimator',
    notes: ['streaming-algorithms-for-learning'],
  },
  sequence.momentsVariance,
)
fn(
  { key: 'runningMean', name: 'Running mean', role: 'estimator', notes: ['law-of-large-numbers'] },
  sequence.runningMean,
)
fn(
  { key: 'runningVariance', name: 'Running variance', role: 'estimator', notes: ['streaming-algorithms-for-learning'] },
  sequence.runningVariance,
)
fn(
  {
    key: 'sampleAcf',
    name: 'Sample ACF with bands',
    summary: "Sample autocorrelations with Bartlett's confidence bands.",
    role: 'estimator',
    notes: ['autocorrelation-and-partial-autocorrelation', 'box-jenkins-method'],
  },
  acf.sampleAcf,
)
fn(
  {
    key: 'samplePacf',
    name: 'Sample PACF',
    summary: 'Sample partial autocorrelations by the Durbin–Levinson recursion.',
    role: 'estimator',
    notes: ['autocorrelation-and-partial-autocorrelation', 'box-jenkins-method'],
  },
  acf.samplePacf,
)

// ── Resampling ───────────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  { key: 'resampleIndices', name: 'Resample indices', role: 'simulation', random: true, notes: ['bootstrap'] },
  resampling.resampleIndices,
)
fn(
  { key: 'shuffled', name: 'Shuffle', role: 'simulation', random: true, notes: ['permutation-tests'] },
  resampling.shuffled,
)
fn(
  {
    key: 'bootstrap',
    name: 'Bootstrap',
    summary: 'The bootstrap distribution of a statistic over resamples drawn with replacement.',
    role: 'estimator',
    random: true,
    notes: ['bootstrap', 'standard-error'],
    cite: ['efron1979'],
  },
  resampling.bootstrap,
)
fn(
  {
    key: 'bootstrapInterval',
    name: 'Bootstrap interval',
    role: 'estimator',
    random: true,
    notes: ['bootstrap', 'confidence-intervals'],
    cite: ['efron1979'],
  },
  resampling.bootstrapInterval,
)
fn(
  {
    key: 'permutationTest',
    name: 'Permutation test',
    summary: 'A p-value from the statistic recomputed over random relabellings.',
    role: 'test',
    random: true,
    notes: ['permutation-tests', 'p-value'],
  },
  resampling.permutationTest,
)
fn(
  {
    key: 'importanceEffectiveSampleSize',
    name: "Kish's effective sample size",
    summary: '(Σw)²/Σw² of importance weights.',
    role: 'estimator',
    notes: ['importance-sampling'],
  },
  resampling.importanceEffectiveSampleSize,
)

// ── Power transforms ─────────────────────────────────────────────────────────────────────────────────────────────────

const POWER = ['feature-scaling', 'stationarity']
fn({ key: 'boxCox', name: 'Box–Cox transform', role: 'transform', notes: POWER }, power.boxCox)
fn({ key: 'boxCoxInverse', name: 'Inverse Box–Cox transform', role: 'transform', notes: POWER }, power.boxCoxInverse)
fn({ key: 'boxCoxLambda', name: 'Box–Cox λ by maximum likelihood', role: 'fit', notes: POWER }, power.boxCoxLambda)
fn({ key: 'yeoJohnson', name: 'Yeo–Johnson transform', role: 'transform', notes: POWER }, power.yeoJohnson)
fn(
  { key: 'yeoJohnsonInverse', name: 'Inverse Yeo–Johnson transform', role: 'transform', notes: POWER },
  power.yeoJohnsonInverse,
)
fn(
  { key: 'yeoJohnsonLambda', name: 'Yeo–Johnson λ by maximum likelihood', role: 'fit', notes: POWER },
  power.yeoJohnsonLambda,
)

// ── Robust location and scatter ──────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'minimumCovarianceDeterminant',
    name: 'Minimum covariance determinant',
    summary: 'Robust mean and covariance from the h-subset with the smallest determinant (FastMCD), reweighted.',
    role: 'estimator',
    notes: ['mahalanobis-distance-outliers', 'shrinkage-covariance-estimation'],
    cite: ['rousseeuw1984', 'rousseeuw1999'],
  },
  robust.minimumCovarianceDeterminant,
)
fn(
  {
    key: 'squaredMahalanobis',
    name: 'Squared Mahalanobis distances',
    tex: '(\\mathbf{x}_i - \\boldsymbol{\\mu})^\\top \\Sigma^{-1} (\\mathbf{x}_i - \\boldsymbol{\\mu})',
    summary: 'The squared Mahalanobis distance of every row under a location and covariance, by a Cholesky solve.',
    role: 'transform',
    notes: ['mahalanobis-distance-outliers'],
  },
  robust.squaredMahalanobis,
)

/** The functions of the module, keyed by name. */
export const statsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>(
    'function',
    descriptive,
    quantile,
    expectile,
    ranks,
    density,
    sequence,
    acf,
    resampling,
    power,
    robust,
  ) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
