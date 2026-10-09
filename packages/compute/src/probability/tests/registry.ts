/**
 * The registry of `aifn-compute/probability/tests`: every hypothesis test (`kind: 'test'`, with what it reads, its
 * statistic, the family of its null law and the alternatives it accepts) and every multiple-testing procedure; the
 * sequential tests as algorithms over a data stream (`problem: 'sequence'`); and the intervals, effect sizes, null
 * laws and design calculations as functions.
 */

import {
  definer,
  entries,
  type AlgorithmInfo,
  type Entry,
  type FunctionInfo,
  type TestInfo,
} from 'aifn-compute/foundation/registry'
import * as contingency from './contingency'
import * as effect from './effect'
import * as goodness from './goodness'
import * as multiple from './multiple'
import * as protocol from './protocol'
import * as proportions from './proportions'
import * as ranks from './ranks'
import * as samples from './samples'
import * as sequential from './sequential'
import * as survival from './survival'
import * as t from './t'

const MODULE = 'probability/tests'
const test = definer<TestInfo>('test', MODULE)
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)
const fn = definer<FunctionInfo>('function', MODULE)

const ALL = ['two-sided', 'less', 'greater'] as const
const TWO = ['two-sided'] as const
const BASICS = ['p-value', 'significance-level', 'one-and-two-sided-tests']

// ── Means ────────────────────────────────────────────────────────────────────────────────────────────────────────────

test(
  {
    key: 'oneSampleTTest',
    name: 'One-sample t-test',
    stability: 'stable',
    data: 'one-sample',
    statistic: 't',
    null: 'StudentT',
    alternatives: ALL,
    parametric: true,
    notes: ['one-sample-t-test', 'student-t-distribution', ...BASICS],
    cite: ['student1908'],
  },
  t.oneSampleTTest,
)
test(
  {
    key: 'pairedTTest',
    name: 'Paired t-test',
    stability: 'stable',
    data: 'paired',
    statistic: 't',
    null: 'StudentT',
    alternatives: ALL,
    parametric: true,
    notes: ['paired-t-test', 'one-sample-t-test'],
    cite: ['student1908'],
  },
  t.pairedTTest,
)
test(
  {
    key: 'pooledTTest',
    name: 'Two-sample t-test (pooled variance)',
    stability: 'stable',
    data: 'two-sample',
    statistic: 't',
    null: 'StudentT',
    alternatives: ALL,
    parametric: true,
    notes: ['pooled-two-sample-t-test', 'welch-t-test'],
    cite: ['student1908'],
  },
  t.pooledTTest,
)
test(
  {
    key: 'welchTTest',
    name: "Welch's t-test",
    stability: 'stable',
    data: 'two-sample',
    statistic: 't',
    null: 'StudentT',
    alternatives: ALL,
    parametric: true,
    notes: ['welch-t-test', 'pooled-two-sample-t-test', 'a-b-test'],
    cite: ['welch1947', 'satterthwaite1946'],
  },
  t.welchTTest,
)
test(
  {
    key: 'zTest',
    name: 'z-test (known σ)',
    stability: 'stable',
    data: 'one-sample',
    statistic: 'z',
    null: 'Normal',
    alternatives: ALL,
    parametric: true,
    notes: ['z-test', 'standard-error'],
  },
  t.zTest,
)
test(
  {
    key: 'oneWayAnova',
    name: 'One-way analysis of variance',
    data: 'k-sample',
    statistic: 'F',
    null: 'FisherSnedecor',
    alternatives: TWO,
    parametric: true,
    notes: ['f-distribution'],
    cite: ['fisher1924'],
  },
  samples.oneWayAnova,
)

// ── Proportions and tables ───────────────────────────────────────────────────────────────────────────────────────────

test(
  {
    key: 'binomialTest',
    name: 'Exact binomial test',
    stability: 'stable',
    data: 'one-sample',
    statistic: 'k',
    null: 'Binomial',
    alternatives: ALL,
    parametric: false,
    notes: ['binomial-test', 'binomial-distribution'],
    cite: ['clopper1934'],
  },
  proportions.binomialTest,
)
test(
  {
    key: 'twoProportionZTest',
    name: 'Two-proportion z-test',
    stability: 'stable',
    data: 'two-sample',
    statistic: 'z',
    null: 'Normal',
    alternatives: ALL,
    parametric: false,
    notes: ['conversion-rate-metrics', 'z-test', 'a-b-test'],
  },
  proportions.twoProportionZTest,
)
test(
  {
    key: 'chiSquareGoodnessOfFit',
    name: 'Pearson χ² goodness-of-fit test',
    tex: '\\chi^2',
    stability: 'stable',
    data: 'table',
    statistic: '\\chi^2',
    null: 'ChiSquare',
    alternatives: TWO,
    parametric: false,
    notes: ['chi-squared-distribution', 'sample-ratio-mismatch'],
    cite: ['pearson1900'],
  },
  contingency.chiSquareGoodnessOfFit,
)
test(
  {
    key: 'chiSquareIndependence',
    name: 'Pearson χ² test of independence',
    tex: '\\chi^2',
    stability: 'stable',
    data: 'table',
    statistic: '\\chi^2',
    null: 'ChiSquare',
    alternatives: TWO,
    parametric: false,
    notes: ['chi-squared-distribution', 'nominal-association'],
    cite: ['pearson1900'],
  },
  contingency.chiSquareIndependence,
)
test(
  {
    key: 'gTestGoodnessOfFit',
    name: 'G-test of goodness of fit',
    stability: 'stable',
    data: 'table',
    statistic: 'G',
    null: 'ChiSquare',
    alternatives: TWO,
    parametric: false,
    notes: ['chi-squared-distribution'],
  },
  contingency.gTestGoodnessOfFit,
)
test(
  {
    key: 'gTestIndependence',
    name: 'G-test of independence',
    stability: 'stable',
    data: 'table',
    statistic: 'G',
    null: 'ChiSquare',
    alternatives: TWO,
    parametric: false,
    notes: ['chi-squared-distribution', 'nominal-association'],
  },
  contingency.gTestIndependence,
)
test(
  {
    key: 'fisherExact',
    name: "Fisher's exact test",
    stability: 'stable',
    data: 'table',
    statistic: 'a',
    null: 'Hypergeometric',
    alternatives: ALL,
    parametric: false,
    notes: ['hypergeometric-distribution', 'conversion-rate-metrics'],
    cite: ['fisher1922'],
  },
  contingency.fisherExact,
)

// ── Ranks and distributions ──────────────────────────────────────────────────────────────────────────────────────────

test(
  {
    key: 'mannWhitneyU',
    name: 'Mann–Whitney U test',
    stability: 'stable',
    data: 'two-sample',
    statistic: 'U',
    null: 'Categorical',
    alternatives: ALL,
    parametric: false,
    notes: ['revenue-and-heavy-tailed-metrics', 'permutation-tests'],
    cite: ['mann1947'],
  },
  ranks.mannWhitneyU,
)
test(
  {
    key: 'wilcoxonSignedRank',
    name: 'Wilcoxon signed-rank test',
    stability: 'stable',
    data: 'paired',
    statistic: 'T^+',
    null: 'Categorical',
    alternatives: ALL,
    parametric: false,
    notes: ['paired-t-test'],
  },
  ranks.wilcoxonSignedRank,
)
test(
  {
    key: 'ksTest',
    name: 'Kolmogorov–Smirnov test',
    stability: 'stable',
    data: 'one-sample',
    statistic: 'D',
    null: 'exact',
    alternatives: ALL,
    parametric: false,
    notes: ['drift-detection', 'permutation-tests'],
    cite: ['smirnov1948', 'massey1951'],
  },
  goodness.ksTest,
)
test(
  {
    key: 'shapiroWilk',
    name: 'Shapiro–Wilk test of normality',
    data: 'one-sample',
    statistic: 'W',
    null: 'Transformed',
    alternatives: TWO,
    parametric: false,
    notes: ['residual-diagnostics'],
  },
  goodness.shapiroWilk,
)
test(
  {
    key: 'ljungBox',
    name: 'Ljung–Box test',
    stability: 'stable',
    data: 'series',
    statistic: 'Q',
    null: 'ChiSquare',
    alternatives: TWO,
    parametric: false,
    notes: ['ljung-box-test', 'residual-diagnostics'],
    cite: ['ljung1978', 'box1970'],
  },
  samples.ljungBox,
)
test(
  {
    key: 'grubbs',
    name: "Grubbs' test",
    stability: 'stable',
    data: 'one-sample',
    statistic: 'G',
    null: 'exact',
    alternatives: ALL,
    parametric: true,
    notes: ['grubbs-test', 'z-score-and-robust-outlier-detection'],
    cite: ['grubbs1950', 'grubbs1969'],
  },
  samples.grubbs,
)
test(
  {
    key: 'logRankTest',
    name: 'Log-rank test',
    stability: 'stable',
    data: 'survival',
    statistic: '\\chi^2',
    null: 'ChiSquare',
    alternatives: TWO,
    parametric: false,
    notes: ['log-rank-test', 'censoring', 'survival-and-hazard-functions'],
    cite: ['mantel1966', 'peto1972'],
  },
  survival.logRankTest,
)

// ── Multiple testing ─────────────────────────────────────────────────────────────────────────────────────────────────

const MT = ['multiple-testing', 'many-metrics-and-variants']
/**
 * Register a multiple-testing procedure as a test on p-values, with the fields every procedure shares.
 *
 * @param key The procedure's registry key, its function's name.
 * @param name Its display name.
 * @param cite The keys of its references, or undefined for none.
 * @param f The procedure's function.
 */
const procedure = (key: string, name: string, cite: string[] | undefined, f: object) =>
  test(
    {
      key,
      name,
      stability: 'stable',
      data: 'p-values',
      statistic: 'p',
      null: 'Uniform',
      alternatives: TWO,
      parametric: false,
      notes: MT,
      ...(cite ? { cite } : {}),
    },
    f,
  )
procedure('bonferroni', 'Bonferroni correction', undefined, multiple.bonferroni)
procedure('holm', "Holm's step-down procedure", ['holm1979'], multiple.holm)
procedure('hochberg', "Hochberg's step-up procedure", undefined, multiple.hochberg)
procedure('benjaminiHochberg', 'Benjamini–Hochberg procedure', ['benjamini1995'], multiple.benjaminiHochberg)
procedure('benjaminiYekutieli', 'Benjamini–Yekutieli procedure', ['benjamini2001'], multiple.benjaminiYekutieli)

// ── Sequential tests ─────────────────────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'sprt',
    name: 'Sequential probability ratio test',
    problem: 'sequence',
    state: { iterate: 'llr', flags: ['terminated'] },
    notes: ['mixture-sequential-probability-ratio-test', 'peeking-and-optional-stopping'],
    cite: ['wald1945'],
  },
  sequential.sprt,
)
algorithm(
  {
    key: 'msprt',
    name: 'Mixture sequential probability ratio test',
    problem: 'sequence',
    state: { iterate: 'mean', objective: 'logLikelihoodRatio', flags: ['terminated'] },
    notes: ['mixture-sequential-probability-ratio-test', 'peeking-and-optional-stopping'],
    cite: ['robbins1970', 'johari2017', 'johari2022'],
  },
  sequential.msprt,
)
algorithm(
  {
    key: 'confidenceSequence',
    name: 'Normal-mixture confidence sequence',
    problem: 'sequence',
    state: { iterate: 'mean', objective: 'logEValue', flags: ['terminated'] },
    notes: ['e-values-and-confidence-sequences', 'martingale', 'peeking-and-optional-stopping'],
    cite: ['robbins1970', 'howard2021', 'ramdas2023'],
  },
  sequential.confidenceSequence,
)
algorithm(
  {
    key: 'groupSequentialTest',
    name: 'Group-sequential z-test',
    problem: 'sequence',
    state: { iterate: 'z', flags: ['terminated'] },
    notes: ['group-sequential-designs', 'peeking-and-optional-stopping'],
    cite: ['pocock1977', 'obrien1979', 'jennison2000'],
  },
  sequential.groupSequentialTest,
)
algorithm(
  {
    key: 'cusum',
    name: 'CUSUM control chart',
    problem: 'sequence',
    state: { iterate: 'upper', flags: ['terminated'] },
    notes: ['cumulative-sum-control-chart', 'change-point-detection'],
    cite: ['page1954'],
  },
  sequential.cusum,
)

// ── Functions: intervals, effect sizes, null laws, design ────────────────────────────────────────────────────────────

fn(
  {
    key: 'meanInterval',
    name: 'Confidence interval for a mean',
    role: 'estimator',
    notes: ['confidence-intervals', 'standard-error'],
  },
  t.meanInterval,
)
fn(
  {
    key: 'differenceOfMeansInterval',
    name: 'Confidence interval for a difference of means',
    role: 'estimator',
    notes: ['confidence-intervals', 'welch-t-test'],
  },
  t.differenceOfMeansInterval,
)
fn(
  {
    key: 'proportionInterval',
    name: 'Confidence interval for a proportion',
    role: 'estimator',
    notes: ['confidence-intervals', 'binomial-test', 'metric-confidence-intervals'],
    cite: ['wilson1927', 'clopper1934', 'brown2001'],
  },
  proportions.proportionInterval,
)
fn(
  {
    key: 'differenceOfProportionsInterval',
    name: 'Confidence interval for a difference of proportions',
    role: 'estimator',
    notes: ['confidence-intervals', 'conversion-rate-metrics'],
  },
  proportions.differenceOfProportionsInterval,
)
fn(
  { key: 'cohensD', name: "Cohen's d", role: 'estimator', notes: ['effect-size'], cite: ['cohen1988'] },
  effect.cohensD,
)
fn(
  { key: 'hedgesG', name: "Hedges' g", role: 'estimator', notes: ['effect-size'], cite: ['hedges1981'] },
  effect.hedgesG,
)
fn(
  { key: 'cohensH', name: "Cohen's h", role: 'estimator', notes: ['effect-size'], cite: ['cohen1988'] },
  proportions.cohensH,
)
fn(
  { key: 'oddsRatio', name: 'Odds ratio', role: 'estimator', notes: ['effect-size', 'nominal-association'] },
  contingency.oddsRatio,
)
fn(
  {
    key: 'powerDivergence',
    name: 'Cressie–Read power divergence',
    role: 'estimator',
    notes: ['chi-squared-distribution'],
  },
  contingency.powerDivergence,
)
fn(
  {
    key: 'expectedCounts',
    name: 'Expected counts under independence',
    role: 'estimator',
    notes: ['nominal-association'],
  },
  contingency.expectedCounts,
)
fn(
  { key: 'ksStatistic', name: 'Kolmogorov–Smirnov statistic', role: 'estimator', notes: ['drift-detection'] },
  goodness.ksStatistic,
)
fn(
  { key: 'kolmogorovNull', name: "Kolmogorov's distribution", role: 'construction', notes: ['drift-detection'] },
  goodness.kolmogorovNull,
)
fn(
  { key: 'twoSampleKsNull', name: 'Two-sample Kolmogorov–Smirnov distribution', role: 'construction' },
  goodness.twoSampleKsNull,
)
fn({ key: 'kolmogorovSf', name: 'Kolmogorov survival function', role: 'property' }, goodness.kolmogorovSf)
fn(
  { key: 'kolmogorovLimitSf', name: "Kolmogorov's limiting survival function", role: 'property' },
  goodness.kolmogorovLimitSf,
)
fn(
  { key: 'smirnovSf', name: 'Smirnov one-sided survival function', role: 'property', cite: ['smirnov1948'] },
  goodness.smirnovSf,
)
fn(
  { key: 'twoSampleKsSf', name: 'Two-sample Kolmogorov–Smirnov survival function', role: 'property' },
  goodness.twoSampleKsSf,
)
fn(
  {
    key: 'shapiroWilkCoefficients',
    name: 'Shapiro–Wilk coefficients',
    role: 'construction',
    notes: ['residual-diagnostics'],
  },
  goodness.shapiroWilkCoefficients,
)
fn({ key: 'shapiroWilkNull', name: 'Shapiro–Wilk null distribution', role: 'construction' }, goodness.shapiroWilkNull)
fn(
  { key: 'mannWhitneyNull', name: 'Exact Mann–Whitney U distribution', role: 'construction', cite: ['mann1947'] },
  ranks.mannWhitneyNull,
)
fn({ key: 'signedRankNull', name: 'Exact signed-rank distribution', role: 'construction' }, ranks.signedRankNull)
fn(
  {
    key: 'pValueOf',
    name: 'p-value from a null distribution',
    role: 'property',
    notes: ['p-value', 'one-and-two-sided-tests'],
  },
  protocol.pValueOf,
)
fn(
  {
    key: 'rejectionRegion',
    name: 'Rejection region at level α',
    role: 'property',
    notes: ['significance-level', 'type-i-and-type-ii-errors'],
  },
  protocol.rejectionRegion,
)
fn(
  { key: 'pivotInterval', name: 'Interval from a pivot', role: 'estimator', notes: ['confidence-intervals'] },
  protocol.pivotInterval,
)
fn(
  {
    key: 'welchDegreesOfFreedom',
    name: 'Welch–Satterthwaite degrees of freedom',
    role: 'estimator',
    notes: ['welch-t-test'],
    cite: ['satterthwaite1946'],
  },
  t.welchDegreesOfFreedom,
)
fn(
  {
    key: 'hedgesCorrection',
    name: "Hedges' small-sample correction",
    role: 'property',
    notes: ['effect-size'],
    cite: ['hedges1981'],
  },
  effect.hedgesCorrection,
)
fn(
  {
    key: 'kaplanMeier',
    name: 'Kaplan–Meier estimator',
    role: 'estimator',
    notes: ['kaplan-meier-estimator', 'censoring'],
    cite: ['kaplan1958', 'greenwood1926'],
  },
  survival.kaplanMeier,
)
fn(
  {
    key: 'nelsonAalen',
    name: 'Nelson–Aalen estimator',
    role: 'estimator',
    notes: ['nelson-aalen-estimator', 'survival-and-hazard-functions'],
    cite: ['nelson1972', 'aalen1978'],
  },
  survival.nelsonAalen,
)
fn(
  {
    key: 'waldBoundaries',
    name: "Wald's SPRT boundaries",
    role: 'construction',
    notes: ['mixture-sequential-probability-ratio-test'],
    cite: ['wald1945'],
  },
  sequential.waldBoundaries,
)
fn(
  {
    key: 'normalMixtureLogLikelihoodRatio',
    name: 'Normal-mixture likelihood ratio',
    role: 'estimator',
    notes: ['mixture-sequential-probability-ratio-test', 'e-values-and-confidence-sequences'],
    cite: ['robbins1970'],
  },
  sequential.normalMixtureLogLikelihoodRatio,
)
fn(
  {
    key: 'normalMixtureRadius',
    name: 'Normal-mixture confidence-sequence radius',
    role: 'property',
    notes: ['e-values-and-confidence-sequences'],
    cite: ['robbins1970', 'howard2021'],
  },
  sequential.normalMixtureRadius,
)
fn(
  {
    key: 'spentAlpha',
    name: 'Alpha-spending function',
    role: 'property',
    notes: ['group-sequential-designs'],
    cite: ['lan1983'],
  },
  sequential.spentAlpha,
)
fn(
  {
    key: 'groupSequentialBoundaries',
    name: 'Group-sequential boundaries by alpha spending',
    role: 'construction',
    notes: ['group-sequential-designs'],
    cite: ['lan1983', 'armitage1969', 'jennison2000'],
  },
  sequential.groupSequentialBoundaries,
)
fn(
  {
    key: 'constantBoundaries',
    name: 'Pocock and O’Brien–Fleming boundaries',
    role: 'construction',
    notes: ['group-sequential-designs'],
    cite: ['pocock1977', 'obrien1979'],
  },
  sequential.constantBoundaries,
)
fn(
  {
    key: 'cusumAverageRunLength',
    name: 'CUSUM average run length',
    role: 'property',
    notes: ['cumulative-sum-control-chart'],
    cite: ['siegmund1985'],
  },
  sequential.cusumAverageRunLength,
)

/** Every hypothesis test and multiple-testing procedure of the module, keyed by name. */
export const testRegistry: Readonly<Record<string, Entry<(...args: never[]) => unknown, TestInfo>>> = entries<TestInfo>(
  'test',
  t,
  proportions,
  contingency,
  ranks,
  goodness,
  samples,
  multiple,
  survival,
) as Readonly<Record<string, Entry<(...args: never[]) => unknown, TestInfo>>>

/** The sequential tests, keyed by factory name. */
export const testsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', sequential) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The intervals, effect sizes, null laws and design calculations, keyed by name. */
export const testsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>(
    'function',
    t,
    proportions,
    contingency,
    effect,
    goodness,
    ranks,
    protocol,
    survival,
    sequential,
  ) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
