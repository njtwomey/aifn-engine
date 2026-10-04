/**
 * `aifn-compute/probability/tests`: hypothesis tests, intervals, effect sizes, multiple testing, sequential tests and survival
 * estimators, on one protocol.
 *
 * ```ts
 * const r = welchTTest(control, treatment, { alternative: 'two-sided', level: 0.95 })
 * r.statistic, r.df, r.pValue, r.ci, r.effectSize // t, Welch–Satterthwaite df, p, interval for μ₁ − μ₂, Cohen's d
 * r.null.cdf(r.statistic)                         // the statistic's null law is a distribution object
 * rejectionRegion(r, 0.05)                        // [[−∞, −2.03], [2.03, ∞]]
 * ```
 *
 * - **The protocol.** Every test returns a `TestResult`: `statistic` (with its TeX `symbol` and `df`), `null` (the
 *   statistic's law under H₀, a `Univariate` from `aifn-compute/probability/distributions`: StudentT, Normal, ChiSquare,
 *   FisherSnedecor, Binomial, Hypergeometric, a Categorical for exact rank laws, or a law built here for Kolmogorov's
 *   distributions and Grubbs' bound), `pValue` (the `tail` probability of `null`: upper, lower, both, or likelihood for
 *   exact two-sided tests), `alternative`, `ci` for the `estimand` with its `estimate`, `effectSize`, `method` and
 *   `n`. `pValueOf` and `rejectionRegion` read any null law the same way. Tests are registered (`testRegistry`,
 *   kind `test`).
 * - **Classical tests.** `oneSampleTTest`, `pairedTTest`, `pooledTTest`, `welchTTest`, `zTest`, `oneWayAnova`;
 *   `binomialTest`, `twoProportionZTest`; `chiSquareGoodnessOfFit`, `chiSquareIndependence`, `gTestGoodnessOfFit`,
 *   `gTestIndependence`, `fisherExact`; `mannWhitneyU`, `wilcoxonSignedRank`; `ksTest` (one and two samples, one- and
 *   two-sided), `shapiroWilk`; `ljungBox`; `grubbs`; `logRankTest`.
 * - **Intervals and effect sizes.** `meanInterval`, `differenceOfMeansInterval`, `proportionInterval` (Wald, Wilson,
 *   Wilson with continuity correction, Clopper–Pearson), `differenceOfProportionsInterval` (Wald, Newcombe);
 *   `cohensD`, `hedgesG`, `cohensH`, `oddsRatio` (sample or conditional MLE, with intervals); the independence tests report Cramér's V.
 * - **Power.** `tTestPower`, `zTestPower`: two-sided power against a shift in standard errors.
 * - **Multiple testing.** `bonferroni`, `holm`, `hochberg`, `benjaminiHochberg`, `benjaminiYekutieli`: adjusted
 *   p-values and rejections.
 * - **Sequential tests** (Algorithms over a data stream, `testsAlgorithms`): `sprt`, `msprt`, `confidenceSequence`,
 *   `groupSequentialTest` with `groupSequentialBoundaries` (alpha spending: O'Brien–Fleming, Pocock, power) and
 *   `constantBoundaries` (Pocock, O'Brien–Fleming), and `cusum` with `cusumAverageRunLength`.
 * - **Survival.** `kaplanMeier` (Greenwood, linear or log–log intervals), `nelsonAalen`, `logRankTest`.
 *
 * Conventions follow scipy.stats where it has the test (the alternative's direction, Yates' correction on 2 × 2
 * tables, the exact two-sided rule of `binomtest` and `fisher_exact`, the automatic choice of exact rank laws).
 */

export {
  checkLevel,
  continuousLaw,
  lowerTail,
  pivotInterval,
  pValueOf,
  rejectionRegion,
  rejects,
  supportValues,
  tailOf,
  upperTail,
  type Alternative,
  type EffectSize,
  type Interval,
  type Tail,
  type TestOptions,
  type TestResult,
} from './protocol'
export { cohensD, hedgesCorrection, hedgesG } from './effect'
export {
  differenceOfMeansInterval,
  meanInterval,
  oneSampleTTest,
  pairedTTest,
  pooledTTest,
  welchDegreesOfFreedom,
  welchTTest,
  zTest,
} from './t'
export {
  binomialTest,
  cohensH,
  differenceOfProportionsInterval,
  proportionInterval,
  twoProportionZTest,
  type ProportionIntervalMethod,
} from './proportions'
export {
  chiSquareGoodnessOfFit,
  chiSquareIndependence,
  expectedCounts,
  fisherExact,
  gTestGoodnessOfFit,
  gTestIndependence,
  oddsRatio,
  powerDivergence,
  type Table,
} from './contingency'
export { mannWhitneyNull, mannWhitneyU, signedRankNull, wilcoxonSignedRank, type RankMethod } from './ranks'
export {
  kolmogorovLimitSf,
  kolmogorovNull,
  kolmogorovSf,
  ksStatistic,
  ksTest,
  shapiroWilk,
  shapiroWilkCoefficients,
  shapiroWilkNull,
  smirnovSf,
  twoSampleKsNull,
  twoSampleKsSf,
  type KsReference,
  type KsStatistic,
  type KsTest,
} from './goodness'
export { grubbs, ljungBox, oneWayAnova, type GrubbsTest } from './samples'
export { benjaminiHochberg, benjaminiYekutieli, bonferroni, hochberg, holm, type MultipleTesting } from './multiple'
export {
  confidenceSequence,
  constantBoundaries,
  cusum,
  cusumAverageRunLength,
  groupSequentialBoundaries,
  groupSequentialTest,
  msprt,
  normalMixtureLogLikelihoodRatio,
  normalMixtureRadius,
  spentAlpha,
  sprt,
  waldBoundaries,
  type ConfidenceSequenceState,
  type CusumState,
  type Decision,
  type GroupSequentialBoundaries,
  type GroupSequentialState,
  type MsprtState,
  type NormalMixture,
  type Spending,
  type SprtState,
} from './sequential'
export { kaplanMeier, logRankTest, nelsonAalen, type KaplanMeier, type LogRankTest, type NelsonAalen } from './survival'
export { tTestPower, zTestPower } from './power'
export { testRegistry, testsAlgorithms, testsFunctions } from './registry'
