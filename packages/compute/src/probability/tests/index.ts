/**
 * `aifn-compute/probability/tests`: hypothesis tests, intervals, effect sizes, multiple testing, sequential tests and
 * survival estimators, on one protocol.
 *
 * - **The protocol.** Every test returns a `TestResult`: `statistic` (with its TeX `symbol` and `df`), `null` (the
 *   statistic's law under $H_0$, a `Univariate` from `aifn-compute/probability/distributions`: StudentT, Normal,
 *   ChiSquare, FisherSnedecor, Binomial, Hypergeometric, a Categorical for exact rank laws, a transformed normal for
 *   Shapiro–Wilk's $W$, or a `continuousLaw` built here for Kolmogorov's distributions and Grubbs' bound), `pValue`
 *   (the `tail` probability of `null`: upper, lower, both, or likelihood for exact two-sided tests), `alternative`,
 *   `ci` for the `estimand` with its `estimate`, `effectSize`, `method` and `n`. `pValueOf`, `upperTail`, `lowerTail`
 *   and `rejectionRegion` read any null law the same way (`tailOf` maps an alternative to its rule, `supportValues`
 *   lists a discrete law's values), `rejects` compares $p$ with $\alpha$, and `pivotInterval` inverts a pivot into an
 *   interval. Tests are registered (`testRegistry`, kind `test`).
 * - **Classical tests.** `oneSampleTTest`, `pairedTTest`, `pooledTTest`, `welchTTest` (with
 *   `welchDegreesOfFreedom`), `zTest`, `oneWayAnova`; `binomialTest`, `twoProportionZTest`;
 *   `chiSquareGoodnessOfFit`, `chiSquareIndependence`, `gTestGoodnessOfFit`, `gTestIndependence` (on
 *   `powerDivergence` and `expectedCounts`), `fisherExact`; `mannWhitneyU`, `wilcoxonSignedRank`; `ksTest` (one and
 *   two samples, one- and two-sided, with `ksStatistic`), `shapiroWilk`; `ljungBox`; `grubbs`; `logRankTest`.
 * - **Null laws.** The exact laws behind the rank and goodness-of-fit tests, for critical values and tail
 *   probabilities: `mannWhitneyNull`, `signedRankNull`; `kolmogorovNull` (on `kolmogorovSf`, `smirnovSf` and
 *   `kolmogorovLimitSf`), `twoSampleKsNull` (on `twoSampleKsSf`); `shapiroWilkNull` with `shapiroWilkCoefficients`.
 * - **Intervals and effect sizes.** `meanInterval`, `differenceOfMeansInterval`, `proportionInterval` (Wald, Wilson,
 *   Wilson with continuity correction, Clopper–Pearson), `differenceOfProportionsInterval` (Wald, Newcombe);
 *   `cohensD`, `hedgesG` (with the exact `hedgesCorrection`), `cohensH`, `oddsRatio` (sample or conditional MLE, with
 *   intervals); the independence tests report Cramér's $V$.
 * - **Power.** `tTestPower`, `zTestPower`: two-sided power against a shift in standard errors.
 * - **Multiple testing.** `bonferroni`, `holm`, `hochberg`, `benjaminiHochberg`, `benjaminiYekutieli`: adjusted
 *   p-values and rejections.
 * - **Sequential tests** (Algorithms over a data stream, `testsAlgorithms`): `sprt` with `waldBoundaries`, `msprt`
 *   and `confidenceSequence` (on `normalMixtureLogLikelihoodRatio` and `normalMixtureRadius`),
 *   `groupSequentialTest` with `groupSequentialBoundaries` (alpha spending by `spentAlpha`: O'Brien–Fleming, Pocock,
 *   power) and `constantBoundaries` (Pocock, O'Brien–Fleming), and `cusum` with `cusumAverageRunLength`.
 * - **Survival.** `kaplanMeier` (Greenwood, linear or log–log intervals), `nelsonAalen`, `logRankTest`.
 *
 * Inputs are checked rather than repaired: a non-finite value, a count out of range or a level outside $(0, 1)$
 * throws `DomainError` (`checkLevel`), and nothing is dropped silently. Conventions follow scipy.stats where it has
 * the test (the alternative's direction, Yates' correction on $2 \times 2$ tables, the exact two-sided rule of
 * `binomtest` and `fisher_exact`, the automatic choice of exact rank laws). The intervals, effect sizes, null laws and
 * design calculations are registered as functions (`testsFunctions`).
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
