/**
 * `aifn-compute/learning/metrics`: evaluation metrics, each defined once with its registry metadata, as
 * sklearn.metrics.
 *
 * - Confusion counts, from which every threshold metric is built: `confusionMatrix` (rows true classes) and
 *   `confusionMargins`; `binaryCounts`, `countsAtThreshold` and every rate of one table, `binaryRates`; per class,
 *   `oneVsRest`, `tallies`, `perClass` and the averaging rules of `averaged`.
 * - Metrics of predicted labels: `accuracy`, `errorRate`, `balancedAccuracy`, `precision`, `recall`, `fBeta`, `f1`,
 *   `precisionRecallFscoreSupport` (per class), `specificity`, `negativePredictiveValue`, `falsePositiveRate`,
 *   `falseNegativeRate`, `informedness`, `markedness`, `jaccardScore`, `matthewsCorrelation`, `cohensKappa` and
 *   `kappaFromTable`, the multi-label `hammingLoss` and `exactMatch`, and the likelihood ratios
 *   `positiveLikelihoodRatio`, `negativeLikelihoodRatio` and `diagnosticOddsRatio`.
 * - Curves of scores over every threshold, as `Curve` values a chart draws: `rocCurve` with `auroc` (binary or
 *   multiclass), `partialAuroc` and `rocConvexHull`; `precisionRecallCurve` with `averagePrecision` and
 *   `precisionRecallTrapezoid`; `precisionRecallGainCurve`, `detCurve`, `costCurve` with `normalisedExpectedCost`, and
 *   `gainCurve` (with lift). Operating points: `equalErrorRate` and `eer`, `youdenPoint`, `operatingPoint`,
 *   `specificityAtSensitivity`, `tprAtFpr`, `recallAtPrecision`. The binormal model's closed forms: `binormalRates`,
 *   `binormalCurves`, `binormalAuroc`, `binormalEqualErrorRate`, `binormalAveragePrecision`.
 * - Proper scoring rules for probabilities and predictive distributions: `logLoss`, `brierScore` with
 *   `brierDecomposition`, `sphericalScore`, `logScore` (any predictive), `gaussianLogScore`, `crpsGaussian`,
 *   `crpsEnsemble`, `intervalScore` with `coverage`, `pitValues` and `perplexity`. Calibration, which depends on the
 *   binning: `expectedCalibrationError`, `maximumCalibrationError`, `rmsCalibrationError`,
 *   `debiasedSquaredCalibrationError`, `sweepCalibrationError`, `confidenceCalibrationError`,
 *   `classwiseCalibrationError`, drawn by `reliabilityDiagram` with `consistencyBars`.
 * - Regression errors and deviances: `meanSquaredError`, `rootMeanSquaredError`, `meanAbsoluteError`,
 *   `medianAbsoluteError`, `maxError`, `r2Score`, `adjustedR2Score`, `explainedVariance`,
 *   `normalisedRootMeanSquaredError`, `huberLoss`, `logCoshError`, `meanSquaredLogError`, `pinballLoss`,
 *   `tweedieDeviance`, `tweedieUnitDeviance`, `poissonDeviance`, `gammaDeviance`; percentage and scaled errors for
 *   forecasts, `meanAbsolutePercentageError`, `symmetricMeanAbsolutePercentageError`,
 *   `weightedMeanAbsolutePercentageError`, `meanAbsoluteScaledError`, `rootMeanSquaredScaledError`.
 * - Ranking: `precisionAtK`, `recallAtK`, `rPrecision`, `hitRate`, `meanAveragePrecision`, `meanReciprocalRank`,
 *   `dcg` and `ndcg` (with `gainFunction` and `positionDiscount`), `expectedReciprocalRank`.
 * - Clustering, against true classes: `contingencyTable`, `pairConfusion`, `randIndex`, `adjustedRandIndex`,
 *   `fowlkesMallows`, `mutualInformationScore`, `normalisedMutualInformation`, `adjustedMutualInformation`,
 *   `homogeneity`, `completeness`, `vMeasure`, `homogeneityCompletenessV`, `variationOfInformation`; and from the
 *   features alone, `silhouetteSamples`, `silhouetteScore`, `calinskiHarabasz`, `daviesBouldin`, `dunnIndex`.
 * - Agreement and association: `pearsonCorrelation`, `spearmanCorrelation`, `kendallCorrelation`,
 *   `concordanceCorrelation`, `intraclassCorrelation`, `fleissKappa`, `krippendorffAlpha`, `chiSquareStatistic`,
 *   `cramersV`, `tschuprowT`, `contingencyCoefficient`, `theilsU`.
 * - Ordinal classes: `ordinalMeanAbsoluteError`, `macroMeanAbsoluteError`, `withinToleranceAccuracy`,
 *   `quadraticWeightedKappa`, `rankedProbabilityScore`, `ordinalConcordanceIndex`.
 * - Distances and embeddings: `euclideanDistance`, `manhattanDistance`, `chebyshevDistance`, `minkowskiDistance`,
 *   `cosineSimilarity`, `cosineDistance`, `angularDistance`, `mahalanobisDistance`, `hausdorffDistance` and
 *   `hausdorffDistances`, `orthogonalProcrustes` and `procrustesDisparity`; `alignment` and `uniformity` of
 *   representations.
 * - Uncertainty of a metric: `bootstrapMetric`, `pairedBootstrap`, `aurocDeLong` and `delongTest`, `waldInterval`,
 *   `wilsonInterval`.
 * - The registry: `defineMetric` and `isMetric`, `metricRegistry`, `getMetric`, `listMetrics`, `isBetter`, and
 *   `metricsFunctions` for the functions that are not metrics. The input helpers `classesOf`, `compareLabels`,
 *   `labelList`, `encodeLabels`, `positiveOf`, `isMatrixLike`, `denseMatrix`, `matrix`, `metricValues`, `sameLength`
 *   and `divide` let metrics defined elsewhere follow the same conventions.
 *
 * Every metric is a function returning a number, with `info` stating what it reads (`inputs`), whether `higher` or
 * `lower` is better, its range and its notes. Labels may be numbers, strings or booleans, and classes are listed in
 * sorted order; the positive class of a binary problem is `1`, else `true`, else the last class, unless `positive`
 * is given. A threshold $t$ predicts positive the cases with score $s_i \ge t$. A ratio with a zero denominator is
 * NaN (or `zeroDivision`, where offered), so an undefined value is reported rather than hidden; inputs of different
 * lengths throw `ShapeError`. Application metrics (text, detection, quality, generative, fairness, beyond-accuracy)
 * are in `aifn-methods/evaluation`.
 */

export {
  defineMetric,
  isMetric,
  type Capability,
  type Data,
  type InputKind,
  type Label,
  type Labels,
  type Metric,
  type MetricFunction,
  type MetricInfo,
  type MetricSpec,
  type Rows,
} from './core'
export {
  averaged,
  binaryCounts,
  binaryRates,
  confusionMargins,
  confusionMatrix,
  countsAtThreshold,
  oneVsRest,
  perClass,
  tallies,
  type Average,
  type AverageOptions,
  type BinaryCounts,
  type BinaryRates,
  type ConfusionMargins,
  type ConfusionMatrix,
  type ConfusionOptions,
  type CountStatistic,
  type OneVsRestCounts,
  type Tallies,
} from './confusion'
export {
  accuracy,
  balancedAccuracy,
  cohensKappa,
  diagnosticOddsRatio,
  errorRate,
  exactMatch,
  f1,
  fBeta,
  falseNegativeRate,
  falsePositiveRate,
  hammingLoss,
  informedness,
  jaccardScore,
  kappaFromTable,
  markedness,
  matthewsCorrelation,
  negativeLikelihoodRatio,
  negativePredictiveValue,
  positiveLikelihoodRatio,
  precision,
  precisionRecallFscoreSupport,
  recall,
  specificity,
  type ClassificationInput,
  type KappaWeights,
} from './classification'
export {
  auroc,
  averagePrecision,
  binormalAuroc,
  binormalAveragePrecision,
  binormalCurves,
  binormalEqualErrorRate,
  binormalRates,
  costCurve,
  detCurve,
  eer,
  equalErrorRate,
  gainCurve,
  normalisedExpectedCost,
  operatingPoint,
  partialAuroc,
  precisionRecallCurve,
  precisionRecallGainCurve,
  precisionRecallTrapezoid,
  recallAtPrecision,
  rocConvexHull,
  rocCurve,
  specificityAtSensitivity,
  tprAtFpr,
  youdenPoint,
  type AurocOptions,
  type OperatingPoint,
  type PrecisionRecallCurve,
  type RocCurve,
} from './curves'
export {
  brierDecomposition,
  brierScore,
  classwiseCalibrationError,
  confidenceCalibrationError,
  consistencyBars,
  coverage,
  crpsEnsemble,
  crpsGaussian,
  debiasedSquaredCalibrationError,
  expectedCalibrationError,
  gaussianLogScore,
  intervalScore,
  logLoss,
  logScore,
  maximumCalibrationError,
  perplexity,
  pitValues,
  reliabilityDiagram,
  rmsCalibrationError,
  sphericalScore,
  sweepCalibrationError,
  type BinStrategy,
  type BrierDecomposition,
  type GaussianForecast,
  type Interval,
  type PerCase,
  type Probabilities,
  type ReliabilityDiagram,
} from './probabilistic'
export {
  adjustedR2Score,
  explainedVariance,
  gammaDeviance,
  huberLoss,
  logCoshError,
  maxError,
  meanAbsoluteError,
  meanAbsolutePercentageError,
  meanAbsoluteScaledError,
  meanSquaredError,
  meanSquaredLogError,
  medianAbsoluteError,
  normalisedRootMeanSquaredError,
  pinballLoss,
  poissonDeviance,
  r2Score,
  rootMeanSquaredError,
  rootMeanSquaredScaledError,
  symmetricMeanAbsolutePercentageError,
  tweedieDeviance,
  tweedieUnitDeviance,
  weightedMeanAbsolutePercentageError,
  type RegressionOptions,
  type ScaledErrorOptions,
} from './regression'
export {
  dcg,
  gainFunction,
  positionDiscount,
  expectedReciprocalRank,
  hitRate,
  meanAveragePrecision,
  meanReciprocalRank,
  ndcg,
  precisionAtK,
  rPrecision,
  recallAtK,
  type AtKOptions,
  type DcgOptions,
  type Gain,
  type RankingInput,
  type AveragePrecisionOptions,
  type RecallOptions,
} from './ranking'
export {
  adjustedMutualInformation,
  adjustedRandIndex,
  calinskiHarabasz,
  completeness,
  contingencyTable,
  daviesBouldin,
  dunnIndex,
  fowlkesMallows,
  homogeneity,
  homogeneityCompletenessV,
  mutualInformationScore,
  normalisedMutualInformation,
  pairConfusion,
  randIndex,
  silhouetteSamples,
  silhouetteScore,
  vMeasure,
  variationOfInformation,
  type Contingency,
  type EntropyMean,
} from './clustering'
export {
  chiSquareStatistic,
  concordanceCorrelation,
  contingencyCoefficient,
  cramersV,
  fleissKappa,
  intraclassCorrelation,
  kendallCorrelation,
  krippendorffAlpha,
  pearsonCorrelation,
  spearmanCorrelation,
  theilsU,
  tschuprowT,
  type IccForm,
  type MeasurementLevel,
} from './agreement'
export {
  angularDistance,
  chebyshevDistance,
  cosineDistance,
  cosineSimilarity,
  euclideanDistance,
  hausdorffDistance,
  hausdorffDistances,
  mahalanobisDistance,
  manhattanDistance,
  minkowskiDistance,
  orthogonalProcrustes,
  procrustesDisparity,
  type HausdorffDistances,
  type Procrustes,
} from './distances'
export { alignment, uniformity, type AlignmentOptions, type UniformityOptions } from './representation'
export {
  aurocDeLong,
  bootstrapMetric,
  delongTest,
  pairedBootstrap,
  waldInterval,
  wilsonInterval,
  type BootstrapOptions,
  type Cases,
  type MetricBootstrap,
} from './uncertainty'
export { getMetric, isBetter, listMetrics, metricRegistry } from './registry'
export {
  ordinalMeanAbsoluteError,
  macroMeanAbsoluteError,
  withinToleranceAccuracy,
  quadraticWeightedKappa,
  rankedProbabilityScore,
  ordinalConcordanceIndex,
} from './ordinal'

// Helpers for defining metrics outside this module (the application metrics of `aifn-methods/evaluation`): the input
// conventions every metric here follows.
export {
  classesOf,
  compareLabels,
  dense as denseMatrix,
  divide,
  encodeLabels,
  isMatrixLike,
  labelList,
  matrix,
  positiveOf,
  sameLength,
  values as metricValues,
  type Dense,
} from './core'
export { metricsFunctions } from './function-registry'
