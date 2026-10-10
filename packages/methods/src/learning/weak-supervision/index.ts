/**
 * `aifn-methods/learning/weak-supervision`: learning without clean labels, from noisy votes, positive and unlabelled
 * data, bag proportions, bag labels or noisy labels.
 *
 * - Label models that combine noisy votes into class posteriors: `majorityVote`, Dawid–Skene EM (`dawidSkene`, or
 *   `dawidSkeneSteps` to step through it; a confusion matrix per voter) and the data-programming label model of Snorkel
 *   (`labelModel`; an accuracy and a coverage per labelling function). `votesOf` reads the vote matrix and
 *   `labelModelReport` runs all three side by side.
 * - Classifiers from weak labels, each a linear model or small MLP: positive–unlabelled learning by Elkan–Noto
 *   calibration (`elkanNoto`, which also estimates the class prior) or the uPU and nnPU risks (`puClassifier`, given
 *   the prior); `proportionClassifier` from bag proportions; `complementaryClassifier` from classes each example is
 *   not.
 * - Learning from label proportions, transductively: label propagation (`lpllp`, `lpllpSteps`, on the graph of
 *   `lpllpGraph`, with $\gamma$ chosen by `lpllpGammaSearch`) and the baselines `inverseCalibration`,
 *   `alterProportionSvm` and `meanMap` (two classes for the first two); the bags themselves from `readBags`,
 *   `bagProportionsOf` and `bagsByProportion`; and the paper's comparison, `llpComparison`.
 * - Active learning with an oracle that answers with proportions: `activeProportionsProblem` sets up the start,
 *   `activeProportionsSteps` and `activeProportionsRun` query by one of `ACTIVE_STRATEGIES`, and
 *   `activeProportionsCurves` compares them.
 * - Multiple-instance learning: `attentionMil` on bags of feature vectors, and MILLET on time series
 *   (`milletModel`, trained by `milletRun`, explained by `milletInterpretation` and scored by `milletScores`, with
 *   `milletLogits`, `milletProbabilities` and the padding `replicatePad`).
 * - Label noise: the noise matrix and likely mislabelled examples by confident learning (`confidentLearning`, with
 *   `classCount`); anchor-point tests for class-conditional noise (`classConditionalNoiseTest`, its power by
 *   `noiseTestPower` and `anchorsForPower`, and `noiseTestSimulation`), on the local likelihood logistic regression of
 *   `localLogistic`, `localLogisticCovariance` and `localLogisticBandwidth` (built on `weightedLogistic` and
 *   `localPolynomialBasis`).
 * - `weakSupervisionAlgorithms` and `weakSupervisionFunctions`: the module's registry entries.
 *
 * Classes are $0, \dots, K - 1$ (two-class methods use 0 and 1), and $-1$ marks an abstaining vote or a point in no
 * bag. Points are the rows of an $n \times d$ matrix. Every fit is deterministic: randomness comes from a `Stream`
 * argument or a `seed` option. Long experiments (`llpComparison`, `activeProportionsCurves`, `noiseTestSimulation`,
 * `milletRun`) are generators that yield partial results, so a figure fills in while a worker computes.
 */

export {
  complementaryClassifier,
  elkanNoto,
  proportionClassifier,
  puClassifier,
  type ElkanNoto,
  type PuClassifierOptions,
  type WeakClassifier,
  type WeakModelOptions,
} from './classifiers'
export { labelModelReport, type DawidSkeneFrame, type LabelModelReport } from './compare'
export {
  dawidSkene,
  dawidSkeneSteps,
  labelModel,
  majorityVote,
  votesOf,
  type DawidSkeneOptions,
  type DawidSkeneState,
  type LabelModel,
  type LabelModelOptions,
  type LabelModelParams,
  type Votes,
} from './label-models'
export {
  alterProportionSvm,
  bagProportionsOf,
  bagsByProportion,
  inverseCalibration,
  llpComparison,
  lpllp,
  lpllpGammaSearch,
  lpllpGraph,
  lpllpSteps,
  meanMap,
  readBags,
  type BagProportions,
  type LlpComparison,
  type LlpComparisonOptions,
  type LlpMethod,
  type LlpPrediction,
  type LpLlpGraph,
  type LpLlpOptions,
  type LpLlpState,
} from './label-proportions'
export {
  ACTIVE_STRATEGIES,
  activeProportionsCurves,
  activeProportionsProblem,
  activeProportionsRun,
  activeProportionsSteps,
  type ActiveCurves,
  type ActiveCurvesOptions,
  type ActiveProportionsOptions,
  type ActiveProportionsProblem,
  type ActiveProportionsState,
  type ActiveStrategy,
} from './active-proportions'
export {
  milletInterpretation,
  milletLogits,
  milletModel,
  milletProbabilities,
  milletRun,
  milletScores,
  replicatePad,
  type MilletCheckpoint,
  type MilletModel,
  type MilletParams,
  type MilletRunOptions,
  type MilletScores,
  type MilletSnapshot,
  type MilletSpec,
  type SeriesSet,
} from './millet'
export { attentionMil, type AttentionMil, type AttentionMilOptions, type AttentionMilParams } from './mil'
export { classCount, confidentLearning, type ConfidentLearning } from './noise'
export {
  localLogistic,
  localLogisticBandwidth,
  localLogisticCovariance,
  localPolynomialBasis,
  weightedLogistic,
  type LocalLogisticFit,
  type LocalLogisticOptions,
} from './local-likelihood'
export {
  anchorsForPower,
  classConditionalNoiseTest,
  noiseTestPower,
  noiseTestSimulation,
  type NoiseTestCell,
  type NoiseTestModel,
  type NoiseTestOptions,
  type NoiseTestResult,
  type NoiseTestSimulation,
} from './noise-tests'
export { weakSupervisionAlgorithms, weakSupervisionFunctions } from './registry'
