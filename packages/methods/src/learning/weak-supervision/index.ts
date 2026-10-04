/**
 * `aifn-methods/learning/weak-supervision`: learning without clean labels. Label models that combine noisy votes
 * (majority vote, Dawid–Skene EM, the data-programming label model of Snorkel); classifiers from positive and
 * unlabelled data (Elkan–Noto, uPU, nnPU), from label proportions and from complementary labels; attention-based
 * multiple-instance learning; and noise-rate estimation by confident learning.
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
