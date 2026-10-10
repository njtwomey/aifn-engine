/**
 * `aifn-methods/data/synthetic`: seeded synthetic datasets with known ground truth, and the modifiers that change them.
 *
 * - Point clouds for classification and clustering: `blobs`, `anisotropicBlobs`, `gaussians`, `moons`, `circles`,
 *   `rings`, `spirals`, `xor`, `checkerboard` and `halfKernel`, the 3-d manifolds `swissRoll` and `sCurve`, and
 *   `shuffleDataset` (generators group points by class).
 * - Two-dimensional densities for generative models, each with its exact density: `gaussianRing`, `gaussianGrid`,
 *   `pinwheel`, `swissRoll2d`, and `annulus` for out-of-distribution points.
 * - Regression: `regression1d`, `linearRegressionData`, `friedman1`, additive models (`additiveData`,
 *   `ADDITIVE_SHAPES`), smooth 1-d laws for GAMs and expectiles (`curve1d`, `CURVE1D_CASES`), a growth chart
 *   (`growthChart`, `GROWTH_TRUTH`), mixtures of experts (`piecewiseLinear`, `quadrantPlanes`, `quadrantOf`,
 *   `interleavedFunctions`, `regressionMixture`) and inverse problems with several answers (`bishopInverse`,
 *   `twoLinkArm`, `twoLinkInverse`, `twoLinkJoints`).
 * - Sequences and series: hidden Markov models (`hmmSample`, `casino`), `arSeries`, `seasonalSeries`, `randomWalk`,
 *   `motifSeries`; series with known changepoints (`meanShifts`, `varianceShifts`, `poissonShifts`, `arRegimes`); named
 *   Markov chains (`weatherChain`, `gamblersRuinChain`, `randomWalkChain`, `ehrenfestChain`); flows that split and
 *   cross (`odeFailureCase`, `floorplanWalks`, `FLOORPLAN`); and labelled web traffic (`webTraffic`,
 *   `WEB_TRAFFIC_CLASSES`).
 * - Recommendation, decisions and ratings: `zipfWeights`, `zipfCatalogue`, `ratings`, `clickLog`, `implicitFeedback`,
 *   `simulatedImpressions`; bandit logs and expert games (`banditProblem`, `logBandit`, `loggedBandit`, `slateBandit`,
 *   `expertGame`); skill rating (`tournament`, `plackettLuceRankings`, `irtResponses`, `ratingPopulation`,
 *   `ratingMatches`, `skillPath`, `winProbability`, `focalPlayerStream`, `focalPlayerMatches`); and learners'
 *   responses (`learnerResponses`, `LEARNER_CONDITIONS`, `LEARNER_ITEM_FEATURES`, `LEARNER_UNSUITABILITY`).
 * - Weak and noisy supervision: `labellingFunctions`, `crowdLabels`, `positiveUnlabelled`, `proportionBags`,
 *   `instanceBags`, `complementaryLabels`; class-conditional label noise with its anchor points
 *   (`classConditionalNoise`, `classConditionalNoiseTruth`, `noisyPosterior`, `noiseLayoutAnchors`,
 *   `noiseLayoutPosterior`); and shifted domains and task sequences (`shiftedMoons`, `labelShiftDomains`,
 *   `rotatingTasks`).
 * - Evaluation and explanation test beds: model outputs to calibrate (`classifierOutputs`, `quantileModelOutputs`),
 *   censored survival (`censoredSurvival`, `weibullPhSurvival`), planted anomalies (`plantedAnomalies`,
 *   `ANOMALY_SHAPES`), planted subgroups (`plantedSubgroups`, `plantedModelFlip` and the `PLANTED_*` patterns), and
 *   tasks with known explanations (`attributionTask`, `attributionScore`, `feasibilityTask`, `correlatedEffects`,
 *   `correlatedEffectsTerm`, `correlatedEffectsTruth`, `interactionTask`, `interactionTaskTruth`, `plantedPatterns`,
 *   `plantedMask`, `plantedShape`, `conceptExamples`, `conceptImages`, `CONCEPTS`).
 * - Images, latent factors and paired views: test images (`checkerboardImage`, `gradientImage`, `shapesImage`,
 *   `geometricScene`, `digits`, `digitGlyphs`, `barsAndStripes`), sources and factors (`cocktailParty`,
 *   `strokeGlyphs`, `glyphStrokes`, `latentFactors`, `latentFactorModel`), and image and caption pairs
 *   (`pairedShapes`, `PAIRED_COLOURS`, `PAIRED_RGB`, `PAIRED_SHAPES`, `PAIRED_SIZES`).
 * - Algorithmic tasks for small transformers and grokking: `sequenceTasks` (`SEQUENCE_TASKS`, `SEQUENCE_VOCABULARY`,
 *   `encodeSequence`, `decodeSequence`, `sequenceTaskTruth`) and $a \circ b \bmod p$ tables (`modularArithmetic`,
 *   `MODULAR_OPERATIONS`, `modularValue`, `modularTruth`).
 * - Modifiers, from one `Dataset` to another: label noise (`withLabelNoise`, `symmetricNoise`, `flippedMask`), class
 *   balance (`withPrevalence`, `withLabelShift`), `withOutliers`, `withNuisanceFeatures`, linear maps (`withTransform`,
 *   `rotation2d`, `shear2d`), `withMissing`, `withCovariateShift`, and `split` into train and test parts.
 *
 * Every random generator takes a `Stream` first and is a pure function of it, drawing its parts from named child
 * streams. Labelled point generators take class sizes alike (a total with `prevalence` or `classWeights`, or per-class
 * counts) and give exact counts. Where the generating process is known in closed form, the truth is attached in
 * `meta.truth` (the Bayes posterior and error, the regression function, the segments) or returned beside the data, and
 * the modifiers keep it consistent. Generators and modifiers are registered by key and record themselves in
 * `meta.recipe`; recipes, which replay them, are in `aifn-methods/data`.
 */

export {
  anisotropicBlobs,
  blobs,
  checkerboard,
  circles,
  gaussians,
  moons,
  rings,
  sCurve,
  shuffleDataset,
  spirals,
  swissRoll,
  xor,
  type AnisotropicOptions,
  type BlobsOptions,
  type CheckerboardOptions,
  type CirclesOptions,
  type GaussiansOptions,
  type ManifoldOptions,
  type MoonsOptions,
  type RingsOptions,
  type SpiralsOptions,
  type XorOptions,
} from './points'
export {
  annulus,
  gaussianGrid,
  gaussianRing,
  pinwheel,
  swissRoll2d,
  type AnnulusOptions,
  type GaussianGridOptions,
  type GaussianRingOptions,
  type PinwheelOptions,
  type SwissRoll2dOptions,
} from './densities'
export {
  ADDITIVE_SHAPES,
  additiveData,
  type AdditiveFamily,
  type AdditiveOptions,
  type AdditiveShape,
} from './additive'
export { CURVE1D_CASES, curve1d, type Curve1dCase, type Curve1dOptions } from './curves'
export { GROWTH_TRUTH, growthChart, type GrowthChartOptions } from './growth'
export {
  friedman1,
  linearRegressionData,
  regression1d,
  type LinearRegressionOptions,
  type Regression1dOptions,
  type RegressionFunction,
} from './regression'
export {
  arSeries,
  casino,
  hmmSample,
  motifSeries,
  randomWalk,
  seasonalSeries,
  type ArOptions,
  type CasinoOptions,
  type DiscreteHmm,
  type HmmSample,
  type MotifSeries,
  type MotifSeriesOptions,
  type SeasonalOptions,
  type TimeSeries,
} from './sequences'
export {
  arRegimes,
  meanShifts,
  poissonShifts,
  varianceShifts,
  type ArRegimeOptions,
  type MeanShiftOptions,
  type PoissonShiftOptions,
  type SegmentOptions,
  type VarianceShiftOptions,
} from './changepoints'
export {
  clickLog,
  implicitFeedback,
  ratings,
  zipfCatalogue,
  zipfWeights,
  type ClickLog,
  type ClickLogOptions,
  type ImplicitFeedback,
  type ImplicitFeedbackOptions,
  type Ratings,
  type RatingsOptions,
  type ZipfCatalogue,
} from './recsys'
export {
  banditProblem,
  expertGame,
  logBandit,
  loggedBandit,
  slateBandit,
  type BanditLogDraw,
  type BanditOptions,
  type BanditProblem,
  type ExpertGame,
  type ExpertGameOptions,
  type LoggedBandit,
  type SlateBandit,
  type SlateBanditOptions,
} from './bandits'
export {
  classifierOutputs,
  quantileModelOutputs,
  type ClassifierOutputs,
  type ClassifierOutputsOptions,
  type QuantileModelOptions,
  type QuantileModelOutputs,
} from './predictions'
export { ehrenfestChain, gamblersRuinChain, randomWalkChain, weatherChain, type NamedChain } from './chains'
export {
  interleavedFunctions,
  piecewiseLinear,
  quadrantOf,
  quadrantPlanes,
  regressionMixture,
  type InterleavedOptions,
  type PiecewiseLinearOptions,
  type QuadrantPlanesOptions,
  type RegimeDataset,
  type RegressionMixtureOptions,
} from './regimes'
export {
  bishopInverse,
  twoLinkArm,
  twoLinkInverse,
  twoLinkJoints,
  type BishopInverseOptions,
  type TwoLinkArmOptions,
} from './inverse'
export {
  barsAndStripes,
  checkerboardImage,
  digitGlyphs,
  digits,
  geometricScene,
  gradientImage,
  shapesImage,
  type GeometricScene,
} from './images'
export { attributionScore, attributionTask } from './attribution'
export {
  CONCEPTS,
  conceptExamples,
  conceptImages,
  correlatedEffects,
  correlatedEffectsTerm,
  correlatedEffectsTruth,
  feasibilityTask,
  interactionTask,
  interactionTaskTruth,
  plantedMask,
  plantedPatterns,
  plantedShape,
  type Concept,
  type ConceptRule,
} from './explanation'
export {
  cocktailParty,
  glyphStrokes,
  latentFactorModel,
  latentFactors,
  strokeGlyphs,
  type CocktailParty,
  type LatentFactorModel,
} from './factors'
export {
  PAIRED_COLOURS,
  PAIRED_RGB,
  PAIRED_SHAPES,
  PAIRED_SIZES,
  pairedShapes,
  type Attribute,
  type PairTruth,
  type PairedShapesOptions,
  type PairedViews,
  type View,
} from './paired'
export {
  flippedMask,
  rotation2d,
  shear2d,
  split,
  symmetricNoise,
  withCovariateShift,
  withLabelNoise,
  withLabelShift,
  withMissing,
  withNuisanceFeatures,
  withOutliers,
  withPrevalence,
  withTransform,
  type CovariateShiftOptions,
  type LabelNoiseOptions,
  type MissingMechanism,
  type MissingOptions,
  type NuisanceOptions,
  type OutlierOptions,
  type PrevalenceOptions,
  type SplitOptions,
  type TransformOptions,
} from './modifiers'
export {
  decodeSequence,
  encodeSequence,
  MODULAR_OPERATIONS,
  modularArithmetic,
  modularTruth,
  modularValue,
  SEQUENCE_TASKS,
  SEQUENCE_VOCABULARY,
  sequenceTasks,
  sequenceTaskTruth,
  type ModularArithmeticData,
  type ModularArithmeticOptions,
  type ModularOperation,
  type ModularPart,
  type ModularTruth,
  type SequenceExamples,
  type SequenceScore,
  type SequenceTaskData,
  type SequenceTaskName,
  type SequenceTaskOptions,
  type SequenceTaskTruth,
} from './algorithmic'
export {
  FLOORPLAN,
  floorplanWalks,
  odeFailureCase,
  type FloorPoint,
  type Floorplan,
  type FloorplanWalksOptions,
  type OdeFailureKind,
  type OdeFailureOptions,
} from './walks'
export { irtResponses, plackettLuceRankings, tournament, type Tournament, type TournamentOptions } from './ratings'
export {
  focalPlayerMatches,
  focalPlayerStream,
  ratingMatches,
  ratingPopulation,
  skillPath,
  winProbability,
  type FocalPlayerOptions,
  type FocalPlayerStream,
  type OutcomeOptions,
  type RatingPopulation,
  type RatingPopulationOptions,
  type SkillPath,
} from './rating-streams'
export {
  LEARNER_CONDITIONS,
  LEARNER_ITEM_FEATURES,
  LEARNER_UNSUITABILITY,
  learnerResponses,
  type LearnerCondition,
  type LearnerResponses,
  type LearnerResponsesOptions,
} from './learners'
export { censoredSurvival, weibullPhSurvival, type CensoredSurvival, type CensoredSurvivalOptions } from './survival'
export { ANOMALY_SHAPES, plantedAnomalies, type AnomalyShape, type PlantedAnomaliesOptions } from './anomalies'
export {
  complementaryLabels,
  crowdLabels,
  instanceBags,
  labellingFunctions,
  positiveUnlabelled,
  proportionBags,
  type CrowdLabelOptions,
  type CrowdLabels,
  type InstanceBags,
  type LabellingFunctionOptions,
  type LabellingFunctions,
  type PositiveUnlabelled,
  type ProportionBags,
} from './weak'
export { halfKernel, type HalfKernelOptions } from './half-kernel'
export {
  WEB_TRAFFIC_CLASSES,
  webTraffic,
  type WebTrafficClass,
  type WebTrafficOptions,
  type WebTrafficSample,
} from './web-traffic'
export {
  classConditionalNoise,
  classConditionalNoiseTruth,
  noiseLayoutAnchors,
  noiseLayoutPosterior,
  noisyPosterior,
  type NoiseLayout,
  type ClassConditionalNoiseOptions,
  type ClassConditionalNoiseSample,
} from './label-noise'
export {
  labelShiftDomains,
  rotatingTasks,
  shiftedMoons,
  type DomainPair,
  type LabelShiftOptions,
  type ShiftedMoonsOptions,
} from './shift'
export { simulatedImpressions } from './impressions'
export {
  PLANTED_CLASSIFIER,
  PLANTED_CORRELATION,
  PLANTED_MAIN,
  PLANTED_SECONDARY,
  plantedModelFlip,
  plantedSubgroups,
  type PlantedModelFlipOptions,
  type PlantedSubgroupsOptions,
} from './subgroups'
