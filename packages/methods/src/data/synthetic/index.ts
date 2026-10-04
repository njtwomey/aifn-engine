/**
 * `aifn-methods/data/synthetic`: seeded synthetic datasets: points (blobs, moons, circles, spirals, …), regression,
 * 2-d densities for generative models (ring and grid of Gaussians, pinwheel, Swiss-roll slice, annulus), inverse
 * problems with multi-valued answers (Bishop's folded sine, two-link arm kinematics),
 * sequences, piecewise series with known changepoints, learners with zero-inflated responses, images, paired views
 * (image and caption) and recommender interactions; and modifiers (noise, outliers, shifts, missingness, linear maps).
 * Recipes, which replay generators and modifiers by key, are in `aifn-methods/data`.
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
