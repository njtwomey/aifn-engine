/**
 * `aifn-methods/neural/ode-mixtures`: neural ODEs with stochastic vector field mixtures (Twomey, Kozłowski &
 * Santos-Rodríguez, 2020, ECAI). `svfm` (VF and SVF units, K components, pick and stick or forward filtering, the
 * moments carried over the grid), the losses (`svfmObjective`: MDLoss, TLoss, VLoss, FLoss), realised paths and the
 * per-instance work of solving them (`samplePaths`, `instanceWork`), the streamed run (`svfmRun`) and the
 * forward-evaluation study (`nfeStudy`). The row-wise adaptive solver is compute (`aifn-compute/dynamics/ode`
 * `dormandPrinceRows`); the datasets are `aifn-methods/data` (`odeFailureCase`, `floorplanWalks`).
 */

export {
  stackedMlp,
  stackedMlpInit,
  svfHeads,
  svfm,
  flatOf,
  type ComponentSelection,
  type FieldActivation,
  type Propagation,
  type StackedMlpParams,
  type Svfm,
  type SvfmOptions,
  type SvfmParams,
} from './model'
export {
  checkLossSettings,
  classMixtureLoss,
  interpolatePaths,
  mixtureDensityLoss,
  squaredErrorAt,
  svfmObjective,
  transportLoss,
  varianceLoss,
  type SvfmBatch,
  type SvfmLossParts,
  type SvfmLossSettings,
} from './losses'
export {
  instanceWork,
  realisation,
  realisedRhs,
  realisedVariance,
  samplePaths,
  type InstanceWork,
  type Realisation,
  type RealisationMode,
  type RealisedOptions,
  type RealisedPaths,
} from './sampling'
export { svfmRun, type SvfmCheckpoint, type SvfmRun, type SvfmRunData, type SvfmRunOptions, type SvfmTask } from './run'
export { classificationTask, endpointTask, walkTask, type DatasetLike, type WalkTaskOptions } from './tasks'
export {
  NFE_STUDY_MODELS,
  nfeStudy,
  type NfeStudy,
  type NfeStudyModel,
  type NfeStudyOptions,
  type NfeStudyResult,
} from './efficiency'
export { odeMixtureFunctions } from './registry'
