/**
 * `aifn-methods/neural/ode-mixtures`: neural ODEs with stochastic vector field mixtures (Twomey, Kozłowski &
 * Santos-Rodríguez, 2020, ECAI), whose vector field (VF) is one of $K$ components, each deterministic or stochastic.
 *
 * - The model: `svfm` (VF and SVF units, $K$ components chosen by pick and stick or forward filtering, the moments
 *   carried over the grid $t_0, \dots, t_T$); its parts `stackedMlp` and `stackedMlpInit` ($K$ MLPs evaluated at once)
 *   and `svfHeads` (an SVF unit's mean direction, length and variances); `flatOf` reads a value out as numbers.
 * - The losses: `svfmObjective` adds the predictive loss and the path regularisers as `checkLossSettings` allows;
 *   `mixtureDensityLoss` and `classMixtureLoss` (MDLoss for targets and for labels), `squaredErrorAt` (the baselines'
 *   loss), `transportLoss` and `varianceLoss` (TLoss and VLoss), and `interpolatePaths` (FLoss's targets on the grid).
 * - Realised paths, with randomness frozen per instance: `realisation` draws it, `realisedRhs` is the VF each instance
 *   follows, `samplePaths` solves the paths, `instanceWork` counts each instance's function evaluations alone and as a
 *   batch, and `realisedVariance` measures how much each instance's VF varies along its path.
 * - Runs: `svfmRun` streams training on a task made by `classificationTask`, `endpointTask` or `walkTask`; `nfeStudy`
 *   trains several models (by default `NFE_STUDY_MODELS`) and measures their work over solver tolerances.
 *
 * The model and its losses are differentiable in the parameters; realised paths and the work of solving them are plain
 * arithmetic. The row-wise adaptive solver is compute's (`dormandPrinceRows` in `aifn-compute/dynamics/ode`); the
 * datasets are `aifn-methods/data` (`odeFailureCase`, `floorplanWalks`). Runs are generators of plain-data snapshots,
 * deterministic in their seed (wall times aside).
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
