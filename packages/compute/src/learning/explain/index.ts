/**
 * `aifn-compute/learning/explain`: explanations of a model's predictions.
 *
 * - Shapley attributions: `exactShapley` and `shapleyInteractions` of any set function (by enumeration),
 *   `interventionalValue` to turn a model into one, `kernelShap` (exact when it enumerates, sampled otherwise, with
 *   `shapleyKernelWeight`) and `kernelShapVariance` for its sampling spread; for trees, `treeShap`,
 *   `treeShapInteractions`, `ensembleTreeShap`, `ensembleTreeShapInteractions`, with `pathDependentValue`,
 *   `expectedValue` and `treeEnsembleOutput`.
 * - Other local attributions: `lime`; gradients by autodiff, `inputGradient` (saliency and gradient times input),
 *   `integratedGradients`, `expectedGradients` and `smoothGrad`; `deepLift` and `deepShap` through a `DenseNetwork`;
 *   `occlusion` of windows.
 * - Networks as data, for the methods that look inside one: `denseLayers`, `denseForward`, `denseFunction`,
 *   `denseOutput`, `fromMlpParams`, `activate`, `activateDerivative`.
 * - Counterfactuals: `wachterCounterfactual` (one, by gradient), `diverseCounterfactuals` (DiCE, several at once),
 *   `face` (`faceGraph` and `faceSearch`: an actual data point reached through dense regions), `growingSpheres`
 *   (model-agnostic, by sampling); `isActionable`, `projectActionable` and `medianAbsoluteDeviation` for the
 *   constraints and the distance.
 * - Rules: `anchor` (with `klLucb`, `klBounds`, `bernoulliKl`), `ruleList` and `applyRuleList` (sequential covering),
 *   `treeRules`, `fidelity` of a surrogate; predicates with `satisfies`, `ruleCoverage`, `simplifyRule`,
 *   `quantileEdges` and `binPredicates`.
 * - Concepts: `tcav` with its random-concept t-test, built from `linearProbe`, `conceptActivationVector`,
 *   `activationGradients`, `conceptSensitivity` and `tcavScore`.
 * - Data: `influenceFunctions` (exact or LiSSA) on `exampleGradients`, `tracIn`, `dataShapley` (truncated Monte Carlo)
 *   and `knnShapley` (exact).
 * - Global: `permutationImportance`, `partialDependence` with ICE on a `featureGrid`, `accumulatedLocalEffects`
 *   (faithful under correlated features), Friedman's `hStatistic`, `functionalAnova`.
 * - Evaluation: `deletionCurve` (deletion and insertion) in `attributionOrder`, `perturbationCurve`, `aopc` and `aopcr`
 *   for a masked predictor, `faithfulnessCorrelation`, `randomiseNetwork` with `explanationSimilarity` (sanity
 *   checks), `localLipschitz` stability, `relevanceMass` against a ground-truth mask.
 *
 * Models are plain functions: a batch of rows ($m \times d$) to one output per row (`ScalarModel`), a differentiable
 * function of one input (`Differentiable`), or a classifier returning a label per row. Inputs are numbers, nested
 * arrays or tensors; results are plain arrays and records. Every random method takes a `Stream`, so a seed fixes its
 * result. `explainFunctions` is the module's registry.
 */

export {
  exactShapley,
  interventionalValue,
  kernelShap,
  shapleyKernelWeight,
  type Attribution,
  type KernelShapOptions,
  type ScalarModel,
} from './shapley'
export {
  ensembleTreeShap,
  expectedValue,
  pathDependentValue,
  ensembleTreeShapInteractions,
  treeEnsembleOutput,
  treeShap,
  treeShapInteractions,
  type ShapNode,
  type ShapTree,
} from './tree'
export { lime, type LimeExplanation, type LimeOptions } from './lime'
export { inputGradient, integratedGradients, smoothGrad, type Differentiable, type PathRule } from './gradients'
export { featureGrid, partialDependence, permutationImportance, type PermutationImportanceOptions } from './importance'
export {
  aopc,
  aopcr,
  perturbationCurve,
  type Aopcr,
  type MaskedPredictor,
  type PerturbationCurve,
  type PerturbationOptions,
} from './perturbation'
export {
  activate,
  activateDerivative,
  denseForward,
  denseFunction,
  denseLayers,
  denseOutput,
  fromMlpParams,
  type DenseActivation,
  type DenseNetwork,
} from './network'
export {
  deepLift,
  deepShap,
  expectedGradients,
  kernelShapVariance,
  occlusion,
  shapleyInteractions,
  type OcclusionOptions,
} from './attribution'
export {
  diverseCounterfactuals,
  face,
  faceGraph,
  faceSearch,
  growingSpheres,
  isActionable,
  medianAbsoluteDeviation,
  projectActionable,
  wachterCounterfactual,
  type Actionability,
  type DiverseOptions,
  type FaceGraph,
  type FaceGraphOptions,
  type FaceResult,
  type FaceSearchOptions,
  type GrowingSpheresOptions,
  type WachterOptions,
} from './counterfactual'
export {
  anchor,
  applyRuleList,
  bernoulliKl,
  binPredicates,
  fidelity,
  klBounds,
  klLucb,
  quantileEdges,
  ruleCoverage,
  ruleList,
  satisfies,
  simplifyRule,
  treeRules,
  type AnchorCandidate,
  type AnchorOptions,
  type AnchorResult,
  type ArmStats,
  type ListRule,
  type Predicate,
  type RuleList,
  type RuleListOptions,
} from './rules'
export {
  activationGradients,
  conceptActivationVector,
  conceptSensitivity,
  linearProbe,
  tcav,
  tcavScore,
  type LinearProbe,
  type TcavResult,
} from './concepts'
export {
  dataShapley,
  exampleGradients,
  influenceFunctions,
  knnShapley,
  tracIn,
  type DataShapleyOptions,
  type ExampleLoss,
  type Examples,
  type InfluenceOptions,
} from './data'
export { accumulatedLocalEffects, functionalAnova, hStatistic, type FunctionalAnova } from './global'
export {
  attributionOrder,
  deletionCurve,
  explanationSimilarity,
  faithfulnessCorrelation,
  localLipschitz,
  randomiseNetwork,
  relevanceMass,
} from './evaluation'
export { explainFunctions } from './registry'
