/**
 * `aifn-compute/learning/explain`: explanations of a model's predictions.
 *
 * - Local attributions: exact Shapley values, KernelSHAP over an interventional value function (and its sampling
 *   spread), TreeSHAP (one tree or an additive ensemble) and exact tree SHAP interaction values, Shapley interaction
 *   values of any set function, LIME, gradient attributions by autodiff (saliency, gradient × input, integrated
 *   gradients, expected gradients, SmoothGrad), DeepLIFT (rescale) and DeepSHAP through a `DenseNetwork`, occlusion.
 * - Counterfactuals: Wachter et al., DiCE, FACE, Growing Spheres, with actionability constraints.
 * - Rules: anchors (beam search with KL-LUCB), decision lists by sequential covering, tree rules, surrogate fidelity.
 * - Concepts: concept activation vectors and TCAV with its random-concept t-test.
 * - Data: influence functions (exact or LiSSA), TracIn, TMC data Shapley, exact KNN-Shapley.
 * - Global: permutation importance, partial dependence with ICE, accumulated local effects, Friedman's H-statistic,
 *   functional ANOVA.
 * - Evaluation: deletion and insertion curves, faithfulness correlation, cascading model randomisation with rank
 *   similarity (sanity checks), local Lipschitz stability.
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
