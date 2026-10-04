/**
 * The registry of `aifn-compute/learning/explain`: attributions (Shapley values, KernelSHAP, TreeSHAP and interaction values,
 * LIME, gradients, DeepLIFT, occlusion), counterfactuals, rules, concepts (TCAV), data attribution, global effects and
 * the evaluation of explanations.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as attribution from './attribution'
import * as concepts from './concepts'
import * as counterfactual from './counterfactual'
import * as data from './data'
import * as evaluation from './evaluation'
import * as global from './global'
import * as gradients from './gradients'
import * as network from './network'
import * as rules from './rules'
import * as importance from './importance'
import * as lime from './lime'
import * as perturbation from './perturbation'
import * as shapley from './shapley'
import * as tree from './tree'

const fn = definer<FunctionInfo>('function', 'learning/explain')
const ATTRIBUTION = ['feature-attribution-methods', 'interpretability']
const TREES = ['shapley-additive-explanations-for-trees', ...ATTRIBUTION]
const COUNTERFACTUAL = ['counterfactual-explanations', 'interpretability']
const RULES = ['intrinsically-interpretable-models', 'interpretability']
const CONCEPTS = ['concept-based-explanations', 'interpretability']
const DATA = ['leverage-and-influence', 'detecting-and-cleaning-label-errors', 'interpretability']
const GLOBAL = ['interpretability', 'interpreting-generalised-additive-models']
const EVALUATION = ['evaluating-explanations', 'feature-attribution-methods']
const NETWORK = ['feature-attribution-methods', 'concept-based-explanations']

fn(
  {
    key: 'exactShapley',
    name: 'Exact Shapley values',
    summary: 'Each player’s marginal contribution averaged over all coalitions, by enumerating the 2^d of them.',
    role: 'estimator',
    notes: ATTRIBUTION,
    cite: ['lundberg2017'],
  },
  shapley.exactShapley,
)
fn(
  {
    key: 'shapleyKernelWeight',
    name: 'Shapley kernel weight',
    summary: 'The weight (d − 1)/(C(d, s) s (d − s)) under which weighted least squares recovers Shapley values.',
    role: 'property',
    notes: ATTRIBUTION,
    cite: ['lundberg2017'],
  },
  shapley.shapleyKernelWeight,
)
fn(
  {
    key: 'interventionalValue',
    name: 'Interventional value function',
    summary: 'Mean output with the coalition’s features from x and the rest from background rows.',
    role: 'construction',
    notes: ATTRIBUTION,
    cite: ['lundberg2017'],
  },
  shapley.interventionalValue,
)
fn(
  {
    key: 'kernelShap',
    name: 'KernelSHAP',
    summary: 'Shapley values as a kernel-weighted linear regression over coalitions, sampled or enumerated.',
    role: 'estimator',
    random: true,
    notes: ATTRIBUTION,
    cite: ['lundberg2017'],
  },
  shapley.kernelShap,
)
fn(
  {
    key: 'treeShap',
    name: 'TreeSHAP',
    summary: 'Exact path-dependent Shapley values of a tree in polynomial time, one pass over root-to-leaf paths.',
    role: 'estimator',
    notes: TREES,
    cite: ['lundberg2020'],
  },
  tree.treeShap,
)
fn(
  {
    key: 'ensembleTreeShap',
    name: 'TreeSHAP for ensembles',
    summary: 'The scaled sum of each tree’s TreeSHAP values, for forests and boosted trees.',
    role: 'estimator',
    notes: TREES,
    cite: ['lundberg2020'],
  },
  tree.ensembleTreeShap,
)
fn(
  {
    key: 'pathDependentValue',
    name: 'Path-dependent value function',
    summary: 'E[f(x) | x_S] for a tree, averaging unknown splits over both branches by cover.',
    role: 'construction',
    notes: TREES,
    cite: ['lundberg2020'],
  },
  tree.pathDependentValue,
)
fn(
  {
    key: 'expectedValue',
    name: 'Tree expected value',
    summary: 'The cover-weighted mean leaf output: the base value of TreeSHAP.',
    role: 'property',
    notes: TREES,
  },
  tree.expectedValue,
)
fn(
  {
    key: 'treeEnsembleOutput',
    name: 'Tree ensemble output',
    summary: 'offset + Σ scale_t f_t(x) over trees: the additive function TreeSHAP for ensembles explains.',
    role: 'transform',
    notes: TREES,
  },
  tree.treeEnsembleOutput,
)
fn(
  {
    key: 'lime',
    name: 'LIME (tabular)',
    summary: 'A kernel-weighted ridge surrogate fitted to the model on Gaussian perturbations of the instance.',
    role: 'estimator',
    random: true,
    notes: ATTRIBUTION,
    cite: ['ribeiro2016'],
  },
  lime.lime,
)
fn(
  {
    key: 'inputGradient',
    name: 'Saliency (input gradient)',
    summary: 'The gradient of the output with respect to the input, and gradient × input.',
    role: 'estimator',
    notes: ATTRIBUTION,
    cite: ['simonyan2014'],
  },
  gradients.inputGradient,
)
fn(
  {
    key: 'integratedGradients',
    name: 'Integrated gradients',
    summary: 'Gradients integrated along the straight path from a baseline; the attributions sum to f(x) − f(x′).',
    role: 'estimator',
    notes: ATTRIBUTION,
    cite: ['sundararajan2017'],
  },
  gradients.integratedGradients,
)
fn(
  {
    key: 'smoothGrad',
    name: 'SmoothGrad',
    summary: 'The gradient averaged over Gaussian-noised copies of the input.',
    role: 'estimator',
    random: true,
    notes: ATTRIBUTION,
    cite: ['smilkov2017'],
  },
  gradients.smoothGrad,
)
fn(
  {
    key: 'permutationImportance',
    name: 'Permutation importance',
    summary: 'The drop in score when one feature’s column is shuffled, over repeats.',
    role: 'estimator',
    random: true,
    notes: ['interpretability'],
    cite: ['breiman2001', 'molnar2022'],
  },
  importance.permutationImportance,
)
fn(
  {
    key: 'partialDependence',
    name: 'Partial dependence and ICE',
    summary: 'Predictions with one feature swept over a grid for every row (ICE) and their mean (partial dependence).',
    role: 'estimator',
    notes: ['interpretability', 'interpreting-generalised-additive-models'],
    cite: ['friedman2001', 'goldstein2015'],
  },
  importance.partialDependence,
)
fn(
  { key: 'featureGrid', name: 'Feature grid', role: 'construction', notes: ['interpretability'] },
  importance.featureGrid,
)

fn(
  {
    key: 'perturbationCurve',
    name: 'Perturbation curve (MoRF)',
    summary: 'The prediction as the positions an explanation ranks first are removed, block by block.',
    role: 'estimator',
    notes: ['interpretability', 'millet'],
    cite: ['samek2017', 'early2024'],
  },
  perturbation.perturbationCurve,
)
fn(
  {
    key: 'aopc',
    name: 'Area over the perturbation curve',
    tex: '\\mathrm{AOPC} = \\frac{1}{J}\\sum_{j=1}^{J} f(X) - f(\\mathrm{MoRF}_j)',
    summary: 'The mean drop of the prediction along a perturbation curve: larger for a more faithful ordering.',
    role: 'estimator',
    notes: ['interpretability', 'millet'],
    cite: ['samek2017'],
  },
  perturbation.aopc,
)
fn(
  {
    key: 'aopcr',
    name: 'AOPC relative to random orderings',
    summary: 'AOPC of an importance ordering minus the mean AOPC of random orderings: zero is no better than chance.',
    role: 'estimator',
    random: true,
    notes: ['interpretability', 'millet'],
    cite: ['early2024'],
  },
  perturbation.aopcr,
)

// ── More attributions ───────────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'occlusion',
    name: 'Occlusion',
    summary:
      'Slide a baseline window over the input; credit each feature with the mean output drop of windows covering it.',
    role: 'estimator',
    notes: ATTRIBUTION,
  },
  attribution.occlusion,
)
fn(
  {
    key: 'deepLift',
    name: 'DeepLIFT (rescale rule)',
    summary: 'Multipliers Δa/Δz through each nonlinearity against a reference; the attributions sum to f(x) − f(x′).',
    role: 'estimator',
    notes: ATTRIBUTION,
    cite: ['shrikumar2017'],
  },
  attribution.deepLift,
)
fn(
  {
    key: 'deepShap',
    name: 'DeepSHAP',
    summary: 'DeepLIFT attributions averaged over background rows as references.',
    role: 'estimator',
    notes: ATTRIBUTION,
    cite: ['lundberg2017', 'shrikumar2017'],
  },
  attribution.deepShap,
)
fn(
  {
    key: 'expectedGradients',
    name: 'Expected gradients',
    summary: 'Integrated gradients averaged over baselines drawn from the data, at random points along each path.',
    role: 'estimator',
    random: true,
    notes: ATTRIBUTION,
    cite: ['sundararajan2017'],
  },
  attribution.expectedGradients,
)
fn(
  {
    key: 'shapleyInteractions',
    name: 'Shapley interaction values',
    summary:
      'Pairwise Shapley interaction values Φᵢⱼ of a set function by enumeration; rows sum to the Shapley values.',
    role: 'estimator',
    notes: TREES,
    cite: ['lundberg2020'],
  },
  attribution.shapleyInteractions,
)
fn(
  {
    key: 'kernelShapVariance',
    name: 'KernelSHAP sampling spread',
    summary: 'KernelSHAP repeated on independent streams: the mean and standard deviation of each estimate.',
    role: 'estimator',
    random: true,
    notes: ATTRIBUTION,
    cite: ['lundberg2017'],
  },
  attribution.kernelShapVariance,
)
fn(
  {
    key: 'treeShapInteractions',
    name: 'TreeSHAP interaction values',
    summary: 'Exact SHAP interaction values of a tree from TreeSHAP passes with one feature held in or out.',
    role: 'estimator',
    notes: TREES,
    cite: ['lundberg2020'],
  },
  tree.treeShapInteractions,
)
fn(
  {
    key: 'ensembleTreeShapInteractions',
    name: 'TreeSHAP interaction values for ensembles',
    summary: 'The scaled sum of each tree’s SHAP interaction values.',
    role: 'estimator',
    notes: TREES,
    cite: ['lundberg2020'],
  },
  tree.ensembleTreeShapInteractions,
)

// ── Networks as data ─────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'denseForward',
    name: 'Dense network forward pass',
    summary: 'Pre-activations and activations of every layer of a feedforward network given as weights and biases.',
    role: 'transform',
    notes: NETWORK,
  },
  network.denseForward,
)
fn(
  {
    key: 'denseFunction',
    name: 'Dense network as a function',
    summary: 'The network from a hidden layer on, as a differentiable function of that layer’s activations.',
    role: 'construction',
    notes: NETWORK,
  },
  network.denseFunction,
)
fn({ key: 'denseOutput', name: 'Dense network output', role: 'transform', notes: NETWORK }, network.denseOutput)
fn({ key: 'denseLayers', name: 'Dense network layers', role: 'construction', notes: NETWORK }, network.denseLayers)
fn(
  {
    key: 'fromMlpParams',
    name: 'Dense network from MLP parameters',
    summary: 'Read the parameter list of an aifn-compute/nn Mlp as weights and biases.',
    role: 'construction',
    notes: NETWORK,
  },
  network.fromMlpParams,
)
fn({ key: 'activate', name: 'Dense activation', role: 'transform', notes: NETWORK }, network.activate)
fn(
  { key: 'activateDerivative', name: 'Dense activation derivative', role: 'transform', notes: NETWORK },
  network.activateDerivative,
)

// ── Counterfactuals ──────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'wachterCounterfactual',
    name: 'Wachter counterfactual',
    summary: 'Minimise λ(f(x′) − y′)² + MAD-scaled L1 distance by gradient steps, raising λ until the target is met.',
    role: 'solver',
    notes: COUNTERFACTUAL,
    cite: ['wachter2017'],
  },
  counterfactual.wachterCounterfactual,
)
fn(
  {
    key: 'diverseCounterfactuals',
    name: 'DiCE diverse counterfactuals',
    summary: 'k counterfactuals at once: hinge validity, proximity and a determinantal diversity term.',
    role: 'solver',
    random: true,
    notes: COUNTERFACTUAL,
    cite: ['mothilal2020'],
  },
  counterfactual.diverseCounterfactuals,
)
fn(
  {
    key: 'faceGraph',
    name: 'FACE graph',
    summary: 'An ε-, kNN- or KDE-weighted graph over the data whose edge weights f(p̂)‖xᵢ − xⱼ‖ favour dense regions.',
    role: 'construction',
    notes: COUNTERFACTUAL,
  },
  counterfactual.faceGraph,
)
fn(
  {
    key: 'faceSearch',
    name: 'FACE search',
    summary: 'The cheapest data point predicted as the target with enough density, by Dijkstra over a FACE graph.',
    role: 'solver',
    notes: COUNTERFACTUAL,
  },
  counterfactual.faceSearch,
)
fn(
  {
    key: 'face',
    name: 'FACE counterfactual',
    summary: 'A feasible, actionable counterfactual: a real data point reached by short steps through dense regions.',
    role: 'solver',
    notes: COUNTERFACTUAL,
  },
  counterfactual.face,
)
fn(
  {
    key: 'growingSpheres',
    name: 'Growing Spheres',
    summary: 'The closest enemy found in growing spherical shells around x, then sparsified.',
    role: 'solver',
    random: true,
    notes: COUNTERFACTUAL,
  },
  counterfactual.growingSpheres,
)
fn(
  {
    key: 'isActionable',
    name: 'Actionability test',
    summary: 'Whether a change keeps immutable features, monotone directions and bounds.',
    role: 'property',
    notes: COUNTERFACTUAL,
    cite: ['ustun2019'],
  },
  counterfactual.isActionable,
)
fn(
  {
    key: 'projectActionable',
    name: 'Actionable projection',
    summary: 'Map a candidate onto the allowed changes: reset immutable features, clip monotone ones and bounds.',
    role: 'transform',
    notes: COUNTERFACTUAL,
    cite: ['ustun2019'],
  },
  counterfactual.projectActionable,
)
fn(
  {
    key: 'medianAbsoluteDeviation',
    name: 'Median absolute deviation',
    summary: 'Per-column med |x − med x|, the feature scale of Wachter’s and DiCE’s distances.',
    role: 'property',
    notes: COUNTERFACTUAL,
    cite: ['wachter2017'],
  },
  counterfactual.medianAbsoluteDeviation,
)

// ── Rules ────────────────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'anchor',
    name: 'Anchors',
    summary: 'A rule holding at x under which the prediction stays with precision ≥ τ, by beam search and KL-LUCB.',
    role: 'estimator',
    random: true,
    notes: RULES,
    cite: ['ribeiro2016'],
  },
  rules.anchor,
)
fn(
  {
    key: 'klLucb',
    name: 'KL-LUCB',
    summary: 'Best-arm identification of the top arms by sampling the two whose KL confidence bounds overlap most.',
    role: 'solver',
    notes: ['multi-armed-bandit', ...RULES],
    cite: ['kaufmann2016', 'garivier2011'],
  },
  rules.klLucb,
)
fn(
  {
    key: 'klBounds',
    name: 'KL confidence bounds',
    summary: 'Lower and upper q with KL(p̂ ‖ q) equal to a level, by bisection.',
    role: 'property',
    notes: ['multi-armed-bandit'],
    cite: ['garivier2011'],
  },
  rules.klBounds,
)
fn(
  { key: 'bernoulliKl', name: 'Bernoulli KL divergence', role: 'property', notes: ['multi-armed-bandit'] },
  rules.bernoulliKl,
)
fn(
  {
    key: 'ruleList',
    name: 'Decision list by sequential covering',
    summary:
      'Repeatedly the purest conjunction on uncovered rows (beam search, Laplace estimate), then a default class.',
    role: 'fit',
    notes: RULES,
    cite: ['letham2015'],
  },
  rules.ruleList,
)
fn({ key: 'applyRuleList', name: 'Apply a decision list', role: 'transform', notes: RULES }, rules.applyRuleList)
fn(
  {
    key: 'treeRules',
    name: 'Tree rules',
    summary: 'The root-to-leaf rules of a tree, each with its value and cover.',
    role: 'transform',
    notes: ['decision-tree', ...RULES],
  },
  rules.treeRules,
)
fn(
  {
    key: 'fidelity',
    name: 'Surrogate fidelity',
    summary: 'Agreement of a surrogate’s labels with a model’s, or the R² of its scores.',
    role: 'property',
    notes: RULES,
    cite: ['molnar2022'],
  },
  rules.fidelity,
)
fn({ key: 'satisfies', name: 'Rule test', role: 'property', notes: RULES }, rules.satisfies)
fn({ key: 'ruleCoverage', name: 'Rule coverage', role: 'property', notes: RULES }, rules.ruleCoverage)
fn({ key: 'simplifyRule', name: 'Simplify a rule', role: 'transform', notes: RULES }, rules.simplifyRule)
fn({ key: 'quantileEdges', name: 'Quantile bin edges', role: 'construction', notes: RULES }, rules.quantileEdges)
fn({ key: 'binPredicates', name: 'Bin predicates', role: 'construction', notes: RULES }, rules.binPredicates)

// ── Concepts ─────────────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'tcav',
    name: 'TCAV',
    summary: 'The fraction of a class’s examples whose logit rises along a concept’s CAV, tested against random CAVs.',
    role: 'test',
    notes: CONCEPTS,
    cite: ['kim2018tcav'],
  },
  concepts.tcav,
)
fn(
  {
    key: 'conceptActivationVector',
    name: 'Concept activation vector',
    summary: 'The unit normal of a linear probe separating a concept’s activations from random ones.',
    role: 'fit',
    notes: CONCEPTS,
    cite: ['kim2018tcav'],
  },
  concepts.conceptActivationVector,
)
fn(
  {
    key: 'linearProbe',
    name: 'Linear probe',
    summary: 'L2-regularised logistic regression on hidden activations, fitted by L-BFGS.',
    role: 'fit',
    notes: CONCEPTS,
  },
  concepts.linearProbe,
)
fn(
  { key: 'tcavScore', name: 'TCAV score', role: 'property', notes: CONCEPTS, cite: ['kim2018tcav'] },
  concepts.tcavScore,
)
fn(
  {
    key: 'conceptSensitivity',
    name: 'Conceptual sensitivity',
    role: 'property',
    notes: CONCEPTS,
    cite: ['kim2018tcav'],
  },
  concepts.conceptSensitivity,
)
fn(
  { key: 'activationGradients', name: 'Activation gradients', role: 'transform', notes: CONCEPTS },
  concepts.activationGradients,
)

// ── Data attribution ─────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'influenceFunctions',
    name: 'Influence functions',
    summary: '−∇ℓ_testᵀ H⁻¹ ∇ℓ(z): the effect of upweighting a training point, exact or by LiSSA.',
    role: 'estimator',
    notes: DATA,
  },
  data.influenceFunctions,
)
fn(
  {
    key: 'tracIn',
    name: 'TracIn',
    summary: 'Σ_t η_t ∇ℓ(z, θ_t) · ∇ℓ(z′, θ_t) over checkpoints, and the self-influence.',
    role: 'estimator',
    notes: DATA,
  },
  data.tracIn,
)
fn(
  {
    key: 'dataShapley',
    name: 'Data Shapley (TMC)',
    summary: 'Shapley values of training points for a model’s score, by truncated Monte Carlo over permutations.',
    role: 'estimator',
    random: true,
    notes: DATA,
  },
  data.dataShapley,
)
fn(
  {
    key: 'knnShapley',
    name: 'KNN-Shapley',
    summary: 'Exact Shapley values of training points for a k-nearest-neighbour classifier, by a sorted recursion.',
    role: 'estimator',
    notes: ['k-nearest-neighbours', ...DATA],
  },
  data.knnShapley,
)
fn({ key: 'exampleGradients', name: 'Per-example gradients', role: 'transform', notes: DATA }, data.exampleGradients)

// ── Global effects ───────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'accumulatedLocalEffects',
    name: 'Accumulated local effects',
    summary: 'Local changes of f across quantile bins, averaged over the rows in each bin, accumulated and centred.',
    role: 'estimator',
    notes: GLOBAL,
    cite: ['molnar2022'],
  },
  global.accumulatedLocalEffects,
)
fn(
  {
    key: 'hStatistic',
    name: 'Friedman’s H-statistic',
    summary: 'The share of joint partial dependence not explained by the separate ones, pairwise and overall.',
    role: 'estimator',
    notes: GLOBAL,
    cite: ['friedman2008'],
  },
  global.hStatistic,
)
fn(
  {
    key: 'functionalAnova',
    name: 'Functional ANOVA',
    summary: 'Main effects and pairwise terms of f under a product grid measure, with their variances (Sobol indices).',
    role: 'estimator',
    notes: GLOBAL,
    cite: ['lengerich2020'],
  },
  global.functionalAnova,
)

// ── Evaluating explanations ──────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'deletionCurve',
    name: 'Deletion and insertion curves',
    summary: 'The output as features are removed (or restored) in attribution order, and its area.',
    role: 'estimator',
    notes: EVALUATION,
    cite: ['petsiuk2018', 'samek2017'],
  },
  evaluation.deletionCurve,
)
fn(
  {
    key: 'faithfulnessCorrelation',
    name: 'Faithfulness correlation',
    summary: 'Correlation between summed attributions of random subsets and the output drop when they are removed.',
    role: 'estimator',
    random: true,
    notes: EVALUATION,
  },
  evaluation.faithfulnessCorrelation,
)
fn(
  {
    key: 'randomiseNetwork',
    name: 'Cascading model randomisation',
    summary: 'Copies of a network with its top k layers re-initialised, for the model-randomisation sanity check.',
    role: 'construction',
    random: true,
    notes: EVALUATION,
    cite: ['adebayo2018'],
  },
  evaluation.randomiseNetwork,
)
fn(
  {
    key: 'explanationSimilarity',
    name: 'Explanation similarity',
    summary: 'Spearman correlation (signed and absolute) and top-k overlap of two attributions.',
    role: 'property',
    notes: EVALUATION,
    cite: ['adebayo2018'],
  },
  evaluation.explanationSimilarity,
)
fn(
  {
    key: 'localLipschitz',
    name: 'Local Lipschitz stability',
    summary: 'The largest change of an explanation per unit change of the input, over a sampled ball.',
    role: 'estimator',
    random: true,
    notes: EVALUATION,
  },
  evaluation.localLipschitz,
)
fn(
  {
    key: 'relevanceMass',
    name: 'Relevance mass',
    summary: 'The share of an attribution’s absolute mass that falls on a ground-truth mask.',
    role: 'property',
    notes: EVALUATION,
  },
  evaluation.relevanceMass,
)
fn(
  { key: 'attributionOrder', name: 'Attribution order', role: 'transform', notes: EVALUATION },
  evaluation.attributionOrder,
)

/** The functions of the module, keyed by name. */
export const explainFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', shapley, tree, lime, gradients, importance, perturbation) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
