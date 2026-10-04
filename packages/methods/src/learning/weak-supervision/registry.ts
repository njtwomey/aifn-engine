/** The registry of `aifn-methods/learning/weak-supervision`: label models, weak-label classifiers, MIL and noise. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as classifiers from './classifiers'
import * as active from './active-proportions'
import * as compare from './compare'
import * as labelModels from './label-models'
import * as labelProportions from './label-proportions'
import * as local from './local-likelihood'
import * as mil from './mil'
import * as millet from './millet'
import * as noiseTests from './noise-tests'
import * as noise from './noise'

const MODULE = 'learning/weak-supervision'
const fn = definer<FunctionInfo>('function', MODULE)
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)

fn(
  {
    key: 'majorityVote',
    name: 'Majority vote',
    summary: 'Each example’s class distribution is the share of its votes; abstentions are ignored, ties split.',
    role: 'estimator',
    notes: ['label-model-estimation', 'learning-from-crowds', 'weak-supervision'],
  },
  labelModels.majorityVote,
)
algorithm(
  {
    key: 'dawidSkeneSteps',
    name: 'Dawid–Skene EM',
    summary: 'Alternate posteriors over the true class and each voter’s confusion matrix; the likelihood never falls.',
    problem: 'objective',
    state: { iterate: 'posteriors', objective: 'logLikelihood', flags: ['converged'] },
    notes: ['dawid-skene-model', 'learning-from-crowds', 'expectation-maximisation'],
    cite: ['dawid1979'],
  },
  labelModels.dawidSkeneSteps,
)
fn(
  { key: 'dawidSkene', name: 'Dawid–Skene model', role: 'fit', notes: ['dawid-skene-model'], cite: ['dawid1979'] },
  labelModels.dawidSkene,
)
fn(
  {
    key: 'labelModel',
    name: 'Data-programming label model',
    summary: 'Labelling-function accuracies and coverages by maximum marginal likelihood (Snorkel’s generative model).',
    role: 'fit',
    notes: ['label-model-estimation', 'data-programming', 'snorkel'],
    cite: ['ratner2016', 'ratner2017'],
  },
  labelModels.labelModel,
)
fn(
  {
    key: 'elkanNoto',
    name: 'Elkan–Noto PU learning',
    summary: 'Train labelled-versus-unlabelled, then divide by the label frequency c estimated on the labelled points.',
    role: 'fit',
    notes: ['positive-unlabelled-learning'],
    cite: ['elkan2008'],
  },
  classifiers.elkanNoto,
)
fn(
  {
    key: 'puClassifier',
    name: 'PU classifier (uPU, nnPU)',
    summary: 'Minimise the unbiased or non-negative PU risk, or treat unlabelled as negative.',
    role: 'fit',
    notes: ['non-negative-positive-unlabelled-learning', 'positive-unlabelled-learning'],
    cite: ['kiryo2017'],
  },
  classifiers.puClassifier,
)
fn(
  {
    key: 'proportionClassifier',
    name: 'Classifier from label proportions',
    summary: 'A softmax classifier fitted so each bag’s mean prediction matches its known class proportions.',
    role: 'fit',
    notes: ['learning-from-label-proportions'],
    cite: ['quadrianto2009'],
  },
  classifiers.proportionClassifier,
)
fn(
  {
    key: 'complementaryClassifier',
    name: 'Classifier from complementary labels',
    summary: 'A softmax classifier trained from classes each example does not belong to.',
    role: 'fit',
    notes: ['complementary-labels'],
    cite: ['ishida2017', 'ishida2019'],
  },
  classifiers.complementaryClassifier,
)
fn(
  {
    key: 'attentionMil',
    name: 'Attention-based multiple-instance learning',
    summary: 'Embed instances, pool each bag with learned attention weights, classify the bag.',
    role: 'fit',
    notes: ['attention-based-multiple-instance-learning', 'multiple-instance-learning'],
    cite: ['ilse2018'],
  },
  mil.attentionMil,
)
fn(
  {
    key: 'confidentLearning',
    name: 'Confident learning',
    summary:
      'The joint of noisy and true labels, the noise rates and likely label errors from predicted probabilities.',
    role: 'estimator',
    notes: ['estimating-noise-rates', 'learning-with-noisy-labels', 'label-noise-models'],
    cite: ['northcutt2021'],
  },
  noise.confidentLearning,
)
fn(
  { key: 'votesOf', name: 'Votes from rows', role: 'construction', notes: ['label-model-estimation'] },
  labelModels.votesOf,
)
fn(
  { key: 'classCount', name: 'Class count of labels', role: 'property', notes: ['estimating-noise-rates'] },
  noise.classCount,
)

fn(
  {
    key: 'labelModelReport',
    name: 'Label models compared',
    summary: 'Majority vote, Dawid–Skene EM step by step and the label model on the same votes, against the truth.',
    role: 'estimator',
    notes: ['label-model-estimation', 'dawid-skene-model', 'snorkel'],
  },
  compare.labelModelReport,
)

// ── Label proportions (Poyiadzi et al. 2018) and active learning with them (Poyiadzis et al. 2019) ───────────────────

const LLP = ['label-propagation-for-label-proportions', 'learning-from-label-proportions']

algorithm(
  {
    key: 'lpllpSteps',
    name: 'Label propagation for label proportions (LP-LLP)',
    summary:
      'Start each point at its bag’s proportions; propagate over the graph, project back onto the bags’ class masses, repeat.',
    problem: 'graph',
    state: { iterate: 'scores', flags: ['converged'] },
    notes: LLP,
    cite: ['poyiadzi2018'],
  },
  labelProportions.lpllpSteps,
)
fn(
  { key: 'lpllp', name: 'LP-LLP to convergence', role: 'fit', notes: LLP, cite: ['poyiadzi2018'] },
  labelProportions.lpllp,
)
fn(
  {
    key: 'lpllpGraph',
    name: 'LP-LLP graph',
    tex: 'W_{ij} = e^{-\\gamma\\lVert x_i - x_j\\rVert^2},\\; S = D^{-1}W,\\; P = (1-\\alpha)(I - \\alpha S)^{-1}',
    summary: 'The affinities, the random-walk matrix and the propagation matrix LP-LLP uses.',
    role: 'construction',
    notes: LLP,
    cite: ['poyiadzi2018'],
  },
  labelProportions.lpllpGraph,
)
fn(
  {
    key: 'lpllpGammaSearch',
    name: 'LP-LLP bandwidth by smoothness',
    summary: 'Run LP-LLP over a grid of γ and keep the one whose converged scores are smoothest on its graph.',
    role: 'estimator',
    notes: LLP,
    cite: ['poyiadzi2018'],
  },
  labelProportions.lpllpGammaSearch,
)
fn(
  {
    key: 'bagsByProportion',
    name: 'Bags with given class proportions',
    summary: 'Split labelled points into bags of given sizes whose class proportions are as requested.',
    role: 'construction',
    random: true,
    notes: LLP,
  },
  labelProportions.bagsByProportion,
)
fn(
  { key: 'bagProportionsOf', name: 'Bag class proportions', role: 'property', notes: LLP },
  labelProportions.bagProportionsOf,
)
fn({ key: 'readBags', name: 'Read bags and proportions', role: 'construction', notes: LLP }, labelProportions.readBags)
fn(
  {
    key: 'inverseCalibration',
    name: 'Inverse calibration (InvCal)',
    summary:
      'Support vector regression from bag means to the log-odds of the bag proportions; a point is 1 when f(x) > 0.',
    role: 'fit',
    notes: ['mean-map-and-invcal', ...LLP],
    cite: ['ruping2010'],
  },
  labelProportions.inverseCalibration,
)
fn(
  {
    key: 'alterProportionSvm',
    name: 'Alternating ∝SVM',
    summary: 'Alternate an SVM fit and relabelling each bag to its proportion by decision value, from random starts.',
    role: 'fit',
    random: true,
    notes: LLP,
    cite: ['yu2013'],
  },
  labelProportions.alterProportionSvm,
)
fn(
  {
    key: 'meanMap',
    name: 'MeanMap',
    summary: 'Class means from bag means by least squares, then a conditional exponential family fitted to them.',
    role: 'fit',
    random: true,
    notes: ['mean-map-and-invcal', ...LLP],
    cite: ['quadrianto2009'],
  },
  labelProportions.meanMap,
)
fn(
  {
    key: 'llpComparison',
    name: 'LP-LLP against the baselines',
    summary: 'Accuracy of LP-LLP, InvCal, alter-∝SVM, MeanMap and the proportion loss as bag size and purity vary.',
    role: 'estimator',
    notes: LLP,
    cite: ['poyiadzi2018'],
  },
  labelProportions.llpComparison,
)
algorithm(
  {
    key: 'activeProportionsSteps',
    name: 'Active learning with label proportions',
    summary:
      'Query a bag of pool points (US-Mass, US-LP, random or exact), add the oracle’s proportion as a bag, refit LP-LLP.',
    problem: 'graph',
    state: { iterate: 'scores', objective: 'accuracy', flags: ['terminated'] },
    notes: LLP,
    cite: ['poyiadzi2019', 'settles2009'],
  },
  active.activeProportionsSteps,
)
fn(
  {
    key: 'activeProportionsRun',
    name: 'One active-learning run, query by query',
    role: 'simulation',
    notes: LLP,
    cite: ['poyiadzi2019'],
  },
  active.activeProportionsRun,
)
fn(
  {
    key: 'activeProportionsProblem',
    name: 'Starting bags, pool and test split',
    summary: 'Two small bags with set class shares, a held-out test set, and every other point in the query pool.',
    role: 'construction',
    random: true,
    notes: LLP,
    cite: ['poyiadzi2019'],
  },
  active.activeProportionsProblem,
)
fn(
  {
    key: 'activeProportionsCurves',
    name: 'Active learning curves with an LLP-oracle',
    summary: 'Test accuracy after each query for every query strategy, averaged over datasets (the paper’s Fig. 2).',
    role: 'estimator',
    notes: LLP,
    cite: ['poyiadzi2019'],
  },
  active.activeProportionsCurves,
)

// ── Testing for class-conditional label noise (Poyiadzi et al. 2022; Yang et al. 2024) ──────────────────────────────

const NOISE = ['hypothesis-testing-for-class-conditional-label-noise', 'label-noise-models']

fn(
  {
    key: 'classConditionalNoiseTest',
    name: 'Anchor-point test for class-conditional label noise',
    tex: 'z = \\frac{\\bar\\eta - 1/2}{\\sqrt{v}},\\quad v = \\tfrac{1}{16}\\,\\bar x^\\top \\hat H \\bar x',
    summary:
      'A z-test that the noisy posterior averages ½ over anchor points, from a logistic or local likelihood fit.',
    role: 'test',
    notes: [...NOISE, 'local-maximum-likelihood-noise-test', 'z-test'],
    cite: ['poyiadzi2022', 'yang2024'],
  },
  noiseTests.classConditionalNoiseTest,
)
fn(
  {
    key: 'noiseTestPower',
    name: 'Power of the noise test',
    summary: 'The probability of rejecting uniform noise at a level, given the noise rates and the variance of η̄.',
    role: 'property',
    notes: NOISE,
    cite: ['poyiadzi2022'],
  },
  noiseTests.noiseTestPower,
)
fn(
  {
    key: 'anchorsForPower',
    name: 'Anchors needed for a target power',
    role: 'property',
    notes: NOISE,
    cite: ['poyiadzi2022'],
  },
  noiseTests.anchorsForPower,
)
fn(
  {
    key: 'noiseTestSimulation',
    name: 'Noise test by simulation',
    summary: 'p-values of the test over many noisy datasets per cell: its size under uniform noise and its power.',
    role: 'simulation',
    notes: [...NOISE, 'local-maximum-likelihood-noise-test'],
    cite: ['poyiadzi2022', 'yang2024'],
  },
  noiseTests.noiseTestSimulation,
)
fn(
  {
    key: 'localLogistic',
    name: 'Local likelihood logistic regression',
    tex: '\\hat r(x) = \\sigma(\\hat\\beta_0),\\; \\hat\\beta = \\arg\\max_\\beta \\textstyle\\sum_i K_h(x - x_i)\\, \\ell(y_i, \\langle \\beta, A_p(x_i - x) \\rangle)',
    summary:
      'A kernel-weighted logistic fit of a local polynomial in x_i − x, with the sandwich variance of its log-odds.',
    role: 'fit',
    notes: ['local-regression', 'local-maximum-likelihood-noise-test', 'logistic-regression'],
    cite: ['loader1999', 'yang2024'],
  },
  local.localLogistic,
)
fn(
  {
    key: 'localLogisticCovariance',
    name: 'Covariance of two local logistic fits',
    role: 'estimator',
    notes: ['local-maximum-likelihood-noise-test'],
    cite: ['yang2024'],
  },
  local.localLogisticCovariance,
)
fn(
  {
    key: 'localLogisticBandwidth',
    name: 'Local logistic bandwidth by leave-one-out',
    role: 'estimator',
    random: true,
    notes: ['local-regression', 'cross-validation'],
    cite: ['loader1999'],
  },
  local.localLogisticBandwidth,
)
fn(
  { key: 'localPolynomialBasis', name: 'Local polynomial basis', role: 'construction', notes: ['local-regression'] },
  local.localPolynomialBasis,
)
fn(
  {
    key: 'weightedLogistic',
    name: 'Weighted logistic MLE by Newton',
    role: 'fit',
    notes: ['logistic-regression', 'newtons-method'],
  },
  local.weightedLogistic,
)

// ── MILLET (Early et al. 2024) ─────────────────────────────────────────────────────────────────────────────────────

const MILLET = ['millet', 'multiple-instance-learning']

fn(
  {
    key: 'milletModel',
    name: 'MILLET model',
    summary: 'A convolutional time-series classifier whose pooling is a MIL pooling, so it explains its time points.',
    role: 'construction',
    notes: MILLET,
    cite: ['early2024'],
  },
  millet.milletModel,
)
fn(
  {
    key: 'milletRun',
    name: 'MILLET training run',
    summary: 'Train a MILLET model by Adam with checkpoints, then score its interpretations (AOPCR, NDCG@n).',
    role: 'fit',
    notes: MILLET,
    cite: ['early2024'],
  },
  millet.milletRun,
)
fn(
  {
    key: 'milletScores',
    name: 'MILLET interpretability scores',
    summary: 'Accuracy, AOPCR against random orderings and NDCG@n against the planted discriminatory points.',
    role: 'estimator',
    random: true,
    notes: [...MILLET, 'interpretability'],
    cite: ['early2024', 'samek2017'],
  },
  millet.milletScores,
)
fn(
  {
    key: 'milletInterpretation',
    name: 'MILLET time-point interpretation',
    role: 'transform',
    notes: MILLET,
    cite: ['early2024'],
  },
  millet.milletInterpretation,
)
fn({ key: 'milletLogits', name: 'MILLET logits of a bag', role: 'transform', notes: MILLET }, millet.milletLogits)
fn(
  { key: 'milletProbabilities', name: 'MILLET class probabilities', role: 'transform', notes: MILLET },
  millet.milletProbabilities,
)
fn({ key: 'replicatePad', name: 'Replicate padding', role: 'transform', notes: ['millet'] }, millet.replicatePad)

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const sources = [labelModels, classifiers, mil, noise, compare, labelProportions, active, noiseTests, local, millet]
/** The algorithms of the module. */
export const weakSupervisionAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  ...sources,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const weakSupervisionFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  ...sources,
) as Table<FunctionInfo>
