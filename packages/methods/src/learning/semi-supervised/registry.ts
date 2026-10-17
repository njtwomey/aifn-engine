/** The registry of `aifn-methods/learning/semi-supervised`: self-training, co-training and the other wrappers. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as selfTraining from './self-training'
import * as clusterThenLabel from './cluster-then-label'
import * as coTraining from './co-training'
import * as gaussianNovelty from './gaussian-novelty'
import * as lapRls from './laplacian-rls'
import * as studentNoise from './student-noise'
import * as thresholdLearning from './threshold-learning'
import * as densityDistance from './density-sensitive-distance'
import * as entropyMinimisation from './entropy-minimisation'
import * as transductiveSvm from './transductive-svm'

const MODULE = 'learning/semi-supervised'
const fn = definer<FunctionInfo>('function', MODULE)
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)

// ── Self-training (self-training.ts) ─────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'selfTrainingSteps',
    name: 'Self-training',
    summary: 'Fit on the labelled points, pseudo-label the confident unlabelled ones, refit on both, and repeat.',
    problem: 'objective',
    state: { iterate: 'proposed', objective: 'unlabelledAccuracy', flags: ['converged'] },
    random: true,
    notes: [
      'self-training',
      'pseudo-labelling',
      'co-training',
      'noisy-student',
      'semi-supervised-assumptions',
      'self-supervised-pretraining-for-semi-supervised-learning',
    ],
    cite: ['scudder1965', 'yarowsky1995', 'lee2013pseudo', 'xie2020'],
  },
  selfTraining.selfTrainingSteps,
)
fn(
  {
    key: 'rowArgmax',
    name: 'Arg max and confidence of each row',
    role: 'transform',
    notes: ['pseudo-labelling', 'fixmatch'],
  },
  selfTraining.rowArgmax,
)

// ── Co-training and tri-training (co-training.ts) ────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'coTrainingSteps',
    name: 'Co-training',
    summary: 'Two classifiers on two views label their most confident pool points of each class for each other.',
    problem: 'objective',
    state: { iterate: 'assigned', flags: ['terminated'] },
    random: true,
    notes: ['co-training'],
    cite: ['blum1998'],
  },
  coTraining.coTrainingSteps,
)
algorithm(
  {
    key: 'triTrainingSteps',
    name: 'Tri-training',
    summary: 'Three bootstrap learners; each takes the points the other two agree on when their joint error allows.',
    problem: 'objective',
    state: { iterate: 'decisions', objective: 'unlabelledAccuracy', flags: ['converged'] },
    random: true,
    notes: ['tri-training-and-disagreement-methods'],
    cite: ['zhou2005tritraining'],
  },
  coTraining.triTrainingSteps,
)
fn(
  {
    key: 'triTrainingRule',
    name: 'Tri-training update rule',
    tex: "e_i \\lvert L_i \\rvert < e'_i \\lvert L'_i \\rvert",
    summary: 'Whether a learner takes the agreed points, all of them or a subsample, from the error bound.',
    role: 'property',
    notes: ['tri-training-and-disagreement-methods'],
    cite: ['zhou2005tritraining'],
  },
  coTraining.triTrainingRule,
)
fn(
  { key: 'productOfExperts', name: 'Product of two views’ experts', role: 'transform', notes: ['co-training'] },
  coTraining.productOfExperts,
)
fn(
  {
    key: 'majority',
    name: 'Majority vote of three',
    role: 'transform',
    notes: ['tri-training-and-disagreement-methods'],
  },
  coTraining.majority,
)

// ── Student noise (student-noise.ts) ─────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'inputJitter',
    name: 'Gaussian input jitter',
    summary: 'Noisy copies of every training row, for a Noisy Student’s student.',
    role: 'transform',
    random: true,
    notes: ['noisy-student'],
    cite: ['xie2020'],
  },
  studentNoise.inputJitter,
)
fn(
  {
    key: 'featureDropout',
    name: 'Feature dropout as data',
    summary: 'Masked copies of every training row under inverted dropout of the features.',
    role: 'transform',
    random: true,
    notes: ['noisy-student'],
    cite: ['xie2020', 'srivastava2014'],
  },
  studentNoise.featureDropout,
)

// ── Cluster-then-label (cluster-then-label.ts) ───────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'clusterClasses',
    name: 'Cluster classes from a few labels',
    tex: 'c(k) = \\arg\\max_c \\sum_{i : y_i = c} r_{ik}',
    summary: 'Each cluster takes the class whose labelled points give it the most responsibility.',
    role: 'fit',
    notes: ['semi-supervised-assumptions'],
    cite: ['zhu2009intro', 'castelli1995exponential'],
  },
  clusterThenLabel.clusterClasses,
)
fn(
  {
    key: 'clusterThenLabelProbabilities',
    name: 'Cluster-then-label probabilities',
    tex: 'P(c \\mid x) = \\sum_{k : c(k) = c} r_k(x)',
    role: 'transform',
    notes: ['semi-supervised-assumptions'],
    cite: ['zhu2009intro'],
  },
  clusterThenLabel.clusterThenLabelProbabilities,
)

// ── Thresholds on a line (threshold-learning.ts) ─────────────────────────────────────────────────────────────────────

const THRESHOLDS = ['when-unlabelled-data-helps']
fn(
  {
    key: 'ermInterval',
    name: 'Interval of empirical risk minimisers',
    role: 'fit',
    notes: THRESHOLDS,
    cite: ['bendavid2008unlabeled'],
  },
  thresholdLearning.ermInterval,
)
fn(
  {
    key: 'thresholdErm',
    name: 'Supervised threshold (ERM midpoint)',
    role: 'fit',
    notes: THRESHOLDS,
    cite: ['bendavid2008unlabeled'],
  },
  thresholdLearning.thresholdErm,
)
fn(
  {
    key: 'lowDensityThreshold',
    name: 'Low-density threshold',
    summary: 'Among the troughs of the unlabelled density, the one with the fewest training errors.',
    role: 'fit',
    notes: THRESHOLDS,
    cite: ['balcan2010discriminative'],
  },
  thresholdLearning.lowDensityThreshold,
)
fn(
  {
    key: 'medianThreshold',
    name: 'Median threshold under a known distribution',
    tex: 't = F^{-1}\\big((F(\\ell) + F(r))/2\\big)',
    role: 'fit',
    notes: THRESHOLDS,
    cite: ['bendavid2008unlabeled'],
  },
  thresholdLearning.medianThreshold,
)
fn(
  {
    key: 'unlabelledDensity',
    name: 'Density of the unlabelled inputs on a grid',
    role: 'estimator',
    notes: THRESHOLDS,
  },
  thresholdLearning.unlabelledDensity,
)
fn(
  { key: 'thresholdErrors', name: 'Training errors of a threshold', role: 'property', notes: THRESHOLDS },
  thresholdLearning.thresholdErrors,
)

// ── Manifold regularisation (laplacian-rls.ts) ───────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'laplacianRegularisedLeastSquares',
    name: 'Laplacian regularised least squares',
    tex: '(JK + \\gamma_A l I + \\tfrac{\\gamma_I l}{n^2} L K)\\,\\alpha = J y',
    summary: 'Kernel least squares on the labelled points with a graph-smoothness penalty over all points.',
    role: 'fit',
    notes: ['manifold-regularisation', 'semi-supervised-regression', 'evaluating-semi-supervised-learning'],
    cite: ['belkin2006manifold'],
  },
  lapRls.laplacianRegularisedLeastSquares,
)

// ── Open-set filtering (gaussian-novelty.ts) ─────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'gaussianNoveltyScores',
    name: 'Gaussian novelty scores',
    tex: 's(x) = \\min_k \\lVert x - \\hat\\mu_k \\rVert^2 / \\hat\\sigma^2,\\; p = P(\\chi^2_d \\ge s)',
    summary: 'The smallest standardised distance to a labelled class mean, with its chi-squared p-value.',
    role: 'estimator',
    notes: ['open-set-semi-supervised-learning'],
    cite: ['hastie2009'],
  },
  gaussianNovelty.gaussianNoveltyScores,
)

// ── Low-density separation (transductive-svm.ts, entropy-minimisation.ts, density-sensitive-distance.ts) ────────────

const TSVM = ['transductive-support-vector-machine', 'low-density-separation']
fn(
  {
    key: 'transductiveSvmFit',
    name: 'Transductive SVM (one stage)',
    tex: '\\tfrac12\\lVert w \\rVert^2 + C\\textstyle\\sum_i \\max(0, 1 - y_i f_i)^2 + C^*\\sum_j e^{-s f_j^2}',
    summary: 'The primal TSVM objective with the balancing constraint, minimised by L-BFGS.',
    role: 'fit',
    notes: TSVM,
    cite: ['chapelle2005lds'],
  },
  transductiveSvm.transductiveSvmFit,
)
fn(
  {
    key: 'transductiveSvmStages',
    name: '∇TSVM continuation',
    summary: 'Refit the TSVM as the unlabelled cost doubles up to C*, each stage from the last one’s weights.',
    role: 'fit',
    notes: TSVM,
    cite: ['chapelle2005lds'],
  },
  transductiveSvm.transductiveSvmStages,
)
fn(
  { key: 'transductiveSvmObjective', name: 'TSVM objective and gradient', role: 'construction', notes: TSVM },
  transductiveSvm.transductiveSvmObjective,
)
fn(
  { key: 'transductiveSvmDecision', name: 'TSVM decision values', role: 'transform', notes: TSVM },
  transductiveSvm.transductiveSvmDecision,
)
fn(
  {
    key: 'symmetricHinge',
    name: 'Symmetric hinge',
    tex: '\\max(0, 1 - \\lvert t \\rvert)',
    role: 'property',
    notes: TSVM,
    cite: ['collobert2006tsvm'],
  },
  transductiveSvm.symmetricHinge,
)
fn(
  {
    key: 'symmetricSigmoid',
    name: 'Symmetric sigmoid',
    tex: 'e^{-s t^2}',
    role: 'property',
    notes: TSVM,
    cite: ['chapelle2005lds'],
  },
  transductiveSvm.symmetricSigmoid,
)
fn(
  {
    key: 'symmetricRamp',
    name: 'Symmetric ramp',
    tex: '\\min(1 + s, \\max(0, 1 - \\lvert t \\rvert))',
    role: 'property',
    notes: TSVM,
    cite: ['collobert2006tsvm'],
  },
  transductiveSvm.symmetricRamp,
)
const ENTROPY = ['entropy-minimisation', 'low-density-separation']
fn(
  {
    key: 'entropyMinimisationFit',
    name: 'Entropy minimisation',
    tex:
      '-\\tfrac1\\ell\\textstyle\\sum_i \\log p(y_i \\mid x_i) + \\tfrac\\lambda u \\sum_j H(p(\\cdot \\mid x_j))' +
      ' + \\tfrac\\gamma2 \\lVert W \\rVert^2',
    summary: 'A multinomial logistic model whose objective adds the mean entropy of its unlabelled predictions.',
    role: 'fit',
    notes: ENTROPY,
    cite: ['grandvalet2005'],
  },
  entropyMinimisation.entropyMinimisationFit,
)
fn(
  {
    key: 'entropyMinimisationPath',
    name: 'Entropy minimisation path',
    summary: 'Fits over an increasing list of entropy weights, each warm-started from the last.',
    role: 'fit',
    notes: ENTROPY,
    cite: ['grandvalet2005'],
  },
  entropyMinimisation.entropyMinimisationPath,
)
fn(
  {
    key: 'entropyMinimisationObjective',
    name: 'Entropy minimisation objective and gradient',
    role: 'construction',
    notes: ENTROPY,
  },
  entropyMinimisation.entropyMinimisationObjective,
)
fn(
  {
    key: 'entropyMinimisationProbabilities',
    name: 'Entropy minimisation class probabilities',
    role: 'transform',
    notes: ENTROPY,
  },
  entropyMinimisation.entropyMinimisationProbabilities,
)
const RHO = ['low-density-separation']
fn(
  {
    key: 'rhoPathDistances',
    name: 'Density-sensitive ρ-path distance',
    tex:
      'D_\\rho(i, j) = \\tfrac1\\rho \\log\\big(1 + ' +
      '\\min_{\\text{paths}} \\textstyle\\sum_{\\text{edges}} (e^{\\rho d} - 1)\\big)',
    summary: 'Shortest paths with edge costs e^{ρd} − 1: short within dense regions, long across gaps.',
    role: 'estimator',
    notes: RHO,
    cite: ['chapelle2005lds'],
  },
  densityDistance.rhoPathDistances,
)
fn(
  { key: 'rhoPathExtend', name: 'ρ-path distance to new points', role: 'transform', notes: RHO },
  densityDistance.rhoPathExtend,
)
fn(
  { key: 'rhoQueryNeighbours', name: 'Nearest points of queries', role: 'construction', notes: RHO },
  densityDistance.rhoQueryNeighbours,
)
fn({ key: 'rhoPathNodes', name: 'Cheapest ρ-path', role: 'property', notes: RHO }, densityDistance.rhoPathNodes)
fn(
  { key: 'rhoEdgeCost', name: 'ρ-path edge cost', tex: 'e^{\\rho d} - 1', role: 'property', notes: RHO },
  densityDistance.rhoEdgeCost,
)
fn(
  {
    key: 'rhoDistance',
    name: 'ρ-path distance from a path cost',
    tex: '\\tfrac1\\rho \\log(1 + S)',
    role: 'property',
    notes: RHO,
  },
  densityDistance.rhoDistance,
)

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const sources = [
  selfTraining,
  coTraining,
  studentNoise,
  clusterThenLabel,
  thresholdLearning,
  lapRls,
  gaussianNovelty,
  transductiveSvm,
  entropyMinimisation,
  densityDistance,
]
/** The algorithms of the module. */
export const semiSupervisedAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  ...sources,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const semiSupervisedFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  ...sources,
) as Table<FunctionInfo>
