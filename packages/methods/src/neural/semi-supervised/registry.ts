/** The registry of `aifn-methods/neural/semi-supervised`: consistency-regularised training and its rules. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as consistency from './consistency'
import * as runs from './run'

const MODULE = 'neural/semi-supervised'
const fn = definer<FunctionInfo>('function', MODULE)
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)

const NOTES = [
  'consistency-regularisation',
  'fixmatch',
  'mean-teacher',
  'virtual-adversarial-training',
  'unsupervised-data-augmentation',
  'mixmatch',
  'adaptive-confidence-thresholds',
]

algorithm(
  {
    key: 'consistencyTraining',
    name: 'Consistency-regularised training',
    summary:
      'Supervised cross-entropy plus a consistency term on unlabelled points: Π-model, temporal ensembling, mean teacher, VAT, UDA, MixMatch or FixMatch (with FlexMatch, FreeMatch or SoftMatch thresholds).',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', flags: ['diverged'] },
    random: true,
    notes: NOTES,
    cite: [
      'laine2017temporal',
      'sajjadi2016stochastic',
      'tarvainen2017meanteacher',
      'miyato2019vat',
      'xie2020uda',
      'berthelot2019mixmatch',
      'sohn2020',
      'zhang2021flexmatch',
      'wang2023freematch',
      'chen2023softmatch',
    ],
  },
  consistency.consistencyTraining,
)
fn(
  {
    key: 'consistencyRun',
    name: 'Consistency-training run',
    summary: 'A consistency-training run from plain data, streaming losses, masks, targets and probabilities.',
    role: 'simulation',
    random: true,
    notes: NOTES,
    cite: ['laine2017temporal', 'sohn2020'],
  },
  runs.consistencyRun,
)
fn(
  {
    key: 'predictProbabilities',
    name: 'Network class probabilities',
    role: 'transform',
    notes: ['consistency-regularisation', 'fixmatch'],
  },
  consistency.predictProbabilities,
)
fn(
  {
    key: 'sharpen',
    name: 'Sharpening',
    tex: '\\operatorname{Sharpen}(p, T)_k = p_k^{1/T} \\big/ \\textstyle\\sum_j p_j^{1/T}',
    summary: 'Lower the entropy of probability rows by a temperature; T → 0 gives the one-hot arg max.',
    role: 'transform',
    notes: ['mixmatch', 'unsupervised-data-augmentation'],
    cite: ['berthelot2019mixmatch'],
  },
  consistency.sharpen,
)
fn(
  {
    key: 'annealingThreshold',
    name: 'Training-signal annealing threshold',
    tex: '\\eta_t = \\alpha_t \\left(1 - \\tfrac{1}{K}\\right) + \\tfrac{1}{K}',
    summary: 'UDA’s threshold on the true-class probability above which a labelled point leaves the loss.',
    role: 'property',
    notes: ['unsupervised-data-augmentation'],
    cite: ['xie2020uda'],
  },
  consistency.annealingThreshold,
)
fn(
  {
    key: 'flexMatchThresholds',
    name: 'FlexMatch class thresholds',
    tex: '\\tau_t(c) = \\frac{\\beta_t(c)}{2 - \\beta_t(c)}\\,\\tau',
    summary: 'Per-class confidence thresholds scaled by how many points each class has learnt.',
    role: 'property',
    notes: ['adaptive-confidence-thresholds', 'fixmatch'],
    cite: ['zhang2021flexmatch'],
  },
  consistency.flexMatchThresholds,
)
fn(
  {
    key: 'rampUpWeight',
    name: 'Gaussian ramp-up',
    tex: 'w(t) = \\lambda\\, e^{-5 (1 - t/R)^2}',
    summary: 'The unsupervised weight rising from λe⁻⁵ to λ over the first R steps.',
    role: 'property',
    notes: ['consistency-regularisation', 'mean-teacher'],
    cite: ['laine2017temporal'],
  },
  consistency.rampUpWeight,
)

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
/** The algorithms of the module. */
export const semiSupervisedNetworkAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  consistency,
  runs,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const semiSupervisedNetworkFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  consistency,
  runs,
) as Table<FunctionInfo>
