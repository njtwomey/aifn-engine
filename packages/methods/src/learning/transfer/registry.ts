/** The registry of `aifn-methods/learning/transfer`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as adaptation from './adaptation'
import * as alignment from './alignment'
import * as continual from './continual'
import * as labelShift from './label-shift'
import * as meta from './meta'
import * as prototypical from './prototypical'

const fn = definer<FunctionInfo>('function', 'learning/transfer')
const DA = ['domain-adversarial-training', 'domain-adaptation-theory']

fn(
  {
    key: 'mmdSquared',
    name: 'Squared MMD (Gaussian kernel)',
    summary: 'The biased V-statistic of the squared maximum mean discrepancy, differentiable in both samples.',
    role: 'estimator',
    notes: DA,
    cite: ['gretton2012', 'long2015dan'],
  },
  alignment.mmdSquared,
)
fn(
  {
    key: 'coralLoss',
    name: 'CORAL loss',
    summary: '‖Cov(source) − Cov(target)‖²_F / (4d²) between feature covariances.',
    role: 'estimator',
    notes: DA,
    cite: ['sun2016deepcoral', 'sun2016coral'],
  },
  alignment.coralLoss,
)
fn(
  {
    key: 'gradientReversal',
    name: 'Gradient-reversal layer',
    summary: 'The identity forward and −λ times the gradient backward: adversarial features in one loss.',
    role: 'transform',
    notes: ['domain-adversarial-training'],
    cite: ['ganin2016'],
  },
  alignment.gradientReversal,
)
fn(
  {
    key: 'domainAdaptationRun',
    name: 'Domain adaptation (source only, MMD, CORAL, DANN)',
    summary:
      'A feature extractor and classifier trained on labelled source and unlabelled target data with an alignment penalty.',
    role: 'simulation',
    random: true,
    notes: DA,
    cite: ['ganin2016', 'long2015dan', 'sun2016deepcoral'],
  },
  adaptation.domainAdaptationRun,
)
fn(
  {
    key: 'blackBoxShiftEstimate',
    name: 'Black-box shift estimation',
    summary: 'Target priors from the source confusion matrix and the target’s predicted-label frequencies: C w = μ.',
    role: 'estimator',
    notes: ['label-shift-and-target-shift'],
    cite: ['lipton2018'],
  },
  labelShift.blackBoxShiftEstimate,
)
fn(
  {
    key: 'priorShiftEm',
    name: 'EM for prior shift',
    summary: 'Alternate prior-reweighted posteriors and their mean until the target priors settle.',
    role: 'estimator',
    notes: ['label-shift-and-target-shift'],
    cite: ['saerens2002'],
  },
  labelShift.priorShiftEm,
)
fn(
  {
    key: 'reweightPosteriors',
    name: 'Prior-shift posterior correction',
    role: 'transform',
    notes: ['label-shift-and-target-shift'],
    cite: ['saerens2002'],
  },
  labelShift.reweightPosteriors,
)
fn(
  {
    key: 'continualRun',
    name: 'Continual learning (naive, EWC, replay)',
    summary: 'One network trained on tasks in turn, with accuracy on every task as training proceeds.',
    role: 'simulation',
    random: true,
    notes: ['continual-learning', 'regularisation-based-continual-learning', 'replay-and-architectural-methods'],
    cite: ['kirkpatrick2017', 'rolnick2019'],
  },
  continual.continualRun,
)
fn(
  {
    key: 'mamlRun',
    name: 'MAML on sinusoids',
    summary: 'Second- or first-order MAML against a pretrained baseline: meta-loss, adapted fits and few-shot curves.',
    role: 'simulation',
    random: true,
    notes: ['meta-learning'],
    cite: ['finn2017maml'],
  },
  meta.mamlRun,
)
fn(
  {
    key: 'sineTasks',
    name: 'Sinusoid tasks',
    role: 'simulation',
    random: true,
    notes: ['meta-learning'],
    cite: ['finn2017maml'],
  },
  meta.sineTasks,
)
fn(
  {
    key: 'sineSamples',
    name: 'Sinusoid samples',
    role: 'simulation',
    random: true,
    notes: ['meta-learning'],
    cite: ['finn2017maml'],
  },
  meta.sineSamples,
)
fn(
  {
    key: 'fewShotEpisode',
    name: 'Few-shot episode (directions)',
    summary: 'N classes as directions from the origin with a nuisance radius; K support and Q query points per class.',
    role: 'simulation',
    random: true,
    notes: ['metric-based-few-shot-learning'],
  },
  prototypical.fewShotEpisode,
)
fn(
  {
    key: 'prototypeLogits',
    name: 'Prototype logits',
    summary: 'Negative squared distances from embedded queries to the mean embedding of each class’s support points.',
    role: 'transform',
    notes: ['metric-based-few-shot-learning'],
    cite: ['snell2017'],
  },
  prototypical.prototypeLogits,
)
fn(
  {
    key: 'prototypicalRun',
    name: 'Prototypical networks (episodic training)',
    summary:
      'An embedding trained on a new N-way K-shot episode per update, with test-episode accuracy against raw-input prototypes and prototype maps.',
    role: 'simulation',
    random: true,
    notes: ['metric-based-few-shot-learning', 'meta-learning'],
    cite: ['snell2017'],
  },
  prototypical.prototypicalRun,
)

/** The functions of the module, keyed by name. */
export const transferFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', alignment, adaptation, labelShift, continual, meta, prototypical) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
