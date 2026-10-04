/**
 * The registry of `aifn-methods/generative/energy`: JEM (or plain cross-entropy) training as a traceable algorithm,
 * and the classifier, its energy and score, the logit shift and the streamed run as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as jem from './jem'
import * as run from './run'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'generative/energy')
const fn = definer<FunctionInfo>('function', 'generative/energy')
const JEM = ['joint-energy-models', 'energy-based-models']

algorithm(
  {
    key: 'jemTraining',
    name: 'JEM training',
    summary: 'Cross-entropy plus persistent contrastive divergence on the energy −logsumexp of the logits.',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', flags: ['diverged'] },
    random: true,
    notes: JEM,
    cite: ['grathwohl2019'],
  },
  jem.jemTraining,
)

fn(
  {
    key: 'classifierEnergy',
    name: 'Classifier energy',
    summary: 'E(x) = −logsumexp_y f(x)[y]: the energy of p(x) a softmax classifier defines.',
    tex: 'E(x) = -\\log\\sum_y e^{f(x)[y]}',
    role: 'property',
    notes: JEM,
    cite: ['grathwohl2019'],
  },
  jem.classifierEnergy,
)
fn(
  {
    key: 'classifierScore',
    name: 'Classifier score',
    summary: '∇ₓ log p(x) or ∇ₓ log p(x | y) of a classifier read as an energy-based model, for Langevin sampling.',
    role: 'property',
    notes: JEM,
  },
  jem.classifierScore,
)
fn(
  {
    key: 'logitShift',
    name: 'Logit shift',
    summary: 'A function c(x) added to every logit: p(y | x) is unchanged, the energy becomes E(x) − c(x).',
    role: 'transform',
    notes: JEM,
  },
  jem.logitShift,
)
fn(
  {
    key: 'jemRun',
    name: 'Streamed JEM and cross-entropy run',
    summary: 'Train both classifiers side by side and yield fields, Langevin samples, calibration and OOD scores.',
    role: 'simulation',
    random: true,
    notes: JEM,
    cite: ['grathwohl2019'],
  },
  run.jemRun,
)

/** The algorithms of the module, keyed by factory name. */
export const energyAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>('algorithm', jem) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const energyFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', jem, run) as Table<FunctionInfo>
