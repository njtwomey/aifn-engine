/** The registry of `aifn-methods/generative/boltzmann`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as dbn from './dbn'
import * as hopfield from './hopfield'
import * as rbm from './rbm'

const fn = definer<FunctionInfo>('function', 'generative/boltzmann')
const RBM = ['restricted-boltzmann-machine', 'boltzmann-machine']
const DBN = ['deep-belief-network', 'restricted-boltzmann-machine']
const HOP = ['hopfield-network']
const MODERN = ['modern-hopfield-network']

fn(
  { key: 'rbm', name: 'Bernoulli RBM', role: 'construction', random: true, notes: RBM, cite: ['smolensky1986'] },
  rbm.rbm,
)
fn({ key: 'hiddenProbabilities', name: 'RBM p(h | v)', role: 'property', notes: RBM }, rbm.hiddenProbabilities)
fn({ key: 'visibleProbabilities', name: 'RBM p(v | h)', role: 'property', notes: RBM }, rbm.visibleProbabilities)
fn({ key: 'freeEnergy', name: 'RBM free energy', role: 'property', notes: RBM }, rbm.freeEnergy)
fn(
  {
    key: 'logPartition',
    name: 'RBM log partition function',
    summary: 'Exact log Z summed over the hidden states.',
    role: 'property',
    notes: RBM,
  },
  rbm.logPartition,
)
fn({ key: 'rbmLogLikelihood', name: 'RBM exact log-likelihood', role: 'estimator', notes: RBM }, rbm.rbmLogLikelihood)
fn(
  { key: 'gibbsChain', name: 'RBM block Gibbs sampling', role: 'simulation', random: true, notes: RBM },
  rbm.gibbsChain,
)
fn(
  {
    key: 'contrastiveDivergenceStep',
    name: 'Contrastive divergence (CD-k, PCD)',
    summary: 'Data statistics minus statistics after k Gibbs steps from the data (or from persistent chains).',
    role: 'fit',
    random: true,
    notes: RBM,
    cite: ['hinton2002poe', 'tieleman2008'],
  },
  rbm.contrastiveDivergenceStep,
)
fn(
  {
    key: 'rbmRun',
    name: 'Streamed RBM training',
    summary: 'CD-k or PCD-k epochs with the exact log-likelihood, weights and long-chain samples over training.',
    role: 'simulation',
    random: true,
    notes: RBM,
    cite: ['hinton2002poe', 'hinton2012practical'],
  },
  rbm.rbmRun,
)
fn(
  {
    key: 'dbnUp',
    name: 'DBN recognition pass',
    summary: 'Hidden probabilities layer by layer up the stack.',
    role: 'transform',
    notes: DBN,
  },
  dbn.dbnUp,
)
fn(
  {
    key: 'dbnSample',
    name: 'DBN ancestral sampling',
    summary: 'Gibbs sampling in the top RBM, then one directed pass down the sigmoid belief layers.',
    role: 'simulation',
    random: true,
    notes: DBN,
    cite: ['hinton2006dbn'],
  },
  dbn.dbnSample,
)
fn(
  {
    key: 'dbnRun',
    name: 'Deep belief network (greedy layer-wise)',
    summary:
      'RBMs trained one layer at a time by CD on the hidden probabilities of the layer below, with samples over training and a discriminative fine-tune against random initialisation.',
    role: 'simulation',
    random: true,
    notes: DBN,
    cite: ['hinton2006dbn', 'bengio2007'],
  },
  dbn.dbnRun,
)
fn(
  { key: 'hebbianWeights', name: 'Hebbian weights', role: 'fit', notes: HOP, cite: ['hopfield1982'] },
  hopfield.hebbianWeights,
)
fn(
  { key: 'hopfieldEnergy', name: 'Hopfield energy', role: 'property', notes: HOP, cite: ['hopfield1982'] },
  hopfield.hopfieldEnergy,
)
fn(
  {
    key: 'hopfieldRecall',
    name: 'Hopfield recall',
    summary: 'Asynchronous sign updates from a cue, sweep by sweep, never raising the energy.',
    role: 'simulation',
    random: true,
    notes: HOP,
    cite: ['hopfield1982'],
  },
  hopfield.hopfieldRecall,
)
fn(
  {
    key: 'modernHopfieldUpdate',
    name: 'Modern Hopfield update',
    summary: 'ξ ← X softmax(β Xᵀξ): one attention step retrieves the best-matching stored pattern.',
    role: 'transform',
    notes: MODERN,
    cite: ['ramsauer2021', 'krotov2016'],
  },
  hopfield.modernHopfieldUpdate,
)
fn(
  {
    key: 'modernHopfieldEnergy',
    name: 'Modern Hopfield energy',
    role: 'property',
    notes: MODERN,
    cite: ['ramsauer2021'],
  },
  hopfield.modernHopfieldEnergy,
)
fn(
  { key: 'corruptPattern', name: 'Corrupt a pattern', role: 'simulation', random: true, notes: HOP },
  hopfield.corruptPattern,
)
fn({ key: 'overlaps', name: 'Pattern overlaps', role: 'property', notes: HOP }, hopfield.overlaps)
fn(
  {
    key: 'capacityCurve',
    name: 'Hopfield capacity curve',
    summary: 'Recall overlap against the number of stored random patterns, classical against modern.',
    role: 'simulation',
    random: true,
    notes: [...HOP, ...MODERN],
    cite: ['hopfield1982', 'ramsauer2021'],
  },
  hopfield.capacityCurve,
)

/** The functions of the module, keyed by name. */
export const boltzmannFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', rbm, dbn, hopfield) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
