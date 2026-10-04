/**
 * The registry of `aifn-methods/inference/sequence-models`.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as crf from './crf'
import * as hmm from './hmm'
import * as templateCrf from './template-crf'
import * as toyTagging from './toy-tagging'

const fn = definer<FunctionInfo>('function', 'inference/sequence-models')
const HMM = ['hidden-markov-model']
const CRF = ['conditional-random-field']

fn({ key: 'hmm', name: 'Hidden Markov model', role: 'construction', notes: HMM, cite: ['rabiner1989'] }, hmm.hmm)
fn({ key: 'hmmChain', name: 'HMM chain potentials', role: 'construction', notes: HMM }, hmm.hmmChain)
fn({ key: 'hmmModel', name: 'HMM model specification', role: 'construction', notes: HMM }, hmm.hmmModel)
fn(
  {
    key: 'dishonestCasino',
    name: 'Occasionally dishonest casino',
    role: 'construction',
    notes: ['occasionally-dishonest-casino', ...HMM],
    cite: ['durbin1998'],
  },
  hmm.dishonestCasino,
)
fn({ key: 'sampleHmm', name: 'Sample an HMM', role: 'simulation', random: true, notes: HMM }, hmm.sampleHmm)
fn({ key: 'crfStructure', name: 'Linear-chain CRF structure', role: 'construction', notes: CRF }, crf.crfStructure)
fn(
  {
    key: 'linearChainCrf',
    name: 'Linear-chain CRF',
    role: 'construction',
    notes: CRF,
    cite: ['lafferty2001', 'sutton2012'],
  },
  crf.linearChainCrf,
)
fn({ key: 'crfPotentials', name: 'CRF log-potentials', role: 'construction', notes: CRF }, crf.crfPotentials)
fn(
  { key: 'crfFactorGraph', name: 'CRF factor graph', role: 'construction', notes: [...CRF, 'factor-graph'] },
  crf.crfFactorGraph,
)
fn({ key: 'crfMarginals', name: 'CRF marginals (forward–backward)', role: 'inference', notes: CRF }, crf.crfMarginals)
fn({ key: 'crfViterbi', name: 'CRF decoding (Viterbi)', role: 'inference', notes: CRF }, crf.crfViterbi)
fn({ key: 'crfScore', name: 'CRF score of a labelling', role: 'estimator', notes: CRF }, crf.crfScore)
fn(
  {
    key: 'crfLogLikelihood',
    name: 'CRF conditional log-likelihood',
    role: 'estimator',
    notes: CRF,
    cite: ['sutton2012'],
  },
  crf.crfLogLikelihood,
)
fn(
  {
    key: 'crfGradient',
    name: 'CRF log-likelihood gradient',
    summary: 'Observed minus expected feature counts.',
    role: 'estimator',
    notes: CRF,
    cite: ['sutton2012'],
  },
  crf.crfGradient,
)

const TCRF = ['conditional-random-field', 'conditional-random-field-variants']
fn(
  {
    key: 'templateCrf',
    name: 'Linear-chain CRF over feature templates',
    summary: 'A CRF whose features are CRF++ template strings conjoined with the current label (U) or label pair (B).',
    role: 'construction',
    notes: TCRF,
    cite: ['lafferty2001', 'sutton2012', 'twomey2016'],
  },
  templateCrf.templateCrf,
)
fn({ key: 'crfProblem', name: 'CRF training problem', role: 'construction', notes: TCRF }, templateCrf.crfProblem)
fn(
  {
    key: 'templateCrfMarginals',
    name: 'Template CRF marginals (forward–backward)',
    role: 'inference',
    notes: TCRF,
    cite: ['twomey2016'],
  },
  templateCrf.templateCrfMarginals,
)
fn(
  { key: 'templateCrfViterbi', name: 'Template CRF decoding (Viterbi)', role: 'inference', notes: TCRF },
  templateCrf.templateCrfViterbi,
)
fn(
  {
    key: 'templateCrfPosterior',
    name: 'Template CRF posterior (max-marginal) decoding',
    role: 'inference',
    notes: TCRF,
  },
  templateCrf.templateCrfPosterior,
)
fn(
  {
    key: 'crfNegLogLikelihood',
    name: 'Template CRF negative log-likelihood and gradient',
    summary: 'Expected minus observed feature counts from the node and pairwise marginals.',
    role: 'estimator',
    notes: TCRF,
    cite: ['sutton2012'],
  },
  templateCrf.crfNegLogLikelihood,
)
fn(
  {
    key: 'firingFeatures',
    name: 'Features firing at a position',
    role: 'property',
    notes: TCRF,
  },
  templateCrf.firingFeatures,
)
fn(
  {
    key: 'crfTrainingRun',
    name: 'Template CRF training run',
    role: 'fit',
    summary: 'Index, encode and train a template CRF, yielding the model and its curves after every step.',
    notes: TCRF,
  },
  templateCrf.crfTrainingRun,
)

fn(
  {
    key: 'toyPosCorpus',
    name: 'Toy part-of-speech corpus',
    summary: 'Short tagged English sentences with ambiguous words, as CRF++ rows (word, suffix, shape).',
    role: 'construction',
    notes: ['sequence-labelling', ...TCRF],
  },
  toyTagging.toyPosCorpus,
)

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/sequence-models')
algorithm(
  {
    key: 'crfTraining',
    name: 'Template CRF training',
    summary: 'L-BFGS, OWL-QN (L1 + L2), SGD or Adam on the regularised conditional log-likelihood.',
    problem: 'sequence',
    state: { iterate: 'weights', objective: 'objective', flags: ['converged', 'diverged', 'stalled'] },
    notes: [...TCRF, 'quasi-newton-methods'],
    cite: ['sutton2012', 'liu1989', 'tsuruoka2009'],
  },
  templateCrf.crfTraining,
)

/** The algorithms of the module. */
export const sequenceModelAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', templateCrf) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module. */
export const sequenceModelFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', hmm, crf, templateCrf, toyTagging) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
