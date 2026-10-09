/**
 * The registry of `aifn-compute/probability/markov`: the chain simulator as an algorithm (`problem: 'map'`, random),
 * and the exact computations on a transition matrix as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as chain from './chain'

const MODULE = 'probability/markov'
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)
const fn = definer<FunctionInfo>('function', MODULE)

const CHAIN = ['markov-chain']
const CITE = ['norris1997', 'levin2017']

algorithm(
  {
    key: 'markovChainSteps',
    name: 'Markov chain, step by step',
    summary: 'A walker drawn from the chain, its visit counts, and the exact distribution p₀Pᵗ beside it.',
    problem: 'map',
    state: { iterate: 'distribution', objective: 'distance', flags: [] },
    notes: [...CHAIN, 'random-walk', 'markov-chain-monte-carlo'],
    cite: CITE,
    random: true,
  },
  chain.markovChainSteps,
)

fn(
  {
    key: 'transitionMatrix',
    name: 'Transition matrix',
    summary: 'Check that a matrix is square, non-negative and row-stochastic.',
    role: 'construction',
    notes: CHAIN,
    cite: CITE,
  },
  chain.transitionMatrix,
)
fn(
  {
    key: 'nStepTransition',
    name: 'n-step transition matrix',
    tex: 'P^n',
    summary: 'The matrix power Pⁿ by repeated squaring.',
    role: 'property',
    notes: CHAIN,
    cite: CITE,
  },
  chain.nStepTransition,
)
fn(
  {
    key: 'distributionAfter',
    name: 'Distribution after t steps',
    tex: 'p_0 P^t',
    summary: 'The distribution of X_t from an initial distribution or state.',
    role: 'property',
    notes: CHAIN,
    cite: CITE,
  },
  chain.distributionAfter,
)
fn(
  {
    key: 'classifyStates',
    name: 'Classification of states',
    summary: 'Communicating classes, closed (recurrent) and transient states, absorbing states and periods.',
    role: 'property',
    notes: CHAIN,
    cite: CITE,
  },
  chain.classifyStates,
)
fn(
  {
    key: 'stationaryDistribution',
    name: 'Stationary distribution',
    tex: '\\pi = \\pi P',
    summary: 'The unique π with πP = π of a chain with one closed class, by one linear solve.',
    role: 'property',
    notes: [...CHAIN, 'markov-chain-monte-carlo', 'pixie-random-walk'],
    cite: CITE,
  },
  chain.stationaryDistribution,
)
fn(
  {
    key: 'stationaryDistributions',
    name: 'Stationary distributions',
    summary: 'One stationary distribution per closed class; every stationary distribution mixes them.',
    role: 'property',
    notes: CHAIN,
    cite: CITE,
  },
  chain.stationaryDistributions,
)
fn(
  {
    key: 'meanReturnTimes',
    name: 'Mean return times',
    tex: '1/\\pi_i',
    summary: "Kac's formula: the expected return time to each state of an irreducible chain is 1/πᵢ.",
    role: 'property',
    notes: CHAIN,
    cite: CITE,
  },
  chain.meanReturnTimes,
)
fn(
  {
    key: 'absorption',
    name: 'Absorption probabilities and times',
    tex: 'N = (I - Q)^{-1}',
    summary: 'The fundamental matrix, absorption probabilities B = NR and expected steps t = N1, with their variance.',
    role: 'property',
    notes: ['gamblers-ruin', ...CHAIN],
    cite: CITE,
  },
  chain.absorption,
)
fn(
  {
    key: 'hittingProbabilities',
    name: 'Hitting probabilities',
    summary: 'The probability of ever reaching a target set, by first-step analysis.',
    role: 'property',
    notes: [...CHAIN, 'gamblers-ruin'],
    cite: CITE,
  },
  chain.hittingProbabilities,
)
fn(
  {
    key: 'expectedHittingTimes',
    name: 'Expected hitting times',
    summary: 'The expected time to reach a target set (∞ where it may never be reached).',
    role: 'property',
    notes: [...CHAIN, 'random-walk'],
    cite: CITE,
  },
  chain.expectedHittingTimes,
)
fn(
  {
    key: 'distanceToStationarity',
    name: 'Distance to stationarity',
    tex: 'd(t) = \\max_x \\lVert P^t(x, \\cdot) - \\pi \\rVert_{TV}',
    summary: 'The total-variation distance to π from every start, for t = 0 … T.',
    role: 'property',
    notes: [...CHAIN, 'markov-chain-monte-carlo'],
    cite: ['levin2017'],
  },
  chain.distanceToStationarity,
)
fn(
  {
    key: 'mixingTime',
    name: 'Mixing time',
    tex: 't_{mix}(\\varepsilon)',
    summary: 'The first t with d(t) ≤ ε, with the relaxation-time bounds for a reversible chain.',
    role: 'property',
    notes: [...CHAIN, 'markov-chain-monte-carlo'],
    cite: ['levin2017'],
  },
  chain.mixingTime,
)
fn(
  {
    key: 'spectralGap',
    name: 'Spectral gap',
    tex: '1 - \\lambda_\\star',
    summary: 'The eigenvalues of P, its spectral and absolute spectral gaps, and the relaxation time.',
    role: 'property',
    notes: [...CHAIN, 'eigendecomposition'],
    cite: ['levin2017'],
  },
  chain.spectralGap,
)
fn(
  {
    key: 'isReversible',
    name: 'Reversibility (detailed balance)',
    tex: '\\pi_i P_{ij} = \\pi_j P_{ji}',
    summary: 'Whether the chain satisfies detailed balance with its stationary distribution.',
    role: 'property',
    notes: [...CHAIN, 'metropolis-hastings'],
    cite: CITE,
  },
  chain.isReversible,
)
fn(
  {
    key: 'simulateChain',
    name: 'Simulate a chain',
    summary: 'A path X₀, …, X_T drawn from the chain.',
    role: 'simulation',
    notes: [...CHAIN, 'random-walk'],
    cite: CITE,
    random: true,
  },
  chain.simulateChain,
)

/** A registry table: the entries of one kind, keyed by name. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>

/** The algorithms of the module, keyed by factory name. */
export const markovAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>('algorithm', chain) as Table<AlgorithmInfo>

/** The functions of the module, keyed by name. */
export const markovFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', chain) as Table<FunctionInfo>
