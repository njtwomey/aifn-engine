/** The decoders (step-through algorithms) and logit processors of `aifn-compute/nn/decoding`, registered with their notes. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as decoders from './decoders'
import * as processors from './processors'

const algorithm = definer<AlgorithmInfo>('algorithm', 'nn/decoding')
const fn = definer<FunctionInfo>('function', 'nn/decoding')
const notes = ['decoding-strategies']

algorithm(
  {
    key: 'greedyDecoding',
    name: 'Greedy decoding',
    summary: 'Append the most probable next token, one token per step.',
    problem: 'sequence',
    state: { iterate: 'tokens', objective: 'logProb', flags: ['terminated'] },
    notes,
  },
  decoders.greedyDecoding,
)
algorithm(
  {
    key: 'samplingDecoding',
    name: 'Sampling with temperature, top-k, top-p and repetition penalty',
    summary: 'Draw each next token from the processed next-token distribution.',
    problem: 'sequence',
    state: { iterate: 'tokens', objective: 'logProb', flags: ['terminated'] },
    random: true,
    notes,
    cite: ['holtzman2020', 'fan2018'],
  },
  decoders.samplingDecoding,
)
algorithm(
  {
    key: 'beamSearch',
    name: 'Beam search',
    summary: 'Keep the B best partial sequences by length-penalised log-probability, recording the search tree.',
    problem: 'sequence',
    state: { iterate: 'beams', flags: ['terminated'] },
    notes,
    cite: ['wu2016'],
  },
  decoders.beamSearch,
)
algorithm(
  {
    key: 'speculativeDecoding',
    name: 'Speculative decoding',
    summary: 'Draft γ tokens with a small model, verify them with the target, keep the target’s distribution exactly.',
    problem: 'sequence',
    state: { iterate: 'tokens', flags: ['terminated'] },
    random: true,
    notes: ['speculative-decoding', ...notes],
    cite: ['leviathan2023', 'chen2023b'],
  },
  decoders.speculativeDecoding,
)

fn(
  {
    key: 'nextTokenDistribution',
    name: 'Processed next-token distribution',
    role: 'transform',
    summary: 'Repetition penalty, temperature, top-k and top-p applied to logits, with the support kept.',
    notes,
    cite: ['holtzman2020', 'fan2018'],
  },
  processors.nextTokenDistribution,
)
fn({ key: 'applyTemperature', name: 'Temperature', role: 'transform', notes }, processors.applyTemperature)
fn({ key: 'applyTopK', name: 'Top-k filtering', role: 'transform', notes, cite: ['fan2018'] }, processors.applyTopK)
fn(
  { key: 'applyTopP', name: 'Nucleus (top-p) filtering', role: 'transform', notes, cite: ['holtzman2020'] },
  processors.applyTopP,
)
fn(
  { key: 'applyRepetitionPenalty', name: 'Repetition penalty', role: 'transform', notes },
  processors.applyRepetitionPenalty,
)
fn(
  {
    key: 'lengthPenalty',
    name: 'GNMT length penalty',
    tex: '((5 + |Y|)/6)^\\alpha',
    role: 'property',
    notes,
    cite: ['wu2016'],
  },
  decoders.lengthPenalty,
)
fn(
  {
    key: 'expectedTokensPerCall',
    name: 'Speculative decoding speed-up',
    tex: '(1 - \\alpha^{\\gamma+1})/(1 - \\alpha)',
    role: 'property',
    notes: ['speculative-decoding'],
    cite: ['leviathan2023'],
  },
  decoders.expectedTokensPerCall,
)

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>

/** The decoders, keyed by factory name. */
export const decodingAlgorithms = entries<AlgorithmInfo>('algorithm', decoders) as Table<AlgorithmInfo>
/** The processors and helpers, keyed by name. */
export const decodingFunctions = entries<FunctionInfo>('function', decoders, processors) as Table<FunctionInfo>
