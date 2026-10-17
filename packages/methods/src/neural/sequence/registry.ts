/**
 * The registry of `aifn-methods/neural/sequence`: the recall task and probe, the finite-precision parity study, the
 * low-rank and hashing attention matrices and the convolution counts, as functions.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as approximate from './approximate'
import * as cost from './cost'
import * as parity from './parity'
import * as recall from './recall'

/** A registry table: each registered export of the module by name, with its description. */
type Table = Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

const fn = definer<FunctionInfo>('function', 'neural/sequence')
const LOW_RANK = 'low-rank-and-hashing-attention'
const PARITY = 'expressiveness-of-linear-recurrences'

fn(
  {
    key: 'multiQueryAssociativeRecall',
    name: 'Multi-query associative recall',
    summary: 'N key–value pairs, then every key again as a query whose answer is the value that followed it.',
    role: 'simulation',
    random: true,
    notes: ['long-range-and-recall-benchmarks'],
    cite: ['arora2024zoology'],
  },
  recall.multiQueryAssociativeRecall,
)
fn(
  {
    key: 'associativeRecallByDimension',
    name: 'Recall by state size',
    summary: 'MQAR accuracy of softmax attention, linear attention and the delta rule with ideal projections.',
    role: 'simulation',
    random: true,
    notes: ['long-range-and-recall-benchmarks', 'delta-rule-and-test-time-training'],
    cite: ['arora2024zoology', 'schlag2021'],
  },
  recall.associativeRecallByDimension,
)
fn(
  {
    key: 'roundToGrid',
    name: 'Fixed-point truncation',
    tex: '\\operatorname{trunc}(x 2^b)/2^b',
    role: 'transform',
    notes: [PARITY],
  },
  parity.roundToGrid,
)
fn(
  {
    key: 'parityStates',
    name: 'Parity recurrence states',
    tex: 'h_t = a(x_t) h_{t-1}',
    role: 'simulation',
    notes: [PARITY],
    cite: ['grazzi2024'],
  },
  parity.parityStates,
)
fn({ key: 'stateValues', name: 'States by count of ones', role: 'property', notes: [PARITY] }, parity.stateValues)
fn(
  { key: 'onesUntilZero', name: 'Ones until the state truncates to zero', role: 'property', notes: [PARITY] },
  parity.onesUntilZero,
)
fn(
  {
    key: 'parityAccuracyByLength',
    name: 'Parity accuracy at finite precision',
    summary: 'The best parity accuracy of a one-state linear recurrence stored in fixed point, exactly, by length.',
    role: 'property',
    notes: [PARITY],
    cite: ['grazzi2024', 'sarrof2024'],
  },
  parity.parityAccuracyByLength,
)
fn(
  { key: 'topicTokens', name: 'Topic-structured tokens', role: 'simulation', random: true, notes: [LOW_RANK] },
  approximate.topicTokens,
)
fn(
  {
    key: 'lowRankSeries',
    name: 'Truncated SVD of an attention matrix',
    role: 'transform',
    summary: 'The singular values and vectors of P, from which lowRankAt builds the best rank-k approximation.',
    notes: [LOW_RANK],
  },
  approximate.lowRankSeries,
)
fn(
  {
    key: 'lowRankAt',
    name: 'Best rank-k approximation',
    tex: 'P_k = \\sum_{t \\le k} \\sigma_t u_t v_t^\\top',
    role: 'construction',
    notes: [LOW_RANK],
  },
  approximate.lowRankAt,
)
fn(
  { key: 'segmentMeans', name: 'Segment-mean projection', role: 'construction', notes: [LOW_RANK] },
  approximate.segmentMeans,
)
fn(
  {
    key: 'linformerMatrix',
    name: 'Linformer attention matrix',
    tex: '\\operatorname{softmax}(\\beta Q (E K)^\\top) E',
    role: 'transform',
    notes: [LOW_RANK, 'efficient-attention'],
    cite: ['wang2020linformer'],
  },
  approximate.linformerMatrix,
)
fn(
  {
    key: 'nystromMatrix',
    name: 'Nyströmformer attention matrix',
    tex: '\\tilde F \\tilde A^{+} \\tilde B',
    role: 'transform',
    notes: [LOW_RANK],
    cite: ['xiong2021nystromformer'],
  },
  approximate.nystromMatrix,
)
fn(
  {
    key: 'angularBuckets',
    name: 'Angular LSH buckets',
    tex: 'h(x) = \\operatorname{argmax}([xR; -xR])',
    role: 'transform',
    notes: [LOW_RANK],
    cite: ['andoni2015falconn', 'kitaev2020reformer'],
  },
  approximate.angularBuckets,
)
fn(
  {
    key: 'angularRotations',
    name: 'Random rotations for angular LSH',
    role: 'construction',
    random: true,
    notes: [LOW_RANK],
    cite: ['andoni2015falconn'],
  },
  approximate.angularRotations,
)
fn(
  {
    key: 'lshAttentionMatrix',
    name: 'LSH attention matrix',
    role: 'transform',
    summary: 'Softmax attention restricted to the keys that share a hash bucket with the query in some round.',
    notes: [LOW_RANK],
    cite: ['kitaev2020reformer'],
  },
  approximate.lshAttentionMatrix,
)
fn(
  {
    key: 'rowTotalVariation',
    name: 'Mean row total variation',
    tex: '\\tfrac12 \\sum_j |\\hat P_{ij} - P_{ij}|',
    role: 'property',
    notes: [LOW_RANK],
  },
  approximate.rowTotalVariation,
)
fn(
  {
    key: 'convolutionMultiplies',
    name: 'Long-convolution multiply counts',
    role: 'property',
    summary: 'Multiplies of one causal convolution of length L, directly (L(L + 1)/2) and by zero-padded FFT.',
    notes: ['long-convolution-models'],
    cite: ['poli2023hyena'],
  },
  cost.convolutionMultiplies,
)

/** The functions of the module, keyed by name. */
export const efficientSequenceFunctions: Table = entries<FunctionInfo>(
  'function',
  approximate,
  cost,
  parity,
  recall,
) as Table
