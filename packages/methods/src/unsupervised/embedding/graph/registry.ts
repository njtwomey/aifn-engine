/** The registry of `aifn-methods/unsupervised/embedding/graph`: node embeddings by factorisation and random walks. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as netmf from './netmf'
import * as walks from './random-walk-embedding'
import * as shallow from './shallow-embedding'

const fn = definer<FunctionInfo>('function', 'unsupervised/embedding/graph')

fn(
  {
    key: 'netmfMatrix',
    name: 'NetMF matrix',
    tex: '\\log\\max\\Big(\\tfrac{\\mathrm{vol}(G)}{bT}\\big(\\textstyle\\sum_{r=1}^{T} P^r\\big) D^{-1}, 1\\Big)',
    summary: 'The matrix DeepWalk with window T and b negatives implicitly factorises, in closed form.',
    role: 'construction',
    notes: ['graph-embedding-as-matrix-factorisation', 'random-walk-embeddings', 'node-embeddings'],
    cite: ['levy2014'],
  },
  netmf.netmfMatrix,
)
fn(
  {
    key: 'nodeSimilarity',
    name: 'Node similarity matrix',
    summary: 'Adjacency, two-hop path counts or the NetMF walk similarity of an undirected graph.',
    role: 'construction',
    notes: ['node-embeddings', 'graph-embedding-as-matrix-factorisation'],
    cite: ['hamilton2020grl'],
  },
  shallow.nodeSimilarity,
)
fn(
  {
    key: 'dotProductEmbedding',
    name: 'Dot-product node embedding',
    tex: 'Z = V_d \\Lambda_d^{1/2},\\quad ZZ^\\top \\approx S',
    summary: 'The best rank-d fit ZZᵀ of a symmetric similarity, from its leading positive eigenpairs.',
    role: 'fit',
    notes: ['node-embeddings', 'graph-embedding-as-matrix-factorisation'],
    cite: ['eckart1936', 'hamilton2020grl'],
  },
  shallow.dotProductEmbedding,
)
fn(
  {
    key: 'reconstructionError',
    name: 'Relative reconstruction error',
    tex: '\\lVert S - ZZ^\\top \\rVert_F^2 / \\lVert S \\rVert_F^2',
    role: 'property',
    notes: ['node-embeddings'],
  },
  shallow.reconstructionError,
)
fn(
  {
    key: 'node2vecTransition',
    name: 'node2vec transition',
    tex: 'P(x \\mid t, v) \\propto w_{vx}\\,\\alpha_{pq}(t, x)',
    summary: 'The probabilities of the next node of a second-order walk, biased by the return and in-out parameters.',
    role: 'property',
    notes: ['random-walk-embeddings'],
    cite: ['grover2016node2vec'],
  },
  walks.node2vecTransition,
)
fn(
  {
    key: 'node2vecWalks',
    name: 'node2vec walks',
    summary: 'Truncated second-order random walks from every node, DeepWalk’s uniform walks when p = q = 1.',
    role: 'simulation',
    random: true,
    notes: ['random-walk-embeddings', 'node-embeddings'],
    cite: ['grover2016node2vec'],
  },
  walks.node2vecWalks,
)
fn(
  {
    key: 'walkCooccurrence',
    name: 'Walk co-occurrence counts',
    summary: 'Symmetric counts of node pairs within a window of each other on random walks.',
    role: 'estimator',
    notes: ['random-walk-embeddings', 'graph-embedding-as-matrix-factorisation'],
  },
  walks.walkCooccurrence,
)
fn(
  {
    key: 'skipGram',
    name: 'Skip-gram with negative sampling on walks',
    tex: "-\\log\\sigma(u_c^\\top v_w) - \\textstyle\\sum_{c'} \\log\\sigma(-u_{c'}^\\top v_w)",
    summary: 'Node vectors fitted by SGD to the (centre, context) pairs of random walks, with unigram^0.75 negatives.',
    role: 'fit',
    random: true,
    notes: ['random-walk-embeddings', 'node-embeddings'],
    cite: ['mikolov2013b', 'levy2014', 'grover2016node2vec'],
  },
  walks.skipGram,
)

/** A registry table of the module: entries keyed by name, each a function with its `info`. */
type Table = Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

/**
 * The functions of the module, keyed by name, each with its `info`.
 *
 * @example The functions and the notes they serve
 * for (const [key, entry] of Object.entries(graphEmbeddingFunctions)) print(key, entry.info.notes)
 */
export const graphEmbeddingFunctions: Table = entries<FunctionInfo>('function', netmf, shallow, walks) as Table
