/**
 * The registry of `aifn-methods/unsupervised/embedding/neighbour`: the t-SNE, UMAP and PaCMAP optimisations as
 * step-through algorithms (the roles of their states' fields in `state`), and the pieces they are built from
 * (affinities, the fuzzy graph, the curve parameters, the spectral start, the pairs and the phase weights) as
 * functions with the notes they serve. The estimators (`tsne`, `umap`, `pacmap`) are registered as models beside their
 * code.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as pacmap from './pacmap'
import * as tsne from './tsne'
import * as umap from './umap'

/** A registry table of the module: entries keyed by name, each a function with its `info`. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
/** Registers a step-through algorithm of the module under `unsupervised/embedding/neighbour`. */
const algorithm = definer<AlgorithmInfo>('algorithm', 'unsupervised/embedding/neighbour')
/** Registers a function of the module under `unsupervised/embedding/neighbour`. */
const fn = definer<FunctionInfo>('function', 'unsupervised/embedding/neighbour')
const TSNE = ['t-distributed-stochastic-neighbour-embedding']
const UMAP = ['uniform-manifold-approximation-and-projection']

algorithm(
  {
    key: 'tsneSteps',
    name: 't-SNE',
    summary: 'Gradient steps with gains and early exaggeration on the KL divergence between neighbour distributions.',
    problem: 'objective',
    state: { iterate: 'embedding', objective: 'kl', flags: [] },
    random: true,
    notes: TSNE,
    cite: ['vandermaaten2008'],
  },
  tsne.tsneSteps,
)
algorithm(
  {
    key: 'umapSteps',
    name: 'UMAP layout',
    summary: 'One epoch of attractive edge samples and negative samples per step.',
    problem: 'graph',
    state: { iterate: 'embedding', flags: [] },
    random: true,
    notes: UMAP,
    cite: ['mcinnes2018'],
  },
  umap.umapSteps,
)
fn(
  { key: 'jointProbabilities', name: 't-SNE joint probabilities', role: 'construction', notes: TSNE },
  tsne.jointProbabilities,
)
fn(
  {
    key: 'perplexityCalibration',
    name: 'Perplexity calibration',
    summary: 'Per-point Gaussian bandwidths by bisection to a target perplexity.',
    role: 'solver',
    notes: TSNE,
  },
  tsne.perplexityCalibration,
)
fn({ key: 'fuzzyGraph', name: 'UMAP fuzzy graph', role: 'construction', notes: UMAP }, umap.fuzzyGraph)
fn({ key: 'curveParameters', name: 'UMAP curve parameters (a, b)', role: 'fit', notes: UMAP }, umap.curveParameters)
fn(
  {
    key: 'spectralLayout',
    name: 'Spectral initial layout',
    role: 'construction',
    notes: [...UMAP, 'laplacian-eigenmaps'],
  },
  umap.spectralLayout,
)

const PACMAP = ['pacmap-trimap-and-largevis']
algorithm(
  {
    key: 'pacmapSteps',
    name: 'PaCMAP',
    summary: 'Adam steps on neighbour, mid-near and further pairs with loss weights that change in three phases.',
    problem: 'objective',
    state: { iterate: 'embedding', objective: 'loss', flags: ['diverged'] },
    random: true,
    notes: PACMAP,
    cite: ['wang2021b'],
  },
  pacmap.pacmapSteps,
)
fn(
  { key: 'pacmapPairs', name: 'PaCMAP pairs', role: 'construction', random: true, notes: PACMAP, cite: ['wang2021b'] },
  pacmap.pacmapPairs,
)
fn({ key: 'pacmapWeights', name: 'PaCMAP phase weights', role: 'property', notes: PACMAP }, pacmap.pacmapWeights)

/**
 * The step-through algorithms of the module, keyed by factory name: `tsneSteps`, `umapSteps` and `pacmapSteps`, each
 * with its `info`.
 *
 * @example The algorithms and what their states report
 * for (const [key, entry] of Object.entries(neighbourEmbeddingAlgorithms)) print(key, entry.info.state)
 */
export const neighbourEmbeddingAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  tsne,
  umap,
  pacmap,
) as Table<AlgorithmInfo>
/**
 * The functions of the module, keyed by name: `jointProbabilities`, `perplexityCalibration`, `fuzzyGraph`,
 * `curveParameters`, `spectralLayout`, `pacmapPairs` and `pacmapWeights`, each with its `info`.
 *
 * @example The functions and their roles
 * for (const [key, entry] of Object.entries(neighbourEmbeddingFunctions)) print(key, entry.info.role)
 */
export const neighbourEmbeddingFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  tsne,
  umap,
  pacmap,
) as Table<FunctionInfo>
