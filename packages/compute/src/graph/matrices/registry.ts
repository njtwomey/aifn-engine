/**
 * The functions of `aifn-compute/graph/matrices`, registered with their display names, roles and the notes they serve,
 * so a worker or a note can address each by key.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as encodings from './encodings'
import * as matrices from './matrices'

const fn = definer<FunctionInfo>('function', 'graph/matrices')
const SPECTRAL = ['spectral-clustering', 'laplacian-eigenmaps']

fn(
  { key: 'adjacencyMatrix', name: 'Adjacency matrix', role: 'construction', notes: ['graph-convolutional-network'] },
  matrices.adjacencyMatrix,
)
fn({ key: 'degreeMatrix', name: 'Degree matrix', role: 'construction', notes: SPECTRAL }, matrices.degreeMatrix)
fn({ key: 'degrees', name: 'Degrees', role: 'property' }, matrices.degrees)
fn({ key: 'incidenceMatrix', name: 'Incidence matrix', role: 'construction' }, matrices.incidenceMatrix)
fn(
  {
    key: 'laplacian',
    name: 'Graph Laplacian',
    tex: 'L = D - A',
    summary: 'The combinatorial, symmetric-normalised or random-walk Laplacian, of a graph or an affinity matrix.',
    role: 'construction',
    notes: [
      ...SPECTRAL,
      'graph-convolutional-network',
      'diffusion-maps',
      'manifold-regularisation',
      'semi-supervised-regression',
      'evaluating-semi-supervised-learning',
    ],
  },
  matrices.laplacian,
)
fn(
  {
    key: 'rescaledLaplacian',
    name: 'Rescaled Laplacian',
    tex: '\\tilde L = 2L/\\lambda_{\\max} - I',
    summary: 'The Laplacian with its spectrum moved to [-1, 1]: the argument of a Chebyshev graph filter.',
    role: 'construction',
    notes: ['spectral-graph-convolution'],
    cite: ['hammond2011', 'defferrard2016'],
  },
  matrices.rescaledLaplacian,
)
fn(
  {
    key: 'hypergraphIncidence',
    name: 'Hypergraph incidence matrix',
    tex: 'H_{ve} = [v \\in e]',
    summary: 'One row per node and one column per hyperedge, 1 where the node is a member.',
    role: 'construction',
    notes: ['hypergraph-neural-networks'],
    cite: ['zhou2006hypergraph'],
  },
  matrices.hypergraphIncidence,
)
const ENCODINGS = ['positional-and-structural-encodings', 'expressive-power-of-message-passing']
fn(
  {
    key: 'randomWalkEncoding',
    name: 'Random-walk structural encoding',
    tex: 'p_v = \\big((P^k)_{vv}\\big)_{k=1}^{K},\\; P = D^{-1}A',
    summary: 'The probability that a random walk from each node is back after 1 to K steps.',
    role: 'construction',
    notes: ENCODINGS,
    cite: ['dwivedi2022'],
  },
  encodings.randomWalkEncoding,
)
fn(
  {
    key: 'laplacianEncoding',
    name: 'Laplacian positional encoding',
    summary: 'The eigenvectors of the normalised Laplacian for the smallest non-zero eigenvalues, one column each.',
    role: 'construction',
    notes: ENCODINGS,
    cite: ['dwivedi2020benchmarking', 'dwivedi2022'],
  },
  encodings.laplacianEncoding,
)
fn(
  {
    key: 'flipSigns',
    name: 'Random sign flips',
    summary: 'Flip the sign of each eigenvector column with probability one half, for training-time augmentation.',
    role: 'transform',
    random: true,
    notes: ['positional-and-structural-encodings'],
    cite: ['dwivedi2020benchmarking'],
  },
  encodings.flipSigns,
)

/** The functions of the module, keyed by name. */
export const matricesFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', matrices, encodings) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
