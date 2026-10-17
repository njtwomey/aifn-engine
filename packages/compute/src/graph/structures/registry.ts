/** The graph generators of `aifn-compute/graph/structures`, registered as functions. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as structures from './structures'

const fn = definer<FunctionInfo>('function', 'graph/structures')

fn({ key: 'chainGraph', name: 'Chain graph', role: 'construction' }, structures.chainGraph)
fn({ key: 'cycleGraph', name: 'Cycle graph', role: 'construction' }, structures.cycleGraph)
fn({ key: 'starGraph', name: 'Star graph', role: 'construction' }, structures.starGraph)
fn({ key: 'completeGraph', name: 'Complete graph', role: 'construction' }, structures.completeGraph)
fn(
  { key: 'completeBipartiteGraph', name: 'Complete bipartite graph', role: 'construction' },
  structures.completeBipartiteGraph,
)
fn(
  { key: 'gridGraph', name: 'Grid graph', role: 'construction', notes: ['markov-random-field', 'ising-model'] },
  structures.gridGraph,
)
fn({ key: 'balancedTreeGraph', name: 'Balanced tree', role: 'construction' }, structures.balancedTreeGraph)
fn(
  { key: 'erdosRenyiGraph', name: 'Erdős–Rényi random graph', role: 'simulation', random: true },
  structures.erdosRenyiGraph,
)
fn(
  { key: 'barabasiAlbertGraph', name: 'Barabási–Albert preferential attachment', role: 'simulation', random: true },
  structures.barabasiAlbertGraph,
)
fn(
  { key: 'wattsStrogatzGraph', name: 'Watts–Strogatz small world', role: 'simulation', random: true },
  structures.wattsStrogatzGraph,
)
fn(
  { key: 'randomDag', name: 'Random DAG', role: 'simulation', random: true, notes: ['bayesian-network'] },
  structures.randomDag,
)
fn(
  {
    key: 'kNearestNeighbourGraph',
    name: 'k-nearest-neighbour graph',
    role: 'construction',
    notes: ['k-nearest-neighbours', 'spectral-clustering', 'isomap', 'uniform-manifold-approximation-and-projection'],
  },
  structures.kNearestNeighbourGraph,
)
fn(
  {
    key: 'epsilonBallGraph',
    name: 'ε-ball graph',
    role: 'construction',
    notes: ['spectral-clustering', 'laplacian-eigenmaps'],
  },
  structures.epsilonBallGraph,
)
fn(
  {
    key: 'bMatchingGraph',
    name: 'Greedy b-matching graph',
    summary: 'Join the closest pairs first while both ends have fewer than b edges: a near-regular graph.',
    role: 'construction',
    notes: [
      'graph-construction-for-semi-supervised-learning',
      'manifold-regularisation',
      'semi-supervised-regression',
      'evaluating-semi-supervised-learning',
    ],
    cite: ['jebara2009bmatching'],
  },
  structures.bMatchingGraph,
)
fn(
  {
    key: 'stochasticBlockModel',
    name: 'Stochastic block model',
    tex: '\\Pr[i \\sim j] = P_{c_i c_j}',
    summary: 'A random graph whose edge probabilities depend only on the blocks of the two ends.',
    role: 'simulation',
    random: true,
    notes: [
      'semi-supervised-node-classification',
      'graph-neural-networks-under-heterophily',
      'graph-learning-benchmarks',
      'link-prediction-with-graph-neural-networks',
      'oversmoothing',
      'residual-and-jumping-knowledge-graph-networks',
    ],
    cite: ['deshpande2018csbm'],
  },
  structures.stochasticBlockModel,
)
fn({ key: 'blockMembership', name: 'Block membership', role: 'construction' }, structures.blockMembership)
fn(
  {
    key: 'randomRecursiveTree',
    name: 'Random recursive tree',
    summary: 'Each new node joins a uniformly drawn earlier node.',
    role: 'simulation',
    random: true,
    notes: ['graph-pooling-and-readout'],
  },
  structures.randomRecursiveTree,
)
fn(
  {
    key: 'barbellGraph',
    name: 'Barbell graph',
    summary: 'Two cliques joined by a path: a bottleneck that squashes messages.',
    role: 'construction',
    notes: ['oversquashing-and-graph-rewiring'],
    cite: ['topping2022curvature'],
  },
  structures.barbellGraph,
)
fn(
  {
    key: 'cliqueExpansion',
    name: 'Clique expansion of a hypergraph',
    summary: 'Join every two members of a hyperedge, weighted by the number of hyperedges they share.',
    role: 'construction',
    notes: ['hypergraph-neural-networks'],
    cite: ['zhou2006hypergraph'],
  },
  structures.cliqueExpansion,
)
fn(
  {
    key: 'starExpansion',
    name: 'Star expansion of a hypergraph',
    summary: 'A node per hyperedge, joined to its members, as member-to-hyperedge and hyperedge-to-member graphs.',
    role: 'construction',
    notes: ['hypergraph-neural-networks'],
    cite: ['zhou2006hypergraph'],
  },
  structures.starExpansion,
)

/** The functions of the module, keyed by name. */
export const structuresFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', structures) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
