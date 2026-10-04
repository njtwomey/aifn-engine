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

/** The functions of the module, keyed by name. */
export const structuresFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', structures) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
