/**
 * `aifn-compute/graph/structures`: standard graphs as data graphs: chains, cycles, stars, complete and complete bipartite
 * graphs, balanced trees, lattices (4 or 8 neighbours, periodic), random DAGs and random graphs (Erdős–Rényi,
 * Watts–Strogatz, Barabási–Albert), and k-nearest-neighbour and ε-ball graphs from points, as networkx's generators.
 */

export {
  balancedTreeGraph,
  barabasiAlbertGraph,
  chainGraph,
  completeBipartiteGraph,
  completeGraph,
  cycleGraph,
  epsilonBallGraph,
  erdosRenyiGraph,
  gridGraph,
  kNearestNeighbourGraph,
  randomDag,
  starGraph,
  wattsStrogatzGraph,
  type GridOptions,
  type PointGraphOptions,
  type StructureOptions,
} from './structures'
export { structuresFunctions } from './registry'
