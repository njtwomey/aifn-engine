/**
 * `aifn-compute/graph/structures`: standard, random and point-cloud graphs as data graphs, as networkx's generators.
 *
 * - Standard graphs: `chainGraph`, `cycleGraph`, `starGraph`, `completeGraph`, `completeBipartiteGraph`,
 *   `balancedTreeGraph` (level order, rooted at 0) and `gridGraph` (lattices with 4 or 8 neighbours, optionally
 *   periodic).
 * - Random graphs, drawn from a `Stream`: `erdosRenyiGraph` (independent edges), `randomDag` (oriented along a
 *   topological order), `wattsStrogatzGraph` (a rewired ring: small world), `barabasiAlbertGraph` (preferential
 *   attachment: a heavy-tailed degree distribution), `stochasticBlockModel` (edge probabilities set by the blocks of
 *   the two ends; `blockMembership` gives each node's block) and `randomRecursiveTree` (each node joins a uniform
 *   earlier one).
 * - Bottlenecks: `barbellGraph`, two cliques joined by a path.
 * - Graphs from points, edges weighted by distance: `kNearestNeighbourGraph` ($k$ nearest, directed, symmetric or
 *   mutual), `epsilonBallGraph` (every pair within $\varepsilon$) and `bMatchingGraph` (shortest pairs first, at most
 *   $b$ edges per node).
 * - Hypergraphs as graphs: `cliqueExpansion` (members of a hyperedge joined pairwise, weighted by the hyperedges they
 *   share) and `starExpansion` (a node per hyperedge, joined to its members).
 *
 * Every generator returns a plain `Graph` on nodes $0, \dots, n - 1$, undirected unless asked otherwise, and throws
 * `AifnError` on a parameter out of range. The same stream gives the same random graph: the random generators draw
 * from named children of it, except `randomRecursiveTree`, which advances it so that one stream yields many trees.
 */

export {
  balancedTreeGraph,
  barabasiAlbertGraph,
  barbellGraph,
  blockMembership,
  bMatchingGraph,
  chainGraph,
  cliqueExpansion,
  completeBipartiteGraph,
  completeGraph,
  cycleGraph,
  epsilonBallGraph,
  erdosRenyiGraph,
  gridGraph,
  kNearestNeighbourGraph,
  randomDag,
  randomRecursiveTree,
  starExpansion,
  starGraph,
  stochasticBlockModel,
  wattsStrogatzGraph,
  type BlockProbabilities,
  type GridOptions,
  type PointGraphOptions,
  type StructureOptions,
} from './structures'
export { structuresFunctions } from './registry'
