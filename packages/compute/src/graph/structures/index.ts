/**
 * `aifn-compute/graph/structures`: standard, random and point-cloud graphs as data graphs, as networkx's generators.
 *
 * - Standard graphs: `chainGraph`, `cycleGraph`, `starGraph`, `completeGraph`, `completeBipartiteGraph`,
 *   `balancedTreeGraph` (level order, rooted at 0) and `gridGraph` (lattices with 4 or 8 neighbours, optionally
 *   periodic).
 * - Random graphs, drawn from a `Stream`: `erdosRenyiGraph` (independent edges), `randomDag` (oriented along a
 *   topological order), `wattsStrogatzGraph` (a rewired ring: small world) and `barabasiAlbertGraph` (preferential
 *   attachment: a heavy-tailed degree distribution).
 * - Graphs from points, edges weighted by distance: `kNearestNeighbourGraph` ($k$ nearest, directed, symmetric or
 *   mutual) and `epsilonBallGraph` (every pair within $\varepsilon$).
 *
 * Every generator returns a plain `Graph` on nodes $0, \dots, n - 1$, undirected unless asked otherwise, and throws
 * `AifnError` on a parameter out of range. The same stream gives the same random graph.
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
