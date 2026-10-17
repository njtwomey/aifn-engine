/**
 * `aifn-compute/graph/isomorphism`: the Weisfeiler-Leman tests, which bound what message-passing networks can tell
 * apart.
 *
 * - Colour refinement (1-WL): `weisfeilerLeman` (every round at once, with histograms, the stable round and the first
 *   round that separates the graphs) and `weisfeilerLemanSteps` (one round per step, for a figure).
 * - The folklore 2-WL test on ordered pairs, as strong as 3-WL: `folkloreWeisfeilerLeman2`.
 * - Subgraph networks: `subgraphBag` (node-deleted, edge-deleted and ego-network policies) and
 *   `subgraphWeisfeilerLeman` (the DS-WL test over the bags).
 *
 * Several graphs are refined with one shared palette, numbered canonically (sorted signatures), so colours compare
 * across graphs and do not depend on the order of the nodes or of the graphs. Results are plain data; the tests
 * throw `ShapeError` for initial labels that do not match the graphs and `DomainError` where an undirected graph is
 * needed.
 */

export {
  folkloreWeisfeilerLeman2,
  weisfeilerLeman,
  weisfeilerLemanSteps,
  type FolkloreTrace,
  type WeisfeilerLemanOptions,
  type WeisfeilerLemanState,
  type WeisfeilerLemanTrace,
} from './weisfeiler-leman'
export {
  subgraphBag,
  subgraphWeisfeilerLeman,
  type Subgraph,
  type SubgraphOptions,
  type SubgraphPolicy,
  type SubgraphWlResult,
} from './subgraphs'
export { isomorphismAlgorithms, isomorphismFunctions } from './registry'
