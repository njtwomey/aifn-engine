/**
 * `aifn-compute/graph/structured`: structured graphs, the one structure for intentional models, with typed node roles,
 * plates and templates held compactly.
 *
 * - The structure: a `StructuredGraph` is a `Graph` whose nodes carry a `NodeRole` (observed, latent, factor,
 *   deterministic, parameter; `VARIABLE_ROLES` are the random ones), whose edges may be directed or not and carry a
 *   `Lag` inside a template, and whose `Group`s are plates (sizes, index symbols, nesting, ragged sizes) or templates
 *   (chains, lattices, trees).
 * - Building: `structured` (a builder), `structuredGraph` (a plain specification), and the templates `chainTemplate`
 *   (an HMM or a linear-chain CRF), `latticeTemplate` (an Ising or Potts model), `treeTemplate` and `repeatedSlices`
 *   (a dynamic Bayesian network). Lookups: `nodeIndex`, `nodesWithRole`, `groupChain`, `groupSizes`.
 * - Expanding: `unroll` makes one node per copy at the given sizes (`SizeBindings`); `copyName` and `copiesOf` relate
 *   copies to their compact node.
 * - Queries: `shape` (chain, tree, lattice, DAG or general, so inference takes a fast path), `chainOrder`,
 *   `interactionScopes`, `markovBlanket` and `groupCounts`.
 * - Drawing: `toDiagram` gives the lab plain diagram data (plate notation, highlighted Markov blankets).
 * - `structuredFunctions`: the registry entries of the functions.
 *
 * Nodes and groups are referred to by name, and a malformed specification or an unbound size throws `AifnError`. The
 * model language of `aifn-compute/inference/model`, factor graphs, LDA's plate diagram and a linear-chain CRF are all
 * built on it.
 */

export {
  VARIABLE_ROLES,
  type GraphShape,
  type Group,
  type GroupKind,
  type Lag,
  type NodeRole,
  type SizeSpec,
  type StructuredEdge,
  type StructuredGraph,
  type StructuredNode,
} from './types'
export {
  chainTemplate,
  groupChain,
  groupSizes,
  latticeTemplate,
  nodeIndex,
  nodesWithRole,
  repeatedSlices,
  structured,
  structuredGraph,
  treeTemplate,
  type EdgeSpec,
  type GroupSpec,
  type NodeSpec,
  type StructuredBuilder,
  type StructuredScope,
  type StructuredSpec,
  type TemplateOptions,
} from './build'
export { copiesOf, copyName, unroll, type SizeBindings } from './unroll'
export { chainOrder, groupCounts, interactionScopes, markovBlanket, shape, type Blanket } from './shape'
export {
  toDiagram,
  type DiagramData,
  type DiagramEdgeData,
  type DiagramGroupData,
  type DiagramNodeData,
  type DiagramOptions,
} from './diagram'
export { structuredFunctions } from './registry'
