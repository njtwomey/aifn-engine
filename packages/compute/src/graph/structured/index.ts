/**
 * `aifn-compute/graph/structured`: structured graphs, the one structure for intentional models: typed node roles (observed,
 * latent, factor, deterministic, parameter), groups (plates with sizes, index symbols and nesting) and templates
 * (chains, lattices, trees, repeated slices) held compactly; `unroll` expands them, `shape` reports chain, tree,
 * lattice, DAG or general so inference takes a fast path, and `toDiagram` gives the lab plain diagram data. The model
 * language of `aifn-compute/inference/model`, factor graphs, LDA's plate diagram and a linear-chain CRF are all built on it.
 *
 * ```ts
 * const hmm = chainTemplate('T', { observed: 'x' })   // z_{t−1} → z_t, z_t → x_t
 * shape(hmm)                                           // 'chain'
 * const explicit = unroll(hmm, { T: 5 })               // z[0] … z[4], x[0] … x[4]
 * ```
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
