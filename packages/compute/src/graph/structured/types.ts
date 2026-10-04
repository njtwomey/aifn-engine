/**
 * Structured graphs: the one structure for intentional models (design: "graphs as one core structure, intentional
 * first"). A structured graph is a base `Graph` whose nodes carry a role, a group and the caller's data, whose edges
 * may be directed or undirected and may couple neighbouring copies of a template (`lag`), and whose groups are plates
 * (exchangeable copies) or templates (chains, lattices, trees). It is held compactly and expanded by `unroll`.
 *
 * Plate notation follows Buntine (1994), "Operations for learning with graphical models", JAIR 2; factor graphs follow
 * Kschischang, Frey & Loeliger (2001); templates (dynamic Bayesian networks and repeated slices) follow Koller &
 * Friedman (2009), "Probabilistic Graphical Models", ch. 6.
 */

import type { Graph, Index, Size } from 'aifn-compute/foundation/contracts'

/**
 * What a node stands for: an `observed` or a `latent` random variable, a `factor` (a potential over its neighbours,
 * as in a factor graph), a `deterministic` function of its parents, or a `parameter` (a fixed quantity or
 * hyperparameter, drawn small).
 */
export type NodeRole = 'observed' | 'latent' | 'factor' | 'deterministic' | 'parameter'

/** The roles that are random variables. */
export const VARIABLE_ROLES: readonly NodeRole[] = ['observed', 'latent', 'deterministic']

/**
 * The attributes of one node (`graph.attributes[i]` for node i). `name` is unique in the graph; `group` is the
 * innermost group holding the node (null for none); `data` is the caller's payload (a distribution, a table).
 * `source` and `index` are set by `unroll`: the compact node a copy came from and its index along each axis of its
 * groups, outermost first.
 */
export interface StructuredNode<D = unknown> {
  name: string
  role: NodeRole
  group: string | null
  /** TeX for display (default: the name). */
  label?: string
  data?: D
  /** Set on copies made by `unroll`: the name of the compact node. */
  source?: string
  /** Set on copies made by `unroll`: the copy's index along each axis of its groups, outermost first. */
  index?: readonly Index[]
}

/**
 * How an edge couples copies inside a template: `1` (or any k ≥ 1) joins copy t − k of `from` to copy t of `to` along
 * a chain; `[di, dj]` joins site (i − di, j − dj) to (i, j) on a lattice; `'parent'` joins a tree node's parent to it.
 */
export type Lag = number | readonly number[] | 'parent'

/** The extra fields of a structured edge: its direction, an optional lag inside a template and a label. */
export interface StructuredEdge {
  /** True for a conditional dependence (parent → child), false for a symmetric link (a factor or an MRF edge). */
  directed: boolean
  lag?: Lag
  label?: string
}

/** A size: a fixed count, or a name bound when the graph is unrolled. */
export type SizeSpec = Size | string

/**
 * The kinds of group: a `plate` holds exchangeable copies with no links between them; a `chain` holds copies
 * 0 … T − 1 in order (a cycle when `periodic`); a `lattice` holds an m × n grid of sites (a torus when `periodic`);
 * a `tree` holds the nodes of a complete `arity`-ary tree of the given depth in level order.
 */
export type GroupKind = 'plate' | 'chain' | 'lattice' | 'tree'

/**
 * A group of nodes repeated together. `size` is one size (plate, chain), `[rows, cols]` (lattice) or the depth
 * (tree: depth 0 is the root alone). `index` names the index symbol of each axis (`n`; `i`, `j` for a lattice).
 * Groups nest through `parent`: a nested plate's size may differ per index of its parent (a ragged size).
 */
export interface Group {
  name: string
  kind: GroupKind
  size: SizeSpec | readonly SizeSpec[]
  index: readonly string[]
  parent: string | null
  /** TeX for display (default: the size). */
  label?: string
  /** Chain: a cycle (copy T − 1 links to copy 0); lattice: a torus. */
  periodic?: boolean
  /** Lattice: 4 or 8 neighbours (informative: the lag edges carry the links). */
  neighbourhood?: 4 | 8
  /** Tree: children per node. */
  arity?: Size
}

/**
 * A structured graph (`kind: 'graph'`): a base graph with typed nodes (`attributes`), structured edges and groups.
 * `sizes` lists the named sizes its groups use. `unrolled` marks a graph made by `unroll`, whose groups are kept only
 * so that a diagram can draw them around the copies.
 */
export interface StructuredGraph<D = unknown> extends Graph<StructuredNode<D>, StructuredEdge> {
  attributes: readonly StructuredNode<D>[]
  groups: readonly Group[]
  sizes: readonly string[]
  name?: string
  unrolled?: boolean
}

/**
 * The shape of a graph's latent structure, which picks an inference path: `chain` (forward–backward, Viterbi),
 * `tree` (exact belief propagation), `lattice` (sweeps; loopy BP), `dag` (a directed acyclic graph with loops in its
 * moral graph: elimination) or `general`.
 */
export type GraphShape = 'chain' | 'tree' | 'lattice' | 'dag' | 'general'
