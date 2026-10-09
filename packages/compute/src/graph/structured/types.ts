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

/** The roles that are random variables: `observed`, `latent` and `deterministic` (not factors or parameters). */
export const VARIABLE_ROLES: readonly NodeRole[] = ['observed', 'latent', 'deterministic']

/**
 * The attributes of one node (`graph.attributes[i]` for node $i$). `name` is unique in the graph; `group` is the
 * innermost group holding the node (null for none); `data` is the caller's payload (a distribution, a table).
 * `source` and `index` are set by `unroll`: the compact node a copy came from and its index along each axis of its
 * groups, outermost first.
 */
export interface StructuredNode<D = unknown> {
  /** The node's name, unique in the graph; edges and queries refer to nodes by it. */
  name: string
  /** What the node stands for. */
  role: NodeRole
  /** The innermost group holding the node, or null for a node outside every group. */
  group: string | null
  /** TeX for display (default: the name). */
  label?: string
  /** The caller's payload: a distribution, a table. */
  data?: D
  /** Set on copies made by `unroll`: the name of the compact node. */
  source?: string
  /** Set on copies made by `unroll`: the copy's index along each axis of its groups, outermost first. */
  index?: readonly Index[]
}

/**
 * How an edge couples copies inside a template: `1` (or any integer $k \ge 1$) joins copy $t - k$ of `from` to copy
 * $t$ of `to` along a chain; `[di, dj]` joins site $(i - d_i, j - d_j)$ to $(i, j)$ on a lattice; `'parent'` joins a
 * tree node's parent to it.
 */
export type Lag = number | readonly number[] | 'parent'

/** The extra fields of a structured edge: its direction, an optional lag inside a template and a label. */
export interface StructuredEdge {
  /** True for a conditional dependence (parent to child), false for a symmetric link (a factor or an MRF edge). */
  directed: boolean
  /** Inside a template, which copies the edge joins; left out, the edge joins copies at the same index. */
  lag?: Lag
  /** A label to draw on the edge. */
  label?: string
}

/** A size: a fixed count, or a name bound when the graph is unrolled. */
export type SizeSpec = Size | string

/**
 * The kinds of group: a `plate` holds exchangeable copies with no links between them; a `chain` holds copies
 * $0, \dots, T - 1$ in order (a cycle when `periodic`); a `lattice` holds an $m \times n$ grid of sites (a torus when
 * `periodic`); a `tree` holds the nodes of a complete `arity`-ary tree of the given depth in level order.
 */
export type GroupKind = 'plate' | 'chain' | 'lattice' | 'tree'

/**
 * A group of nodes repeated together. `size` is one size (plate, chain), `[rows, cols]` (lattice) or the depth
 * (tree: depth 0 is the root alone). `index` names the index symbol of each axis (`n`; `i`, `j` for a lattice).
 * Groups nest through `parent`: a nested plate's size may differ per index of its parent (a ragged size).
 */
export interface Group {
  /** The group's name, unique in the graph; nodes and nested groups refer to it. */
  name: string
  /** A plate or a kind of template. */
  kind: GroupKind
  /** One size (plate, chain), `[rows, cols]` (lattice) or the depth (tree); a number or a named size. */
  size: SizeSpec | readonly SizeSpec[]
  /** The index symbol of each axis. */
  index: readonly string[]
  /** The group this one is nested in, or null at the top level. */
  parent: string | null
  /** TeX for display (default: the size). */
  label?: string
  /** Chain: a cycle (copy $T - 1$ links to copy 0); lattice: a torus. */
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
  /** One record per node: its name, role, group and data. */
  attributes: readonly StructuredNode<D>[]
  /** The plates and templates, in any order (nesting is by `parent`). */
  groups: readonly Group[]
  /** The named sizes the groups use, bound by `unroll`. */
  sizes: readonly string[]
  /** The model's name. */
  name?: string
  /** True for a graph made by `unroll`: one node per copy. */
  unrolled?: boolean
}

/**
 * The shape of a graph's latent structure, which picks an inference path: `chain` (forward–backward, Viterbi),
 * `tree` (exact belief propagation), `lattice` (sweeps; loopy BP), `dag` (a directed acyclic graph with loops in its
 * moral graph: elimination) or `general`.
 */
export type GraphShape = 'chain' | 'tree' | 'lattice' | 'dag' | 'general'
