/**
 * Graphs and rooted trees as plain data (design S §2.11): the structures that classification trees, dendrograms,
 * Huffman codes, search trees, factor graphs and computation graphs all return. aifn produces structure; the lab draws
 * it. Both carry the `kind` brand, which their constructors set.
 */

import type { Kinded } from './kinds'
import type { Index, Size } from './numbers'

/** An edge from `from` to `to` (or between them, in an undirected graph), with an optional weight (default 1). */
export interface Edge {
  /** The node the edge leaves. */
  from: Index
  /** The node the edge enters. */
  to: Index
  /** The edge's weight; 1 when absent. */
  weight?: number
}

/**
 * A graph on nodes $0, \dots, n - 1$, $n$ = `nodes` (`kind: 'graph'`). `directed` defaults to true; in an undirected
 * graph every edge is used in both directions. `labels`, when given, has one entry per node and is for display only.
 *
 * Nodes and edges are typed: `N` is the attribute record of a node (`attributes[i]` belongs to node $i$) and `E` the
 * extra fields of an edge, so one base carries data graphs (no attributes), factor graphs and the structured graphs of
 * `aifn-compute/graph/structured` (roles, groups and templates). Algorithms read only `nodes`, `edges` and `directed`.
 */
export interface Graph<N extends object = object, E extends object = object> extends Kinded<'graph'> {
  /** The number of nodes. */
  nodes: Size
  /** The edges, each with the extra fields `E`. */
  edges: readonly (Edge & E)[]
  /** False for an undirected graph; true when absent. */
  directed?: boolean
  /** A display name per node. */
  labels?: readonly string[]
  /** One attribute record per node, when the graph has typed nodes. */
  attributes?: readonly N[]
}

/** A node of a tree: the caller's data `N` plus the structure. */
export type TreeNode<N extends object = object> = N & {
  /** The node's index in `tree.nodes`. */
  id: Index
  /** The index of the parent, or null at the root. */
  parent: Index | null
  /** The indices of the children, in order. */
  children: Index[]
  /**
   * For ordered $k$-ary trees (`tree.arity` set): the node's position among its parent's slots (binary: 0 left, 1
   * right).
   */
  slot?: Index
  /** A display label. */
  label?: string
  /** Dendrogram convention: the height at which the node's children merge (a distance); leaves at 0. */
  height?: number
}

/** The edge from a node's parent to it: the caller's data `E` plus an optional label and weight (branch length). */
export type TreeEdge<E extends object = object> = E & { label?: string; weight?: number }

/**
 * A rooted tree (`kind: 'tree'`). `nodes[i].id === i`; `edges[c]` is the edge from node $c$'s parent to $c$ (null at
 * the root). With `arity` set (e.g. 2), children carry a `slot` so a lone child is still a left or a right child.
 */
export interface Tree<N extends object = object, E extends object = object> extends Kinded<'tree'> {
  /** Every node, indexed by its `id`. */
  nodes: TreeNode<N>[]
  /** The index of the root. */
  root: Index
  /** The edge into each node from its parent, indexed by the child's `id`; null at the root. */
  edges: (TreeEdge<E> | null)[]
  /** The number of child slots of an ordered tree (2 for binary); absent for an unordered tree. */
  arity?: Size
}
