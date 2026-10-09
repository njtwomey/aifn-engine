/**
 * Rooted trees as plain data: one type for every tree in aifn (Huffman codes, decision trees, dendrograms,
 * branch-and-bound search trees, breadth- and depth-first parent trees, spanning trees, expression trees), so one
 * renderer can draw them all.
 *
 * The types `Tree`, `TreeNode` and `TreeEdge` are defined in `aifn-compute/foundation/contracts`. A node is known by
 * its `id`, its index in `tree.nodes`; it holds its `parent` (null at the root) and its `children`, ordered left to
 * right, and in an ordered $k$-ary tree (`tree.arity` set) its `slot` among its parent's (0 left and 1 right in a
 * binary tree). `tree.edges[c]` is the edge from node $c$'s parent to $c$ (null at the root), with an optional label
 * and weight (a branch length). Nodes and edges carry the caller's data beside these fields.
 *
 * The constructors check the structure and throw `DomainError` or `ShapeError`; queries and traversals return node
 * ids. Trees are serialisable (JSON round-trips them) and treated as immutable: every function returns new data.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Graph } from './graph'
import type { Tree, TreeEdge, TreeNode } from 'aifn-compute/foundation/contracts'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { TreeNode, TreeEdge, Tree } from 'aifn-compute/foundation/contracts'

/** Options shared by the constructors: per-node labels and data, per-edge data (keyed by child). */
export interface TreeOptions<N extends object, E extends object> {
  /** A display label for node $i$ at index $i$ (TeX between dollar signs allowed); undefined for none. */
  labels?: readonly (string | undefined)[]
  /** Node data for node $i$. */
  data?: (i: number) => N
  /** Edge data for the edge from the parent of node $c$ to $c$. */
  edge?: (child: number, parent: number) => TreeEdge<E>
}

/**
 * Check that a tree is one: the root is a node, every child is a node whose `parent` points back, and every node is
 * reached exactly once from the root. Throws `DomainError` otherwise.
 *
 * @param t The tree.
 * @param where The caller's name for error messages.
 * @returns `t`, unchanged.
 */
function check<N extends object, E extends object>(t: Tree<N, E>, where: string): Tree<N, E> {
  const n = t.nodes.length
  if (!(t.root >= 0 && t.root < n)) throw new DomainError(where, `${where}: root ${t.root} is not a node`)
  // Every node must be reached exactly once from the root.
  const seen = new Uint8Array(n)
  const stack = [t.root]
  let count = 0
  while (stack.length) {
    const v = stack.pop()!
    if (seen[v]) throw new DomainError(where, `${where}: node ${v} is reached twice (a cycle or a shared child)`)
    seen[v] = 1
    count++
    for (const c of t.nodes[v].children) {
      if (!(c >= 0 && c < n)) throw new DomainError(where, `${where}: node ${v} has an unknown child ${c}`)
      if (t.nodes[c].parent !== v) throw new DomainError(where, `${where}: node ${c}'s parent is not ${v}`)
      stack.push(c)
    }
  }
  if (count !== n) throw new DomainError(where, `${where}: ${n - count} node(s) are not reachable from the root`)
  return t
}

/**
 * Assemble and check a tree from its parent and child lists, with the options' labels and data.
 *
 * @param parent The parent of each node, null at the root.
 * @param children The ordered children of each node (used as given, not copied).
 * @param root The root's id.
 * @param options Labels, node data and edge data.
 * @param where The caller's name for error messages.
 * @param slots The slot of each node among its parent's, when the tree is ordered (left out: no slots).
 * @param arity The tree's arity, when ordered (left out: none).
 * @returns The tree, checked as `check` does.
 */
function build<N extends object, E extends object>(
  parent: readonly (number | null)[],
  children: number[][],
  root: number,
  options: TreeOptions<N, E>,
  where: string,
  slots?: readonly (number | undefined)[],
  arity?: number,
): Tree<N, E> {
  const nodes = parent.map((p, i) => {
    const node = {
      ...(options.data?.(i) ?? ({} as N)),
      id: i,
      parent: p,
      children: children[i],
    } as TreeNode<N>
    const label = options.labels?.[i]
    if (label !== undefined) node.label = label
    if (slots?.[i] !== undefined) node.slot = slots[i]
    return node
  })
  const edges = parent.map((p, c) => (p === null ? null : (options.edge?.(c, p) ?? ({} as TreeEdge<E>))))
  return check({ kind: 'tree', nodes, root, edges, ...(arity !== undefined && { arity }) }, where)
}

/**
 * A tree from a parent array: `parents[i]` is node $i$'s parent, or a negative number or null for the root (exactly
 * one, else `ShapeError`). Children are ordered by id unless `order` (a list of every node id) says otherwise, e.g. a
 * visit order. Throws `DomainError` on an unknown parent, a cycle, or a node that `order` leaves out.
 *
 * @param parents The parent of each node.
 * @param options Labels, node and edge data, and `order`, the order in which children join their parents.
 * @returns The tree, with node $i$ as id $i$.
 *
 * @example A root with two children, the first with two of its own
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print('children:', t.nodes.map((n) => n.children))
 * print('height:', height(t))
 *
 * @example A visit order sets the order of siblings
 * const t = treeFromParents([-1, 0, 0], { order: [0, 2, 1] })
 * print("root's children:", t.nodes[0].children)
 */
export function treeFromParents<N extends object = object, E extends object = object>(
  parents: ArrayLike<number | null>,
  options: TreeOptions<N, E> & { order?: ArrayLike<number> } = {},
): Tree<N, E> {
  const n = parents.length
  const parent: (number | null)[] = Array.from({ length: n }, (_, i) => {
    const p = parents[i]
    return p === null || p < 0 ? null : p
  })
  const roots = parent.flatMap((p, i) => (p === null ? [i] : []))
  if (roots.length !== 1)
    throw new ShapeError('treeFromParents', `treeFromParents: expected one root, found ${roots.length}`)
  const children: number[][] = parent.map(() => [])
  const order = options.order ? Array.from(options.order) : parent.map((_, i) => i)
  for (const i of order) {
    const p = parent[i]
    if (p === null) continue
    if (!(p >= 0 && p < n))
      throw new DomainError('treeFromParents', `treeFromParents: node ${i} has an unknown parent ${p}`)
    children[p].push(i)
  }
  return build(parent, children, roots[0], options, 'treeFromParents')
}

/**
 * A tree from ordered child lists: `children[i]` lists node $i$'s children, left to right. Throws `DomainError` on an
 * unknown child, a node listed as a child twice, or a node not reached from `root`.
 *
 * @param children The ordered children of each node; the number of lists is the number of nodes.
 * @param root The root's id.
 * @param options Labels, node data and edge data.
 * @returns The tree, with node $i$ as id $i$.
 *
 * @example Child lists to parents
 * const t = treeFromChildren([[1, 2], [3], [], []])
 * print('parents:', t.nodes.map((n) => n.parent))
 * print('leaves:', leaves(t))
 */
export function treeFromChildren<N extends object = object, E extends object = object>(
  children: readonly (readonly number[])[],
  root = 0,
  options: TreeOptions<N, E> = {},
): Tree<N, E> {
  const parent: (number | null)[] = children.map(() => null)
  children.forEach((cs, p) =>
    cs.forEach((c) => {
      if (!(c >= 0 && c < children.length))
        throw new DomainError('treeFromChildren', `treeFromChildren: node ${p} has an unknown child ${c}`)
      if (parent[c] !== null) throw new DomainError('treeFromChildren', `treeFromChildren: node ${c} has two parents`)
      parent[c] = p
    }),
  )
  return build(
    parent,
    children.map((cs) => [...cs]),
    root,
    options,
    'treeFromChildren',
  )
}

/** A nested description of a tree: node data, an optional label and edge (from its parent), and children. */
export type NestedTree<N extends object = object, E extends object = object> = N & {
  /** Display text (TeX between dollar signs allowed). */
  label?: string
  /** A dendrogram's merge distance (0 at the leaves). */
  height?: number
  /** The edge from the parent to this node. */
  edge?: TreeEdge<E>
  /** The subtrees, left to right (none for a leaf). */
  children?: readonly NestedTree<N, E>[]
}

/** A binary tree described by nesting: `left` and `right` subtrees, either may be absent. */
export type NestedBinaryTree<N extends object = object, E extends object = object> = N & {
  /** Display text (TeX between dollar signs allowed). */
  label?: string
  /** A dendrogram's merge distance (0 at the leaves). */
  height?: number
  /** The edge from the parent to this node. */
  edge?: TreeEdge<E>
  /** The left subtree, or none. */
  left?: NestedBinaryTree<N, E> | null
  /** The right subtree, or none. */
  right?: NestedBinaryTree<N, E> | null
}

/**
 * Splits a nested node into its data and structure (the data keeps every field that is not structural).
 *
 * @param x The nested node.
 * @returns Its data without `children`, `left`, `right`, `edge`, `label` and `height`, and those three last apart.
 */
function unnest<N extends object, E extends object>(
  x: NestedTree<N, E> | NestedBinaryTree<N, E>,
): { data: N; edge?: TreeEdge<E>; label?: string; height?: number } {
  const { edge, label, height, ...rest } = x as NestedTree<N, E> & NestedBinaryTree<N, E>
  const data = { ...rest } as Record<string, unknown>
  delete data.children
  delete data.left
  delete data.right
  return {
    data: data as N,
    edge,
    ...(label !== undefined && { label }),
    ...(height !== undefined && { height }),
  }
}

/**
 * A tree from nested objects, e.g. `{ label: '+', children: [{ label: 'x' }, { label: '1' }] }`. Ids in pre-order, so
 * the root is 0. Every field other than the structural ones (`children`, `edge`, `label`, `height`) is node data.
 *
 * @param nested The root, with its subtrees nested in `children`.
 * @returns The tree.
 *
 * @example An expression tree: a sum of a variable and a product
 * const product = { label: '*', children: [{ label: '2' }, { label: 'y' }] }
 * const t = treeFromNested({ label: '+', children: [{ label: 'x' }, product] })
 * print('pre-order:', preOrder(t).map((i) => t.nodes[i].label))
 * print('post-order:', postOrder(t).map((i) => t.nodes[i].label))
 */
export function treeFromNested<N extends object = object, E extends object = object>(
  nested: NestedTree<N, E>,
): Tree<N, E> {
  const nodes: TreeNode<N>[] = []
  const edges: (TreeEdge<E> | null)[] = []
  const visit = (x: NestedTree<N, E>, parent: number | null): number => {
    const id = nodes.length
    const { data, edge, label, height } = unnest(x)
    const node = { ...data, id, parent, children: [] as number[] } as TreeNode<N>
    if (label !== undefined) node.label = label
    if (height !== undefined) node.height = height
    nodes.push(node)
    edges.push(parent === null ? null : (edge ?? ({} as TreeEdge<E>)))
    for (const c of x.children ?? []) node.children.push(visit(c, id))
    return id
  }
  visit(nested, null)
  return { kind: 'tree', nodes, root: 0, edges }
}

/**
 * A binary tree (`arity: 2`) from nested objects with `left` and `right`; a lone child keeps its side (`slot` 0 left,
 * 1 right). Ids in pre-order, the left subtree before the right.
 *
 * @param nested The root, with its subtrees in `left` and `right`.
 * @returns The tree.
 *
 * @example A search tree read in order
 * const t = binaryTree({ label: '2', left: { label: '1' }, right: { label: '3', right: { label: '4' } } })
 * print('in-order:', inOrder(t).map((i) => t.nodes[i].label))
 * print('node 3 is a right child:', t.nodes[3].slot === 1)
 */
export function binaryTree<N extends object = object, E extends object = object>(
  nested: NestedBinaryTree<N, E>,
): Tree<N, E> {
  const nodes: TreeNode<N>[] = []
  const edges: (TreeEdge<E> | null)[] = []
  const visit = (x: NestedBinaryTree<N, E>, parent: number | null, slot?: number): number => {
    const id = nodes.length
    const { data, edge, label, height } = unnest(x)
    const node = { ...data, id, parent, children: [] as number[] } as TreeNode<N>
    if (slot !== undefined) node.slot = slot
    if (label !== undefined) node.label = label
    if (height !== undefined) node.height = height
    nodes.push(node)
    edges.push(parent === null ? null : (edge ?? ({} as TreeEdge<E>)))
    if (x.left) node.children.push(visit(x.left, id, 0))
    if (x.right) node.children.push(visit(x.right, id, 1))
    return id
  }
  visit(nested, null)
  return { kind: 'tree', nodes, root: 0, edges, arity: 2 }
}

/**
 * The left child (slot 0) of a node in a binary tree, or null. A child without a slot is read by its position among
 * the children, so a lone child without one is the left.
 *
 * @param tree The tree.
 * @param id The node.
 * @returns The left child's id, or null.
 *
 * @example A node with only a right child
 * const t = binaryTree({ label: 'a', right: { label: 'b' } })
 * print('left:', leftChild(t, 0), 'right:', rightChild(t, 0))
 */
export function leftChild(tree: Tree<object, object>, id: number): number | null {
  return tree.nodes[id].children.find((c) => (tree.nodes[c].slot ?? tree.nodes[id].children.indexOf(c)) === 0) ?? null
}

/**
 * The right child (slot 1) of a node in a binary tree, or null. A child without a slot is read by its position among
 * the children.
 *
 * @param tree The tree.
 * @param id The node.
 * @returns The right child's id, or null.
 *
 * @example Children of the root
 * const t = binaryTree({ label: 'a', left: { label: 'b' }, right: { label: 'c' } })
 * print('left:', leftChild(t, 0), 'right:', rightChild(t, 0))
 */
export function rightChild(tree: Tree<object, object>, id: number): number | null {
  return tree.nodes[id].children.find((c) => (tree.nodes[c].slot ?? tree.nodes[id].children.indexOf(c)) === 1) ?? null
}

// ── Queries ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Whether a node has no children.
 *
 * @param tree The tree.
 * @param id The node.
 * @returns True for a leaf.
 *
 * @example The root and a leaf
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print('root:', isLeaf(t, 0), 'node 3:', isLeaf(t, 3))
 */
export const isLeaf = (tree: Tree<object, object>, id: number): boolean => tree.nodes[id].children.length === 0

/**
 * Edges from the root to the node (the root has depth 0).
 *
 * @param tree The tree.
 * @param id The node.
 * @returns Its depth.
 *
 * @example Depths along a branch
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print('root:', depth(t, 0), 'node 1:', depth(t, 1), 'node 4:', depth(t, 4))
 */
export function depth(tree: Tree<object, object>, id: number): number {
  let d = 0
  for (let v = tree.nodes[id].parent; v !== null; v = tree.nodes[v].parent) d++
  return d
}

/**
 * The depth of every node, indexed by id.
 *
 * @param tree The tree.
 * @returns One depth per node.
 *
 * @example All depths at once
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print(depths(t))
 */
export function depths(tree: Tree<object, object>): number[] {
  const out = new Array<number>(tree.nodes.length).fill(0)
  for (const v of preOrder(tree)) {
    const p = tree.nodes[v].parent
    out[v] = p === null ? 0 : out[p] + 1
  }
  return out
}

/**
 * Edges on the longest downward path from the node (default the root) to a leaf: 0 for a leaf.
 *
 * @param tree The tree.
 * @param id The node whose subtree is measured.
 * @returns The height.
 *
 * @example The whole tree and a subtree
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print('tree:', height(t), 'under node 1:', height(t, 1), 'leaf 2:', height(t, 2))
 */
export function height(tree: Tree<object, object>, id: number = tree.root): number {
  return foldTree(tree, (_, hs) => (hs.length ? 1 + Math.max(...hs) : 0), id)
}

/**
 * The leaves under a node (default the root), left to right.
 *
 * @param tree The tree.
 * @param id The node whose subtree is searched (a leaf gives itself).
 * @returns The leaves' ids.
 *
 * @example Leaves of the tree and of a subtree
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print('all:', leaves(t), 'under node 1:', leaves(t, 1))
 */
export function leaves(tree: Tree<object, object>, id: number = tree.root): number[] {
  return preOrder(tree, id).filter((v) => tree.nodes[v].children.length === 0)
}

/**
 * The node and its ancestors, from the node up to the root.
 *
 * @param tree The tree.
 * @param id The node.
 * @returns Ids from `id` to the root.
 *
 * @example Up from a leaf
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print(pathToRoot(t, 4))
 */
export function pathToRoot(tree: Tree<object, object>, id: number): number[] {
  const path: number[] = []
  for (let v: number | null = id; v !== null; v = tree.nodes[v].parent) path.push(v)
  return path
}

/**
 * The path from the root down to the node (inclusive): the reverse of `pathToRoot`.
 *
 * @param tree The tree.
 * @param id The node.
 * @returns Ids from the root to `id`.
 *
 * @example Down to a leaf
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print(pathFromRoot(t, 4))
 */
export const pathFromRoot = (tree: Tree<object, object>, id: number): number[] => pathToRoot(tree, id).reverse()

/**
 * The node's proper ancestors, nearest first (its parent, and so on up to the root).
 *
 * @param tree The tree.
 * @param id The node.
 * @returns Ids from the parent to the root; empty for the root.
 *
 * @example Ancestors of a leaf and of the root
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print('node 4:', ancestors(t, 4), 'root:', ancestors(t, 0))
 */
export const ancestors = (tree: Tree<object, object>, id: number): number[] => pathToRoot(tree, id).slice(1)

/**
 * The lowest common ancestor of two nodes (a node is its own ancestor here). Throws `DomainError` when the walks up
 * never meet, which a checked tree rules out.
 *
 * @param tree The tree.
 * @param a A node.
 * @param b Another node.
 * @returns The deepest node above both (or equal to one of them).
 *
 * @example Siblings, a node and its parent's sibling, and a node with its ancestor
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print('3 and 4:', lca(t, 3, 4), '3 and 2:', lca(t, 3, 2), '1 and 4:', lca(t, 1, 4))
 */
export function lca(tree: Tree<object, object>, a: number, b: number): number {
  const up = new Set(pathToRoot(tree, a))
  for (let v: number | null = b; v !== null; v = tree.nodes[v].parent) if (up.has(v)) return v
  throw new DomainError('lca', 'lca: the nodes are not in one tree')
}

/**
 * The number of nodes in the subtree rooted at the node (itself included).
 *
 * @param tree The tree.
 * @param id The subtree's root (default the tree's).
 * @returns The node count.
 *
 * @example The whole tree and a subtree
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print('tree:', subtreeSize(t), 'under node 1:', subtreeSize(t, 1))
 */
export function subtreeSize(tree: Tree<object, object>, id: number = tree.root): number {
  return foldTree(tree, (_, sizes) => 1 + sizes.reduce((a, b) => a + b, 0), id)
}

// ── Traversals ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Node ids in pre-order (a node before its children, children left to right), from `id` (default the root).
 *
 * @param tree The tree.
 * @param id The subtree's root.
 * @returns The ids of the subtree.
 *
 * @example Pre-order of a small tree
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print(preOrder(t))
 */
export function preOrder(tree: Tree<object, object>, id: number = tree.root): number[] {
  const out: number[] = []
  const stack = [id]
  while (stack.length) {
    const v = stack.pop()!
    out.push(v)
    const cs = tree.nodes[v].children
    for (let i = cs.length - 1; i >= 0; i--) stack.push(cs[i])
  }
  return out
}

/**
 * Node ids in post-order (children left to right, then the node), from `id` (default the root).
 *
 * @param tree The tree.
 * @param id The subtree's root.
 * @returns The ids of the subtree.
 *
 * @example Post-order of a small tree
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print(postOrder(t))
 */
export function postOrder(tree: Tree<object, object>, id: number = tree.root): number[] {
  // Reverse of a pre-order that visits children right to left.
  const out: number[] = []
  const stack = [id]
  while (stack.length) {
    const v = stack.pop()!
    out.push(v)
    for (const c of tree.nodes[v].children) stack.push(c)
  }
  return out.reverse()
}

/**
 * Node ids level by level (breadth-first), each level left to right, from `id` (default the root).
 *
 * @param tree The tree.
 * @param id The subtree's root.
 * @returns The ids of the subtree.
 *
 * @example Level order against pre-order
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print('level order:', levelOrder(t))
 * print('pre-order:', preOrder(t))
 */
export function levelOrder(tree: Tree<object, object>, id: number = tree.root): number[] {
  const out = [id]
  for (let i = 0; i < out.length; i++) out.push(...tree.nodes[out[i]].children)
  return out
}

/**
 * Node ids in in-order (left subtree, node, right subtree) of a binary tree. Sides come from `slot` when set, else
 * from the child's position. Throws `DomainError` on a node with more than two children.
 *
 * @param tree The tree.
 * @param id The subtree's root (default the tree's).
 * @returns The ids of the subtree.
 *
 * @example A binary search tree's keys come out sorted
 * const two = { label: '2', left: { label: '1' }, right: { label: '3' } }
 * const t = binaryTree({ label: '4', left: two, right: { label: '5' } })
 * print(inOrder(t).map((i) => t.nodes[i].label))
 */
export function inOrder(tree: Tree<object, object>, id: number = tree.root): number[] {
  const out: number[] = []
  const visit = (v: number) => {
    if (tree.nodes[v].children.length > 2)
      throw new DomainError('inOrder', `inOrder: node ${v} has more than two children`)
    const l = leftChild(tree, v)
    const r = rightChild(tree, v)
    if (l !== null) visit(l)
    out.push(v)
    if (r !== null) visit(r)
  }
  visit(id)
  return out
}

// ── Transforms ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A tree with the same shape and new node (and optionally edge) data. `node(n)` returns the new data for node $n$;
 * the structural fields (`id`, `parent`, `children`, `slot`) are kept, and `label` and `height` too unless the new
 * data sets them. `edge(e, child)` maps each edge (default: keep). The old node data is not carried over.
 *
 * @param tree The tree (not modified).
 * @param node Gives the new data of a node, from the node and the tree.
 * @param edge Gives the new data of an edge, from the edge, its child's id and the tree; left out, edges are kept.
 * @returns The new tree, with the same arity.
 *
 * @example Label each node with its depth
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * const d = mapTree(t, (n) => ({ label: `depth ${depth(t, n.id)}` }))
 * print(d.nodes.map((n) => n.label))
 */
export function mapTree<N extends object, E extends object, M extends object, F extends object = E>(
  tree: Tree<N, E>,
  node: (n: TreeNode<N>, tree: Tree<N, E>) => M & { label?: string; height?: number },
  edge?: (e: TreeEdge<E>, child: number, tree: Tree<N, E>) => TreeEdge<F>,
): Tree<M, F> {
  const nodes = tree.nodes.map((n) => {
    const out = {
      ...(n.label !== undefined && { label: n.label }),
      ...(n.height !== undefined && { height: n.height }),
      ...node(n, tree),
      id: n.id,
      parent: n.parent,
      children: [...n.children],
    } as TreeNode<M>
    if (n.slot !== undefined) out.slot = n.slot
    return out
  })
  const edges = tree.edges.map((e, c) => (e === null ? null : edge ? edge(e, c, tree) : (e as unknown as TreeEdge<F>)))
  return { kind: 'tree', nodes, root: tree.root, edges, ...(tree.arity !== undefined && { arity: tree.arity }) }
}

/**
 * Folds a tree bottom-up: `f(node, childResults)` gets the results of the node's children, left to right, and the
 * fold returns the result at `id` (default the root). E.g. the subtree size is `foldTree(t, (_, s) => 1 + sum(s))`.
 *
 * @param tree The tree.
 * @param f Combines a node with its children's results (an empty list at a leaf).
 * @param id The subtree's root.
 * @returns The result at `id`.
 *
 * @example Count the leaves
 * const t = treeFromParents([-1, 0, 0, 1, 1])
 * print(foldTree(t, (_, below) => (below.length ? below.reduce((a, b) => a + b, 0) : 1)))
 */
export function foldTree<N extends object, E extends object, R>(
  tree: Tree<N, E>,
  f: (node: TreeNode<N>, children: R[]) => R,
  id: number = tree.root,
): R {
  const result = new Map<number, R>()
  for (const v of postOrder(tree, id)) {
    const n = tree.nodes[v]
    result.set(
      v,
      f(
        n,
        n.children.map((c) => result.get(c)!),
      ),
    )
  }
  return result.get(id)!
}

// ── Trees from graph searches ────────────────────────────────────────────────────────────────────────────────────────

/** A node of a spanning tree: the graph vertex it stands for. Labels come from `graph.labels`. */
export type SpanningTreeNode = { vertex: number }
/** An edge of a spanning tree: the index of the graph edge it uses ($-1$ if none was given) and its weight. */
export type SpanningTreeEdge = { edge: number; weight: number }

/**
 * How a search's tree is given: a parent per vertex ($-1$ for roots and unreached vertices), optionally with the graph
 * edge used to reach each vertex (`parentEdge`) and a visit order that orders siblings; or a set of undirected tree
 * edges (indices into `graph.edges`, e.g. a minimum spanning tree), oriented away from the root.
 */
export type SpanningInput =
  | { parent: ArrayLike<number>; parentEdge?: ArrayLike<number>; order?: ArrayLike<number> }
  | { edges: ArrayLike<number> }

/**
 * A parent array for either form of input. A parent array is used as given (`parentEdge` $-1$ and the order of the
 * vertex ids when left out); an edge set is oriented by a breadth-first walk from each root in turn, siblings in edge
 * order, skipping roots already reached.
 *
 * @param graph The graph whose edges `input.edges` refers to.
 * @param input The parent array or the edge set.
 * @param roots The vertices to walk an edge set from, in order (unused for a parent array).
 * @returns The parent and the graph edge into each vertex ($-1$ for none), and the visit order.
 */
function parentsOf(
  graph: Graph,
  input: SpanningInput,
  roots: readonly number[],
): { parent: Int32Array; parentEdge: Int32Array; order: number[] } {
  const V = graph.nodes
  if ('parent' in input) {
    const parent = Int32Array.from(input.parent)
    const parentEdge = input.parentEdge ? Int32Array.from(input.parentEdge) : new Int32Array(V).fill(-1)
    const order = input.order ? Array.from(input.order) : Array.from({ length: V }, (_, v) => v)
    return { parent, parentEdge, order }
  }
  // Orient the edge set by a breadth-first walk from each root, siblings in edge order.
  const adj: { to: number; edge: number }[][] = Array.from({ length: V }, () => [])
  for (const k of Array.from(input.edges)) {
    const e = graph.edges[k]
    adj[e.from].push({ to: e.to, edge: k })
    adj[e.to].push({ to: e.from, edge: k })
  }
  const parent = new Int32Array(V).fill(-1)
  const parentEdge = new Int32Array(V).fill(-1)
  const seen = new Uint8Array(V)
  const order: number[] = []
  for (const r of roots) {
    if (seen[r]) continue
    seen[r] = 1
    const queue = [r]
    for (let i = 0; i < queue.length; i++) {
      const v = queue[i]
      order.push(v)
      for (const a of adj[v])
        if (!seen[a.to]) {
          seen[a.to] = 1
          parent[a.to] = v
          parentEdge[a.to] = a.edge
          queue.push(a.to)
        }
    }
  }
  return { parent, parentEdge, order }
}

/**
 * The tree of the vertices under `root`, with tree ids in pre-order and siblings in visit order.
 *
 * @param graph The graph, for labels and edge weights.
 * @param root The vertex at the root.
 * @param parent The parent of each vertex ($-1$ for none).
 * @param parentEdge The graph edge into each vertex ($-1$ for none: the tree edge then has weight 1).
 * @param order The visit order, which orders siblings; a vertex left out of it is left out of the tree.
 * @returns The tree, its nodes carrying `vertex` and its edges `edge` and `weight`.
 */
function treeUnder(
  graph: Graph,
  root: number,
  parent: Int32Array,
  parentEdge: Int32Array,
  order: readonly number[],
): Tree<SpanningTreeNode, SpanningTreeEdge> {
  // Vertices in the root's tree, in visit order (the root first).
  const kids: number[][] = Array.from({ length: graph.nodes }, () => [])
  for (const v of order) if (parent[v] >= 0) kids[parent[v]].push(v)
  const members: number[] = []
  const stack = [root]
  while (stack.length) {
    const v = stack.pop()!
    members.push(v)
    for (let i = kids[v].length - 1; i >= 0; i--) stack.push(kids[v][i])
  }
  const idOf = new Map(members.map((v, i) => [v, i]))
  return treeFromChildren<SpanningTreeNode, SpanningTreeEdge>(
    members.map((v) => kids[v].map((c) => idOf.get(c)!)),
    0,
    {
      labels: graph.labels ? members.map((v) => graph.labels![v]) : undefined,
      data: (i) => ({ vertex: members[i] }),
      edge: (c) => {
        const k = parentEdge[members[c]]
        return { edge: k, weight: k >= 0 ? (graph.edges[k].weight ?? 1) : 1 }
      },
    },
  )
}

/**
 * The tree of a graph search rooted at `root`, over graph vertices (node `vertex`, label from `graph.labels`, edge
 * `edge` and `weight`). Node ids are the tree's own (pre-order); the root is node 0. Give the search's `parent`
 * (and `parentEdge`, and its visit `order` so siblings keep the order they were found in), e.g. a
 * `breadthFirstSearch` or `depthFirstSearch` result; or a spanning tree's `edges`, e.g. from `minimumSpanningTree`.
 * Default root: the first root in visit order for a parent array, vertex 0 for an edge set. Throws `DomainError` when
 * the root has a parent, or when a parent array has no root.
 *
 * @param graph The graph searched, for labels and edge weights.
 * @param input The search's parent array, or a spanning tree's edge set.
 * @param root The vertex at the root of the tree.
 * @returns The tree.
 *
 * @example A spanning tree from its edge set
 * const g = fromEdges(4, [[0, 1, 2], [1, 2, 5], [0, 3, 1]], { directed: false })
 * const t = spanningTreeOf(g, { edges: [0, 1, 2] })
 * print('vertices in pre-order:', t.nodes.map((n) => n.vertex))
 * print('edge weights:', t.edges.map((e) => e && e.weight))
 */
export function spanningTreeOf(
  graph: Graph,
  input: SpanningInput,
  root?: number,
): Tree<SpanningTreeNode, SpanningTreeEdge> {
  const start = root ?? ('parent' in input ? firstRoot(input) : 0)
  const { parent, parentEdge, order } = parentsOf(graph, input, [start])
  if (parent[start] >= 0)
    throw new DomainError('spanningTreeOf', `spanningTreeOf: vertex ${start} has a parent, so it is not a root`)
  return treeUnder(graph, start, parent, parentEdge, order)
}

/**
 * The first vertex in visit order (or in id order, without one) whose parent is negative; throws `DomainError` when
 * there is none.
 *
 * @param input The parent array and its optional visit order.
 * @returns The root.
 */
function firstRoot(input: { parent: ArrayLike<number>; order?: ArrayLike<number> }): number {
  const order = input.order ? Array.from(input.order) : Array.from({ length: input.parent.length }, (_, v) => v)
  const r = order.find((v) => input.parent[v] < 0)
  if (r === undefined) throw new DomainError('spanningTreeOf', 'spanningTreeOf: no root')
  return r
}

/**
 * Every tree of a search forest (one per root, in visit order), as `spanningTreeOf`. With a parent array, vertices
 * that are roots in it and appear in `order` start a tree (without an order, every vertex with parent $-1$ does, so an
 * unreached vertex is a tree of one node). With an edge set, trees start at vertex 0, then at each vertex not yet
 * covered.
 *
 * @param graph The graph searched, for labels and edge weights.
 * @param input The search's parent array, or a spanning forest's edge set.
 * @returns The trees.
 *
 * @example Two components give two trees
 * const g = fromEdges(4, [[0, 1], [2, 3]], { directed: false })
 * const forest = spanningForestOf(g, { edges: [0, 1] })
 * print('trees:', forest.length)
 * print('vertices:', forest.map((t) => t.nodes.map((n) => n.vertex)))
 */
export function spanningForestOf(graph: Graph, input: SpanningInput): Tree<SpanningTreeNode, SpanningTreeEdge>[] {
  const all = Array.from({ length: graph.nodes }, (_, v) => v)
  const { parent, parentEdge, order } = parentsOf(graph, input, all)
  return order.filter((v) => parent[v] < 0).map((r) => treeUnder(graph, r, parent, parentEdge, order))
}
