/**
 * Rooted trees as plain data: one type for every tree in aifn (Huffman codes, decision trees, dendrograms,
 * branch-and-bound search trees, breadth- and depth-first parent trees, spanning trees, expression trees), so one
 * renderer can draw them all.
 *
 * ```ts
 * type TreeNode<N> = N & {
 *   id: number              // index into tree.nodes
 *   parent: number | null   // null for the root
 *   children: number[]      // ordered (left to right)
 *   slot?: number           // ordered k-ary trees (tree.arity set): position among the parent's slots; binary 0 left, 1 right
 *   label?: string          // display text ($…$ maths allowed)
 *   height?: number         // dendrograms: the merge distance (leaves 0); layouts may place nodes by it
 * }
 * type TreeEdge<E> = E & { label?: string; weight?: number }   // the edge parent → child (weight: a branch length)
 * interface Tree<N, E> {
 *   nodes: TreeNode<N>[]
 *   root: number
 *   edges: (TreeEdge<E> | null)[]   // keyed by child id; null at the root
 *   arity?: number                  // 2 for binary trees whose left/right order matters (see `slot`)
 * }
 * ```
 *
 * - Constructors: `treeFromParents`, `treeFromChildren`, `treeFromNested` (nested objects with `children`),
 *   `binaryTree` (nested objects with `left`/`right`), `spanningTreeOf` / `spanningForestOf` (a BFS/DFS parent array or
 *   an MST edge set as a tree over graph vertices).
 * - Queries: `depth`, `depths`, `height`, `leaves`, `isLeaf`, `ancestors`, `pathToRoot`, `pathFromRoot`, `lca`,
 *   `subtreeSize`, `leftChild`, `rightChild`.
 * - Traversals (node ids): `preOrder`, `postOrder`, `levelOrder`, `inOrder` (binary trees).
 * - Transforms: `mapTree` (node and edge data), `foldTree` (bottom-up).
 *
 * Trees are serialisable (JSON round-trips them) and treated as immutable: every function returns new data.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Graph } from './graph'
import type { Tree, TreeEdge, TreeNode } from 'aifn-compute/foundation/contracts'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { TreeNode, TreeEdge, Tree } from 'aifn-compute/foundation/contracts'

/** Options shared by the constructors: per-node labels and data, per-edge data (keyed by child). */
export interface TreeOptions<N extends object, E extends object> {
  labels?: readonly (string | undefined)[]
  /** Node data for node i. */
  data?: (i: number) => N
  /** Edge data for the edge parent(c) → c. */
  edge?: (child: number, parent: number) => TreeEdge<E>
}

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
 * A tree from a parent array: `parents[i]` is node i's parent, or −1 / null for the root (exactly one). Children are
 * ordered by id unless `order` (a list of every node id) says otherwise, e.g. a visit order.
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

/** A tree from ordered child lists: `children[i]` lists node i's children, left to right. */
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
  label?: string
  height?: number
  /** The edge from the parent to this node. */
  edge?: TreeEdge<E>
  children?: readonly NestedTree<N, E>[]
}

/** A binary tree described by nesting: `left` and `right` subtrees, either may be absent. */
export type NestedBinaryTree<N extends object = object, E extends object = object> = N & {
  label?: string
  height?: number
  edge?: TreeEdge<E>
  left?: NestedBinaryTree<N, E> | null
  right?: NestedBinaryTree<N, E> | null
}

/** Splits a nested node into its data and structure (the data keeps every field that is not structural). */
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

/** A tree from nested objects, e.g. `{ label: '+', children: [{ label: 'x' }, { label: '1' }] }`. Ids in pre-order. */
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
 * 1 right). Ids in pre-order.
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

/** The left child (slot 0) of a node in a binary tree, or null. */
export function leftChild(tree: Tree<object, object>, id: number): number | null {
  return tree.nodes[id].children.find((c) => (tree.nodes[c].slot ?? tree.nodes[id].children.indexOf(c)) === 0) ?? null
}

/** The right child (slot 1) of a node in a binary tree, or null. */
export function rightChild(tree: Tree<object, object>, id: number): number | null {
  return tree.nodes[id].children.find((c) => (tree.nodes[c].slot ?? tree.nodes[id].children.indexOf(c)) === 1) ?? null
}

// ── Queries ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Whether a node has no children. */
export const isLeaf = (tree: Tree<object, object>, id: number): boolean => tree.nodes[id].children.length === 0

/** Edges from the root to the node (the root has depth 0). */
export function depth(tree: Tree<object, object>, id: number): number {
  let d = 0
  for (let v = tree.nodes[id].parent; v !== null; v = tree.nodes[v].parent) d++
  return d
}

/** The depth of every node, indexed by id. */
export function depths(tree: Tree<object, object>): number[] {
  const out = new Array<number>(tree.nodes.length).fill(0)
  for (const v of preOrder(tree)) {
    const p = tree.nodes[v].parent
    out[v] = p === null ? 0 : out[p] + 1
  }
  return out
}

/** Edges on the longest downward path from the node (default the root) to a leaf: 0 for a leaf. */
export function height(tree: Tree<object, object>, id: number = tree.root): number {
  return foldTree(tree, (_, hs) => (hs.length ? 1 + Math.max(...hs) : 0), id)
}

/** The leaves under a node (default the root), left to right. */
export function leaves(tree: Tree<object, object>, id: number = tree.root): number[] {
  return preOrder(tree, id).filter((v) => tree.nodes[v].children.length === 0)
}

/** The node and its ancestors, from the node up to the root. */
export function pathToRoot(tree: Tree<object, object>, id: number): number[] {
  const path: number[] = []
  for (let v: number | null = id; v !== null; v = tree.nodes[v].parent) path.push(v)
  return path
}

/** The path from the root down to the node (inclusive): the reverse of `pathToRoot`. */
export const pathFromRoot = (tree: Tree<object, object>, id: number): number[] => pathToRoot(tree, id).reverse()

/** The node's proper ancestors, nearest first (its parent, …, the root). */
export const ancestors = (tree: Tree<object, object>, id: number): number[] => pathToRoot(tree, id).slice(1)

/** The lowest common ancestor of two nodes (a node is its own ancestor here). */
export function lca(tree: Tree<object, object>, a: number, b: number): number {
  const up = new Set(pathToRoot(tree, a))
  for (let v: number | null = b; v !== null; v = tree.nodes[v].parent) if (up.has(v)) return v
  throw new DomainError('lca', 'lca: the nodes are not in one tree')
}

/** The number of nodes in the subtree rooted at the node (itself included). */
export function subtreeSize(tree: Tree<object, object>, id: number = tree.root): number {
  return foldTree(tree, (_, sizes) => 1 + sizes.reduce((a, b) => a + b, 0), id)
}

// ── Traversals ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** Node ids in pre-order (a node before its children, children left to right), from `id` (default the root). */
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

/** Node ids in post-order (children left to right, then the node). */
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

/** Node ids level by level (breadth-first), each level left to right. */
export function levelOrder(tree: Tree<object, object>, id: number = tree.root): number[] {
  const out = [id]
  for (let i = 0; i < out.length; i++) out.push(...tree.nodes[out[i]].children)
  return out
}

/**
 * Node ids in in-order (left subtree, node, right subtree) of a binary tree. Sides come from `slot` when set, else
 * from the child's position. Throws on a node with more than two children.
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
 * A tree with the same shape and new node (and optionally edge) data. `node(n)` returns the new data for node n; the
 * structural fields (`id`, `parent`, `children`, `slot`) are kept, and `label` and `height` too unless the new data
 * sets them. `edge(e, child)` maps each edge (default: keep).
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
/** An edge of a spanning tree: the index of the graph edge it uses (−1 if none was given) and its weight. */
export type SpanningTreeEdge = { edge: number; weight: number }

/**
 * How a search's tree is given: a parent per vertex (−1 for roots and unreached vertices), optionally with the graph
 * edge used to reach each vertex (`parentEdge`) and a visit order that orders siblings; or a set of undirected tree
 * edges (indices into `graph.edges`, e.g. a minimum spanning tree), oriented away from the root.
 */
export type SpanningInput =
  | { parent: ArrayLike<number>; parentEdge?: ArrayLike<number>; order?: ArrayLike<number> }
  | { edges: ArrayLike<number> }

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
 * Default root: the first root in visit order.
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

function firstRoot(input: { parent: ArrayLike<number>; order?: ArrayLike<number> }): number {
  const order = input.order ? Array.from(input.order) : Array.from({ length: input.parent.length }, (_, v) => v)
  const r = order.find((v) => input.parent[v] < 0)
  if (r === undefined) throw new DomainError('spanningTreeOf', 'spanningTreeOf: no root')
  return r
}

/**
 * Every tree of a search forest (one per root, in visit order), as `spanningTreeOf`. With a parent array, vertices
 * that are roots in it and appear in `order` start a tree (without an order, every vertex with parent −1 does, so an
 * unreached vertex is a tree of one node). With an edge set,
 * trees start at vertex 0, then at each vertex not yet covered.
 */
export function spanningForestOf(graph: Graph, input: SpanningInput): Tree<SpanningTreeNode, SpanningTreeEdge>[] {
  const all = Array.from({ length: graph.nodes }, (_, v) => v)
  const { parent, parentEdge, order } = parentsOf(graph, input, all)
  return order.filter((v) => parent[v] < 0).map((r) => treeUnder(graph, r, parent, parentEdge, order))
}
