/**
 * Graphs as plain data: a node count, an edge list, a direction flag and optional labels. Every algorithm in
 * `aifn-compute/graph` reads a graph through `adjacency`, whose neighbour order is fixed by the edge list (see there), so
 * traversals are reproducible.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Edge, Graph } from 'aifn-compute/foundation/contracts'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { Edge, Graph } from 'aifn-compute/foundation/contracts'

/** One entry of a node's neighbour list: the neighbour, the index of the edge in `graph.edges`, and its weight. */
export interface Arc {
  to: number
  edge: number
  weight: number
}

/** Options shared by the constructors. */
export interface GraphOptions {
  /** Default true. */
  directed?: boolean
  labels?: readonly string[]
}

/** An edge given as an object, a pair `[from, to]` or a triple `[from, to, weight]`. */
export type EdgeInput = Edge | readonly [number, number] | readonly [number, number, number]

/** The weight of an edge (1 when unset). */
export const weightOf = (e: Edge): number => e.weight ?? 1

/** True unless the graph says `directed: false`. */
export const isDirected = (g: Graph): boolean => g.directed !== false

function check(g: Graph): void {
  if (!Number.isInteger(g.nodes) || g.nodes < 0)
    throw new DomainError('graph', `graph: node count ${g.nodes} is not a count`)
  if (g.labels && g.labels.length !== g.nodes)
    throw new ShapeError('graph', 'graph: labels must have one entry per node')
  g.edges.forEach((e, k) => {
    if (
      !Number.isInteger(e.from) ||
      !Number.isInteger(e.to) ||
      e.from < 0 ||
      e.to < 0 ||
      e.from >= g.nodes ||
      e.to >= g.nodes
    )
      throw new DomainError('graph', `graph: edge ${k} (${e.from} → ${e.to}) is outside 0…${g.nodes - 1}`)
    if (Number.isNaN(weightOf(e))) throw new DomainError('graph', `graph: edge ${k} has a NaN weight`)
  })
}

/**
 * A graph from an edge list. Edges keep their order, which fixes the order of every node's neighbours (see
 * `adjacency`). Throws on an endpoint outside 0 … nodes − 1 or a NaN weight.
 */
export function fromEdges(nodes: number, edges: readonly EdgeInput[], options: GraphOptions = {}): Graph {
  const list: Edge[] = edges.map((e) =>
    Array.isArray(e)
      ? e.length === 3
        ? { from: e[0], to: e[1], weight: e[2] }
        : { from: e[0], to: e[1] }
      : { ...(e as Edge) },
  )
  const g: Graph = {
    kind: 'graph',
    nodes,
    edges: list,
    directed: options.directed ?? true,
    ...(options.labels && { labels: options.labels }),
  }
  check(g)
  return g
}

/**
 * A graph from adjacency lists: `lists[v]` holds the neighbours of v, and `weights[v][k]` (optional) the weight of the
 * edge to `lists[v][k]`. Directed: one edge per entry, in list order, so neighbour order is exactly the list order.
 * Undirected: each unordered pair becomes one edge the first time it is listed (listing it from both ends is allowed),
 * so a node's neighbours come in the order its edges were created.
 */
export function fromAdjacency(
  lists: readonly (readonly number[])[],
  options: GraphOptions & { weights?: readonly (readonly number[])[] } = {},
): Graph {
  const directed = options.directed ?? true
  const edges: Edge[] = []
  const seen = new Set<string>()
  lists.forEach((row, v) =>
    row.forEach((w, k) => {
      const weight = options.weights?.[v]?.[k]
      if (!directed) {
        const key = v < w ? `${v},${w}` : `${w},${v}`
        if (seen.has(key)) return
        seen.add(key)
      }
      edges.push(weight === undefined ? { from: v, to: w } : { from: v, to: w, weight })
    }),
  )
  return fromEdges(lists.length, edges, { ...options, directed })
}

/** Reads a square matrix given as a tensor or rows into rows of numbers. */
function readSquare(m: Tensor | readonly (readonly number[])[], where: string): number[][] {
  let rows: number[][]
  if (isTensor(m)) {
    if (m.shape.length !== 2)
      throw new ShapeError(where, `${where}: expected a matrix, got shape [${m.shape.join(', ')}]`)
    rows = Array.from({ length: m.shape[0] }, (_, i) =>
      Array.from({ length: m.shape[1] }, (_, j) => m.data[m.offset + i * m.strides[0] + j * m.strides[1]]),
    )
  } else rows = m.map((r) => [...r])
  if (rows.some((r) => r.length !== rows.length)) throw new ShapeError(where, `${where}: the matrix must be square`)
  return rows
}

/**
 * A graph from a dense V × V adjacency or weight matrix (a tensor or rows): an edge i → j with weight `m[i][j]` for
 * every entry that is not `absent` (default 0, as `scipy.sparse.csgraph` reads dense input; pass Infinity for a
 * distance matrix). Edges come in row-major order, so neighbours are in ascending order. Undirected: only the upper
 * triangle (i ≤ j) is read.
 */
export function fromMatrix(
  m: Tensor | readonly (readonly number[])[],
  options: GraphOptions & { absent?: number } = {},
): Graph {
  const rows = readSquare(m, 'fromMatrix')
  const absent = options.absent ?? 0
  const directed = options.directed ?? true
  const edges: Edge[] = []
  rows.forEach((row, i) =>
    row.forEach((w, j) => {
      if (!directed && j < i) return
      const missing = w === absent || (Number.isNaN(absent) && Number.isNaN(w))
      if (!missing) edges.push({ from: i, to: j, weight: w })
    }),
  )
  return fromEdges(rows.length, edges, { ...options, directed })
}

/**
 * Every node's out-neighbours (all neighbours when undirected) as arcs. **Neighbour order:** a node's arcs follow the
 * edge list: edge k contributes `from → to` to `from`'s list and, when undirected, `to → from` to `to`'s list, in
 * increasing k. A self-loop in an undirected graph appears once. Every traversal visits neighbours in this order.
 */
export function adjacency(g: Graph): Arc[][] {
  check(g)
  const out: Arc[][] = Array.from({ length: g.nodes }, () => [])
  const directed = isDirected(g)
  g.edges.forEach((e, k) => {
    const weight = weightOf(e)
    out[e.from].push({ to: e.to, edge: k, weight })
    if (!directed && e.from !== e.to) out[e.to].push({ to: e.from, edge: k, weight })
  })
  return out
}

/** The neighbours of `v` (out-neighbours when directed) in adjacency order; int32 vector. */
export function neighbours(g: Graph, v: number): Tensor {
  return ints(adjacency(g)[v].map((a) => a.to))
}

/** Out-degrees (edge ends leaving each node; an undirected self-loop counts once), int32 vector of length V. */
export function outDegree(g: Graph): Tensor {
  return ints(adjacency(g).map((a) => a.length))
}

/** In-degrees (edges entering each node), int32 vector of length V; equals `outDegree` for an undirected graph. */
export function inDegree(g: Graph): Tensor {
  if (!isDirected(g)) return outDegree(g)
  check(g)
  const d = new Int32Array(g.nodes)
  for (const e of g.edges) d[e.to]++
  return fromData(d)
}

/** The graph with every edge reversed (same edge order and weights); an undirected graph is returned as it is. */
export function reverse(g: Graph): Graph {
  if (!isDirected(g)) return g
  return { ...g, edges: g.edges.map((e) => ({ ...e, from: e.to, to: e.from })) }
}

/**
 * The subgraph induced by `nodes` (distinct node indices): node i of the result is `nodes[i]`, edges are those with
 * both ends kept, in their original order. Labels are carried over; without labels, the result is labelled with the
 * original indices so the mapping is not lost.
 */
export function subgraph(g: Graph, nodes: readonly number[] | Tensor): Graph {
  const keep = isTensor(nodes) ? Array.from(nodes.data) : [...nodes]
  const index = new Map<number, number>()
  keep.forEach((v, i) => {
    if (index.has(v)) throw new DomainError('subgraph', `subgraph: node ${v} is listed twice`)
    if (!(v >= 0 && v < g.nodes)) throw new DomainError('subgraph', `subgraph: node ${v} is outside 0…${g.nodes - 1}`)
    index.set(v, i)
  })
  const edges = g.edges
    .filter((e) => index.has(e.from) && index.has(e.to))
    .map((e) => ({ ...e, from: index.get(e.from)!, to: index.get(e.to)! }))
  return {
    kind: 'graph',
    nodes: keep.length,
    edges,
    directed: isDirected(g),
    labels: keep.map((v) => (g.labels ? g.labels[v] : String(v))),
  }
}

/**
 * The path to `target` read from a parent (predecessor) array, as node indices from the root of `target`'s tree to
 * `target`; int32 vector. Parents are −1 at roots. Empty when `target` is unreached (its parent is −1 and it is not a
 * root: pass `root` to say which node is the root, e.g. the source).
 */
export function path(parents: Tensor | readonly number[], target: number, root?: number): Tensor {
  const p = isTensor(parents)
    ? (i: number) => parents.data[parents.offset + i * (parents.strides[0] ?? 1)]
    : (i: number) => parents[i]
  const length = isTensor(parents) ? parents.shape[0] : parents.length
  const out = [target]
  let v = target
  while (p(v) >= 0) {
    v = p(v)
    out.push(v)
    if (out.length > length) throw new DomainError('path', 'path: the parent array has a cycle')
  }
  if (root !== undefined && v !== root) return ints([])
  return ints(out.reverse())
}

/** An int32 vector holding a copy of `a`. */
export function ints(a: ArrayLike<number>): Tensor {
  return fromData(Int32Array.from(a))
}

/** A float64 vector holding a copy of `a`. */
export function floats(a: ArrayLike<number>): Tensor {
  return fromData(Float64Array.from(a))
}

/** A mutable copy of a (contiguous) int32 tensor's data. */
export const intsOf = (t: Tensor): Int32Array => Int32Array.from(t.data)

/** A mutable copy of a (contiguous) tensor's data as float64. */
export const floatsOf = (t: Tensor): Float64Array => Float64Array.from(t.data)
