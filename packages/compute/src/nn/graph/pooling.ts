/**
 * Graph-level readout and hierarchical pooling, on the segment reductions of `aifn-compute/graph/propagation`.
 *
 * - Batching: `batchGraphs` joins several graphs into one block-diagonal graph with a batch vector (the graph of each
 *   node), so one message-passing pass runs over all of them.
 * - Readout: `globalPool` reduces each graph's node rows by a sum, mean or max,
 *   $\rvec_b = \bigoplus_{v : b_v = b} \hvec_v$ (`aggregateEdges` with the batch vector as destinations; the sum is
 *   the injective readout of Xu et al. 2019); `attentionPool` weights each graph's nodes by a softmax of a gate
 *   score over that graph's nodes and sums (the soft-attention readout of Li et al. 2016, "Gated graph sequence
 *   neural networks", ICLR, eq. 7, with a softmax over the graph's nodes in place of a sigmoid gate).
 * - Hierarchical pooling: `diffPool` (Ying et al. 2018, "Hierarchical graph representation learning with
 *   differentiable pooling", NeurIPS, §3.2 to 3.3) coarsens a dense graph by a soft assignment
 *   $\Smat = \operatorname{softmax}(\text{logits})$ of $n$ nodes to $c$ clusters, $\Amat' = \Smat^\top\Amat\Smat$ and
 *   $\Xmat' = \Smat^\top\Zmat$, with the link-prediction and entropy losses; its link loss is
 *   $\lVert\Amat - \Smat\Smat^\top\rVert_F^2 / n^2$ (the paper's norm is not squared), so its scale does not grow with
 *   the graph. `topKPool` (Gao and Ji 2019, "Graph U-Nets", ICML, §3.1) scores nodes by
 *   $y_v = \hvec_v^\top\pvec / \lVert\pvec\rVert$, keeps the $k$ highest, gates their rows by $\tanh(y_v)$ and returns
 *   the induced subgraph.
 *
 * Everything is differentiable in the features and parameters (the choice of nodes in `topKPool` is not).
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { softmax } from 'aifn-compute/numerics/special'
import {
  add,
  div,
  fromData,
  log,
  matmul,
  mean,
  mul,
  neg,
  shapeOfValue,
  sqrt,
  square,
  sub,
  sum,
  take,
  tanh,
  toFlat,
  transpose,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { subgraph, type Edge, type Graph } from 'aifn-compute/graph'
import { aggregateEdges, edgeSoftmax } from 'aifn-compute/graph/propagation'

/** Several graphs as one: the block-diagonal union and the graph each node came from. */
export interface GraphBatch {
  /** The union graph, nodes numbered graph by graph, as directed as the inputs. */
  readonly graph: Graph
  /** The graph index of every node, $V$ entries in $0, \dots, B - 1$ (int32). */
  readonly batch: Int32Array
  /** The number of graphs $B$. */
  readonly graphs: number
  /** The index of each graph's first node in the union. */
  readonly offsets: readonly number[]
}

/**
 * Join graphs into one block-diagonal graph (no edges between graphs) with its batch vector. Edges keep their weights
 * and other fields; labels are kept when every graph has them. Throws `DomainError` when the graphs mix directed and
 * undirected ones.
 *
 * @param gs The graphs, each on its own nodes $0, \dots, n_b - 1$.
 * @returns The union, the batch vector, the number of graphs and the node offsets.
 *
 * @example Two triangles in one batch
 * import { cycleGraph } from 'aifn-compute/graph/structures'
 * const { graph, batch, offsets } = batchGraphs([cycleGraph(3), cycleGraph(3)])
 * print('nodes:', graph.nodes, ' edges:', graph.edges.length, ' offsets:', offsets)
 * print('batch:', batch)
 */
export function batchGraphs(gs: readonly Graph[]): GraphBatch {
  const directed = gs.map((g) => g.directed !== false)
  if (directed.some((d) => d !== directed[0]))
    throw new DomainError('batchGraphs', 'batchGraphs: the graphs mix directed and undirected ones')
  const edges: Edge[] = []
  const batch: number[] = []
  const offsets: number[] = []
  let offset = 0
  gs.forEach((g, b) => {
    offsets.push(offset)
    for (const e of g.edges) edges.push({ ...e, from: e.from + offset, to: e.to + offset })
    for (let v = 0; v < g.nodes; v++) batch.push(b)
    offset += g.nodes
  })
  const labels = gs.every((g) => g.labels) ? gs.flatMap((g) => [...g.labels!]) : undefined
  return {
    graph: { kind: 'graph', nodes: offset, edges, directed: directed[0] ?? false, ...(labels ? { labels } : {}) },
    batch: Int32Array.from(batch),
    graphs: gs.length,
    offsets,
  }
}

/** A permutation-invariant reduction over each graph's nodes. */
export type GlobalPooling = 'sum' | 'mean' | 'max'

/**
 * Reduce the node rows of every graph in a batch to one row, $\rvec_b = \bigoplus_{v : b_v = b} \hvec_v$: a named
 * readout over `aggregateEdges` with the batch vector as destinations. A graph with no nodes gets a zero row.
 * Differentiable in `h`.
 *
 * @param h The node rows, $V \times d$.
 * @param batch The graph of each node, from `batchGraphs`.
 * @param graphs The number of graphs $B$.
 * @param mode The reduction (default `sum`).
 * @returns The $B \times d$ graph rows.
 *
 * @example Sum, mean and max of one graph's four node rows
 * const h = tensor([[6], [6], [10], [7]])
 * const one = Int32Array.from([0, 0, 0, 0])
 * print('sum:', globalPool(h, one, 1), ' mean:', globalPool(h, one, 1, 'mean'), ' max:', globalPool(h, one, 1, 'max'))
 */
export function globalPool(h: Value, batch: Int32Array, graphs: Size, mode: GlobalPooling = 'sum'): Value {
  return aggregateEdges(h, batch, graphs, mode)
}

/**
 * Attention readout: weights $\alpha_v = \exp(s_v) / \sum_{u : b_u = b_v} \exp(s_u)$ from gate scores $s_v$, and
 * $\rvec_b = \sum_{v : b_v = b} \alpha_v \hvec_v$. The weights in each graph sum to 1, so the readout is a weighted
 * mean. Differentiable in `h` and `scores`.
 *
 * @param h The node rows, $V \times d$.
 * @param scores The gate scores, $V \times 1$ (e.g. `matmul(h, w)`).
 * @param batch The graph of each node.
 * @param graphs The number of graphs $B$.
 * @returns The $B \times d$ graph rows `pooled` and the $V \times 1$ `weights`.
 *
 * @example The node with the highest score counts most
 * const h = tensor([[6], [6], [10], [7]])
 * const { pooled, weights } = attentionPool(h, tensor([[0.6], [0.6], [1], [0.7]]), Int32Array.from([0, 0, 0, 0]), 1)
 * print('weights:', weights)
 * print('readout:', pooled)
 */
export function attentionPool(
  h: Value,
  scores: Value,
  batch: Int32Array,
  graphs: Size,
): { pooled: Value; weights: Value } {
  const weights = edgeSoftmax(scores, batch, graphs)
  return { pooled: aggregateEdges(mul(h, weights), batch, graphs, 'sum'), weights }
}

/** The result of one DiffPool step. */
export interface DiffPooled {
  /** The soft assignment $\Smat$, $n \times c$, rows summing to 1. */
  readonly s: Value
  /** The coarsened adjacency $\Smat^\top \Amat \Smat$, $c \times c$. */
  readonly adjacency: Value
  /** The coarsened features $\Smat^\top \Zmat$, $c \times d$. */
  readonly features: Value
  /** The link-prediction loss $\lVert \Amat - \Smat\Smat^\top \rVert_F^2 / n^2$. */
  readonly linkLoss: Value
  /** The mean row entropy $-\frac{1}{n}\sum_i \sum_j S_{ij} \log S_{ij}$, in nats. */
  readonly entropy: Value
}

/**
 * One DiffPool coarsening from assignment logits (the output of a pooling network), a dense adjacency and node
 * embeddings (see the file comment). Differentiable in all three. Throws `ShapeError` unless the adjacency is
 * $n \times n$ and the embeddings and logits have $n$ rows.
 *
 * @param adjacency The dense adjacency $\Amat$, $n \times n$.
 * @param z The node embeddings $\Zmat$ (the embedding network's output), $n \times d$.
 * @param logits The assignment logits, $n \times c$; $\Smat$ is their row softmax.
 * @returns The assignment, the coarsened graph and the two auxiliary losses.
 *
 * @example Three nodes into two clusters
 * const A = tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
 * const d = diffPool(A, tensor([[1], [2], [3]]), tensor([[5, 0], [0, 0], [0, 5]]))
 * print('S^T A S:', d.adjacency)
 * print('link loss:', d.linkLoss, ' entropy:', d.entropy)
 */
export function diffPool(adjacency: Value, z: Value, logits: Value): DiffPooled {
  const [n, m] = shapeOfValue(adjacency)
  if (n !== m || shapeOfValue(z)[0] !== n || shapeOfValue(logits)[0] !== n)
    throw new ShapeError('diffPool', 'diffPool: need an n × n adjacency and n rows of embeddings and logits')
  const s = softmax(logits, { axis: -1 })
  const st = transpose(s)
  const residual = sub(adjacency, matmul(s, st))
  return {
    s,
    adjacency: matmul(matmul(st, adjacency), s),
    features: matmul(st, z),
    linkLoss: div(sum(square(residual)), n * n),
    entropy: neg(mean(sum(mul(s, log(add(s, 1e-12))), -1))),
  }
}

/** The result of `topKPool`. */
export interface TopKPooled {
  /** The kept node indices, highest score first, ties to the smaller index (int32). */
  readonly kept: Tensor
  /** Every node's score $y_v = \hvec_v^\top \pvec / \lVert \pvec \rVert$ (float64). */
  readonly scores: Tensor
  /** The kept rows gated by $\tanh(y_v)$, $k \times d$. */
  readonly features: Value
  /**
   * The subgraph induced by the kept nodes, renumbered $0, \dots, k - 1$ in `kept` order, with the original edge
   * weights and, as labels, the original indices (or labels).
   */
  readonly graph: Graph
}

/**
 * Top-k pooling (gPool; see the file comment): project, keep the $k$ highest-scoring nodes, gate their rows and keep
 * the edges among them. Differentiable in `h` and `p` through the gate (the choice of nodes is not). Throws
 * `DomainError` unless $1 \le k \le V$.
 *
 * @param g The graph.
 * @param h The node rows, $V \times d$.
 * @param p The projection vector $\pvec$, $d \times 1$.
 * @param k The number of nodes to keep.
 * @returns The kept nodes, the scores, the gated rows and the induced subgraph.
 *
 * @example Keep the two highest of four scores on a triangle with a tail
 * const g = { kind: 'graph', nodes: 4, edges: [[0, 1], [0, 2], [1, 2], [2, 3]].map(([from, to]) => ({ from, to })) }
 * const r = topKPool({ ...g, directed: false }, tensor([[0.2], [-0.5], [0.9], [0.4]]), tensor([[1]]), 2)
 * print('kept:', r.kept, ' gated rows:', r.features)
 * print('edges kept:', r.graph.edges.map((e) => [e.from, e.to]))
 */
export function topKPool(g: Graph, h: Value, p: Value, k: Size): TopKPooled {
  if (!(Number.isInteger(k) && k >= 1 && k <= g.nodes))
    throw new DomainError('topKPool', `topKPool: k = ${k} must be an integer in 1…${g.nodes}`)
  const y = div(matmul(h, p), sqrt(sum(square(p))))
  const scores = Float64Array.from(toFlat(unwrap(y) as Tensor))
  const kept = Array.from(scores, (s, v) => [s, v] as const)
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .slice(0, k)
    .map(([, v]) => v)
  const gate = tanh(take(y, kept))
  return {
    kept: fromData(Int32Array.from(kept)),
    scores: fromData(scores),
    features: mul(take(h, kept), gate),
    graph: subgraph(g, kept),
  }
}
