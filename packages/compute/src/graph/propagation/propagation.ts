/**
 * Message passing over a graph's edges as a differentiable primitive composition: gather each edge's source features,
 * apply an edge function, and aggregate the messages at each destination by sum, mean or max (Gilmer et al. 2017,
 * "Neural message passing for quantum chemistry", ICML; Kipf & Welling 2017, "Semi-supervised classification with
 * graph convolutional networks", ICLR). Built only from `take`, `gather`, `scatterAdd` and elementwise primitives, so
 * gradients reach the node features and whatever the edge function closes over, to any order.
 *
 * Features are matrices with one row per node ($V \times F$) and messages one row per directed edge ($E \times G$).
 * An undirected edge carries a message each way, and each edge's weight (1 when unset) is passed to the edge function.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { AifnError } from 'aifn-compute/foundation/errors'
import {
  div,
  exp,
  fromData,
  gather,
  unwrap,
  mul,
  scatterAdd,
  shapeOfValue,
  sub,
  take,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { isDirected, weightOf, type Graph } from '../graph'

/**
 * How messages arriving at a node are combined: `'sum'`, `'mean'` over the incoming edges, or `'max'` of each
 * feature.
 */
export type Aggregation = 'sum' | 'mean' | 'max'

/**
 * An edge function: the messages ($E \times G$) from the source features (`source`, $E \times F$), the destination
 * features (`destination`, $E \times F$) and the edge weights (`weight`, a float64 column, $E \times 1$), one row per
 * directed edge. It must return a matrix with one row per edge; $G$ may differ from $F$.
 */
export type EdgeFunction = (source: Value, destination: Value, weight: Tensor) => Value

/** Options of {@link propagate}. */
export interface PropagateOptions {
  /** How the messages at each node are combined (default `'sum'`). */
  aggregate?: Aggregation
  /** The edge function; default the weighted source features, $w_{uv} \hvec_u$. */
  message?: EdgeFunction
  /** Also pass a message from every node to itself (weight 1), as in a graph convolution. Default false. */
  selfLoops?: boolean
}

/**
 * The directed edges message passing uses: each edge in its own direction, then (in an undirected graph) the reverse,
 * except for a self-loop, which is used once; with `selfLoops`, an edge from every node to itself after them.
 *
 * @param g The graph. Edges keep their order, and an undirected edge's reverse follows it directly.
 * @param selfLoops Whether to append an edge $v \to v$ of weight 1 for every node, as a graph convolution does. It is
 *   added even when the graph already has a self-loop at $v$.
 * @returns The `source` and `destination` node of each directed edge ($E$ entries each) and its `weight`, a float64
 *   $E \times 1$ column (1 where the edge has none).
 *
 * @example An undirected path, with self-loops
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1, weight: 2 }, { from: 1, to: 2 }], directed: false }
 * const { source, destination, weight } = messageEdges(g, true)
 * print('source:', source)
 * print('destination:', destination)
 * print('weight:', weight)
 */
export function messageEdges(
  g: Graph,
  selfLoops = false,
): { source: Int32Array; destination: Int32Array; weight: Tensor } {
  const src: number[] = []
  const dst: number[] = []
  const w: number[] = []
  for (const e of g.edges) {
    src.push(e.from)
    dst.push(e.to)
    w.push(weightOf(e))
    if (!isDirected(g) && e.from !== e.to) {
      src.push(e.to)
      dst.push(e.from)
      w.push(weightOf(e))
    }
  }
  if (selfLoops)
    for (let v = 0; v < g.nodes; v++) {
      src.push(v)
      dst.push(v)
      w.push(1)
    }
  return {
    source: Int32Array.from(src),
    destination: Int32Array.from(dst),
    weight: fromData(Float64Array.from(w), [w.length, 1]),
  }
}

/**
 * The concrete tensor under a value, traced or not, for reading its entries; throws `AifnError` when it is a scalar.
 *
 * @param x The messages or scores, a matrix.
 * @returns The tensor of its values.
 */
const raw = (x: Value): Tensor => {
  const v = unwrap(x)
  if (typeof v === 'number') throw new AifnError('propagate', 'propagate: messages must be a matrix')
  return v
}

/**
 * Flat indices of rows `rows` of a row-major matrix with `width` columns: the entries of each listed row in turn.
 *
 * @param rows The row numbers, in the order wanted; a row may repeat.
 * @param width The number of columns of the matrix.
 * @returns `rows.length * width` indices: entry $k \cdot w + j$ is `rows[k] * width + j` ($w$ = `width`).
 */
function rowIndices(rows: Int32Array, width: Size): Int32Array {
  const out = new Int32Array(rows.length * width)
  for (let k = 0; k < rows.length; k++) for (let j = 0; j < width; j++) out[k * width + j] = rows[k] * width + j
  return out
}

/**
 * Combine per-edge rows ($E \times G$) at their destination nodes,
 * $\ovec_v = \bigoplus_{k : d_k = v} \mvec_k$ ($d_k$ the destination of edge $k$), by `sum`, `mean` (over the
 * incoming edges) or `max` (each feature's largest; its gradient goes to that row only, ties to the first edge). A
 * node with no incoming edge gets zeros. The segment reduction behind {@link propagate}, attention and pooling.
 * Differentiable in `messages`. Throws `AifnError` unless `messages` is a matrix with one row per entry of
 * `destination`.
 *
 * @param messages The per-edge rows $\mvec_k$, an $E \times G$ matrix (traced or not).
 * @param destination The destination node of each row, $E$ entries in $0, \dots, V - 1$.
 * @param nodes The number of nodes $V$, the number of rows of the result.
 * @param aggregate How the rows arriving at a node are combined (default `'sum'`).
 * @returns The $V \times G$ matrix of combined rows.
 *
 * @example Sum, mean and max at each destination
 * // Rows 0 and 1 arrive at node 0, row 2 at node 1; nothing arrives at node 2.
 * const m = tensor([[1, 10], [5, 0], [2, 3]])
 * const destination = Int32Array.from([0, 0, 1])
 * print('sum:', aggregateEdges(m, destination, 3))
 * print('mean:', aggregateEdges(m, destination, 3, 'mean'))
 * print('max:', aggregateEdges(m, destination, 3, 'max'))
 *
 * @example The gradient of max reaches only the winning row
 * const destination = Int32Array.from([0, 0, 1])
 * const g = grad((m) => sum(aggregateEdges(m, destination, 2, 'max')))
 * print('d/dm:', g(tensor([[1], [5], [2]])))
 */
export function aggregateEdges(
  messages: Value,
  destination: Int32Array,
  nodes: Size,
  aggregate: Aggregation = 'sum',
): Value {
  const mShape = shapeOfValue(messages)
  if (mShape.length !== 2 || mShape[0] !== destination.length)
    throw new AifnError('aggregateEdges', 'aggregateEdges: messages must have one row per edge')
  const V = nodes
  const G = mShape[1]
  const m = messages
  if (aggregate !== 'max') {
    const total = scatterAdd(m, rowIndices(destination, G), [V, G])
    if (aggregate === 'sum') return total
    const count = new Float64Array(V)
    for (const v of destination) count[v]++
    return div(
      total,
      fromData(
        count.map((c) => Math.max(c, 1)),
        [V, 1],
      ),
    )
  }
  // Max: pick each (node, feature)'s largest message on the raw values, then gather it so the gradient follows it.
  const values = toFlat(raw(m))
  const best = new Int32Array(V * G).fill(-1)
  for (let k = 0; k < destination.length; k++)
    for (let j = 0; j < G; j++) {
      const at = destination[k] * G + j
      if (best[at] < 0 || values[k * G + j] > values[best[at]]) best[at] = k * G + j
    }
  const empty = best.map((b) => (b < 0 ? 1 : 0))
  if (!empty.some((x) => x === 1)) return gather(m, best, [V, G])
  const picked = gather(
    m,
    best.map((b) => Math.max(b, 0)),
    [V, G],
  )
  return mul(
    picked,
    fromData(
      Float64Array.from(empty, (x) => 1 - x),
      [V, G],
    ),
  )
}

/**
 * The softmax of edge scores ($E \times H$, one column per head) over each destination's incoming edges:
 * $\alpha_k = \exp(e_k) / \sum_{k' : d_{k'} = d_k} \exp(e_{k'})$, $d_k$ the destination of edge $k$. The
 * per-destination maximum is subtracted first (a constant, so gradients are unchanged). Differentiable in the scores.
 * Throws `AifnError` unless `scores` is a matrix with one row per entry of `destination`.
 *
 * @param scores The edge scores $e_k$, an $E \times H$ matrix with a column per attention head (traced or not).
 * @param destination The destination node of each edge, $E$ entries in $0, \dots, V - 1$.
 * @param nodes The number of nodes $V$.
 * @returns The weights $\alpha_k$, $E \times H$: in each column, the weights of the edges into one node sum to 1.
 *
 * @example Attention weights over the edges into each node
 * // Edges 0 and 1 enter node 0 with scores 0 and log 3; edge 2 is the only edge into node 1.
 * const scores = tensor([[0], [Math.log(3)], [5]])
 * print('alpha:', edgeSoftmax(scores, Int32Array.from([0, 0, 1]), 2))
 */
export function edgeSoftmax(scores: Value, destination: Int32Array, nodes: Size): Value {
  const shape = shapeOfValue(scores)
  if (shape.length !== 2 || shape[0] !== destination.length)
    throw new AifnError('edgeSoftmax', 'edgeSoftmax: scores must have one row per edge')
  const H = shape[1]
  const e = toFlat(raw(scores))
  const top = new Float64Array(nodes * H).fill(-Infinity)
  for (let k = 0; k < destination.length; k++)
    for (let h = 0; h < H; h++) top[destination[k] * H + h] = Math.max(top[destination[k] * H + h], e[k * H + h])
  const shift = new Float64Array(destination.length * H)
  for (let k = 0; k < destination.length; k++)
    for (let h = 0; h < H; h++) shift[k * H + h] = top[destination[k] * H + h]
  const ex = exp(sub(scores, fromData(shift, [destination.length, H])))
  const total = scatterAdd(ex, rowIndices(destination, H), [nodes, H])
  return div(ex, take(total, destination))
}

/**
 * One round of message passing, from node features $\Hmat$ ($V \times F$) to aggregated messages ($V \times G$):
 * $\mvec_{uv} = \mathrm{message}(\hvec_u, \hvec_v, w_{uv})$ on every directed edge $u \to v$ (see
 * {@link messageEdges}) and $\ovec_v = \bigoplus_{u \to v} \mvec_{uv}$ ({@link aggregateEdges}).
 * Differentiable in $\Hmat$ and in everything `message` closes over. Throws `AifnError` when `h` is not
 * $V \times F$ or the edge function does not return one row per edge.
 *
 * @param g The graph, with $V$ nodes; an undirected edge passes a message each way.
 * @param h The node features $\Hmat$, a $V \times F$ matrix with row $v$ the features of node $v$ (traced or not).
 * @param options The aggregation (default `'sum'`), the edge function (default the weighted source features) and
 *   whether every node also sends itself a message.
 * @returns The $V \times G$ matrix of aggregated messages; a node that receives none gets a zero row.
 *
 * @example Sum, mean and max over a path's neighbours
 * // The path 0 - 1 - 2 with features 1, 2, 3: node 1 hears from 0 and 2, the ends from node 1 only.
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const h = tensor([[1], [2], [3]])
 * print('sum:', propagate(g, h))
 * print('mean:', propagate(g, h, { aggregate: 'mean' }))
 * print('max:', propagate(g, h, { aggregate: 'max' }))
 *
 * @example A graph convolution step: self-loops, mean, and its gradient
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const layer = (h) => propagate(g, h, { aggregate: 'mean', selfLoops: true })
 * print('out:', layer(tensor([[1], [2], [3]])))
 * print('d sum(out) / dh:', grad((h) => sum(layer(h)))(tensor([[1], [2], [3]])))
 *
 * @example An edge function: differences to the neighbour
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const message = (source, destination) => sub(source, destination)
 * print('sum of h_u - h_v:', propagate(g, tensor([[1], [2], [4]]), { message }))
 */
export function propagate(g: Graph, h: Value, options: PropagateOptions = {}): Value {
  const shape = shapeOfValue(h)
  if (shape.length !== 2 || shape[0] !== g.nodes)
    throw new AifnError('propagate', `propagate: features must be ${g.nodes} × F, got [${shape.join(', ')}]`)
  const { source, destination, weight } = messageEdges(g, options.selfLoops)
  const hs = take(h, source)
  const m = options.message ? options.message(hs, take(h, destination), weight) : mul(hs, weight)
  const mShape = shapeOfValue(m)
  if (mShape.length !== 2 || mShape[0] !== source.length)
    throw new AifnError('propagate', 'propagate: the edge function must return one message row per edge')
  return aggregateEdges(m, destination, g.nodes, options.aggregate ?? 'sum')
}
