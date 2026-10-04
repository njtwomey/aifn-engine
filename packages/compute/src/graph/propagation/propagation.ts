/**
 * Message passing over a graph's edges as a differentiable primitive composition: gather each edge's source features,
 * apply an edge function, and aggregate the messages at each destination by sum, mean or max (Gilmer et al. 2017,
 * "Neural message passing for quantum chemistry", ICML; Kipf & Welling 2017, "Semi-supervised classification with
 * graph convolutional networks", ICLR). Built only from `take`, `gather`, `scatterAdd` and elementwise primitives, so
 * gradients reach the node features and whatever the edge function closes over, to any order.
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

/** How messages arriving at a node are combined. */
export type Aggregation = 'sum' | 'mean' | 'max'

/**
 * An edge function: the messages (E × G) from the source features (E × F), the destination features (E × F) and the
 * edge weights (a float64 column, E × 1), one row per directed edge.
 */
export type EdgeFunction = (source: Value, destination: Value, weight: Tensor) => Value

/** Options of {@link propagate}. */
export interface PropagateOptions {
  /** Default `sum`. */
  aggregate?: Aggregation
  /** The edge function; default the weighted source features, w_{uv} h_u. */
  message?: EdgeFunction
  /** Also pass a message from every node to itself (weight 1), as in a graph convolution. Default false. */
  selfLoops?: boolean
}

/** The directed edges message passing uses: both directions of an undirected edge (a self-loop once). */
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

const raw = (x: Value): Tensor => {
  const v = unwrap(x)
  if (typeof v === 'number') throw new AifnError('propagate', 'propagate: messages must be a matrix')
  return v
}

/** Flat indices of rows `rows` of a (· × width) matrix, row-major. */
function rowIndices(rows: Int32Array, width: Size): Int32Array {
  const out = new Int32Array(rows.length * width)
  for (let k = 0; k < rows.length; k++) for (let j = 0; j < width; j++) out[k * width + j] = rows[k] * width + j
  return out
}

/**
 * Combine per-edge rows (E × G) at their destination nodes: out_v = ⊕_{k: destination_k = v} m_k, by `sum`, `mean` (over
 * the incoming edges) or `max` (each feature's largest; its gradient goes to that row only, ties to the first edge). A
 * node with no incoming edge gets zeros. The segment reduction behind {@link propagate}, attention and pooling.
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
 * The softmax of edge scores (E × H, one column per head) over each destination's incoming edges:
 * α_k = exp(e_k) / Σ_{k′ → same destination} exp(e_k′). The per-destination maximum is subtracted first (a constant, so
 * gradients are unchanged). Differentiable in the scores.
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
 * One round of message passing: node features h (V × F) → aggregated messages (V × G), with
 * m_{uv} = message(h_u, h_v, w_{uv}) on every directed edge u → v and out_v = ⊕_{u → v} m_{uv} ({@link aggregateEdges}).
 * Differentiable in h and in everything `message` closes over.
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
