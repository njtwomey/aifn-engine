/**
 * Graph neural network layers on `aifn-compute/graph/propagation`'s message passing, each a functional form (graph, features,
 * parameters) and a `Layer` that closes over one graph (transductive use, as a citation network):
 *
 * - **Graph convolution** (Kipf and Welling 2017, "Semi-supervised classification with graph convolutional networks",
 *   ICLR): H′ = Â H W + b with Â = D̃^{−1/2}(A + I)D̃^{−1/2} (`symmetric`), D̃⁻¹(A + I) (`random-walk`, the mean over
 *   the neighbourhood) or A + I (`none`, the sum).
 * - **Graph attention** (Veličković et al. 2018, "Graph attention networks", ICLR): z = hW per head;
 *   e_{uv} = LeakyReLU(a_srcᵀ z_u + a_dstᵀ z_v); α = softmax of e over v's incoming edges (itself included);
 *   h′_v = Σ_u α_{uv} z_u, heads concatenated or averaged. `v2` is GATv2 (Brody, Alon and Yahav 2022, "How attentive are
 *   graph attention networks?", ICLR), e_{uv} = aᵀ LeakyReLU(z_u + z_v), whose ranking of neighbours can depend on v.
 * - **GraphSAGE** (Hamilton, Ying and Leskovec 2017, "Inductive representation learning on large graphs", NeurIPS):
 *   h′_v = W_self h_v + W_neigh AGG{h_u : u → v} + b, with AGG the mean, the sum, the elementwise max, or Hamilton's
 *   max-pooling max{ReLU(W_pool h_u + b_pool)}; `normalise` scales each output row to unit length. `sampleNeighbours`
 *   keeps a uniform sample of at most S incoming edges per node.
 * - **Message passing** (Gilmer et al. 2017, "Neural message passing for quantum chemistry", ICML):
 *   m_v = AGG_{u → v} M([h_u, h_v, w_{uv}]), h′_v = U([h_v, m_v]) for any message and update layers M and U.
 *
 * Every layer is differentiable in its parameters and in the node features, to any order.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, integers, type Stream } from 'aifn-compute/foundation/random'
import type { Params } from 'aifn-compute/foundation/pytree'
import {
  add,
  concat,
  div,
  fromData,
  matmul,
  maximum,
  mean,
  mul,
  reshape,
  shapeOfValue,
  sqrt,
  square,
  sum,
  take,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { fromEdges, type Graph } from 'aifn-compute/graph'
import { aggregateEdges, edgeSoftmax, messageEdges, type Aggregation } from 'aifn-compute/graph/propagation'
import { leakyRelu, relu } from 'aifn-compute/nn/functional'
import { lecunUniform, xavierUniform, zerosInit } from 'aifn-compute/nn/init'
import { childContext, linear, tap, type Context, type Layer, type LinearParams } from 'aifn-compute/nn/layers'

/** The directed message edges of a graph (both directions of an undirected edge), with optional self-loops. */
type Edges = ReturnType<typeof messageEdges>

function checkFeatures(g: Graph, h: Value, where: string): Size {
  const s = shapeOfValue(h)
  if (s.length !== 2 || s[0] !== g.nodes)
    throw new ShapeError(where, `${where}: features must be ${g.nodes} × F, got [${s.join(', ')}]`)
  return s[1]
}

const column = (values: Float64Array): Tensor => fromData(values, [values.length, 1])

// ── Graph convolution ────────────────────────────────────────────────────────────────────────────────────────────────

/** How a graph convolution weights each neighbour. */
export type GcnNormalisation = 'symmetric' | 'random-walk' | 'none'

/** Options of {@link graphConv}. */
export interface GraphConvOptions {
  /** Default `symmetric`. */
  normalisation?: GcnNormalisation
  /** Add a self-loop of weight 1 to every node (Ã = A + I). Default true. */
  selfLoops?: boolean
  /** Use the graph's edge weights in A (default false: 1 per edge). */
  weighted?: boolean
}

/** The propagation coefficients Â_{vu} of every message edge u → v of a graph convolution, and the edges. */
export function gcnCoefficients(g: Graph, options: GraphConvOptions = {}): { edges: Edges; coefficients: Tensor } {
  const { normalisation = 'symmetric', selfLoops = true, weighted = false } = options
  const edges = messageEdges(g, selfLoops)
  const w = weighted
    ? Float64Array.from(edges.weight.data as Float64Array)
    : new Float64Array(edges.source.length).fill(1)
  const deg = new Float64Array(g.nodes)
  edges.destination.forEach((v, k) => (deg[v] += w[k]))
  const coef = w.map((x, k) => {
    const u = edges.source[k]
    const v = edges.destination[k]
    if (normalisation === 'none') return x
    if (normalisation === 'random-walk') return deg[v] > 0 ? x / deg[v] : 0
    return deg[u] > 0 && deg[v] > 0 ? x / Math.sqrt(deg[u] * deg[v]) : 0
  })
  return { edges, coefficients: column(coef) }
}

/** The graph convolution Â (h W) + b of features h (V × F) with W (F × G); see the module notes. */
export function graphConv(g: Graph, h: Value, weight: Value, bias?: Value, options: GraphConvOptions = {}): Value {
  checkFeatures(g, h, 'graphConv')
  const { edges, coefficients } = gcnCoefficients(g, options)
  const z = matmul(h, weight)
  const out = aggregateEdges(mul(take(z, edges.source), coefficients), edges.destination, g.nodes, 'sum')
  return bias === undefined ? out : add(out, bias)
}

/** A graph convolution layer over a fixed graph: Linear(F → G) parameters, Glorot-initialised as in Kipf and Welling. */
export function GraphConv(
  g: Graph,
  inFeatures: Size,
  outFeatures: Size,
  options: GraphConvOptions & { bias?: boolean } = {},
): Layer<LinearParams> {
  const { bias = true, ...conv } = options
  const fans = { fanIn: inFeatures, fanOut: outFeatures }
  return {
    kind: 'GraphConv',
    label: `GraphConv(${inFeatures} → ${outFeatures}, ${conv.normalisation ?? 'symmetric'})`,
    init: (s) => ({
      weight: xavierUniform()(child(s, 'weight'), [inFeatures, outFeatures], fans),
      ...(bias ? { bias: zerosInit()(child(s, 'bias'), [outFeatures], fans) } : {}),
    }),
    apply: (p, x, ctx) => tap(ctx, graphConv(g, x, p.weight, p.bias, conv)),
  }
}

// ── Graph attention ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Parameters of a graph attention layer with H heads of width G. */
export type GraphAttentionParams = {
  /** The shared map W (F × H·G). */
  weight: Tensor
  /** GAT: the source and destination halves of the attention vector a, one row per head (H × G). */
  attSource?: Tensor
  attTarget?: Tensor
  /** GATv2: the attention vector, one row per head (H × G). */
  att?: Tensor
  /** H·G when heads are concatenated, G when averaged. */
  bias?: Tensor
}

/** Options of {@link graphAttention}. */
export interface GraphAttentionOptions {
  /** Heads H (default 1). */
  heads?: Size
  /** Concatenate the heads (default true) or average them (the output layer of Veličković et al.). */
  concat?: boolean
  /** The LeakyReLU slope (default 0.2). */
  slope?: number
  /** Attend to every node itself as well (default true). */
  selfLoops?: boolean
  /** `gat` (default) or `v2`. */
  variant?: 'gat' | 'v2'
}

/** A graph attention pass: the output and the attention weights on every message edge. */
export interface GraphAttentionResult {
  /** V × H·G (concatenated) or V × G (averaged). */
  readonly output: Value
  /** α per message edge and head, E × H; each destination's column sums to one. */
  readonly attention: Value
  readonly source: Int32Array
  readonly destination: Int32Array
}

/** One graph attention pass over features h (V × F); see the module notes. */
export function graphAttention(
  g: Graph,
  h: Value,
  params: GraphAttentionParams,
  options: GraphAttentionOptions = {},
): GraphAttentionResult {
  checkFeatures(g, h, 'graphAttention')
  const { heads: H = 1, concat: joined = true, slope = 0.2, selfLoops = true, variant = 'gat' } = options
  const width = shapeOfValue(params.weight)[1]
  if (width % H !== 0)
    throw new ShapeError('graphAttention', `graphAttention: W's ${width} columns do not split into ${H} heads`)
  const G = width / H
  const { source, destination } = messageEdges(g, selfLoops)
  const E = source.length
  const z = matmul(h, params.weight)
  const zs = reshape(take(z, source), [E, H, G])
  const zd = reshape(take(z, destination), [E, H, G])
  let scores: Value
  if (variant === 'v2') {
    if (!params.att) throw new DomainError('graphAttention', 'graphAttention: GATv2 needs `att`')
    scores = sum(mul(leakyRelu(add(zs, zd), slope), reshape(params.att, [1, H, G])), 2)
  } else {
    if (!params.attSource || !params.attTarget)
      throw new DomainError('graphAttention', 'graphAttention: GAT needs `attSource` and `attTarget`')
    const es = sum(mul(zs, reshape(params.attSource, [1, H, G])), 2)
    const ed = sum(mul(zd, reshape(params.attTarget, [1, H, G])), 2)
    scores = leakyRelu(add(es, ed), slope)
  }
  const alpha = edgeSoftmax(scores, destination, g.nodes)
  const messages = reshape(mul(zs, reshape(alpha, [E, H, 1])), [E, H * G])
  let out = aggregateEdges(messages, destination, g.nodes, 'sum')
  if (!joined) out = mean(reshape(out, [g.nodes, H, G]), 1)
  if (params.bias) out = add(out, params.bias)
  return { output: out, attention: alpha, source, destination }
}

/** A graph attention layer over a fixed graph; `forward` also returns the attention weights. */
export interface GraphAttentionLayer extends Layer<GraphAttentionParams> {
  readonly options: GraphAttentionOptions
  forward(params: GraphAttentionParams, x: Value, ctx?: Context): GraphAttentionResult
}

/** A graph attention layer (F → H heads of width G), Glorot-initialised as in Veličković et al. */
export function GraphAttention(
  g: Graph,
  inFeatures: Size,
  outPerHead: Size,
  options: GraphAttentionOptions & { bias?: boolean } = {},
): GraphAttentionLayer {
  const { bias = true, ...att } = options
  const H = att.heads ?? 1
  const outWidth = (att.concat ?? true) ? H * outPerHead : outPerHead
  const glorot = xavierUniform()
  const forward = (p: GraphAttentionParams, x: Value, ctx?: Context) => {
    const r = graphAttention(g, x, p, att)
    return { ...r, output: tap(ctx, r.output) }
  }
  return {
    kind: 'GraphAttention',
    label: `GraphAttention(${inFeatures} → ${H} × ${outPerHead}${att.variant === 'v2' ? ', v2' : ''})`,
    options: att,
    init: (s) => ({
      weight: glorot(child(s, 'weight'), [inFeatures, H * outPerHead], { fanIn: inFeatures, fanOut: H * outPerHead }),
      ...(att.variant === 'v2'
        ? { att: glorot(child(s, 'att'), [H, outPerHead], { fanIn: outPerHead, fanOut: 1 }) }
        : {
            attSource: glorot(child(s, 'attSource'), [H, outPerHead], { fanIn: outPerHead, fanOut: 1 }),
            attTarget: glorot(child(s, 'attTarget'), [H, outPerHead], { fanIn: outPerHead, fanOut: 1 }),
          }),
      ...(bias ? { bias: zerosInit()(child(s, 'bias'), [outWidth], { fanIn: outWidth, fanOut: outWidth }) } : {}),
    }),
    apply: (p, x, ctx) => forward(p, x, ctx).output,
    forward,
  }
}

// ── GraphSAGE ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** How GraphSAGE aggregates the neighbours: `pool` is Hamilton's max-pooling of ReLU(W_pool h_u + b_pool). */
export type SageAggregator = Aggregation | 'pool'

/** Parameters of a GraphSAGE layer. */
export type SageParams = { self: LinearParams; neighbour: LinearParams; pool?: LinearParams }

/** Options of {@link sageConv}. */
export interface SageOptions {
  /** Default `mean`. */
  aggregate?: SageAggregator
  /** Scale each output row to unit Euclidean length (Hamilton et al., Algorithm 1, line 7). Default false. */
  normalise?: boolean
}

/** One GraphSAGE pass: W_self h_v + W_neigh AGG{h_u : u → v} + b, without self-loops; see the module notes. */
export function sageConv(g: Graph, h: Value, params: SageParams, options: SageOptions = {}): Value {
  checkFeatures(g, h, 'sageConv')
  const { aggregate = 'mean', normalise = false } = options
  const { source, destination } = messageEdges(g, false)
  let neigh = take(h, source)
  if (aggregate === 'pool') {
    if (!params.pool) throw new DomainError('sageConv', 'sageConv: the pool aggregator needs `pool` parameters')
    neigh = relu(linear(neigh, params.pool.weight, params.pool.bias))
  }
  const agg = aggregateEdges(neigh, destination, g.nodes, aggregate === 'pool' ? 'max' : aggregate)
  let out = add(
    linear(h, params.self.weight, params.self.bias),
    linear(agg, params.neighbour.weight, params.neighbour.bias),
  )
  if (normalise) out = div(out, maximum(sqrt(sum(square(out), 1, true)), 1e-12))
  return out
}

/** A GraphSAGE layer (F → G) over a fixed graph; the `pool` aggregator adds a Linear(F → F) pooling map. */
export function SageConv(g: Graph, inFeatures: Size, outFeatures: Size, options: SageOptions = {}): Layer<SageParams> {
  const init = lecunUniform()
  const lin = (s: Stream, i: Size, o: Size, bias: boolean): LinearParams => ({
    weight: init(child(s, 'weight'), [i, o], { fanIn: i, fanOut: o }),
    ...(bias ? { bias: zerosInit()(child(s, 'bias'), [o], { fanIn: i, fanOut: o }) } : {}),
  })
  return {
    kind: 'SageConv',
    label: `SageConv(${inFeatures} → ${outFeatures}, ${options.aggregate ?? 'mean'})`,
    init: (s) => ({
      self: lin(child(s, 'self'), inFeatures, outFeatures, true),
      neighbour: lin(child(s, 'neighbour'), inFeatures, outFeatures, false),
      ...(options.aggregate === 'pool' ? { pool: lin(child(s, 'pool'), inFeatures, inFeatures, true) } : {}),
    }),
    apply: (p, x, ctx) => tap(ctx, sageConv(g, x, p, options)),
  }
}

/**
 * A neighbourhood sample (Hamilton et al. 2017, §3.1): the graph with at most `size` incoming message edges per node,
 * drawn uniformly without replacement from `child(stream, 'node', v)`; directed, so each node aggregates exactly its
 * sample.
 */
export function sampleNeighbours(g: Graph, size: Size, stream: Stream): Graph {
  if (!(Number.isInteger(size) && size >= 1))
    throw new DomainError('sampleNeighbours', 'sampleNeighbours: the sample size must be a positive integer')
  const { source, destination } = messageEdges(g, false)
  const incoming: number[][] = Array.from({ length: g.nodes }, () => [])
  destination.forEach((v, k) => incoming[v].push(source[k]))
  const edges: [number, number][] = []
  incoming.forEach((list, v) => {
    const pool = [...list]
    const s = child(stream, 'node', v)
    const m = Math.min(size, pool.length)
    for (let r = 0; r < m; r++) {
      const t = r + integers(s, pool.length - r)
      ;[pool[r], pool[t]] = [pool[t], pool[r]]
      edges.push([pool[r], v])
    }
  })
  return fromEdges(g.nodes, edges, { directed: true, ...(g.labels ? { labels: g.labels } : {}) })
}

// ── Message passing ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of {@link messagePassing}. */
export interface MessagePassingOptions {
  /** Default `sum` (Gilmer et al.). */
  aggregate?: Aggregation
  /** Pass each node a message from itself too. Default false. */
  selfLoops?: boolean
}

/**
 * One generic message-passing step: m_v = AGG_{u → v} message([h_u, h_v, w_{uv}]) and h′_v = update([h_v, m_v]), where
 * `message` maps rows of width 2F + 1 and `update` rows of width F + M (any differentiable functions, such as layers'
 * `apply` with their parameters).
 */
export function messagePassing(
  g: Graph,
  h: Value,
  message: (x: Value) => Value,
  update: (x: Value) => Value,
  options: MessagePassingOptions = {},
): Value {
  checkFeatures(g, h, 'messagePassing')
  const { source, destination, weight } = messageEdges(g, options.selfLoops)
  const m = message(concat([take(h, source), take(h, destination), weight], 1))
  const agg = aggregateEdges(m, destination, g.nodes, options.aggregate ?? 'sum')
  return update(concat([h, agg], 1))
}

/** Parameters of a message-passing layer: those of its message and update layers. */
export type MessagePassingParams = { message: Params; update: Params }

/**
 * A message-passing layer over a fixed graph from a message layer (2F + 1 → M) and an update layer (F + M → G), e.g.
 * small `Mlp`s.
 */
export function MessagePassing(
  g: Graph,
  message: Layer,
  update: Layer,
  options: MessagePassingOptions = {},
): Layer<MessagePassingParams> {
  return {
    kind: 'MessagePassing',
    label: `MessagePassing(${message.label}, ${update.label}, ${options.aggregate ?? 'sum'})`,
    init: (s) => ({ message: message.init(child(s, 'message')), update: update.init(child(s, 'update')) }),
    apply: (p, x, ctx?: Context) =>
      tap(
        ctx,
        messagePassing(
          g,
          x,
          (y) => message.apply(p.message, y, childContext(ctx, 'message')),
          (y) => update.apply(p.update, y, childContext(ctx, 'update')),
          options,
        ),
      ),
  }
}
