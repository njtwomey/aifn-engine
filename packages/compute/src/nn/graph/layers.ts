/**
 * Graph neural network layers on the message passing of `aifn-compute/graph/propagation`, each a functional form
 * (graph, features, parameters) and a `Layer` that closes over one graph (transductive use, as on a citation network).
 * Node features are a $V \times F$ matrix, one row per node; the messages run along the directed message edges
 * $u \to v$ (both directions of an undirected edge).
 *
 * - **Graph convolution** (Kipf and Welling 2017, "Semi-supervised classification with graph convolutional networks",
 *   ICLR): $\Hmat' = \hat{\Amat} \Hmat \Wmat + \bvec$ with
 *   $\hat{\Amat} = \tilde{\Dmat}^{-1/2}(\Amat + \Imat)\tilde{\Dmat}^{-1/2}$ (`symmetric`),
 *   $\tilde{\Dmat}^{-1}(\Amat + \Imat)$ (`random-walk`, the mean over the neighbourhood) or $\Amat + \Imat$
 *   (`none`, the sum), $\tilde{\Dmat}$ the degrees of $\Amat + \Imat$.
 * - **Graph attention** (Veličković et al. 2018, "Graph attention networks", ICLR): $\zvec = \hvec\Wmat$ per head;
 *   $e_{uv} = \mathrm{LeakyReLU}(\avec_{\mathrm{src}}^\top \zvec_u + \avec_{\mathrm{dst}}^\top \zvec_v)$;
 *   $\alpha_{uv}$ the softmax of $e_{uv}$ over $v$'s incoming edges (itself included);
 *   $\hvec'_v = \sum_u \alpha_{uv} \zvec_u$, heads concatenated or averaged. `v2` is GATv2 (Brody, Alon and Yahav
 *   2022, "How attentive are graph attention networks?", ICLR),
 *   $e_{uv} = \avec^\top \mathrm{LeakyReLU}(\zvec_u + \zvec_v)$, whose ranking of neighbours can depend on $v$.
 * - **GraphSAGE** (Hamilton, Ying and Leskovec 2017, "Inductive representation learning on large graphs", NeurIPS):
 *   $\hvec'_v = \Wmat_{\mathrm{self}} \hvec_v + \Wmat_{\mathrm{neigh}} \mathrm{AGG}\{\hvec_u : u \to v\} + \bvec$,
 *   with AGG the mean, the sum, the elementwise max, or Hamilton's max-pooling
 *   $\max\{\mathrm{ReLU}(\Wmat_{\mathrm{pool}} \hvec_u + \bvec_{\mathrm{pool}})\}$; `normalise` scales each output
 *   row to unit length. `sampleNeighbours` keeps a uniform sample of at most $S$ incoming edges per node.
 * - **Message passing** (Gilmer et al. 2017, "Neural message passing for quantum chemistry", ICML):
 *   $\mvec_v = \mathrm{AGG}_{u \to v} M([\hvec_u, \hvec_v, w_{uv}])$, $\hvec'_v = U([\hvec_v, \mvec_v])$ for any
 *   message and update layers $M$ and $U$.
 *
 * Every layer is differentiable in its parameters and in the node features, to any order. Features that are not a
 * $V \times F$ matrix throw `ShapeError`.
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

/**
 * Check that the features have one row per node, and return their width. Throws `ShapeError` otherwise.
 *
 * @param g The graph the features belong to.
 * @param h The node features, expected to be $V \times F$ for the $V$ nodes of `g`.
 * @param where The caller's name, for the error message.
 * @returns The feature width $F$.
 */
function checkFeatures(g: Graph, h: Value, where: string): Size {
  const s = shapeOfValue(h)
  if (s.length !== 2 || s[0] !== g.nodes)
    throw new ShapeError(where, `${where}: features must be ${g.nodes} × F, got [${s.join(', ')}]`)
  return s[1]
}

/**
 * Per-edge values as an $E \times 1$ column, to scale the $E$ rows of messages.
 *
 * @param values One value per edge.
 * @returns The values as an $E \times 1$ tensor.
 */
const column = (values: Float64Array): Tensor => fromData(values, [values.length, 1])

// ── Graph convolution ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * How a graph convolution weights each neighbour: `symmetric` $1/\sqrt{\tilde{d}_u \tilde{d}_v}$, `random-walk`
 * $1/\tilde{d}_v$ (the mean) or `none` (the sum), with $\tilde{d}$ the degrees counting self-loops.
 */
export type GcnNormalisation = 'symmetric' | 'random-walk' | 'none'

/** Options of {@link graphConv}. */
export interface GraphConvOptions {
  /** How each neighbour is weighted (default `symmetric`). */
  normalisation?: GcnNormalisation
  /** Add a self-loop of weight 1 to every node ($\tilde{\Amat} = \Amat + \Imat$). Default true. */
  selfLoops?: boolean
  /**
   * Use the graph's edge weights in $\Amat$ (default false: 1 per edge); the self-loops keep weight 1, and the degrees
   * are then weighted degrees.
   */
  weighted?: boolean
}

/**
 * The propagation coefficients $\hat{A}_{vu}$ of every message edge $u \to v$ of a graph convolution, and the edges.
 * A node of degree 0 (no self-loop, no neighbours) gets coefficient 0 instead of a division by zero.
 *
 * @param g The graph; an undirected edge carries messages both ways, a directed one from `from` to `to`.
 * @param options The normalisation, self-loops and edge weights.
 * @returns The message `edges` (`source`, `destination` and `weight` per edge, the self-loops last) and the
 *   `coefficients` $\hat{A}_{vu}$, an $E \times 1$ column in the same order.
 *
 * @example The symmetric coefficients of a path of three nodes
 * // 0 - 1 - 2, with self-loops: degrees 2, 3, 2.
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const { edges, coefficients } = gcnCoefficients(g)
 * print('source:', edges.source)
 * print('destination:', edges.destination)
 * print('coefficients:', coefficients)
 */
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

/**
 * The graph convolution $\hat{\Amat} (\Hmat \Wmat) + \bvec$ (Kipf and Welling, 2017): each node's new features
 * are the normalised sum over its neighbourhood of the transformed features. Differentiable in `h`, `weight` and
 * `bias`. Throws `ShapeError` unless `h` has one row per node.
 *
 * @param g The graph.
 * @param h The node features $\Hmat$, $V \times F$.
 * @param weight The weight $\Wmat$, $F \times G$.
 * @param bias The bias $\bvec$, $G$ values added to every row (default none).
 * @param options The normalisation, self-loops and edge weights, as for `gcnCoefficients`.
 * @returns The new features, $V \times G$.
 *
 * @example A graph convolution on a path of three nodes
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const h = tensor([[1, 0], [0, 1], [1, 1]])
 * const W = tensor([[1], [1]]) // sums the two features
 * print('sum:', graphConv(g, h, W, undefined, { normalisation: 'none' }))
 * print('mean:', graphConv(g, h, W, undefined, { normalisation: 'random-walk' }))
 * print('symmetric:', graphConv(g, h, W))
 */
export function graphConv(g: Graph, h: Value, weight: Value, bias?: Value, options: GraphConvOptions = {}): Value {
  checkFeatures(g, h, 'graphConv')
  const { edges, coefficients } = gcnCoefficients(g, options)
  const z = matmul(h, weight)
  const out = aggregateEdges(mul(take(z, edges.source), coefficients), edges.destination, g.nodes, 'sum')
  return bias === undefined ? out : add(out, bias)
}

/**
 * A graph convolution layer over a fixed graph: `Linear`-style parameters ($F \to G$), the weight Glorot-uniform
 * initialised as in Kipf and Welling and the bias zero; `apply` is `graphConv` on `g`.
 *
 * @param g The graph every call runs on.
 * @param inFeatures The input width $F$.
 * @param outFeatures The output width $G$.
 * @param options The `graphConv` options, and `bias` (default true) to include the bias.
 * @returns The layer: `init` draws `weight` ($F \times G$) and `bias` ($G$), `apply` maps $V \times F$ features to
 *   $V \times G$.
 *
 * @example A layer of width four on a 3-node graph
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const layer = GraphConv(g, 2, 4)
 * const p = layer.init(stream(0))
 * print(layer.label)
 * print('output:', layer.apply(p, tensor([[1, 0], [0, 1], [1, 1]])))
 */
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

/** Parameters of a graph attention layer with $H$ heads of width $G$. */
export type GraphAttentionParams = {
  /** The shared map $\Wmat$, $F \times HG$: head $k$ is columns $kG$ to $kG + G - 1$. */
  weight: Tensor
  /** GAT: the source half $\avec_{\mathrm{src}}$ of the attention vector, one row per head ($H \times G$). */
  attSource?: Tensor
  /** GAT: the destination half $\avec_{\mathrm{dst}}$ of the attention vector, one row per head ($H \times G$). */
  attTarget?: Tensor
  /** GATv2: the attention vector $\avec$, one row per head ($H \times G$). */
  att?: Tensor
  /** The bias: $HG$ values when heads are concatenated, $G$ when averaged (default none). */
  bias?: Tensor
}

/** Options of {@link graphAttention}. */
export interface GraphAttentionOptions {
  /** Heads $H$ (default 1). */
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
  /** The new features, $V \times HG$ (heads concatenated) or $V \times G$ (averaged). */
  readonly output: Value
  /** $\alpha$ per message edge and head, $E \times H$: in each column, the weights into one node sum to one. */
  readonly attention: Value
  /** The source node $u$ of each message edge ($E$ entries, the self-loops last). */
  readonly source: Int32Array
  /** The destination node $v$ of each message edge ($E$ entries). */
  readonly destination: Int32Array
}

/**
 * One graph attention pass (GAT, or GATv2 with `variant: 'v2'`; see the file notes). Differentiable in `h` and the
 * parameters. Throws `ShapeError` unless `h` has one row per node and $\Wmat$'s columns split into the heads, and
 * `DomainError` when the attention vectors of the variant are missing.
 *
 * @param g The graph.
 * @param h The node features, $V \times F$.
 * @param params The map $\Wmat$, the attention vectors of the variant and the optional bias.
 * @param options The number of heads, how they are combined, the LeakyReLU slope, self-loops and the variant.
 * @returns The output with the attention weights on every message edge and the edges they belong to.
 *
 * @example Zero attention vectors attend uniformly: the mean over each neighbourhood
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const h = tensor([[1, 0], [0, 1], [1, 1]])
 * const params = { weight: tensor([[1, 0], [0, 1]]), attSource: tensor([[0, 0]]), attTarget: tensor([[0, 0]]) }
 * const { output, attention, source, destination } = graphAttention(g, h, params)
 * print('edges:', source, '->', destination)
 * print('attention:', attention)
 * print('output:', output)
 */
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

/**
 * A graph attention layer over a fixed graph; `forward` also returns the attention weights. `options` are the
 * `graphAttention` options the layer was built with.
 */
export interface GraphAttentionLayer extends Layer<GraphAttentionParams> {
  readonly options: GraphAttentionOptions
  forward(params: GraphAttentionParams, x: Value, ctx?: Context): GraphAttentionResult
}

/**
 * A graph attention layer ($F$ inputs to $H$ heads of width $G$) over a fixed graph, Glorot-uniform initialised as in
 * Veličković et al., with a zero bias; `apply` returns the output of `graphAttention`, `forward` the whole result.
 *
 * @param g The graph every call runs on.
 * @param inFeatures The input width $F$.
 * @param outPerHead The width $G$ of each head.
 * @param options The `graphAttention` options, and `bias` (default true) to include the bias.
 * @returns The layer, with `forward` for the attention weights.
 *
 * @example Two heads on a 3-node graph
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const layer = GraphAttention(g, 2, 3, { heads: 2 })
 * const { output, attention } = layer.forward(layer.init(stream(0)), tensor([[1, 0], [0, 1], [1, 1]]))
 * print('output shape:', shapeOf(output))
 * print('attention (edge x head):', attention)
 */
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

/**
 * How GraphSAGE aggregates the neighbours: `mean`, `sum`, `max`, or `pool`, Hamilton's max-pooling of
 * $\mathrm{ReLU}(\Wmat_{\mathrm{pool}} \hvec_u + \bvec_{\mathrm{pool}})$.
 */
export type SageAggregator = Aggregation | 'pool'

/**
 * Parameters of a GraphSAGE layer: `self` ($\Wmat_{\mathrm{self}}$, $F \times G$, whose bias is the layer's),
 * `neighbour` ($\Wmat_{\mathrm{neigh}}$, $F \times G$) and, for the `pool` aggregator, `pool` ($F \times F$ with
 * bias).
 */
export type SageParams = { self: LinearParams; neighbour: LinearParams; pool?: LinearParams }

/** Options of {@link sageConv}. */
export interface SageOptions {
  /** How the neighbours are aggregated (default `mean`). */
  aggregate?: SageAggregator
  /** Scale each output row to unit Euclidean length (Hamilton et al., Algorithm 1, line 7). Default false. */
  normalise?: boolean
}

/**
 * One GraphSAGE pass,
 * $\hvec'_v = \Wmat_{\mathrm{self}} \hvec_v + \Wmat_{\mathrm{neigh}} \mathrm{AGG}\{\hvec_u : u \to v\} + \bvec$,
 * the neighbourhood without self-loops (a node with no neighbours aggregates zeros). Differentiable in `h` and the
 * parameters. Throws `ShapeError` unless `h` has one row per node, and `DomainError` for `pool` without `pool`
 * parameters.
 *
 * @param g The graph, or a neighbourhood sample of it from `sampleNeighbours`.
 * @param h The node features, $V \times F$.
 * @param params The self, neighbour and (for `pool`) pooling maps.
 * @param options The aggregator and whether to scale each output row to unit length.
 * @returns The new features, $V \times G$.
 *
 * @example Mean and max of the neighbours alone
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const h = tensor([[1, 0], [0, 1], [1, 1]])
 * // No self term, identity on the neighbours: the output is the aggregate itself.
 * const params = { self: { weight: tensor([[0, 0], [0, 0]]) }, neighbour: { weight: tensor([[1, 0], [0, 1]]) } }
 * print('mean:', sageConv(g, h, params))
 * print('max:', sageConv(g, h, params, { aggregate: 'max' }))
 */
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

/**
 * A GraphSAGE layer ($F \to G$) over a fixed graph, LeCun-uniform initialised with zero biases; the `pool`
 * aggregator adds an $F \to F$ pooling map. `apply` is `sageConv` on `g`.
 *
 * @param g The graph every call runs on.
 * @param inFeatures The input width $F$.
 * @param outFeatures The output width $G$.
 * @param options The aggregator and the row normalisation.
 * @returns The layer: `init` draws `SageParams`, `apply` maps $V \times F$ features to $V \times G$.
 *
 * @example A max-pooling layer on a 3-node graph
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const layer = SageConv(g, 2, 3, { aggregate: 'pool' })
 * const p = layer.init(stream(0))
 * print(layer.label, Object.keys(p))
 * print('output:', layer.apply(p, tensor([[1, 0], [0, 1], [1, 1]])))
 */
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
 * sample. Edge weights are dropped and node labels kept. Throws `DomainError` unless `size` is a positive integer.
 *
 * @param g The graph to sample from (its message edges, both directions of an undirected edge).
 * @param size The largest number $S$ of incoming edges kept per node; a node with fewer keeps them all.
 * @param stream The random stream; node $v$ draws from its own child stream, so its sample does not depend on the
 *   other nodes.
 * @returns A directed graph on the same nodes with the sampled edges.
 *
 * @example Node 0 of a star keeps two of its three neighbours
 * const star = [{ from: 0, to: 1 }, { from: 0, to: 2 }, { from: 0, to: 3 }]
 * const s = sampleNeighbours({ kind: 'graph', nodes: 4, edges: star, directed: false }, 2, stream(0))
 * print('directed:', s.directed)
 * print('edges:', s.edges.map((e) => `${e.from} -> ${e.to}`).join(', '))
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
  /** How the messages into a node are combined (default `sum`, as Gilmer et al.). */
  aggregate?: Aggregation
  /** Pass each node a message from itself too, with edge weight 1. Default false. */
  selfLoops?: boolean
}

/**
 * One generic message-passing step:
 * $\mvec_v = \mathrm{AGG}_{u \to v}\, \mathrm{message}([\hvec_u, \hvec_v, w_{uv}])$ and
 * $\hvec'_v = \mathrm{update}([\hvec_v, \mvec_v])$, with $w_{uv}$ the edge weight (1 where the edge has none).
 * Differentiable whenever `message` and `update` are. Throws `ShapeError` unless `h` has one row per node.
 *
 * @param g The graph.
 * @param h The node features, $V \times F$.
 * @param message The message function, from the $E \times (2F + 1)$ rows $[\hvec_u, \hvec_v, w_{uv}]$ (one per
 *   message edge) to $E \times M$ messages; any differentiable function, such as a layer's `apply` with its parameters.
 * @param update The update function, from the $V \times (F + M)$ rows $[\hvec_v, \mvec_v]$ to the new features.
 * @param options The aggregation and self-loops.
 * @returns What `update` returns, normally $V \times G$.
 *
 * @example Identity message and update show what each node receives
 * // Node 0 receives [h1, h0, w01] = [2, 1, 2] from node 1 over the edge of weight 2.
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1, weight: 2 }, { from: 1, to: 2 }], directed: false }
 * const h = tensor([[1], [2], [3]])
 * print('[h_v, m_v] =', messagePassing(g, h, (x) => x, (x) => x))
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

/** Parameters of a message-passing layer: those of its `message` and `update` layers. */
export type MessagePassingParams = { message: Params; update: Params }

/**
 * A message-passing layer over a fixed graph from a message layer ($2F + 1 \to M$) and an update layer
 * ($F + M \to G$), such as small `Mlp`s; `apply` is `messagePassing` on `g` with the two layers' `apply`.
 *
 * @param g The graph every call runs on.
 * @param message The message layer, applied to the $E \times (2F + 1)$ rows $[\hvec_u, \hvec_v, w_{uv}]$.
 * @param update The update layer, applied to the $V \times (F + M)$ rows $[\hvec_v, \mvec_v]$.
 * @param options The aggregation and self-loops.
 * @returns The layer, whose parameters are those of the two layers.
 *
 * @example Two hand-written linear layers as message and update
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const lin = (i, o) => ({
 *   kind: 'lin',
 *   label: `lin(${i}, ${o})`,
 *   init: (s) => ({ w: normals(s, [i, o]) }),
 *   apply: (p, x) => matmul(x, p.w),
 * })
 * const layer = MessagePassing(g, lin(3, 2), lin(3, 2), { aggregate: 'mean' })
 * const out = layer.apply(layer.init(stream(0)), tensor([[1], [0], [-1]]))
 * print(layer.label)
 * print('output:', out)
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
