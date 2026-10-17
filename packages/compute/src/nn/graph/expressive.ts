/**
 * Aggregators that tell more neighbourhoods apart than the mean of a graph convolution: the graph isomorphism network
 * and principal neighbourhood aggregation.
 *
 * - **GIN** (Xu, Hu, Leskovec and Jegelka 2019, "How powerful are graph neural networks?", ICLR, §4.1, eq. 4.1):
 *   $\hvec'_v = \mathrm{MLP}\big((1 + \epsilon)\hvec_v + \sum_{u \to v} \hvec_u\big)$, injective on multisets and as
 *   strong as the 1-WL test. `aggregate` swaps the sum for a mean or a max, which is not GIN but makes the
 *   comparison of aggregators that motivates it a one-line change. The MLP is any function the caller passes (the
 *   paper uses a 2-layer MLP with batch normalisation).
 * - **PNA** (Corso, Cavalleri, Beaini, Liò and Veličković 2020, "Principal neighbourhood aggregation for graph nets",
 *   NeurIPS, eq. 4 to 8): several aggregators (mean, max, min, standard deviation, sum) of the incoming messages, each
 *   multiplied by degree scalers $(\log(d + 1)/\delta)^{\alpha}$ with $\alpha = 0$ (identity), $1$ (amplification)
 *   or $-1$ (attenuation), $d$ the in-degree and $\delta$ the mean of $\log(d + 1)$ over the training graphs.
 *
 * Messages run along the message edges $u \to v$ (both directions of an undirected edge). GIN's sum weights each
 * message by its edge weight, as `propagate` does; PNA's aggregators ignore the weights. Both are differentiable in
 * the features and the parameters.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  fromData,
  maximum,
  mul,
  neg,
  scalar,
  sqrt,
  square,
  sub,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { messageEdges, propagate, type Aggregation } from 'aifn-compute/graph/propagation'
import { Mlp, tap, type Layer, type MlpOptions } from 'aifn-compute/nn/layers'
import { checkFeatures } from './layers'

// ── GIN ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `ginConv`. */
export interface GinOptions {
  /** The weight $\epsilon$ on the node's own vector, $(1 + \epsilon)\hvec_v$ (default 0): a number or a scalar. */
  epsilon?: Value
  /** The neighbour aggregation (default `sum`, the GIN layer). */
  aggregate?: Aggregation
}

/**
 * One GIN pass: `mlp` of $(1 + \epsilon)\hvec_v$ plus the aggregate of the neighbours' vectors (see the file
 * comment). Differentiable in `h`, `epsilon` and whatever `mlp` closes over. Throws `ShapeError` unless `h` has one
 * row per node.
 *
 * @param g The graph; an undirected edge passes a message each way.
 * @param h The node features, $V \times F$.
 * @param mlp The map applied to the combined $V \times F$ matrix, row by row (a layer's `apply` with its parameters).
 * @param options The weight on the node's own vector and the aggregation.
 * @returns The new features, $V \times G$.
 *
 * @example An edge whose ends carry 1 and 2: GIN-0 gives both ends 3, GIN with $\epsilon = 0.5$ separates them
 * const edge = { kind: 'graph', nodes: 2, edges: [{ from: 0, to: 1 }], directed: false }
 * const h = tensor([[1], [2]])
 * print('GIN-0:', ginConv(edge, h, (x) => x))
 * print('epsilon 0.5:', ginConv(edge, h, (x) => x, { epsilon: 0.5 }))
 */
export function ginConv(g: Graph, h: Value, mlp: (x: Value) => Value, options: GinOptions = {}): Value {
  checkFeatures(g, h, 'ginConv')
  const epsilon = options.epsilon ?? 0
  const neighbours = propagate(g, h, { aggregate: options.aggregate ?? 'sum' })
  return mlp(add(mul(add(1, epsilon), h), neighbours))
}

/** Parameters of `GinConv`: the MLP's and, when learned, $\epsilon$. */
export type GinParams = {
  /** The parameters of the MLP, as `Mlp` lays them out. */
  mlp: Params[]
  /** The learned $\epsilon$, a scalar; absent when $\epsilon$ is fixed. */
  epsilon?: Tensor
}

/** Options of `GinConv`: the MLP's (activations, initialiser, dropout) and the GIN-specific ones. */
export interface GinLayerOptions extends MlpOptions {
  /** The hidden widths of the MLP (default one hidden layer as wide as the output). */
  hidden?: readonly number[]
  /** The starting (or fixed) $\epsilon$ (default 0). */
  epsilon?: number
  /** Learn $\epsilon$ as a parameter, GIN-$\epsilon$ (default false: fixed, GIN-0 when 0). */
  learnEpsilon?: boolean
  /** The neighbour aggregation (default `sum`). */
  aggregate?: Aggregation
}

/**
 * A GIN layer ($F \to G$) over a fixed graph, its MLP an `Mlp` of widths $F$, the hidden widths, $G$; `apply` is
 * `ginConv` on `g`.
 *
 * @param g The graph every call runs on.
 * @param inFeatures The input width $F$.
 * @param outFeatures The output width $G$.
 * @param options The MLP's hidden widths and activations, $\epsilon$ and whether it is learned, and the aggregation.
 * @returns The layer: `init` draws the MLP from the child stream `'mlp'` (and sets $\epsilon$ when learned), `apply`
 *   maps $V \times F$ features to $V \times G$.
 *
 * @example A GIN-epsilon layer on a path of three nodes
 * const g = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const layer = GinConv(g, 1, 4, { hidden: [8], learnEpsilon: true, epsilon: 0.1 })
 * const p = layer.init(stream(3))
 * print(layer.label, ' epsilon:', p.epsilon)
 * print('output shape:', shapeOf(layer.apply(p, tensor([[1], [2], [3]]))))
 */
export function GinConv(
  g: Graph,
  inFeatures: Size,
  outFeatures: Size,
  options: GinLayerOptions = {},
): Layer<GinParams> {
  const { hidden = [outFeatures], epsilon = 0, learnEpsilon = false, aggregate = 'sum', ...mlpOptions } = options
  const mlp = Mlp([inFeatures, ...hidden, outFeatures], mlpOptions)
  return {
    kind: 'GinConv',
    label: `GinConv(${inFeatures} → ${outFeatures}${learnEpsilon ? ', learned ε' : `, ε = ${epsilon}`})`,
    init: (s: Stream): GinParams => ({
      mlp: mlp.init(child(s, 'mlp')),
      ...(learnEpsilon ? { epsilon: scalar(epsilon) } : {}),
    }),
    apply: (p, x, ctx) =>
      tap(
        ctx,
        ginConv(g, x, (z) => mlp.apply(p.mlp, z), {
          epsilon: learnEpsilon && p.epsilon ? p.epsilon : epsilon,
          aggregate,
        }),
      ),
  }
}

// ── PNA ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A PNA aggregator of the incoming messages. */
export type PnaAggregator = 'mean' | 'max' | 'min' | 'std' | 'sum'

/** A PNA degree scaler: $\alpha = 0$, $1$ or $-1$ in $(\log(d + 1)/\delta)^{\alpha}$. */
export type PnaScaler = 'identity' | 'amplification' | 'attenuation'

/** Options of `pnaAggregate`. */
export interface PnaOptions {
  /** The aggregators, in output order (default mean, max, min, std). */
  aggregators?: readonly PnaAggregator[]
  /** The scalers, in output order (default identity, amplification, attenuation). */
  scalers?: readonly PnaScaler[]
  /** The normaliser $\delta$ (default `pnaDelta` of this graph). */
  delta?: number
  /** The $\varepsilon$ under the square root of the standard deviation (default 1e-5). */
  epsilon?: number
}

/**
 * The in-degree of every node: the number of message edges into it (both directions of an undirected edge).
 *
 * @param g The graph.
 * @returns The count of incoming message edges per node.
 */
function inDegrees(g: Graph): Float64Array {
  const d = new Float64Array(g.nodes)
  for (const v of messageEdges(g, false).destination) d[v]++
  return d
}

/**
 * PNA's degree normaliser $\delta = \frac{1}{N}\sum_v \log(d_v + 1)$ over all $N$ nodes of the given (training)
 * graphs, $d_v$ the number of messages into node $v$.
 *
 * @param graphs The training graphs.
 * @returns The mean log degree, or 1 when there are no nodes.
 *
 * @example A star with three leaves: $(\log 4 + 3 \log 2)/4$
 * import { starGraph } from 'aifn-compute/graph/structures'
 * print('delta =', pnaDelta([starGraph(3)]), ' check:', (Math.log(4) + 3 * Math.log(2)) / 4)
 */
export function pnaDelta(graphs: readonly Graph[]): number {
  let total = 0
  let count = 0
  for (const g of graphs)
    for (const d of inDegrees(g)) {
      total += Math.log(d + 1)
      count++
    }
  return count > 0 ? total / count : 1
}

/** The exponent $\alpha$ of each scaler. */
const ALPHA: Record<PnaScaler, number> = { identity: 0, amplification: 1, attenuation: -1 }

/**
 * The PNA aggregate of every node's incoming messages (see the file comment): for each scaler, then each aggregator,
 * an $F$-wide block, concatenated. The standard deviation is
 * $\sqrt{\max(\mathrm{E}[h^2] - \mathrm{E}[h]^2, 0) + \varepsilon}$. Edge weights are ignored; a node with no
 * incoming messages gets 0 in every block. Differentiable in `h`. Throws `ShapeError` unless `h` has one row per node.
 *
 * @param g The graph; an undirected edge passes a message each way.
 * @param h The node features, $V \times F$.
 * @param options The aggregators, scalers, $\delta$ and $\varepsilon$.
 * @returns A $V \times (F \cdot S \cdot A)$ matrix for $S$ scalers and $A$ aggregators.
 *
 * @example The centre of a star sees {-1, 1} or {-2, 2}: equal mean and sum, different max, min and std
 * const star = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 0, to: 2 }], directed: false }
 * const options = { aggregators: ['mean', 'max', 'min', 'std', 'sum'], scalers: ['identity'], epsilon: 0 }
 * print('{-1, 1}:', toRows(pnaAggregate(star, tensor([[0], [-1], [1]]), options))[0])
 * print('{-2, 2}:', toRows(pnaAggregate(star, tensor([[0], [-2], [2]]), options))[0])
 */
export function pnaAggregate(g: Graph, h: Value, options: PnaOptions = {}): Value {
  checkFeatures(g, h, 'pnaAggregate')
  const aggregators = options.aggregators ?? ['mean', 'max', 'min', 'std']
  const scalers = options.scalers ?? ['identity', 'amplification', 'attenuation']
  const delta = options.delta ?? pnaDelta([g])
  const epsilon = options.epsilon ?? 1e-5
  const degree = inDegrees(g)
  const plain = { message: (source: Value) => source }
  const mean = () => propagate(g, h, { ...plain, aggregate: 'mean' })
  const parts: Record<PnaAggregator, () => Value> = {
    mean,
    sum: () => propagate(g, h, { ...plain, aggregate: 'sum' }),
    max: () => propagate(g, h, { ...plain, aggregate: 'max' }),
    min: () => neg(propagate(g, neg(h), { ...plain, aggregate: 'max' })),
    std: () => {
      const m = mean()
      const m2 = propagate(g, square(h), { ...plain, aggregate: 'mean' })
      return sqrt(add(maximum(sub(m2, square(m)), 0), epsilon))
    },
  }
  const aggregated = aggregators.map((a) => parts[a]())
  const blocks: Value[] = []
  for (const s of scalers) {
    const column = fromData(
      degree.map((d) => (d > 0 ? Math.pow(Math.log(d + 1) / delta, ALPHA[s]) : 0)),
      [g.nodes, 1],
    )
    for (const a of aggregated) blocks.push(mul(a, column))
  }
  return concat(blocks, 1)
}
