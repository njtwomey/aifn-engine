/**
 * Link prediction with a graph autoencoder (Kipf and Welling 2016, "Variational graph auto-encoders", NIPS Bayesian
 * Deep Learning workshop: the non-variational GAE of §2 and the evaluation protocol of §3).
 *
 * `splitEdges` holds out a share of the edges as positive test pairs, draws as many non-edges of the full graph as
 * negative test pairs, and returns the graph of the remaining edges, the only graph the model passes messages over.
 * `linkPredictionRun` trains the GAE, a two-layer graph-convolution encoder
 * $\Zmat = \hat\Amat\,\mathrm{ReLU}(\hat\Amat\Xmat\Wmat_1)\Wmat_2$ with the inner-product decoder
 * $\sigma(\zvec_u^\top\zvec_v)$, by full-batch Adam on the binary cross-entropy of the training edges against an equal
 * number of non-edges drawn afresh at every step, and yields the test AUC at checkpoints. The classical baselines,
 * common neighbours and Adamic-Adar, are `pairHeuristics` of `aifn-compute/graph/measures`.
 *
 * Differences from the paper: negatives are resampled each step instead of fixed per epoch, and the loss weighs
 * positives and negatives equally instead of reweighting the full adjacency. As is common practice, a training
 * negative is any pair that is not a training edge, so it may be a held-out positive, and negatives are not
 * deduplicated. A run takes about a quarter of a second on a 60-node graph; it is a generator of plain inputs, so a
 * worker can stream it.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, integers, permutation, stream, type Stream } from 'aifn-compute/foundation/random'
import { mul, shapeOfValue, sum, take, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { fromEdges, type Graph } from 'aifn-compute/graph'
import type { NodePair } from 'aifn-compute/graph/measures'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { auroc } from 'aifn-compute/learning/metrics'
import { relu } from 'aifn-compute/nn/functional'
import { GraphConv } from 'aifn-compute/nn/graph'
import type { Context, LinearParams } from 'aifn-compute/nn/layers'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'

/** The result of `splitEdges`. */
export interface EdgeSplit {
  /** The graph of the edges kept for message passing and training (undirected, unweighted). */
  readonly train: Graph
  /** The held-out edges: the positive test pairs. */
  readonly testPositive: NodePair[]
  /** Non-edges of the full graph, as many as the held-out edges: the negative test pairs, each $u < v$. */
  readonly testNegative: NodePair[]
}

/**
 * The key of an unordered pair.
 *
 * @param u One end.
 * @param v The other end.
 * @returns The same string for $(u, v)$ and $(v, u)$.
 */
const key = (u: number, v: number) => (u < v ? `${u},${v}` : `${v},${u}`)

/**
 * Draw distinct pairs $u \ne v$ that are not in `present`, by rejection from uniform node pairs. Throws `DomainError`
 * when there are fewer such pairs than asked for.
 *
 * @param s The stream the node draws come from (advanced).
 * @param n The number of nodes.
 * @param count The number of pairs to draw.
 * @param present The keys of the pairs to avoid.
 * @param distinct Whether the pairs drawn must differ from one another.
 * @param where The caller's name, for error messages.
 * @returns The pairs, in the order drawn.
 */
function nonEdges(
  s: Stream,
  n: Size,
  count: Size,
  present: ReadonlySet<string>,
  distinct: boolean,
  where: string,
): NodePair[] {
  const available = (n * (n - 1)) / 2 - present.size
  if (available < (distinct ? count : 1))
    throw new DomainError(where, `${where}: the graph has ${available} non-edges, too few to draw ${count}`)
  const out: NodePair[] = []
  const chosen = new Set<string>()
  while (out.length < count) {
    const u = integers(s, n)
    const v = integers(s, n)
    const k = key(u, v)
    if (u === v || present.has(k) || (distinct && chosen.has(k))) continue
    chosen.add(k)
    out.push(distinct ? [Math.min(u, v), Math.max(u, v)] : [u, v])
  }
  return out
}

/**
 * Hold out a share of the edges of an undirected graph and draw as many non-edges (see the file comment). Self-loops
 * are dropped. Throws `DomainError` when the share leaves no test or no training edges, or the graph has too few
 * non-edges.
 *
 * @param s The random stream: the held-out edges are a permutation drawn from its child `'edges'`, the non-edges come
 *   from its child `'negatives'`.
 * @param g The undirected graph, each edge listed once.
 * @param holdOut The share of edges to hold out, in $(0, 1)$.
 * @returns The training graph and the positive and negative test pairs.
 *
 * @example Hold out 15% of a block model's edges
 * import { stochasticBlockModel } from 'aifn-compute/graph/structures'
 * const g = stochasticBlockModel(stream('link/graph'), [20, 20, 20], { pIn: 0.3, pOut: 0.02 })
 * const split = splitEdges(stream('link/split'), g, 0.15)
 * print('edges:', g.edges.length, ' train:', split.train.edges.length, ' test:', split.testPositive.length)
 */
export function splitEdges(s: Stream, g: Graph, holdOut: number): EdgeSplit {
  const edges = g.edges.filter((e) => e.from !== e.to)
  const m = Math.round(holdOut * edges.length)
  if (!(m >= 1 && m < edges.length))
    throw new DomainError('splitEdges', 'splitEdges: the hold-out share leaves no test or no training edges')
  const order = toFlat(permutation(child(s, 'edges'), edges.length))
  const held = new Set(Array.from(order.slice(0, m)))
  const present = new Set(edges.map((e) => key(e.from, e.to)))
  const testNegative = nonEdges(child(s, 'negatives'), g.nodes, m, present, true, 'splitEdges')
  return {
    train: fromEdges(
      g.nodes,
      edges.filter((_, i) => !held.has(i)).map((e) => [e.from, e.to] as const),
      { directed: false },
    ),
    testPositive: edges.filter((_, i) => held.has(i)).map((e) => [e.from, e.to] as const),
    testNegative,
  }
}

/** Options of `linkPredictionRun`. */
export interface LinkPredictionOptions {
  /** The edge split; the encoder passes messages over `split.train` only. */
  readonly split: EdgeSplit
  /** The node features $\Xmat$, $V \times F$ (the identity matrix for one-hot node identities). */
  readonly x: Tensor
  /** The hidden width of the first layer (default 16). */
  readonly hidden?: number
  /** The embedding width $k$ (default 8). */
  readonly embedding?: number
  /** The number of Adam steps (default 200). */
  readonly steps?: number
  /** The Adam step size (default 0.01). */
  readonly stepSize?: number
  /** The number of steps between checkpoints (default 5). */
  readonly every?: number
  /** The seed of the initial weights and of the negative samples: the root stream is `stream('link/' + seed)`. */
  readonly seed?: number | string
}

/** A checkpoint of `linkPredictionRun`. */
export interface LinkCheckpoint {
  /** The step. */
  readonly step: number
  /** The training loss: the mean binary cross-entropy over training edges and sampled non-edges. */
  readonly loss: number
  /** The AUC of the inner-product scores of the held-out edges against the held-out non-edges. */
  readonly auc: number
  /** The scores $\zvec_u^\top\zvec_v$ of the positive test pairs, then of the negative ones. */
  readonly scores: number[]
}

/** The parameters of the two encoder layers. */
type Params = [LinearParams, LinearParams]

/**
 * Train a graph autoencoder by full-batch Adam and yield the held-out AUC every `every` steps, from step 0, and at the
 * last step (see the file comment). Each step samples one non-edge per training edge, uniformly among the node pairs
 * of the training graph that are not training edges. Throws `DomainError` when the training graph has no non-edges.
 *
 * @param options The split, the features, the widths, the training and the seed.
 * @returns A generator of checkpoints.
 *
 * @example Training lifts the held-out AUC above chance
 * import { stochasticBlockModel } from 'aifn-compute/graph/structures'
 * const g = stochasticBlockModel(stream('link/graph'), [20, 20, 20], { pIn: 0.3, pOut: 0.02 })
 * const split = splitEdges(stream('link/split'), g, 0.15)
 * let last
 * for (const c of linkPredictionRun({ split, x: eye(60), steps: 40, stepSize: 0.02, every: 20, seed: 1 })) last = c
 * print('step', last.step, ' AUC', last.auc)
 */
export function* linkPredictionRun(options: LinkPredictionOptions): Generator<LinkCheckpoint> {
  const { split, x, hidden = 16, embedding = 8, steps = 200, stepSize = 0.01, every = 5, seed = 0 } = options
  const g = split.train
  const n = g.nodes
  const features = shapeOfValue(x)[1]
  const first = GraphConv(g, features, hidden, { bias: false })
  const second = GraphConv(g, hidden, embedding, { bias: false })
  const encode = (p: Params): Value => second.apply(p[1], relu(first.apply(p[0], x)))
  const positives = g.edges.map((e) => [e.from, e.to] as const)
  const present = new Set(positives.map(([u, v]) => key(u, v)))
  const pairScores = (z: Value, pairs: readonly NodePair[]): Value =>
    sum(
      mul(
        take(
          z,
          pairs.map((p) => p[0]),
        ),
        take(
          z,
          pairs.map((p) => p[1]),
        ),
      ),
      1,
    )
  const targets = tensor([...positives.map(() => 1), ...positives.map(() => 0)])
  const loss = (p: Params, _batch: unknown, ctx: Context) => {
    const z = encode(p)
    const s = child(ctx.stream ?? stream('link/negatives'), 'negatives')
    const negatives = nonEdges(s, n, positives.length, present, false, 'linkPredictionRun')
    return binaryCrossEntropyWithLogits(pairScores(z, [...positives, ...negatives]), targets)
  }
  const root = stream(`link/${seed}`)
  const start = { params: [first.init(child(root, 'first')), second.init(child(root, 'second'))] as Params }
  const alg = trainingLoop({ loss, data: { i: tensor([0]) }, optimizer: adamRule({ stepSize }) })
  const test = [...split.testPositive, ...split.testNegative]
  const labels = [...split.testPositive.map(() => 1), ...split.testNegative.map(() => 0)]
  for (const { step, state } of live(alg, start, { stream: child(root, 'run') })) {
    const last = step >= steps || state.diverged
    if (step % every === 0 || last) {
      const scores = Array.from(toFlat(pairScores(encode(state.params), test) as Tensor))
      yield { step, loss: state.loss, auc: auroc(labels, scores), scores }
    }
    if (last) return
  }
}
