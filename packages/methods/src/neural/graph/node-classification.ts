/**
 * Semi-supervised node classification with a two-layer graph neural network, the experiment of Kipf and Welling
 * (2017): a few nodes carry labels, every node's features pass through two message-passing layers, and the
 * cross-entropy on the labelled nodes trains both layers by Adam. The first layer is a graph convolution, a graph
 * attention layer or a GraphSAGE layer (`aifn-compute/nn/graph`) followed by tanh; the second maps to one logit per class. With
 * two classes the logits are a 2-D embedding of every node, so a page can draw the classes separating.
 *
 * `nodeClassificationRun` is a generator: it yields snapshots (curves, and the logits, hidden features and attention
 * weights at checkpoints), so a worker can stream the run to a page.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  dense,
  fromData,
  mul,
  neg,
  sum,
  take,
  tanh,
  toRows,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { Graph } from 'aifn-compute/graph'
import { logSoftmax } from 'aifn-compute/numerics/special'
import {
  GraphAttention,
  GraphConv,
  SageConv,
  type GraphAttentionLayer,
  type GraphAttentionParams,
  type SageAggregator,
} from 'aifn-compute/nn/graph'
import type { Layer } from 'aifn-compute/nn/layers'
import { trainingLoop, type TrainingState } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The first layer's kind. */
export type GnnKind = 'gcn' | 'gat' | 'sage'

/** The network: its layers over one graph. */
export interface GnnSpec {
  readonly kind: GnnKind
  readonly features: Size
  readonly hidden: Size
  readonly classes: Size
  /** GAT: heads of the first layer (concatenated). */
  readonly heads: Size
  /** GraphSAGE: the aggregator of both layers. */
  readonly aggregate: SageAggregator
}

/** Two layers over a graph, and what a forward pass exposes. */
export interface GnnModel {
  readonly spec: GnnSpec
  readonly first: Layer
  readonly second: Layer
  init(s: ReturnType<typeof stream>): Params[]
  /** Hidden features after tanh (V × width) and the logits (V × C). */
  forward(params: Params[], x: Value): { hidden: Value; logits: Value }
  /** GAT: the first layer's attention, E × heads, and its edges. */
  attention?(params: Params[], x: Value): { weights: Tensor; source: Int32Array; destination: Int32Array }
}

/** A two-layer GNN of the given kind over a graph (module notes). */
export function gnnModel(graph: Graph, spec: GnnSpec): GnnModel {
  const { kind, features, hidden, classes, heads, aggregate } = spec
  const width = kind === 'gat' ? heads * hidden : hidden
  const first: Layer =
    kind === 'gcn'
      ? GraphConv(graph, features, hidden)
      : kind === 'gat'
        ? GraphAttention(graph, features, hidden, { heads })
        : SageConv(graph, features, hidden, { aggregate })
  const second: Layer =
    kind === 'gcn'
      ? GraphConv(graph, width, classes)
      : kind === 'gat'
        ? GraphAttention(graph, width, classes, { heads: 1, concat: false })
        : SageConv(graph, width, classes, { aggregate })
  const model: GnnModel = {
    spec,
    first,
    second,
    init: (s) => [first.init(child(s, 'first')), second.init(child(s, 'second'))],
    forward: (p, x) => {
      const h = tanh(first.apply(p[0], x))
      return { hidden: h, logits: second.apply(p[1], h) }
    },
  }
  if (kind === 'gat')
    return {
      ...model,
      attention: (p, x) => {
        const r = (first as GraphAttentionLayer).forward(p[0] as GraphAttentionParams, x)
        return { weights: r.attention as Tensor, source: r.source, destination: r.destination }
      },
    }
  return model
}

/** The mean cross-entropy of the logits on the labelled nodes `train` with one-hot targets [L, C]. */
export function nodeLoss(model: GnnModel, params: Params[], x: Value, train: Int32Array, targets: Tensor): Value {
  const logp = logSoftmax(model.forward(params, x).logits)
  return neg(sum(mul(take(logp, train), targets)) as Value) as Value
}

/** Options of `nodeClassificationTraining`. */
export interface NodeTrainingOptions {
  /** Adam's step size (default 0.01). */
  stepSize?: number
}

/** Full-batch Adam on the labelled nodes' mean cross-entropy, as a traceable `trainingLoop`. */
export function nodeClassificationTraining(
  model: GnnModel,
  x: Tensor,
  labels: ArrayLike<number>,
  train: readonly number[],
  options: NodeTrainingOptions = {},
): Algorithm<{ params: Params[] }, TrainingState<Params[]>> {
  const C = model.spec.classes
  const idx = Int32Array.from(train)
  const onehot = new Float64Array(idx.length * C)
  idx.forEach((v, r) => (onehot[r * C + labels[v]] = 1 / idx.length))
  const targets = fromData(onehot, [idx.length, C])
  return trainingLoop<Params[], { targets: Tensor }>({
    loss: (p, b) => nodeLoss(model, p, x, idx, b.targets),
    data: { targets },
    optimizer: adamRule({ stepSize: options.stepSize ?? 0.01 }) as never,
  })
}

/** Options of `nodeClassificationRun`. */
export interface NodeClassificationOptions {
  /**
   * The graph, the node features x [V, F] and the node classes y [V] (every node's, for the accuracy curves; only
   * `train`'s are used to fit), e.g. `aifn-methods/data/real`'s `karateClub()`.
   */
  data: { graph: Graph; x: Tensor; y?: Tensor }
  /** The labelled nodes. */
  train: readonly number[]
  kind?: GnnKind
  /** Hidden width (per head for GAT). Default 8. */
  hidden?: Size
  heads?: Size
  aggregate?: SageAggregator
  /** Adam updates (default 200). */
  steps?: Size
  stepSize?: number
  /** Keep a checkpoint every this many steps (default steps/100, at least 1). */
  every?: Size
  seed?: string | number
}

/** Curves of a run, one entry per step. */
export interface NodeHistory {
  step: number[]
  /** Cross-entropy on the labelled nodes. */
  loss: number[]
  /** Accuracy on the labelled nodes and on the unlabelled ones. */
  trainAccuracy: number[]
  testAccuracy: number[]
}

/** The network's view of every node at one step. */
export interface NodeCheckpoint {
  readonly step: Size
  /** Logits [V][C]. */
  readonly logits: number[][]
  /** Hidden features after the first layer [V][width]. */
  readonly hidden: number[][]
  /** GAT: the first layer's attention per message edge, averaged over heads. */
  readonly attention?: number[]
}

/** A snapshot of `nodeClassificationRun`. */
export interface NodeSnapshot {
  readonly step: Size
  readonly steps: Size
  readonly done: boolean
  readonly spec: GnnSpec
  readonly history: NodeHistory
  readonly checkpoints: readonly NodeCheckpoint[]
  /** GAT: the message edges the attention weights belong to. */
  readonly edges?: { source: number[]; destination: number[] }
}

const argmax = (row: readonly number[]) => row.reduce((b, v, i) => (v > row[b] ? i : b), 0)

/** Train a two-layer GNN on the labelled nodes by full-batch Adam, yielding snapshots (module notes). */
export function* nodeClassificationRun(options: NodeClassificationOptions): Generator<NodeSnapshot> {
  const { data, train, kind = 'gcn', hidden = 8, heads = 2, aggregate = 'mean', steps = 200 } = options
  const { graph, x, y } = data
  if (!y) throw new DomainError('nodeClassificationRun', 'nodeClassificationRun: the data need node classes y')
  const labels = Array.from(dense.data(y))
  const classes = Math.max(...labels) + 1
  const spec: GnnSpec = { kind, features: x.shape[1], hidden, classes, heads, aggregate }
  const model = gnnModel(graph, spec)
  const every = Math.max(1, options.every ?? Math.round(steps / 100))
  const root = stream(options.seed ?? 'gnn')
  const alg = nodeClassificationTraining(model, x, labels, train, { stepSize: options.stepSize })
  const isTrain = new Set(train)
  const history: NodeHistory = { step: [], loss: [], trainAccuracy: [], testAccuracy: [] }
  const checkpoints: NodeCheckpoint[] = []
  let edges: NodeSnapshot['edges']
  const record = (t: Size, params: Params[], loss: number, keep: boolean) => {
    const out = model.forward(params, x)
    const logits = toRows(out.logits as Tensor) as number[][]
    let a = 0
    let b = 0
    logits.forEach((row, v) => {
      const ok = argmax(row) === labels[v] ? 1 : 0
      if (isTrain.has(v)) a += ok
      else b += ok
    })
    history.step.push(t)
    history.loss.push(loss)
    history.trainAccuracy.push(a / Math.max(1, isTrain.size))
    history.testAccuracy.push(b / Math.max(1, labels.length - isTrain.size))
    if (!keep) return
    let attention: number[] | undefined
    if (model.attention) {
      const r = model.attention(params, x)
      const w = toRows(r.weights) as number[][]
      attention = w.map((row) => row.reduce((s, v) => s + v, 0) / row.length)
      edges ??= { source: Array.from(r.source), destination: Array.from(r.destination) }
    }
    checkpoints.push({
      step: t,
      logits,
      hidden: toRows(out.hidden as Tensor) as number[][],
      ...(attention ? { attention } : {}),
    })
  }
  const snapshot = (t: Size, done: boolean): NodeSnapshot => ({
    step: t,
    steps,
    done,
    spec,
    history: {
      step: [...history.step],
      loss: [...history.loss],
      trainAccuracy: [...history.trainAccuracy],
      testAccuracy: [...history.testAccuracy],
    },
    checkpoints: [...checkpoints],
    ...(edges ? { edges } : {}),
  })
  let state = alg.init({ params: model.init(child(root, 'init')) }, child(root, 'init'))
  record(0, state.params, Number(state.loss), true)
  yield snapshot(0, false)
  for (let t = 0; t < steps; t++) {
    state = alg.step(state, { t, stream: child(root, 'step', t) })
    const k = t + 1
    record(k, state.params, Number(state.loss), k % every === 0 || k === steps)
    if (k % every === 0 || k === steps || state.diverged) yield snapshot(k, k === steps || state.diverged)
    if (state.diverged) return
  }
}
