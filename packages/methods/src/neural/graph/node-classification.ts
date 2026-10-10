/**
 * Semi-supervised node classification with a two-layer graph neural network, the experiment of Kipf and Welling
 * (2017): a few nodes carry labels, every node's features pass through two message-passing layers, and the
 * cross-entropy on the labelled nodes trains both layers by Adam. The first layer is a graph convolution, a graph
 * attention layer or a GraphSAGE layer (`aifn-compute/nn/graph`) followed by tanh; the second, of the same kind, maps
 * to one logit per class. With node features $\Xmat$ ($V \times F$) the network computes
 * $\Hmat = \tanh(f_1(\Xmat))$ and the logits $\Zmat = f_2(\Hmat)$ ($V \times C$), and the loss is
 * $-\frac{1}{L} \sum_{v \in \text{train}} \log \operatorname{softmax}(\zvec_v)_{y_v}$ over the $L$ labelled nodes. With
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
  /** The kind of both layers: graph convolution, graph attention or GraphSAGE. */
  readonly kind: GnnKind
  /** Input features per node $F$. */
  readonly features: Size
  /** Hidden units of the first layer (per head for GAT). */
  readonly hidden: Size
  /** Classes $C$, the second layer's outputs. */
  readonly classes: Size
  /** GAT: heads of the first layer (concatenated); the second has one. Read only for GAT. */
  readonly heads: Size
  /** GraphSAGE: the aggregator of both layers. Read only for GraphSAGE. */
  readonly aggregate: SageAggregator
}

/** Two layers over a graph, and what a forward pass exposes. */
export interface GnnModel {
  /** The network's specification. */
  readonly spec: GnnSpec
  /** The first layer, $F$ inputs to the hidden width. */
  readonly first: Layer
  /** The second layer, the hidden width to $C$ logits. */
  readonly second: Layer
  /** Fresh parameters of both layers, `[first, second]`, from stream `s`. */
  init(s: ReturnType<typeof stream>): Params[]
  /**
   * Hidden features after tanh ($V \times$ width, the width `hidden` times `heads` for GAT) and the logits
   * ($V \times C$), from node features `x` ($V \times F$).
   */
  forward(params: Params[], x: Value): { hidden: Value; logits: Value }
  /**
   * GAT only: the first layer's attention, $E \times$ `heads`, over its message edges (self-loops included), with
   * each edge's source and destination node.
   */
  attention?(params: Params[], x: Value): { weights: Tensor; source: Int32Array; destination: Int32Array }
}

/**
 * A two-layer GNN of the given kind over a graph (module notes): GCN, GAT (the first layer's heads concatenated, the
 * second's one head averaged) or GraphSAGE.
 *
 * @param graph The graph the layers pass messages over.
 * @param spec The kind, the sizes, and the GAT heads or the GraphSAGE aggregator.
 * @returns The model, with `attention` for GAT.
 *
 * @example A GCN on a path of three nodes
 * // The path 0 - 1 - 2 as plain data (`aifn-compute/graph`'s `fromEdges` builds the same).
 * const graph = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const model = gnnModel(graph, { kind: 'gcn', features: 3, hidden: 2, classes: 2, heads: 1, aggregate: 'mean' })
 * const { hidden, logits } = model.forward(model.init(stream(0)), eye(3))
 * print('hidden:', hidden)
 * print('logits:', logits)
 *
 * @example GAT's attention over each message edge, for two heads
 * const graph = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const model = gnnModel(graph, { kind: 'gat', features: 3, hidden: 2, classes: 2, heads: 2, aggregate: 'mean' })
 * const att = model.attention(model.init(stream(0)), eye(3))
 * print('source:', att.source, ' destination:', att.destination)
 * print('weights:', att.weights)
 */
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

/**
 * The cross-entropy of the logits on the labelled nodes, $-\sum_{r, c} T_{rc} \log p_{v_r c}$ with $v_r$ the $r$-th
 * labelled node: a sum weighted by the targets, so the mean cross-entropy when each target row is one-hot divided by
 * $L$, as `nodeClassificationTraining` builds them (differentiable in the parameters).
 *
 * @param model The two-layer GNN.
 * @param params Its parameters.
 * @param x The node features, $V \times F$.
 * @param train The $L$ labelled nodes.
 * @param targets The target weights $\Tmat$, $L \times C$, row $r$ for node `train[r]`.
 * @returns The weighted cross-entropy, a scalar.
 *
 * @example The two end nodes of a path, weighted $1/L$ each
 * const graph = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const model = gnnModel(graph, { kind: 'gcn', features: 3, hidden: 2, classes: 2, heads: 1, aggregate: 'mean' })
 * const train = Int32Array.of(0, 2)
 * print('loss:', nodeLoss(model, model.init(stream(0)), eye(3), train, tensor([[0.5, 0], [0, 0.5]])))
 * print('chance, log 2 =', Math.log(2))
 */
export function nodeLoss(model: GnnModel, params: Params[], x: Value, train: Int32Array, targets: Tensor): Value {
  const logp = logSoftmax(model.forward(params, x).logits)
  return neg(sum(mul(take(logp, train), targets)) as Value) as Value
}

/** Options of `nodeClassificationTraining`. */
export interface NodeTrainingOptions {
  /** Adam's step size (default 0.01). */
  stepSize?: number
}

/**
 * Full-batch Adam on the labelled nodes' mean cross-entropy (`nodeLoss`), as a traceable `trainingLoop`. Only the
 * labels of the `train` nodes are read.
 *
 * @param model The two-layer GNN.
 * @param x The node features, $V \times F$.
 * @param labels Every node's class, $0, \dots, C - 1$ (only the labelled nodes' are read).
 * @param train The labelled nodes.
 * @param options Adam's step size.
 * @returns The algorithm, to run with `run` or `trace` from `{ params }`.
 *
 * @example Two labelled ends of a path of three nodes
 * const graph = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const model = gnnModel(graph, { kind: 'gcn', features: 3, hidden: 2, classes: 2, heads: 1, aggregate: 'mean' })
 * const alg = nodeClassificationTraining(model, eye(3), [0, 0, 1], [0, 2], { stepSize: 0.1 })
 * const tr = trace(alg, { params: model.init(stream(0)) }, 60, { every: 20, record: { loss: (s) => s.loss } })
 * print('step:', tr.index)
 * print('loss:', tr.series.loss)
 */
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
   * The graph, the node features `x` ($V \times F$) and the node classes `y` ($V$ values, required: every node's, for
   * the accuracy curves; only `train`'s are used to fit), e.g. `aifn-methods/data/real`'s `karateClub()`. The number
   * of classes is the largest class plus one.
   */
  data: { graph: Graph; x: Tensor; y?: Tensor }
  /** The labelled nodes. */
  train: readonly number[]
  /** The layers' kind (default `'gcn'`). */
  kind?: GnnKind
  /** Hidden width (per head for GAT). Default 8. */
  hidden?: Size
  /** GAT: heads of the first layer (default 2). */
  heads?: Size
  /** GraphSAGE: the aggregator (default `'mean'`). */
  aggregate?: SageAggregator
  /** Adam updates (default 200). */
  steps?: Size
  /** Adam's step size (default 0.01). */
  stepSize?: number
  /** Keep a checkpoint, and yield, every this many steps (default `steps` / 100 rounded, at least 1). */
  every?: Size
  /** The root stream's seed (default `'gnn'`). */
  seed?: string | number
}

/** Curves of a run, one entry per step. */
export interface NodeHistory {
  /** The steps, from 0. */
  step: number[]
  /** Cross-entropy on the labelled nodes. */
  loss: number[]
  /** Accuracy on the labelled nodes. */
  trainAccuracy: number[]
  /** Accuracy on the unlabelled nodes. */
  testAccuracy: number[]
}

/** The network's view of every node at one step. */
export interface NodeCheckpoint {
  /** The step. */
  readonly step: Size
  /** Logits, one row of $C$ per node. */
  readonly logits: number[][]
  /** Hidden features after the first layer and tanh, one row per node. */
  readonly hidden: number[][]
  /** GAT: the first layer's attention per message edge, averaged over heads. */
  readonly attention?: number[]
}

/** A snapshot of `nodeClassificationRun`. */
export interface NodeSnapshot {
  /** Steps taken. */
  readonly step: Size
  /** Steps in the whole run. */
  readonly steps: Size
  /** Whether this is the last snapshot: the run finished or diverged. */
  readonly done: boolean
  /** The network's specification. */
  readonly spec: GnnSpec
  /** The curves, one entry per step so far. */
  readonly history: NodeHistory
  /** The checkpoints so far: step 0, every `every` steps and the last. */
  readonly checkpoints: readonly NodeCheckpoint[]
  /** GAT: the message edges the attention weights belong to. */
  readonly edges?: { source: number[]; destination: number[] }
}

/**
 * The index of a row's largest entry (the first on a tie).
 *
 * @param row The values.
 * @returns The index of the largest.
 */
const argmax = (row: readonly number[]) => row.reduce((b, v, i) => (v > row[b] ? i : b), 0)

/**
 * Train a two-layer GNN on the labelled nodes by full-batch Adam, yielding snapshots (module notes): one at step 0,
 * one every `every` steps and one at the end, or when training diverges (which stops it). Throws `DomainError` when
 * the data have no node classes. Deterministic from the seed.
 *
 * @param options The data, the labelled nodes, the network, the training, the checkpoint interval and the seed.
 * @returns A generator of snapshots.
 *
 * @example Classify the middle node of a path from its two labelled ends
 * const graph = { kind: 'graph', nodes: 3, edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }], directed: false }
 * const data = { graph, x: eye(3), y: tensor([0, 0, 1]) }
 * const options = { data, train: [0, 2], steps: 60, every: 20, stepSize: 0.1, seed: 0 }
 * let last
 * for (const s of nodeClassificationRun(options)) last = s
 * print('loss every 20 steps:', last.history.loss.filter((_, i) => i % 20 === 0))
 * print('accuracy on node 1, the unlabelled one:', last.history.testAccuracy.at(-1))
 * print('final logits:', last.checkpoints.at(-1).logits)
 */
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
