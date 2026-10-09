/**
 * Training as a traceable algorithm: minibatch stochastic gradient descent (Robbins and Monro, 1951, "A stochastic
 * approximation method", Ann. Math. Statist. 22) on a loss of the parameters, with any pytree update rule of
 * `aifn-compute/optim/first-order` (`adamRule`, `sgdRule`, ...; optax's gradient transformations, Babuschkin et al.,
 * 2020).
 *
 * The state is plain data: each epoch's shuffled order is drawn from the stream of the step that starts the epoch, and
 * dropout masks from the stream of the step that evaluates the loss, so a step is a pure function of its state and
 * context, and `seek`, `extend` and replays agree exactly. Non-trainable layer state (batch norm's running statistics,
 * `ctx.buffers`) is part of the state too: each loss evaluation reads the previous buffers and its writes become the
 * next ones.
 */

import type { Scalar, Size, Status, StepContext } from 'aifn-compute/foundation/contracts'
import { valueAndGrad, type ValueAndGrad } from 'aifn-compute/foundation/autodiff'
import { treeLeaves, type Params } from 'aifn-compute/foundation/pytree'
import { child, permutation, type Stream } from 'aifn-compute/foundation/random'
import { fromData, norm, take, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { Buffers, Context } from 'aifn-compute/nn/layers'
import {
  adamRule,
  applyUpdates,
  chainRules,
  clipByGlobalNorm,
  globalNorm,
  type UpdateRule,
} from 'aifn-compute/optim/first-order'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A dataset: named tensors whose first axis indexes the examples (all the same length), e.g. `{ x, y }`. */
export type Batch = Record<string, Tensor>

/** Options of `trainingLoop`. */
export type TrainingOptions<P extends Params, B extends Batch> = {
  /**
   * The loss of parameters on a minibatch, a number or rank-0 value. `ctx` carries `train: true`, a stream (for
   * dropout) and the layer buffers with a sink for their updates (for batch norm); pass it to the model's `apply`.
   */
  loss: (params: P, batch: B, ctx: Context) => Value
  /** The training set. */
  data: B
  /** Examples per step (default: the whole set, i.e. full-batch gradient descent). */
  batchSize?: Size
  /** The update rule, a pytree rule of `aifn-compute/optim/first-order`. Default `adamRule({ stepSize: 0.01 })`. */
  optimizer?: UpdateRule<unknown>
  /**
   * Rescale the gradient when its global norm exceeds this (Pascanu, Mikolov and Bengio, 2013), by chaining
   * `clipByGlobalNorm` before the optimiser. The unclipped norm is reported. Default: no clipping.
   */
  clipNorm?: Scalar
  /** Flag divergence when the loss's absolute value exceeds this or the loss is not finite. Default 1e12. */
  divergeAbove?: Scalar
}

/** The state of `trainingLoop` after $t$ updates. */
export interface TrainingState<P extends Params> extends Status {
  /** Updates applied so far. */
  readonly t: Size
  /** The parameters after $t$ updates. */
  readonly params: P
  /** The update rule's state (its step count and running moments). */
  readonly optimizer: unknown
  /** The layers' non-trainable state after evaluating this step's loss (running statistics); `{}` when none. */
  readonly buffers: Buffers
  /** The loss of `params` on this step's minibatch (the one the next update uses). */
  readonly loss: Scalar
  /** Its gradient with respect to the parameters. */
  readonly grads: P
  /** The global gradient norm $\lVert \nabla \rVert_2$ over every parameter (before clipping). */
  readonly gradNorm: Scalar
  /** The gradient norm of each parameter leaf, by path (e.g. `[0].weight`). */
  readonly gradNorms: Readonly<Record<string, Scalar>>
  /** The global parameter norm $\lVert \thetavec \rVert_2$. */
  readonly paramNorm: Scalar
  /** Indices of this step's minibatch in the dataset (null for full-batch training). */
  readonly batch: Tensor | null
  /** The epoch this step's minibatch belongs to (the step itself in full-batch training). */
  readonly epoch: Size
  /** The current epoch's shuffled order of the examples (null for full-batch training). */
  readonly order: Tensor | null
  /** Whether the loss is not finite or its absolute value exceeds `divergeAbove`; a run stops here. */
  readonly diverged: boolean
}

/**
 * The data's number of examples, checking every field agrees. Throws `ShapeError` when two fields differ in length and
 * `DomainError` when there are no examples.
 *
 * @param data The dataset, whose fields' first axes index the examples.
 * @returns The common length of the first axes.
 */
function examplesOf(data: Batch): Size {
  let n = -1
  for (const [name, t] of Object.entries(data)) {
    const m = t.shape[0]
    if (n >= 0 && m !== n)
      throw new ShapeError('trainingLoop', `trainingLoop: data field '${name}' has ${m} rows, expected ${n}`)
    n = m
  }
  if (n <= 0) throw new DomainError('trainingLoop', 'trainingLoop: the data has no examples')
  return n
}

/**
 * The Euclidean norm of a parameter leaf.
 *
 * @param v The leaf: a tensor, or a number (whose norm is its absolute value).
 * @returns The norm, $\lVert v \rVert_2$.
 */
const leafNorm = (v: Tensor | number) => (typeof v === 'number' ? Math.abs(v) : norm(v))

/**
 * Minibatch training of parameters on `loss`. `init` takes `{ params }` (e.g. `model.init(stream)`), and optionally the
 * initial `buffers` (default `{}`: each stateful layer starts from its initial values); step $t$ applies one update
 * from the gradient on minibatch $t$ and evaluates the loss on minibatch $t + 1$, so the initial state already holds
 * the loss and gradient of minibatch 0. Epochs are shuffled and cut into consecutive batches, so every example is used
 * once per epoch (a short last batch is dropped). The shuffle of epoch $e$ is drawn from `child(s, 'epoch')` of the
 * stream of the step that starts it (the init stream for epoch 0); the dropout masks of a loss evaluation from
 * `child(s, 'dropout')`. Throws `ShapeError` when the data's fields differ in length.
 *
 * Record what a figure needs with the trace's recorders, e.g. `{ loss: (s) => s.loss, gradNorm: (s) => s.gradNorm }`.
 *
 * @param options The loss, the data, the batch size, the update rule, gradient clipping and the divergence threshold.
 * @returns The algorithm, to run with `run` or `trace` from `{ params }`.
 *
 * @example Fit $y = 2x + 1$ by full-batch Adam: the loss falls
 * const data = { x: tensor([[0], [1], [2], [3]]), y: tensor([[1], [3], [5], [7]]) }
 * const loss = (p, b) => mean(square(sub(add(matmul(b.x, p.w), p.b), b.y)))
 * const start = { params: { w: zeros([1, 1]), b: zeros([1]) } }
 * const tr = trace(trainingLoop({ loss, data }), start, 200, { every: 50, record: { loss: (s) => s.loss } })
 * print('steps:', tr.index)
 * print('loss:', tr.series.loss)
 *
 * @example Minibatches of two: each epoch is a fresh shuffle cut into two batches
 * const data = { x: tensor([[0], [1], [2], [3]]), y: tensor([[1], [3], [5], [7]]) }
 * const loss = (p, b) => mean(square(sub(add(matmul(b.x, p.w), p.b), b.y)))
 * const alg = trainingLoop({ loss, data, batchSize: 2 })
 * const start = { params: { w: zeros([1, 1]), b: zeros([1]) } }
 * for (const t of [0, 1, 2, 3]) {
 *   const s = run(alg, start, t)
 *   print('step', t, ' epoch', s.epoch, ' batch', s.batch, ' loss', s.loss)
 * }
 */
export function trainingLoop<P extends Params, B extends Batch>(
  options: TrainingOptions<P, B>,
): Algorithm<{ params: P; buffers?: Buffers }, TrainingState<P>> {
  const { loss, data, clipNorm, divergeAbove = 1e12 } = options
  const rule = options.optimizer ?? (adamRule({ stepSize: 0.01 }) as UpdateRule<unknown>)
  const optimizer: UpdateRule<unknown> =
    clipNorm === undefined ? rule : (chainRules(clipByGlobalNorm(clipNorm), rule) as UpdateRule<unknown>)
  const n = examplesOf(data)
  const size = Math.min(options.batchSize ?? n, n)
  const perEpoch = Math.floor(n / size)
  const lossAndGrad: (params: P, batch: B, ctx: Context) => ValueAndGrad<Value, unknown> = valueAndGrad(
    (params: P, batch: B, ctx: Context) => loss(params, batch, ctx),
    {},
  )

  /**
   * Evaluate the loss and gradient at step $t$ on its minibatch, drawing a new epoch order from `s` when $t$ starts
   * one.
   */
  const evaluate = (params: P, buffers: Buffers, t: Size, previous: Tensor | null, s: Stream) => {
    let batch = data
    let indices: Tensor | null = null
    let order: Tensor | null = null
    const epoch = size >= n ? t : Math.floor(t / perEpoch)
    if (size < n) {
      const k = t % perEpoch
      order = k === 0 || previous === null ? permutation(child(s, 'epoch'), n) : previous
      const ids = toFlat(order).slice(k * size, (k + 1) * size)
      indices = fromData(Int32Array.from(ids), [size])
      batch = Object.fromEntries(Object.entries(data).map(([key, v]) => [key, unwrap(take(v, ids)) as Tensor])) as B
    }
    const bufferUpdates: Record<string, Params> = {}
    const ctx: Context = { train: true, stream: child(s, 'dropout'), buffers, bufferUpdates }
    const { value, grad } = lossAndGrad(params, batch, ctx)
    const grads = grad as P
    const gradNorms: Record<string, number> = {}
    for (const { path, value: g } of treeLeaves(grads)) gradNorms[path] = leafNorm(g)
    const raw = unwrap(value)
    const lossValue = typeof raw === 'number' ? raw : toFlat(raw)[0]
    return {
      loss: lossValue,
      buffers: Object.keys(bufferUpdates).length ? { ...buffers, ...bufferUpdates } : buffers,
      grads,
      gradNorm: globalNorm(grads),
      gradNorms,
      paramNorm: globalNorm(params),
      batch: indices,
      epoch,
      order,
      diverged: !Number.isFinite(lossValue) || Math.abs(lossValue) > divergeAbove,
    }
  }

  return {
    name: `training-${rule.name}`,
    init: ({ params, buffers = {} }, s) => ({
      t: 0,
      params,
      optimizer: optimizer.init(params),
      ...evaluate(params, buffers, 0, null, s),
    }),
    step: (state, ctx: StepContext) => {
      const { updates, state: next } = optimizer.update(state.grads, state.optimizer, state.params)
      const params = applyUpdates(state.params, updates)
      const t = state.t + 1
      return { t, params, optimizer: next, ...evaluate(params, state.buffers, t, state.order, ctx.stream) }
    },
  }
}
