/**
 * A tiny GPT trained on a prompt–answer task (copying, reversing, sorting, bracket completion, addition, associative
 * recall; `aifn-methods/data`'s `sequenceTasks`): each example is one row `^ prompt = answer .`, the model reads it
 * causally, and the loss is the next-token cross-entropy at the answer positions only, so the model is never asked to
 * predict a random prompt. Accuracy is teacher-forced: an example is exact when the most probable next token is right
 * at every answer position, which is when greedy decoding from the prompt reproduces the answer.
 *
 * With $w_i$ the weight of position $i$ (1 at an answer token, else 0) and $\ell_i$ its cross-entropy, the loss is
 * $\sum_i w_i \ell_i / \max(1, \sum_i w_i)$, in nats per answer token.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  div,
  mul,
  reshape,
  shapeOfValue,
  sum,
  take,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { trainingLoop, type TrainingState } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { Gpt, type GptConfig, type GptParams } from './gpt'

/**
 * Examples of a prompt–answer task as padded rows: `tokens` ($n \times L$ ids), next-token `targets` ($n \times L$,
 * the tokens shifted left by one) and `weights` ($n \times L$), 1 at the positions whose target is an answer token
 * (the shape of `sequenceTasks`' examples).
 */
export type TaskExamples = { readonly tokens: Tensor; readonly targets: Tensor; readonly weights: Tensor }

/**
 * The weighted mean next-token cross-entropy (nats per answer token): $\sum_i w_i \ell_i / \max(1, \sum_i w_i)$
 * over every position $i$ (differentiable in the logits).
 *
 * @param logits The logits, $N \times T \times V$.
 * @param targets The target ids, $N \times T$.
 * @param weights The weight $w_i$ of each position, $N \times T$: 1 at answer positions, 0 elsewhere.
 * @returns The weighted mean cross-entropy, a scalar.
 *
 * @example Only the weighted position counts
 * const logits = fromData(Float64Array.from([0, 0, 0, 0, 5, 0, 0, 0, 5]), [1, 3, 3])
 * const targets = fromData(Int32Array.from([0, 1, 1]), [1, 3])
 * print('the answer position only:', taskLoss(logits, targets, tensor([[0, 1, 0]])))
 * print('every position:', taskLoss(logits, targets, tensor([[1, 1, 1]])))
 */
export function taskLoss(logits: Value, targets: Tensor, weights: Tensor): Value {
  const V = shapeOfValue(logits).at(-1)!
  const each = softmaxCrossEntropy(reshape(logits, [-1, V]), reshape(targets, [-1]) as Tensor, { reduction: 'none' })
  const w = reshape(weights, [-1]) as Tensor
  return div(
    sum(mul(each, w)),
    Math.max(
      1,
      toFlat(w).reduce((a, b) => a + b, 0),
    ),
  )
}

/** Options of `taskTraining`. */
export type TaskTrainingOptions = {
  /** Rows per step (default 32). */
  batchSize?: Size
  /** Adam's step size (default 0.003). */
  stepSize?: number
  /** Decoupled weight decay $\lambda$ (AdamW; default 0, plain Adam). */
  weightDecay?: number
  /** Rescale gradients above this global norm (default 1). */
  clipNorm?: number
}

/**
 * Minibatch AdamW on `taskLoss` (Adam when the weight decay is 0), as a traceable `trainingLoop`, with gradients
 * clipped to a global norm. Each epoch shuffles the rows; the batch size is capped at the number of rows.
 *
 * @param model The GPT; its context must be at least the row length $L$.
 * @param examples The training rows.
 * @param options The batch size, step size, weight decay and clipping norm.
 * @returns The algorithm, to run with `run` or `trace` from `{ params }`.
 *
 * @example Learn to copy one letter
 * const one = (rows) => fromData(Int32Array.from(rows.flat()), [rows.length, rows[0].length])
 * const ex = {
 *   tokens: one([[0, 3, 1, 3], [0, 4, 1, 4]]),
 *   targets: one([[3, 1, 3, 2], [4, 1, 4, 2]]),
 *   weights: tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
 * }
 * const model = Gpt({ vocabulary: 5, context: 4, width: 8, layers: 1, heads: 2 })
 * const alg = taskTraining(model, ex, { stepSize: 0.05 })
 * const tr = trace(alg, { params: model.init(stream(0)) }, 30, { every: 10, record: { loss: (s) => s.loss } })
 * print('step:', tr.index)
 * print('loss:', tr.series.loss)
 */
export function taskTraining(
  model: Gpt,
  examples: TaskExamples,
  options: TaskTrainingOptions = {},
): Algorithm<{ params: GptParams }, TrainingState<GptParams>> {
  const { batchSize = 32, stepSize = 0.003, weightDecay = 0, clipNorm = 1 } = options
  return trainingLoop<GptParams, { x: Tensor; y: Tensor; w: Tensor }>({
    loss: (p, b, ctx) => taskLoss(model.apply(p, b.x, ctx), b.y, b.w),
    data: { x: examples.tokens, y: examples.targets, w: examples.weights },
    batchSize: Math.min(batchSize, examples.tokens.shape[0]),
    optimizer: adamRule({ stepSize, weightDecay, decoupled: weightDecay > 0 }) as never,
    clipNorm,
  })
}

/**
 * Teacher-forced accuracy: `token`, the share of answer tokens predicted right; `exact`, the share of examples right
 * at every one; and `loss`, the mean cross-entropy per answer token. Each is 0 when there is nothing to score.
 */
export type TaskAccuracy = { readonly token: number; readonly exact: number; readonly loss: number }

/**
 * The teacher-forced accuracy and loss of a GPT on the first `limit` examples (default all), evaluated in chunks of
 * 64 rows. A prediction is the most probable next token; an example with no answer positions counts as exact.
 *
 * @param model The GPT.
 * @param params Its parameters.
 * @param examples The rows to score.
 * @param limit How many rows to score, from the first (default all of them).
 * @returns The token and exact accuracies and the loss.
 *
 * @example A copy task before and after training
 * // Rows "^ x = x ." over the vocabulary ^ = . a b, with weight on the answer and the full stop.
 * const one = (rows) => fromData(Int32Array.from(rows.flat()), [rows.length, rows[0].length])
 * const ex = {
 *   tokens: one([[0, 3, 1, 3], [0, 4, 1, 4]]),
 *   targets: one([[3, 1, 3, 2], [4, 1, 4, 2]]),
 *   weights: tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
 * }
 * const model = Gpt({ vocabulary: 5, context: 4, width: 8, layers: 1, heads: 2 })
 * const params = model.init(stream(0))
 * print('untrained:', taskAccuracy(model, params, ex))
 * const trained = run(taskTraining(model, ex, { stepSize: 0.05 }), { params }, 30, { stream: stream(1) }).params
 * print('after 30 steps:', taskAccuracy(model, trained, ex))
 */
export function taskAccuracy(model: Gpt, params: GptParams, examples: TaskExamples, limit?: Size): TaskAccuracy {
  const [n, L] = examples.tokens.shape
  const m = Math.min(n, limit ?? n)
  const V = model.config.vocabulary
  let right = 0
  let total = 0
  let exact = 0
  let loss = 0
  for (let start = 0; start < m; start += 64) {
    const rows = Array.from({ length: Math.min(64, m - start) }, (_, i) => start + i)
    const x = unwrap(take(examples.tokens, rows)) as Tensor
    const logits = toFlat(unwrap(model.apply(params, x)) as Tensor)
    const y = toFlat(unwrap(take(examples.targets, rows)) as Tensor)
    const w = toFlat(unwrap(take(examples.weights, rows)) as Tensor)
    rows.forEach((_, r) => {
      let all = true
      for (let t = 0; t < L; t++) {
        const k = r * L + t
        if (w[k] === 0) continue
        let best = 0
        let max = -Infinity
        let z = 0
        for (let v = 0; v < V; v++) {
          const l = logits[k * V + v]
          if (l > max) {
            max = l
            best = v
          }
        }
        for (let v = 0; v < V; v++) z += Math.exp(logits[k * V + v] - max)
        loss += max + Math.log(z) - logits[k * V + y[k]]
        total++
        if (best === y[k]) right++
        else all = false
      }
      if (all) exact++
    })
  }
  return { token: total ? right / total : 0, exact: m ? exact / m : 0, loss: total ? loss / total : 0 }
}

/** A checkpoint of `taskTrainingRun`: the parameters after `step` updates and their accuracy on both splits. */
export type TaskCheckpoint = {
  /** Updates taken. */
  readonly step: Size
  /** The parameters after `step` updates. */
  readonly params: GptParams
  /** Accuracy and loss on the first `evaluate` training rows. */
  readonly train: TaskAccuracy
  /** Accuracy and loss on the first `evaluate` test rows. */
  readonly test: TaskAccuracy
}

/** A snapshot of `taskTrainingRun`: the steps so far, every step's minibatch loss, and every checkpoint. */
export type TaskSnapshot = {
  /** Steps taken. */
  readonly step: Size
  /** Steps in the whole run. */
  readonly steps: Size
  /** The minibatch loss at the start and after each step so far: `step` $+ 1$ values. */
  readonly losses: readonly number[]
  /** The model's full configuration. */
  readonly config: Required<GptConfig>
  /** Every checkpoint so far, the last at `step`. */
  readonly checkpoints: readonly TaskCheckpoint[]
}

/** Options of `taskTrainingRun`. */
export type TaskTrainingRunOptions = Omit<GptConfig, 'vocabulary' | 'context'> &
  TaskTrainingOptions & {
    /** Adam steps (default 400). */
    steps?: Size
    /** Checkpoint every this many steps (default 25), and at step 0 and the end. */
    every?: Size
    /** Examples of each split scored at a checkpoint (default 256). */
    evaluate?: Size
    /** The root stream's seed (default 'task-gpt'). */
    seed?: string | number
  }

/**
 * Train a tiny GPT on a prompt–answer task, yielding a snapshot at every checkpoint (step 0 first): a generator, so a
 * worker can stream the run to a page that plays the checkpoints. The context is the row length of the data, and the
 * run is deterministic from its seed.
 *
 * @param data The task: its vocabulary (whose length sets the model's), and its training and test rows.
 * @param options The architecture (without vocabulary and context), the training options, the number of steps, the
 *   checkpoint interval, the rows scored per checkpoint and the seed.
 * @returns A generator of snapshots, one per checkpoint.
 *
 * @example A copy task, checkpointed every ten steps
 * const one = (rows) => fromData(Int32Array.from(rows.flat()), [rows.length, rows[0].length])
 * const split = {
 *   tokens: one([[0, 3, 1, 3], [0, 4, 1, 4]]),
 *   targets: one([[3, 1, 3, 2], [4, 1, 4, 2]]),
 *   weights: tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
 * }
 * const data = { vocabulary: ['^', '=', '.', 'a', 'b'], train: split, test: split }
 * const options = { width: 8, layers: 1, heads: 2, steps: 30, every: 10, stepSize: 0.05, seed: 0 }
 * for (const s of taskTrainingRun(data, options)) {
 *   const c = s.checkpoints.at(-1)
 *   print('step', s.step, ' loss', s.losses.at(-1), ' exact on train', c.train.exact)
 * }
 */
export function* taskTrainingRun(
  data: { readonly vocabulary: readonly string[]; readonly train: TaskExamples; readonly test: TaskExamples },
  options: TaskTrainingRunOptions = {},
): Generator<TaskSnapshot> {
  const {
    steps = 400,
    every = 25,
    evaluate = 256,
    seed = 'task-gpt',
    batchSize,
    stepSize,
    weightDecay,
    clipNorm,
    ...arch
  } = options
  const model = Gpt({ ...arch, vocabulary: data.vocabulary.length, context: data.train.tokens.shape[1] })
  const root = stream(seed)
  const alg = taskTraining(model, data.train, { batchSize, stepSize, weightDecay, clipNorm })
  let state = alg.init({ params: model.init(child(root, 'init')) }, child(root, 'init'))
  const losses = [state.loss]
  const checkpoints: TaskCheckpoint[] = []
  const checkpoint = (t: Size) =>
    checkpoints.push({
      step: t,
      params: state.params,
      train: taskAccuracy(model, state.params, data.train, evaluate),
      test: taskAccuracy(model, state.params, data.test, evaluate),
    })
  checkpoint(0)
  yield { step: 0, steps, losses: [...losses], config: model.config, checkpoints: [...checkpoints] }
  for (let t = 0; t < steps; t++) {
    state = alg.step(state, { t, stream: child(root, 'step', t) })
    losses.push(state.loss)
    if ((t + 1) % every === 0 || t + 1 === steps) {
      checkpoint(t + 1)
      yield { step: t + 1, steps, losses: [...losses], config: model.config, checkpoints: [...checkpoints] }
    }
  }
}
