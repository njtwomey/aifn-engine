/**
 * A tiny GPT trained on a prompt–answer task (copying, reversing, sorting, bracket completion, addition, associative
 * recall; `aifn-methods/data`'s `sequenceTasks`): each example is one row `^ prompt = answer .`, the model reads it
 * causally, and the loss is the next-token cross-entropy at the answer positions only, so the model is never asked to
 * predict a random prompt. Accuracy is teacher-forced: an example is exact when the most probable next token is right
 * at every answer position, which is when greedy decoding from the prompt reproduces the answer.
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
 * Examples of a prompt–answer task as padded rows: tokens [n, L], next-token targets [n, L] and weights [n, L], 1 at
 * the positions whose target is an answer token (the shape of `sequenceTasks`' examples).
 */
export type TaskExamples = { readonly tokens: Tensor; readonly targets: Tensor; readonly weights: Tensor }

/** The weighted mean next-token cross-entropy (nats per answer token) of logits [N, T, V]. */
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
  /** Decoupled weight decay λ (AdamW; default 0). */
  weightDecay?: number
  /** Rescale gradients above this global norm (default 1). */
  clipNorm?: number
}

/** Minibatch AdamW on `taskLoss`, as a traceable `trainingLoop`. */
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

/** Teacher-forced accuracy: the share of answer tokens predicted right, and of examples right at every one. */
export type TaskAccuracy = { readonly token: number; readonly exact: number; readonly loss: number }

/** The accuracy and loss of a GPT on the first `limit` examples (default all), evaluated in chunks of 64 rows. */
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
  readonly step: Size
  readonly params: GptParams
  readonly train: TaskAccuracy
  readonly test: TaskAccuracy
}

/** A snapshot of `taskTrainingRun`: the steps so far, every step's minibatch loss, and every checkpoint. */
export type TaskSnapshot = {
  readonly step: Size
  readonly steps: Size
  readonly losses: readonly number[]
  readonly config: Required<GptConfig>
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
 * worker can stream the run to a page that plays the checkpoints. The context is the row length of the data.
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
