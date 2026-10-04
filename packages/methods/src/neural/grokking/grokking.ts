/**
 * Grokking (Power, Burda, Edwards, Babuschkin and Misra, 2022): a network trained on part of the table of a ∘ b mod p
 * fits its training pairs early, then, under weight decay, generalises to the held-out pairs much later. The model
 * here is the small embedding MLP that groks fastest in a browser: a shared embedding E [p, d] of the residues, the
 * concatenation [E_a, E_b] through one ReLU layer of `width` units, and logits over the p answers. Full-batch AdamW
 * (Loshchilov and Hutter, 2019) shrinks every weight a little per step, which favours the low-norm solution that
 * generalises; on addition that solution represents residues by a few Fourier frequencies (Nanda, Chan, Lieberum, Smith
 * and Steinhardt, 2023; Gromov, 2023, "Grokking modular arithmetic"). A 1-layer transformer groks on the same data
 * but needs far more steps than a page can afford.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { child, stream, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  fromData,
  matmul,
  mul,
  norm,
  square,
  sum,
  take,
  toFlat,
  unwrap,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { relu } from 'aifn-compute/nn/functional'
import { normalInit } from 'aifn-compute/nn/init'
import { childContext, tap, type Context } from 'aifn-compute/nn/layers'
import { methodTraining } from 'aifn-compute/nn/training'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Parameters of the modular MLP: embedding [p, d], hidden layer [2d, width] + [width], output [width, p] + [p]. */
export type ModularMlpParams = { embedding: Tensor; hidden: Tensor; hiddenBias: Tensor; out: Tensor; outBias: Tensor }

/** The architecture of the modular MLP. */
export type ModularMlpConfig = {
  /** The modulus p (residues in and answers out). */
  p: Size
  /** Embedding width d (default 32). */
  embed?: Size
  /** Hidden ReLU units (default 128). */
  width?: Size
}

/** Pairs as two int32 columns a and b [n]. */
export type Pairs = { readonly a: Tensor; readonly b: Tensor }

/** The modular MLP: initialise parameters; logits [n, p] for pairs (hidden units tapped at `hidden`). */
export type ModularMlp = {
  readonly config: Required<ModularMlpConfig>
  init(s: Stream): ModularMlpParams
  apply(params: ModularMlpParams, pairs: Pairs, ctx?: Context): Value
}

/** The embedding MLP for a ∘ b mod p (see the module comment). Initialised at variance 1/fan-in throughout. */
export function ModularMlp(config: ModularMlpConfig): ModularMlp {
  const c = { embed: 32, width: 128, ...config }
  const { p, embed: d, width } = c
  const init = (s: Stream, name: string, shape: number[], fanIn: number) =>
    normalInit(1 / Math.sqrt(fanIn))(child(s, name), shape, { fanIn, fanOut: shape[1] })
  return {
    config: c,
    init: (s) => ({
      embedding: init(s, 'embedding', [p, d], d),
      hidden: init(s, 'hidden', [2 * d, width], 2 * d),
      hiddenBias: zeros([width]),
      out: init(s, 'out', [width, p], width),
      outBias: zeros([p]),
    }),
    apply: (q, pairs, ctx) => {
      const x = concat([take(q.embedding, pairs.a), take(q.embedding, pairs.b)], -1)
      const h = tap(childContext(ctx, 'hidden'), relu(add(matmul(x, q.hidden), q.hiddenBias)))
      return tap(childContext(ctx, 'logits'), add(matmul(h, q.out), q.outBias))
    },
  }
}

/** Pairs from a modular-arithmetic part: features x [n, 2] (a and b) and labels y [n]. */
export function pairsOf(part: { readonly x: Tensor }): Pairs {
  const v = toFlat(part.x)
  const n = v.length / 2
  return {
    a: fromData(
      Int32Array.from({ length: n }, (_, i) => v[2 * i]),
      [n],
    ),
    b: fromData(
      Int32Array.from({ length: n }, (_, i) => v[2 * i + 1]),
      [n],
    ),
  }
}

/** Accuracy and mean cross-entropy of logits [n, p] against labels y [n]. */
function score(logits: Tensor, y: Tensor): { accuracy: number; loss: number } {
  const z = toFlat(logits)
  const labels = toFlat(y)
  const p = logits.shape[1]
  let right = 0
  let loss = 0
  labels.forEach((t, i) => {
    let max = -Infinity
    let best = 0
    for (let k = 0; k < p; k++)
      if (z[i * p + k] > max) {
        max = z[i * p + k]
        best = k
      }
    let s = 0
    for (let k = 0; k < p; k++) s += Math.exp(z[i * p + k] - max)
    loss += max + Math.log(s) - z[i * p + t]
    if (best === t) right++
  })
  return { accuracy: right / labels.length, loss: loss / labels.length }
}

/** The curves of a grokking run, one entry per recorded step. */
export type GrokkingCurves = {
  readonly steps: number[]
  readonly trainAccuracy: number[]
  readonly testAccuracy: number[]
  readonly trainLoss: number[]
  readonly testLoss: number[]
  /** The global parameter norm ‖θ‖₂. */
  readonly weightNorm: number[]
}

/** A snapshot of `grokkingRun`: the curves so far and the parameters at every checkpoint. */
export type GrokkingSnapshot = {
  readonly step: Size
  readonly steps: Size
  readonly config: Required<ModularMlpConfig>
  readonly curves: GrokkingCurves
  readonly checkpoints: readonly { readonly step: Size; readonly params: ModularMlpParams }[]
}

/** Options of `grokkingRun`. */
export type GrokkingRunOptions = Omit<ModularMlpConfig, 'p'> & {
  /** Full-batch AdamW steps or L-BFGS iterations (default 1500). */
  steps?: Size
  /**
   * `adamw` (default) or `lbfgs`: full-batch L-BFGS on the cross-entropy plus the coupled penalty (λ/2)‖θ‖², the
   * weight decay's fixed point, which it reaches directly rather than by the slow drift that grokking rides.
   */
  method?: 'adamw' | 'lbfgs'
  /** L-BFGS's memory m (default 10). */
  memory?: Size
  /** AdamW's step size (default 0.01) and decoupled weight decay λ (default 2). */
  stepSize?: number
  weightDecay?: number
  /** Record the curves every this many steps (default 10) and keep a checkpoint every `checkpointEvery` (default 50). */
  recordEvery?: Size
  checkpointEvery?: Size
  /** The root stream's seed (default 'grokking'). */
  seed?: string | number
}

/**
 * Train the modular MLP by full-batch AdamW on the training pairs of a modular-arithmetic table (`aifn-methods/data`'s
 * `modularArithmetic`), yielding a snapshot at every checkpoint (step 0 first) with the training and test accuracy,
 * loss and weight norm so far.
 */
export function* grokkingRun(
  data: {
    readonly p: Size
    readonly train: { readonly x: Tensor; readonly y?: Tensor }
    readonly test: { readonly x: Tensor; readonly y?: Tensor }
  },
  options: GrokkingRunOptions = {},
): Generator<GrokkingSnapshot> {
  const {
    steps = 1500,
    stepSize = 0.01,
    weightDecay = 2,
    recordEvery = 10,
    checkpointEvery = 50,
    seed = 'grokking',
    method = 'adamw',
    memory,
    ...arch
  } = options
  if (!data.train.y || !data.test.y) throw new DomainError('grokkingRun', 'grokkingRun: both parts need labels')
  const model = ModularMlp({ ...arch, p: data.p })
  const train = { ...pairsOf(data.train), y: data.train.y }
  const test = { ...pairsOf(data.test), y: data.test.y }
  const root = stream(seed)
  const crossEntropy = (q: ModularMlpParams, batch: { a: Tensor; b: Tensor; y: Tensor }) =>
    softmaxCrossEntropy(model.apply(q, batch), batch.y)
  const penalised = (q: ModularMlpParams, batch: { a: Tensor; b: Tensor; y: Tensor }) => {
    let total: Value = 0
    for (const w of Object.values(q)) total = add(total, sum(square(w)))
    return add(crossEntropy(q, batch), mul(weightDecay / 2, total))
  }
  const alg =
    method === 'lbfgs'
      ? methodTraining(penalised, train, { method: 'lbfgs', memory })
      : // β₂ = 0.98, as in Nanda et al.'s runs: a shorter memory of the squared gradient than Adam's default 0.999.
        methodTraining(crossEntropy, train, { method: 'adam', stepSize, weightDecay, decoupled: true, beta2: 0.98 })
  let state = alg.init({ params: model.init(child(root, 'init')) }, child(root, 'init'))
  const curves: GrokkingCurves = {
    steps: [],
    trainAccuracy: [],
    testAccuracy: [],
    trainLoss: [],
    testLoss: [],
    weightNorm: [],
  }
  const checkpoints: { step: Size; params: ModularMlpParams }[] = []
  const record = (t: Size) => {
    const tr = score(unwrap(model.apply(state.params, train)) as Tensor, train.y)
    const te = score(unwrap(model.apply(state.params, test)) as Tensor, test.y)
    curves.steps.push(t)
    curves.trainAccuracy.push(tr.accuracy)
    curves.testAccuracy.push(te.accuracy)
    curves.trainLoss.push(tr.loss)
    curves.testLoss.push(te.loss)
    curves.weightNorm.push(Math.sqrt(Object.values(state.params).reduce((a, w) => a + norm(w) ** 2, 0)))
  }
  // A run that stops early (converged L-BFGS) reports its last step as the total, so a page reads it as finished.
  let total = steps
  const snapshot = (t: Size): GrokkingSnapshot => ({
    step: t,
    steps: total,
    config: model.config,
    curves: {
      steps: [...curves.steps],
      trainAccuracy: [...curves.trainAccuracy],
      testAccuracy: [...curves.testAccuracy],
      trainLoss: [...curves.trainLoss],
      testLoss: [...curves.testLoss],
      weightNorm: [...curves.weightNorm],
    },
    checkpoints: [...checkpoints],
  })
  record(0)
  checkpoints.push({ step: 0, params: state.params })
  yield snapshot(0)
  for (let t = 0; t < steps; t++) {
    state = alg.step(state, { t, stream: child(root, 'step', t) })
    const done = t + 1 === steps || state.stopped
    if (done) total = t + 1
    if ((t + 1) % recordEvery === 0 || done) record(t + 1)
    if ((t + 1) % checkpointEvery === 0 || done) {
      checkpoints.push({ step: t + 1, params: state.params })
      yield snapshot(t + 1)
    }
    if (done) return
  }
}
