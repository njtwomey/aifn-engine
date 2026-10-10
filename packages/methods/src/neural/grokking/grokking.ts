/**
 * Grokking (Power, Burda, Edwards, Babuschkin and Misra, 2022): a network trained on part of the table of
 * $a \circ b \bmod p$ fits its training pairs early, then, under weight decay, generalises to the held-out pairs much
 * later. The model here is the small embedding MLP that groks fastest in a browser: a shared embedding $\Emat$
 * ($p \times d$) of the residues, the concatenation $[\evec_a, \evec_b]$ of rows $a$ and $b$ through one ReLU layer
 * of `width` units, and logits over the $p$ answers,
 * $\zvec = \operatorname{relu}([\evec_a, \evec_b] \Wmat_1 + \bvec_1) \Wmat_2 + \bvec_2$. Full-batch AdamW
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

/**
 * Parameters of the modular MLP: the `embedding` $\Emat$ ($p \times d$), the `hidden` layer $\Wmat_1$
 * ($2d \times$ `width`) and its `hiddenBias` $\bvec_1$ (`width`), and the `out` layer $\Wmat_2$ (`width` $\times p$)
 * and its `outBias` $\bvec_2$ ($p$).
 */
export type ModularMlpParams = { embedding: Tensor; hidden: Tensor; hiddenBias: Tensor; out: Tensor; outBias: Tensor }

/** The architecture of the modular MLP. */
export type ModularMlpConfig = {
  /** The modulus $p$ (residues in and answers out). */
  p: Size
  /** Embedding width $d$ (default 32). */
  embed?: Size
  /** Hidden ReLU units (default 128). */
  width?: Size
}

/** Pairs as two int32 columns `a` and `b`, $n$ values each: the operands of each row. */
export type Pairs = { readonly a: Tensor; readonly b: Tensor }

/** The modular MLP: initialise parameters; logits for pairs. */
export type ModularMlp = {
  /** The architecture, with the defaults filled in. */
  readonly config: Required<ModularMlpConfig>
  /** Fresh parameters from stream `s`. */
  init(s: Stream): ModularMlpParams
  /**
   * The logits, $n \times p$, of $n$ pairs. With a tapping context it records the hidden units at `hidden` and the
   * logits at `logits`.
   */
  apply(params: ModularMlpParams, pairs: Pairs, ctx?: Context): Value
}

/**
 * The embedding MLP for $a \circ b \bmod p$ (see the file comment). Weights are drawn from $\Gauss(0, 1/m)$ with $m$
 * the fan-in ($d$ for the embedding, $2d$ for the hidden layer, `width` for the output); biases start at zero.
 *
 * @param config The modulus $p$, the embedding width and the hidden width.
 * @returns The model: its full configuration, `init` and `apply`.
 *
 * @example Logits over the five residues for two pairs
 * const model = ModularMlp({ p: 5, embed: 4, width: 8 })
 * const pairs = pairsOf({ x: tensor([[1, 2], [3, 4]]) })
 * print('logits:', model.apply(model.init(stream(0)), pairs))
 */
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

/**
 * Pairs from a modular-arithmetic part: its features `x` ($n \times 2$, the operands $a$ and $b$ of each row) as two
 * int32 columns. The labels are not read.
 *
 * @param part A part of the table, with `x` ($n \times 2$).
 * @returns The columns `a` and `b`.
 *
 * @example Four rows of the table of addition mod 5
 * const rows = [[0, 1], [2, 2], [3, 4], [4, 4]]
 * const pairs = pairsOf({ x: tensor(rows) })
 * print('a:', pairs.a, ' b:', pairs.b)
 * print('(a + b) mod 5:', rows.map(([a, b]) => (a + b) % 5))
 */
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

/**
 * Accuracy and mean cross-entropy of logits against labels; a prediction is the largest logit (the first on a tie).
 *
 * @param logits The logits, $n \times p$.
 * @param y The labels, $n$ residues.
 * @returns The share predicted right, and the mean cross-entropy in nats.
 */
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
  /** The recorded steps, from 0. */
  readonly steps: number[]
  /** Accuracy on the training pairs. */
  readonly trainAccuracy: number[]
  /** Accuracy on the held-out pairs. */
  readonly testAccuracy: number[]
  /** Mean cross-entropy on the training pairs (without the weight penalty). */
  readonly trainLoss: number[]
  /** Mean cross-entropy on the held-out pairs. */
  readonly testLoss: number[]
  /** The global parameter norm $\norm{\thetavec}_2$, biases and embedding included. */
  readonly weightNorm: number[]
}

/** A snapshot of `grokkingRun`: the curves so far and the parameters at every checkpoint. */
export type GrokkingSnapshot = {
  /** Steps taken. */
  readonly step: Size
  /** Steps in the whole run, or the step it stopped at once L-BFGS has converged. */
  readonly steps: Size
  /** The model's architecture. */
  readonly config: Required<ModularMlpConfig>
  /** The curves so far. */
  readonly curves: GrokkingCurves
  /** The parameters at step 0, every `checkpointEvery` steps and the last. */
  readonly checkpoints: readonly { readonly step: Size; readonly params: ModularMlpParams }[]
}

/** Options of `grokkingRun`. */
export type GrokkingRunOptions = Omit<ModularMlpConfig, 'p'> & {
  /** Full-batch AdamW steps or L-BFGS iterations (default 1500). */
  steps?: Size
  /**
   * `adamw` (default) or `lbfgs`: full-batch L-BFGS on the cross-entropy plus the coupled penalty
   * $(\lambda/2)\norm{\thetavec}^2$, the weight decay's fixed point, which it reaches directly rather than by the slow
   * drift that grokking rides.
   */
  method?: 'adamw' | 'lbfgs'
  /** L-BFGS's memory $m$ (default 10). */
  memory?: Size
  /** AdamW's step size (default 0.01). */
  stepSize?: number
  /** The weight decay $\lambda$ (default 2): AdamW's decoupled decay, or L-BFGS's penalty weight. */
  weightDecay?: number
  /** Record the curves every this many steps (default 10), and at the last step. */
  recordEvery?: Size
  /** Keep a checkpoint, and yield a snapshot, every this many steps (default 50), and at the last step. */
  checkpointEvery?: Size
  /** The root stream's seed (default 'grokking'). */
  seed?: string | number
}

/**
 * Train the modular MLP by full-batch AdamW on the training pairs of a modular-arithmetic table (`aifn-methods/data`'s
 * `modularArithmetic`), yielding a snapshot at every checkpoint (step 0 first) with the training and test accuracy,
 * loss and weight norm so far. AdamW uses $\beta_2 = 0.98$, as in Nanda et al.'s runs; with `method: 'lbfgs'` the run
 * stops early once L-BFGS converges. Throws `DomainError` when either part has no labels. Deterministic from the seed.
 *
 * @param data The modulus $p$, and the training and held-out parts, each with features `x` ($n \times 2$) and labels
 *   `y` (required).
 * @param options The architecture, the method, the steps, the optimiser's settings, the recording and checkpoint
 *   intervals and the seed.
 * @returns A generator of snapshots, one per checkpoint.
 *
 * @example Addition mod 7: the training pairs are fitted first
 * const p = 7
 * const rows = []
 * for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) rows.push([a, b, (a + b) % p])
 * const order = Array.from(toFlat(permutation(stream(0), rows.length)))
 * const part = (ids) => ({
 *   x: tensor(ids.map((i) => rows[i].slice(0, 2))),
 *   y: fromData(Int32Array.from(ids, (i) => rows[i][2]), [ids.length]),
 * })
 * const data = { p, train: part(order.slice(0, 35)), test: part(order.slice(35)) }
 * const options = { embed: 8, width: 32, steps: 100, recordEvery: 25, checkpointEvery: 25, stepSize: 0.03 }
 * let last
 * for (const s of grokkingRun(data, { ...options, weightDecay: 1 })) last = s
 * print('step:', last.curves.steps)
 * print('train accuracy:', last.curves.trainAccuracy)
 * print('test accuracy:', last.curves.testAccuracy)
 * print('weight norm:', last.curves.weightNorm)
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
