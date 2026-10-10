/**
 * MILLET: multiple instance learning for locally explainable time series classification (Early, Cheung, Cutajar, Xie,
 * Kandola and Twomey 2024, ICLR). A time series $\Xmat_i = \{x_i^1, \dots, x_i^t\}$ is a MIL bag whose instances are
 * its time points. A convolutional feature extractor $\psi_{\text{FE}}$ maps it to time-point embeddings $\Zmat_i$
 * ($t \times d$, Eq. 1); fixed sinusoidal positional encodings are added and dropout (0.1) applied (App. B.1, Eq. A.1);
 * one of the five MIL poolings of `aifn-compute/nn/layers` (`milPool`: embedding/GAP, attention, instance, additive,
 * conjunctive) gives the series' logits and, inherently, a time-point interpretation (§3.3). Convolutions pad by
 * replicating the boundary value, not zeros (§3.4).
 *
 * The backbone here is a browser-sized FCN (Wang et al. 2017: convolution, ReLU, length kept), a few thousand
 * parameters instead of the paper's 128-wide FCN/ResNet/InceptionTime. The interpretability metrics are the paper's
 * (App. D.1): AOPCR from `aifn-compute/learning/explain` (MoRF in blocks of 5% up to 50%, removed time points dropped
 * from the bag with their positional encodings kept, against three random orders, on the logit of the predicted class)
 * and NDCG@$n$ against the planted discriminatory points ($n$ of them), from `aifn-compute/learning/metrics`.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream as makeStream, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  dense,
  fromData,
  gather,
  mul,
  reshape,
  shapeOfValue,
  toFlat,
  transpose,
  type Tensor,
  unwrap,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { aopcr } from 'aifn-compute/learning/explain'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { ndcg } from 'aifn-compute/learning/metrics'
import { sinusoidalPositions } from 'aifn-compute/nn/attention'
import { relu } from 'aifn-compute/nn/functional'
import {
  Conv1d,
  dropout,
  milPool,
  MilPooling,
  type Context,
  type ConvParams,
  type MilPooled,
  type MilPoolingKind,
  type MilPoolingParams,
} from 'aifn-compute/nn/layers'
import { methodTraining, type TrainingMethod } from 'aifn-compute/nn/training'
import { softmax } from 'aifn-compute/numerics/special'

/** The shape of a MILLET model. */
export interface MilletSpec {
  /** The series length $t$. */
  readonly length: Size
  /** The number of classes $c$. */
  readonly classes: Size
  /**
   * Channel widths of the convolutions, one per layer; the last is the embedding width $d$, which must be even when
   * `positional` is set.
   */
  readonly widths: readonly Size[]
  /** The kernel size of each convolution, one per entry of `widths`. */
  readonly kernels: readonly Size[]
  /** The MIL pooling. */
  readonly pooling: MilPoolingKind
  /** Add sinusoidal positional encodings to the embeddings (`milletRun` sets it for every pooling but `embedding`). */
  readonly positional: boolean
  /** The dropout rate after them in training (`milletRun`: 0.1, as the paper, or 0 for `embedding`). */
  readonly dropout: number
  /** Inputs are standardised by subtracting this mean (from the training set). */
  readonly mean: number
  /** ... and dividing by this scale. */
  readonly scale: number
}

/** A MILLET model's parameters: `convs`, one entry per convolution, and `pool`, the pooling's. */
export type MilletParams = { convs: ConvParams[]; pool: MilPoolingParams }

/** A MILLET model: its spec, initialiser and forward pass. */
export interface MilletModel {
  /** The shape it was built from. */
  readonly spec: MilletSpec
  /** Fresh parameters, the convolutions drawn from `child(s, 'conv', l)` and the pooling from `child(s, 'pool')`. */
  init(s: Stream): MilletParams
  /**
   * Series $N \times t$ to the pooling's outputs (logits $N \times c$ and the interpretation). `mask`, $N \times t$ of
   * 1 (kept) and 0, drops time points from their bag; dropout is applied only with a training `ctx` that has a stream.
   */
  forward(params: MilletParams, x: Value, options?: { ctx?: Context; mask?: Tensor }): MilPooled
}

/**
 * Pad the last axis of a batch of signals by repeating its first and last values, as MILLET's convolutions pad
 * (differentiable: a gather).
 *
 * @param x The signals, $N \times C \times L$.
 * @param left The number of copies of the first value put before it.
 * @param right The number of copies of the last value put after it.
 * @returns The padded signals, $N \times C \times (L + \text{left} + \text{right})$.
 *
 * @example Two copies of the first value, one of the last
 * print('padded:', replicatePad(tensor([[[1, 2, 3, 4]]]), 2, 1))
 */
export function replicatePad(x: Value, left: Size, right: Size): Value {
  const [N, C, L] = shapeOfValue(x)
  const W = L + left + right
  const idx = new Int32Array(N * C * W)
  for (let r = 0; r < N * C; r++)
    for (let j = 0; j < W; j++) idx[r * W + j] = r * L + Math.min(L - 1, Math.max(0, j - left))
  return gather(x, idx, [N, C, W])
}

/**
 * Build a MILLET model (see the file's notes): standardisation, the convolutions (each padded by replication to keep
 * the length, then ReLU), positional encodings and dropout, and the MIL pooling. Throws when there is not one kernel
 * size per convolution, or none.
 *
 * @param spec The model's shape: length, classes, widths and kernels, pooling, positional encodings, dropout and
 *   standardisation.
 * @returns The model, with its initialiser and forward pass.
 *
 * @example An untrained model's logits and interpretation for two series
 * const model = milletModel({
 *   length: 8, classes: 2, widths: [4], kernels: [3], pooling: 'conjunctive',
 *   positional: true, dropout: 0, mean: 0, scale: 1,
 * })
 * const params = model.init(stream(0))
 * const out = model.forward(params, normals(stream(1), [2, 8]))
 * print('logits:', out.logits)
 * print('interpretation shape:', out.interpretation.shape)
 */
export function milletModel(spec: MilletSpec): MilletModel {
  const { length: t, classes, widths, kernels, pooling, positional, dropout: rate, mean, scale } = spec
  if (widths.length !== kernels.length || widths.length === 0)
    throw new Error('milletModel: give one kernel size per convolution')
  const d = widths[widths.length - 1]
  const convs = widths.map((w, l) => Conv1d(l === 0 ? 1 : widths[l - 1], w, kernels[l]))
  const pool = MilPooling(d, classes, pooling)
  const pe = positional
    ? sinusoidalPositions(
        Array.from({ length: t }, (_, j) => j + 1),
        d,
      )
    : null
  return {
    spec,
    init: (s) => ({
      convs: convs.map((c, l) => c.init(child(s, 'conv', l))),
      pool: pool.init(child(s, 'pool')),
    }),
    forward: (p, x, options = {}) => {
      const [N] = shapeOfValue(x)
      let h: Value = reshape(mul(add(x, -mean), 1 / scale), [N, 1, t])
      convs.forEach((c, l) => {
        const k = kernels[l]
        const padded = replicatePad(h, Math.floor((k - 1) / 2), k - 1 - Math.floor((k - 1) / 2))
        h = relu(c.apply(p.convs[l], padded))
      })
      // [N, d, t] → [N, t, d]: one embedding per time point.
      let z: Value = transpose(h, [0, 2, 1])
      if (pe) z = add(z, pe)
      if (rate > 0 && options.ctx?.train && options.ctx.stream)
        z = dropout(child(options.ctx.stream, 'dropout'), z, rate)
      return milPool(pooling, p.pool, z, options.mask)
    },
  }
}

// ── Training ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A labelled set of series: `x`, $N \times t$, and `y`, $N$ labels in $0, \dots, c - 1$. */
export type SeriesSet = { x: Tensor; y: Tensor }

/** Options of {@link milletRun}. */
export interface MilletRunOptions {
  /** The training series. */
  readonly train: SeriesSet
  /** The test series, scored at every checkpoint and for interpretability at the end. */
  readonly test: SeriesSet & {
    /** The planted discriminatory points, $N \times t$ (1 = discriminatory), for NDCG@$n$; optional. */
    discriminatory?: Tensor
  }
  /** The MIL pooling. */
  readonly pooling: MilPoolingKind
  /** Channel widths of the convolutions (default $[8, 16]$). */
  readonly widths?: readonly Size[]
  /** Their kernel sizes (default $[7, 5]$). */
  readonly kernels?: readonly Size[]
  /** Positional encodings (default: on for the MIL poolings; off for `embedding`, the GAP baseline). */
  readonly positional?: boolean
  /** The dropout rate (default 0.1 for the MIL poolings; 0 for `embedding`). */
  readonly dropout?: number
  /** Optimiser steps (default 300). */
  readonly steps?: Size
  /** Adam's step size (default 0.01). */
  readonly stepSize?: number
  /** The minibatch size (default 32, or the training set's size when smaller). */
  readonly batchSize?: Size
  /** Steps between checkpoints (default `steps / 30` rounded, so about 30 checkpoints; at least 1). */
  readonly every?: Size
  /** Test series scored for AOPCR at the end (default 30; NDCG@$n$ uses every test series with planted points). */
  readonly evaluate?: Size
  /** Seed of the initial weights, the minibatches, dropout and AOPCR's random orders (default `'millet'`). */
  readonly seed?: number | string
}

/** One checkpoint: the `step` and the `params` after it. */
export type MilletCheckpoint = { step: Size; params: MilletParams }

/** The interpretability scores of a model on the test set (App. D.1). */
export interface MilletScores {
  /** The accuracy of the argmax of the logits on the test set. */
  readonly accuracy: number
  /** Mean AOPCR over the scored test series. */
  readonly aopcr: number
  /** Mean NDCG@$n$ over the series with planted points (NaN when none has any). */
  readonly ndcg: number
  /** AOPCR per series, in test order (NaN after the first `evaluate`). */
  readonly aopcrPerSeries: number[]
  /** NDCG@$n$ per series, in test order (NaN for a series with no planted points). */
  readonly ndcgPerSeries: number[]
}

/** What {@link milletRun} yields: the run so far. */
export interface MilletSnapshot {
  /** The step reached. */
  readonly step: Size
  /** The steps of the whole run. */
  readonly steps: Size
  /** Whether training is over (the final snapshot, with `scores`). */
  readonly done: boolean
  /** The model's shape, with the training set's standardisation. */
  readonly spec: MilletSpec
  /**
   * At each checkpoint so far: the step, the training loss on the whole training set (without dropout), and the
   * training and test accuracies.
   */
  readonly history: { step: number[]; loss: number[]; trainAccuracy: number[]; testAccuracy: number[] }
  /** The parameters at each checkpoint so far. */
  readonly checkpoints: MilletCheckpoint[]
  /** Set on the final snapshot. */
  readonly scores?: MilletScores
}

/**
 * The accuracy of the argmax of the logits (the first class on a tie).
 *
 * @param logits The logits, $N \times c$.
 * @param y The true class of each series.
 * @returns The share of series whose argmax is their class.
 */
function accuracyOf(logits: Tensor, y: ArrayLike<number>): number {
  const [N, c] = logits.shape
  const z = dense.data(logits)
  let hit = 0
  for (let i = 0; i < N; i++) {
    let best = 0
    for (let k = 1; k < c; k++) if (z[i * c + k] > z[i * c + best]) best = k
    if (best === y[i]) hit++
  }
  return hit / N
}

/**
 * The interpretation of one series for one class: $t$ scores, one per time point (the class-agnostic attention for
 * `attention` pooling, whatever the class).
 *
 * @param model The model.
 * @param params Its parameters.
 * @param series The series, $t$ values.
 * @param label The class the interpretation is for.
 * @returns The score of each time point.
 *
 * @example Where an untrained model looks, for each class
 * const model = milletModel({
 *   length: 6, classes: 2, widths: [4], kernels: [3], pooling: 'instance',
 *   positional: false, dropout: 0, mean: 0, scale: 1,
 * })
 * const params = model.init(stream(0))
 * const series = [0, 0, 3, 3, 0, 0]
 * print('class 0:', milletInterpretation(model, params, series, 0))
 * print('class 1:', milletInterpretation(model, params, series, 1))
 */
export function milletInterpretation(
  model: MilletModel,
  params: MilletParams,
  series: ArrayLike<number>,
  label: Size,
): Float64Array {
  const t = model.spec.length
  const out = model.forward(params, fromData(Float64Array.from(series), [1, t]))
  const v = toFlat(out.interpretation as Tensor)
  if (model.spec.pooling === 'attention') return Float64Array.from(v)
  const c = model.spec.classes
  return Float64Array.from({ length: t }, (_, j) => v[j * c + label])
}

/**
 * Logits ($c$ values) of one series, with some time points dropped from the bag.
 *
 * @param model The model.
 * @param params Its parameters.
 * @param series The series, $t$ values.
 * @param kept Which time points stay in the bag: `kept[j] = 1` keeps $j$, 0 drops it; left out, all stay.
 * @returns The logits, one per class.
 *
 * @example Dropping time points from the bag changes the logits
 * const model = milletModel({
 *   length: 6, classes: 2, widths: [4], kernels: [3], pooling: 'additive',
 *   positional: false, dropout: 0, mean: 0, scale: 1,
 * })
 * const params = model.init(stream(0))
 * const series = [0, 0, 3, 3, 0, 0]
 * print('every point:', milletLogits(model, params, series))
 * print('the bump only:', milletLogits(model, params, series, Uint8Array.from([0, 0, 1, 1, 0, 0])))
 */
export function milletLogits(
  model: MilletModel,
  params: MilletParams,
  series: ArrayLike<number>,
  kept?: Uint8Array,
): Float64Array {
  const t = model.spec.length
  const mask = kept ? fromData(Float64Array.from(kept), [1, t]) : undefined
  return Float64Array.from(
    toFlat(model.forward(params, fromData(Float64Array.from(series), [1, t]), { mask }).logits as Tensor),
  )
}

/**
 * The interpretability scores of a trained model on a test set (App. D.1): AOPCR on the logit of the predicted class
 * (blocks of 5% up to 50%, three random orders) for the first `evaluate` series, and NDCG@$n$ of the interpretation for
 * the true class against the planted points.
 *
 * @param s The stream AOPCR's random orders come from (series $i$ uses `child(s, 'aopcr', i)`).
 * @param model The model.
 * @param params Its parameters.
 * @param test The test series, with their planted points when known.
 * @param evaluate The number of test series, from the first, scored for AOPCR.
 * @returns The accuracy and the mean and per-series AOPCR and NDCG@$n$.
 *
 * @example The scores' shape, from an untrained model on four series
 * const model = milletModel({
 *   length: 8, classes: 2, widths: [4], kernels: [3], pooling: 'conjunctive',
 *   positional: false, dropout: 0, mean: 0, scale: 1,
 * })
 * const params = model.init(stream(0))
 * const marks = [[], [2, 3], [], [5, 6]]
 * const discriminatory = tensor(marks.map((m) => Array.from({ length: 8 }, (_, j) => (m.includes(j) ? 1 : 0))))
 * const x = mul(discriminatory, 3)
 * const scores = milletScores(stream(1), model, params, { x, y: tensor([0, 1, 0, 1]), discriminatory }, 4)
 * print('AOPCR per series:', scores.aopcrPerSeries)
 * print('NDCG@n per series (none without planted points):', scores.ndcgPerSeries)
 * print('means:', scores.aopcr, scores.ndcg)
 */
export function milletScores(
  s: Stream,
  model: MilletModel,
  params: MilletParams,
  test: MilletRunOptions['test'],
  evaluate = 30,
): MilletScores {
  const [N, t] = test.x.shape
  const X = dense.data(test.x)
  const y = toFlat(test.y)
  const logits = model.forward(params, test.x).logits as Tensor
  const accuracy = accuracyOf(logits, y)
  const L = dense.data(logits)
  const c = model.spec.classes
  const mask = test.discriminatory ? dense.data(test.discriminatory) : null
  const aopcrPerSeries: number[] = []
  const ndcgPerSeries: number[] = []
  for (let i = 0; i < N; i++) {
    const series = X.subarray(i * t, (i + 1) * t)
    let pred = 0
    for (let k = 1; k < c; k++) if (L[i * c + k] > L[i * c + pred]) pred = k
    const scores = milletInterpretation(model, params, series, pred)
    if (i < evaluate) {
      const r = aopcr(child(s, 'aopcr', i), (kept) => milletLogits(model, params, series, kept)[pred], scores, {
        block: Math.max(1, Math.round(0.05 * t)),
        until: 0.5,
      })
      aopcrPerSeries.push(r.aopcr)
    } else aopcrPerSeries.push(NaN)
    const rel = mask ? mask.subarray(i * t, (i + 1) * t) : null
    const n = rel ? rel.reduce((a, v) => a + v, 0) : 0
    // NDCG@n of the interpretation for the series' true class against its planted points.
    ndcgPerSeries.push(rel && n > 0 ? ndcg(rel, milletInterpretation(model, params, series, y[i]), { k: n }) : NaN)
  }
  const mean = (v: number[]) => {
    const f = v.filter(Number.isFinite)
    return f.length ? f.reduce((a, x) => a + x, 0) / f.length : NaN
  }
  return { accuracy, aopcr: mean(aopcrPerSeries), ndcg: mean(ndcgPerSeries), aopcrPerSeries, ndcgPerSeries }
}

/**
 * Train a MILLET model end to end by Adam on the softmax cross-entropy of minibatches, as a generator yielding the run
 * at each checkpoint (the first at step 0, before training) and, at the end, the interpretability scores. Series are
 * standardised by the training set's mean and standard deviation (over every value). The number of classes is one
 * more than the largest label of the training and test sets.
 *
 * @param options The data, the model, the training and the evaluation.
 * @returns A generator of snapshots: one at step 0, one every `every` steps before the last, and a final one, with
 *   `done` and `scores`, which it also returns.
 *
 * @example Learn where a bump sits in a short series
 * // Class 1 series have a bump of height 3 at a random place; class 0 series are noise alone.
 * const s = stream(3)
 * const make = (count) => {
 *   const [x, y, d] = [[], [], []]
 *   for (let i = 0; i < count; i++) {
 *     const label = i % 2
 *     const at = 2 + Math.floor(uniform(s) * 10)
 *     const row = Array.from({ length: 16 }, () => normal(s, 0, 0.3))
 *     const mark = row.map((_, j) => (label === 1 && j >= at && j < at + 3 ? 1 : 0))
 *     x.push(row.map((v, j) => v + 3 * mark[j]))
 *     y.push(label)
 *     d.push(mark)
 *   }
 *   return { x: tensor(x), y: tensor(y), discriminatory: tensor(d) }
 * }
 * const options = { train: make(16), test: make(6), pooling: 'conjunctive', widths: [4, 8], kernels: [3, 3] }
 * let last
 * for (const snapshot of milletRun({ ...options, steps: 30, batchSize: 8, every: 10, evaluate: 2 })) last = snapshot
 * print('loss by checkpoint:', last.history.loss)
 * print('test accuracy by checkpoint:', last.history.testAccuracy)
 * print('AOPCR:', last.scores.aopcr, ' NDCG@n:', last.scores.ndcg)
 */
export function* milletRun(options: MilletRunOptions): Generator<MilletSnapshot, MilletSnapshot> {
  const {
    train,
    test,
    pooling,
    widths = [8, 16],
    kernels = [7, 5],
    // As the paper, positional encodings and dropout belong to the MIL poolings, not to the GAP backbone.
    positional = pooling !== 'embedding',
    dropout: rate = pooling === 'embedding' ? 0 : 0.1,
    steps = 300,
    stepSize = 0.01,
    batchSize = 32,
    evaluate = 30,
    seed = 'millet',
  } = options
  const every = Math.max(1, options.every ?? Math.round(steps / 30))
  const [, t] = train.x.shape
  const yTrain = toFlat(train.y)
  const yTest = toFlat(test.y)
  let classes = 0
  for (const v of [...yTrain, ...yTest]) classes = Math.max(classes, v + 1)
  const X = dense.data(train.x)
  const mean = X.reduce((a, v) => a + v, 0) / X.length
  const scale = Math.sqrt(X.reduce((a, v) => a + (v - mean) ** 2, 0) / X.length) || 1
  const model = milletModel({ length: t, classes, widths, kernels, pooling, positional, dropout: rate, mean, scale })
  const root = makeStream(seed)
  const method: TrainingMethod = { method: 'adam', stepSize, batchSize: Math.min(batchSize, yTrain.length) }
  const data = { x: train.x, y: fromData(Float64Array.from(yTrain), [yTrain.length]) }
  const alg = methodTraining<MilletParams & Params, typeof data>(
    (p, batch, ctx) => softmaxCrossEntropy(model.forward(p, batch.x, { ctx }).logits, toFlat(batch.y)),
    data,
    method,
  )
  let state = alg.init({ params: model.init(child(root, 'init')) as MilletParams & Params }, child(root, 'train'))
  const history = {
    step: [] as number[],
    loss: [] as number[],
    trainAccuracy: [] as number[],
    testAccuracy: [] as number[],
  }
  const checkpoints: MilletCheckpoint[] = []
  const record = (k: Size) => {
    const lt = model.forward(state.params, train.x).logits as Tensor
    history.step.push(k)
    history.loss.push(Number(unwrap(softmaxCrossEntropy(lt, yTrain))))
    history.trainAccuracy.push(accuracyOf(lt, yTrain))
    history.testAccuracy.push(accuracyOf(model.forward(state.params, test.x).logits as Tensor, yTest))
    checkpoints.push({ step: k, params: state.params })
  }
  const snapshot = (k: Size, done: boolean, scores?: MilletScores): MilletSnapshot => ({
    step: k,
    steps,
    done,
    spec: model.spec,
    history: {
      step: [...history.step],
      loss: [...history.loss],
      trainAccuracy: [...history.trainAccuracy],
      testAccuracy: [...history.testAccuracy],
    },
    checkpoints: [...checkpoints],
    ...(scores ? { scores } : {}),
  })
  record(0)
  yield snapshot(0, false)
  for (let k = 1; k <= steps; k++) {
    state = alg.step(state, { t: k - 1, stream: child(root, 'step', k - 1) })
    if (k % every === 0 || k === steps) {
      record(k)
      if (k < steps) yield snapshot(k, false)
    }
  }
  const scores = milletScores(child(root, 'scores'), model, state.params, test, evaluate)
  // The worker keeps the last value yielded, not the return value.
  const final = snapshot(steps, true, scores)
  yield final
  return final
}

/**
 * The softmax probabilities ($c$ values) of one series, for readouts.
 *
 * @param model The model.
 * @param params Its parameters.
 * @param series The series, $t$ values.
 * @returns The class probabilities.
 *
 * @example An untrained model's class probabilities
 * const model = milletModel({
 *   length: 6, classes: 3, widths: [4], kernels: [3], pooling: 'attention',
 *   positional: false, dropout: 0, mean: 0, scale: 1,
 * })
 * print('p =', milletProbabilities(model, model.init(stream(0)), [0, 1, 2, 3, 2, 1]))
 */
export const milletProbabilities = (
  model: MilletModel,
  params: MilletParams,
  series: ArrayLike<number>,
): Float64Array =>
  Float64Array.from(toFlat(softmax(fromData(milletLogits(model, params, series), [model.spec.classes])) as Tensor))
