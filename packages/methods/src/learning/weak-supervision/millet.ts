/**
 * MILLET: multiple instance learning for locally explainable time series classification (Early, Cheung, Cutajar, Xie,
 * Kandola and Twomey 2024, ICLR). A time series X_i = {x_i¹ … x_iᵗ} is a MIL bag whose instances are its time points.
 * A convolutional feature extractor ψ_FE maps it to time-point embeddings Z_i [t, d] (Eq. 1); fixed sinusoidal
 * positional encodings are added and dropout (0.1) applied (App. B.1, Eq. A.1); one of the five MIL poolings of
 * `aifn-compute/nn/layers` (`milPool`: embedding/GAP, attention, instance, additive, conjunctive) gives the series' logits and,
 * inherently, a time-point interpretation (§3.3). Convolutions pad by replicating the boundary value, not zeros (§3.4).
 *
 * The backbone here is a browser-sized FCN (Wang et al. 2017: convolution, ReLU, length kept), a few thousand
 * parameters instead of the paper's 128-wide FCN/ResNet/InceptionTime. The interpretability metrics are the paper's
 * (App. D.1): AOPCR from `aifn-compute/learning/explain` (MoRF in blocks of 5% up to 50%, removed time points dropped from the
 * bag with their positional encodings kept, against three random orders, on the logit of the predicted class) and
 * NDCG@n against the planted discriminatory points (n of them), from `aifn-compute/learning/metrics`.
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
  /** Series length t and number of classes c. */
  readonly length: Size
  readonly classes: Size
  /** Channel widths of the convolutions (the last is the embedding width d, which must be even) and their kernel sizes. */
  readonly widths: readonly Size[]
  readonly kernels: readonly Size[]
  readonly pooling: MilPoolingKind
  /** Add positional encodings (default true) and the dropout rate after them (default 0.1, as the paper). */
  readonly positional: boolean
  readonly dropout: number
  /** Inputs are standardised by this mean and scale (from the training set). */
  readonly mean: number
  readonly scale: number
}

/** A MILLET model's parameters: the convolutions and the pooling. */
export type MilletParams = { convs: ConvParams[]; pool: MilPoolingParams }

/** A MILLET model: its spec, initialiser and forward pass. */
export interface MilletModel {
  readonly spec: MilletSpec
  init(s: Stream): MilletParams
  /** Series [N, t] → the pooling's outputs (logits [N, c] and the interpretation); `mask` [N, t] drops time points. */
  forward(params: MilletParams, x: Value, options?: { ctx?: Context; mask?: Tensor }): MilPooled
}

/** Pad the last axis of x [N, C, L] by repeating its first and last values (`left`, `right` copies). */
export function replicatePad(x: Value, left: Size, right: Size): Value {
  const [N, C, L] = shapeOfValue(x)
  const W = L + left + right
  const idx = new Int32Array(N * C * W)
  for (let r = 0; r < N * C; r++)
    for (let j = 0; j < W; j++) idx[r * W + j] = r * L + Math.min(L - 1, Math.max(0, j - left))
  return gather(x, idx, [N, C, W])
}

/** Build a MILLET model (module notes). */
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

/** A labelled set of series: x [N, t], labels 0 … c − 1. */
export type SeriesSet = { x: Tensor; y: Tensor }

/** Options of {@link milletRun}. */
export interface MilletRunOptions {
  readonly train: SeriesSet
  readonly test: SeriesSet & {
    /** The planted discriminatory points [N, t] (1 = discriminatory), for NDCG@n; optional. */
    discriminatory?: Tensor
  }
  readonly pooling: MilPoolingKind
  readonly widths?: readonly Size[]
  readonly kernels?: readonly Size[]
  /** Positional encodings and dropout (default: on, rate 0.1, for the MIL poolings; off for `embedding`, the GAP baseline). */
  readonly positional?: boolean
  readonly dropout?: number
  /** Optimiser steps (default 300), Adam's step size (default 0.01) and the minibatch size (default 32). */
  readonly steps?: Size
  readonly stepSize?: number
  readonly batchSize?: Size
  /** Checkpoints (default about 30). */
  readonly every?: Size
  /** Test series scored for AOPCR at the end (default 30; NDCG@n uses every test series with planted points). */
  readonly evaluate?: Size
  readonly seed?: number | string
}

/** One checkpoint: the step and the parameters. */
export type MilletCheckpoint = { step: Size; params: MilletParams }

/** The interpretability scores of a model on the test set (App. D.1). */
export interface MilletScores {
  readonly accuracy: number
  /** Mean AOPCR over the scored test series, and mean NDCG@n over the series with planted points. */
  readonly aopcr: number
  readonly ndcg: number
  /** Per-series values (NaN where undefined), in test order (AOPCR for the first `evaluate` only). */
  readonly aopcrPerSeries: number[]
  readonly ndcgPerSeries: number[]
}

/** What {@link milletRun} yields: the run so far. */
export interface MilletSnapshot {
  readonly step: Size
  readonly steps: Size
  readonly done: boolean
  readonly spec: MilletSpec
  readonly history: { step: number[]; loss: number[]; trainAccuracy: number[]; testAccuracy: number[] }
  readonly checkpoints: MilletCheckpoint[]
  /** Set on the final snapshot. */
  readonly scores?: MilletScores
}

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

/** The interpretation of one series for one class: [t] scores (class-agnostic attention for `attention` pooling). */
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

/** Logits [c] of one series with some time points dropped from the bag (`kept[j] = 1` keeps j). */
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
 * (blocks of 5% up to 50%, three random orders) for the first `evaluate` series, and NDCG@n against the planted points.
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
 * standardised by the training set's mean and standard deviation.
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

/** The softmax probabilities [c] of one series, for readouts. */
export const milletProbabilities = (
  model: MilletModel,
  params: MilletParams,
  series: ArrayLike<number>,
): Float64Array =>
  Float64Array.from(toFlat(softmax(fromData(milletLogits(model, params, series), [model.spec.classes])) as Tensor))
