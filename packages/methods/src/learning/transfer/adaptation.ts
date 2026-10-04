/**
 * Unsupervised domain adaptation on small 2-d problems: labelled source data, unlabelled target data. A feature
 * extractor f (an MLP to a low-dimensional feature space), a linear classifier head g and, for DANN, a domain
 * discriminator d are trained together by Adam on
 *
 *   CE(g(f(xₛ)), yₛ) + λ · alignment(f(xₛ), f(xₜ)),
 *
 * where the alignment is nothing (source only), the squared MMD, the CORAL loss, or DANN's adversarial term
 * BCE(d(R(f(x))), domain) through the gradient-reversal layer R, which trains d to tell the domains apart and f to make
 * that impossible (Ganin et al., 2016). The run reports source and target accuracy (the target labels are used only
 * to score), the classifier's probability field on a grid, and the features of both domains.
 */

import type { Params } from 'aifn-compute/foundation/pytree'
import { child, integers, stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  fromData,
  mul,
  take,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { binaryCrossEntropyWithLogits, softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { softmax } from 'aifn-compute/numerics/special'
import { Linear, Mlp, type Layer } from 'aifn-compute/nn/layers'
import { coralLoss, gradientReversal, mmdSquared } from './alignment'
import { adamTrainer } from './shared'

/** The alignment of `domainAdaptationRun`. */
export type AdaptationMethod = 'source-only' | 'mmd' | 'coral' | 'dann'

/** Options of `domainAdaptationRun`. */
export interface AdaptationOptions {
  method?: AdaptationMethod
  /** The alignment weight λ (default 10 for MMD, 100 for CORAL, 1 for DANN). */
  lambda?: number
  /** Feature dimension (default 2, so the features can be drawn) and hidden widths (default [32, 32]). */
  features?: number
  hidden?: readonly number[]
  /** Adam updates (default 1500), step size (default 3e-3), rows per domain per step (default 64). */
  steps?: number
  stepSize?: number
  batchSize?: number
  /** The MMD kernel bandwidth (default 1). */
  bandwidth?: number
  seed?: number | string
  checkpoints?: number
  /** Grid cells per side of the probability field (default 40) and its half-width (default 2.5). */
  grid?: number
  box?: number
}

/** One checkpoint. */
export interface AdaptationCheckpoint {
  step: number
  sourceAccuracy: number
  targetAccuracy: number
  /** P(class 1 | x) on the grid, row-major in (y, x). */
  field: Float64Array
  /** Features of the source and target points, [n × features]. */
  sourceFeatures: Float64Array
  targetFeatures: Float64Array
}

/** A run so far. */
export interface AdaptationRun {
  method: AdaptationMethod
  steps: number
  done: number
  finished: boolean
  gridX: Float64Array
  history: { step: number[]; classification: number[]; alignment: number[]; targetAccuracy: number[] }
  checkpoints: AdaptationCheckpoint[]
  source: { x: Float64Array; y: Int32Array }
  target: { x: Float64Array; y: Int32Array }
}

type Nets = { feature: Layer<Params[]>; head: Layer<Params>; discriminator: Layer<Params[]> }
type AdaptationParams = { feature: Params[]; head: Params; discriminator: Params[] }

const rowsOf = (t: Tensor, ids: ArrayLike<number>) => unwrap(take(t, Array.from(ids))) as Tensor

/** Train with the chosen alignment and yield snapshots (module docs). Deterministic in `seed`. */
export function* domainAdaptationRun(
  data: { source: { x: Tensor; y: Tensor }; target: { x: Tensor; y: Tensor } },
  options: AdaptationOptions = {},
): Generator<AdaptationRun, AdaptationRun> {
  const { method = 'dann', features = 2, hidden = [32, 32], steps = 1500, stepSize = 3e-3 } = options
  const { batchSize = 64, bandwidth = 1, seed = 0, checkpoints = 30, grid = 40, box = 2.5 } = options
  const lambda = options.lambda ?? { 'source-only': 0, mmd: 10, coral: 100, dann: 1 }[method]
  const { source, target } = data
  const ns = source.x.shape[0]
  const nt = target.x.shape[0]
  const ys = Int32Array.from(toFlat(source.y))
  const yt = Int32Array.from(toFlat(target.y))
  const classes = Math.max(...ys) + 1
  const nets: Nets = {
    feature: Mlp([2, ...hidden, features], { activation: 'tanh' }),
    head: Linear(features, classes),
    discriminator: Mlp([features, 32, 1], { activation: 'relu' }),
  }
  const root = stream(seed)
  const init: AdaptationParams = {
    feature: nets.feature.init(child(root, 'feature')),
    head: nets.head.init(child(root, 'head')),
    discriminator: nets.discriminator.init(child(root, 'discriminator')),
  }
  const parts = (p: AdaptationParams, xs: Tensor, ysb: Tensor, xt: Tensor) => {
    const fs = nets.feature.apply(p.feature, xs)
    const ft = nets.feature.apply(p.feature, xt)
    const classification = softmaxCrossEntropy(nets.head.apply(p.head, fs), ysb, { reduction: 'mean' })
    const m = xs.shape[0]
    let alignment: Value = 0
    if (method === 'mmd') alignment = mmdSquared(fs, ft, bandwidth)
    else if (method === 'coral') alignment = coralLoss(fs, ft, { source: m, target: xt.shape[0], dimension: features })
    else if (method === 'dann') {
      const both = gradientReversal(concat([fs, ft], 0), 1)
      const logits = nets.discriminator.apply(p.discriminator, both)
      const domain = fromData(
        Float64Array.from({ length: m + xt.shape[0] }, (_, i) => (i < m ? 0 : 1)),
        [m + xt.shape[0], 1],
      )
      alignment = binaryCrossEntropyWithLogits(logits, domain, { reduction: 'mean' })
    }
    return { classification, alignment }
  }
  const trainer = adamTrainer<AdaptationParams>(stepSize, init)
  const scalar = (v: Value) => {
    const u = unwrap(v)
    return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
  }
  const gx = Float64Array.from({ length: grid }, (_, i) => -box + (2 * box * i) / (grid - 1))
  const gridPoints = fromData(
    Float64Array.from({ length: grid * grid * 2 }, (_, k) =>
      k % 2 === 0 ? gx[(k >> 1) % grid] : gx[Math.floor((k >> 1) / grid)],
    ),
    [grid * grid, 2],
  )
  const probabilities = (p: AdaptationParams, x: Tensor) =>
    toFlat(unwrap(softmax(nets.head.apply(p.head, nets.feature.apply(p.feature, x)))) as Tensor)
  const accuracy = (p: AdaptationParams, x: Tensor, y: Int32Array) => {
    const pr = probabilities(p, x)
    let right = 0
    for (let i = 0; i < y.length; i++) {
      let best = 0
      for (let c = 1; c < classes; c++) if (pr[i * classes + c] > pr[i * classes + best]) best = c
      if (best === y[i]) right++
    }
    return right / y.length
  }
  const history = {
    step: [] as number[],
    classification: [] as number[],
    alignment: [] as number[],
    targetAccuracy: [] as number[],
  }
  const shots: AdaptationCheckpoint[] = []
  const checkpoint = (t: number, p: AdaptationParams): AdaptationCheckpoint => {
    const pr = probabilities(p, gridPoints)
    return {
      step: t,
      sourceAccuracy: accuracy(p, source.x, ys),
      targetAccuracy: accuracy(p, target.x, yt),
      field: Float64Array.from({ length: grid * grid }, (_, i) => pr[i * classes + Math.min(1, classes - 1)]),
      sourceFeatures: Float64Array.from(toFlat(unwrap(nets.feature.apply(p.feature, source.x)) as Tensor)),
      targetFeatures: Float64Array.from(toFlat(unwrap(nets.feature.apply(p.feature, target.x)) as Tensor)),
    }
  }
  const snapshot = (done: number, finished: boolean): AdaptationRun => ({
    method,
    steps,
    done,
    finished,
    gridX: gx,
    history: {
      step: [...history.step],
      classification: [...history.classification],
      alignment: [...history.alignment],
      targetAccuracy: [...history.targetAccuracy],
    },
    checkpoints: shots.slice(),
    source: { x: Float64Array.from(toFlat(source.x)), y: ys },
    target: { x: Float64Array.from(toFlat(target.x)), y: yt },
  })
  let params = init
  shots.push(checkpoint(0, params))
  yield snapshot(0, false)
  const every = Math.max(1, Math.round(steps / checkpoints))
  const recordEvery = Math.max(1, Math.floor(steps / 150))
  const chunk = Math.max(1, Math.ceil(steps / 20))
  for (let t = 1; t <= steps; t++) {
    const si = toFlat(integers(child(root, 'source', t), ns, { shape: [Math.min(batchSize, ns)] }))
    const ti = toFlat(integers(child(root, 'target', t), nt, { shape: [Math.min(batchSize, nt)] }))
    const xs = rowsOf(source.x, si)
    const ysb = fromData(
      Int32Array.from(si, (i) => ys[i]),
      [si.length],
    )
    const xt = rowsOf(target.x, ti)
    let pieces = { classification: NaN, alignment: NaN }
    params = trainer.step(params, (p) => {
      const r = parts(p, xs, ysb, xt)
      pieces = { classification: scalar(r.classification), alignment: scalar(r.alignment) }
      return method === 'source-only' ? r.classification : add(r.classification, mul(lambda, r.alignment))
    })
    if (t % recordEvery === 0 || t === steps) {
      history.step.push(t)
      history.classification.push(pieces.classification)
      history.alignment.push(pieces.alignment)
      history.targetAccuracy.push(accuracy(params, target.x, yt))
    }
    if (t % every === 0 || t === steps) shots.push(checkpoint(t, params))
    if (t === steps) {
      const last = snapshot(t, true)
      yield last
      return last
    }
    if (t % chunk === 0) yield snapshot(t, false)
  }
  return snapshot(steps, true)
}
