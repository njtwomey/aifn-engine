/**
 * Unsupervised domain adaptation on small 2-d problems: labelled source data, unlabelled target data. A feature
 * extractor $f$ (an MLP to a low-dimensional feature space), a linear classifier head $g$ and, for DANN, a domain
 * discriminator $d$ are trained together by Adam on
 *
 * $\mathrm{CE}(g(f(\xvec_s)), y_s) + \lambda \cdot \mathrm{alignment}(f(\xvec_s), f(\xvec_t))$,
 *
 * where the alignment is nothing (source only), the squared MMD, the CORAL loss, or DANN's adversarial term
 * $\mathrm{BCE}(d(R(f(\xvec))), \mathrm{domain})$ through the gradient-reversal layer $R$, which trains $d$ to tell
 * the domains apart and $f$ to make that impossible (Ganin et al., 2016). The run reports source and target accuracy
 * (the target labels are used only to score), the classifier's probability field on a grid, and the features of both
 * domains.
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
  /** The alignment (default `dann`). */
  method?: AdaptationMethod
  /** The alignment weight $\lambda$ (default 10 for MMD, 100 for CORAL, 1 for DANN; unused for source only). */
  lambda?: number
  /** Feature dimension (default 2, so the features can be drawn). */
  features?: number
  /** Hidden widths of the feature extractor (default $[32, 32]$, tanh). */
  hidden?: readonly number[]
  /** Adam updates (default 1500). */
  steps?: number
  /** Adam's step size (default 3e-3). */
  stepSize?: number
  /** Rows per domain per step, drawn with replacement (default 64, at most the domain's size). */
  batchSize?: number
  /** The MMD kernel bandwidth (default 1). */
  bandwidth?: number
  /** The root stream's seed (default 0): the run is deterministic in it. */
  seed?: number | string
  /** About how many checkpoints to keep after the start (default 30): one every `steps / checkpoints` steps. */
  checkpoints?: number
  /** Grid cells per side of the probability field (default 40). */
  grid?: number
  /** The grid's half-width: it spans $[-\mathit{box}, \mathit{box}]^2$ (default 2.5). */
  box?: number
}

/** One checkpoint. */
export interface AdaptationCheckpoint {
  /** Adam updates taken. */
  step: number
  /** Accuracy on every source point. */
  sourceAccuracy: number
  /** Accuracy on every target point (the target labels are used only here). */
  targetAccuracy: number
  /** $P(\text{class } 1 \mid \xvec)$ on the grid, row-major in $(y, x)$: $\mathit{grid}^2$ values. */
  field: Float64Array
  /** Features of the source points, row-major, $n_s$ rows of `features` values. */
  sourceFeatures: Float64Array
  /** Features of the target points, row-major, $n_t$ rows of `features` values. */
  targetFeatures: Float64Array
}

/** A run so far. */
export interface AdaptationRun {
  /** The alignment. */
  method: AdaptationMethod
  /** The updates the run will take. */
  steps: number
  /** The updates taken. */
  done: number
  /** True for the last snapshot. */
  finished: boolean
  /** The grid's coordinates along each axis, `grid` values. */
  gridX: Float64Array
  /**
   * Per recorded step (about 150 of them): the minibatch classification and alignment losses before the update, and
   * the target accuracy after it.
   */
  history: { step: number[]; classification: number[]; alignment: number[]; targetAccuracy: number[] }
  /** The checkpoints so far, the first at step 0. */
  checkpoints: AdaptationCheckpoint[]
  /** The source points (row-major, 2 per row) and labels. */
  source: { x: Float64Array; y: Int32Array }
  /** The target points (row-major, 2 per row) and labels. */
  target: { x: Float64Array; y: Int32Array }
}

/** The three networks: `feature` the extractor $f$, `head` the classifier $g$, `discriminator` the critic $d$. */
type Nets = { feature: Layer<Params[]>; head: Layer<Params>; discriminator: Layer<Params[]> }
/** The parameters of the three networks, trained together. */
type AdaptationParams = { feature: Params[]; head: Params; discriminator: Params[] }

/**
 * The rows of a matrix at the given indices, as a plain tensor.
 *
 * @param t The matrix.
 * @param ids The row indices, repeats allowed.
 * @returns The selected rows, in the order of `ids`.
 */
const rowsOf = (t: Tensor, ids: ArrayLike<number>) => unwrap(take(t, Array.from(ids))) as Tensor

/**
 * Train with the chosen alignment and yield snapshots: at the start, every `steps / 20` updates and at the end (the
 * last also returned). Each update draws a minibatch from each domain, with replacement, from the root stream of
 * `seed`, so the run is deterministic in it. Inputs must have two columns; the classes are $0, \dots, K - 1$ with
 * $K$ one more than the largest source label, and the probability field is that of class 1.
 *
 * @param data The labelled `source` and the `target` domain: inputs $n \times 2$ and integer labels (the target's
 *   used only to score).
 * @param options The alignment, the networks, the training and the grid.
 * @returns A generator of snapshots of the run.
 *
 * @example MMD pulls the target's features onto the source's while the classifier trains
 * const centres = tensor(Array.from({ length: 40 }, (_, i) => (i < 20 ? [-1, 0] : [1, 0])))
 * const xs = add(normals(stream(1), [40, 2], 0, 0.4), centres)
 * const y = tensor(Array.from({ length: 40 }, (_, i) => (i < 20 ? 0 : 1)))
 * const data = { source: { x: xs, y }, target: { x: add(xs, tensor([0.8, 0.8])), y } }
 * const options = { method: 'mmd', steps: 60, batchSize: 20, hidden: [8], grid: 2, checkpoints: 1, seed: 1 }
 * let run
 * for (const snapshot of domainAdaptationRun(data, options)) run = snapshot
 * const meanOf = (f) => [0, 1].map((j) => f.filter((_, i) => i % 2 === j).reduce((a, b) => a + b, 0) / 40)
 * const [start, end] = [run.checkpoints[0], run.checkpoints.at(-1)]
 * print('feature means at the start, source:', meanOf(start.sourceFeatures), 'target:', meanOf(start.targetFeatures))
 * print('feature means at the end, source:', meanOf(end.sourceFeatures), 'target:', meanOf(end.targetFeatures))
 * print('target accuracy at the start and the end:', start.targetAccuracy, end.targetAccuracy)
 */
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
