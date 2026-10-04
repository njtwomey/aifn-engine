/**
 * Two classifiers of one architecture trained side by side on the same minibatches, one by cross-entropy and one by
 * JEM, as a generator of plain-data snapshots for a worker to stream. At checkpoints each model reports its logits on a
 * grid (decision regions, p(y | x) and the energy −logsumexp f), Langevin samples from p(x) and from each p(x | y),
 * test accuracy and confidence ECE, and how well its energy separates test points from out-of-distribution ones.
 */

import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { live, run } from 'aifn-compute/foundation/trace'
import { langevinParticles } from 'aifn-compute/inference/stochastic'
import { auroc, confidenceCalibrationError } from 'aifn-compute/learning/metrics'
import { softmax } from 'aifn-compute/numerics/special'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { knownDensity, mixtureLogDensityOf, squareGrid } from '../densities'
import { classifier, classifierScore, jemTraining, classifierLogits, uniformBox, type Classifier } from './jem'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A labelled dataset as the run reads it, with the class densities in `meta.truth.model` when known. */
export type JemData = {
  x: Tensor
  /** Integer labels (required). */
  y?: Tensor
  /** A truth with a labelled density (`knownDensity`) gives the true energy. */
  meta?: { labelNames?: readonly string[]; truth?: unknown }
}

/** Options of `jemRun`; plain data. */
export type JemRunOptions = {
  /** Updates of each model. Default 600. */
  steps?: number
  batchSize?: number
  hidden?: readonly number[]
  /** Adam's step size. Default 1e-3. */
  stepSize?: number
  /** The Langevin sampler of JEM's negatives and of the shown samples. */
  langevinSteps?: number
  langevinStepSize?: number
  /** Noise sd; default √(2α), proper Langevin. */
  langevinNoise?: number
  reinitialise?: number
  bufferSize?: number
  /** The energy-magnitude penalty of the CD term (Du & Mordatch, 2019). Default 0.1; 0 often diverges. */
  regularisation?: number
  seed?: number | string
  /** Grid cells per side. Default 40. */
  grid?: number
  /** Checkpoints besides step 0. Default 12. */
  checkpoints?: number
  /** Unconditional samples shown per checkpoint (class-conditional: half as many per class). Default 150. */
  samples?: number
  /** Langevin steps for the shown samples. Default 40. */
  sampleSteps?: number
}

/** One model's state at a checkpoint. */
export type JemCheckpoint = {
  step: number
  /** The parameters, so a page can evaluate the model at a probe point. */
  params: Params[]
  /** Logits on the grid, row-major [g² × K]. */
  logits: Float64Array
  /** Langevin samples from p(x), [m × 2], started uniform on the box. */
  samples: Float64Array
  /** Langevin samples from each p(x | y), [K][m/2 × 2]. */
  conditional: Float64Array[]
  accuracy: number
  /** Confidence (top-label) ECE on the test points. */
  ece: number
  /** AUROC of −E(x) separating test points (positive) from out-of-distribution points. */
  oodAuroc: number
  /** E(x) at the test and the out-of-distribution points. */
  testEnergy: Float64Array
  oodEnergy: Float64Array
}

/** One model's run. */
export type JemTrack = {
  /** Cross-entropy (the supervised part) per update. */
  crossEntropy: Float64Array
  /** The contrastive-divergence term per update (NaN for the cross-entropy model). */
  contrastive: Float64Array
  /** Mean energy of the data minibatch and of the negatives per update (NaN for the cross-entropy model). */
  dataEnergy: Float64Array
  sampleEnergy: Float64Array
  checkpoints: JemCheckpoint[]
}

/** A run so far. */
export type JemRun = {
  steps: number
  done: number
  finished: boolean
  classes: number
  labelNames: string[]
  box: number
  gridX: Float64Array
  gridY: Float64Array
  /** The true log p(x) on the grid, when the class densities are known; else null. */
  trueLogDensity: Float64Array | null
  data: Float64Array
  labels: Int32Array
  test: Float64Array
  testLabels: Int32Array
  ood: Float64Array
  crossEntropy: JemTrack
  jem: JemTrack
}

/** Rows of a [n, d] tensor as a fresh Float64Array. */
const rows = (t: Tensor | Value) => Float64Array.from(toFlat(unwrap(t as Value) as Tensor))

/** The model's energy −logsumexp f at rows, from logits [n × K]. */
function energiesFrom(logits: Float64Array, K: number): Float64Array {
  const n = logits.length / K
  return Float64Array.from({ length: n }, (_, i) => {
    let m = -Infinity
    for (let k = 0; k < K; k++) m = Math.max(m, logits[i * K + k])
    let s = 0
    for (let k = 0; k < K; k++) s += Math.exp(logits[i * K + k] - m)
    return -(m + Math.log(s))
  })
}

/**
 * Train a cross-entropy classifier and a JEM classifier (same network, initial parameters and minibatches) on `data`
 * and yield snapshots about every twentieth of the run and at the end. `test` scores accuracy and calibration; `ood`
 * holds out-of-distribution points for the energy's separation.
 */
export function* jemRun(
  data: JemData,
  test: JemData,
  ood: { x: Tensor },
  options: JemRunOptions = {},
): Generator<JemRun, JemRun> {
  const {
    steps = 600,
    batchSize = 64,
    hidden = [64, 64],
    stepSize = 1e-3,
    langevinSteps = 20,
    langevinStepSize = 0.02,
    reinitialise = 0.05,
    bufferSize = 1000,
    regularisation = 0.1,
    seed = 0,
    grid = 40,
    checkpoints = 12,
    samples = 150,
    sampleSteps = 40,
  } = options
  const noise = options.langevinNoise ?? Math.sqrt(2 * langevinStepSize)
  const xs = rows(data.x)
  if (!data.y || !test.y) throw new DomainError('jemRun', 'jemRun: the training and test data need labels')
  const trainY = data.y
  const labels = Int32Array.from(toFlat(trainY))
  const K = labels.reduce((m, v) => Math.max(m, v + 1), 0)
  let extent = 0
  for (const v of xs) extent = Math.max(extent, Math.abs(v))
  const bound = 1.2 * extent
  const box = Math.ceil(extent * 1.3 * 2) / 2
  const g = squareGrid(box, grid)
  const G = g.points
  const model = knownDensity(data.meta?.truth)
  const trueLogDensity = model ? mixtureLogDensityOf(model, G) : null
  const testY = Int32Array.from(toFlat(test.y))
  const net: Classifier = classifier(2, K, { hidden })
  const root = stream(seed)
  const params0 = net.layer.init(child(root, 'init'))
  const rule = () => adamRule({ stepSize }) as UpdateRule<unknown>
  const sampler = { steps: langevinSteps, stepSize: langevinStepSize, noise, reinitialise }
  const fresh = uniformBox(bound, 2)

  // Langevin samples from p(x) (or p(x | y)), started uniform on the box: the shown samples of a checkpoint.
  const draw = (params: Params[], s: Stream, m: number, y?: number) =>
    rows(
      run(
        langevinParticles(classifierScore(net, params, y), { stepSize: langevinStepSize, noise, bound }),
        { x: fresh(child(s, 'start'), m) },
        sampleSteps,
        { stream: s },
      ).x,
    )
  const checkpoint = (step: number, params: Params[], s: Stream): JemCheckpoint => {
    const logits = rows(classifierLogits(net, params, G))
    const testLogits = rows(classifierLogits(net, params, test.x))
    const probs = rows(softmax(fromData(testLogits, [testY.length, K])))
    let correct = 0
    for (let i = 0; i < testY.length; i++) {
      let best = 0
      for (let k = 1; k < K; k++) if (testLogits[i * K + k] > testLogits[i * K + best]) best = k
      if (best === testY[i]) correct++
    }
    const testEnergy = energiesFrom(testLogits, K)
    const oodEnergy = energiesFrom(rows(classifierLogits(net, params, ood.x)), K)
    const truth = Int32Array.from([...Array(testEnergy.length).fill(1), ...Array(oodEnergy.length).fill(0)])
    const score = Float64Array.from([...testEnergy, ...oodEnergy], (e) => -e)
    return {
      step,
      params,
      logits,
      samples: draw(params, child(s, 'samples'), samples),
      conditional: Array.from({ length: K }, (_, y) => draw(params, child(s, 'class', y), Math.ceil(samples / 2), y)),
      accuracy: correct / testY.length,
      ece: confidenceCalibrationError(testY, fromData(probs, [testY.length, K])),
      oodAuroc: auroc(truth, score),
      testEnergy,
      oodEnergy,
    }
  }
  const make = (objective: 'jem' | 'cross-entropy') =>
    jemTraining({
      net,
      x: data.x,
      y: trainY,
      objective,
      batchSize,
      optimizer: rule(),
      sampler: { ...sampler, bound, fresh },
      bufferSize,
      regularisation,
      bound,
    })
  // The same root stream for both: identical minibatches (the cross-entropy model draws no negatives).
  const trainStream = child(root, 'train')
  const runs = {
    crossEntropy: live(make('cross-entropy'), { params: params0 }, { stream: trainStream }),
    jem: live(make('jem'), { params: params0 }, { stream: trainStream }),
  }
  const tracks = {
    crossEntropy: {
      ce: [] as number[],
      cd: [] as number[],
      de: [] as number[],
      se: [] as number[],
      shots: [] as JemCheckpoint[],
    },
    jem: {
      ce: [] as number[],
      cd: [] as number[],
      de: [] as number[],
      se: [] as number[],
      shots: [] as JemCheckpoint[],
    },
  }
  const every = Math.max(1, Math.round(steps / checkpoints))
  const chunk = Math.max(1, Math.ceil(steps / 20))
  const labelNames = data.meta?.labelNames
    ? [...data.meta.labelNames]
    : Array.from({ length: K }, (_, k) => `class ${k + 1}`)
  const track = (t: (typeof tracks)['jem']): JemTrack => ({
    crossEntropy: Float64Array.from(t.ce),
    contrastive: Float64Array.from(t.cd),
    dataEnergy: Float64Array.from(t.de),
    sampleEnergy: Float64Array.from(t.se),
    checkpoints: t.shots.slice(),
  })
  const snapshot = (done: number, finished: boolean): JemRun => ({
    steps,
    done,
    finished,
    classes: K,
    labelNames,
    box,
    gridX: g.x,
    gridY: g.y,
    trueLogDensity,
    data: xs,
    labels,
    test: rows(test.x),
    testLabels: testY,
    ood: rows(ood.x),
    crossEntropy: track(tracks.crossEntropy),
    jem: track(tracks.jem),
  })
  for (let step = 0; step <= steps; step++) {
    for (const key of ['crossEntropy', 'jem'] as const) {
      const next = runs[key].next()
      if (next.done) continue
      const { state } = next.value
      const t = tracks[key]
      if (step > 0) {
        t.ce.push(state.supervisedLoss)
        t.cd.push(state.generativeLoss)
        t.de.push(state.dataEnergy)
        t.se.push(state.sampleEnergy)
      }
      if (step % every === 0 || step === steps)
        t.shots.push(checkpoint(step, state.params, child(root, 'checkpoint', step)))
    }
    if (step === steps) {
      const last = snapshot(step, true)
      yield last
      return last
    }
    if (step > 0 && step % chunk === 0) yield snapshot(step, false)
  }
  return snapshot(steps, true)
}
