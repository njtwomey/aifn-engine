/**
 * Two classifiers of one architecture trained side by side on the same minibatches, one by cross-entropy and one by
 * JEM, as a generator of plain-data snapshots for a worker to stream. At checkpoints each model reports its logits on a
 * grid (decision regions, $p(y \mid \xvec)$ and the energy $-\operatorname{logsumexp} f$), Langevin samples from
 * $p(\xvec)$ and from each $p(\xvec \mid y)$, test accuracy and confidence ECE, and how well its energy separates
 * test points from out-of-distribution ones. The points are 2-d.
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
  /** The points, $n \times 2$. */
  x: Tensor
  /** Integer labels from 0, $n$ of them (required: `jemRun` throws without them). */
  y?: Tensor
  /**
   * Class names, and the dataset's truth: one with a labelled density (`knownDensity`) gives the true log-density.
   */
  meta?: { labelNames?: readonly string[]; truth?: unknown }
}

/** Options of `jemRun`; plain data. */
export type JemRunOptions = {
  /** Updates of each model. Default 600. */
  steps?: number
  /** Points per minibatch. Default 64. */
  batchSize?: number
  /** The classifier's hidden widths. Default $[64, 64]$. */
  hidden?: readonly number[]
  /** Adam's step size. Default $10^{-3}$. */
  stepSize?: number
  /** Langevin steps per draw of JEM's negatives. Default 20. */
  langevinSteps?: number
  /** The Langevin step size $\epsilon$, of JEM's negatives and of the shown samples. Default 0.02. */
  langevinStepSize?: number
  /** The Langevin noise's standard deviation. Default $\sqrt{2\epsilon}$, proper Langevin. */
  langevinNoise?: number
  /** The probability that a chain of the replay buffer restarts from uniform on the box. Default 0.05. */
  reinitialise?: number
  /** Persistent chains in JEM's replay buffer. Default 1000. */
  bufferSize?: number
  /** The energy-magnitude penalty of the CD term (Du & Mordatch, 2019). Default 0.1; 0 often diverges. */
  regularisation?: number
  /** The seed of the run's stream. Default 0. */
  seed?: number | string
  /** Grid cells per side $g$. Default 40. */
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
  /** The updates applied so far. */
  step: number
  /** The parameters, so a page can evaluate the model at a probe point. */
  params: Params[]
  /** Logits on the grid's cell centres, row-major $g^2 \times K$, one grid row after another. */
  logits: Float64Array
  /** Langevin samples from $p(\xvec)$, row-major $m \times 2$ ($m$ = `samples`), started uniform on the box. */
  samples: Float64Array
  /** Langevin samples from each $p(\xvec \mid y)$: per class, row-major $\lceil m/2 \rceil \times 2$. */
  conditional: Float64Array[]
  /** The share of test points classified correctly. */
  accuracy: number
  /** Confidence (top-label) ECE on the test points. */
  ece: number
  /** AUROC of $-E(\xvec)$ separating test points (positive) from out-of-distribution points. */
  oodAuroc: number
  /** $E(\xvec)$ at the test points. */
  testEnergy: Float64Array
  /** $E(\xvec)$ at the out-of-distribution points. */
  oodEnergy: Float64Array
}

/** One model's run. */
export type JemTrack = {
  /** Cross-entropy (the supervised part) per update. */
  crossEntropy: Float64Array
  /** The contrastive-divergence term per update (NaN for the cross-entropy model). */
  contrastive: Float64Array
  /** Mean energy of the data minibatch per update (NaN for the cross-entropy model). */
  dataEnergy: Float64Array
  /** Mean energy of the negatives per update (NaN for the cross-entropy model). */
  sampleEnergy: Float64Array
  /** The checkpoints so far. */
  checkpoints: JemCheckpoint[]
}

/** A run so far. */
export type JemRun = {
  /** The updates the run will make. */
  steps: number
  /** The updates made so far. */
  done: number
  /** True at the end of the run. */
  finished: boolean
  /** The number of classes $K$ (one more than the largest training label). */
  classes: number
  /** The class names: the data's, or `class 1`, `class 2`, and so on. */
  labelNames: string[]
  /** The grid's half-width: the square $[-\text{box}, \text{box}]^2$ holds the data with a margin. */
  box: number
  /** The grid's cell centres along $x_1$. */
  gridX: Float64Array
  /** The grid's cell centres along $x_2$. */
  gridY: Float64Array
  /** The true $\log p(\xvec)$ on the grid, when the class densities are known; else null. */
  trueLogDensity: Float64Array | null
  /** The training points, row-major $n \times 2$. */
  data: Float64Array
  /** The training labels. */
  labels: Int32Array
  /** The test points, row-major. */
  test: Float64Array
  /** The test labels. */
  testLabels: Int32Array
  /** The out-of-distribution points, row-major. */
  ood: Float64Array
  /** The cross-entropy model's track. */
  crossEntropy: JemTrack
  /** The JEM model's track. */
  jem: JemTrack
}

/**
 * The entries of a tensor (or a traced value's tensor) as a fresh array, row-major.
 *
 * @param t The tensor, such as $n \times d$ points.
 * @returns Its values, row-major.
 */
const rows = (t: Tensor | Value) => Float64Array.from(toFlat(unwrap(t as Value) as Tensor))

/**
 * The model's energy $E = -\operatorname{logsumexp} f$ at each row, from its logits.
 *
 * @param logits The logits, row-major $n \times K$.
 * @param K The number of classes $K$.
 * @returns The $n$ energies.
 */
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
 * holds out-of-distribution points for the energy's separation. Checkpoints are taken at step 0, about every
 * `steps / checkpoints` updates and at the end. Throws `DomainError` when `data` or `test` has no labels.
 * Deterministic in `seed`.
 *
 * @param data The training points, $n \times 2$, with their labels (and, when known, their true densities).
 * @param test The test points and labels, for accuracy, calibration and the energy's separation.
 * @param ood Out-of-distribution points, $m \times 2$.
 * @param options The run's length, the network and optimiser, the Langevin sampler and replay buffer, and what the
 *   checkpoints hold.
 * @returns A generator of `JemRun` snapshots; the last has `finished` set.
 *
 * @example A tiny run on two clusters, both models side by side
 * const cluster = (s, c) => normal(s, c, 0.3, { shape: [8, 2] })
 * const x = concat([cluster(stream(1), -1), cluster(stream(2), 1)], 0)
 * const y = tensor([...Array(8).fill(0), ...Array(8).fill(1)])
 * const ood = { x: normal(stream(3), 0, 4, { shape: [8, 2] }) }
 * const options = { steps: 10, batchSize: 8, hidden: [8], langevinSteps: 3, bufferSize: 16, grid: 4 }
 * let last
 * for (const r of jemRun({ x, y }, { x, y }, ood, { ...options, checkpoints: 1, samples: 2, sampleSteps: 2 })) last = r
 * const [ce, jem] = [last.crossEntropy.checkpoints.at(-1), last.jem.checkpoints.at(-1)]
 * print('test accuracy, cross-entropy and JEM:', ce.accuracy, jem.accuracy)
 * print('JEM cross-entropy by update:', last.jem.crossEntropy)
 * print('OOD AUROC of the energy, cross-entropy and JEM:', ce.oodAuroc, jem.oodAuroc)
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
