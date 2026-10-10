/**
 * A RealNVP training run as a generator of plain-data snapshots: Adam on minibatches of the negative log-likelihood
 * (`aifn-compute/nn/training`'s `trainingLoop`, gradients clipped to norm 50), with the training NLL and, at
 * checkpoints, the model density on a grid, samples from fixed base draws, and the data after every coupling layer
 * (from the data to the base). Snapshots hold typed arrays only, so they can cross to a worker or a page as they are.
 */

import type { Params } from 'aifn-compute/foundation/pytree'
import { child, standardNormals, stream } from 'aifn-compute/foundation/random'
import { fromData, mean, mul, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { knownDensity, mixtureLogDensityOf, squareGrid } from '../densities'
import { flowForward, flowLogDensity, flowLogDensityValues, flowSample, initRealNvp, realNvp } from './realnvp'

/** Options of `realNvpRun`. */
export interface RealNvpRunOptions {
  /** Coupling layers (default `realNvp`'s 6). */
  layers?: number
  /** Hidden widths of each conditioner (default `realNvp`'s `[32, 32]`). */
  hidden?: readonly number[]
  /** Adam updates (default 2000). */
  steps?: number
  /** Adam's step size (default 1e-3). */
  stepSize?: number
  /** Rows per step (default 128, at most the number of data points). */
  batchSize?: number
  /** The seed of the run's root stream (default 0): the run is deterministic in it. */
  seed?: number | string
  /** Roughly how many checkpoints after the first (default 30): one every `round(steps / checkpoints)` steps. */
  checkpoints?: number
  /** Samples drawn at each checkpoint (default 1000). */
  samples?: number
  /** Grid cells per side (default 48). */
  grid?: number
}

/** One checkpoint. */
export interface RealNvpCheckpoint {
  /** The updates applied when it was taken. */
  step: number
  /** Model density $p(\xvec)$ on the grid, row-major in $(y, x)$: $g^2$ values. */
  density: Float64Array
  /** Samples from the same base draws at every checkpoint, row-major $[\mathit{samples}, 2]$. */
  samples: Float64Array
  /**
   * The followed data points (up to 500, an even stride through the data) after each layer, from the data (entry 0)
   * to the base (entry $K$): $K + 1$ arrays, each row-major $[m, 2]$ for the $m$ points followed.
   */
  layers: Float64Array[]
}

/** A run so far. */
export interface RealNvpRun {
  /** The updates asked for. */
  steps: number
  /** The updates applied so far. */
  done: number
  /** Whether the run is over: all its steps taken, or stopped early because the loss diverged. */
  finished: boolean
  /**
   * The half-width $b$ of the grid's square $[-b, b]^2$: 1.2 times the data's largest $\lvert x_{ij} \rvert$, rounded
   * up to a multiple of 0.5.
   */
  box: number
  /** The grid's cell centres along $x$ ($g$ values). */
  gridX: Float64Array
  /** The grid's cell centres along $y$ ($g$ values). */
  gridY: Float64Array
  /** The true density on the grid when the data's density is known; else null. */
  trueDensity: Float64Array | null
  /** The training data, row-major $[n, 2]$. */
  data: Float64Array
  /** The data's labels, $n$ values (all 0 when the data has none). */
  labels: Int32Array
  /** The labels of the 500 points followed through the layers (an even stride through the data). */
  layerLabels: Int32Array
  /**
   * The model's negative log-likelihood per point on the followed data points (up to 500, an even stride through the
   * data), by step: at step 0, about every fiftieth of the run, and at the last step.
   */
  nll: { step: number[]; value: number[] }
  /**
   * The true density's negative log-likelihood per point on all the training data (an estimate of its entropy, the
   * floor of `nll`) when the data's density is known; else NaN.
   */
  trueNll: number
  /** The checkpoints so far, the first at step 0. */
  checkpoints: RealNvpCheckpoint[]
}

/**
 * Train RealNVP on 2-d data and yield snapshots: one before training, one every twentieth of the run, and the last.
 * Stops early, finished, when the loss diverges. Deterministic in `seed`.
 *
 * @param data The dataset: `x`, the points $[n, 2]$; `y`, optional labels (for colouring); `meta.truth`, the data's
 *   true density when known (see `knownDensity`), for `trueDensity` and `trueNll`.
 * @param options The flow's shape, the training and what is recorded.
 * @returns A generator of snapshots of the run so far; its return value is the last one.
 *
 * @example A short run on a shifted, narrow Gaussian: the training NLL falls
 * const data = { x: normal(stream(1), 2, 0.5, { shape: [200, 2] }) }
 * const options = { steps: 50, layers: 2, hidden: [8], stepSize: 0.02, samples: 10, grid: 8, checkpoints: 2 }
 * const last = [...realNvpRun(data, options)].at(-1)
 * print('steps done', last.done, ' checkpoints at', last.checkpoints.map((c) => c.step))
 * for (const i of [0, 10, 25, 50]) print('step', last.nll.step[i], ' NLL', last.nll.value[i])
 */
export function* realNvpRun(
  data: { x: Tensor; y?: Tensor; meta?: { truth?: unknown } },
  options: RealNvpRunOptions = {},
): Generator<RealNvpRun, RealNvpRun> {
  const { steps = 2000, stepSize = 1e-3, batchSize = 128, seed = 0, checkpoints = 30 } = options
  const { samples = 1000, grid = 48 } = options
  const flow = realNvp({ layers: options.layers, hidden: options.hidden })
  const x = data.x
  const n = x.shape[0]
  const xs = Float64Array.from(toFlat(x))
  const labels = data.y ? Int32Array.from(toFlat(data.y)) : new Int32Array(n)
  let extent = 0
  for (const v of xs) extent = Math.max(extent, Math.abs(v))
  const box = Math.ceil(extent * 1.2 * 2) / 2
  const g = squareGrid(box, grid)
  const model = knownDensity(data.meta?.truth)
  const trueLog = model ? mixtureLogDensityOf(model, g.points) : null
  const trueNll = model ? -mixtureLogDensityOf(model, x).reduce((a, b) => a + b, 0) / n : NaN
  const root = stream(seed)
  const alg = trainingLoop<Params[][] & Params, { x: Tensor }>({
    loss: (p, b) => mul(-1, mean(flowLogDensity(flow, p, b.x))),
    data: { x },
    batchSize: Math.min(batchSize, n),
    optimizer: adamRule({ stepSize }) as UpdateRule<unknown>,
    clipNorm: 50,
  })
  const base = fromData(standardNormals(child(root, 'base'), samples * 2), [samples, 2])
  const shownRows = Math.min(500, n)
  // An even stride through the data, so every mode is represented (generators often list the data mode by mode).
  const shownIds = Array.from({ length: shownRows }, (_, i) => Math.floor((i * n) / shownRows))
  const shown = fromData(Float64Array.from(shownIds.flatMap((i) => [xs[2 * i], xs[2 * i + 1]])), [shownRows, 2])
  const shownLabels = Int32Array.from(shownIds, (i) => labels[i])
  const nll = { step: [] as number[], value: [] as number[] }
  const shots: RealNvpCheckpoint[] = []
  const record = (t: number, p: Params[][]) => {
    nll.step.push(t)
    nll.value.push(-flowLogDensityValues(flow, p, shown).reduce((a, b) => a + b, 0) / shownRows)
  }
  const checkpoint = (t: number, p: Params[][]): RealNvpCheckpoint => ({
    step: t,
    density: Float64Array.from(flowLogDensityValues(flow, p, g.points), Math.exp),
    samples: Float64Array.from(toFlat(flowSample(flow, p, base))),
    layers: flowForward(flow, p, shown).map((v) => Float64Array.from(toFlat(v))),
  })
  const snapshot = (done: number, finished: boolean): RealNvpRun => ({
    steps,
    done,
    finished,
    box,
    gridX: g.x,
    gridY: g.y,
    trueDensity: trueLog ? Float64Array.from(trueLog, Math.exp) : null,
    data: xs,
    labels,
    layerLabels: shownLabels,
    nll: { step: [...nll.step], value: [...nll.value] },
    trueNll,
    checkpoints: shots.slice(),
  })
  const every = Math.max(1, Math.round(steps / checkpoints))
  const recordEvery = Math.max(1, Math.floor(steps / 50))
  const chunk = Math.max(1, Math.ceil(steps / 20))
  let state = alg.init({ params: initRealNvp(flow, child(root, 'init')) as Params[][] & Params }, child(root, 'init'))
  record(0, state.params)
  shots.push(checkpoint(0, state.params))
  yield snapshot(0, false)
  for (let t = 0; t < steps; t++) {
    state = alg.step(state, { t, stream: child(root, 'step', t) })
    const k = t + 1
    if (k % recordEvery === 0 || k === steps) record(k, state.params)
    if (k % every === 0 || k === steps) shots.push(checkpoint(k, state.params))
    if (k === steps || state.diverged) {
      const last = snapshot(k, true)
      yield last
      return last
    }
    if (k % chunk === 0) yield snapshot(k, false)
  }
  return snapshot(steps, true)
}
