/**
 * A RealNVP training run as a generator of plain-data snapshots: Adam on minibatches of the negative log-likelihood
 * (`aifn-compute/nn/training`'s `trainingLoop`), with the training NLL and, at checkpoints, the model density on a grid,
 * samples from fixed base draws, and the data after every coupling layer (from the data to the base).
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
  layers?: number
  hidden?: readonly number[]
  /** Adam updates (default 2000), step size (default 1e-3), rows per step (default 128). */
  steps?: number
  stepSize?: number
  batchSize?: number
  seed?: number | string
  checkpoints?: number
  /** Samples drawn at each checkpoint (default 1000) and grid cells per side (default 48). */
  samples?: number
  grid?: number
}

/** One checkpoint. */
export interface RealNvpCheckpoint {
  step: number
  /** Model density p(x) on the grid, row-major in (y, x). */
  density: Float64Array
  /** Samples from fixed base draws, [samples × 2]. */
  samples: Float64Array
  /** 500 data points (an even stride) after each layer, [layers + 1][500 × 2]. */
  layers: Float64Array[]
}

/** A run so far. */
export interface RealNvpRun {
  steps: number
  done: number
  finished: boolean
  box: number
  gridX: Float64Array
  gridY: Float64Array
  /** The true density on the grid when the data's density is known; else null. */
  trueDensity: Float64Array | null
  data: Float64Array
  labels: Int32Array
  /** The labels of the 500 points followed through the layers (an even stride through the data). */
  layerLabels: Int32Array
  /** Negative log-likelihood per point on (the first 500 points of) the training data, by step; and the true density's (entropy), if known. */
  nll: { step: number[]; value: number[] }
  trueNll: number
  checkpoints: RealNvpCheckpoint[]
}

/** Train RealNVP on 2-d data { x, y?, meta? } and yield snapshots. Deterministic in `seed`. */
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
