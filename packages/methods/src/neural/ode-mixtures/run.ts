/**
 * Streamed training of stochastic vector field mixtures and their baselines on small problems, for a worker
 * (Twomey, Kozłowski & Santos-Rodríguez, 2020): classification of 2-d points (moons, nested circles, XOR), the 1-d
 * failure cases of fig. 1 (crossing, splitting, scaling: an end target per start), and forecasting paths with FLoss.
 * Each checkpoint holds realised paths of the shown points (§3: frozen randomness per path), the component posterior
 * along them, the predictive mixtures on the grid, each component's mean VF, the prior $\pivec(t_0)$ over the inputs,
 * and the per-instance work of solving the realised paths with Dormand–Prince.
 */

import { treeLeaves } from 'aifn-compute/foundation/pytree'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, take, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import {
  checkLossSettings,
  interpolatePaths,
  svfmObjective,
  transportLoss,
  varianceLoss,
  type SvfmLossSettings,
} from './losses'
import {
  flatOf,
  svfm,
  type ComponentSelection,
  type FieldActivation,
  type Svfm,
  type SvfmOptions,
  type SvfmParams,
} from './model'
import { instanceWork, realisation, samplePaths } from './sampling'
import { DomainError, NumericalError } from 'aifn-compute/foundation/errors'

/** The kinds of problem a run trains on. */
export type SvfmTask = 'classification' | 'endpoint' | 'forecast'

/** The data of a run, as `classificationTask`, `endpointTask` and `walkTask` make it. */
export type SvfmRunData = {
  /** Starts $\xvec$, $[n, D]$ (for forecasting, the paths' first samples). */
  x: Tensor
  /** Class labels $[n]$ (int32: classification) or end targets $[n]$ or $[n, D]$ (endpoint). */
  y?: Tensor
  /** Sampled paths $[n, M, D]$ (forecasting). */
  paths?: Tensor
  /** The paths' $M$ sample times on $[0, 1]$ (default evenly spaced). */
  pathTimes?: readonly number[]
  /** Context $[n, C]$ (e.g. the time of day), given to every network. */
  context?: Tensor
  /** Colour groups $[n]$ (default the labels, or zeros). */
  groups?: Tensor
}

/** Options of {@link svfmRun}; plain data. */
export type SvfmRunOptions = {
  /** Default forecasting with `paths`, classification with int32 `y`, endpoint otherwise. */
  task?: SvfmTask
  /** Components $K$ (fig. 6's lattice). Default 1. */
  components?: number
  /** SVF units rather than VF units. Default false. */
  stochastic?: boolean
  /** The selection method. Default `'pick-and-stick'`. */
  selection?: ComponentSelection
  /** Zero-padded extra state dimensions. Default 0. */
  augment?: number
  /** The largest SVF variance $\tau_{\max}$. Default 0.5. */
  maxVariance?: number
  /**
   * Further architecture (`svfm` options): the $\pivec$ networks (prior kind, width, depth, activation, temperature,
   * emissions, transitions, stickiness), shared trunk, time dependence, the variance heads, the training solver.
   */
  architecture?: Omit<
    SvfmOptions,
    'dim' | 'components' | 'stochastic' | 'selection' | 'augment' | 'context' | 'classes'
  >
  /** Hidden units per layer of the component VFs. Default 32. */
  hidden?: number
  /** Hidden layers of the component VFs. Default 1. */
  layers?: number
  /** Activation of the component VFs. Default `'relu'`. */
  activation?: FieldActivation
  /** Grid intervals $T$ on $[0, 1]$. Default 5. */
  grid?: number
  /** RK4 step within the grid during training. Default $1/T$ (one step per interval). */
  stepSize?: number
  /** The losses (`svfmObjective`). Default the predictive loss alone. */
  losses?: SvfmLossSettings
  /** Optimiser steps. Default 300. */
  steps?: number
  /** Minibatch size (at most $n$). Default 50. */
  batchSize?: number
  /** Adam's step size. Default 0.01. */
  learningRate?: number
  /** Rescale gradients whose global norm exceeds this. Default 1. */
  clipNorm?: number
  /** Seed of the initialisation, the minibatches and the realised paths. Default 0. */
  seed?: number
  /** Standardise the data (centre, unit overall sd). Default true except for the 1-d endpoint tasks. */
  standardise?: boolean
  /** Points whose realised paths are drawn. Default 160. */
  shown?: number
  /** Frames per grid interval of the drawn paths. Default 4. */
  framesPerInterval?: number
  /** Side of the field grid. Default 13. */
  fieldGrid?: number
  /** Side of the 2-d input rasters (1-d data uses 61 points). Default 32. */
  decisionGrid?: number
  /** Dormand–Prince's relative tolerance at checkpoints (the absolute one is a hundredth of it). Default $10^{-4}$. */
  rtol?: number
  /** Checkpoints over the run (from step 0). Default 12. */
  checkpoints?: number
}

/** A checkpoint of a run. */
export type SvfmCheckpoint = {
  /** The optimiser step it was taken at. */
  step: number
  /** The predictive loss on the whole set. */
  predictive: number
  /** TLoss on the whole set, unweighted. */
  transport: number
  /** VLoss on the whole set, unweighted. */
  variance: number
  /** Accuracy on the whole set (classification), or NaN. */
  accuracy: number
  /** Realised paths of the $P$ shown points, $[\text{frames} \times P \times S]$. */
  paths: Float64Array
  /** $\pivec$ along them at the grid times, $[(T + 1) \times P \times K]$. */
  weights: Float64Array
  /** The component each shown point followed per interval, $[T \times P]$. */
  components: Int32Array
  /**
   * The predictive mixture of the shown points at each grid time: weight, mean and standard deviation,
   * $[(T + 1) \times P \times K \times (2D + 1)]$.
   */
  mixture: Float64Array
  /**
   * Each component's mean VF in the data's $D$ coordinates at each grid time, `fields[k][i]`: on the field grid,
   * $[g^2 \times D]$ (2-d), or along the $x$ axis, $[g]$ (1-d).
   */
  fields: Float64Array[][]
  /** $\pivec(t_0)$ over the input raster, $[g^2 \times K]$ (2-d), or along the $x$ axis (1-d); empty when $K = 1$. */
  prior: Float64Array
  /** $P(\text{class } 1)$ over the input raster (2-d classification), else empty. */
  decision: Float64Array
  /** NFE of each shown point's realised path solved alone. */
  nfe: Int32Array
  /** NFE of the shown points' realised paths solved as one batch. */
  batchNfe: number
}

/** A snapshot of a run. */
export type SvfmRun = {
  /** The task trained on. */
  task: SvfmTask
  /** The steps the run was asked for. */
  steps: number
  /** The steps taken so far. */
  done: number
  /** True on the last snapshot. */
  finished: boolean
  /** The message of the error that ended the run early (a non-finite loss among them), or null. */
  error: string | null
  /** The data's dimension $D$. */
  dim: number
  /** The state's dimension $S$. */
  stateDim: number
  /** Components $K$. */
  components: number
  /** SVF units or VF units. */
  stochastic: boolean
  /** The selection method. */
  selection: ComponentSelection
  /** Classes (at least 2) of a classification task, else 0. */
  classes: number
  /** The number of parameters of the model. */
  parameters: number
  /** Data in model coordinates, $[n \times D]$; the original is `shift` plus `scale` times it. */
  data: Float64Array
  /** The per-coordinate shift of the standardisation, $[D]$ (zeros without it). */
  shift: Float64Array
  /** The overall scale of the standardisation (1 without it). */
  scale: number
  /** Colour groups, $[n]$. */
  groups: Int32Array
  /** The shown points' indices, $[P]$. */
  shown: Int32Array
  /**
   * End targets $[n \times D]$ (endpoint) or grid targets of the shown points $[P \times (T + 1) \times D]$ (forecast),
   * in model coordinates; empty for classification.
   */
  targets: Float64Array
  /** Grid times on $[0, 1]$. */
  gridTimes: Float64Array
  /** Frame times of the realised paths on $[0, 1]$. */
  frameTimes: Float64Array
  /**
   * Half-width of the drawing box in model coordinates, centred on 0 (2-d), or of the $x$ range (1-d); it holds the
   * data and the targets.
   */
  box: number
  /** The field grid's axis (2-d: both axes; 1-d: $x$, against `gridTimes`). */
  fieldAxis: Float64Array
  /** The input raster's axis. */
  rasterAxis: Float64Array
  /** Per iteration: the minibatch objective. */
  loss: Float64Array
  /** Per iteration: wall milliseconds. */
  wallMs: Float64Array
  /** The checkpoints so far. */
  checkpoints: SvfmCheckpoint[]
  /** The trained parameters (on the finished snapshot only). */
  params: SvfmParams | null
}

/**
 * The wall clock in milliseconds: `performance.now()` where it exists, else `Date.now()`.
 *
 * @returns The time, for differences only.
 */
const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now())
/**
 * Evenly spaced values from `a` to `b`, both included.
 *
 * @param a The first value.
 * @param b The last value.
 * @param m How many values; with 1, just `a`.
 * @returns The `m` values.
 */
const spaced = (a: number, b: number, m: number): number[] =>
  Array.from({ length: m }, (_, i) => (m === 1 ? a : a + ((b - a) * i) / (m - 1)))

/**
 * Train an SVFM (or a baseline) on `data` and yield snapshots (about every twentieth of the run and at the end);
 * checkpoints are evenly spaced from step 0. The data is standardised (one shift per coordinate, one overall scale,
 * paths included) unless the task is 1-d endpoint regression, and the objective (`svfmObjective`) is minimised by Adam
 * on minibatches with the gradient's global norm clipped. Invalid loss settings throw `DomainError` before training,
 * as does FLoss without a forecasting task; an error during training, a non-finite loss among them, ends the run with a
 * finished snapshot carrying its message. Deterministic in `seed` (wall times aside).
 *
 * @param data The starts and the labels, targets or paths, with optional context and groups; see `SvfmRunData`.
 * @param options The task, the model (components, units, selection, augmentation, widths, grid, `architecture`), the
 *   losses, the optimisation and what checkpoints record; see `SvfmRunOptions`.
 * @returns A generator of `SvfmRun` snapshots; the last has `finished` set and carries the trained parameters.
 *
 * @example A single VF learns to scale its starts: the loss falls
 * const data = endpointTask({ x: tensor([[-1], [-0.5], [0.5], [1]]), y: tensor([-2, -1, 1, 2]) })
 * const options = { steps: 20, learningRate: 0.05, hidden: 8, grid: 2, checkpoints: 1, shown: 2, fieldGrid: 3 }
 * let run
 * for (const r of svfmRun(data, options)) run = r
 * print('loss at steps 0, 10, 20:', [0, 10, 20].map((k) => run.loss[k]))
 * const last = run.checkpoints.at(-1)
 * print('NFE of the shown points, alone and as one batch:', last.nfe, last.batchNfe)
 */
export function* svfmRun(data: SvfmRunData, options: SvfmRunOptions = {}): Generator<SvfmRun, SvfmRun> {
  const task: SvfmTask =
    options.task ?? (data.paths ? 'forecast' : data.y && data.y.dtype === 'int32' ? 'classification' : 'endpoint')
  const {
    components = 1,
    stochastic = false,
    selection = 'pick-and-stick',
    augment = 0,
    maxVariance = 0.5,
    hidden = 32,
    layers = 1,
    activation = 'relu',
    grid = 5,
    stepSize = 1 / grid,
    losses = {},
    steps = 300,
    batchSize = 50,
    learningRate = 0.01,
    clipNorm = 1,
    seed = 0,
    shown: shownCount = 160,
    framesPerInterval = 4,
    fieldGrid = 13,
    decisionGrid = 32,
    rtol = 1e-4,
    checkpoints = 12,
  } = options
  checkLossSettings(losses)
  if (losses.forecast && task !== 'forecast')
    throw new DomainError('svfmRun', 'svfmRun: FLoss needs a forecasting task')
  const D = data.x.shape[1]
  const n = data.x.shape[0]
  const standardise = options.standardise ?? !(task === 'endpoint' && D === 1)

  // Standardise with one shift per coordinate and one overall scale (paths included).
  const raw = Float64Array.from(toFlat(data.x))
  const shift = new Float64Array(D)
  let scale = 1
  if (standardise) {
    const pool = data.paths ? Float64Array.from(toFlat(data.paths)) : raw
    const rows = pool.length / D
    for (let i = 0; i < rows; i++) for (let j = 0; j < D; j++) shift[j] += pool[i * D + j] / rows
    let ss = 0
    for (let i = 0; i < rows; i++) for (let j = 0; j < D; j++) ss += (pool[i * D + j] - shift[j]) ** 2
    scale = Math.sqrt(ss / (rows * D)) || 1
  }
  const toModel = (a: ArrayLike<number>) => Float64Array.from(a, (v, i) => (v - shift[i % D]) / scale)
  const xs = toModel(raw)
  const x = fromData(xs, [n, D])
  const labels = task === 'classification' && data.y ? Int32Array.from(toFlat(data.y)) : new Int32Array(n)
  const classes =
    task === 'classification'
      ? Math.max(
          2,
          labels.reduce((m, v) => Math.max(m, v + 1), 0),
        )
      : 0
  const groups = data.groups ? Int32Array.from(toFlat(data.groups)) : labels
  const context = data.context ? Float64Array.from(toFlat(data.context)) : null
  const C = data.context ? data.context.shape[1] : 0

  const model: Svfm = svfm({
    dim: D,
    components,
    stochastic,
    selection,
    augment,
    context: C,
    hidden,
    layers,
    activation,
    classes,
    grid,
    stepSize,
    maxVariance,
    ...options.architecture,
  })
  const gridTimes = Float64Array.from({ length: grid + 1 }, (_, i) => i / grid)
  const S = model.stateDim
  const K = components

  // Targets on the grid (forecasting) or at the end.
  let endTargets = new Float64Array(0)
  let gridTargets: Tensor | null = null
  if (task === 'endpoint') endTargets = toModel(toFlat(data.y!))
  if (task === 'forecast') {
    const M = data.paths!.shape[1]
    const pathTimes = data.pathTimes ?? spaced(0, 1, M)
    gridTargets = interpolatePaths(toModel(toFlat(data.paths!)), pathTimes, D, Array.from(gridTimes))
  }
  const batchData: Record<string, Tensor> = { x }
  if (task === 'classification') batchData.labels = fromData(labels, [n])
  if (task === 'endpoint') batchData.targets = fromData(endTargets, [n, D])
  if (gridTargets) batchData.path = gridTargets
  if (data.context) batchData.c = data.context

  const objective = svfmObjective(model, losses)
  const loss = (params: SvfmParams, batch: Record<string, Tensor>): Value =>
    objective(params, { x: batch.x, labels: batch.labels, targets: batch.targets, path: batch.path, c: batch.c }).total

  const root = stream(seed)
  const alg = trainingLoop({
    loss: loss as never,
    data: batchData,
    batchSize: Math.min(batchSize, n),
    optimizer: adamRule({ stepSize: learningRate }) as UpdateRule<unknown>,
    clipNorm,
  })

  // What the figures show.
  const shown = Int32Array.from(spaced(0, n - 1, Math.min(shownCount, n)).map(Math.round))
  const P = shown.length
  const shownX = Float64Array.from({ length: P * D }, (_, k) => xs[shown[Math.floor(k / D)] * D + (k % D)])
  const shownC = context
    ? Float64Array.from({ length: P * C }, (_, k) => context[shown[Math.floor(k / C)] * C + (k % C)])
    : null
  const real = realisation(child(root, 'paths'), P, S)
  let m = 0
  for (const v of xs) m = Math.max(m, Math.abs(v))
  if (gridTargets) for (const v of toFlat(gridTargets)) m = Math.max(m, Math.abs(v))
  if (endTargets.length) for (const v of endTargets) m = Math.max(m, Math.abs(v))
  const box = Math.ceil(m * 1.25 * 2) / 2 || 1
  const fieldAxis = Float64Array.from(spaced(-box, box, fieldGrid))
  const rasterAxis = Float64Array.from(spaced(-box, box, D === 1 ? 61 : decisionGrid))
  const plane = (axis: Float64Array) => {
    const g = axis.length
    const pts = new Float64Array(g * g * 2)
    for (let i = 0; i < g; i++)
      for (let j = 0; j < g; j++) {
        pts[2 * (i * g + j)] = axis[j]
        pts[2 * (i * g + j) + 1] = axis[i]
      }
    return pts
  }
  const fieldPoints = D === 1 ? fieldAxis : plane(fieldAxis)
  const rasterPoints = D === 1 ? rasterAxis : plane(rasterAxis)
  const targets =
    task === 'endpoint'
      ? endTargets
      : gridTargets
        ? Float64Array.from(toFlat(take(gridTargets, Array.from(shown)) as Tensor))
        : new Float64Array(0)
  const meanContext = (rows: number): Tensor | null => {
    if (!context) return null
    // Rasters use the first shown point's context (forecasts share one start; the time of day is set per run).
    const c = Float64Array.from({ length: rows * C }, (_, k) => context[k % C])
    return fromData(c, [rows, C])
  }

  const evaluate = (params: SvfmParams) => {
    const parts = objective(params, {
      x,
      labels: batchData.labels,
      targets: batchData.targets,
      path: batchData.path,
      c: batchData.c,
    })
    let accuracy = NaN
    const p = model.propagate(params, x, data.context ?? null)
    if (task === 'classification') {
      const ll = flatOf(model.classLogLikelihoods(params, p)) // [n, K, classes]
      const lw = flatOf(p.logWeights[grid])
      let correct = 0
      for (let i = 0; i < n; i++) {
        let best = 0
        let bestP = -Infinity
        for (let c = 0; c < classes; c++) {
          let s = 0
          for (let k = 0; k < K; k++) s += Math.exp(lw[i * K + k] + ll[(i * K + k) * classes + c])
          if (s > bestP) {
            bestP = s
            best = c
          }
        }
        if (best === labels[i]) correct++
      }
      accuracy = correct / n
    }
    const num = (v: Value) => flatOf(v)[0]
    return {
      predictive: num(parts.predictive),
      transport: num(transportLoss(p)),
      variance: num(varianceLoss(p)),
      accuracy,
    }
  }

  const checkpoint = (step: number, params: SvfmParams): SvfmCheckpoint => {
    const all = evaluate(params)
    const realised = samplePaths(model, params, shownX, shownC, real, {
      mode: stochastic ? 'sample' : 'mean',
      rtol,
      atol: rtol * 1e-2,
      framesPerInterval,
    })
    const work = instanceWork(model, params, shownX, shownC, real, {
      mode: stochastic ? 'sample' : 'mean',
      rtol,
      atol: rtol * 1e-2,
    })
    // The predictive mixtures of the shown points.
    const p = model.propagate(params, fromData(shownX, [P, D]), shownC ? fromData(shownC, [P, C]) : null)
    const mixture = new Float64Array((grid + 1) * P * K * (2 * D + 1))
    for (let i = 0; i <= grid; i++) {
      const o = model.output(params, p, i)
      const w = flatOf(o.logWeights)
      const mu = flatOf(o.means)
      const ls = flatOf(o.logScales)
      for (let b = 0; b < P; b++)
        for (let k = 0; k < K; k++) {
          const at = ((i * P + b) * K + k) * (2 * D + 1)
          mixture[at] = Math.exp(w[b * K + k])
          for (let d = 0; d < D; d++) {
            mixture[at + 1 + d] = mu[(b * K + k) * D + d]
            mixture[at + 1 + D + d] = Math.exp(ls[(b * K + k) * D + d])
          }
        }
    }
    // Each component's mean VF on the field grid (2-d: the plane; 1-d: the x axis) at every grid time.
    const g = fieldPoints.length / D
    const fieldStates = new Float64Array(g * S)
    for (let r = 0; r < g; r++) for (let d = 0; d < D; d++) fieldStates[r * S + d] = fieldPoints[r * D + d]
    const fieldC = meanContext(g)
    const fields: Float64Array[][] = Array.from({ length: K }, () => [])
    for (let i = 0; i <= grid; i++) {
      const z = fromData(
        Float64Array.from({ length: K * g * S }, (_, q) => fieldStates[q % (g * S)]),
        [K, g, S],
      )
      const mean = flatOf(model.moments(params, gridTimes[i], z, fieldC).mean)
      for (let k = 0; k < K; k++) {
        const f = new Float64Array(g * D)
        for (let r = 0; r < g; r++) for (let d = 0; d < D; d++) f[r * D + d] = mean[(k * g + r) * S + d]
        fields[k].push(f)
      }
    }
    const rg = rasterPoints.length / D
    const rasterX = fromData(Float64Array.from(rasterPoints), [rg, D])
    const rasterC = meanContext(rg)
    const prior =
      K > 1
        ? Float64Array.from(flatOf(model.prior(params, model.lift(rasterX), rasterC)), Math.exp)
        : new Float64Array(0)
    let decision = new Float64Array(0)
    if (task === 'classification' && D === 2) {
      const q = model.propagate(params, rasterX, rasterC)
      const ll = flatOf(model.classLogLikelihoods(params, q))
      const lw = flatOf(q.logWeights[grid])
      decision = Float64Array.from({ length: rg }, (_, r) => {
        let s = 0
        for (let k = 0; k < K; k++) s += Math.exp(lw[r * K + k] + ll[(r * K + k) * classes + 1])
        return s
      })
    }
    return {
      step,
      ...all,
      paths: realised.paths,
      weights: realised.weights,
      components: realised.components,
      mixture,
      fields,
      prior,
      decision,
      nfe: work.perInstance,
      batchNfe: work.batch,
    }
  }

  const every = Math.max(1, Math.round(steps / checkpoints))
  const chunk = Math.max(1, Math.ceil(steps / 20))
  const losses_: number[] = []
  const wall: number[] = []
  const shots: SvfmCheckpoint[] = []
  let error: string | null = null
  let parameters = 0
  const frameTimes = Float64Array.from(
    { length: grid * framesPerInterval + 1 },
    (_, f) => f / (grid * framesPerInterval),
  )
  let latest: SvfmParams | null = null
  const snapshot = (done: number, finished: boolean): SvfmRun => ({
    task,
    steps,
    done,
    finished,
    error,
    dim: D,
    stateDim: S,
    components: K,
    stochastic,
    selection,
    classes,
    parameters,
    data: xs,
    shift,
    scale,
    groups,
    shown,
    targets,
    gridTimes,
    frameTimes,
    box,
    fieldAxis,
    rasterAxis,
    loss: Float64Array.from(losses_),
    wallMs: Float64Array.from(wall),
    checkpoints: shots.slice(),
    params: finished ? latest : null,
  })

  const init = model.init(child(root, 'init'))
  parameters = treeLeaves(init as never).reduce((s, l) => s + flatOf(l.value as Value).length, 0)
  let lastEnd = now()
  const iterator = live(alg, { params: init as never }, { stream: child(root, 'train') })
  for (;;) {
    type Item = { step: number; state: { params: unknown; loss: number }; stopped?: unknown }
    let next: IteratorResult<Item, void>
    try {
      next = iterator.next() as IteratorResult<Item, void>
      if (!next.done && !Number.isFinite(next.value.state.loss))
        throw new NumericalError('snapshot', 'the loss is not finite', 'not-finite')
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
      const last = snapshot(losses_.length, true)
      yield last
      return last
    }
    if (next.done) break
    const { step, state, stopped } = next.value
    latest = state.params as SvfmParams
    losses_.push(state.loss)
    wall.push(now() - lastEnd)
    if (step % every === 0 || step === steps || stopped) shots.push(checkpoint(step, state.params as SvfmParams))
    if (step === steps || stopped) {
      const last = snapshot(step, true)
      yield last
      return last
    }
    if (step > 0 && step % chunk === 0) yield snapshot(step, false)
    lastEnd = now()
  }
  return snapshot(steps, true)
}
