/**
 * Streamed training of stochastic vector field mixtures and their baselines on small problems, for a worker
 * (Twomey, Kozłowski & Santos-Rodríguez, 2020): classification of 2-d points (moons, nested circles, XOR), the 1-d
 * failure cases of fig. 1 (crossing, splitting, scaling: an end target per start), and forecasting paths with FLoss.
 * Each checkpoint holds realised paths of the shown points (§3: frozen randomness per path), the component posterior
 * along them, the predictive mixtures on the grid, each component's mean VF, the prior π(t₀) over the inputs, and the
 * per-instance work of solving the realised paths with Dormand–Prince.
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

/**
 * The data of a run: starts x [n, D]; class labels y [n] (classification) or end targets y [n] / [n, D] (endpoint);
 * for forecasting, sampled paths [n, M, D] at `pathTimes` [M] on [0, 1] (x is then their first sample); `context`
 * [n, C] (e.g. the time of day) and `groups` [n] (colours; labels by default).
 */
export type SvfmRunData = {
  x: Tensor
  y?: Tensor
  paths?: Tensor
  pathTimes?: readonly number[]
  context?: Tensor
  groups?: Tensor
}

/** Options of {@link svfmRun}; plain data. */
export type SvfmRunOptions = {
  task?: SvfmTask
  /** Components K, SVF units, the selection method and augmentation (fig. 6's lattice). */
  components?: number
  stochastic?: boolean
  selection?: ComponentSelection
  augment?: number
  /** The largest SVF variance τ. Default 0.5. */
  maxVariance?: number
  /**
   * Further architecture (`svfm` options): the π networks (prior kind, width, depth, activation, temperature, emissions,
   * transitions, stickiness), shared trunk, time dependence, the variance heads, the training solver.
   */
  architecture?: Omit<
    SvfmOptions,
    'dim' | 'components' | 'stochastic' | 'selection' | 'augment' | 'context' | 'classes'
  >
  hidden?: number
  layers?: number
  activation?: FieldActivation
  /** Grid intervals T on [0, 1]. Default 5. */
  grid?: number
  /** RK4 step within the grid during training. Default 0.2 (one step per interval at T = 5). */
  stepSize?: number
  losses?: SvfmLossSettings
  steps?: number
  batchSize?: number
  learningRate?: number
  clipNorm?: number
  seed?: number
  /** Standardise the data (centre, unit overall sd). Default true except for the 1-d endpoint tasks. */
  standardise?: boolean
  /** Points whose realised paths are drawn. Default 160. */
  shown?: number
  /** Frames per grid interval of the drawn paths. Default 4. */
  framesPerInterval?: number
  /** Side of the field grid and of the input rasters. Defaults 13 and 32. */
  fieldGrid?: number
  decisionGrid?: number
  /** Dormand–Prince tolerance of the per-instance work at checkpoints. Default 1e-4. */
  rtol?: number
  checkpoints?: number
}

/** A checkpoint of a run. */
export type SvfmCheckpoint = {
  step: number
  /** Losses on the whole set: predictive, TLoss, VLoss (unweighted), and accuracy (classification) or NaN. */
  predictive: number
  transport: number
  variance: number
  accuracy: number
  /** Realised paths of the shown points [frames × P × S]. */
  paths: Float64Array
  /** π along them at the grid times [(T + 1) × P × K] and the component followed per interval [T × P]. */
  weights: Float64Array
  components: Int32Array
  /** The predictive mixture of the shown points at each grid time: π, mean, sd [(T + 1) × P × K × (2D + 1)]. */
  mixture: Float64Array
  /** Each component's mean VF on the field grid at each grid time [K][T + 1][g² × S] (2-d), or over (t, x) (1-d). */
  fields: Float64Array[][]
  /** π(t₀) over the input raster [g² × K] (2-d) or the x axis (1-d); empty when K = 1. */
  prior: Float64Array
  /** P(class 1) over the input raster (classification), else empty. */
  decision: Float64Array
  /** NFE of each shown point's realised path solved alone, and of all of them as one batch. */
  nfe: Int32Array
  batchNfe: number
}

/** A snapshot of a run. */
export type SvfmRun = {
  task: SvfmTask
  steps: number
  done: number
  finished: boolean
  error: string | null
  dim: number
  stateDim: number
  components: number
  stochastic: boolean
  selection: ComponentSelection
  classes: number
  parameters: number
  /** Data in model coordinates [n × D], with the map back: original = shift + scale · model. */
  data: Float64Array
  shift: Float64Array
  scale: number
  /** Colour groups [n] and the shown points' indices [P]. */
  groups: Int32Array
  shown: Int32Array
  /** End targets [n × D] (endpoint) or grid targets of the shown points [P × (T + 1) × D] (forecast). */
  targets: Float64Array
  /** Grid times and frame times on [0, 1]. */
  gridTimes: Float64Array
  frameTimes: Float64Array
  /** Half-width of the drawing box in model coordinates (2-d), or of the x range (1-d), and its centre. */
  box: number
  /** The field grid's axis (2-d: both axes; 1-d: x, against `gridTimes`) and the raster's axis. */
  fieldAxis: Float64Array
  rasterAxis: Float64Array
  /** Per iteration: the minibatch objective and wall milliseconds. */
  loss: Float64Array
  wallMs: Float64Array
  checkpoints: SvfmCheckpoint[]
  /** The trained parameters (on the finished snapshot only). */
  params: SvfmParams | null
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now())
const spaced = (a: number, b: number, m: number): number[] =>
  Array.from({ length: m }, (_, i) => (m === 1 ? a : a + ((b - a) * i) / (m - 1)))

/**
 * Train an SVFM (or a baseline) on `data` and yield snapshots (about every twentieth of the run and at the end);
 * checkpoints are evenly spaced from step 0. Deterministic in `seed` (wall times aside).
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
