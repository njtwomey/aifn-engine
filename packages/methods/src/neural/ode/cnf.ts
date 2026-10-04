/**
 * A continuous normalising flow on 2-d points (Chen et al., 2018, §4; FFJORD, Grathwohl et al., 2019): x′ = f_θ(t, x)
 * carries the standard normal at t = 0 to the data at t = 1, and the instantaneous change of variables gives the log
 * density, log p₁(x) = log 𝒩(z(0)) + ∫₀¹ tr(∂f/∂x) dt along the trajectory through x (solved from t = 1 back to 0).
 * The trace is exact (two forward products in 2-d) or Hutchinson's estimate; the RNODE regularisers (Finlay et al.,
 * 2020) penalise the kinetic energy and the Jacobian's Frobenius norm. Streamed for a worker with samples moving
 * through the flow and the density p_t on a grid at every frame time.
 */

import { traceProbe, type OdeFlowOptions, type ProbeKind, type TraceEstimator } from 'aifn-compute/dynamics/ode'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, normals, stream, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  mean,
  mul,
  neg,
  square,
  sub,
  sum,
  take,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { Mlp, OdeBlock, type OdeBlockLayer } from 'aifn-compute/nn/layers'
import { xavierUniform } from 'aifn-compute/nn/init'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { boxOf, flatOf, now, planeGrid, spaced, standardise, workCounter } from './shared'

const LOG_2PI = Math.log(2 * Math.PI)

/** Options of {@link cnf}. */
export type CnfOptions = {
  /** Hidden width of the field's two-layer MLP f(t, x). Default 32. */
  hidden?: number
  /** Activation of the field. Default tanh. */
  activation?: 'tanh' | 'softplus'
  /** The solver and gradient method. Default RK4 with step 0.1, backprop. */
  solver?: OdeFlowOptions
}

/** A CNF on 2-d points. */
export type Cnf = {
  block: OdeBlockLayer<Params[]>
  init(s: Stream): Params[]
  /**
   * log p_t(x) for rows x [B, 2] at time t ∈ (0, 1], with the divergence `estimator` (a probe [B, 2] for Hutchinson's),
   * and the RNODE integrals from 0 to t.
   */
  logDensity(
    params: Params[],
    x: Value,
    options?: { t?: number; estimator?: TraceEstimator; probe?: Value; regularise?: boolean; solver?: OdeFlowOptions },
  ): { logDensity: Value; kinetic?: Value; jacobianFrobenius?: Value }
  /** Base points z [B, 2] carried to the given times (each in [0, 1], increasing from 0). */
  sample(params: Params[], z: Value, times: readonly number[], solver?: OdeFlowOptions): Value[]
}

/** log 𝒩(z; 0, I) per row of z [B, 2]. */
const baseLogDensity = (z: Value): Value => sub(mul(-0.5, sum(square(z), 1)), LOG_2PI)

/** A continuous normalising flow with a time-dependent MLP field (FFJORD's free-form Jacobian). */
export function cnf(options: CnfOptions = {}): Cnf {
  const { hidden = 32, activation = 'tanh', solver = {} } = options
  const block = OdeBlock(Mlp([3, hidden, hidden, 2], { activation, init: xavierUniform() }), {
    method: 'rk4',
    stepSize: 0.1,
    ...solver,
    timeDependent: true,
  })
  return {
    block,
    init: (s) => block.init(s),
    logDensity: (params, x, { t = 1, estimator = 'exact', probe, regularise = false, solver: override } = {}) => {
      const parts = block.flowAugmented(
        params,
        x,
        [t, 0],
        {
          logDensity: estimator,
          probe,
          kinetic: regularise,
          jacobianFrobenius: regularise,
        },
        override,
      )[1]
      // Δ runs from 0 at t back to 0: log p_t(x) = log p₀(z(0)) − Δ(0). Integrals taken backwards are negated.
      return {
        logDensity: sub(baseLogDensity(parts.x), parts.logDensityChange!),
        ...(regularise ? { kinetic: neg(parts.kinetic!), jacobianFrobenius: neg(parts.jacobianFrobenius!) } : {}),
      }
    },
    sample: (params, z, times, override) => block.flow(params, z, times, override),
  }
}

/** Options of {@link cnfRun}; plain data. */
export type CnfRunOptions = CnfOptions & {
  /** The trace in training: exact or Hutchinson's. Default exact. */
  estimator?: TraceEstimator
  probe?: ProbeKind
  /** Weights of the kinetic-energy and Jacobian-Frobenius regularisers. Default 0. */
  kinetic?: number
  jacobian?: number
  /** Keep only the points with this label (e.g. one colour of a checkerboard). Default all. */
  keepLabel?: number
  steps?: number
  batchSize?: number
  learningRate?: number
  /** Rescale gradients whose global norm exceeds this (keeps late Adam steps from spiking). Default 1. */
  clipNorm?: number
  seed?: number
  /** Frame times on [0, 1]. Default 11. */
  frames?: number
  /** Base samples carried through the flow. Default 600. */
  samples?: number
  /** Side of the density grid. Default 32. */
  grid?: number
  /** Side of the quiver grid. Default 11. */
  fieldGrid?: number
  checkpoints?: number
}

/** A checkpoint of a CNF run. */
export type CnfCheckpoint = {
  step: number
  /** Negative log-likelihood per point on the whole set (exact trace), nats. */
  nll: number
  /** Hutchinson's estimate of the NLL on a batch over several probes: mean and standard deviation; and exact. */
  hutchinson: { mean: number; sd: number; exact: number }
  /** Samples at each frame time: [frames × samples × 2]. */
  samples: Float64Array
  /** log p_t on the grid at each frame time: [frames][g²] (row-major, y outer). */
  density: Float64Array[]
  /** The field on the quiver grid at each frame time: [frames][g² × 2]. */
  field: Float64Array[]
  /** Forward evaluations of one solve of the samples. */
  evaluations: number
}

/** A snapshot of a CNF run. */
export type CnfRun = {
  steps: number
  done: number
  finished: boolean
  error: string | null
  estimator: TraceEstimator
  gradientMethod: 'backprop' | 'adjoint'
  data: Float64Array
  box: number
  times: Float64Array
  gridAxis: Float64Array
  fieldAxis: Float64Array
  /** Per iteration: minibatch loss (NLL plus regularisers), evaluations forward and backward, wall milliseconds. */
  loss: Float64Array
  nfeForward: Float64Array
  nfeBackward: Float64Array
  wallMs: Float64Array
  checkpoints: CnfCheckpoint[]
}

/**
 * Train a CNF on 2-d points by maximum likelihood and yield snapshots (about every twentieth of the run and at the end).
 */
export function* cnfRun(data: { x: Tensor; y?: Tensor }, options: CnfRunOptions = {}): Generator<CnfRun, CnfRun> {
  const {
    estimator = 'exact',
    probe: probeKind = 'rademacher',
    kinetic = 0,
    jacobian = 0,
    keepLabel,
    steps = 300,
    batchSize = 128,
    learningRate = 0.01,
    clipNorm = 1,
    seed = 0,
    frames = 11,
    samples: sampleCount = 600,
    grid = 32,
    fieldGrid = 11,
    checkpoints = 10,
  } = options
  const solver: OdeFlowOptions = { method: 'rk4', stepSize: 0.1, gradient: 'backprop', ...options.solver }
  const gradientMethod = solver.gradient ?? 'backprop'
  const counter = workCounter()
  let x = data.x
  if (keepLabel !== undefined && data.y) {
    const labels = toFlat(data.y)
    x = take(
      x,
      Array.from({ length: labels.length }, (_, i) => i).filter((i) => labels[i] === keepLabel),
    ) as Tensor
  }
  x = standardise(x)
  const n = x.shape[0]
  const model = cnf({ ...options, solver: { ...solver, onSolve: counter.onSolve } })
  const regularise = kinetic > 0 || jacobian > 0
  const loss = (params: Params[], batch: { x: Tensor }, ctx: { stream?: Stream }): Value => {
    const needsProbe = estimator === 'hutchinson' || regularise
    const probe = needsProbe ? traceProbe(ctx.stream!, [batch.x.shape[0], 2], probeKind) : undefined
    const r = model.logDensity(params, batch.x, { estimator, probe, regularise })
    let total = neg(mean(r.logDensity))
    if (kinetic > 0) total = add(total, mul(kinetic, mean(r.kinetic!)))
    if (jacobian > 0) total = add(total, mul(jacobian, mean(r.jacobianFrobenius!)))
    return total
  }
  const root = stream(seed)
  const alg = trainingLoop({
    loss: loss as never,
    data: { x },
    batchSize: Math.min(batchSize, n),
    optimizer: adamRule({ stepSize: learningRate }) as UpdateRule<unknown>,
    clipNorm,
  })

  const xs = Float64Array.from(toFlat(x))
  const box = Math.max(boxOf(xs, 1.15), 3)
  const times = Float64Array.from(spaced(0, 1, frames))
  const densityGrid = planeGrid(box, grid)
  const quiver = planeGrid(box, fieldGrid)
  const base = normals(child(root, 'samples'), [sampleCount, 2])
  const evalBatch = take(x, Array.from(spaced(0, n - 1, Math.min(128, n)).map(Math.round))) as Tensor

  const checkpoint = (step: number, params: Params[]): CnfCheckpoint => {
    counter.take('none')
    const nll = -flatOf(mean(model.logDensity(params, x).logDensity))[0]
    const exact = -flatOf(mean(model.logDensity(params, evalBatch).logDensity))[0]
    const estimates = Array.from({ length: 6 }, (_, k) => {
      const probe = traceProbe(child(child(root, 'hutchinson'), k), [evalBatch.shape[0], 2], probeKind)
      return -flatOf(mean(model.logDensity(params, evalBatch, { estimator: 'hutchinson', probe }).logDensity))[0]
    })
    const m = estimates.reduce((a, b) => a + b, 0) / estimates.length
    const sd = Math.sqrt(estimates.reduce((a, b) => a + (b - m) ** 2, 0) / (estimates.length - 1))
    const work = workCounter()
    const states = model.sample(params, base, Array.from(times), { onSolve: work.onSolve })
    const evaluations = work.take('none').forward
    const sampleArray = new Float64Array(frames * sampleCount * 2)
    states.forEach((s, f) => sampleArray.set(flatOf(s), f * sampleCount * 2))
    const density = Array.from(times, (t) =>
      t === 0
        ? flatOf(baseLogDensity(densityGrid.points))
        : flatOf(model.logDensity(params, densityGrid.points, { t }).logDensity),
    )
    const field = Array.from(times, (t) => flatOf(model.block.field(params, t, quiver.points)))
    counter.take('none')
    return { step, nll, hutchinson: { mean: m, sd, exact }, samples: sampleArray, density, field, evaluations }
  }

  const every = Math.max(1, Math.round(steps / checkpoints))
  const chunk = Math.max(1, Math.ceil(steps / 20))
  const losses: number[] = []
  const nfeF: number[] = []
  const nfeB: number[] = []
  const wall: number[] = []
  const shots: CnfCheckpoint[] = []
  let error: string | null = null
  const snapshot = (done: number, finished: boolean): CnfRun => ({
    steps,
    done,
    finished,
    error,
    estimator,
    gradientMethod,
    data: xs,
    box,
    times,
    gridAxis: densityGrid.axis,
    fieldAxis: quiver.axis,
    loss: Float64Array.from(losses),
    nfeForward: Float64Array.from(nfeF),
    nfeBackward: Float64Array.from(nfeB),
    wallMs: Float64Array.from(wall),
    checkpoints: shots.slice(),
  })
  type Item = { step: number; state: { params: unknown; loss: number }; stopped?: unknown }
  const iterator = live(alg, { params: model.init(child(root, 'init')) as never }, { stream: child(root, 'train') })
  let lastEnd = now()
  for (;;) {
    let next: IteratorResult<Item, void>
    try {
      next = iterator.next() as IteratorResult<Item, void>
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
      const last = snapshot(losses.length, true)
      yield last
      return last
    }
    if (next.done) break
    const { step, state, stopped } = next.value
    const ms = now() - lastEnd
    const work = counter.take(gradientMethod)
    losses.push(state.loss)
    nfeF.push(work.forward)
    nfeB.push(work.backward)
    wall.push(ms)
    if (step % every === 0 || step === steps || stopped) shots.push(checkpoint(step, state.params as Params[]))
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
