/**
 * Streamed training of the neural ODE family on small problems, for a worker: a classifier on 2-d points (nested
 * circles, moons, …) or a regressor on the reflection g(x) = −x (Dupont et al., 2019, §3), with the trajectories, the
 * field and the decision function at checkpoints, the work of every iteration (function evaluations forward and
 * backward, wall time) and, at checkpoints, the adjoint gradient against backprop's.
 */

import type { OdeFlowOptions, OdeSolveInfo } from 'aifn-compute/dynamics/ode'
import { traceProbe } from 'aifn-compute/dynamics/ode'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { treeLeaves, type Params } from 'aifn-compute/foundation/pytree'
import { child, stream, units, type Stream } from 'aifn-compute/foundation/random'
import { add, fromData, mean, mul, take, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { softmax } from 'aifn-compute/numerics/special'
import { meanSquaredErrorLoss, softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { odeModel, type OdeModel, type OdeModelKind, type OdeModelParams } from './models'
import { boxOf, flatOf, now, planeGrid, scalarOf, spaced, standardise, workCounter } from './shared'

/** The data of a run: points x [n, d] and integer labels y [n] (classification) or targets y [n, d] (regression). */
export type OdeRunData = { x: Tensor; y?: Tensor }

/**
 * The reflection g(x) = −x on [−1, 1] (Dupont et al., 2019, §3): a 1-d neural ODE cannot fit it, since its flow is
 * an increasing map of the line (trajectories cannot cross), while one extra dimension lets x rotate past −x.
 */
export function reflectionData(n = 40): OdeRunData {
  const xs = spaced(-1, 1, n)
  return {
    x: fromData(Float64Array.from(xs), [n, 1]),
    y: fromData(
      Float64Array.from(xs, (v) => -v),
      [n, 1],
    ),
  }
}

/**
 * A disc inside a ring (Dupont et al., 2019, §3, the concentric-spheres function in 2-d): points uniform on the disc
 * r ≤ r₁ (label 0) and on the annulus r₂ ≤ r ≤ r₃ (label 1). The classes are filled regions, so a flow of the plane
 * (a homeomorphism) cannot make them linearly separable: a plain 2-d neural ODE can only squeeze the ring through
 * itself approximately, while one extra dimension lifts the disc out.
 */
export function discInRing(
  s: Stream,
  n = 400,
  [r1, r2, r3]: readonly [number, number, number] = [0.5, 1, 1.5],
): OdeRunData {
  const u = units(s, 2 * n)
  const x = new Float64Array(2 * n)
  const y = new Int32Array(n)
  const inner = Math.round((n * (r1 * r1)) / (r1 * r1 + r3 * r3 - r2 * r2))
  for (let i = 0; i < n; i++) {
    const outer = i >= inner
    // Uniform over area: r² uniform between the radii squared.
    const [a, b] = outer ? [r2, r3] : [0, r1]
    const r = Math.sqrt(a * a + u[2 * i] * (b * b - a * a))
    const phi = 2 * Math.PI * u[2 * i + 1]
    x[2 * i] = r * Math.cos(phi)
    x[2 * i + 1] = r * Math.sin(phi)
    y[i] = outer ? 1 : 0
  }
  return { x: fromData(x, [n, 2]), y: fromData(y, [n]) }
}

/** Options of {@link odeRun}; plain data, so a worker task can carry them. */
export type OdeRunOptions = {
  kind?: OdeModelKind
  task?: 'classification' | 'regression'
  /** The solver and gradient method (ignored by the ResNet). Default RK4, step 0.1, backprop. */
  solver?: OdeFlowOptions
  hidden?: number
  augment?: number
  timeDependent?: boolean
  /** Blocks of the ResNet. Default 10. */
  depth?: number
  /** Weights of the kinetic-energy and Jacobian-Frobenius regularisers (RNODE). Default 0. */
  kinetic?: number
  jacobian?: number
  steps?: number
  batchSize?: number
  learningRate?: number
  /** Rescale gradients whose global norm exceeds this (keeps late Adam steps from spiking). Default 1. */
  clipNorm?: number
  seed?: number
  /** Times at which trajectories are sampled on [0, 1]. Default 21. */
  frames?: number
  /** Points whose trajectories are drawn. Default 160. */
  shown?: number
  /** Side of the field’s quiver grid. Default 11. */
  fieldGrid?: number
  /** Side of the decision raster. Default 36. */
  decisionGrid?: number
  /** Checkpoints over the run (from step 0). Default 16. */
  checkpoints?: number
  /** Compare the adjoint gradient with backprop's at checkpoints. Default true. */
  compareGradients?: boolean
}

/** The adjoint against backprop at a checkpoint, on a fixed batch. */
export type GradientComparison = {
  /** ‖g_adjoint − g_backprop‖ / ‖g_backprop‖. */
  relativeError: number
  /** The largest |x₀ reconstructed by the backward solve − x₀ kept|, over the batch. */
  reconstructionError: number
  /** Function evaluations of each gradient: backprop's forward (its backward replays them); the adjoint's both ways. */
  backpropEvaluations: number
  adjointForward: number
  adjointBackward: number
  /** Floats held for the backward pass (an estimate): every recorded evaluation's inputs and activations for
   * backprop; the checkpointed states and one evaluation for the adjoint. */
  backpropMemory: number
  adjointMemory: number
}

/** A checkpoint of a run. */
export type OdeCheckpoint = {
  step: number
  /** Loss on the whole set (without regularisers) and accuracy (classification) or mean squared error. */
  loss: number
  metric: number
  /** States of the shown points at each frame time: [frames × shown × stateDim], row-major. */
  paths: Float64Array
  /** The field on the quiver grid at each frame time (one frame when autonomous): [frames][g² × 2], or null. */
  field: Float64Array[] | null
  /** P(class 1) on the decision grid ([g²], row-major, y outer), or the prediction on `curveX` (regression). */
  decision: Float64Array | null
  /** Forward evaluations of one solve of the shown points (0 for a ResNet: it takes `depth` blocks). */
  evaluations: number
  gradient: GradientComparison | null
}

/** A snapshot of a run. */
export type OdeRun = {
  kind: OdeModelKind
  task: 'classification' | 'regression'
  gradientMethod: 'backprop' | 'adjoint'
  steps: number
  done: number
  finished: boolean
  error: string | null
  dim: number
  stateDim: number
  classes: number
  /** Standardised data [n × dim] and labels or targets. */
  data: Float64Array
  labels: Int32Array
  targets: Float64Array
  /** Indices of the shown points. */
  shown: Int32Array
  /** The frame times on [0, 1]. */
  times: Float64Array
  /** Half-width of the plotting square (2-d) or of the x range (1-d). */
  box: number
  /** The quiver grid's axis (plane: both axes; 1-d state: x, with `fieldTimes` as the other axis). */
  fieldAxis: Float64Array
  fieldTimes: Float64Array
  /** The decision grid's axis (2-d data) or the regression curve's x. */
  decisionAxis: Float64Array
  /** Per iteration: minibatch loss (with regularisers), evaluations forward and backward, wall milliseconds. */
  loss: Float64Array
  nfeForward: Float64Array
  nfeBackward: Float64Array
  wallMs: Float64Array
  checkpoints: OdeCheckpoint[]
}

/**
 * Train a member of the neural ODE family on `data` and yield snapshots (about every twentieth of the run and at the
 * end); checkpoints are evenly spaced from step 0. Deterministic in `seed` (wall times aside).
 */
export function* odeRun(data: OdeRunData, options: OdeRunOptions = {}): Generator<OdeRun, OdeRun> {
  const {
    kind = 'node',
    task = data.y && data.y.shape.length === 1 ? 'classification' : 'regression',
    hidden = 32,
    augment,
    timeDependent = false,
    depth = 10,
    kinetic = 0,
    jacobian = 0,
    steps = 300,
    batchSize = 128,
    learningRate = 0.01,
    clipNorm = 1,
    seed = 0,
    frames = 21,
    shown: shownCount = 160,
    fieldGrid = 11,
    decisionGrid = 36,
    checkpoints = 16,
    compareGradients = true,
  } = options
  const solver: OdeFlowOptions = { method: 'rk4', stepSize: 0.1, gradient: 'backprop', ...options.solver }
  const gradientMethod = solver.gradient ?? 'backprop'
  const counter = workCounter()
  // Regression keeps its scale (g(x) = −x on [−1, 1]); points are standardised.
  const x = task === 'regression' ? data.x : standardise(data.x)
  const [n, dim] = x.shape
  const labelsRaw = task === 'classification' && data.y ? Int32Array.from(toFlat(data.y)) : new Int32Array(n)
  const classes =
    task === 'classification'
      ? Math.max(
          2,
          labelsRaw.reduce((m, v) => Math.max(m, v + 1), 0),
        )
      : 0
  const targets = task === 'regression' && data.y ? Float64Array.from(toFlat(data.y)) : new Float64Array(0)
  const y: Tensor = task === 'classification' ? fromData(labelsRaw, [n]) : fromData(targets, [n, targets.length / n])

  const model: OdeModel = odeModel({
    kind,
    dim,
    classes,
    hidden,
    augment,
    timeDependent,
    depth,
    solver: { ...solver, onSolve: counter.onSolve },
  })
  const regularised = (kinetic > 0 || jacobian > 0) && model.block !== null
  const taskLoss = (params: OdeModelParams, zT: Value, target: Value): Value =>
    task === 'classification'
      ? softmaxCrossEntropy(model.readout(params, zT), target as Tensor)
      : meanSquaredErrorLoss(model.readout(params, zT), target as Tensor)

  const loss = (params: OdeModelParams, batch: { x: Tensor; y: Tensor }, ctx: { stream?: unknown }): Value => {
    if (!regularised) return taskLoss(params, model.flow(params, batch.x, [0, 1])[1], batch.y)
    const z0 = model.lift(batch.x)
    const probe = traceProbe(ctx.stream as never, [batch.x.shape[0], model.stateDim])
    const parts = model.block!.flowAugmented(params.field, z0, [0, 1], {
      kinetic: true,
      jacobianFrobenius: true,
      probe,
    })[1]
    let total = taskLoss(params, parts.x, batch.y)
    if (kinetic > 0) total = add(total, mul(kinetic, mean(parts.kinetic!)))
    if (jacobian > 0) total = add(total, mul(jacobian, mean(parts.jacobianFrobenius!)))
    return total
  }

  const root = stream(seed)
  const alg = trainingLoop({
    loss: loss as never,
    data: { x, y },
    batchSize: Math.min(batchSize, n),
    optimizer: adamRule({ stepSize: learningRate }) as UpdateRule<unknown>,
    clipNorm,
  })

  // What the figures show.
  const xs = Float64Array.from(toFlat(x))
  const shown = Int32Array.from(spaced(0, n - 1, Math.min(shownCount, n)).map(Math.round))
  const shownX = take(x, Array.from(shown)) as Tensor
  const times = Float64Array.from(spaced(0, 1, frames))
  const box = dim === 1 ? Math.max(1.5, boxOf(xs, 1.6)) : boxOf(xs)
  const planeField = model.stateDim === 2 && kind !== 'resnet' && kind !== 'sonode'
  const timeField = model.stateDim === 1 && kind !== 'resnet'
  // The field spans twice the data's box: trajectories leave it.
  const fieldPlane = planeField ? planeGrid(2 * box, fieldGrid) : null
  const fieldTimes = Float64Array.from(spaced(0, 1, fieldGrid))
  const fieldAxis = fieldPlane ? fieldPlane.axis : Float64Array.from(spaced(-box, box, fieldGrid))
  const decisionPlane = task === 'classification' && dim === 2 ? planeGrid(box, decisionGrid) : null
  const curveX = Float64Array.from(spaced(-box, box, 61))
  const decisionAxis = decisionPlane ? decisionPlane.axis : curveX
  const compareBatch = take(x, Array.from(spaced(0, n - 1, Math.min(64, n)).map(Math.round))) as Tensor
  const compareY = take(y, Array.from(spaced(0, n - 1, Math.min(64, n)).map(Math.round))) as Tensor

  const evaluateAll = (params: OdeModelParams) => {
    const zT = model.flow(params, x, [0, 1])[1]
    const l = scalarOf(taskLoss(params, zT, y))
    if (task === 'regression') return { loss: l, metric: l }
    const p = Float64Array.from(toFlat(softmax(model.readout(params, zT)) as Tensor))
    let correct = 0
    for (let i = 0; i < n; i++) {
      let best = 0
      for (let k = 1; k < classes; k++) if (p[i * classes + k] > p[i * classes + best]) best = k
      if (best === labelsRaw[i]) correct++
    }
    return { loss: l, metric: correct / n }
  }

  const fieldAt = (params: OdeModelParams, t: number): Float64Array => {
    if (fieldPlane) return flatOf(model.field(params, t, fieldPlane.points))
    // 1-d state: f(t, x) on the (t, x) grid, x outer.
    const out = new Float64Array(fieldGrid * fieldGrid * 2)
    for (let i = 0; i < fieldGrid; i++) {
      const f = flatOf(model.field(params, fieldTimes[i], fromData(fieldAxis, [fieldGrid, 1])))
      for (let j = 0; j < fieldGrid; j++) {
        out[2 * (j * fieldGrid + i)] = 1
        out[2 * (j * fieldGrid + i) + 1] = f[j]
      }
    }
    return out
  }

  const gradientOf = (params: OdeModelParams, gradient: 'backprop' | 'adjoint', onSolve: (i: OdeSolveInfo) => void) =>
    (
      valueAndGrad((p: unknown) =>
        taskLoss(
          p as OdeModelParams,
          model.flow(p as OdeModelParams, compareBatch, [0, 1], { gradient, onSolve })[1],
          compareY,
        ),
      ) as (p: unknown) => { grad: unknown }
    )(params).grad as Params

  const activationWidth = model.stateDim + 2 * hidden + (timeDependent ? 1 : 0)
  const compare = (params: OdeModelParams): GradientComparison | null => {
    if (!compareGradients || !model.block) return null
    let bpEvals = 0
    const gb = gradientOf(params, 'backprop', (i) => (bpEvals += i.evaluations))
    let adjF = 0
    let adjB = 0
    let reconstruction = 0
    const start = flatOf(model.lift(compareBatch))
    const ga = gradientOf(params, 'adjoint', (i) => {
      if (i.phase === 'forward') adjF += i.evaluations
      else {
        adjB += i.evaluations
        if (i.to === 0)
          for (let k = 0; k < start.length; k++) reconstruction = Math.max(reconstruction, Math.abs(i.x[k] - start[k]))
      }
    })
    let num = 0
    let den = 0
    const la = treeLeaves(ga)
    treeLeaves(gb).forEach((leaf, k) => {
      const b = flatOf(leaf.value as Value)
      const a = flatOf(la[k].value as Value)
      for (let i = 0; i < b.length; i++) {
        num += (a[i] - b[i]) ** 2
        den += b[i] ** 2
      }
    })
    const batch = compareBatch.shape[0]
    return {
      relativeError: Math.sqrt(num / (den || 1)),
      reconstructionError: reconstruction,
      backpropEvaluations: bpEvals,
      adjointForward: adjF,
      adjointBackward: adjB,
      backpropMemory: bpEvals * batch * activationWidth,
      adjointMemory: ((solver.checkpoints ?? 1) + 1) * batch * model.stateDim + batch * activationWidth,
    }
  }

  const checkpoint = (step: number, params: OdeModelParams): OdeCheckpoint => {
    counter.take('none')
    const all = evaluateAll(params)
    const before = workCounter()
    const states = model.flow(params, shownX, Array.from(times), { onSolve: before.onSolve })
    const evaluations = before.take('none').forward
    const paths = new Float64Array(frames * shown.length * model.stateDim)
    states.forEach((s, f) => paths.set(flatOf(s), f * shown.length * model.stateDim))
    const field =
      planeField || timeField
        ? timeDependent && planeField
          ? Array.from(times, (t) => fieldAt(params, t))
          : [fieldAt(params, 0)]
        : null
    let decision: Float64Array | null = null
    if (decisionPlane) {
      const zT = model.flow(params, decisionPlane.points, [0, 1])[1]
      const p = flatOf(softmax(model.readout(params, zT)))
      decision = Float64Array.from({ length: decisionGrid * decisionGrid }, (_, i) => p[i * classes + 1])
    } else if (task === 'regression') {
      const zT = model.flow(params, fromData(curveX, [curveX.length, 1]), [0, 1])[1]
      decision = flatOf(model.readout(params, zT))
    }
    const gradient = compare(params)
    counter.take('none')
    return { step, loss: all.loss, metric: all.metric, paths, field, decision, evaluations, gradient }
  }

  const every = Math.max(1, Math.round(steps / checkpoints))
  const chunk = Math.max(1, Math.ceil(steps / 20))
  const losses: number[] = []
  const nfeF: number[] = []
  const nfeB: number[] = []
  const wall: number[] = []
  const shots: OdeCheckpoint[] = []
  let error: string | null = null
  const snapshot = (done: number, finished: boolean): OdeRun => ({
    kind,
    task,
    gradientMethod,
    steps,
    done,
    finished,
    error,
    dim,
    stateDim: model.stateDim,
    classes,
    data: xs,
    labels: labelsRaw,
    targets,
    shown,
    times,
    box,
    fieldAxis,
    fieldTimes,
    decisionAxis,
    loss: Float64Array.from(losses),
    nfeForward: Float64Array.from(nfeF),
    nfeBackward: Float64Array.from(nfeB),
    wallMs: Float64Array.from(wall),
    checkpoints: shots.slice(),
  })

  const counting = kind === 'resnet' ? 'none' : gradientMethod
  let lastEnd = now()
  const iterator = live(alg, { params: model.init(child(root, 'init')) as never }, { stream: child(root, 'train') })
  for (;;) {
    type Item = { step: number; state: { params: unknown; loss: number }; stopped?: unknown }
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
    const work = counter.take(counting)
    if (kind === 'resnet') {
      work.forward = depth
      work.backward = depth
    }
    // Step 0's loss is evaluated at init: its work belongs to the first update.
    losses.push(state.loss)
    nfeF.push(work.forward)
    nfeB.push(work.backward)
    wall.push(ms)
    if (step % every === 0 || step === steps || stopped) shots.push(checkpoint(step, state.params as OdeModelParams))
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

/** Total entries of a parameter tree (for captions). */
export function parameterCount(params: Params): number {
  return treeLeaves(params).reduce((s, l) => s + flatOf(l.value as Value).length, 0)
}
