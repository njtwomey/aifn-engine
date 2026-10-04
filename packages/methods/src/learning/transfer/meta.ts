/**
 * Model-agnostic meta-learning (MAML; Finn, Abbeel and Levine, 2017) on sinusoid regression: learn an initialisation
 * θ from which a few gradient steps on K points of a new task fit it. For each task in a meta-batch, the inner step
 * adapts θ′ = θ − α ∇_θ L_support(θ); the meta-objective is the query loss at θ′, averaged over tasks, minimised over θ
 * by Adam. Its gradient runs through the inner gradient step, so it holds second derivatives (taken here by nested
 * autodiff); first-order MAML drops them by treating ∇_θ L_support as a constant. The baseline pretrains one network
 * on all tasks together, whose best guess for an unseen task is the average function (near 0), and fine-tunes it
 * the same way.
 */

import { stopGradient, valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { treeZip, type Params } from 'aifn-compute/foundation/pytree'
import { child, stream, units, type Stream } from 'aifn-compute/foundation/random'
import {
  fromData,
  mean,
  mul,
  square,
  sub,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { adamTrainer } from './shared'

/** A sinusoid regression task y = A sin(x − φ), A ∈ [0.1, 5], φ ∈ [0, π] (Finn, Abbeel and Levine, 2017). */
export interface SineTask {
  readonly amplitude: number
  readonly phase: number
}

/** `count` sinusoid tasks drawn uniformly. */
export function sineTasks(s: Stream, count: number): SineTask[] {
  const u = units(s, 2 * count)
  return Array.from({ length: count }, (_, i) => ({ amplitude: 0.1 + 4.9 * u[2 * i], phase: Math.PI * u[2 * i + 1] }))
}

/** k points of a sinusoid task with x uniform on [−5, 5]. */
export function sineSamples(task: SineTask, s: Stream, k: number): { x: Tensor; y: Tensor } {
  const u = units(s, k)
  const x = Float64Array.from(u, (v) => -5 + 10 * v)
  return {
    x: fromData(x, [k, 1]),
    y: fromData(
      Float64Array.from(x, (v) => task.amplitude * Math.sin(v - task.phase)),
      [k, 1],
    ),
  }
}

/** Mean squared error of the network at parameters p on (x, y). */
// Inputs on [−5, 5] are scaled to [−1, 1] before the network, which keeps the first gradients (and the inner step) tame.
const forward = (net: Layer<Params[]>, p: Params[], x: Tensor): Value => net.apply(p, mul(0.2, x))
const mse = (net: Layer<Params[]>, p: Params[], x: Tensor, y: Tensor): Value => mean(square(sub(forward(net, p, x), y)))

/** θ − α ∇L(θ) on (x, y), keeping (second order) or cutting (first order) the gradient's dependence on θ. */
function adapt(net: Layer<Params[]>, p: Params[], x: Tensor, y: Tensor, alpha: number, firstOrder: boolean): Params[] {
  const { grad } = valueAndGrad((q: Params[]): Value => mse(net, q, x, y), {})(p)
  return treeZip([p, grad as Params[]], ([w, g]) =>
    sub(w as Value, mul(alpha, firstOrder ? stopGradient(g as Value) : (g as Value))),
  ) as Params[]
}

/** Options of `mamlRun`. */
export interface MamlOptions {
  /** `second` (default) or `first`-order MAML. */
  order?: 'first' | 'second'
  /** Meta-updates (default 1000), tasks per meta-batch (default 8), support and query points per task (default 10). */
  steps?: number
  tasksPerBatch?: number
  shots?: number
  /** Inner step size α (default 0.01) and outer Adam step size (default 1e-3). */
  innerRate?: number
  outerRate?: number
  hidden?: readonly number[]
  seed?: number | string
  /** Adaptation steps shown at checkpoints (default 10). */
  adaptationSteps?: number
  checkpoints?: number
}

/** One checkpoint: the fixed test task's fits after 0, 1 and `adaptationSteps` steps, for MAML and the baseline. */
export interface MamlCheckpoint {
  step: number
  maml: Float64Array[]
  pretrained: Float64Array[]
  /** Query MSE on 20 held-out tasks after each adaptation step 0 … adaptationSteps. */
  mamlCurve: Float64Array
  pretrainedCurve: Float64Array
}

/** A MAML run so far. */
export interface MamlRun {
  order: 'first' | 'second'
  steps: number
  done: number
  finished: boolean
  grid: Float64Array
  testTask: SineTask
  support: { x: Float64Array; y: Float64Array }
  truth: Float64Array
  /** The meta-loss (query MSE after one inner step) and the baseline's training MSE, per recorded step. */
  history: { step: number[]; meta: number[]; pretrained: number[] }
  checkpoints: MamlCheckpoint[]
}

/** Meta-train on sinusoids and yield snapshots (module docs). Deterministic in `seed`. */
export function* mamlRun(options: MamlOptions = {}): Generator<MamlRun, MamlRun> {
  const { order = 'second', steps = 1000, tasksPerBatch = 8, shots = 10, innerRate = 0.01, outerRate = 1e-3 } = options
  const { hidden = [40, 40], seed = 0, adaptationSteps = 10, checkpoints = 20 } = options
  const firstOrder = order === 'first'
  const net = Mlp([1, ...hidden, 1], { activation: 'relu' })
  const root = stream(seed)
  let theta: Params[] = net.init(child(root, 'maml'))
  let base: Params[] = net.init(child(root, 'pretrained'))
  const meta = adamTrainer<Params[]>(outerRate, theta)
  const joint = adamTrainer<Params[]>(outerRate, base)
  const testTask = sineTasks(child(root, 'test task'), 1)[0]
  const support = sineSamples(testTask, child(root, 'test support'), shots)
  const grid = Float64Array.from({ length: 101 }, (_, i) => -5 + i / 10)
  const gridX = fromData(grid, [101, 1])
  const truth = Float64Array.from(grid, (x) => testTask.amplitude * Math.sin(x - testTask.phase))
  const heldOut = sineTasks(child(root, 'held out'), 20).map((task, i) => ({
    support: sineSamples(task, child(root, 'held support', i), shots),
    query: sineSamples(task, child(root, 'held query', i), 50),
  }))
  const scalar = (v: Value) => {
    const u = unwrap(v)
    return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
  }
  const predict = (p: Params[]) => Float64Array.from(toFlat(unwrap(forward(net, p, gridX)) as Tensor))
  const fits = (p0: Params[]) => {
    const out = [predict(p0)]
    let p = p0
    for (let k = 1; k <= adaptationSteps; k++) {
      p = adapt(net, p, support.x, support.y, innerRate, true)
      if (k === 1 || k === adaptationSteps) out.push(predict(p))
    }
    return out
  }
  const curve = (p0: Params[]) => {
    const c = new Float64Array(adaptationSteps + 1)
    for (const h of heldOut) {
      let p = p0
      for (let k = 0; k <= adaptationSteps; k++) {
        c[k] += scalar(mse(net, p, h.query.x, h.query.y)) / heldOut.length
        if (k < adaptationSteps) p = adapt(net, p, h.support.x, h.support.y, innerRate, true)
      }
    }
    return c
  }
  const history = { step: [] as number[], meta: [] as number[], pretrained: [] as number[] }
  const shots_: MamlCheckpoint[] = []
  const checkpoint = (t: number): MamlCheckpoint => ({
    step: t,
    maml: fits(theta),
    pretrained: fits(base),
    mamlCurve: curve(theta),
    pretrainedCurve: curve(base),
  })
  const snapshot = (done: number, finished: boolean): MamlRun => ({
    order,
    steps,
    done,
    finished,
    grid,
    testTask,
    support: { x: Float64Array.from(toFlat(support.x)), y: Float64Array.from(toFlat(support.y)) },
    truth,
    history: { step: [...history.step], meta: [...history.meta], pretrained: [...history.pretrained] },
    checkpoints: shots_.slice(),
  })
  shots_.push(checkpoint(0))
  yield snapshot(0, false)
  const every = Math.max(1, Math.round(steps / checkpoints))
  const recordEvery = Math.max(1, Math.floor(steps / 100))
  const chunk = Math.max(1, Math.ceil(steps / 20))
  for (let t = 1; t <= steps; t++) {
    const tasks = sineTasks(child(root, 'tasks', t), tasksPerBatch)
    const batches = tasks.map((task, i) => ({
      support: sineSamples(task, child(root, 'support', t, i), shots),
      query: sineSamples(task, child(root, 'query', t, i), shots),
    }))
    let metaLoss = 0
    theta = meta.step(theta, (p) => {
      let total: Value = 0
      for (const b of batches) {
        const adapted = adapt(net, p, b.support.x, b.support.y, innerRate, firstOrder)
        total = sub(total, mul(-1 / batches.length, mse(net, adapted, b.query.x, b.query.y)))
      }
      metaLoss = scalar(total)
      return total
    })
    // The baseline sees the same points as one regression problem.
    let jointLoss = 0
    base = joint.step(base, (p) => {
      let total: Value = 0
      for (const b of batches) total = sub(total, mul(-1 / batches.length, mse(net, p, b.query.x, b.query.y)))
      jointLoss = scalar(total)
      return total
    })
    if (t % recordEvery === 0 || t === steps) {
      history.step.push(t)
      history.meta.push(metaLoss)
      history.pretrained.push(jointLoss)
    }
    if (t % every === 0 || t === steps) shots_.push(checkpoint(t))
    if (t === steps) {
      const last = snapshot(t, true)
      yield last
      return last
    }
    if (t % chunk === 0) yield snapshot(t, false)
  }
  return snapshot(steps, true)
}
