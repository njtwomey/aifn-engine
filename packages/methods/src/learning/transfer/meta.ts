/**
 * Model-agnostic meta-learning (MAML; Finn, Abbeel and Levine, 2017) on sinusoid regression: learn an initialisation
 * $\thetavec$ from which a few gradient steps on $K$ points of a new task fit it. For each task in a meta-batch, the
 * inner step adapts $\thetavec' = \thetavec - \alpha \nabla_{\thetavec} L_{\mathrm{support}}(\thetavec)$; the
 * meta-objective is the query loss at $\thetavec'$, averaged over tasks, minimised over $\thetavec$ by Adam. Its
 * gradient runs through the inner gradient step, so it holds second derivatives (taken here by nested autodiff);
 * first-order MAML drops them by treating $\nabla_{\thetavec} L_{\mathrm{support}}$ as a constant. The baseline
 * pretrains one network on all tasks together (on the same meta-batches' query points), whose best guess for an
 * unseen task is the average function (near 0), and fine-tunes it the same way.
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

/**
 * A sinusoid regression task $y = A \sin(x - \varphi)$, $A \in [0.1, 5]$, $\varphi \in [0, \pi]$ (Finn, Abbeel and
 * Levine, 2017).
 */
export interface SineTask {
  /** The amplitude $A$. */
  readonly amplitude: number
  /** The phase $\varphi$, radians. */
  readonly phase: number
}

/**
 * Sinusoid tasks with amplitude uniform on $[0.1, 5]$ and phase uniform on $[0, \pi]$.
 *
 * @param s The stream the tasks are drawn from.
 * @param count The number of tasks.
 * @returns The tasks.
 *
 * @example Two tasks
 * print(sineTasks(stream(1), 2))
 */
export function sineTasks(s: Stream, count: number): SineTask[] {
  const u = units(s, 2 * count)
  return Array.from({ length: count }, (_, i) => ({ amplitude: 0.1 + 4.9 * u[2 * i], phase: Math.PI * u[2 * i + 1] }))
}

/**
 * Points of a sinusoid task, noiseless, with $x$ uniform on $[-5, 5]$.
 *
 * @param task The task.
 * @param s The stream the inputs are drawn from.
 * @param k The number of points.
 * @returns Inputs `x` and targets `y`, both $k \times 1$.
 *
 * @example Three points of a task with amplitude 2 and phase 0
 * print(sineSamples({ amplitude: 2, phase: 0 }, stream(1), 3))
 */
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

/**
 * The network's output on inputs $x$. Inputs on $[-5, 5]$ are scaled to $[-1, 1]$ before the network, which keeps
 * the first gradients (and the inner step) tame.
 *
 * @param net The network.
 * @param p Its parameters.
 * @param x The inputs, $n \times 1$.
 * @returns The outputs, $n \times 1$ (differentiable in `p`).
 */
const forward = (net: Layer<Params[]>, p: Params[], x: Tensor): Value => net.apply(p, mul(0.2, x))
/**
 * Mean squared error of the network at parameters `p` on $(x, y)$ (differentiable in `p`).
 *
 * @param net The network.
 * @param p Its parameters.
 * @param x The inputs, $n \times 1$.
 * @param y The targets, $n \times 1$.
 * @returns The mean squared error, a scalar.
 */
const mse = (net: Layer<Params[]>, p: Params[], x: Tensor, y: Tensor): Value => mean(square(sub(forward(net, p, x), y)))

/**
 * One inner gradient step $\thetavec - \alpha \nabla L(\thetavec)$ of the mean squared error on $(x, y)$, keeping
 * (second order) or cutting (first order) the gradient's dependence on $\thetavec$.
 *
 * @param net The network.
 * @param p The parameters $\thetavec$ (traced when the meta-gradient is being taken).
 * @param x The support inputs, $n \times 1$.
 * @param y The support targets, $n \times 1$.
 * @param alpha The inner step size $\alpha$.
 * @param firstOrder Treat the gradient as a constant (first-order MAML).
 * @returns The adapted parameters.
 */
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
  /** Meta-updates (default 1000). */
  steps?: number
  /** Tasks per meta-batch (default 8). */
  tasksPerBatch?: number
  /** Support points, and query points, per task (default 10 each); also the test task's support. */
  shots?: number
  /** Inner step size $\alpha$ (default 0.01), also of the fine-tuning at checkpoints. */
  innerRate?: number
  /** Outer Adam step size (default 1e-3), for MAML and the baseline alike. */
  outerRate?: number
  /** Hidden widths of the network (default $[40, 40]$, ReLU). */
  hidden?: readonly number[]
  /** The root stream's seed (default 0): the run is deterministic in it. */
  seed?: number | string
  /** Adaptation steps shown at checkpoints (default 10). */
  adaptationSteps?: number
  /** About how many checkpoints to keep after the start (default 20): one every `steps / checkpoints` steps. */
  checkpoints?: number
}

/**
 * One checkpoint: the fixed test task's fits on the grid after 0, 1 and `adaptationSteps` fine-tuning steps on its
 * support points, for MAML and the baseline.
 */
export interface MamlCheckpoint {
  /** Meta-updates taken. */
  step: number
  /** MAML's fits, 101 values each. */
  maml: Float64Array[]
  /** The baseline's fits, 101 values each. */
  pretrained: Float64Array[]
  /** MAML's mean query MSE on 20 held-out tasks after each adaptation step $0, \dots,$ `adaptationSteps`. */
  mamlCurve: Float64Array
  /** The baseline's curve, likewise. */
  pretrainedCurve: Float64Array
}

/** A MAML run so far. */
export interface MamlRun {
  /** First- or second-order MAML. */
  order: 'first' | 'second'
  /** The meta-updates the run will take. */
  steps: number
  /** The meta-updates taken. */
  done: number
  /** True for the last snapshot. */
  finished: boolean
  /** The 101 inputs on $[-5, 5]$ where the fits are drawn. */
  grid: Float64Array
  /** The fixed test task. */
  testTask: SineTask
  /** The test task's support points. */
  support: { x: Float64Array; y: Float64Array }
  /** The test task's function on the grid. */
  truth: Float64Array
  /**
   * The meta-loss (query MSE after one inner step) and the baseline's training MSE, on the step's meta-batch before
   * its update, per recorded step (about 100 of them).
   */
  history: { step: number[]; meta: number[]; pretrained: number[] }
  /** The checkpoints so far, the first at step 0. */
  checkpoints: MamlCheckpoint[]
}

/**
 * Meta-train on sinusoids, beside the pretrained baseline, and yield snapshots: at the start, every `steps / 20`
 * meta-updates and at the end (the last also returned). Every task and point is drawn from the root stream of
 * `seed`, so the run is deterministic in it. Checkpoints fine-tune by plain gradient steps of size `innerRate`.
 *
 * @param options The MAML variant, the meta-training, the network and the checkpoints.
 * @returns A generator of snapshots of the run.
 *
 * @example Ten second-order meta-updates of a small network: the held-out error after 0, 1 and 2 adaptation steps
 * const options = { steps: 10, tasksPerBatch: 2, shots: 5, hidden: [8, 8], adaptationSteps: 2, checkpoints: 1 }
 * let run
 * for (const snapshot of mamlRun(options)) run = snapshot
 * const last = run.checkpoints.at(-1)
 * print('MAML:', last.mamlCurve)
 * print('pretrained baseline:', last.pretrainedCurve)
 * print('meta-loss per step:', run.history.meta)
 */
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
