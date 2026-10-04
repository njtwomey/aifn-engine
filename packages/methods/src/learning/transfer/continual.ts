/**
 * Continual learning on a sequence of classification tasks with one network (an MLP), trained on each task in turn:
 *
 * - **naive** fine-tuning, which forgets earlier tasks (catastrophic forgetting);
 * - **elastic weight consolidation** (EWC; Kirkpatrick et al., 2017): after task t, the diagonal of the empirical
 *   Fisher information Fᵢ = mean over the task of (∂ log p(y | x)/∂θᵢ)² measures how much each weight mattered, and later
 *   tasks add (λ/2) Σᵢ Fᵢ (θᵢ − θ*ᵢ)², a quadratic anchor to the old solution θ* (the Fishers and anchors of all earlier
 *   tasks are summed);
 * - **experience replay**: a small memory of examples from earlier tasks is mixed into every minibatch.
 *
 * The run reports the accuracy on every task as training proceeds, so forgetting shows as falling curves.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { treeLeaves, treeZip, type Params } from 'aifn-compute/foundation/pytree'
import { child, integers, permutation, stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  fromData,
  mul,
  square,
  sub,
  sum,
  take,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { softmax } from 'aifn-compute/numerics/special'
import { Mlp } from 'aifn-compute/nn/layers'
import { adamTrainer } from './shared'

/** The strategy of `continualRun`. */
export type ContinualMethod = 'naive' | 'ewc' | 'replay'

/** Options of `continualRun`. */
export interface ContinualOptions {
  method?: ContinualMethod
  /** EWC's λ (default 1000). */
  lambda?: number
  /** Replay memory per earlier task (default 20 examples). */
  memory?: number
  /** Adam updates per task (default 400), step size (default 1e-2), rows per step (default 32). */
  stepsPerTask?: number
  stepSize?: number
  batchSize?: number
  hidden?: readonly number[]
  /** Examples used to estimate each task's Fisher (default 100). */
  fisherSamples?: number
  seed?: number | string
  /** Grid of the decision fields (default 30 cells on [−box, box]², box 4). */
  grid?: number
  box?: number
}

/** A continual run so far. */
export interface ContinualRun {
  method: ContinualMethod
  tasks: number
  stepsPerTask: number
  /** Accuracy on each task (rows) after each recorded step (columns), [tasks][records]. */
  accuracy: number[][]
  steps: number[]
  /** The probability field of class 1 on the grid after each task, [tasks][grid²]. */
  fields: Float64Array[]
  gridX: Float64Array
  done: boolean
}

type Task = { x: Tensor; y: Tensor }

/** Train on the tasks in order with the chosen strategy, yielding after each task (module docs). */
export function* continualRun(
  tasks: readonly Task[],
  options: ContinualOptions = {},
): Generator<ContinualRun, ContinualRun> {
  const { method = 'ewc', lambda = 1000, memory = 20, stepsPerTask = 400, stepSize = 1e-2, batchSize = 32 } = options
  const { hidden = [32, 32], fisherSamples = 100, seed = 0, grid = 30, box = 4 } = options
  const net = Mlp([2, ...hidden, 2], { activation: 'tanh' })
  const root = stream(seed)
  let params: Params[] = net.init(child(root, 'init'))
  const trainer = adamTrainer<Params[]>(stepSize, params)
  const labels = tasks.map((t) => Int32Array.from(toFlat(t.y)))
  const accuracyOn = (p: Params[], k: number) => {
    const pr = toFlat(unwrap(softmax(net.apply(p, tasks[k].x))) as Tensor)
    let right = 0
    for (let i = 0; i < labels[k].length; i++) if ((pr[2 * i + 1] > pr[2 * i] ? 1 : 0) === labels[k][i]) right++
    return right / labels[k].length
  }
  const gx = Float64Array.from({ length: grid }, (_, i) => -box + (2 * box * i) / (grid - 1))
  const gridPoints = fromData(
    Float64Array.from({ length: grid * grid * 2 }, (_, k) =>
      k % 2 === 0 ? gx[(k >> 1) % grid] : gx[Math.floor((k >> 1) / grid)],
    ),
    [grid * grid, 2],
  )
  const run: ContinualRun = {
    method,
    tasks: tasks.length,
    stepsPerTask,
    accuracy: tasks.map(() => []),
    steps: [],
    fields: [],
    gridX: gx,
    done: false,
  }
  const record = (step: number) => {
    run.steps.push(step)
    tasks.forEach((_, k) => run.accuracy[k].push(accuracyOn(params, k)))
  }
  record(0)
  // EWC's anchors: the summed Fisher diagonal and Fisher-weighted anchor, as parameter pytrees.
  const anchors: { fisher: Params[]; theta: Params[] }[] = []
  const memoryX: Tensor[] = []
  const memoryY: Int32Array[] = []
  const recordEvery = Math.max(1, Math.floor(stepsPerTask / 20))
  for (let k = 0; k < tasks.length; k++) {
    const { x } = tasks[k]
    const n = x.shape[0]
    for (let t = 1; t <= stepsPerTask; t++) {
      const ids = Array.from(toFlat(integers(child(root, 'batch', k, t), n, { shape: [Math.min(batchSize, n)] })))
      let xb = unwrap(take(x, ids)) as Tensor
      let yb = Int32Array.from(ids, (i) => labels[k][i])
      if (method === 'replay' && memoryX.length > 0) {
        xb = concat([xb, ...memoryX], 0) as Tensor
        yb = Int32Array.from([...yb, ...memoryY.flatMap((m) => Array.from(m))])
      }
      const target = fromData(yb, [yb.length])
      params = trainer.step(params, (p) => {
        let loss: Value = softmaxCrossEntropy(net.apply(p, xb), target, { reduction: 'mean' })
        if (method === 'ewc')
          for (const a of anchors) {
            const penalty = treeLeaves(
              treeZip([p, a.theta, a.fisher], ([q, th, f]) => mul(f as Tensor, square(sub(q as Value, th as Tensor)))),
            )
            for (const leaf of penalty) loss = add(loss, mul(lambda / 2, sum(leaf.value)))
          }
        return loss
      })
      if (t % recordEvery === 0) record(k * stepsPerTask + t)
    }
    if (method === 'ewc') {
      // The empirical Fisher diagonal: mean squared per-example gradients of log p(y | x).
      const order = Array.from(toFlat(permutation(child(root, 'fisher', k), n))).slice(0, Math.min(fisherSamples, n))
      let fisher: Params[] | null = null
      for (const i of order) {
        const xi = unwrap(take(x, [i])) as Tensor
        const yi = fromData(Int32Array.of(labels[k][i]), [1])
        const { grad } = valueAndGrad(
          (p: Params[]): Value => softmaxCrossEntropy(net.apply(p, xi), yi, { reduction: 'sum' }),
          {},
        )(params)
        const sq = treeZip([grad as Params[]], ([g]) => mul(1 / order.length, square(g as Tensor))) as Params[]
        fisher = fisher === null ? sq : (treeZip([fisher, sq], ([a, b]) => add(a as Tensor, b as Tensor)) as Params[])
      }
      anchors.push({ fisher: fisher!, theta: params })
    }
    if (method === 'replay') {
      const keep = Array.from(toFlat(permutation(child(root, 'memory', k), n))).slice(0, Math.min(memory, n))
      memoryX.push(unwrap(take(x, keep)) as Tensor)
      memoryY.push(Int32Array.from(keep, (i) => labels[k][i]))
    }
    const pr = toFlat(unwrap(softmax(net.apply(params, gridPoints))) as Tensor)
    run.fields.push(Float64Array.from({ length: grid * grid }, (_, i) => pr[2 * i + 1]))
    run.done = k === tasks.length - 1
    yield { ...run, accuracy: run.accuracy.map((r) => [...r]), steps: [...run.steps], fields: run.fields.slice() }
  }
  return run
}
