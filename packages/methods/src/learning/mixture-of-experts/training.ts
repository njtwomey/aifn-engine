/**
 * Training a mixture of experts by gradient descent (Adam through aifn's autodiff, with the auxiliary losses of
 * `aifn-compute/nn/experts`), and `mixtureOfExpertsRun`, a generator that trains by EM, Adam or L-BFGS and yields
 * snapshots with the curves a figure needs: the data loss, the auxiliary losses, each expert's load, the router's
 * entropy and how well the gate's assignments recover the true regimes (adjusted Rand index), plus the parameters at
 * checkpoints for a player.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { adjustedRandIndex } from 'aifn-compute/learning/metrics'
import { routingStatistics } from 'aifn-compute/nn/experts'
import type { Context } from 'aifn-compute/nn/layers'
import { methodTraining, trainingLoop, type TrainingState } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { moeEm, type MoeData } from './em'
import {
  moeLoss,
  moeModel,
  type AuxWeights,
  type MoeConfig,
  type MoeModel,
  type MoeParams,
  type MoeSpec,
} from './model'

/** Options of `moeTraining`. */
export type MoeTrainingOptions = {
  /** Adam's step size (default 0.03). */
  stepSize?: number
  /** Rows per step (default all: full-batch). */
  batchSize?: Size
  /** Weights of the auxiliary losses (default none). */
  aux?: AuxWeights
  /** Clip the gradient's global norm (default 10). */
  clipNorm?: number
}

/**
 * Adam on the data loss plus the weighted auxiliary losses (`moeLoss`'s total), as a traceable `trainingLoop`: the
 * minibatches' shuffles and any gate noise come from the steps' streams, as `trainingLoop` describes.
 *
 * @param model The model: any experts, gate and objective.
 * @param data The training data.
 * @param options Adam's step size, the minibatch size, the auxiliary losses' weights and the gradient clipping.
 * @returns The algorithm; `init` takes `{ params }` (such as `model.init(stream)`), and the state's `loss` is the
 *   objective on the next minibatch (the whole set when full-batch).
 *
 * @example Adam fits two experts to a V of two regimes
 * const xs = Array.from({ length: 16 }, (_, t) => -1 + (2 * t) / 15)
 * const x = tensor(xs.map((v) => [v]))
 * const y = add(tensor(xs.map((v) => Math.abs(2 * v))), normals(stream(1), 16, 0, 0.1))
 * const model = moeModel({ inputs: 1, task: 'regression', experts: 2 })
 * const alg = moeTraining(model, { x, y }, { stepSize: 0.05 })
 * const s = run(alg, { params: model.init(stream(2)) }, 120, { stream: stream(3) })
 * print('loss after 120 steps:', s.loss)
 * print('gate at x = -0.5 and 0.5:', moePredict(model, s.params, tensor([[-0.5], [0.5]])).gate)
 */
export function moeTraining(
  model: MoeModel,
  data: MoeData,
  options: MoeTrainingOptions = {},
): Algorithm<{ params: MoeParams }, TrainingState<MoeParams>> {
  const { stepSize = 0.03, batchSize, aux = {}, clipNorm = 10 } = options
  return trainingLoop<MoeParams, { x: Tensor; y: Tensor }>({
    loss: (p, b, ctx) => moeLoss(model, p, b.x, b.y, aux, ctx).total,
    data: { x: data.x, y: data.y },
    batchSize,
    optimizer: adamRule({ stepSize }) as never,
    clipNorm,
  })
}

/** Options of `mixtureOfExpertsRun`. */
export type MoeRunOptions = Omit<MoeConfig, 'inputs' | 'task'> & {
  /**
   * Inputs `x` ($T \times d$), targets `y` ($T$), and optionally the true regime of each row (`regime`, $T$ integers),
   * for the agreement curve.
   */
  data: MoeData & { regime?: Tensor }
  /** Real targets or 0/1 labels. The input width is read from the data. */
  task: 'regression' | 'classification'
  /**
   * `em` (linear experts, dense gate, mixture objective), `adam` (default) or `lbfgs` (full batch, the gate without
   * noise: L-BFGS needs one deterministic objective).
   */
  method?: 'em' | 'adam' | 'lbfgs'
  /** L-BFGS's memory m (default 10). */
  memory?: Size
  /** Steps: EM iterations, Adam updates or L-BFGS iterations (default 40 for EM, 600 otherwise). */
  steps?: Size
  /** Keep the parameters, and yield a snapshot, every this many steps (default `steps` / 60 rounded, at least 1). */
  every?: Size
  /** Adam's step size (default 0.03). */
  stepSize?: number
  /** Adam's rows per step (default all). */
  batchSize?: Size
  /** Weight of the load-balancing loss (default 0). */
  balance?: number
  /** Weight of the importance loss (default 0). */
  importance?: number
  /** Weight of the router z-loss (default 0). */
  z?: number
  /** The root stream's seed (default 'moe'). */
  seed?: string | number
}

/**
 * Curves of a run, one entry per recorded step: every step of a run of fewer than 600 steps, every
 * $\lfloor \mathit{steps}/300 \rfloor$-th of a longer one, and the last step.
 */
export type MoeHistory = {
  /** The recorded steps (0 for the start). */
  step: number[]
  /** The data term on the whole training set. */
  loss: number[]
  /** The load-balancing loss on the whole training set. */
  balance: number[]
  /** The importance loss on the whole training set. */
  importance: number[]
  /** The router z-loss on the whole training set. */
  z: number[]
  /**
   * Each expert's load [step][N]: its share of the processed assignments for a sparse gate; for a dense gate, where
   * every expert processes every row, its mean gate weight.
   */
  load: number[][]
  /** Mean router entropy (nats). */
  entropy: number[]
  /** Experts that processed nothing (dense gate: that hold the largest weight for no row). */
  idle: number[]
  /** Share of assignments dropped for capacity. */
  dropped: number[]
  /** Adjusted Rand index of the gate's argmax assignments against the true regimes (NaN without them). */
  agreement: number[]
}

/** A snapshot of `mixtureOfExpertsRun`. */
export type MoeSnapshot = {
  /** Steps taken. */
  readonly step: Size
  /**
   * Steps the run will take: `steps`, or, once an Adam or L-BFGS run has stopped early (converged or diverged), the
   * step it stopped at.
   */
  readonly steps: Size
  /** True for the last snapshot. */
  readonly done: boolean
  /** The training method. */
  readonly method: 'em' | 'adam' | 'lbfgs'
  /** The model's structure. */
  readonly spec: MoeSpec
  /** The curves so far (a copy). */
  readonly history: MoeHistory
  /** Parameters at step 0, every `every` steps and at the last step. */
  readonly checkpoints: readonly { step: Size; params: MoeParams }[]
}

/**
 * A scalar value as a number.
 *
 * @param v The value (traced or not; its primal is read).
 * @returns The number.
 */
const scalar = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/**
 * Train a mixture of experts on `data` by EM, Adam or L-BFGS, yielding a snapshot every `every` steps and at the end:
 * a generator, so a worker can stream the run to a page that plots the curves and plays the checkpoints. Every draw
 * (the initial parameters, minibatches, gate noise) comes from the root stream of `seed`. A run that diverges, or an
 * L-BFGS run that converges, stops early. Throws as `moeModel` does, and as `moeEm` does for EM on a model it does
 * not apply to.
 *
 * @param options The data, the model's structure (as `MoeConfig`, less `inputs`) and the training settings.
 * @returns A generator of snapshots, the last with `done` set.
 *
 * @example Ten EM steps on a V of two regimes: the gate's assignments recover the regimes
 * const xs = Array.from({ length: 16 }, (_, t) => -1 + (2 * t) / 15)
 * const x = tensor(xs.map((v) => [v]))
 * const y = add(tensor(xs.map((v) => Math.abs(2 * v))), normals(stream(1), 16, 0, 0.1))
 * const regime = tensor(xs.map((v) => (v < 0 ? 0 : 1)))
 * const options = { data: { x, y, regime }, task: 'regression', experts: 2, method: 'em', steps: 10, every: 5 }
 * for (const snap of mixtureOfExpertsRun(options)) {
 *   print(`step ${snap.step}: loss`, snap.history.loss.at(-1), 'agreement', snap.history.agreement.at(-1))
 * }
 */
export function* mixtureOfExpertsRun(options: MoeRunOptions): Generator<MoeSnapshot> {
  const {
    data,
    task,
    method = 'adam',
    memory,
    seed = 'moe',
    stepSize,
    batchSize,
    balance = 0,
    importance = 0,
    z = 0,
    every: everyOption,
    steps: stepsOption,
    ...structure
  } = options
  const model = moeModel({ ...structure, task, inputs: data.x.shape[1] })
  const steps = stepsOption ?? (method === 'em' ? 40 : 600)
  const every = Math.max(1, everyOption ?? Math.round(steps / 60))
  const root = stream(seed)
  const regime = data.regime ? Array.from(toFlat(data.regime)) : null
  const history: MoeHistory = {
    step: [],
    loss: [],
    balance: [],
    importance: [],
    z: [],
    load: [],
    entropy: [],
    idle: [],
    dropped: [],
    agreement: [],
  }
  const checkpoints: { step: Size; params: MoeParams }[] = []
  const record = (t: Size, params: MoeParams) => {
    const parts = moeLoss(model, params, data.x, data.y)
    const stats = routingStatistics(parts.forward.routing)
    history.step.push(t)
    history.loss.push(scalar(parts.data))
    history.balance.push(scalar(parts.balance))
    history.importance.push(scalar(parts.importance))
    history.z.push(scalar(parts.z))
    const N = model.spec.experts
    const w = toFlat(unwrap(parts.forward.routing.combine) as Tensor)
    const T = data.x.shape[0]
    const assignment = Array.from({ length: T }, (_, t) => {
      const row = w.slice(t * N, (t + 1) * N)
      return row.indexOf(Math.max(...row))
    })
    const dense = parts.forward.routing.gate === 'softmax'
    history.load.push(dense ? [...stats.importance] : [...stats.load])
    history.entropy.push(stats.entropy)
    history.idle.push(
      dense ? Array.from({ length: N }, (_, i) => i).filter((i) => !assignment.includes(i)).length : stats.idle,
    )
    history.dropped.push(stats.dropped)
    if (regime) {
      history.agreement.push(adjustedRandIndex(regime, assignment))
    } else history.agreement.push(NaN)
  }
  // A run that stops early (diverged, or converged L-BFGS) reports its last step as the total.
  let total = steps
  const snapshot = (t: Size, done: boolean): MoeSnapshot => ({
    step: t,
    steps: total,
    done,
    method,
    spec: model.spec,
    history: {
      step: [...history.step],
      loss: [...history.loss],
      balance: [...history.balance],
      importance: [...history.importance],
      z: [...history.z],
      load: [...history.load],
      entropy: [...history.entropy],
      idle: [...history.idle],
      dropped: [...history.dropped],
      agreement: [...history.agreement],
    },
    checkpoints: [...checkpoints],
  })

  const start = { params: model.init(child(root, 'init')) }
  // Record every step of a short run, and about 300 points of a long one.
  const recordEvery = Math.max(1, Math.floor(steps / 300))
  if (method === 'em') {
    const alg = moeEm(model, data)
    let state = alg.init(start, child(root, 'init'))
    record(0, state.params)
    checkpoints.push({ step: 0, params: state.params })
    for (let t = 0; t < steps; t++) {
      state = alg.step(state, { t, stream: child(root, 'step', t) })
      const k = t + 1
      // A diverged run ends here, with its last step recorded and a final snapshot.
      const last = k === steps || Boolean(state.diverged)
      if (state.diverged) total = k
      if (k % recordEvery === 0 || last) record(k, state.params)
      if (k % every === 0 || last) {
        checkpoints.push({ step: k, params: state.params })
        yield snapshot(k, last)
      }
      if (last) break
    }
    return
  }
  const aux = { balance, importance, z }
  // Adam's loss takes the layers' context (the noisy gate's stream); L-BFGS's sees none, so its gate is noiseless.
  const alg = methodTraining(
    (p: MoeParams, b: { x: Tensor; y: Tensor }, ctx?: Context) => moeLoss(model, p, b.x, b.y, aux, ctx).total,
    { x: data.x, y: data.y },
    method === 'lbfgs'
      ? { method: 'lbfgs', memory }
      : { method: 'adam', stepSize: stepSize ?? 0.03, batchSize, clipNorm: 10 },
  )
  let state = alg.init(start, child(root, 'init'))
  record(0, state.params)
  checkpoints.push({ step: 0, params: state.params })
  for (let t = 0; t < steps; t++) {
    state = alg.step(state, { t, stream: child(root, 'step', t) })
    const k = t + 1
    // A run that diverges or stops ends here, with its last step recorded and a final snapshot.
    const early = state.diverged || state.stopped
    const last = k === steps || early
    if (early) total = k
    if (k % recordEvery === 0 || last) record(k, state.params)
    if (k % every === 0 || last) {
      checkpoints.push({ step: k, params: state.params })
      yield snapshot(k, last)
    }
    if (last) break
  }
}
