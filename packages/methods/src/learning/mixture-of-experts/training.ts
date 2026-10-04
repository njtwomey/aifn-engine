/**
 * Training a mixture of experts by gradient descent (Adam through aifn's autodiff, with the auxiliary losses of
 * `aifn-compute/nn/experts`), and `mixtureOfExpertsRun`, a generator that trains by EM or by Adam and yields snapshots with the
 * curves a figure needs: the data loss, the auxiliary losses, each expert's load, the router's entropy and how well the
 * gate's assignments recover the true regimes (adjusted Rand index), plus the parameters at checkpoints for a player.
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

/** Adam on the data loss plus the weighted auxiliary losses, as a traceable `trainingLoop`. */
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
  /** Inputs [T, d], targets [T], and optionally the true regime of each row (int [T]), for the agreement curve. */
  data: MoeData & { regime?: Tensor }
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
  /** Keep the parameters every this many steps for the player (default steps/60, at least 1). */
  every?: Size
  stepSize?: number
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

/** Curves of a run, one entry per recorded step. */
export type MoeHistory = {
  step: number[]
  /** The data term on the whole training set. */
  loss: number[]
  balance: number[]
  importance: number[]
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
  readonly step: Size
  readonly steps: Size
  readonly done: boolean
  readonly method: 'em' | 'adam' | 'lbfgs'
  readonly spec: MoeSpec
  readonly history: MoeHistory
  /** Parameters at step 0, every `every` steps and at the last step. */
  readonly checkpoints: readonly { step: Size; params: MoeParams }[]
}

const scalar = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/**
 * Train a mixture of experts on `data` by EM or Adam, yielding a snapshot every `every` steps and at the end: a
 * generator, so a worker can stream the run to a page that plots the curves and plays the checkpoints.
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
  // A run that stops early (converged L-BFGS) reports its last step as the total.
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
      if (k % recordEvery === 0 || k === steps) record(k, state.params)
      if (k % every === 0 || k === steps) {
        checkpoints.push({ step: k, params: state.params })
        yield snapshot(k, k === steps)
      }
      if (state.diverged) break
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
    if (k % recordEvery === 0 || k === steps) record(k, state.params)
    if (k % every === 0 || k === steps) {
      checkpoints.push({ step: k, params: state.params })
      yield snapshot(k, k === steps)
    }
    if (state.diverged || state.stopped) {
      total = k
      if (k % every !== 0 && k !== steps) {
        if (k % recordEvery !== 0) record(k, state.params)
        checkpoints.push({ step: k, params: state.params })
        yield snapshot(k, true)
      }
      break
    }
  }
}
