/**
 * A consistency-training run from plain data, for a worker to stream: the network is built from its hidden widths
 * (tanh units, Xavier-uniform weights, the classifier of the consistency-regularisation figures), trained by Adam with
 * `consistencyTraining`, and every checkpoint holds plain arrays (`Float64Array`, `Int32Array`) only: the step's
 * losses and per-point diagnostics, the student's (and mean teacher's) class probabilities on the unlabelled points
 * and on optional evaluation points such as a grid, and their accuracies.
 */

import type { Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { shapeOf, type Tensor } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { adamRule } from 'aifn-compute/optim/first-order'
import { rowArgmax } from 'aifn-methods/learning/semi-supervised'
import {
  consistencyTraining,
  predictProbabilities,
  type ConsistencyMethod,
  type ConsistencyProblem,
  type ConsistencyState,
} from './consistency'

/** Options of `consistencyRun`; plain data, so a worker task can carry them. */
export type ConsistencyRunOptions = {
  /** The unsupervised term. */
  method: ConsistencyMethod
  /** Hidden widths of the classifier, tanh units (default `[32, 32]`). */
  hidden?: readonly number[]
  /** Updates to make (default 300); also UDA's planned steps $T$. */
  steps?: number
  /** Adam's step size (default 0.01). */
  rate?: number
  /** The unsupervised weight $\lambda_u$ (default 1). */
  weight?: number
  /** Steps of the Gaussian ramp-up of $\lambda_u$ (default 0: none). */
  rampUp?: number
  /** Standard deviation of the weak jitter (default 0.05). */
  weak?: number
  /** Standard deviation of the strong jitter (default 0.25). */
  strong?: number
  /** Unlabelled points per step (default all of them). */
  batch?: number
  /** The key of the stream the initial parameters are drawn from, `stream(seed)` (default 0). */
  seed?: number | string
  /** The key of the run's root stream, from which every step's draws derive (default 0, as `live`'s default). */
  stepSeed?: number | string
  /** Steps between checkpoints (default 10); step 0 and the last step are always kept. */
  every?: number
  /** Further inputs ($m \times d$, e.g. a grid) whose class probabilities each checkpoint holds. */
  evaluate?: Tensor
}

/** One checkpoint of a run: plain arrays only. */
export type ConsistencyCheckpoint = {
  /** Updates made (0 for the initial network). */
  step: number
  /** The step's objective (NaN at step 0). */
  loss: number
  /** Its supervised part. */
  supervisedLoss: number
  /** Its unsupervised part, before weighting. */
  unsupervisedLoss: number
  /** The unsupervised weight of the step. */
  weight: number
  /** The share of the step's batch the unsupervised term used. */
  maskRate: number
  /** The mask-weighted accuracy of the batch's pseudo-labels. */
  pseudoAccuracy: number
  /** The share of labelled points UDA's annealing kept. */
  labelledKept: number
  /** The per-class thresholds of the step (`fixmatch`). */
  thresholds: number[]
  /** Each unlabelled point's weight in the unsupervised term. */
  mask: Float64Array
  /** Each unlabelled point's target distribution, row-major $n_U \times K$. */
  targets: Float64Array
  /** Each unlabelled point's pseudo-label ($-1$: none yet). */
  pseudo: Int32Array
  /** VAT's perturbation of each unlabelled point, row-major $n_U \times d$ (empty otherwise). */
  perturbation: Float64Array
  /** The student's class probabilities on the unlabelled points, row-major $n_U \times K$. */
  probabilities: Float64Array
  /** The mean teacher's class probabilities on the unlabelled points, or null for the other methods. */
  teacherProbabilities: Float64Array | null
  /** The student's class probabilities on the `evaluate` inputs, row-major $m \times K$ (empty without them). */
  evaluate: Float64Array
  /** The mean teacher's on the `evaluate` inputs, or null. */
  teacherEvaluate: Float64Array | null
  /** The student's arg max accuracy on the unlabelled points against `truth` (NaN without it). */
  accuracy: number
  /** The mean teacher's (NaN for the other methods or without truth). */
  teacherAccuracy: number
  /** Whether the loss stopped being finite, which ends the run. */
  diverged: boolean
}

/**
 * The arg max accuracy of probability rows against true classes; NaN without the truth.
 *
 * @param p The probabilities, row-major $n \times K$.
 * @param K The number of classes.
 * @param truth The true class of each row, or undefined.
 * @returns The share of rows whose arg max is their true class.
 */
function accuracyAgainst(p: Float64Array, K: number, truth: ArrayLike<number> | undefined): number {
  if (!truth) return NaN
  const { cls } = rowArgmax(p, K)
  let right = 0
  for (let i = 0; i < cls.length; i++) if (cls[i] === truth[i]) right++
  return cls.length ? right / cls.length : NaN
}

/**
 * Train a classifier by `consistencyTraining` and yield a checkpoint at step 0, every `every` steps, at the last step
 * and when training diverges (which ends the run). The network is `Mlp([d, ...hidden, K])` with tanh units and
 * Xavier-uniform weights, drawn from `stream(seed)`; the optimiser is Adam with step size `rate`. Deterministic in
 * `seed` and `stepSeed`.
 *
 * @param problem The labelled and unlabelled inputs, the classes and optional truth (for the accuracies).
 * @param options The method, the network's widths, the training, the seeds and the checkpoints.
 * @returns A generator of checkpoints, one per kept step.
 *
 * @example A short FixMatch run on six points
 * const problem = {
 *   labelledX: tensor([[-1, 0], [1, 0]]),
 *   labelledY: [0, 1],
 *   unlabelledX: tensor([[-1.2, 0.3], [-0.8, -0.2], [0.9, 0.4], [1.1, -0.3]]),
 *   classes: 2,
 *   truth: [0, 0, 1, 1],
 * }
 * const options = { method: { kind: 'fixmatch', threshold: 0.7 }, hidden: [8], steps: 20, every: 10 }
 * for (const c of consistencyRun(problem, options)) print('step', c.step, ' accuracy', c.accuracy, ' mask', c.maskRate)
 */
export function* consistencyRun(
  problem: ConsistencyProblem,
  options: ConsistencyRunOptions,
): Generator<ConsistencyCheckpoint> {
  const { hidden = [32, 32], steps = 300, rate = 0.01, every = 10, seed = 0, stepSeed = 0, evaluate } = options
  const K = problem.classes
  const d = shapeOf(problem.unlabelledX)[1]
  const net: Layer<Params[]> = Mlp([d, ...hidden, K], { activation: 'tanh', init: xavierUniform() })
  const alg = consistencyTraining(problem, {
    model: net,
    method: options.method,
    weight: options.weight ?? 1,
    rampUp: options.rampUp ?? 0,
    steps,
    weak: options.weak ?? 0.05,
    strong: options.strong ?? 0.25,
    batch: options.batch,
    optimizer: adamRule({ stepSize: rate }),
  })
  const checkpoint = (k: number, state: ConsistencyState): ConsistencyCheckpoint => {
    const probabilities = predictProbabilities(net, state.params, problem.unlabelledX)
    const teacherProbabilities = state.teacher ? predictProbabilities(net, state.teacher, problem.unlabelledX) : null
    return {
      step: k,
      loss: state.loss,
      supervisedLoss: state.supervisedLoss,
      unsupervisedLoss: state.unsupervisedLoss,
      weight: state.weight,
      maskRate: state.maskRate,
      pseudoAccuracy: state.pseudoAccuracy,
      labelledKept: state.labelledKept,
      thresholds: state.thresholds,
      mask: state.mask,
      targets: state.targets,
      pseudo: state.pseudo,
      perturbation: state.perturbation,
      probabilities,
      teacherProbabilities,
      evaluate: evaluate ? predictProbabilities(net, state.params, evaluate) : new Float64Array(0),
      teacherEvaluate: evaluate && state.teacher ? predictProbabilities(net, state.teacher, evaluate) : null,
      accuracy: accuracyAgainst(probabilities, K, problem.truth),
      teacherAccuracy: teacherProbabilities ? accuracyAgainst(teacherProbabilities, K, problem.truth) : NaN,
      diverged: state.diverged === true,
    }
  }
  const it = live(alg, { params: net.init(stream(seed)) }, { stream: stream(stepSeed) })
  for (let k = 0; k <= steps; k++) {
    const next = it.next()
    if (next.done) return
    const state = next.value.state
    if (k % every === 0 || k === steps || state.diverged) yield checkpoint(k, state)
    if (state.diverged) return
  }
}
