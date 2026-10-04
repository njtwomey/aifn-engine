/**
 * Training a small MLP by full-batch L-BFGS against first-order methods, for a worker. Every optimiser starts from the
 * same initial weights and minimises the same objective, the mean loss on the whole training set plus (λ/2)‖W‖² over
 * the weight matrices: L-BFGS (Liu & Nocedal, 1989) and plain gradient descent on the full batch through compute's
 * `fullBatchTraining`, Adam (Kingma & Ba, 2015) and SGD on minibatches through `trainingLoop`. Each run records, per
 * iteration, the full-set objective, its gradient norm and the work done in full-data gradient evaluations (an L-BFGS
 * line search evaluates the loss and gradient several times; a minibatch step costs its share b/n); L-BFGS also records
 * its step length, its evaluations per iteration and the curvature sᵀy of each pair.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { ravel, type Params } from 'aifn-compute/foundation/pytree'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  add,
  fromData,
  mul,
  square,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { now } from 'aifn-compute/foundation/trace'
import { binaryCrossEntropyWithLogits, meanSquaredErrorLoss } from 'aifn-compute/learning/losses'
import type { ActivationName } from 'aifn-compute/nn/functional'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { fullBatchTraining, trainingLoop, treeObjective } from 'aifn-compute/nn/training'
import { adamRule, sgdRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import type { LbfgsState } from 'aifn-compute/optim/second-order'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The optimisers compared, in their fixed order (and colour slots). */
export const COMPARISON_OPTIMISERS = ['lbfgs', 'gradient-descent', 'adam', 'sgd'] as const
export type ComparisonOptimiser = (typeof COMPARISON_OPTIMISERS)[number]

/** The task: binary classification of points x [n, 2] by labels y ∈ {0, 1}, or regression of y on x [n, 1]. */
export type ComparisonTask = 'classification' | 'regression'

/** The network: an MLP of `depth` hidden layers of `width` units with one output (a logit or a value). */
export type ComparisonNetwork = {
  inputs: Size
  width: Size
  depth: Size
  activation: Extract<ActivationName, 'tanh' | 'gelu' | 'relu'>
}

/** The MLP of a network configuration (Xavier-uniform weights, zero biases) and the map from θ back to its parameters. */
export function comparisonModel(network: ComparisonNetwork): {
  model: Layer<Params[]>
  /** The parameter tree of a flat θ (in `ravel` order). */
  unravel: (theta: ArrayLike<number>) => Params[]
  size: Size
} {
  const { inputs, width, depth, activation } = network
  const sizes = [inputs, ...Array.from({ length: depth }, () => width), 1]
  const model = Mlp(sizes, { activation, init: xavierUniform() })
  const { vector, unravel } = ravel(model.init(stream(0)))
  return { model, unravel, size: vector.length }
}

/** Options of `fullBatchComparison`. */
export type ComparisonOptions = {
  task: ComparisonTask
  network: Omit<ComparisonNetwork, 'inputs'>
  /** The optimisers to train, each from the same initial weights (default all four). */
  optimisers?: readonly ComparisonOptimiser[]
  /** At most this many iterations per optimiser (default 300). */
  iterations?: Size
  /** L-BFGS memory m (default 10) and gradient-norm tolerance (default 1e-6). */
  memory?: Size
  tolerance?: number
  /** Fixed step sizes: gradient descent (default 0.3), Adam (default 0.01), SGD (default 0.1). */
  gdStep?: number
  adamStep?: number
  sgdStep?: number
  /** Minibatch size of Adam and SGD (default 32). */
  batchSize?: Size
  /** The L2 strength λ in (λ/2)‖W‖² (default 0). */
  l2?: number
  /** The seed of the initial weights (default 0). */
  seed?: number
  /** At most this many parameter checkpoints per optimiser (default 120). */
  checkpoints?: Size
}

/** One optimiser's run, recorded at every iteration (minibatch methods: at most about 400 records). */
export type ComparisonRun = {
  optimiser: ComparisonOptimiser
  /** Iterations (updates) at each record. */
  iteration: number[]
  /** Full-data gradient evaluations used by each record. */
  evaluations: number[]
  /** The full-set objective (mean loss + L2) and its gradient norm. */
  loss: number[]
  gradNorm: number[]
  /** L-BFGS only, per iteration 1, 2, …: the accepted step length α, the loss evaluations of the line search, sᵀy, and whether the pair was skipped. */
  stepSize: number[]
  lineEvaluations: number[]
  curvature: number[]
  skipped: boolean[]
  /** θ at checkpoints (in `ravel` order), with their iterations. */
  checkpoints: { iteration: number; theta: Float64Array }[]
  /** Training accuracy (classification) or RMSE (regression) at the last record. */
  score: number
  /** Milliseconds spent in the optimiser's steps (records excluded). */
  ms: number
  /** Why the run ended: the iteration budget, convergence, a stalled line search or divergence. */
  stop: 'budget' | 'converged' | 'stalled' | 'diverged' | null
}

/** A snapshot of `fullBatchComparison`: the runs so far (the current one partial). */
export type ComparisonSnapshot = {
  task: ComparisonTask
  network: ComparisonNetwork
  parameterCount: Size
  examples: Size
  /** Iterations done over every optimiser, and the total budget. */
  done: Size
  total: Size
  runs: ComparisonRun[]
}

/** (λ/2) Σ‖W‖² over the weight matrices (biases unpenalised) of MLP parameters. */
function weightPenalty(params: Params[], l2: number): Value {
  let total: Value = 0
  for (const layer of params) {
    const w = (layer as { weight?: Tensor }).weight
    if (w) total = add(total, sum(square(w)))
  }
  return mul(l2 / 2, total)
}

/**
 * Train a small MLP by each optimiser in turn, from the same initial weights, on the whole of `data` (x [n, d], y [n]),
 * yielding a snapshot every few iterations. Classification uses binary cross-entropy on one logit; regression the mean
 * squared error.
 */
export function* fullBatchComparison(
  data: { readonly x: Tensor; readonly y?: Tensor },
  options: ComparisonOptions,
): Generator<ComparisonSnapshot> {
  const {
    task,
    optimisers = COMPARISON_OPTIMISERS,
    iterations = 300,
    memory = 10,
    tolerance = 1e-6,
    gdStep = 0.3,
    adamStep = 0.01,
    sgdStep = 0.1,
    batchSize = 32,
    l2 = 0,
    seed = 0,
    checkpoints = 120,
  } = options
  if (!data.y) throw new DomainError('fullBatchComparison', 'fullBatchComparison: the data need targets y')
  const [n, d] = data.x.shape
  const network: ComparisonNetwork = { ...options.network, inputs: d }
  const { model, size } = comparisonModel(network)
  const y = fromData(Float64Array.from(toFlat(data.y)), [n, 1])
  const train = { x: data.x, y }
  const fit = (out: Value, target: Tensor) =>
    task === 'classification' ? binaryCrossEntropyWithLogits(out, target) : meanSquaredErrorLoss(out, target)
  const loss = (params: Params[], batch: { x: Tensor; y: Tensor }): Value => {
    const value = fit(model.apply(params, batch.x), batch.y)
    return l2 > 0 ? add(value, weightPenalty(params, l2)) : value
  }
  const initial = model.init(child(stream(`full-batch-init-${seed}`), 'weights'))
  const { unravel } = ravel(initial)
  const objective = treeObjective(loss, train, unravel)
  const targets = toFlat(y)
  const scoreOf = (theta: Float64Array) => {
    const out = toFlat(unwrap(model.apply(unravel(theta), data.x)) as Tensor)
    if (task === 'classification') {
      let right = 0
      out.forEach((z, i) => (right += (z > 0 ? 1 : 0) === targets[i] ? 1 : 0))
      return right / n
    }
    let se = 0
    out.forEach((v, i) => (se += (v - targets[i]) ** 2))
    return Math.sqrt(se / n)
  }
  const batch = Math.max(1, Math.min(batchSize, n))
  const runs: ComparisonRun[] = []
  const total = iterations * optimisers.length
  let done = 0
  const snapshot = (): ComparisonSnapshot => ({
    task,
    network,
    parameterCount: size,
    examples: n,
    done,
    total,
    runs: runs.map((r) => ({ ...r })),
  })
  const every = Math.max(1, Math.ceil(iterations / checkpoints))
  const yieldEvery = Math.max(5, Math.ceil(iterations / 25))
  const root = stream(`full-batch-run-${seed}`)

  for (const optimiser of optimisers) {
    const run: ComparisonRun = {
      optimiser,
      iteration: [],
      evaluations: [],
      loss: [],
      gradNorm: [],
      stepSize: [],
      lineEvaluations: [],
      curvature: [],
      skipped: [],
      checkpoints: [],
      score: NaN,
      ms: 0,
      stop: null,
    }
    runs.push(run)
    const record = (t: number, evaluations: number, theta: Float64Array, value?: number, grad?: ArrayLike<number>) => {
      // Minibatch states carry minibatch values: the full-set objective is evaluated here, outside the timed steps.
      const full = value === undefined || grad === undefined ? objective(fromData(theta)) : { value, grad }
      run.iteration.push(t)
      run.evaluations.push(evaluations)
      run.loss.push(full.value)
      const g = full.grad as ArrayLike<number>
      let g2 = 0
      for (let i = 0; i < g.length; i++) g2 += g[i] * g[i]
      run.gradNorm.push(Math.sqrt(g2))
      if (t % every === 0 || t === iterations) run.checkpoints.push({ iteration: t, theta: Float64Array.from(theta) })
    }
    const finish = (theta: Float64Array, stop: ComparisonRun['stop']) => {
      const last = run.checkpoints.at(-1)
      if (!last || last.iteration !== run.iteration.at(-1))
        run.checkpoints.push({ iteration: run.iteration.at(-1) ?? 0, theta: Float64Array.from(theta) })
      run.score = scoreOf(theta)
      run.stop = stop
    }
    const ctx = (t: number) => ({ t, stream: child(root, optimiser, 'step', t) })

    if (optimiser === 'lbfgs' || optimiser === 'gradient-descent') {
      const alg =
        optimiser === 'lbfgs'
          ? fullBatchTraining({ loss, data: train, method: 'lbfgs', options: { memory, tolerance } })
          : fullBatchTraining({
              loss,
              data: train,
              method: 'gradient-descent',
              options: { stepSize: gdStep, tolerance },
            })
      let t0 = now()
      let state = alg.init({ params: initial }, child(root, optimiser, 'init'))
      run.ms += now() - t0
      const view = (s: typeof state) => s as typeof state & Partial<LbfgsState>
      record(0, state.evaluations, Float64Array.from(toFlat(state.x)), state.value, toFlat(view(state).grad!))
      let stop: ComparisonRun['stop'] = 'budget'
      for (let t = 1; t <= iterations; t++) {
        if (state.diverged || state.converged || alg.done?.(state)) {
          stop = state.diverged ? 'diverged' : state.converged ? 'converged' : 'stalled'
          break
        }
        const before = { x: toFlat(state.x), grad: toFlat(view(state).grad!) }
        t0 = now()
        state = alg.step(state, ctx(t))
        run.ms += now() - t0
        const v = view(state)
        const x = Float64Array.from(toFlat(state.x))
        const g = toFlat(v.grad!)
        if (optimiser === 'lbfgs') {
          let sy = 0
          for (let i = 0; i < x.length; i++) sy += (x[i] - before.x[i]) * (g[i] - before.grad[i])
          run.stepSize.push(v.stepSize ?? NaN)
          run.lineEvaluations.push(v.lineSearch?.evaluations ?? NaN)
          run.curvature.push(sy)
          run.skipped.push(Boolean(v.skipped))
        }
        record(t, state.evaluations, x, state.value, g)
        done++
        if (t % yieldEvery === 0) yield snapshot()
      }
      if (stop === 'budget' && (state.diverged || state.converged)) stop = state.diverged ? 'diverged' : 'converged'
      finish(Float64Array.from(toFlat(state.x)), stop)
    } else {
      const rule = (
        optimiser === 'adam' ? adamRule({ stepSize: adamStep }) : sgdRule({ stepSize: sgdStep })
      ) as UpdateRule<unknown>
      const alg = trainingLoop<Params[], { x: Tensor; y: Tensor }>({
        loss,
        data: train,
        batchSize: batch,
        optimizer: rule,
      })
      const share = batch / n
      let t0 = now()
      let state = alg.init({ params: initial }, child(root, optimiser, 'init'))
      run.ms += now() - t0
      record(0, share, ravel(state.params).vector)
      let stop: ComparisonRun['stop'] = 'budget'
      for (let t = 1; t <= iterations; t++) {
        t0 = now()
        state = alg.step(state, ctx(t))
        run.ms += now() - t0
        if (state.diverged) {
          stop = 'diverged'
          break
        }
        // Step t used minibatch t − 1's gradient and evaluated minibatch t's: t + 1 minibatches so far.
        record(t, (t + 1) * share, ravel(state.params).vector)
        done++
        if (t % yieldEvery === 0) yield snapshot()
      }
      finish(ravel(state.params).vector, stop)
    }
    done = runs.length * iterations
    yield snapshot()
  }
  yield snapshot()
}
