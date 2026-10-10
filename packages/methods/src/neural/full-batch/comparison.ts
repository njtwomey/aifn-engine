/**
 * Training a small MLP by full-batch L-BFGS against first-order methods, for a worker. Every optimiser starts from the
 * same initial weights and minimises the same objective, the mean loss on the whole training set plus
 * $\frac{\lambda}{2}\sum_l \lVert \Wmat_l \rVert_F^2$ over the weight matrices $\Wmat_l$ (biases unpenalised): L-BFGS
 * (Liu and Nocedal, 1989) and plain gradient descent on the full batch through compute's `fullBatchTraining`, Adam
 * (Kingma and Ba, 2015) and SGD on minibatches through `trainingLoop`. Each run records, per iteration, the full-set
 * objective, its gradient norm and the work done in full-data gradient evaluations (an L-BFGS line search evaluates
 * the loss and gradient several times; a minibatch step costs its share $b/n$); L-BFGS also records its step length,
 * its line-search evaluations per iteration and the curvature $\svec^\top\yvec$ of each pair, with
 * $\svec = \thetavec_{t} - \thetavec_{t-1}$ and $\yvec$ the change in the gradient.
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
/** One of the optimisers compared: full-batch L-BFGS or gradient descent, or minibatch Adam or SGD. */
export type ComparisonOptimiser = (typeof COMPARISON_OPTIMISERS)[number]

/**
 * The task: binary classification of points $\xvec_i \in \reals^d$ by labels $y_i \in \{0, 1\}$ (binary cross-entropy
 * on one logit), or regression of $y_i \in \reals$ on $\xvec_i$ (mean squared error).
 */
export type ComparisonTask = 'classification' | 'regression'

/** The network: an MLP of `depth` hidden layers of `width` units with one output (a logit or a value). */
export type ComparisonNetwork = {
  /** The number of input features $d$. */
  inputs: Size
  /** The number of units in each hidden layer. */
  width: Size
  /** The number of hidden layers. */
  depth: Size
  /** The activation after each hidden layer (the output layer has none). */
  activation: Extract<ActivationName, 'tanh' | 'gelu' | 'relu'>
}

/**
 * The MLP of a network configuration (Xavier-uniform weights, zero biases) and the map from a flat parameter vector
 * $\thetavec$ back to its parameter tree.
 *
 * @param network The input size, the hidden width and depth, and the activation.
 * @returns The model, whose `init` draws the weights; `unravel`, which reads a flat $\thetavec$ as the model's
 *   parameters; and `size`, the number of parameters (weights and biases).
 *
 * @example Two hidden layers of 8 tanh units, with every parameter 0.1
 * const { model, unravel, size } = comparisonModel({ inputs: 2, width: 8, depth: 2, activation: 'tanh' })
 * print(model.label, ' parameters:', size)
 * const params = unravel(new Float64Array(size).fill(0.1))
 * print('output at (1, 1):', model.apply(params, tensor([[1, 1]])))
 */
export function comparisonModel(network: ComparisonNetwork): {
  /** The MLP, with Xavier-uniform weights and zero biases from its `init`. */
  model: Layer<Params[]>
  /** The parameter tree of a flat $\thetavec$ (in `ravel` order). */
  unravel: (theta: ArrayLike<number>) => Params[]
  /** The number of parameters, the length of $\thetavec$. */
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
  /** Classification (binary cross-entropy on one logit) or regression (mean squared error). */
  task: ComparisonTask
  /** The hidden width and depth and the activation; the input size is the number of columns of the data's `x`. */
  network: Omit<ComparisonNetwork, 'inputs'>
  /** The optimisers to train, in this order, each from the same initial weights (default all four). */
  optimisers?: readonly ComparisonOptimiser[]
  /** At most this many iterations per optimiser (default 300). */
  iterations?: Size
  /** The L-BFGS memory $m$, the number of curvature pairs kept (default 10). */
  memory?: Size
  /** The gradient-norm tolerance at which L-BFGS and gradient descent stop as converged (default 1e-6). */
  tolerance?: number
  /** The fixed step size of gradient descent (default 0.3). */
  gdStep?: number
  /** The step size of Adam (default 0.01). */
  adamStep?: number
  /** The step size of SGD (default 0.1). */
  sgdStep?: number
  /** Minibatch size of Adam and SGD, clamped to between 1 and the number of examples (default 32). */
  batchSize?: Size
  /**
   * The L2 strength $\lambda$ in $\frac{\lambda}{2}\sum_l \lVert \Wmat_l \rVert_F^2$, over the weight matrices only
   * (default 0, no penalty).
   */
  l2?: number
  /** The seed of the initial weights and of the runs' own draws, such as the minibatches (default 0). */
  seed?: number
  /**
   * About this many parameter checkpoints per optimiser (default 120): one every `ceil(iterations / checkpoints)`
   * iterations from iteration 0, and the last.
   */
  checkpoints?: Size
}

/** One optimiser's run, recorded at every iteration from 0, so at most `iterations + 1` records. */
export type ComparisonRun = {
  /** The optimiser of the run. */
  optimiser: ComparisonOptimiser
  /** Iterations (updates) at each record. */
  iteration: number[]
  /**
   * Full-data gradient evaluations used by each record, cumulative: the optimiser's own count for L-BFGS and gradient
   * descent, and $b/n$ per minibatch drawn for Adam and SGD.
   */
  evaluations: number[]
  /** The full-set objective (mean loss plus the L2 penalty) at each record. */
  loss: number[]
  /** The Euclidean norm of the full-set objective's gradient at each record. */
  gradNorm: number[]
  /** L-BFGS only, per iteration $1, 2, \dots$: the accepted step length $\alpha$ (NaN when not reported). */
  stepSize: number[]
  /** L-BFGS only, per iteration: the loss and gradient evaluations of the line search (NaN when not reported). */
  lineEvaluations: number[]
  /**
   * L-BFGS only, per iteration: the curvature $\svec^\top\yvec$ of the step, with $\svec$ the change in $\thetavec$
   * and $\yvec$ the change in the gradient.
   */
  curvature: number[]
  /** L-BFGS only, per iteration: whether the curvature pair was skipped rather than stored. */
  skipped: boolean[]
  /** $\thetavec$ at checkpoints (in `ravel` order), with their iterations. */
  checkpoints: { iteration: number; theta: Float64Array }[]
  /**
   * Training accuracy (classification, a logit above 0 read as label 1) or RMSE (regression) at the end of the run;
   * NaN until it ends.
   */
  score: number
  /** Milliseconds spent in the optimiser's `init` and steps (records excluded). */
  ms: number
  /**
   * Why the run ended: the iteration budget, convergence, a stalled line search or divergence; `null` while it runs.
   */
  stop: 'budget' | 'converged' | 'stalled' | 'diverged' | null
}

/** A snapshot of `fullBatchComparison`: the runs so far (the current one partial). */
export type ComparisonSnapshot = {
  /** The task, as in the options. */
  task: ComparisonTask
  /** The network, with its input size taken from the data. */
  network: ComparisonNetwork
  /** The number of parameters of the network. */
  parameterCount: Size
  /** The number of training examples $n$. */
  examples: Size
  /**
   * Iterations done over every optimiser so far; a finished run counts its whole budget, even when it stopped early.
   */
  done: Size
  /** The total budget, `iterations` times the number of optimisers. */
  total: Size
  /**
   * One run per optimiser started: shallow copies, whose arrays are the generator's own and keep growing after the
   * snapshot (posting it to another thread copies them).
   */
  runs: ComparisonRun[]
}

/**
 * The penalty $\frac{\lambda}{2}\sum_l \lVert \Wmat_l \rVert_F^2$ over the weight matrices of MLP parameters (biases
 * unpenalised; differentiable).
 *
 * @param params The MLP's parameters, one tree per layer; layers without a `weight` (activations) add nothing.
 * @param l2 The strength $\lambda$.
 * @returns The penalty, as a value that gradients can flow through.
 */
function weightPenalty(params: Params[], l2: number): Value {
  let total: Value = 0
  for (const layer of params) {
    const w = (layer as { weight?: Tensor }).weight
    if (w) total = add(total, sum(square(w)))
  }
  return mul(l2 / 2, total)
}

/**
 * Train a small MLP by each optimiser in turn, from the same initial weights, on the whole of `data` ($\Xmat$ of
 * $n \times d$, $\yvec$ of $n$), yielding a snapshot every few iterations. Classification uses binary cross-entropy on
 * one logit; regression the mean squared error. L-BFGS and gradient descent stop early when they converge (gradient
 * norm below `tolerance`), diverge or stall; Adam and SGD stop early only when they diverge. Deterministic from `seed`.
 * Throws `DomainError` when `data` has no targets.
 *
 * @param data The training set: `x`, the $n \times d$ inputs, and `y`, the $n$ targets (labels 0 or 1 for
 *   classification), in any shape with $n$ values. `y` is required despite being optional in the type.
 * @param options The task and network, the optimisers, the iteration budget, the L-BFGS memory and tolerance, the step
 *   sizes, the minibatch size, the L2 strength, the seed and the number of checkpoints.
 * @returns A generator of snapshots: one every `max(5, ceil(iterations / 25))` iterations, one at the end of each
 *   optimiser's run, and a final one. Each holds every run so far, the current one partial.
 *
 * @example Four optimisers on a small classification problem
 * const x = normals(stream(1), [40, 2])
 * const y = tensor(toArray(x).map(([a, b]) => (a * b > 0 ? 1 : 0)))
 * const options = { task: 'classification', network: { width: 8, depth: 1, activation: 'tanh' }, iterations: 40 }
 * let last
 * for (const snapshot of fullBatchComparison({ x, y }, options)) last = snapshot
 * for (const run of last.runs)
 *   print(run.optimiser, ' loss', run.loss.at(-1), ' evaluations', run.evaluations.at(-1), ' accuracy', run.score)
 *
 * @example The L-BFGS line search on a regression
 * const x = normals(stream(2), [30, 1])
 * const y = tensor(toArray(x).map(([a]) => Math.sin(2 * a)))
 * const network = { width: 8, depth: 1, activation: 'tanh' }
 * const options = { task: 'regression', network, optimisers: ['lbfgs'], iterations: 10 }
 * const snapshots = [...fullBatchComparison({ x, y }, options)]
 * const run = snapshots.at(-1).runs[0]
 * print('step lengths:', run.stepSize)
 * print('line-search evaluations:', run.lineEvaluations)
 * print('curvature:', run.curvature)
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
