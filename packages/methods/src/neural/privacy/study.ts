/**
 * DP-SGD (Abadi et al., 2016) on a small MLP, for a worker: the same network (from the same initial weights) trained
 * by `aifn-compute/nn/training`'s `privateTraining` once per noise multiplier $\sigma$, including $\sigma = 0$
 * (clipping only, no privacy) as the baseline, with Poisson sampling rate $q = \text{batchSize} / n$, clipping norm
 * $C$ and a fixed number of steps. Each step clips every sampled example's gradient to norm at most $C$, sums them,
 * adds $\Gauss(\zeros, \sigma^2 C^2 \Imat)$ noise, divides by the expected batch size $qn$ and hands the result to
 * Adam. Each run records the minibatch loss per step and, at checkpoints, the test accuracy and the $\varepsilon$
 * spent at $\delta$ (RDP accounting, Mironov, 2017; $\infty$ for $\sigma = 0$), so the study traces accuracy against
 * $\varepsilon$ for each $\sigma$.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { ravel, type Params } from 'aifn-compute/foundation/pytree'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { privateTraining, type PrivateTrainingState } from 'aifn-compute/nn/training'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `privateTrainingStudy`. */
export type PrivateStudyOptions = {
  /** Noise multipliers to train with, in order (default [0, 0.6, 1, 2, 4]); 0 is the non-private baseline. */
  noiseMultipliers?: readonly number[]
  /** Hidden units of the one-hidden-layer tanh MLP (default 16). */
  hidden?: Size
  /** DP-SGD steps per run (default 300). */
  steps?: Size
  /** Expected examples per step $qn$ (default 32). */
  batchSize?: Size
  /** Clipping norm $C$ (default 1). */
  clipNorm?: number
  /** Adam's step size (default 0.05). */
  stepSize?: number
  /** $\delta$ for the reported $\varepsilon$ (default 1e-5). */
  delta?: number
  /**
   * Checkpoints per run, besides step 0 (default 30): one every $\lceil \text{steps} / \text{checkpoints} \rceil$
   * steps, and at the last.
   */
  checkpoints?: Size
  /** Seed of the initial weights and the training streams (default 0). */
  seed?: number
}

/** One noise multiplier's run. */
export type PrivateRun = {
  /** The run's noise multiplier $\sigma$. */
  noiseMultiplier: number
  /** Minibatch loss per step (NaN for an empty sample). */
  loss: number[]
  /** Checkpoint steps, from 0. */
  at: number[]
  /** Test accuracy at each checkpoint. */
  accuracy: number[]
  /** $\varepsilon$ spent at $\delta$ by each checkpoint (0 at step 0, $\infty$ without noise). */
  epsilon: number[]
  /** The share of the checkpoint step's sampled examples whose gradients were clipped (0 at step 0). */
  clipped: number[]
  /** Flat parameters $\thetavec$ at each checkpoint (`ravel` order of the MLP's parameters). */
  theta: Float64Array[]
}

/** A snapshot of `privateTrainingStudy`: the runs so far, the current one partial. */
export type PrivateStudySnapshot = {
  /** Steps done, over every run. */
  done: Size
  /** Steps in all: the steps per run times the number of noise multipliers. */
  total: Size
  /** Training examples $n$. */
  examples: Size
  /** Hidden units of the MLP. */
  hidden: Size
  /** The runs so far, one per noise multiplier, the last one partial while it trains (copies of each run's fields). */
  runs: PrivateRun[]
}

/**
 * The study's MLP (`inputs` to `hidden` tanh units to one logit, Xavier-uniform initialisation) and the map from flat
 * $\thetavec$ to its parameters.
 *
 * @param inputs The number of input features.
 * @param hidden The hidden units.
 * @returns The `Mlp`, and `unravel`, which turns a flat vector in `ravel` order (each layer's weight, then its bias)
 *   back into the MLP's parameters.
 *
 * @example Parameters from a flat vector
 * const { model, unravel } = privateStudyModel(2, 3)
 * // A 2 x 3 weight and 3 biases, then a 3 x 1 weight and 1 bias: 13 values.
 * const params = unravel(Float64Array.from({ length: 13 }, (_, i) => (i % 3) - 1))
 * print('first weight:', params[0].weight)
 * print('logit of (1, -1):', model.apply(params, tensor([[1, -1]])))
 */
export function privateStudyModel(inputs: Size, hidden: Size) {
  const model = Mlp([inputs, hidden, 1], { activation: 'tanh', init: xavierUniform() })
  const { unravel } = ravel(model.init(stream(0)))
  return { model, unravel }
}

/**
 * Train the study's MLP by DP-SGD for each noise multiplier on `train`, scoring accuracy on `test`, yielding a
 * snapshot at the start of each run and after each checkpoint. Every run starts from the same initial weights.
 * Throws `DomainError` when either set has no labels. Deterministic from the seed.
 *
 * @param train The training set: features `x` ($n \times d$) and labels `y` ($n$ values in $\{0, 1\}$; required).
 * @param test The test set, in the same form, on which accuracy is measured.
 * @param options The noise multipliers, the network, the training, $\delta$, the checkpoints and the seed.
 * @returns A generator of snapshots.
 *
 * @example Accuracy against the privacy spent, for three noise levels
 * const x = normals(stream(0), [64, 2])
 * const y = fromData(Float64Array.from(toRows(x), ([a, b]) => (a + b > 0 ? 1 : 0)), [64])
 * const options = { noiseMultipliers: [0, 1, 4], steps: 40, batchSize: 16, checkpoints: 2 }
 * let last
 * for (const s of privateTrainingStudy({ x, y }, { x, y }, options)) last = s
 * for (const r of last.runs) print(`sigma ${r.noiseMultiplier}: accuracy`, r.accuracy, ' epsilon', r.epsilon)
 */
export function* privateTrainingStudy(
  train: { readonly x: Tensor; readonly y?: Tensor },
  test: { readonly x: Tensor; readonly y?: Tensor },
  options: PrivateStudyOptions = {},
): Generator<PrivateStudySnapshot> {
  const {
    noiseMultipliers = [0, 0.6, 1, 2, 4],
    hidden = 16,
    steps = 300,
    batchSize = 32,
    clipNorm = 1,
    stepSize = 0.05,
    delta = 1e-5,
    checkpoints = 30,
    seed = 0,
  } = options
  if (!train.y || !test.y) throw new DomainError('privateTrainingStudy', 'privateTrainingStudy: the data need labels y')
  const [n, d] = train.x.shape
  const { model } = privateStudyModel(d, hidden)
  const data = { x: train.x, y: train.y }
  const loss = (params: Params[], e: { x: Tensor; y: Tensor }): Value =>
    binaryCrossEntropyWithLogits(model.apply(params, e.x), e.y)
  const initial = model.init(child(stream(`dp-sgd-init-${seed}`), 'weights'))
  const testY = toFlat(test.y)
  const accuracyOf = (params: Params[]) => {
    const z = toFlat(unwrap(model.apply(params, test.x)) as Tensor)
    let right = 0
    z.forEach((v, i) => (right += (v > 0 ? 1 : 0) === testY[i] ? 1 : 0))
    return right / z.length
  }
  const runs: PrivateRun[] = []
  const total = steps * noiseMultipliers.length
  let done = 0
  const snapshot = (): PrivateStudySnapshot => ({
    done,
    total,
    examples: n,
    hidden,
    runs: runs.map((r) => ({ ...r })),
  })
  const every = Math.max(1, Math.ceil(steps / checkpoints))
  for (const [k, sigma] of noiseMultipliers.entries()) {
    const alg = privateTraining<Params[], { x: Tensor; y: Tensor }>({
      loss,
      data,
      batchSize,
      clipNorm,
      noiseMultiplier: sigma,
      optimizer: adamRule({ stepSize }) as UpdateRule<unknown>,
      delta,
    })
    const root = child(stream(`dp-sgd-run-${seed}`), 'sigma', k)
    let state: PrivateTrainingState<Params[]> = alg.init({ params: initial }, child(root, 'init'))
    const run: PrivateRun = {
      noiseMultiplier: sigma,
      loss: [],
      at: [0],
      accuracy: [accuracyOf(state.params)],
      epsilon: [0],
      clipped: [0],
      theta: [Float64Array.from(ravel(state.params).vector)],
    }
    runs.push(run)
    yield snapshot()
    for (let t = 0; t < steps; t++) {
      state = alg.step(state, { t, stream: child(root, 'step', t) })
      run.loss.push(state.loss)
      done++
      if ((t + 1) % every === 0 || t + 1 === steps) {
        run.at.push(t + 1)
        run.accuracy.push(accuracyOf(state.params))
        run.epsilon.push(state.epsilon)
        run.clipped.push(state.clippedShare)
        run.theta.push(Float64Array.from(ravel(state.params).vector))
        yield snapshot()
      }
    }
  }
}
