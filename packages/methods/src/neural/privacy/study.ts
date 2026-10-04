/**
 * DP-SGD on a small MLP, for a worker: the same network (from the same initial weights) trained by compute
 * `privateTraining` once per noise multiplier σ, including σ = 0 (clipping only, no privacy) as the baseline, with
 * Poisson sampling rate q = batchSize/n, clipping norm C and a fixed number of steps. Each run records the minibatch
 * loss per step and, at checkpoints, the test accuracy and the ε spent at δ (RDP accounting), so the study traces
 * accuracy against ε for each σ.
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
  /** Expected examples per step qn (default 32). */
  batchSize?: Size
  /** Clipping norm C (default 1). */
  clipNorm?: number
  /** Adam's step size (default 0.05). */
  stepSize?: number
  /** δ for the reported ε (default 1e-5). */
  delta?: number
  /** Checkpoints per run (default 30). */
  checkpoints?: Size
  /** Seed of the initial weights and the training streams (default 0). */
  seed?: number
}

/** One noise multiplier's run. */
export type PrivateRun = {
  noiseMultiplier: number
  /** Minibatch loss per step (NaN for an empty sample). */
  loss: number[]
  /** Checkpoint steps, with the test accuracy and ε at each. */
  at: number[]
  accuracy: number[]
  epsilon: number[]
  /** Share of each step's examples whose gradients were clipped, per checkpoint. */
  clipped: number[]
  /** Flat parameters θ at each checkpoint (`ravel` order of the MLP's parameters). */
  theta: Float64Array[]
}

/** A snapshot of `privateTrainingStudy`: the runs so far, the current one partial. */
export type PrivateStudySnapshot = {
  done: Size
  total: Size
  examples: Size
  hidden: Size
  runs: PrivateRun[]
}

/** The study's MLP (2 → hidden → 1, tanh) and the map from flat θ to its parameters. */
export function privateStudyModel(inputs: Size, hidden: Size) {
  const model = Mlp([inputs, hidden, 1], { activation: 'tanh', init: xavierUniform() })
  const { unravel } = ravel(model.init(stream(0)))
  return { model, unravel }
}

/**
 * Train the study's MLP by DP-SGD for each noise multiplier on `train` (x [n, d], y [n] in {0, 1}), scoring accuracy
 * on `test`, yielding a snapshot after each checkpoint.
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
