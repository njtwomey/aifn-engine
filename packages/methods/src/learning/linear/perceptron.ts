/**
 * The perceptron (Rosenblatt, 1958): a linear classifier trained one example at a time, updating only on mistakes.
 * Novikoff's (1962) theorem bounds the number of mistakes on separable data by (R/γ)². The averaged perceptron
 * (Freund and Schapire, 1999, "Large margin classification using the perceptron algorithm", Machine Learning 37)
 * predicts with the mean of the weight vectors over every example visited, which is far less sensitive to the last
 * few updates on data that are not separable.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import type {
  Decides,
  Estimator,
  FitOptions,
  Fitted,
  Scores,
  Supervised,
  Trained,
} from 'aifn-compute/learning/estimators'
import { permutation, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { classLabels, inputs, matrix, vec } from '../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The problem a perceptron run solves: inputs [n, d] and labels ±1. */
export interface PerceptronProblem {
  x: Tensor
  /** Labels −1 or +1, [n]. */
  y: Tensor
  learningRate?: number
  intercept?: boolean
  /** Visit the rows in a fresh random order each epoch (drawn from the run's streams); default false (row order). */
  shuffle?: boolean
}

/** One perceptron state: the weights after visiting one example. */
export interface PerceptronState extends Status {
  /** Examples visited. */
  t: number
  weights: Tensor
  bias: number
  /** The mean of the weights and bias after each of the t steps so far (the averaged perceptron); the start at t = 0. */
  averageWeights: Tensor
  averageBias: number
  /** The example visited in this step (−1 at the start), its margin y(w·x + b) before the update, and whether it updated. */
  example: number
  margin: number
  updated: boolean
  /** Mistakes so far, and mistakes in the current epoch. */
  mistakes: number
  epochMistakes: number
  epoch: number
  /** Position within the epoch's visiting order. */
  position: number
  /** The visiting order of the current epoch. */
  order: Tensor
  /** A full epoch passed without a mistake. */
  converged: boolean
}

/**
 * The perceptron as a traceable algorithm: each step visits one example and, if y(w·x + b) ≤ 0, sets
 * w ← w + η y x and b ← b + η y. It has converged after an epoch without mistakes. `init` takes optional starting
 * weights. With `shuffle`, the first epoch's order comes from the `init` stream and each later one from the stream of
 * the step that ends the previous epoch.
 */
export function perceptronSteps(
  problem: PerceptronProblem,
): Algorithm<{ weights?: Tensor; bias?: number }, PerceptronState> {
  const { n, d, v } = matrix(problem.x, 'perceptronSteps')
  const y = dense.data(problem.y)
  const eta = problem.learningRate ?? 1
  const intercept = problem.intercept ?? true
  const orderFor = (s: Stream): Tensor =>
    problem.shuffle
      ? permutation(s, n)
      : fromData(
          Int32Array.from({ length: n }, (_, i) => i),
          [n],
        )
  return {
    name: 'perceptron',
    init: ({ weights, bias = 0 }, s) => ({
      t: 0,
      weights: weights ?? fromData(new Float64Array(d), [d]),
      bias,
      averageWeights: weights ?? fromData(new Float64Array(d), [d]),
      averageBias: bias,
      example: -1,
      margin: NaN,
      updated: false,
      mistakes: 0,
      epochMistakes: 0,
      epoch: 0,
      position: 0,
      order: orderFor(s),
      converged: false,
    }),
    step: (state, ctx) => {
      const i = state.order.data[state.position]
      const w = Float64Array.from(state.weights.data as Float64Array)
      let f = state.bias
      for (let j = 0; j < d; j++) f += w[j] * v[i * d + j]
      const margin = y[i] * f
      const updated = margin <= 0
      let bias = state.bias
      if (updated) {
        for (let j = 0; j < d; j++) w[j] += eta * y[i] * v[i * d + j]
        if (intercept) bias += eta * y[i]
      }
      const epochMistakes = state.epochMistakes + (updated ? 1 : 0)
      const last = state.position === n - 1
      // Running means: avg_t = avg_{t−1} + (w_t − avg_{t−1})/t.
      const t = state.t + 1
      const avg = Float64Array.from(state.averageWeights.data as Float64Array)
      for (let j = 0; j < d; j++) avg[j] += (w[j] - avg[j]) / t
      return {
        t,
        weights: fromData(w, [d]),
        bias,
        averageWeights: fromData(avg, [d]),
        averageBias: state.averageBias + (bias - state.averageBias) / t,
        example: i,
        margin,
        updated,
        mistakes: state.mistakes + (updated ? 1 : 0),
        epochMistakes: last ? 0 : epochMistakes,
        epoch: last ? state.epoch + 1 : state.epoch,
        position: last ? 0 : state.position + 1,
        order: last ? orderFor(ctx.stream) : state.order,
        converged: last && epochMistakes === 0,
      }
    },
  }
}

/** A fitted binary perceptron (labels 0/1). */
export interface PerceptronModel
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Trained<PerceptronState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'perceptron'
  /** The weights and bias it predicts with: the final ones, or their averages with `average`. */
  readonly weights: Tensor
  readonly bias: number
  readonly averaged: boolean
  readonly mistakes: number
  readonly epochs: number
  readonly converged: boolean
}

/**
 * The perceptron for labels 0/1 (mapped to ∓1): at most `epochs` passes (default 100), stopping after a pass without
 * mistakes. `score` is w·x + b [m]; `decide` is 1 where it is positive. With `average`, w and b are the averaged
 * perceptron's means over every step. With `shuffle`, `fit` needs a stream.
 */
export function perceptron(
  params: { epochs?: number; learningRate?: number; intercept?: boolean; shuffle?: boolean; average?: boolean } = {},
): Estimator<Supervised<Tensor, Tensor>, PerceptronModel> {
  const { epochs = 100, learningRate = 1, intercept = true, shuffle = false, average = false } = params
  return {
    name: 'perceptron',
    params: { epochs, learningRate, intercept, shuffle, average },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'perceptron')
      const { y: labels, k } = classLabels(y, n, 'perceptron')
      if (k > 2) throw new DomainError('perceptron', 'perceptron: binary labels 0/1 only; use a multiclass reduction')
      const signs = vec(Array.from(labels, (c) => (c === 1 ? 1 : -1)))
      const alg = perceptronSteps({ x, y: signs, learningRate, intercept, shuffle })
      const training: Trace<PerceptronState> = trace(alg, {}, epochs * n, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          mistakes: (s) => s.mistakes,
          ...(options.trace?.record as Record<string, (s: PerceptronState, t: number) => number> | undefined),
        },
      })
      const final = training.final
      const weights = average ? final.averageWeights : final.weights
      const bias = average ? final.averageBias : final.bias
      const w = weights.data as Float64Array
      const score = (q: Tensor) => {
        const { n: m, v } = inputs(q, d, 'perceptron')
        const out = new Float64Array(m)
        for (let i = 0; i < m; i++) {
          let s = bias
          for (let j = 0; j < d; j++) s += w[j] * v[i * d + j]
          out[i] = s
        }
        return fromData(out, [m])
      }
      return {
        kind: 'model',
        name: 'perceptron',
        weights,
        bias,
        averaged: average,
        mistakes: final.mistakes,
        epochs: final.epoch,
        converged: final.converged,
        training,
        forward: score,
        score,
        decide: (q: Tensor) =>
          fromData(
            Int32Array.from(score(q).data as Float64Array, (s) => (s > 0 ? 1 : 0)),
            [q.shape[0]],
          ),
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'perceptron',
    module: 'learning/linear',
    name: 'Perceptron',
    summary: "Rosenblatt's perceptron for two classes, trained by mistake-driven updates.",
    task: 'classification',
    capabilities: ['forward', 'decide', 'score'],
    hyper: space({
      epochs: int(1, 1000, { default: 100 }),
      learningRate: real(1e-3, 10, { default: 1, scale: 'log' }),
      intercept: bool({ default: true }),
      shuffle: bool(),
      average: bool(),
    }),
    notes: ['perceptron'],
    cite: ['rosenblatt1958'],
  },
  perceptron,
)
