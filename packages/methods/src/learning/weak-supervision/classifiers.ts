/**
 * Classifiers trained from weak labels, each a linear model or a small MLP fitted through `aifn-compute/nn/training` on a loss
 * of `aifn-compute/learning/losses`: positive–unlabelled learning by Elkan and Noto's calibration (2008) or by the unbiased and
 * non-negative PU risks; learning from label proportions with the proportion loss; and learning from complementary
 * labels. Every fit returns the trained parameters and a function giving class probabilities for new points.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream as makeStream } from 'aifn-compute/foundation/random'
import { dense, fromData, reshape, take, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import {
  binaryCrossEntropyWithLogits,
  complementaryLabelLoss,
  nonNegativePu,
  proportionLoss,
  unbiasedPu,
} from 'aifn-compute/learning/losses'
import { Mlp } from 'aifn-compute/nn/layers'
import { methodTraining, type TrainingMethod } from 'aifn-compute/nn/training'
import { sigmoid, softmax } from 'aifn-compute/numerics/special'

/** The model of a weak-label classifier: hidden widths of an MLP (none: a linear model) and how it is trained. */
export type WeakModelOptions = {
  /** Hidden widths (default [] : linear). */
  hidden?: Size[]
  /** Training steps (default 300). */
  steps?: Size
  /** Default full-batch L-BFGS; PU training with nnPU defaults to Adam (the non-negative correction is not smooth). */
  method?: TrainingMethod
  /** Seed of the initial weights (default 0). */
  seed?: number | string
}

/** A trained weak-label classifier. */
export type WeakClassifier = {
  params: Params[]
  /** Class probabilities of new points: [n] (P(y = 1)) for binary models, [n, K] otherwise. */
  predict: (x: MatrixLike) => Tensor
  /** The training loss after each step. */
  losses: number[]
}

const matrixOf = (x: MatrixLike, where: string) => {
  const m = dense.toMatrixF64(x, where)
  return fromData(Float64Array.from(m.data), [m.m, m.n])
}

/** Train a network with `outputs` logits on `loss(logits)` (all points at once) and return it with its probabilities. */
function fit(
  X: Tensor,
  outputs: Size,
  loss: (logits: Value) => Value,
  options: WeakModelOptions,
  fallback: TrainingMethod = { method: 'lbfgs' },
): WeakClassifier {
  const { hidden = [], steps = 300, method = fallback, seed = 0 } = options
  const net = Mlp([X.shape[1], ...hidden, outputs], { activation: 'tanh' })
  const logits = (p: Params[], x: Value) => net.apply(p, x)
  const alg = methodTraining<Params[], Record<string, Tensor>>((p) => loss(logits(p, X)), { x: X }, method)
  const root = makeStream(seed)
  let s = alg.init({ params: net.init(child(root, 'init')) }, child(root, 'train'))
  const losses = [s.loss]
  for (let t = 0; t < steps && !s.stopped; t++) {
    s = alg.step(s, { t, stream: child(root, 'step', t) })
    losses.push(s.loss)
  }
  const params = s.params
  return {
    params,
    losses,
    predict: (x) => {
      const z = logits(params, matrixOf(x, 'predict')) as Tensor
      return (outputs === 1 ? sigmoid(reshape(z, [z.shape[0]])) : softmax(z)) as Tensor
    },
  }
}

// ── Positive–unlabelled learning ─────────────────────────────────────────────────────────────────────────────────────

/** An Elkan–Noto fit: the non-traditional classifier g(x) = P(s = 1 | x), the label frequency c and the class prior. */
export type ElkanNoto = WeakClassifier & {
  /** c = P(s = 1 | y = 1), estimated as the mean of g over the labelled positives. */
  labelFrequency: number
  /** The class prior P(y = 1) = P(s = 1)/c. */
  prior: number
  /** P(y = 1 | x) = min(1, g(x)/c). */
  positive: (x: MatrixLike) => Float64Array
}

/**
 * Positive–unlabelled learning under "selected completely at random" (Elkan and Noto, 2008): train a classifier g to
 * tell labelled (s = 1) from unlabelled (s = 0) points; since P(s = 1 | x) = c·P(y = 1 | x) with c the probability that a
 * positive is labelled, c is estimated by the mean of g over the labelled positives (their estimator e₁), and
 * P(y = 1 | x) = g(x)/c. `labelled` holds s ∈ {0, 1} per row.
 */
export function elkanNoto(x: MatrixLike, labelled: ArrayLike<number>, options: WeakModelOptions = {}): ElkanNoto {
  const X = matrixOf(x, 'elkanNoto')
  const s = fromData(Float64Array.from(labelled), [labelled.length])
  const g = fit(X, 1, (z) => binaryCrossEntropyWithLogits(reshape(z, [X.shape[0]]), s), options)
  const scores = toFlat(g.predict(X))
  let sum = 0
  let count = 0
  for (let i = 0; i < labelled.length; i++)
    if (labelled[i] === 1) {
      sum += scores[i]
      count++
    }
  const c = count > 0 ? sum / count : NaN
  const prior = count / labelled.length / c
  return {
    ...g,
    labelFrequency: c,
    prior,
    positive: (q) => Float64Array.from(toFlat(g.predict(q)), (v) => Math.min(1, v / c)),
  }
}

/** Options of `puClassifier`. */
export type PuClassifierOptions = WeakModelOptions & {
  /** The class prior π (known, or from `elkanNoto`). */
  prior: number
  /** `unbiased` (uPU), `non-negative` (nnPU) or `naive` (unlabelled treated as negative). Default `non-negative`. */
  risk?: 'unbiased' | 'non-negative' | 'naive'
}

/**
 * A classifier trained on positive and unlabelled data by minimising a PU risk of `aifn-compute/learning/losses` (uPU of du
 * Plessis et al., nnPU of Kiryo et al.) with the sigmoid surrogate, or by treating the unlabelled points as negatives
 * (`naive`, biased towards the negative class). nnPU is trained by Adam on its training objective.
 */
export function puClassifier(x: MatrixLike, labelled: ArrayLike<number>, options: PuClassifierOptions): WeakClassifier {
  const { prior, risk = 'non-negative' } = options
  const X = matrixOf(x, 'puClassifier')
  const pos: number[] = []
  const unl: number[] = []
  for (let i = 0; i < labelled.length; i++) (labelled[i] === 1 ? pos : unl).push(i)
  const n = X.shape[0]
  const s = fromData(Float64Array.from(labelled), [n])
  const loss = (z: Value) => {
    const g = reshape(z, [n])
    if (risk === 'naive') return binaryCrossEntropyWithLogits(g, s)
    const gp = take(g, pos)
    const gu = take(g, unl)
    return risk === 'unbiased' ? unbiasedPu(gp, gu, { prior }) : nonNegativePu(gp, gu, { prior, training: true })
  }
  const fallback: TrainingMethod = risk === 'non-negative' ? { method: 'adam', stepSize: 0.05 } : { method: 'lbfgs' }
  return fit(X, 1, loss, options, fallback)
}

// ── Label proportions and complementary labels ───────────────────────────────────────────────────────────────────────

/**
 * Learning from label proportions: a softmax classifier over K classes trained on bags of instances, each labelled only
 * by its class proportions, by the proportion loss (each bag's mean predicted distribution against its proportions).
 * `bags` gives each row's bag index and `proportions` is [B, K].
 */
export function proportionClassifier(
  x: MatrixLike,
  bags: ArrayLike<number>,
  proportions: MatrixLike,
  options: WeakModelOptions = {},
): WeakClassifier {
  const X = matrixOf(x, 'proportionClassifier')
  const P = matrixOf(proportions, 'proportionClassifier')
  return fit(X, P.shape[1], (z) => proportionLoss(z, Array.from(bags), P), options)
}

/**
 * Learning from complementary labels: a softmax classifier over K classes trained when each example comes with a class
 * it does not belong to, by the forward-corrected or the unbiased complementary-label loss.
 */
export function complementaryClassifier(
  x: MatrixLike,
  complementary: ArrayLike<number>,
  classes: Size,
  options: WeakModelOptions & { loss?: 'forward' | 'unbiased' } = {},
): WeakClassifier {
  const X = matrixOf(x, 'complementaryClassifier')
  const method = options.loss ?? 'forward'
  return fit(X, classes, (z) => complementaryLabelLoss(z, Array.from(complementary), { method }), options)
}
