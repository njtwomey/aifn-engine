/**
 * Classifiers trained from weak labels, each a linear model or a small MLP fitted through `aifn-compute/nn/training` on
 * a loss of `aifn-compute/learning/losses`: positive–unlabelled learning by Elkan and Noto's calibration (2008) or by
 * the unbiased and non-negative PU risks; learning from label proportions with the proportion loss; and learning from
 * complementary labels. Every fit returns the trained parameters, the loss along the way and a function giving class
 * probabilities for new points. Points are the rows of an $n \times d$ matrix; the fits are deterministic for a given
 * `seed`.
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
  /** Hidden widths of the MLP, whose activation is $\tanh$ (default none: a linear model). */
  hidden?: Size[]
  /** The most training steps (default 300); training ends sooner when the method stops. */
  steps?: Size
  /**
   * The training method (default full-batch L-BFGS; `puClassifier` with the nnPU risk defaults to Adam with step size
   * 0.05, as the non-negative correction is not smooth).
   */
  method?: TrainingMethod
  /** Seed of the initial weights and of the training streams (default 0). */
  seed?: number | string
}

/** A trained weak-label classifier. */
export type WeakClassifier = {
  /** The trained parameters, one entry per layer of the MLP. */
  params: Params[]
  /**
   * Class probabilities of new points (the rows of an $n \times d$ matrix): $n$ values of $p(y = 1 \mid \xvec)$ for
   * the binary (positive–unlabelled) models, $n \times K$ otherwise.
   */
  predict: (x: MatrixLike) => Tensor
  /** The training loss at the start and after each step. */
  losses: number[]
}

/**
 * A matrix argument as a float64 tensor $n \times d$ (a copy). A tensor that is not a real matrix, or ragged rows,
 * throw.
 *
 * @param x The matrix: a rank-2 tensor or an array of rows.
 * @param where The caller's name, for error messages.
 * @returns The matrix as a float64 tensor.
 */
const matrixOf = (x: MatrixLike, where: string) => {
  const m = dense.toMatrixF64(x, where)
  return fromData(Float64Array.from(m.data), [m.m, m.n])
}

/**
 * Train a network with `outputs` logits on `loss(logits)` (all points at once) and return it with its probabilities.
 * The network is an MLP with $\tanh$ hidden units, initialised from the stream of `seed`, trained for `steps` steps or
 * until the method stops.
 *
 * @param X The training points, $n \times d$.
 * @param outputs The number of logits per point: 1 for a binary model (probabilities by the sigmoid), $K$ for $K$
 *   classes (by the softmax).
 * @param loss The training loss of the logits of every point ($n \times$ `outputs`), a scalar.
 * @param options The hidden widths, steps, method and seed.
 * @param fallback The training method used when `options` names none.
 * @returns The trained classifier.
 */
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

/**
 * An Elkan–Noto fit: the non-traditional classifier $g(\xvec) = p(s = 1 \mid \xvec)$ (its `predict`), the label
 * frequency $c$ and the class prior.
 */
export type ElkanNoto = WeakClassifier & {
  /** $c = p(s = 1 \mid y = 1)$, estimated as the mean of $g$ over the labelled positives (NaN when there are none). */
  labelFrequency: number
  /** The class prior $p(y = 1) = p(s = 1)/c$, with $p(s = 1)$ the share of points labelled. */
  prior: number
  /** $p(y = 1 \mid \xvec) = \min(1, g(\xvec)/c)$ for each row of new points. */
  positive: (x: MatrixLike) => Float64Array
}

/**
 * Positive–unlabelled learning under "selected completely at random" (Elkan and Noto, 2008): train a classifier $g$ by
 * logistic loss to tell labelled ($s = 1$) from unlabelled ($s = 0$) points; since
 * $p(s = 1 \mid \xvec) = c \, p(y = 1 \mid \xvec)$ with $c$ the probability that a positive is labelled, $c$ is
 * estimated by the mean of $g$ over the labelled positives (their estimator $e_1$), and
 * $p(y = 1 \mid \xvec) = g(\xvec)/c$, capped at 1.
 *
 * @param x The points, $n \times d$.
 * @param labelled $s \in \{0, 1\}$ per point: 1 for a labelled positive, 0 for an unlabelled point.
 * @param options The model and its training (default: a linear model, 300 L-BFGS steps at most).
 * @returns The classifier $g$ with $c$, the class prior and the calibrated `positive`.
 *
 * @example Half of the positives labelled: recover the label frequency and the prior
 * const s = stream(5)
 * const xs = [...normals(s, 60, 2, 1).data, ...normals(s, 60, -2, 1).data]
 * const labelled = xs.map((_, i) => (i < 60 && uniform(s) < 0.5 ? 1 : 0))
 * const fit = elkanNoto(xs.map((v) => [v]), labelled)
 * print('share of positives labelled:', sum(tensor(labelled)) / 60, ' c =', fit.labelFrequency)
 * print('true prior:', 0.5, ' estimated:', fit.prior)
 * print('p(s = 1 | x) at x = -2, 0, 2:', fit.predict([[-2], [0], [2]]))
 * print('p(y = 1 | x) at x = -2, 0, 2:', fit.positive([[-2], [0], [2]]))
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
  /** The class prior $\pi = p(y = 1)$, known or from `elkanNoto`; it must lie in $(0, 1)$ unless `risk` is `naive`. */
  prior: number
  /** `unbiased` (uPU), `non-negative` (nnPU) or `naive` (unlabelled treated as negative). Default `non-negative`. */
  risk?: 'unbiased' | 'non-negative' | 'naive'
}

/**
 * A classifier trained on positive and unlabelled data by minimising a PU risk of `aifn-compute/learning/losses` (uPU
 * of du Plessis et al., nnPU of Kiryo et al.) with the sigmoid surrogate, or by treating the unlabelled points as
 * negatives under logistic loss (`naive`, biased towards the negative class). nnPU is trained by Adam on its training
 * objective unless `method` says otherwise. A prior outside $(0, 1)$ throws `DomainError` for the PU risks.
 *
 * @param x The points, $n \times d$.
 * @param labelled 1 for a labelled positive, anything else for an unlabelled point.
 * @param options The class prior, the risk, and the model and its training.
 * @returns The classifier; its `predict` gives $p(y = 1 \mid \xvec)$ for new points.
 *
 * @example Naive, unbiased and non-negative PU on the same data
 * const s = stream(6)
 * const xs = [...normals(s, 40, 2, 1).data, ...normals(s, 40, -2, 1).data]
 * const labelled = xs.map((_, i) => (i < 40 && uniform(s) < 0.5 ? 1 : 0))
 * const x = xs.map((v) => [v])
 * for (const risk of ['naive', 'unbiased', 'non-negative']) {
 *   const p = puClassifier(x, labelled, { prior: 0.5, risk }).predict([[-2], [0], [2]])
 *   print(risk, ' p(y = 1 | x) at x = -2, 0, 2:', p)
 * }
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
 * Learning from label proportions: a softmax classifier over $K$ classes trained on bags of instances, each labelled
 * only by its class proportions, by the proportion loss (the cross-entropy of each bag's proportions against its mean
 * predicted distribution, averaged over bags). A bag index outside $0, \dots, B - 1$ throws.
 *
 * @param x The instances, $n \times d$.
 * @param bags Each instance's bag index in $0, \dots, B - 1$.
 * @param proportions The bags' class proportions, $B \times K$, each row summing to 1; $K$ is its number of columns.
 * @param options The model and its training.
 * @returns The classifier; its `predict` gives $n \times K$ class probabilities.
 *
 * @example Four bags with known shares of class 1, and no instance label
 * const s = stream(7)
 * const shares = [0.9, 0.7, 0.3, 0.1]
 * const [x, y, bags] = [[], [], []]
 * shares.forEach((p, b) => {
 *   for (let i = 0; i < 20; i++) {
 *     y.push(i < Math.round(20 * p) ? 1 : 0)
 *     x.push([normal(s, y.at(-1) === 1 ? 2 : -2, 1)])
 *     bags.push(b)
 *   }
 * })
 * const fit = proportionClassifier(x, bags, shares.map((p) => [1 - p, p]))
 * print('p(class 0), p(class 1) at x = -2, 2:', fit.predict([[-2], [2]]))
 * const p1 = fit.predict(x).data.filter((_, i) => i % 2 === 1)
 * print('accuracy on the labels the bags hid:', y.filter((c, i) => (p1[i] > 0.5 ? 1 : 0) === c).length / y.length)
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
 * Learning from complementary labels: a softmax classifier over $K$ classes trained when each example comes with a
 * class it does not belong to, by the forward-corrected or the unbiased complementary-label loss.
 *
 * @param x The points, $n \times d$.
 * @param complementary For each point, a class in $0, \dots, K - 1$ that it is known not to belong to.
 * @param classes The number of classes $K$ (at least 2).
 * @param options The model and its training, and `loss`, the complementary-label loss: `forward` (the default, bounded
 *   below) or `unbiased`.
 * @returns The classifier; its `predict` gives $n \times K$ class probabilities.
 *
 * @example Three classes on a line, each point told only one class it is not
 * const s = stream(8)
 * const centres = [-3, 0, 3]
 * const [x, not] = [[], []]
 * for (let i = 0; i < 90; i++) {
 *   const y = i % 3
 *   x.push([normal(s, centres[y], 0.7)])
 *   not.push((y + (uniform(s) < 0.5 ? 1 : 2)) % 3)
 * }
 * const fit = complementaryClassifier(x, not, 3)
 * print('class probabilities at x = -3, 0, 3:', fit.predict([[-3], [0], [3]]))
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
