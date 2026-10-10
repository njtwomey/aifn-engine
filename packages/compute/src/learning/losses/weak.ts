/**
 * Losses for weak supervision, where the labels are not the classes of the examples: positive–unlabelled risks (the
 * unbiased uPU risk of du Plessis, Niu and Sugiyama, and the non-negative nnPU risk of Kiryo et al.), the proportion
 * loss of learning from label proportions (bags labelled only by their class shares), and losses for complementary
 * labels (a class each example does not belong to). Each is a composition of primitives, so it differentiates in the
 * model's outputs.
 */

import { logSoftmax, sigmoid, softmax, softplus } from 'aifn-compute/numerics/special'
import {
  add,
  fromData,
  log,
  logsumexp,
  matmul,
  maximum,
  mean,
  mul,
  neg,
  shapeOfValue,
  sub,
  sum,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { constant, defineLoss, flatValues, oneHot, reduce, type ReductionOptions, type Target } from './core'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// ── Positive–unlabelled risks ────────────────────────────────────────────────────────────────────────────────────────

/**
 * The surrogate $\ell(z, y)$ of the 0–1 loss used inside the PU risks, for a score $z$ and a label $y \in \{-1, +1\}$:
 * `sigmoid` $\ell = \sigma(-yz)$ (bounded, as Kiryo et al. recommend) or `logistic` $\ell = \log(1 + e^{-yz})$.
 */
export type PuSurrogate = 'sigmoid' | 'logistic'

/** Options of `unbiasedPu` and `nonNegativePu`. */
export type PuOptions = {
  /** The class prior $\pi = P(y = +1)$, known or estimated; it must lie in $(0, 1)$. */
  prior: number
  /** The surrogate loss (default `sigmoid`). */
  surrogate?: PuSurrogate
}

/**
 * The surrogate loss $\ell(z, y)$, elementwise.
 *
 * @param z The scores.
 * @param y The label every score is scored against, $+1$ or $-1$.
 * @param kind Which surrogate.
 * @returns $\ell(z, y)$, shaped like `z`.
 */
const surrogate = (z: Value, y: 1 | -1, kind: PuSurrogate): Value =>
  kind === 'sigmoid' ? sigmoid(y === 1 ? neg(z) : z) : softplus(y === 1 ? neg(z) : z)

/**
 * Throw `DomainError` unless the class prior lies in $(0, 1)$.
 *
 * @param prior The class prior $\pi$.
 * @param where The caller's name, for the error message.
 */
function checkPrior(prior: number, where: string) {
  if (!(prior > 0 && prior < 1))
    throw new DomainError(where, `${where}: the class prior must lie in (0, 1), got ${prior}`)
}

/**
 * The three empirical risks of the PU decomposition: $R_P^+$ and $R_P^-$ (the labelled positives scored as positive
 * and as negative) and $R_U^-$ (the unlabelled scored as negative), each a mean of the surrogate.
 *
 * @param positive The scores of the labelled positives.
 * @param unlabelled The scores of the unlabelled examples.
 * @param kind Which surrogate.
 * @returns `positivePlus` $R_P^+$, `positiveMinus` $R_P^-$ and `unlabelledMinus` $R_U^-$.
 */
function puRisks(positive: Value, unlabelled: Value, kind: PuSurrogate) {
  return {
    positivePlus: mean(surrogate(positive, 1, kind)),
    positiveMinus: mean(surrogate(positive, -1, kind)),
    unlabelledMinus: mean(surrogate(unlabelled, -1, kind)),
  }
}

/**
 * The unbiased PU risk (du Plessis, Niu and Sugiyama, 2014, 2015): with scores $g(\xvec)$ on labelled positives and on
 * unlabelled examples drawn from the marginal, $R(g) = \pi R_P^+(g) + R_U^-(g) - \pi R_P^-(g)$, where $R_P^{\pm}$ is
 * the mean surrogate loss of the positives labelled $\pm 1$ and $R_U^-$ that of the unlabelled labelled $-1$. Its
 * expectation is the positive–negative risk, because the unlabelled mean of $\ell(g, -1)$ counts the hidden positives
 * with weight $\pi$, which the last term removes. With a flexible model the estimate goes negative (overfitting); see
 * `nonNegativePu`. A prior outside $(0, 1)$ throws `DomainError`.
 *
 * @param positive The scores $g(\xvec)$ of the labelled positives, of any shape.
 * @param unlabelled The scores of the unlabelled examples, of any shape.
 * @param options The class prior $\pi$ and the surrogate.
 * @returns The risk, a number (or a traced scalar).
 *
 * @example Negative when the unlabelled positives are fitted as negatives
 * const options = { prior: 0.5 }
 * print('a fair fit:', unbiasedPu(tensor([2, 2]), tensor([2, -2]), options))
 * print('every unlabelled as negative:', unbiasedPu(tensor([5, 5]), tensor([-5, -5]), options))
 */
export const unbiasedPu = defineLoss(
  {
    key: 'unbiasedPu',
    name: 'Unbiased PU risk (uPU)',
    family: 'classification',
    inputs: 'logits',
    notes: ['positive-unlabelled-learning', 'non-negative-positive-unlabelled-learning'],
    target: 'the score of the positive class',
  },
  (positive: Value, unlabelled: Value, { prior, surrogate: kind = 'sigmoid' }: PuOptions): Value => {
    checkPrior(prior, 'unbiasedPu')
    const r = puRisks(positive, unlabelled, kind)
    return add(mul(prior, r.positivePlus), sub(r.unlabelledMinus, mul(prior, r.positiveMinus)))
  },
)

/** Options of `nonNegativePu`. */
export type NonNegativePuOptions = PuOptions & {
  /**
   * Return the training objective of Kiryo et al.'s algorithm rather than the risk: when the negative-class part
   * $R_U^- - \pi R_P^-$ falls below $-\beta$, the objective is $-\gamma (R_U^- - \pi R_P^-)$, whose gradient step
   * pushes it back up; otherwise it is $\pi R_P^+ + R_U^- - \pi R_P^-$, unclipped. Default false (the risk itself).
   */
  training?: boolean
  /** The tolerance $\beta \ge 0$ below zero allowed before the correction (default 0). */
  beta?: number
  /** The correction's step factor $\gamma \in (0, 1]$ (default 1). */
  gamma?: number
}

/**
 * The non-negative PU risk (Kiryo, Niu, du Plessis and Sugiyama, 2017):
 * $R(g) = \pi R_P^+(g) + \max(0, R_U^-(g) - \pi R_P^-(g))$. The second term estimates the negative class's risk
 * $(1 - \pi) R_N^-$, which cannot be negative; clipping it at zero stops a flexible model from driving the unbiased
 * estimate below zero by fitting the unlabelled positives as negatives. With `training: true` it returns the objective
 * of their algorithm instead (see `NonNegativePuOptions`). A prior outside $(0, 1)$ throws `DomainError`.
 *
 * @param positive The scores $g(\xvec)$ of the labelled positives, of any shape.
 * @param unlabelled The scores of the unlabelled examples, of any shape.
 * @param options The class prior $\pi$, the surrogate, and the training objective's switch, $\beta$ and $\gamma$.
 * @returns The risk (or the training objective), a number (or a traced scalar).
 *
 * @example The overfitted case of `unbiasedPu`, clipped
 * const [pos, unl] = [tensor([5, 5]), tensor([-5, -5])]
 * print('uPU:', unbiasedPu(pos, unl, { prior: 0.5 }))
 * print('nnPU:', nonNegativePu(pos, unl, { prior: 0.5 }))
 * print('training objective:', nonNegativePu(pos, unl, { prior: 0.5, training: true }))
 */
export const nonNegativePu = defineLoss(
  {
    key: 'nonNegativePu',
    name: 'Non-negative PU risk (nnPU)',
    family: 'classification',
    inputs: 'logits',
    notes: ['non-negative-positive-unlabelled-learning', 'positive-unlabelled-learning'],
    target: 'the score of the positive class',
  },
  (positive: Value, unlabelled: Value, options: NonNegativePuOptions): Value => {
    const { prior, surrogate: kind = 'sigmoid', training = false, beta = 0, gamma = 1 } = options
    checkPrior(prior, 'nonNegativePu')
    const r = puRisks(positive, unlabelled, kind)
    const negative = sub(r.unlabelledMinus, mul(prior, r.positiveMinus))
    const positivePart = mul(prior, r.positivePlus)
    if (!training) return add(positivePart, maximum(negative, 0))
    const value = flatValues(negative)[0]
    return value < -beta ? mul(-gamma, negative) : add(positivePart, negative)
  },
)

// ── Learning from label proportions ──────────────────────────────────────────────────────────────────────────────────

/** Options of `proportionLoss`. */
export type ProportionLossOptions = ReductionOptions & {
  /** Added to the bag's mean probabilities inside the log (default 1e-12), so an empty class gives no $-\infty$. */
  epsilon?: number
}

/**
 * The proportion loss of learning from label proportions (Ardehaly and Culotta, 2017; Tsai and Lin, 2020): instance
 * logits $\zvec_i$ (rows of `[n, K]`) are turned into probabilities, averaged within each bag $b$ into
 * $\bar{\pvec}_b = \tfrac{1}{\lvert b \rvert} \sum_{i \in b} \operatorname{softmax}(\zvec_i)$, and compared with the
 * bag's known class proportions $\pivec_b$ by the cross-entropy $-\sum_k \pi_{bk} \log \bar{p}_{bk}$. `bags` gives each
 * instance's bag index in $0, \dots, B - 1$ and `proportions` is `[B, K]`. The reduction is over bags. Logits that are
 * not a matrix, a bag id per instance missing, or an id outside $[0, B)$ throw.
 *
 * @param logits The instance logits, `[n, K]`.
 * @param bags Each instance's bag index, $n$ integers in $[0, B)$; constants.
 * @param proportions The class proportions of each bag, `[B, K]` (rows summing to one); constants. $B$ is read from
 *   its first axis.
 * @param options The $\varepsilon$ inside the log, and the reduction (over bags).
 * @returns The loss, reduced over bags (mean by default).
 *
 * @example When a bag's mean prediction matches its proportions, its loss is their entropy
 * const logits = tensor([[2, 0], [0, 2], [Math.log(9), 0], [Math.log(9), 0]])
 * const proportions = tensor([[0.5, 0.5], [0.9, 0.1]])
 * print('per bag:', proportionLoss(logits, [0, 0, 1, 1], proportions, { reduction: 'none' }))
 * print('entropies:', Math.log(2), -(0.9 * Math.log(0.9) + 0.1 * Math.log(0.1)))
 */
export const proportionLoss = defineLoss(
  {
    key: 'proportionLoss',
    name: 'Proportion loss (LLP)',
    family: 'classification',
    inputs: 'logits',
    notes: ['learning-from-label-proportions'],
    target: 'the class logits of each instance',
  },
  (logits: Value, bags: Target, proportions: Target, options: ProportionLossOptions = {}): Value => {
    const { reduction, epsilon = 1e-12 } = options
    const shape = shapeOfValue(logits)
    if (shape.length !== 2) throw new ShapeError('proportionLoss', 'proportionLoss: logits must be [n, K]')
    const [n, K] = shape
    const ids = flatValues(bags)
    if (ids.length !== n)
      throw new ShapeError('proportionLoss', `proportionLoss: ${ids.length} bag ids for ${n} instances`)
    const target = constant(proportions)
    const B = typeof target === 'number' ? 1 : target.shape[0]
    // The averaging matrix A [B, n] with A_bi = 1/|b| for i in b: a constant, so p̄ = A softmax(z) differentiates.
    const counts = new Float64Array(B)
    for (const b of ids) {
      if (!Number.isInteger(b) || b < 0 || b >= B)
        throw new DomainError('proportionLoss', `proportionLoss: bag id ${b} not in [0, ${B})`)
      counts[b]++
    }
    const A = new Float64Array(B * n)
    ids.forEach((b, i) => (A[b * n + i] = 1 / counts[b]))
    const pbar = matmul(fromData(A, [B, n]), softmax(logits))
    if (K < 1) throw new DomainError('proportionLoss', 'proportionLoss: no classes')
    return reduce(neg(sum(mul(target, log(add(pbar, epsilon))), -1)), reduction)
  },
)

// ── Complementary labels ─────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `complementaryLabelLoss`. */
export type ComplementaryLabelOptions = ReductionOptions & {
  /**
   * `unbiased`: the unbiased risk estimator with cross-entropy of Ishida et al. (2019),
   * $-(K - 1) \ell(\zvec, \bar{y}) + \sum_k \ell(\zvec, k)$ with $\ell(\zvec, k) = -\log p_k$; `forward`: the forward
   * correction under uniform complements (Yu et al., 2018), $-\log((1 - p_{\bar{y}}) / (K - 1))$, the cross-entropy of
   * the complementary-label distribution the model implies. Default `forward`.
   */
  method?: 'unbiased' | 'forward'
}

/**
 * Losses for complementary labels: each example comes with a class $\bar{y}$ it does not belong to, drawn uniformly
 * from the other $K - 1$ classes. Logits are `[n, K]` and `complementary` holds $\bar{y}$ for each row, with
 * $\pvec = \operatorname{softmax}(\zvec)$. The `unbiased` form equals the ordinary cross-entropy risk in expectation
 * over $\bar{y}$ but goes negative per example; the `forward` form is bounded below. Logits that are not a matrix, or
 * fewer than two classes, throw.
 *
 * @param logits The logits $\zvec$, `[n, K]`.
 * @param complementary The complementary label $\bar{y}$ of each row, an integer in $[0, K)$; constants.
 * @param options The method (`forward` or `unbiased`) and the reduction (over rows).
 * @returns The loss, reduced over rows (mean by default).
 *
 * @example Uniform logits cost log K by either method
 * const z = tensor([[0, 0, 0]])
 * print('forward:', complementaryLabelLoss(z, [0]))
 * print('unbiased:', complementaryLabelLoss(z, [0], { method: 'unbiased' }))
 * print('log 3 =', Math.log(3))
 *
 * @example Avoiding the complementary class: the unbiased form goes negative, the forward one stays above log(K - 1)
 * const z = tensor([[5, 0, 5]])
 * print('forward:', complementaryLabelLoss(z, [1]), ' log 2 =', Math.log(2))
 * print('unbiased:', complementaryLabelLoss(z, [1], { method: 'unbiased' }))
 */
export const complementaryLabelLoss = defineLoss(
  {
    key: 'complementaryLabelLoss',
    name: 'Complementary-label loss',
    family: 'classification',
    inputs: 'logits',
    notes: ['complementary-labels'],
    target: 'the class logits',
  },
  (logits: Value, complementary: Target, { reduction, method = 'forward' }: ComplementaryLabelOptions = {}): Value => {
    const shape = shapeOfValue(logits)
    if (shape.length !== 2)
      throw new ShapeError('complementaryLabelLoss', 'complementaryLabelLoss: logits must be [n, K]')
    const K = shape[1]
    if (K < 2) throw new DomainError('complementaryLabelLoss', 'complementaryLabelLoss: needs at least two classes')
    const bar = oneHot(complementary, K)
    const logp = logSoftmax(logits)
    if (method === 'unbiased') {
      // ℓ(z, k) = −log p_k, so −(K − 1)ℓ(z, ȳ) + Σ_k ℓ(z, k) = (K − 1) log p_ȳ − Σ_k log p_k.
      const atBar = sum(mul(bar, logp), -1)
      return reduce(sub(mul(K - 1, atBar), sum(logp, -1)), reduction)
    }
    // log(1 − p_ȳ) = logsumexp_{k ≠ ȳ} z_k − logsumexp_k z_k, which stays finite when p_ȳ rounds to 1 (forming
    // 1 − p_ȳ from the softmax gives log 0 = −∞ once z_ȳ leads the others by about 37).
    const mask = fromData(
      Float64Array.from(flatValues(bar), (v) => (v === 1 ? -Infinity : 0)),
      shapeOfValue(bar),
    )
    const logRest = sub(logsumexp(add(logits, mask), -1), logsumexp(logits, -1))
    return reduce(sub(Math.log(K - 1), logRest), reduction)
  },
)
