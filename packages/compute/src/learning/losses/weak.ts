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
 * The surrogate ℓ(z, y) of the 0–1 loss used inside the PU risks, for a score z and a label y ∈ {−1, +1}: `sigmoid`
 * ℓ = σ(−yz) (bounded, as Kiryo et al. recommend) or `logistic` ℓ = log(1 + e^{−yz}).
 */
export type PuSurrogate = 'sigmoid' | 'logistic'

/** Options of `unbiasedPu` and `nonNegativePu`. */
export type PuOptions = {
  /** The class prior π = P(y = +1), known or estimated. */
  prior: number
  /** The surrogate loss (default `sigmoid`). */
  surrogate?: PuSurrogate
}

const surrogate = (z: Value, y: 1 | -1, kind: PuSurrogate): Value =>
  kind === 'sigmoid' ? sigmoid(y === 1 ? neg(z) : z) : softplus(y === 1 ? neg(z) : z)

function checkPrior(prior: number, where: string) {
  if (!(prior > 0 && prior < 1))
    throw new DomainError(where, `${where}: the class prior must lie in (0, 1), got ${prior}`)
}

/** The three empirical risks of the PU decomposition: R_P⁺, R_P⁻ (labelled positives) and R_U⁻ (unlabelled). */
function puRisks(positive: Value, unlabelled: Value, kind: PuSurrogate) {
  return {
    positivePlus: mean(surrogate(positive, 1, kind)),
    positiveMinus: mean(surrogate(positive, -1, kind)),
    unlabelledMinus: mean(surrogate(unlabelled, -1, kind)),
  }
}

/**
 * The unbiased PU risk (du Plessis, Niu and Sugiyama, 2014, 2015): with scores g(x) on labelled positives and on
 * unlabelled examples drawn from the marginal, R(g) = π R_P⁺(g) + R_U⁻(g) − π R_P⁻(g), where R_P^± is the mean surrogate
 * loss of the positives labelled ±1 and R_U⁻ that of the unlabelled labelled −1. Its expectation is the
 * positive–negative risk, because the unlabelled mean of ℓ(g, −1) counts the hidden positives with weight π, which the
 * last term removes. With a flexible model the estimate goes negative (overfitting); see `nonNegativePu`.
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
   * R_U⁻ − π R_P⁻ falls below −β, the objective is −γ (R_U⁻ − π R_P⁻), whose gradient step pushes it back up. Default
   * false (the risk itself).
   */
  training?: boolean
  /** The tolerance β ≥ 0 below zero allowed before the correction (default 0). */
  beta?: number
  /** The correction's step factor γ ∈ (0, 1] (default 1). */
  gamma?: number
}

/**
 * The non-negative PU risk (Kiryo, Niu, du Plessis and Sugiyama, 2017): R(g) = π R_P⁺(g) + max(0, R_U⁻(g) − π R_P⁻(g)).
 * The second term estimates the negative class's risk (1 − π) R_N⁻, which cannot be negative; clipping it at zero
 * stops a flexible model from driving the unbiased estimate below zero by fitting the unlabelled positives as negatives.
 * With `training: true` it returns the objective of their algorithm instead (see `NonNegativePuOptions`).
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
  /** Added to the bag's mean probabilities inside the log (default 1e-12), so an empty class does not give −∞. */
  epsilon?: number
}

/**
 * The proportion loss of learning from label proportions (Ardehaly and Culotta, 2017; Tsai and Lin, 2020): instance
 * logits z [n, K] are turned into probabilities, averaged within each bag b into p̄_b = (1/|b|) Σ_{i ∈ b} softmax(z_i),
 * and compared with the bag's known class proportions π_b by the cross-entropy −Σ_k π_bk log p̄_bk. `bags` gives each
 * instance's bag index in 0 … B − 1 and `proportions` is [B, K]. The reduction is over bags.
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
   * `unbiased`: the unbiased risk estimator with cross-entropy of Ishida et al. (2019), −(K − 1) ℓ(z, ȳ) + Σ_k ℓ(z, k);
   * `forward`: the forward correction under uniform complements (Yu et al., 2018), −log((1 − p_ȳ)/(K − 1)), the
   * cross-entropy of the complementary-label distribution the model implies. Default `forward`.
   */
  method?: 'unbiased' | 'forward'
}

/**
 * Losses for complementary labels: each example comes with a class ȳ it does not belong to, drawn uniformly from the
 * other K − 1 classes. Logits are [n, K] and `complementary` holds ȳ for each row. The `unbiased` form equals the
 * ordinary cross-entropy risk in expectation over ȳ but goes negative per example; the `forward` form is bounded below.
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
