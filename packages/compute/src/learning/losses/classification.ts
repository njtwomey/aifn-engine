/**
 * Classification losses: binary cross-entropy (from logits or probabilities), softmax cross-entropy with label
 * smoothing, focal loss, the margin-based surrogates of the 0–1 loss (hinge, squared hinge, logistic, exponential,
 * modified Huber) and the multiclass hinges of Crammer–Singer and Weston–Watkins. Every loss is a composition of
 * `aifn-compute/foundation/tensor` and `aifn-compute/numerics/special` primitives, so it is differentiable (in the predictions) to any order the primitives
 * allow.
 */

import { logSoftmax, sigmoid, softplus } from 'aifn-compute/numerics/special'
import {
  add,
  exp,
  expandDims,
  less,
  lessEqual,
  log,
  log1p,
  max,
  maximum,
  mul,
  neg,
  pow,
  shapeOfValue,
  square,
  sub,
  sum,
  unwrap,
  where,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { constant, defineLoss, expectRank, oneHot, reduce, type ReductionOptions, type Target } from './core'

// ── Binary cross-entropy ─────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `binaryCrossEntropyWithLogits`. */
export type BinaryCrossEntropyOptions = ReductionOptions & {
  /** Weight w on the positive class's term (PyTorch's `pos_weight`), for class imbalance. Default 1. */
  positiveWeight?: number
}

/**
 * Binary cross-entropy (log loss) from logits z and targets y ∈ [0, 1] (0/1 labels or soft targets), elementwise over
 * broadcast shapes: ℓ = (1 − y)z + (1 + (w − 1)y)·softplus(−z), which for w = 1 is softplus(z) − yz = −y log σ(z) −
 * (1 − y) log(1 − σ(z)). Computed through softplus, so it is exact for logits of any size (no log of a rounded
 * probability). Matches `torch.nn.BCEWithLogitsLoss`.
 */
export const binaryCrossEntropyWithLogits = defineLoss(
  {
    key: 'binaryCrossEntropyWithLogits',
    name: 'Binary cross-entropy (logits)',
    family: 'classification',
    inputs: 'logits',
    notes: ['log-loss-and-brier-score'],
    target: 'the log-odds of y = 1',
  },
  (logits: Value, targets: Target, { reduction, positiveWeight = 1 }: BinaryCrossEntropyOptions = {}): Value => {
    const y = constant(targets)
    const weight = positiveWeight === 1 ? 1 : add(1, mul(positiveWeight - 1, y))
    return reduce(add(mul(sub(1, y), logits), mul(weight, softplus(neg(logits)))), reduction)
  },
)

/**
 * Binary cross-entropy from probabilities p ∈ [0, 1] and targets y: −y log p − (1 − y) log(1 − p), elementwise. Nothing
 * is clamped: p = 0 with y = 1 gives +∞ (PyTorch clamps the logs at −100). Prefer the logits form in training.
 */
export const binaryCrossEntropy = defineLoss(
  {
    key: 'binaryCrossEntropy',
    name: 'Binary cross-entropy (probabilities)',
    family: 'classification',
    inputs: 'probabilities',
    notes: ['log-loss-and-brier-score'],
    target: 'P(y = 1 | x)',
  },
  (probabilities: Value, targets: Target, { reduction }: ReductionOptions = {}): Value => {
    const y = constant(targets)
    // y log p with 0 log 0 = 0 on either side, so hard labels do not multiply 0 by −∞. The log's argument is also
    // replaced where its weight is 0 (the "double where"), so its derivative there is finite and the masked branch
    // contributes 0 to the gradient instead of 0·∞ = NaN.
    const term = (w: Value, l: (q: Value) => Value) => where(w, mul(w, l(where(w, probabilities, 0.5))), 0)
    return reduce(
      neg(
        add(
          term(y, log),
          term(sub(1, y), (q) => log1p(neg(q))),
        ),
      ),
      reduction,
    )
  },
)

// ── Softmax cross-entropy ────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `softmaxCrossEntropy`. */
export type SoftmaxCrossEntropyOptions = ReductionOptions & {
  /**
   * Label smoothing ε ∈ [0, 1) (Szegedy et al., 2016, §7): the target becomes (1 − ε)·onehot(y) + ε/K. Default 0.
   */
  labelSmoothing?: number
}

/**
 * The target rows of a softmax loss: integer labels (shape [...batch]) become one-hot rows; probability rows (shape
 * [...batch, K]) are used as given. Smoothing mixes in the uniform distribution.
 */
function targetRows(logits: Value, targets: Target, labelSmoothing: number): Value {
  const shape = shapeOfValue(logits)
  const K = shape[shape.length - 1]
  const t = constant(targets)
  const tShape = typeof t === 'number' ? [] : t.shape
  const rows = tShape.length === shape.length ? t : oneHot(targets, K)
  if (labelSmoothing === 0) return rows
  return add(mul(1 - labelSmoothing, rows), labelSmoothing / K)
}

/**
 * Softmax cross-entropy −Σₖ tₖ log softmax(z)ₖ from logits z (shape [K] or [n, K]) and targets t: integer class labels
 * (a number or shape [n]) or probability rows (shape like z, e.g. soft labels or a teacher's outputs), with optional
 * label smoothing. Log-softmax is computed stably, so logits of any size are safe. Matches
 * `torch.nn.CrossEntropyLoss` (including `label_smoothing`). Its gradient in z is softmax(z) − t.
 */
export const softmaxCrossEntropy = defineLoss(
  {
    key: 'softmaxCrossEntropy',
    name: 'Softmax cross-entropy',
    family: 'classification',
    inputs: 'logits',
    notes: ['cross-entropy-and-perplexity'],
    target: 'the class log-probabilities, up to a constant per example',
  },
  (logits: Value, targets: Target, { reduction, labelSmoothing = 0 }: SoftmaxCrossEntropyOptions = {}): Value => {
    expectRank(logits, [1, 2], 'softmaxCrossEntropy logits')
    const t = targetRows(logits, targets, labelSmoothing)
    return reduce(neg(sum(mul(t, logSoftmax(logits)), -1)), reduction)
  },
)

// ── Focal loss ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the focal losses. */
export type FocalOptions = ReductionOptions & {
  /** The focusing parameter γ ≥ 0; γ = 0 recovers cross-entropy. Default 2. */
  gamma?: number
  /**
   * Binary only: the weight α on positives (1 − α on negatives); `null` for none. Default 0.25, as in Lin et al. and
   * torchvision.
   */
  alpha?: number | null
}

/**
 * The binary focal loss (Lin et al., 2017, eq. 5) from logits z and 0/1 targets y: −α_t (1 − p_t)^γ log p_t, where
 * p_t = σ(z) for y = 1 and 1 − σ(z) for y = 0 (and α_t = α or 1 − α). The factor (1 − p_t)^γ down-weights examples
 * already classified well. Matches `torchvision.ops.sigmoid_focal_loss`.
 */
export const focalLoss = defineLoss(
  {
    key: 'focalLoss',
    name: 'Focal loss (binary)',
    family: 'classification',
    inputs: 'logits',
    notes: ['focal-loss'],
  },
  (logits: Value, targets: Target, { reduction, gamma = 2, alpha = 0.25 }: FocalOptions = {}): Value => {
    const y = constant(targets)
    const p = sigmoid(logits)
    const ce = add(mul(sub(1, y), logits), softplus(neg(logits)))
    const pt = add(mul(p, y), mul(sub(1, p), sub(1, y)))
    let loss = gamma === 0 ? ce : mul(ce, pow(sub(1, pt), gamma))
    if (alpha !== null) loss = mul(loss, add(mul(alpha, y), mul(1 - alpha, sub(1, y))))
    return reduce(loss, reduction)
  },
)

/**
 * The multiclass focal loss −(1 − p_y)^γ log p_y from logits (shape [K] or [n, K]) and integer labels, with
 * p = softmax(z). γ = 0 is softmax cross-entropy.
 */
export const softmaxFocalLoss = defineLoss(
  {
    key: 'softmaxFocalLoss',
    name: 'Focal loss (multiclass)',
    family: 'classification',
    inputs: 'logits',
    notes: ['focal-loss'],
  },
  (logits: Value, labels: Target, { reduction, gamma = 2 }: Omit<FocalOptions, 'alpha'> = {}): Value => {
    expectRank(logits, [1, 2], 'softmaxFocalLoss logits')
    const K = shapeOfValue(logits).at(-1)!
    const logPy = sum(mul(oneHot(labels, K), logSoftmax(logits)), -1)
    const weight = gamma === 0 ? 1 : pow(sub(1, exp(logPy)), gamma)
    return reduce(neg(mul(weight, logPy)), reduction)
  },
)

// ── Margin surrogates of the 0–1 loss ────────────────────────────────────────────────────────────────────────────────

/** The margin-based surrogates φ(m) of the 0–1 loss, as functions of the margin m = y·f(x). */
export type SurrogateName = 'zeroOne' | 'hinge' | 'squaredHinge' | 'logistic' | 'exponential' | 'modifiedHuber'

/**
 * φ(m) for each surrogate, elementwise in the margin m = y·f(x), y ∈ {−1, +1} (Bartlett, Jordan & McAuliffe, 2006):
 *
 * - `zeroOne`: 1[m ≤ 0] (piecewise constant, so its gradient is zero; not differentiable at 0).
 * - `hinge`: max(0, 1 − m) (the SVM).
 * - `squaredHinge`: max(0, 1 − m)².
 * - `logistic`: log₂(1 + e^{−m}), in bits so that it passes through (0, 1) and bounds the 0–1 loss.
 * - `exponential`: e^{−m} (AdaBoost).
 * - `modifiedHuber`: max(0, 1 − m)² for m ≥ −1, −4m below (Zhang, 2004).
 */
export const surrogates: Readonly<Record<SurrogateName, (margin: Value) => Value>> = {
  // 0 · m keeps the result on the tape (with its true, zero, derivative) when m is traced.
  zeroOne: (m) => add(mul(0, m), lessEqual(unwrap(m), 0)),
  hinge: (m) => maximum(sub(1, m), 0),
  squaredHinge: (m) => square(maximum(sub(1, m), 0)),
  logistic: (m) => mul(1 / Math.LN2, softplus(neg(m))),
  exponential: (m) => exp(neg(m)),
  modifiedHuber: (m) => where(less(unwrap(m), -1), mul(-4, m), square(maximum(sub(1, m), 0))),
}

/** Build a margin loss from its surrogate: the mean (or sum, or values) of φ(y·f) for scores f and labels y ∈ {±1}. */
function marginLoss(name: SurrogateName) {
  return (scores: Value, labels: Target, { reduction }: ReductionOptions = {}): Value =>
    reduce(surrogates[name](mul(constant(labels), scores)), reduction)
}

const marginInfo = (key: string, name: string) =>
  ({ key, name, family: 'classification', inputs: 'margins', notes: ['surrogate-losses'] }) as const

/** The hinge loss max(0, 1 − y·f) of scores f and labels y ∈ {−1, +1} (the soft-margin SVM). */
export const hinge = defineLoss(marginInfo('hinge', 'Hinge loss'), marginLoss('hinge'))
/** The squared hinge loss max(0, 1 − y·f)² of scores f and labels y ∈ {−1, +1}. */
export const squaredHinge = defineLoss(marginInfo('squaredHinge', 'Squared hinge loss'), marginLoss('squaredHinge'))
/** The logistic loss log₂(1 + e^{−y·f}) of scores f and labels y ∈ {−1, +1}, in bits. */
export const logisticLoss = defineLoss(marginInfo('logisticLoss', 'Logistic loss'), marginLoss('logistic'))
/** The exponential loss e^{−y·f} of scores f and labels y ∈ {−1, +1} (AdaBoost). */
export const exponentialLoss = defineLoss(marginInfo('exponentialLoss', 'Exponential loss'), marginLoss('exponential'))
/** The modified Huber loss of scores f and labels y ∈ {−1, +1} (Zhang, 2004). */
export const modifiedHuber = defineLoss(marginInfo('modifiedHuber', 'Modified Huber loss'), marginLoss('modifiedHuber'))

// ── Multiclass hinges ────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the multiclass hinges. */
export type MulticlassHingeOptions = ReductionOptions & {
  /** The margin Δ required between the true class and the others. Default 1. */
  margin?: number
}

/** Scores with the true class's entry replaced by −∞, and the true class's score, for [K] or [n, K] scores. */
function splitTrueClass(scores: Value, labels: Target): { others: Value; trueScore: Value } {
  const K = shapeOfValue(scores).at(-1)!
  const hot = oneHot(labels, K)
  return { others: where(hot, -Infinity, scores), trueScore: sum(mul(hot, scores), -1) }
}

/**
 * The Crammer–Singer multiclass hinge (Crammer & Singer, 2001): max(0, Δ + max_{j≠y} s_j − s_y), for scores s of
 * shape [K] or [n, K] and integer labels. It penalises only the most violating class.
 */
export const crammerSingerHinge = defineLoss(
  {
    key: 'crammerSingerHinge',
    name: 'Crammer–Singer multiclass hinge',
    family: 'classification',
    inputs: 'logits',
    notes: ['multiclass-support-vector-machines'],
  },
  (scores: Value, labels: Target, { reduction, margin = 1 }: MulticlassHingeOptions = {}): Value => {
    expectRank(scores, [1, 2], 'crammerSingerHinge scores')
    const { others, trueScore } = splitTrueClass(scores, labels)
    return reduce(maximum(sub(add(margin, max(others, -1)), trueScore), 0), reduction)
  },
)

/**
 * The Weston–Watkins multiclass hinge (Weston & Watkins, 1999): Σ_{j≠y} max(0, Δ + s_j − s_y), for scores of shape [K]
 * or [n, K] and integer labels. It penalises every violating class. (`torch.nn.MultiMarginLoss` is this divided by K.)
 */
export const westonWatkinsHinge = defineLoss(
  {
    key: 'westonWatkinsHinge',
    name: 'Weston–Watkins multiclass hinge',
    family: 'classification',
    inputs: 'logits',
    notes: ['multiclass-support-vector-machines'],
  },
  (scores: Value, labels: Target, { reduction, margin = 1 }: MulticlassHingeOptions = {}): Value => {
    expectRank(scores, [1, 2], 'westonWatkinsHinge scores')
    const { others, trueScore } = splitTrueClass(scores, labels)
    // −∞ at the true class gives max(0, −∞) = 0 there, so the sum runs over j ≠ y.
    const violations = maximum(sub(add(margin, others), expandDims(trueScore, -1)), 0)
    return reduce(sum(violations, -1), reduction)
  },
)
