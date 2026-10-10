/**
 * Classification losses: binary cross-entropy (from logits or probabilities), softmax cross-entropy with label
 * smoothing, focal loss, the margin-based surrogates of the 0–1 loss (hinge, squared hinge, logistic, exponential,
 * modified Huber) and the multiclass hinges of Crammer–Singer and Weston–Watkins. Every loss is a composition of
 * `aifn-compute/foundation/tensor` and `aifn-compute/numerics/special` primitives, so it is differentiable (in the
 * predictions) to any order the primitives allow.
 *
 * Logits $z$ are unnormalised log-odds (binary) or log-probabilities up to a constant per row (multiclass); labels are
 * 0/1 for the binary losses, $\{-1, +1\}$ for the margin losses and class indices in $[0, K)$ for the multiclass ones.
 * Each returns the mean over examples by default, as PyTorch does.
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
  /** Weight $w$ on the positive class's term (PyTorch's `pos_weight`), for class imbalance. Default 1. */
  positiveWeight?: number
}

/**
 * Binary cross-entropy (log loss) from logits $z$ and targets $y \in [0, 1]$ (0/1 labels or soft targets),
 * elementwise over broadcast shapes:
 * $\ell = (1 - y) z + (1 + (w - 1) y) \operatorname{softplus}(-z)$, which for $w = 1$ is
 * $\operatorname{softplus}(z) - yz = -y \log \sigma(z) - (1 - y) \log(1 - \sigma(z))$. Computed through softplus, so
 * it is exact for logits of any size (no log of a rounded probability). Matches `torch.nn.BCEWithLogitsLoss`.
 *
 * @param logits The logits $z$, of any shape.
 * @param targets The targets $y$ in $[0, 1]$, broadcast against `logits`; constants.
 * @param options The positive-class weight $w$ and the reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example A logit of 0 costs log 2 whatever the label
 * print('per example:', binaryCrossEntropyWithLogits(tensor([0, 2]), tensor([1, 0]), { reduction: 'none' }))
 * print('log 2 =', Math.log(2), ' softplus(2) =', Math.log(1 + Math.exp(2)))
 *
 * @example Weighting the positive class
 * print('w = 3:', binaryCrossEntropyWithLogits(0, 1, { positiveWeight: 3 }), ' 3 log 2 =', 3 * Math.log(2))
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
 * Binary cross-entropy from probabilities $p \in [0, 1]$ and targets $y$: $-y \log p - (1 - y) \log(1 - p)$,
 * elementwise, with $0 \log 0 = 0$ so hard labels are exact at $p = 0$ or $1$. Nothing is clamped: $p = 0$ with
 * $y = 1$ gives $+\infty$ (PyTorch clamps the logs at $-100$). Prefer the logits form in training.
 *
 * @param probabilities The predicted probabilities $p$ of the positive class, of any shape.
 * @param targets The targets $y$ in $[0, 1]$, broadcast against `probabilities`; constants.
 * @param options The reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example Against the negative log of the probability given to the true class
 * print('loss =', binaryCrossEntropy(tensor([0.9, 0.2]), tensor([1, 0]), { reduction: 'none' }))
 * print('-log 0.9 =', -Math.log(0.9), ' -log 0.8 =', -Math.log(0.8))
 *
 * @example A certain wrong prediction costs infinity
 * print('p = 0, y = 1:', binaryCrossEntropy(0, 1))
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
   * Label smoothing $\varepsilon \in [0, 1)$ (Szegedy et al., 2016, §7): the target row $\tvec$ becomes
   * $(1 - \varepsilon) \tvec + \varepsilon / K$. Default 0.
   */
  labelSmoothing?: number
}

/**
 * The target rows of a softmax loss: integer labels (shape `[...batch]`) become one-hot rows; probability rows (shape
 * `[...batch, K]`, the rank of the logits) are used as given. Smoothing mixes in the uniform distribution.
 *
 * @param logits The logits; only their shape is read (the rank, and $K$ from the last axis).
 * @param targets Integer labels, or probability rows with the logits' rank.
 * @param labelSmoothing The weight $\varepsilon$ of the uniform distribution, 0 for none.
 * @returns The rows $(1 - \varepsilon) \tvec + \varepsilon / K$, shape like the logits.
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
 * Softmax cross-entropy $-\sum_k t_k \log \operatorname{softmax}(\zvec)_k$ from logits $\zvec$ (shape `[K]` or
 * `[n, K]`) and targets $\tvec$: integer class labels (a number or shape `[n]`) or probability rows (shape like the
 * logits, e.g. soft labels or a teacher's outputs), with optional label smoothing. Log-softmax is computed stably, so
 * logits of any size are safe. Matches `torch.nn.CrossEntropyLoss` (including `label_smoothing`). Its gradient in
 * $\zvec$ is $\operatorname{softmax}(\zvec) - \tvec$. Logits of another rank throw `ShapeError`.
 *
 * @param logits The logits $\zvec$, shape `[K]` or `[n, K]`.
 * @param targets Integer labels in $[0, K)$ (one per row), or probability rows of the logits' shape; constants.
 * @param options The label smoothing $\varepsilon$ and the reduction (over rows).
 * @returns The loss, reduced over rows (mean by default).
 *
 * @example Cross-entropy of a one-hot target against known probabilities
 * // Logits equal to log-probabilities: the loss is -log of the true class's probability.
 * print('loss =', softmaxCrossEntropy(log(tensor([0.7, 0.2, 0.1])), 0))
 * print('-log 0.7 =', -Math.log(0.7))
 *
 * @example The gradient is softmax(z) minus the target
 * print('grad =', grad((z) => softmaxCrossEntropy(z, 0))(tensor([0, 0, 0])))
 *
 * @example A batch with label smoothing
 * const z = tensor([[2, 0, 0], [0, 0, 0]])
 * print('plain:', softmaxCrossEntropy(z, [0, 1], { reduction: 'none' }))
 * print('smoothed:', softmaxCrossEntropy(z, [0, 1], { reduction: 'none', labelSmoothing: 0.1 }))
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
  /** The focusing parameter $\gamma \ge 0$; $\gamma = 0$ recovers cross-entropy. Default 2. */
  gamma?: number
  /**
   * Binary only: the weight $\alpha$ on positives ($1 - \alpha$ on negatives); `null` for none. Default 0.25, as in
   * Lin et al. and torchvision.
   */
  alpha?: number | null
}

/**
 * The binary focal loss (Lin et al., 2017, eq. 5) from logits $z$ and 0/1 targets $y$:
 * $-\alpha_t (1 - p_t)^\gamma \log p_t$, where $p_t = \sigma(z)$ for $y = 1$ and $1 - \sigma(z)$ for $y = 0$ (and
 * $\alpha_t = \alpha$ or $1 - \alpha$). The factor $(1 - p_t)^\gamma$ down-weights examples already classified well.
 * Elementwise over broadcast shapes. Matches `torchvision.ops.sigmoid_focal_loss`.
 *
 * @param logits The logits $z$, of any shape.
 * @param targets The 0/1 targets $y$, broadcast against `logits`; constants.
 * @param options $\gamma$, $\alpha$ and the reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example At a logit of 0 the loss is a quarter of the weighted cross-entropy
 * print('focal =', focalLoss(0, 1), ' 0.25 * 0.25 * log 2 =', 0.25 * 0.25 * Math.log(2))
 *
 * @example A confident correct prediction is down-weighted far more than an uncertain one
 * const z = tensor([3, 0])
 * print('focal:', focalLoss(z, 1, { alpha: null, reduction: 'none' }))
 * print('cross-entropy:', binaryCrossEntropyWithLogits(z, 1, { reduction: 'none' }))
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
 * The multiclass focal loss $-(1 - p_y)^\gamma \log p_y$ from logits $\zvec$ (shape `[K]` or `[n, K]`) and integer
 * labels $y$, with $\pvec = \operatorname{softmax}(\zvec)$. $\gamma = 0$ is softmax cross-entropy.
 *
 * @param logits The logits $\zvec$, shape `[K]` or `[n, K]`.
 * @param labels The class labels $y$ in $[0, K)$, one per row; constants.
 * @param options $\gamma$ and the reduction.
 * @returns The loss, reduced over rows (mean by default).
 *
 * @example With the true class at probability one half
 * const z = log(tensor([0.5, 0.25, 0.25]))
 * print('focal =', softmaxFocalLoss(z, 0), ' 0.5^2 log 2 =', 0.25 * Math.log(2))
 * print('gamma = 0:', softmaxFocalLoss(z, 0, { gamma: 0 }), ' cross-entropy:', softmaxCrossEntropy(z, 0))
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

/** The margin-based surrogates $\phi(m)$ of the 0–1 loss, as functions of the margin $m = y f(\xvec)$. */
export type SurrogateName = 'zeroOne' | 'hinge' | 'squaredHinge' | 'logistic' | 'exponential' | 'modifiedHuber'

/**
 * $\phi(m)$ for each surrogate, elementwise in the margin $m = y f(\xvec)$, $y \in \{-1, +1\}$ (Bartlett, Jordan &
 * McAuliffe, 2006):
 *
 * - `zeroOne`: $\indicator[m \le 0]$ (piecewise constant, so its gradient is zero; not differentiable at 0).
 * - `hinge`: $\max(0, 1 - m)$ (the SVM).
 * - `squaredHinge`: $\max(0, 1 - m)^2$.
 * - `logistic`: $\log_2(1 + e^{-m})$, in bits so that it passes through $(0, 1)$ and bounds the 0–1 loss.
 * - `exponential`: $e^{-m}$ (AdaBoost).
 * - `modifiedHuber`: $\max(0, 1 - m)^2$ for $m \ge -1$, $-4m$ below (Zhang, 2004).
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

/**
 * Build a margin loss from its surrogate: the mean (or sum, or values) of $\phi(y f)$ for scores $f$ and labels
 * $y \in \{-1, +1\}$.
 *
 * @param name The surrogate $\phi$, a key of `surrogates`.
 * @returns The loss function of scores, labels (constants, broadcast against the scores) and the reduction option.
 */
function marginLoss(name: SurrogateName) {
  return (scores: Value, labels: Target, { reduction }: ReductionOptions = {}): Value =>
    reduce(surrogates[name](mul(constant(labels), scores)), reduction)
}

/**
 * The registry metadata shared by the margin losses: family `classification`, inputs `margins`, note
 * `surrogate-losses`.
 *
 * @param key The loss's key, its export name.
 * @param name The loss's display name.
 * @returns The `LossSpec` to pass to `defineLoss`.
 */
const marginInfo = (key: string, name: string) =>
  ({ key, name, family: 'classification', inputs: 'margins', notes: ['surrogate-losses'] }) as const

/**
 * The hinge loss $\max(0, 1 - y f)$ of scores $f$ and labels $y \in \{-1, +1\}$ (the soft-margin SVM); its mean is
 * scikit-learn's `hinge_loss`.
 *
 * @param scores The real-valued scores $f(\xvec)$, of any shape.
 * @param labels The labels $y \in \{-1, +1\}$, broadcast against `scores`; constants.
 * @param options The reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example Zero beyond the margin, linear inside it
 * print(hinge(tensor([2, 0.5, -1]), tensor([1, 1, 1]), { reduction: 'none' }))
 */
export const hinge = defineLoss(marginInfo('hinge', 'Hinge loss'), marginLoss('hinge'))
/**
 * The squared hinge loss $\max(0, 1 - y f)^2$ of scores $f$ and labels $y \in \{-1, +1\}$.
 *
 * @param scores The real-valued scores $f(\xvec)$, of any shape.
 * @param labels The labels $y \in \{-1, +1\}$, broadcast against `scores`; constants.
 * @param options The reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example The hinge, squared
 * print(squaredHinge(tensor([2, 0.5, -1]), tensor([1, 1, 1]), { reduction: 'none' }))
 */
export const squaredHinge = defineLoss(marginInfo('squaredHinge', 'Squared hinge loss'), marginLoss('squaredHinge'))
/**
 * The logistic loss $\log_2(1 + e^{-y f})$ of scores $f$ and labels $y \in \{-1, +1\}$, in bits, so that it is 1 at
 * a margin of 0.
 *
 * @param scores The real-valued scores $f(\xvec)$, of any shape.
 * @param labels The labels $y \in \{-1, +1\}$, broadcast against `scores`; constants.
 * @param options The reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example One bit at a margin of 0
 * print(logisticLoss(tensor([0, 1, -1]), tensor([1, 1, 1]), { reduction: 'none' }))
 * print('log2(1 + e^-1) =', Math.log2(1 + Math.exp(-1)))
 */
export const logisticLoss = defineLoss(marginInfo('logisticLoss', 'Logistic loss'), marginLoss('logistic'))
/**
 * The exponential loss $e^{-y f}$ of scores $f$ and labels $y \in \{-1, +1\}$ (AdaBoost).
 *
 * @param scores The real-valued scores $f(\xvec)$, of any shape.
 * @param labels The labels $y \in \{-1, +1\}$, broadcast against `scores`; constants.
 * @param options The reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example A wrong sign costs exponentially
 * print(exponentialLoss(tensor([0, 1, 1]), tensor([1, 1, -1]), { reduction: 'none' }))
 */
export const exponentialLoss = defineLoss(marginInfo('exponentialLoss', 'Exponential loss'), marginLoss('exponential'))
/**
 * The modified Huber loss of scores $f$ and labels $y \in \{-1, +1\}$ (Zhang, 2004): with $m = y f$,
 * $\max(0, 1 - m)^2$ for $m \ge -1$ and $-4m$ below, as scikit-learn's `SGDClassifier(loss='modified_huber')`.
 *
 * @param scores The real-valued scores $f(\xvec)$, of any shape.
 * @param labels The labels $y \in \{-1, +1\}$, broadcast against `scores`; constants.
 * @param options The reduction.
 * @returns The loss, reduced over every entry (mean by default).
 *
 * @example Quadratic near the margin, linear far on the wrong side
 * print(modifiedHuber(tensor([2, 0, -2]), tensor([1, 1, 1]), { reduction: 'none' }))
 */
export const modifiedHuber = defineLoss(marginInfo('modifiedHuber', 'Modified Huber loss'), marginLoss('modifiedHuber'))

// ── Multiclass hinges ────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the multiclass hinges. */
export type MulticlassHingeOptions = ReductionOptions & {
  /** The margin $\Delta$ required between the true class and the others. Default 1. */
  margin?: number
}

/**
 * Scores with the true class's entry replaced by $-\infty$, and the true class's score, for `[K]` or `[n, K]` scores.
 *
 * @param scores The class scores $\svec$, shape `[K]` or `[n, K]`.
 * @param labels The true class $y$ of each row, an integer in $[0, K)$.
 * @returns `others`, the scores with $s_y$ replaced by $-\infty$, and `trueScore`, $s_y$ per row.
 */
function splitTrueClass(scores: Value, labels: Target): { others: Value; trueScore: Value } {
  const K = shapeOfValue(scores).at(-1)!
  const hot = oneHot(labels, K)
  return { others: where(hot, -Infinity, scores), trueScore: sum(mul(hot, scores), -1) }
}

/**
 * The Crammer–Singer multiclass hinge (Crammer & Singer, 2001): $\max(0, \Delta + \max_{j \ne y} s_j - s_y)$, for
 * scores $\svec$ of shape `[K]` or `[n, K]` and integer labels $y$. It penalises only the most violating class.
 *
 * @param scores The class scores $\svec$, shape `[K]` or `[n, K]`.
 * @param labels The true class $y$ of each row, an integer in $[0, K)$; constants.
 * @param options The margin $\Delta$ and the reduction.
 * @returns The loss, reduced over rows (mean by default).
 *
 * @example Only the closest rival counts
 * // Two rivals each 0.5 short of the margin: 1 + 2.5 - 3.
 * print('loss =', crammerSingerHinge(tensor([3, 2.5, 2.5]), 0))
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
 * The Weston–Watkins multiclass hinge (Weston & Watkins, 1999): $\sum_{j \ne y} \max(0, \Delta + s_j - s_y)$, for
 * scores $\svec$ of shape `[K]` or `[n, K]` and integer labels $y$. It penalises every violating class.
 * (`torch.nn.MultiMarginLoss` is this divided by $K$.)
 *
 * @param scores The class scores $\svec$, shape `[K]` or `[n, K]`.
 * @param labels The true class $y$ of each row, an integer in $[0, K)$; constants.
 * @param options The margin $\Delta$ and the reduction.
 * @returns The loss, reduced over rows (mean by default).
 *
 * @example Every rival inside the margin counts
 * // Two rivals each 0.5 short of the margin: 0.5 + 0.5; MultiMarginLoss would give 1/3.
 * print('loss =', westonWatkinsHinge(tensor([3, 2.5, 2.5]), 0))
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
