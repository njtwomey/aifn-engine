/**
 * Losses for large output spaces and learned representations: sampled softmax with the logQ correction, negative
 * sampling and noise-contrastive estimation, in-batch softmax (InfoNCE itself is compute's), and the triplet and
 * contrastive (siamese) losses on embeddings. Scores and embeddings may be traced; sampling probabilities, labels and
 * masks are constants.
 */

import { softplus } from 'aifn-compute/numerics/special'
import {
  add,
  concat,
  expandDims,
  logsumexp,
  maximum,
  mul,
  neg,
  norm,
  sqrt,
  square,
  sub,
  sum,
  where,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { constantTarget as constant, expectRank, reduce } from 'aifn-compute/learning/losses'
import { defineLoss, type ReductionOptions, type Target } from 'aifn-compute/learning/losses'
import { infoNce, type InfoNceOptions } from 'aifn-compute/learning/losses'

// ── Sampled softmax ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `sampledSoftmax`. */
export type SampledSoftmaxOptions = ReductionOptions & {
  /**
   * log q_j, the log-probability with which each negative was drawn (shape like the negatives, or [m] shared by every
   * row). Given, each negative logit is shifted by −log(m·q_j), the logQ correction, so the loss estimates the full
   * softmax cross-entropy; omitted, the logits are used as they are (the uncorrected loss, which learns log p − log q).
   */
  logQ?: Target
  /** log q_y of each positive (shape [B]); given, the positive's logit is shifted by −log q_y too (TensorFlow's form). */
  positiveLogQ?: Target
  /** 1 to keep a negative, 0 to drop it (e.g. an accidental hit: a sampled negative equal to the positive). */
  mask?: Target
}

/**
 * Sampled softmax (Jean et al., 2015; Bengio & Senécal, 2008): the cross-entropy of the positive against itself and m
 * sampled negatives, −s_y + log(e^{s_y} + Σᵢ e^{s_{jᵢ} − log(m q_{jᵢ})}), for positive logits of shape [B] and negative
 * logits of shape [B, m]. With the logQ correction the sum estimates the partition function Z without bias; keeping
 * the positive in the denominator keeps the loss non-negative (Blanc & Rendle, 2018).
 */
export const sampledSoftmax = defineLoss(
  {
    module: 'applied/retrieval/losses',
    key: 'sampledSoftmax',
    name: 'Sampled softmax',
    family: 'retrieval',
    inputs: 'logits',
    notes: ['sampled-softmax-and-negative-sampling'],
    target: 'the full softmax cross-entropy (with the logQ correction)',
  },
  (positive: Value, negatives: Value, options: SampledSoftmaxOptions = {}): Value => {
    const { reduction, logQ, positiveLogQ, mask } = options
    const m = expectRank(negatives, [1, 2], 'sampledSoftmax negatives').at(-1)!
    let shifted = logQ === undefined ? negatives : sub(negatives, add(constant(logQ), Math.log(m)))
    if (mask !== undefined) shifted = where(constant(mask), shifted, -Infinity)
    const pos = positiveLogQ === undefined ? positive : sub(positive, constant(positiveLogQ))
    const all = concat([expandDims(pos, -1), shifted], -1)
    return reduce(sub(logsumexp(all, -1), pos), reduction)
  },
)

// ── Negative sampling and NCE ────────────────────────────────────────────────────────────────────────────────────────

/**
 * Negative sampling (Mikolov et al., 2013): logistic regression of the positive against k sampled negatives,
 * −log σ(s⁺) − Σᵢ log σ(−sᵢ⁻) = softplus(−s⁺) + Σᵢ softplus(sᵢ⁻), for positive scores [B] and negative scores [B, k].
 * It is noise-contrastive estimation without the noise correction, so it does not recover normalised probabilities.
 */
export const negativeSampling = defineLoss(
  {
    module: 'applied/retrieval/losses',
    key: 'negativeSampling',
    name: 'Negative sampling',
    family: 'retrieval',
    inputs: 'logits',
    notes: ['sampled-softmax-and-negative-sampling'],
  },
  (positive: Value, negatives: Value, { reduction }: ReductionOptions = {}): Value =>
    reduce(add(softplus(neg(positive)), sum(softplus(negatives), -1)), reduction),
)

/** Options of `noiseContrastiveEstimation`. */
export type NceOptions = ReductionOptions & {
  /** log q(y) of each positive under the noise distribution, shape [B]. */
  logNoisePositive: Target
  /** log q of each noise sample, shape [B, k] (or [k]). */
  logNoiseNegatives: Target
}

/**
 * Noise-contrastive estimation (Gutmann & Hyvärinen, 2010; Mnih & Teh, 2012): classify data against k noise samples
 * with the logit Δ = s − log(k·q), so that at the optimum s is the log of the normalised probability:
 * softplus(−Δ⁺) + Σᵢ softplus(Δᵢ⁻), for positive scores [B] and noise scores [B, k].
 */
export const noiseContrastiveEstimation = defineLoss(
  {
    module: 'applied/retrieval/losses',
    key: 'noiseContrastiveEstimation',
    name: 'Noise-contrastive estimation',
    family: 'retrieval',
    inputs: 'logits',
    notes: ['sampled-softmax-and-negative-sampling'],
    target: 'the normalised log-probability',
  },
  (positive: Value, negatives: Value, options: NceOptions): Value => {
    const k = expectRank(negatives, [1, 2], 'noiseContrastiveEstimation negatives').at(-1)!
    const logK = Math.log(k)
    const dPos = sub(positive, add(constant(options.logNoisePositive), logK))
    const dNeg = sub(negatives, add(constant(options.logNoiseNegatives), logK))
    return reduce(add(softplus(neg(dPos)), sum(softplus(dNeg), -1)), options.reduction)
  },
)

// ── In-batch softmax ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * In-batch softmax for two-tower retrieval (Yi et al., 2019): query embeddings [B, d] against the batch's item
 * embeddings [B, d] by inner product, with the logQ correction when `logQ` gives each item's sampling log-probability.
 * It is `aifn-compute/learning/losses`' `infoNce` with retrieval's defaults (dot product, τ = 1).
 */
export const inBatchSoftmax = defineLoss(
  {
    module: 'applied/retrieval/losses',
    key: 'inBatchSoftmax',
    name: 'In-batch softmax',
    family: 'retrieval',
    inputs: 'embeddings',
    notes: ['contrastive-losses-for-retrieval'],
  },
  (queries: Value, items: Value, options: InfoNceOptions = {}): Value =>
    infoNce(queries, items, { temperature: 1, similarity: 'dot', ...options }),
)

// ── Metric-learning losses ───────────────────────────────────────────────────────────────────────────────────────────

/** Options of `triplet` and `contrastive`. */
export type MarginOptions = ReductionOptions & {
  /** The margin. Default 1. */
  margin?: number
}

/**
 * The triplet loss (Weinberger & Saul, 2009; Schroff et al., 2015): max(0, ‖a − p‖ − ‖a − n‖ + Δ) for anchor, positive
 * and negative embeddings of shape [d] or [B, d], with Euclidean distances. Matches `torch.nn.TripletMarginLoss` up to
 * the 1e-6 PyTorch adds inside its distances.
 */
export const triplet = defineLoss(
  {
    module: 'applied/retrieval/losses',
    key: 'triplet',
    name: 'Triplet loss',
    family: 'representation',
    inputs: 'embeddings',
    notes: ['triplet-and-margin-losses'],
  },
  (anchor: Value, positive: Value, negative: Value, { reduction, margin = 1 }: MarginOptions = {}): Value => {
    const dPos = norm(sub(anchor, positive), -1)
    const dNeg = norm(sub(anchor, negative), -1)
    return reduce(maximum(add(sub(dPos, dNeg), margin), 0), reduction)
  },
)

/**
 * The contrastive (siamese) loss (Hadsell, Chopra & LeCun, 2006): ½ d² for a similar pair and ½ max(0, Δ − d)² for a
 * dissimilar one, with d = ‖x₁ − x₂‖ for embeddings of shape [d] or [B, d] and `similar` 1 or 0 per pair.
 */
export const contrastive = defineLoss(
  {
    module: 'applied/retrieval/losses',
    key: 'contrastive',
    name: 'Contrastive (siamese) loss',
    family: 'representation',
    inputs: 'embeddings',
    notes: ['triplet-and-margin-losses'],
  },
  (x1: Value, x2: Value, similar: Target, { reduction, margin = 1 }: MarginOptions = {}): Value => {
    const y = constant(similar)
    const d2 = sum(square(sub(x1, x2)), -1)
    const pull = mul(y, d2)
    const push = mul(sub(1, y), square(maximum(sub(margin, sqrt(d2)), 0)))
    return reduce(mul(0.5, add(pull, push)), reduction)
  },
)
