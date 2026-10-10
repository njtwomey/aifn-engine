/**
 * Losses for large output spaces and learned representations: sampled softmax with the logQ correction, negative
 * sampling and noise-contrastive estimation, in-batch softmax (InfoNCE itself is `aifn-compute/learning/losses`'), and
 * the triplet and contrastive (siamese) losses on embeddings. Scores and embeddings may be traced and are
 * differentiated; sampling probabilities, labels and masks are constants. Positive scores have shape `[B]` and
 * negative scores `[B, m]`, one row of $m$ negatives per positive.
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
   * $\log q_j$, the log-probability with which each negative was drawn (shape like the negatives, or `[m]` shared by
   * every row). Given, each negative logit is shifted by $-\log(m q_j)$, the logQ correction, so the loss estimates the
   * full softmax cross-entropy; omitted, the logits are used as they are (the uncorrected loss, which learns
   * $\log p - \log q$).
   */
  logQ?: Target
  /**
   * $\log q_y$ of each positive (shape `[B]`); given, the positive's logit is shifted by $-\log q_y$ too (TensorFlow's
   * form).
   */
  positiveLogQ?: Target
  /**
   * 1 to keep a negative, 0 to drop it (e.g. an accidental hit: a sampled negative equal to the positive), shaped
   * like the negatives.
   */
  mask?: Target
}

/**
 * Sampled softmax (Jean et al., 2015; Bengio & Senécal, 2008): the cross-entropy of the positive against itself and $m$
 * sampled negatives, $-s_y + \log(e^{s_y} + \sum_i e^{s_{j_i} - \log(m q_{j_i})})$. With the logQ correction the sum
 * estimates the partition function $Z$ without bias; keeping the positive in the denominator keeps the loss
 * non-negative (Blanc & Rendle, 2018). TensorFlow's `tf.nn.sampled_softmax_loss` is the same loss.
 *
 * @param positive The positives' logits $s_y$, shape `[B]`.
 * @param negatives The sampled negatives' logits $s_{j_i}$, shape `[B, m]`.
 * @param options The sampling log-probabilities for the logQ correction, the mask of dropped negatives, and the
 *   reduction over positives.
 * @returns The loss, reduced over positives (mean by default).
 *
 * @example Two negatives drawn uniformly from 100 items: the correction weights each by 50
 * const positive = tensor([2])
 * const negatives = tensor([[0, 1]])
 * print('uncorrected:', sampledSoftmax(positive, negatives))
 * print('logQ corrected:', sampledSoftmax(positive, negatives, { logQ: [Math.log(0.01), Math.log(0.01)] }))
 * print('log(e^2 + 50 (e^0 + e^1)) - 2 =', Math.log(Math.exp(2) + 50 * (1 + Math.E)) - 2)
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
 * Negative sampling (Mikolov et al., 2013): logistic regression of the positive against $k$ sampled negatives,
 * $-\log \operatorname{sigmoid}(s^+) - \sum_i \log \operatorname{sigmoid}(-s_i^-)
 * = \operatorname{softplus}(-s^+) + \sum_i \operatorname{softplus}(s_i^-)$. It is noise-contrastive estimation without
 * the noise correction, so it does not recover normalised probabilities.
 *
 * @param positive The positives' scores $s^+$, shape `[B]`.
 * @param negatives The negatives' scores $s_i^-$, shape `[B, k]`.
 * @param options The reduction over positives.
 * @returns The loss, reduced over positives (mean by default).
 *
 * @example Small when the positive scores high and the negatives low
 * print('separated:', negativeSampling(tensor([3]), tensor([[-3, -3]])))
 * print('confused:', negativeSampling(tensor([0]), tensor([[0, 0]])), ' 3 log 2 =', 3 * Math.log(2))
 * print('inverted:', negativeSampling(tensor([-3]), tensor([[3, 3]])))
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
  /** $\log q(y)$ of each positive under the noise distribution, shape `[B]`. */
  logNoisePositive: Target
  /** $\log q$ of each noise sample, shape `[B, k]` (or `[k]`, shared by every row). */
  logNoiseNegatives: Target
}

/**
 * Noise-contrastive estimation (Gutmann & Hyvärinen, 2010; Mnih & Teh, 2012): classify data against $k$ noise samples
 * with the logit $\Delta = s - \log(k q)$, so that at the optimum $s$ is the log of the normalised probability:
 * $\operatorname{softplus}(-\Delta^+) + \sum_i \operatorname{softplus}(\Delta_i^-)$.
 *
 * @param positive The data's scores $s^+$, shape `[B]`.
 * @param negatives The noise samples' scores $s_i^-$, shape `[B, k]`.
 * @param options The noise log-probabilities of the positives and the noise samples (required), and the reduction.
 * @returns The loss, reduced over positives (mean by default).
 *
 * @example A model equal to the noise, with two noise samples, has every logit at minus log 2
 * const lq = Math.log(0.1)
 * const options = { logNoisePositive: tensor([lq]), logNoiseNegatives: tensor([[lq, lq]]) }
 * print('loss:', noiseContrastiveEstimation(tensor([lq]), tensor([[lq, lq]]), options))
 * print('log 3 + 2 log 1.5 =', Math.log(3) + 2 * Math.log(1.5))
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
 * In-batch softmax for two-tower retrieval (Yi et al., 2019): each query must pick its own item out of the batch's
 * items by inner product, the other $B - 1$ items being its negatives, with the logQ correction when `logQ` gives each
 * item's sampling log-probability. It is `aifn-compute/learning/losses`' `infoNce` with retrieval's defaults (dot
 * product, $\tau = 1$), which `options` may override.
 *
 * @param queries The query embeddings, shape `[B, d]`.
 * @param items The item embeddings, shape `[B, d]`, row $i$ the positive of query $i$.
 * @param options `infoNce`'s options: temperature, similarity, `logQ`, `symmetric` and the reduction.
 * @returns The loss, reduced over queries (mean by default).
 *
 * @example Queries against their own items, and against the items shuffled
 * const queries = normal(stream(0), 0, 1, { shape: [4, 3] })
 * const items = add(queries, normal(stream(1), 0, 0.1, { shape: [4, 3] }))
 * print('aligned:', inBatchSoftmax(queries, items))
 * print('shuffled:', inBatchSoftmax(queries, take(items, [1, 2, 3, 0])))
 * print('chance, log B =', Math.log(4))
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
  /** The margin $\Delta$. Default 1. */
  margin?: number
}

/**
 * The triplet loss (Weinberger & Saul, 2009; Schroff et al., 2015):
 * $\max(0, \lVert \avec - \pvec \rVert - \lVert \avec - \nvec \rVert + \Delta)$ with Euclidean distances. Matches
 * `torch.nn.TripletMarginLoss` up to the 1e-6 PyTorch adds inside its distances.
 *
 * @param anchor The anchors $\avec$, shape `[d]` or `[B, d]`.
 * @param positive The positives $\pvec$, each to be nearer its anchor, of the same shape.
 * @param negative The negatives $\nvec$, each to be at least $\Delta$ further from its anchor, of the same shape.
 * @param options The margin $\Delta$ and the reduction over triplets.
 * @returns The loss, reduced over triplets (mean by default).
 *
 * @example Positives near their anchors, then the same positives shuffled
 * const anchors = normal(stream(0), 0, 1, { shape: [4, 2] })
 * const near = add(anchors, normal(stream(1), 0, 0.1, { shape: [4, 2] }))
 * const shuffled = take(near, [1, 2, 3, 0])
 * print('aligned:', triplet(anchors, near, shuffled))
 * print('shuffled:', triplet(anchors, shuffled, near))
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
 * The contrastive (siamese) loss (Hadsell, Chopra & LeCun, 2006): $\tfrac{1}{2} \delta^2$ for a similar pair and
 * $\tfrac{1}{2} \max(0, \Delta - \delta)^2$ for a dissimilar one, with $\delta = \lVert \xvec_1 - \xvec_2 \rVert$.
 *
 * @param x1 The first embedding of each pair, shape `[d]` or `[B, d]`.
 * @param x2 The second, of the same shape.
 * @param similar 1 for a similar pair and 0 for a dissimilar one, one per pair.
 * @param options The margin $\Delta$ and the reduction over pairs.
 * @returns The loss, reduced over pairs (mean by default).
 *
 * @example Similar pairs aligned and shuffled
 * const x1 = normal(stream(0), 0, 1, { shape: [4, 2] })
 * const x2 = add(x1, normal(stream(1), 0, 0.1, { shape: [4, 2] }))
 * print('aligned:', contrastive(x1, x2, [1, 1, 1, 1]))
 * print('shuffled:', contrastive(x1, take(x2, [1, 2, 3, 0]), [1, 1, 1, 1]))
 * print('shuffled, labelled dissimilar:', contrastive(x1, take(x2, [1, 2, 3, 0]), [0, 0, 0, 0]))
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
