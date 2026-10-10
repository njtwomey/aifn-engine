/**
 * Representation losses on pairs of embeddings: InfoNCE with in-batch negatives (one direction, or symmetric as in
 * CLIP), and CLIP's learnable temperature. The embeddings and the temperature may be traced, so the temperature can be
 * a trained parameter; the logQ correction is a constant.
 */

import {
  add,
  div,
  exp,
  eye,
  matmul,
  minimum,
  mul,
  neg,
  norm,
  shapeOfValue,
  sub,
  transpose,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { softmaxCrossEntropy } from './classification'
import { constant, defineLoss, expectRank, reduce, type ReductionOptions, type Target } from './core'

/** Options of `infoNce`. */
export type InfoNceOptions = ReductionOptions & {
  /**
   * The temperature $\tau > 0$ dividing every similarity (default 0.1). A traced value makes it learnable: CLIP trains
   * $\log(1/\tau)$ (see `learnedTemperature`).
   */
  temperature?: number | Value
  /** `cosine` (default) normalises the embeddings first; `dot` uses raw inner products. */
  similarity?: 'cosine' | 'dot'
  /**
   * $\log q_j$ of each in-batch item (shape `[B]`), its sampling probability: subtracted from column $j$'s logits, the
   * logQ correction for popular items appearing more often as negatives (Yi et al., 2019). A constant.
   */
  logQ?: Target
  /** Average the loss over both directions (anchors to positives and back), as CLIP does. Default false. */
  symmetric?: boolean
}

/**
 * Each row divided by its Euclidean norm (a zero row gives NaN).
 *
 * @param x The rows to normalise, shape `[B, d]`.
 * @returns The unit rows, shape `[B, d]`.
 */
function unitRows(x: Value): Value {
  return div(x, norm(x, -1, true))
}

/**
 * InfoNCE (van den Oord, Li & Vinyals, 2018; SimCLR, Chen et al., 2020): each anchor (row $i$ of `[B, d]`) must pick
 * its own positive (row $i$ of `[B, d]`) out of the batch, by softmax cross-entropy over the logits
 * $\operatorname{sim}(\avec_i, \pvec_j) / \tau$; the other $B - 1$ rows are its negatives. $\log B$ minus InfoNCE
 * lower-bounds the mutual information between the two views. With `symmetric`, the loss is the mean of the
 * anchor-to-positive and positive-to-anchor cross-entropies, CLIP's loss (Radford et al., 2021). Inputs of another rank
 * throw `ShapeError`.
 *
 * @param anchors The anchor embeddings $\avec_i$, shape `[B, d]`.
 * @param positives The positive embeddings $\pvec_i$, shape `[B, d]`, row $i$ paired with anchor $i$.
 * @param options The temperature $\tau$, the similarity (`cosine` or `dot`), the logQ correction, whether the loss is
 *   symmetric, and the reduction (over anchors).
 * @returns The loss, reduced over anchors (mean by default).
 *
 * @example Two orthogonal pairs: each anchor's logits are 1/tau for its positive and 0 for the other
 * const x = tensor([[1, 0], [0, 1]])
 * print('tau = 1:', infoNce(x, x, { temperature: 1 }), ' log(1 + 1/e) =', Math.log(1 + Math.exp(-1)))
 * print('tau = 0.1:', infoNce(x, x), ' log(1 + 1/e^10) =', Math.log(1 + Math.exp(-10)))
 */
export const infoNce = defineLoss(
  {
    key: 'infoNce',
    name: 'InfoNCE',
    family: 'representation',
    inputs: 'embeddings',
    notes: ['contrastive-learning', 'clip'],
  },
  (anchors: Value, positives: Value, options: InfoNceOptions = {}): Value => {
    expectRank(anchors, [2], 'infoNce anchors')
    expectRank(positives, [2], 'infoNce positives')
    const { temperature = 0.1, similarity = 'cosine', logQ, symmetric = false, reduction } = options
    const [a, p] = similarity === 'cosine' ? [unitRows(anchors), unitRows(positives)] : [anchors, positives]
    let logits = div(matmul(a, transpose(p)), temperature)
    if (logQ !== undefined) logits = sub(logits, constant(logQ))
    const labels = eye(shapeOfValue(logits)[0])
    const forward = softmaxCrossEntropy(logits, labels, { reduction: 'none' })
    if (!symmetric) return reduce(forward, reduction)
    const backward = softmaxCrossEntropy(transpose(logits), labels, { reduction: 'none' })
    return reduce(mul(0.5, add(forward, backward)), reduction)
  },
)

/**
 * CLIP's learnable temperature (Radford et al., 2021, §2.5): the trained parameter is the log logit scale
 * $s = \log(1/\tau)$, so $\tau = \exp(-s)$ stays positive, and the scale $e^s$ is clipped at `maxScale` (default
 * 100) so that training cannot sharpen the softmax without bound. CLIP initialises $s$ at $\log(1/0.07)$.
 * Differentiable in $s$ (with zero gradient once clipped).
 *
 * @param logScale The log logit scale $s$, usually a traced parameter.
 * @param options The clip on the scale.
 * @param options.maxScale The largest scale $e^s$ allowed, so the smallest temperature is `1 / maxScale`.
 * @returns The temperature $\tau = \exp(-\min(s, \log \text{maxScale}))$, to pass as `infoNce`'s `temperature`.
 *
 * @example CLIP's initial temperature, and the clip
 * print('s = log(1/0.07):', learnedTemperature(Math.log(1 / 0.07)))
 * print('s = 10, clipped:', learnedTemperature(10))
 */
export function learnedTemperature(logScale: Value, { maxScale = 100 }: { maxScale?: number } = {}): Value {
  return exp(neg(minimum(logScale, Math.log(maxScale))))
}
