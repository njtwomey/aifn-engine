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
   * The temperature τ > 0 dividing every similarity (default 0.1). A traced value makes it learnable: CLIP trains
   * log(1/τ) (see `learnedTemperature`).
   */
  temperature?: number | Value
  /** `cosine` (default) normalises the embeddings first; `dot` uses raw inner products. */
  similarity?: 'cosine' | 'dot'
  /**
   * log q_j of each in-batch item (shape [B]): subtracted from column j's logits, the logQ correction for popular items
   * appearing more often as negatives (Yi et al., 2019).
   */
  logQ?: Target
  /** Average the loss over both directions (anchors → positives and back), as CLIP does. Default false. */
  symmetric?: boolean
}

/** Each row divided by its Euclidean norm. */
function unitRows(x: Value): Value {
  return div(x, norm(x, -1, true))
}

/**
 * InfoNCE (van den Oord, Li & Vinyals, 2018; SimCLR, Chen et al., 2020): each anchor (row i of [B, d]) must pick its
 * own positive (row i of [B, d]) out of the batch, by softmax cross-entropy over the logits sim(aᵢ, pⱼ)/τ; the other
 * B − 1 rows are its negatives. log B − InfoNCE lower-bounds the mutual information between the two views. With
 * `symmetric`, the loss is the mean of the anchor → positive and positive → anchor cross-entropies, CLIP's loss
 * (Radford et al., 2021).
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
 * s = log(1/τ), so τ = exp(−s) stays positive, and the scale e^s is clipped at `maxScale` (default 100) so that
 * training cannot sharpen the softmax without bound. CLIP initialises s at log(1/0.07).
 */
export function learnedTemperature(logScale: Value, { maxScale = 100 }: { maxScale?: number } = {}): Value {
  return exp(neg(minimum(logScale, Math.log(maxScale))))
}
