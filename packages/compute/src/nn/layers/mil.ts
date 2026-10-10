/**
 * Multiple-instance pooling: from a bag of instance embeddings $\Zmat$, `[B, t, d]` ($B$ bags of $t$ instances, e.g.
 * the time points of a series), to bag logits `[B, c]`, in the five ways MILLET compares (Early et al. 2024, ICLR,
 * §3.2 and Tables A.1–A.5). $\psi$ is a linear classifier from $d$ to $c$ features; the attention head is
 * $a(\zvec) = \sigma(\wvec^\top \tanh(\Vmat^\top\zvec + \bvec_V) + b_w) \in (0, 1)$ with an internal width of 8
 * (sigmoid rather than softmax, so long bags do not flatten the weights). With $a_j = a(\zvec_j)$ and
 * $\operatorname{mean}_j$ the mean over a bag's instances:
 *
 * - `embedding` (global average pooling, as a plain CNN): $\hat{Y} = \psi(\operatorname{mean}_j \zvec_j)$.
 * - `attention` (Ilse et al. 2018, with sigmoid weights): $\hat{Y} = \psi(\operatorname{mean}_j a_j \zvec_j)$.
 * - `instance` (mi-Net, Wang et al. 2018): $\hat{y}_j = \psi(\zvec_j)$; $\hat{Y} = \operatorname{mean}_j \hat{y}_j$.
 * - `additive` (Javed et al. 2022): $\hat{y}_j = \psi(a_j \zvec_j)$; $\hat{Y} = \operatorname{mean}_j \hat{y}_j$.
 * - `conjunctive` (MILLET): $\hat{y}_j = \psi(\zvec_j)$; $\hat{Y} = \operatorname{mean}_j a_j \hat{y}_j$; attention and
 *   classifier act in parallel.
 *
 * Every kind returns an instance-level interpretation: the class-specific instance predictions ($\hat{y}_j$ for
 * `instance`, $a_j \hat{y}_j$ for `additive` and `conjunctive`), the class-agnostic weights $a_j$ for `attention`, and
 * for `embedding` the class activation map $\psi(\zvec_j)$ (Zhou et al. 2016, there without the bias), whose mean over
 * the bag is the bag logit since $\psi$ is affine. An optional mask `[B, t]` (1 = kept) removes instances from their
 * bag: means run over the kept instances only, as when a perturbation metric drops time points.
 */

import { child } from 'aifn-compute/foundation/random'
import {
  div,
  expandDims,
  mul,
  reshape,
  shapeOfValue,
  sum,
  tanh,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { sigmoid } from 'aifn-compute/numerics/special'
import { Linear, linear, tap, type Context, type Layer, type LinearParams } from './layers'

/** The five MIL pooling methods. */
export type MilPoolingKind = 'embedding' | 'attention' | 'instance' | 'additive' | 'conjunctive'

/** All pooling kinds, in MILLET's order. */
export const MIL_POOLING_KINDS: readonly MilPoolingKind[] = [
  'embedding',
  'attention',
  'instance',
  'additive',
  'conjunctive',
]

/** Parameters of MIL pooling. */
export type MilPoolingParams = {
  /** The classifier $\psi$, a linear map from $d$ to $c$ features. */
  classifier: LinearParams
  /**
   * The attention head of the attention kinds: `hidden`, from $d$ to the internal width $h$ ($\Vmat$, $\bvec_V$), and
   * `score`, from $h$ to 1 ($\wvec$, $b_w$).
   */
  attention?: { hidden: LinearParams; score: LinearParams }
}

/** The outputs of `milPool`. */
export interface MilPooled {
  /** Bag logits, `[B, c]`. */
  readonly logits: Value
  /** Attention weights $a_j$, `[B, t]` (the attention kinds only). */
  readonly attention?: Value
  /**
   * Instance predictions, `[B, t, c]`, before attention weighting: $\psi(\zvec_j)$, or $\psi(a_j \zvec_j)$ for
   * `additive`. Absent for `attention`.
   */
  readonly predictions?: Value
  /** The instance-level interpretation: `[B, t, c]` (class-specific) or `[B, t]` (`attention`). */
  readonly interpretation: Value
}

/**
 * Whether a pooling kind has an attention head.
 *
 * @param kind The pooling kind.
 * @returns True for `attention`, `additive` and `conjunctive`.
 */
const usesAttention = (kind: MilPoolingKind) => kind === 'attention' || kind === 'additive' || kind === 'conjunctive'

/**
 * The mean over the instances of each bag, over the kept instances only when a mask is given.
 *
 * @param x Per-instance values, `[B, t, k]`.
 * @param mask Which instances are kept, `[B, t]` of 1 (kept) and 0; left out, all are.
 * @returns The bag means, `[B, k]`.
 */
function bagMean(x: Value, mask: Tensor | undefined): Value {
  const [, t] = shapeOfValue(x)
  if (!mask) return div(sum(x, 1), t)
  const m = expandDims(mask, 2)
  return div(sum(mul(x, m), 1), sum(m, 1))
}

/**
 * The attention weights $a(\zvec) = \sigma(\wvec^\top \tanh(\Vmat^\top\zvec + \bvec_V) + b_w)$ of every instance.
 *
 * @param p The attention head's parameters.
 * @param z The instance embeddings, `[B, t, d]`.
 * @returns The weights, in $(0, 1)$, `[B, t, 1]`.
 */
function attend(p: NonNullable<MilPoolingParams['attention']>, z: Value): Value {
  return sigmoid(linear(tanh(linear(z, p.hidden.weight, p.hidden.bias)), p.score.weight, p.score.bias))
}

/**
 * Pool bags of instance embeddings $\Zmat$, `[B, t, d]`, into bag logits by one of the five methods of the file notes,
 * with the instance-level interpretation. Differentiable. Throws `ShapeError` when `z` is not rank 3 or the mask is
 * not `[B, t]`, and `DomainError` when an attention kind is given no attention head.
 *
 * @param kind The pooling method.
 * @param params The classifier and, for the attention kinds, the attention head, as `MilPooling` draws them.
 * @param z The instance embeddings, `[B, t, d]`.
 * @param mask Which instances are kept, `[B, t]` of 1 (kept) and 0; left out, all are.
 * @returns The bag logits, and the attention weights, instance predictions and interpretation the method has.
 *
 * @example Conjunctive pooling: the bag logits are the bag mean of the interpretation
 * const p = MilPooling(3, 2, 'conjunctive').init(stream(0))
 * const out = milPool('conjunctive', p, normals(stream(1), [1, 4, 3]))
 * print('logits:', out.logits)
 * print('attention:', out.attention)
 * print('mean of the interpretation:', mean(out.interpretation, 1))
 *
 * @example A mask drops instances from the bag
 * const p = MilPooling(3, 2, 'instance').init(stream(0))
 * const z = tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]])
 * print('first two kept:', milPool('instance', p, z, tensor([[1, 1, 0, 0]])).logits)
 * print('bag of the first two:', milPool('instance', p, tensor([[[1, 0, 0], [0, 1, 0]]])).logits)
 */
export function milPool(kind: MilPoolingKind, params: MilPoolingParams, z: Value, mask?: Tensor): MilPooled {
  const shape = shapeOfValue(z)
  if (shape.length !== 3)
    throw new ShapeError('milPool', `milPool: embeddings must be [B, t, d], got [${shape.join(', ')}]`)
  if (mask && (mask.shape.length !== 2 || mask.shape[0] !== shape[0] || mask.shape[1] !== shape[1]))
    throw new ShapeError(
      'milPool',
      `milPool: the mask must be [${shape[0]}, ${shape[1]}], got [${mask.shape.join(', ')}]`,
    )
  const psi = (x: Value) => linear(x, params.classifier.weight, params.classifier.bias)
  if (usesAttention(kind) && !params.attention)
    throw new DomainError('milPool', `milPool: ${kind} pooling needs the attention head's parameters`)
  const a = usesAttention(kind) ? attend(params.attention!, z) : undefined
  const flat = (v: Value) => reshape(v, [shape[0], shape[1]])
  switch (kind) {
    case 'embedding': {
      const cam = psi(z)
      return { logits: psi(bagMean(z, mask)), predictions: cam, interpretation: cam }
    }
    case 'attention':
      return { logits: psi(bagMean(mul(a!, z), mask)), attention: flat(a!), interpretation: flat(a!) }
    case 'instance': {
      const y = psi(z)
      return { logits: bagMean(y, mask), predictions: y, interpretation: y }
    }
    case 'additive': {
      const y = psi(mul(a!, z))
      return { logits: bagMean(y, mask), attention: flat(a!), predictions: y, interpretation: mul(a!, y) }
    }
    case 'conjunctive': {
      const y = psi(z)
      const weighted = mul(a!, y)
      return { logits: bagMean(weighted, mask), attention: flat(a!), predictions: y, interpretation: weighted }
    }
  }
}

/**
 * MIL pooling as a layer from embeddings of width `features` to `classes` logits: `apply(params, z)` gives the bag
 * logits `[B, c]` (with no mask); `milPool` with the same parameters gives the interpretation too. The classifier is
 * drawn from the stream's child `classifier`, the attention head from `('attention', 'hidden')` and
 * `('attention', 'score')`.
 *
 * @param features The embedding width $d$.
 * @param classes The number of classes $c$.
 * @param kind The pooling method; the attention kinds also get an attention head.
 * @param options `attentionWidth`, the attention head's internal width $h$ (default 8, as MILLET).
 * @returns The layer, with parameters `MilPoolingParams`.
 *
 * @example Attention pooling of two bags of five instances
 * const layer = MilPooling(4, 3, 'attention')
 * const p = layer.init(stream(0))
 * print(layer.label, ' head:', p.attention.hidden.weight.shape, p.attention.score.weight.shape)
 * print('logits:', layer.apply(p, normals(stream(1), [2, 5, 4])))
 */
export function MilPooling(
  features: number,
  classes: number,
  kind: MilPoolingKind,
  options: { attentionWidth?: number } = {},
): Layer<MilPoolingParams> {
  const h = options.attentionWidth ?? 8
  const classifier = Linear(features, classes)
  const hidden = Linear(features, h)
  const score = Linear(h, 1)
  return {
    kind: 'MilPooling',
    label: `MilPooling(${kind}, ${features} → ${classes})`,
    init: (s) => ({
      classifier: classifier.init(child(s, 'classifier')),
      ...(usesAttention(kind)
        ? {
            attention: {
              hidden: hidden.init(child(s, 'attention', 'hidden')),
              score: score.init(child(s, 'attention', 'score')),
            },
          }
        : {}),
    }),
    apply: (p, x, ctx?: Context) => tap(ctx, milPool(kind, p, x).logits),
  }
}
