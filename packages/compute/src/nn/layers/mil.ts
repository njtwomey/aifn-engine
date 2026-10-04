/**
 * Multiple-instance pooling: from a bag of instance embeddings Z [B, t, d] (B bags of t instances, e.g. the time points
 * of a series) to bag logits [B, c], in the five ways MILLET compares (Early et al. 2024, ICLR, §3.2 and Tables
 * A.1–A.5). ψ is a linear classifier d → c; the attention head is a(z) = σ(wᵀ tanh(Vᵀz)) ∈ (0, 1) with an internal width
 * of 8 (sigmoid rather than softmax, so long bags do not flatten the weights).
 *
 * - `embedding` (global average pooling, as a plain CNN): Ŷ = ψ(mean_j z_j).
 * - `attention` (Ilse et al. 2018, with sigmoid weights): Ŷ = ψ(mean_j a_j z_j).
 * - `instance` (mi-Net, Wang et al. 2018): ŷ_j = ψ(z_j); Ŷ = mean_j ŷ_j.
 * - `additive` (Javed et al. 2022): ŷ_j = ψ(a_j z_j); Ŷ = mean_j ŷ_j.
 * - `conjunctive` (MILLET): ŷ_j = ψ(z_j); Ŷ = mean_j a_j ŷ_j; attention and classifier act in parallel.
 *
 * Every kind returns an instance-level interpretation: the class-specific instance predictions (ŷ for `instance`, aŷ
 * for `additive` and `conjunctive`), the class-agnostic weights a for `attention`, and for `embedding` the class
 * activation map ψ(z_j) (Zhou et al. 2016), which for a linear ψ equals the per-instance class score up to the bias.
 * An optional mask [B, t] (1 = kept) removes instances from their bag: means run over the kept instances only, as
 * when a perturbation metric drops time points.
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

/** Parameters of MIL pooling: the classifier ψ (d → c) and, for the attention kinds, the head (d → h → 1). */
export type MilPoolingParams = {
  classifier: LinearParams
  attention?: { hidden: LinearParams; score: LinearParams }
}

/** The outputs of {@link milPool}. */
export interface MilPooled {
  /** Bag logits [B, c]. */
  readonly logits: Value
  /** Attention weights [B, t] (the attention kinds). */
  readonly attention?: Value
  /** Instance predictions [B, t, c] before attention weighting (ψ(z_j), or ψ(a_j z_j) for `additive`). */
  readonly predictions?: Value
  /** The instance-level interpretation: [B, t, c] (class-specific) or [B, t] (`attention`). */
  readonly interpretation: Value
}

const usesAttention = (kind: MilPoolingKind) => kind === 'attention' || kind === 'additive' || kind === 'conjunctive'

/** The mean over axis 1 of x [B, t, k], over the kept instances when a mask [B, t] is given. */
function bagMean(x: Value, mask: Tensor | undefined): Value {
  const [, t] = shapeOfValue(x)
  if (!mask) return div(sum(x, 1), t)
  const m = expandDims(mask, 2)
  return div(sum(mul(x, m), 1), sum(m, 1))
}

/** The attention weights a = σ(wᵀ tanh(Vᵀz)) of every instance, [B, t, 1]. */
function attend(p: NonNullable<MilPoolingParams['attention']>, z: Value): Value {
  return sigmoid(linear(tanh(linear(z, p.hidden.weight, p.hidden.bias)), p.score.weight, p.score.bias))
}

/** Pool a bag of instance embeddings Z [B, t, d] into bag logits by one of the five methods (module notes). */
export function milPool(kind: MilPoolingKind, params: MilPoolingParams, z: Value, mask?: Tensor): MilPooled {
  const shape = shapeOfValue(z)
  if (shape.length !== 3)
    throw new ShapeError('milPool', `milPool: embeddings must be [B, t, d], got [${shape.join(', ')}]`)
  if (mask && (mask.shape[0] !== shape[0] || mask.shape[1] !== shape[1]))
    throw new ShapeError('milPool', 'milPool: the mask must be [B, t]')
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
 * logits [B, c]; {@link milPool} with the same parameters gives the interpretation too. `attentionWidth` is the
 * attention head's internal width (default 8, as MILLET).
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
