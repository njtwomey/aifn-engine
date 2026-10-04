/**
 * Divergences as losses: KL(p ‖ q) between distribution objects (closed forms from `aifn-compute/probability/distributions`, so they are
 * differentiable in both distributions' parameters), the Jensen–Shannon divergence between probability vectors, and
 * knowledge distillation's temperature-scaled KL between teacher and student softmaxes.
 */

import { Categorical, kl, type Distribution } from 'aifn-compute/probability/distributions'
import { add, div, mul, type Value } from 'aifn-compute/foundation/tensor'
import { defineLoss, reduce, type ReductionOptions } from './core'

/**
 * KL(p ‖ q) in nats between two distributions (batches reduce like examples). Uses the closed form registered in
 * `aifn-compute/probability/distributions` (Normal, MultivariateNormal, Categorical, Bernoulli, Beta, Gamma, …); throws for a pair without
 * one. Minimising it over q's parameters is moment matching (forward KL); over p's, mode seeking (reverse KL).
 */
export const klLoss = defineLoss(
  {
    key: 'klLoss',
    name: 'Kullback–Leibler divergence',
    family: 'divergence',
    inputs: 'distributions',
    notes: ['kullback-leibler-divergence'],
  },
  (p: Distribution, q: Distribution, { reduction }: ReductionOptions = {}): Value => reduce(kl(p, q), reduction),
)

/**
 * The Jensen–Shannon divergence ½ KL(p ‖ m) + ½ KL(q ‖ m), m = (p + q)/2, between probability vectors p and q (shape
 * [K] or [n, K], rows summing to one), in nats; symmetric and bounded by log 2. Computed with `Categorical` KL.
 */
export const jensenShannonLoss = defineLoss(
  {
    key: 'jensenShannonLoss',
    name: 'Jensen–Shannon divergence',
    family: 'divergence',
    inputs: 'distributions',
    notes: ['f-divergences-and-jensen-shannon'],
  },
  (p: Value, q: Value, { reduction }: ReductionOptions = {}): Value => {
    const m = Categorical(mul(0.5, add(p, q)))
    return reduce(mul(0.5, add(kl(Categorical(p), m), kl(Categorical(q), m))), reduction)
  },
)

/** Options of `distillation`. */
export type DistillationOptions = ReductionOptions & {
  /** The softmax temperature T > 0 applied to both sets of logits. Default 2. */
  temperature?: number
}

/**
 * Knowledge distillation (Hinton, Vinyals & Dean, 2015): T²·KL(softmax(t/T) ‖ softmax(s/T)) between teacher logits t
 * and student logits s (shape [K] or [n, K]). The factor T² keeps the gradient's size comparable across temperatures.
 */
export const distillation = defineLoss(
  {
    key: 'distillation',
    name: 'Distillation (temperature-scaled KL)',
    family: 'divergence',
    inputs: 'logits',
    notes: ['kullback-leibler-divergence'],
  },
  (student: Value, teacher: Value, { reduction, temperature = 2 }: DistillationOptions = {}): Value => {
    const t = Categorical({ logits: div(teacher, temperature) })
    const s = Categorical({ logits: div(student, temperature) })
    return reduce(mul(temperature ** 2, kl(t, s)), reduction)
  },
)
