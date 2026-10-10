/**
 * Divergences as losses: $\KL(p \,\Vert\, q)$ between distribution objects (closed forms from
 * `aifn-compute/probability/distributions`, so they are differentiable in both distributions' parameters), the
 * Jensen–Shannon divergence between probability vectors, and knowledge distillation's temperature-scaled KL between
 * teacher and student softmaxes. All are in nats.
 */

import { Categorical, kl, type Distribution } from 'aifn-compute/probability/distributions'
import { add, div, mul, type Value } from 'aifn-compute/foundation/tensor'
import { defineLoss, reduce, type ReductionOptions } from './core'

/**
 * $\KL(p \,\Vert\, q) = \expect_p[\log p(x) - \log q(x)]$ in nats between two distributions (batches reduce like
 * examples). Uses the closed form registered in `aifn-compute/probability/distributions` (Normal, MultivariateNormal,
 * Categorical, Bernoulli, Beta, Gamma, …); a pair without one throws `AifnError`. Minimising it over $q$'s parameters
 * is moment matching (forward KL); over $p$'s, mode seeking (reverse KL).
 *
 * @param p The first distribution, the one the expectation is under.
 * @param q The second distribution, of the same family as `p`.
 * @param options The reduction.
 * @returns The divergence, reduced over the broadcast batch (mean by default).
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
 * The Jensen–Shannon divergence $\tfrac{1}{2} \KL(p \,\Vert\, m) + \tfrac{1}{2} \KL(q \,\Vert\, m)$, with
 * $m = (p + q)/2$, between probability vectors $p$ and $q$ (shape `[K]` or `[n, K]`, rows summing to one), in nats;
 * symmetric and bounded by $\log 2$. Computed with `Categorical` KL, so it is differentiable in both. (SciPy's
 * `jensenshannon` returns its square root, the Jensen–Shannon distance.)
 *
 * @param p The first probability vector, or rows of them.
 * @param q The second, of the same shape (or broadcast against `p`).
 * @param options The reduction (over rows).
 * @returns The divergence, reduced over rows (mean by default).
 *
 * @example Zero for equal distributions, log 2 for disjoint ones
 * print('equal:', jensenShannonLoss(tensor([0.5, 0.5]), tensor([0.5, 0.5])))
 * print('disjoint:', jensenShannonLoss(tensor([1, 0]), tensor([0, 1])), ' log 2 =', Math.log(2))
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
  /** The softmax temperature $T > 0$ applied to both sets of logits. Default 2. */
  temperature?: number
}

/**
 * Knowledge distillation (Hinton, Vinyals & Dean, 2015):
 * $T^2 \KL(\operatorname{softmax}(\tvec/T) \,\Vert\, \operatorname{softmax}(\svec/T))$ between teacher logits $\tvec$
 * and student logits $\svec$ (shape `[K]` or `[n, K]`). The factor $T^2$ keeps the gradient's size comparable across
 * temperatures. The student comes first.
 *
 * @param student The student's logits $\svec$, shape `[K]` or `[n, K]`; the values trained.
 * @param teacher The teacher's logits $\tvec$, of the same shape.
 * @param options The temperature $T$ and the reduction (over rows).
 * @returns The loss, reduced over rows (mean by default).
 *
 * @example A uniform student against a confident teacher, at T = 2
 * const t = [Math.exp(1) / (Math.exp(1) + 1), 1 / (Math.exp(1) + 1)]
 * print('loss =', distillation(tensor([0, 0]), tensor([2, 0])))
 * print('4 KL =', 4 * (t[0] * Math.log(2 * t[0]) + t[1] * Math.log(2 * t[1])))
 * print('student = teacher:', distillation(tensor([2, 0]), tensor([2, 0])))
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
