/**
 * Preference losses for aligning a policy $\pi_\theta$ with pairwise (or single) human judgements, written on the
 * log-probabilities of whole responses: DPO, IPO and KTO against a frozen reference policy $\pi_{\mathrm{ref}}$, and
 * SimPO and ORPO without one.
 *
 * Each takes log-probabilities as numbers or as tensors of one value per example, and returns the mean over examples
 * (or the sum, or every value, by `reduction`). They are compositions of differentiable primitives, so `grad` gives
 * the update of the policy's parameters through the log-probabilities. The reference's log-probabilities are inputs
 * like any other: hold them constant (compute them once, outside the differentiated function) as the methods do.
 *
 * With $y_w$ the chosen and $y_l$ the rejected response to a prompt $x$, the implicit reward of DPO is
 * $r(x, y) = \beta \log(\pi_\theta(y \mid x) / \pi_{\mathrm{ref}}(y \mid x))$, and the losses compare
 * $h = [\log \pi_\theta(y_w) - \log \pi_{\mathrm{ref}}(y_w)] - [\log \pi_\theta(y_l) - \log \pi_{\mathrm{ref}}(y_l)]$.
 * Every $-\log \sigma(z)$ is computed as $\operatorname{softplus}(-z)$, exact for margins of any size.
 */

import { add, expm1, log, mul, neg, sub, type Value } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { sigmoid, softplus } from 'aifn-compute/numerics/special'
import { constant, defineLoss, reduce, type ReductionOptions, type Target } from './core'

const PREFERENCE = ['direct-preference-optimisation']

/**
 * The log-ratio difference $h = (\log \pi_w - \log \pi^{\mathrm{ref}}_w) - (\log \pi_l - \log \pi^{\mathrm{ref}}_l)$.
 *
 * @param logpW The policy's log-probability of the chosen response.
 * @param logpL The policy's log-probability of the rejected response.
 * @param refW The reference's log-probability of the chosen response.
 * @param refL The reference's log-probability of the rejected response.
 * @returns $h$, elementwise.
 */
function logRatioMargin(logpW: Value, logpL: Value, refW: Value, refL: Value): Value {
  return sub(sub(logpW, refW), sub(logpL, refL))
}

/** Options of `dpo`. */
export type DpoOptions = ReductionOptions & {
  /** The inverse temperature $\beta > 0$: the strength of the implicit KL penalty to the reference. */
  beta: number
  /**
   * The assumed probability $\varepsilon \in [0, \tfrac{1}{2})$ that a label is flipped (conservative DPO); default 0,
   * the original loss.
   */
  labelSmoothing?: number
}

/**
 * Direct preference optimisation (Rafailov et al., 2023, eq. 7):
 * $\ell = -\log \sigma(\beta h)$, the Bradley–Terry likelihood of the preference under the implicit reward. With
 * label smoothing $\varepsilon$ (conservative DPO, as TRL's `label_smoothing`) it is
 * $-(1 - \varepsilon) \log \sigma(\beta h) - \varepsilon \log \sigma(-\beta h)$, which no longer rewards pushing $h$
 * to infinity.
 *
 * @param logpW The policy's summed log-probability of each chosen response.
 * @param logpL The policy's summed log-probability of each rejected response.
 * @param refW The reference policy's summed log-probability of each chosen response (a constant).
 * @param refL The reference policy's summed log-probability of each rejected response (a constant).
 * @param options $\beta$, the label smoothing and the reduction.
 * @returns The loss, reduced over examples (mean by default).
 *
 * @example The loss falls as the policy prefers the chosen response more than the reference does
 * for (const h of [-1, 0, 1, 3]) print(`h = ${h}: loss =`, dpo(h, 0, 0, 0, { beta: 1 }))
 *
 * @example The gradient pushes the chosen log-probability up and the rejected one down
 * const g = grad((lp) => dpo(get(lp, 0), get(lp, 1), -2, -2, { beta: 0.5 }))(tensor([-2, -2]))
 * print('d loss / d (log p_w, log p_l) =', g)
 */
export const dpo = defineLoss(
  {
    key: 'dpo',
    name: 'Direct preference optimisation',
    family: 'preference',
    inputs: 'log-probabilities',
    notes: PREFERENCE,
    cite: ['rafailov2023'],
  },
  (logpW: Value, logpL: Value, refW: Value, refL: Value, options: DpoOptions): Value => {
    const { beta, labelSmoothing = 0, reduction } = options
    if (!(beta > 0)) throw new DomainError('dpo', `dpo: beta must be positive, got ${beta}`)
    if (!(labelSmoothing >= 0 && labelSmoothing < 0.5))
      throw new DomainError('dpo', `dpo: labelSmoothing must be in [0, 0.5), got ${labelSmoothing}`)
    const z = mul(beta, logRatioMargin(logpW, logpL, refW, refL))
    const loss =
      labelSmoothing === 0
        ? softplus(neg(z))
        : add(mul(1 - labelSmoothing, softplus(neg(z))), mul(labelSmoothing, softplus(z)))
    return reduce(loss, reduction)
  },
)

/** Options of `ipo`. */
export type IpoOptions = ReductionOptions & {
  /** The regularisation $\tau > 0$: the loss pulls $h$ to $1/(2\tau)$, so a smaller $\tau$ allows a larger margin. */
  tau: number
}

/**
 * Identity preference optimisation (Azar et al., 2024, eq. 17): $\ell = (h - 1/(2\tau))^2$, a squared loss that pulls
 * the log-ratio margin to a finite target instead of pushing it without bound, so it does not overfit deterministic
 * preferences.
 *
 * @param logpW The policy's summed log-probability of each chosen response.
 * @param logpL The policy's summed log-probability of each rejected response.
 * @param refW The reference policy's summed log-probability of each chosen response (a constant).
 * @param refL The reference policy's summed log-probability of each rejected response (a constant).
 * @param options $\tau$ and the reduction.
 * @returns The loss, reduced over examples (mean by default).
 *
 * @example Zero at the target margin, positive either side
 * for (const h of [0, 2.5, 5, 10]) print(`h = ${h}: loss =`, ipo(h, 0, 0, 0, { tau: 0.1 }))
 */
export const ipo = defineLoss(
  {
    key: 'ipo',
    name: 'Identity preference optimisation',
    family: 'preference',
    inputs: 'log-probabilities',
    notes: PREFERENCE,
    cite: ['azar2024'],
  },
  (logpW: Value, logpL: Value, refW: Value, refL: Value, options: IpoOptions): Value => {
    const { tau, reduction } = options
    if (!(tau > 0)) throw new DomainError('ipo', `ipo: tau must be positive, got ${tau}`)
    const d = sub(logRatioMargin(logpW, logpL, refW, refL), 1 / (2 * tau))
    return reduce(mul(d, d), reduction)
  },
)

/** Options of `kto`. */
export type KtoOptions = ReductionOptions & {
  /** The inverse temperature $\beta > 0$ of the value function. */
  beta: number
  /** The weight $\lambda_D$ of desirable examples (default 1). */
  lambdaD?: number
  /** The weight $\lambda_U$ of undesirable examples (default 1). */
  lambdaU?: number
}

/**
 * Kahneman–Tversky optimisation (Ethayarajh et al., 2024, eq. 8), for single responses labelled desirable or
 * undesirable rather than pairs. With $r = \log \pi_\theta(y) - \log \pi_{\mathrm{ref}}(y)$ and $z_0$ the reference
 * point (an estimate of $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$, held constant), the value is
 * $v = \lambda_D \sigma(\beta(r - z_0))$ for a desirable response and $v = \lambda_U \sigma(\beta(z_0 - r))$ for an
 * undesirable one, and the loss is $\lambda_y - v$.
 *
 * @param logp The policy's summed log-probability of each response.
 * @param refLogp The reference policy's summed log-probability of each response (a constant).
 * @param desirable Whether each response is desirable: one boolean for all, or one per example (1 or `true` for
 *   desirable).
 * @param z0 The reference point $z_0$, a number (in practice a batch estimate of the KL, not differentiated).
 * @param options $\beta$, $\lambda_D$, $\lambda_U$ and the reduction.
 * @returns The loss, reduced over examples (mean by default).
 *
 * @example A desirable response gains, an undesirable one loses, as its log-ratio rises
 * for (const r of [-1, 0, 1]) {
 *   print(`r = ${r}: desirable`, kto(r, 0, true, 0, { beta: 1 }), ' undesirable', kto(r, 0, false, 0, { beta: 1 }))
 * }
 */
export const kto = defineLoss(
  {
    key: 'kto',
    name: 'Kahneman–Tversky optimisation',
    family: 'preference',
    inputs: 'log-probabilities',
    notes: PREFERENCE,
    cite: ['ethayarajh2024'],
  },
  (logp: Value, refLogp: Value, desirable: boolean | Target, z0: number, options: KtoOptions): Value => {
    const { beta, lambdaD = 1, lambdaU = 1, reduction } = options
    if (!(beta > 0)) throw new DomainError('kto', `kto: beta must be positive, got ${beta}`)
    const d = constant(typeof desirable === 'boolean' ? (desirable ? 1 : 0) : desirable)
    // sign = +1 for desirable, −1 for undesirable; weight = λ_D or λ_U.
    const sign = typeof d === 'number' ? 2 * d - 1 : sub(mul(2, d), 1)
    const weight = typeof d === 'number' ? (d ? lambdaD : lambdaU) : add(lambdaU, mul(lambdaD - lambdaU, d))
    const v = mul(weight, sigmoid(mul(mul(beta, sign), sub(sub(logp, refLogp), z0))))
    return reduce(sub(weight, v), reduction)
  },
)

/** Options of `simpo`. */
export type SimpoOptions = ReductionOptions & {
  /** The reward scale $\beta > 0$. */
  beta: number
  /** The target reward margin $\gamma \ge 0$ (default 0). */
  gamma?: number
}

/**
 * Simple preference optimisation (Meng, Xia and Chen, 2024, eq. 6):
 * $\ell = -\log \sigma(\beta(\bar\ell_w - \bar\ell_l) - \gamma)$ on the length-averaged log-probabilities
 * $\bar\ell = \log \pi_\theta(y) / |y|$, with no reference policy; the margin $\gamma$ asks the chosen response to win
 * by at least $\gamma / \beta$ nats per token.
 *
 * @param avgLogpW The policy's length-averaged log-probability of each chosen response.
 * @param avgLogpL The policy's length-averaged log-probability of each rejected response.
 * @param options $\beta$, $\gamma$ and the reduction.
 * @returns The loss, reduced over examples (mean by default).
 *
 * @example The margin raises the loss of a pair the policy only just prefers
 * for (const gamma of [0, 0.5, 1]) print(`gamma = ${gamma}: loss =`, simpo(-1, -1.2, { beta: 2, gamma }))
 */
export const simpo = defineLoss(
  {
    key: 'simpo',
    name: 'Simple preference optimisation',
    family: 'preference',
    inputs: 'log-probabilities',
    notes: PREFERENCE,
    cite: ['meng2024simpo'],
  },
  (avgLogpW: Value, avgLogpL: Value, options: SimpoOptions): Value => {
    const { beta, gamma = 0, reduction } = options
    if (!(beta > 0)) throw new DomainError('simpo', `simpo: beta must be positive, got ${beta}`)
    if (!(gamma >= 0)) throw new DomainError('simpo', `simpo: gamma must be non-negative, got ${gamma}`)
    return reduce(softplus(neg(sub(mul(beta, sub(avgLogpW, avgLogpL)), gamma))), reduction)
  },
)

/** Options of `orpo`. */
export type OrpoOptions = ReductionOptions & {
  /** The weight $\lambda \ge 0$ of the odds-ratio term against the supervised loss. */
  lambda: number
}

/**
 * The log-odds $\log(p / (1 - p))$ of a probability given by its logarithm $a = \log p < 0$:
 * $a - \log(-\operatorname{expm1}(a))$, accurate for $p$ near 0 and near 1.
 *
 * @param a The log-probability, negative.
 * @returns The log-odds, elementwise.
 */
function logOdds(a: Value): Value {
  return sub(a, log(neg(expm1(a))))
}

/**
 * Odds-ratio preference optimisation (Hong, Lee and Thorne, 2024, eqs. 6–7), supervised fine-tuning and preference
 * alignment in one loss with no reference policy:
 * $\ell = \mathrm{NLL}_w - \lambda \log \sigma(\log \mathrm{odds}_w - \log \mathrm{odds}_l)$, where
 * $\mathrm{odds} = p/(1 - p)$ of the length-averaged likelihood $p = \exp(\bar\ell)$.
 *
 * @param avgLogpW The policy's length-averaged log-probability of each chosen response, negative.
 * @param avgLogpL The policy's length-averaged log-probability of each rejected response, negative.
 * @param nllW The supervised loss on each chosen response (its negative log-likelihood, usually length-averaged).
 * @param options $\lambda$ and the reduction.
 * @returns The loss, reduced over examples (mean by default).
 *
 * @example The odds-ratio term on top of the supervised loss
 * const avgW = -0.5
 * const avgL = -1.5
 * print('lambda = 0:', orpo(avgW, avgL, -avgW, { lambda: 0 }))
 * print('lambda = 0.1:', orpo(avgW, avgL, -avgW, { lambda: 0.1 }))
 */
export const orpo = defineLoss(
  {
    key: 'orpo',
    name: 'Odds-ratio preference optimisation',
    family: 'preference',
    inputs: 'log-probabilities',
    notes: PREFERENCE,
    cite: ['hong2024orpo'],
  },
  (avgLogpW: Value, avgLogpL: Value, nllW: Value, options: OrpoOptions): Value => {
    const { lambda, reduction } = options
    if (!(lambda >= 0)) throw new DomainError('orpo', `orpo: lambda must be non-negative, got ${lambda}`)
    const ratio = softplus(neg(sub(logOdds(avgLogpW), logOdds(avgLogpL))))
    return reduce(add(nllW, mul(lambda, ratio)), reduction)
  },
)
