/**
 * The arithmetic of group-relative policy optimisation and its relatives: advantages from a group of rewards for one
 * prompt (GRPO, RLOO, Dr. GRPO) or from a whole batch (REINFORCE++), how a response's advantage is spread over its
 * tokens, Schulman's three estimators of the KL divergence from one sample, and pass@$k$.
 *
 * GRPO (Shao et al., 2024) samples a group of $G$ responses to each prompt and scores each against the group:
 * $A_i = (r_i - \bar r) / \operatorname{std}(r)$, so no value network is needed. The variants differ in the baseline
 * and the scale: RLOO (Ahmadian et al., 2024) subtracts the mean of the other $G - 1$ rewards, Dr. GRPO (Liu et al.,
 * 2025) drops the division by the standard deviation and the per-response length normalisation, and REINFORCE++ (Hu
 * et al., 2025) normalises by the standard deviation of the whole batch. A group whose rewards are all equal has zero
 * advantage everywhere and so no gradient; DAPO (Yu et al., 2025) drops such groups.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The baseline subtracted from each reward: the group mean, the mean of the other rewards (leave-one-out), or none.
 */
export type Baseline = 'group-mean' | 'leave-one-out' | 'none'

/** The scale each advantage is divided by: the group's standard deviation, the batch's, or none. */
export type Scale = 'group-std' | 'batch-std' | 'none'

/**
 * How a response's advantage is spread over its tokens: averaged over each response's own tokens (GRPO), over all
 * tokens of the batch (DAPO's token-level mean), or divided by a constant (Dr. GRPO).
 */
export type Aggregation = 'sequence-mean' | 'token-mean' | 'constant'

/** Options of `groupAdvantages` and `batchAdvantages`. */
export type AdvantageOptions = {
  /** The baseline (default `'group-mean'`). */
  baseline?: Baseline
  /** The scale (default `'group-std'` for one group, `'batch-std'` for a batch). */
  scale?: Scale
  /**
   * The standard deviation's divisor $G - \mathrm{ddof}$: 0 for the population form, 1 for the sample form
   * (default 0).
   */
  ddof?: 0 | 1
  /** Added to the standard deviation before dividing, so a group of equal rewards gives zeros (default $10^{-4}$). */
  eps?: number
}

/**
 * The mean and standard deviation of some numbers.
 *
 * @param x The numbers, at least one.
 * @param ddof The divisor is the count less this.
 * @returns Their mean and standard deviation (0 for a single number with `ddof` 1).
 */
function moments(x: ArrayLike<number>, ddof: 0 | 1): { mean: number; std: number } {
  let mean = 0
  for (let i = 0; i < x.length; i++) mean += x[i] / x.length
  let ss = 0
  for (let i = 0; i < x.length; i++) ss += (x[i] - mean) ** 2
  const d = x.length - ddof
  return { mean, std: d > 0 ? Math.sqrt(ss / d) : 0 }
}

/**
 * Subtract a baseline from a group's rewards.
 *
 * @param rewards The group's rewards.
 * @param baseline Which baseline.
 * @param where The caller's name, for error messages.
 * @returns The rewards less the baseline.
 */
function centred(rewards: ArrayLike<number>, baseline: Baseline, where: string): Float64Array {
  const G = rewards.length
  if (G === 0) throw new DomainError(where, `${where}: the group is empty`)
  if (baseline === 'leave-one-out' && G < 2)
    throw new DomainError(where, `${where}: the leave-one-out baseline needs at least two rewards`)
  let total = 0
  for (let i = 0; i < G; i++) total += rewards[i]
  return Float64Array.from({ length: G }, (_, i) =>
    baseline === 'none'
      ? rewards[i]
      : baseline === 'group-mean'
        ? rewards[i] - total / G
        : rewards[i] - (total - rewards[i]) / (G - 1),
  )
}

/**
 * The advantages of one group of responses to a prompt: each reward less a baseline, divided by a scale. The default
 * is GRPO's $A_i = (r_i - \bar r) / (\operatorname{std}(r) + \epsilon)$ (Shao et al., 2024, §4.1.2); RLOO is
 * `{ baseline: 'leave-one-out', scale: 'none' }`, whose advantage is $\frac{G}{G - 1}(r_i - \bar r)$, and Dr. GRPO is
 * `{ scale: 'none' }`. A group of equal rewards gets zero advantage with every option.
 *
 * @param rewards The rewards $r_1, \dots, r_G$ of the group's responses.
 * @param options The baseline, the scale (`'batch-std'` is not available for one group: use `batchAdvantages`), the
 *   standard deviation's `ddof` and `eps`.
 * @returns The advantages, one per response.
 *
 * @example GRPO, RLOO and Dr. GRPO on one group
 * const r = [1, 0, 0, 1, 1]
 * print('GRPO:', groupAdvantages(r))
 * print('RLOO:', groupAdvantages(r, { baseline: 'leave-one-out', scale: 'none' }))
 * print('Dr. GRPO:', groupAdvantages(r, { scale: 'none' }))
 *
 * @example A group that is all correct teaches nothing
 * print(groupAdvantages([1, 1, 1, 1]))
 */
export function groupAdvantages(rewards: ArrayLike<number>, options: AdvantageOptions = {}): Float64Array {
  const { baseline = 'group-mean', scale = 'group-std', ddof = 0, eps = 1e-4 } = options
  if (scale === 'batch-std')
    throw new DomainError('groupAdvantages', 'groupAdvantages: the batch scale needs a batch; use batchAdvantages')
  const a = centred(rewards, baseline, 'groupAdvantages')
  if (scale === 'none') return a
  const { std } = moments(rewards, ddof)
  return a.map((v) => v / (std + eps))
}

/**
 * The advantages of a batch of groups, each centred within its group and then scaled: by its own group's standard
 * deviation (`'group-std'`, GRPO), by the standard deviation of every centred reward in the batch (`'batch-std'`, the
 * default here, as REINFORCE++ normalises; Hu et al., 2025), or not at all.
 *
 * @param rewards The rewards, one array per group (groups may differ in size).
 * @param options The baseline, the scale, `ddof` and `eps`.
 * @returns The advantages, one array per group.
 *
 * @example The batch scale keeps an easy group's advantages small
 * const batch = [[1, 0, 0, 0], [5, 0, 5, 0]]
 * print('group scale:', batchAdvantages(batch, { scale: 'group-std' }))
 * print('batch scale:', batchAdvantages(batch))
 */
export function batchAdvantages(rewards: readonly ArrayLike<number>[], options: AdvantageOptions = {}): Float64Array[] {
  const { baseline = 'group-mean', scale = 'batch-std', ddof = 0, eps = 1e-4 } = options
  if (scale !== 'batch-std') return rewards.map((r) => groupAdvantages(r, { baseline, scale, ddof, eps }))
  const groups = rewards.map((r) => centred(r, baseline, 'batchAdvantages'))
  const all = Float64Array.from(groups.flatMap((g) => Array.from(g)))
  const { std } = moments(all, ddof)
  return groups.map((g) => g.map((v) => v / (std + eps)))
}

/**
 * The weight of each token of each response in the policy-gradient loss, given the responses' advantages and lengths:
 * a token of response $i$ carries $A_i / |o_i|$ divided by the number of responses (`'sequence-mean'`, GRPO's mean
 * over each response, then over responses), $A_i$ divided by the total number of tokens (`'token-mean'`, DAPO), or
 * $A_i$ divided by a constant (`'constant'`, Dr. GRPO, which removes the bias towards long wrong answers that the
 * per-response mean introduces).
 *
 * @param advantages The advantage $A_i$ of each response.
 * @param lengths The number of tokens $|o_i|$ of each response, positive.
 * @param aggregation How to spread the advantage.
 * @param constant The divisor for `'constant'` (default the longest length, as Dr. GRPO uses the generation budget).
 * @returns The weight per token of each response; the loss's gradient is $\sum_i w_i \sum_t \nabla \log \pi(o_{i,t})$.
 *
 * @example A long wrong answer is penalised less per token under the sequence mean
 * const adv = [1, -1]
 * const len = [10, 40]
 * for (const agg of ['sequence-mean', 'token-mean', 'constant']) print(agg, tokenWeights(adv, len, agg))
 */
export function tokenWeights(
  advantages: ArrayLike<number>,
  lengths: ArrayLike<number>,
  aggregation: Aggregation,
  constant?: number,
): Float64Array {
  const G = advantages.length
  if (lengths.length !== G)
    throw new DomainError('tokenWeights', `tokenWeights: ${G} advantages but ${lengths.length} lengths`)
  for (let i = 0; i < G; i++)
    if (!(lengths[i] > 0)) throw new DomainError('tokenWeights', 'tokenWeights: every length must be positive')
  let tokens = 0
  let longest = 0
  for (let i = 0; i < G; i++) {
    tokens += lengths[i]
    longest = Math.max(longest, lengths[i])
  }
  const divisor = constant ?? longest
  if (aggregation === 'constant' && !(divisor > 0))
    throw new DomainError('tokenWeights', 'tokenWeights: the constant must be positive')
  return Float64Array.from({ length: G }, (_, i) =>
    aggregation === 'sequence-mean'
      ? advantages[i] / lengths[i] / G
      : aggregation === 'token-mean'
        ? advantages[i] / tokens
        : advantages[i] / divisor,
  )
}

/**
 * Schulman's three estimators of $\mathrm{KL}(\pi \,\|\, \pi_{\mathrm{ref}})$ from one sample $y \sim \pi$, with
 * $\rho = \pi_{\mathrm{ref}}(y) / \pi(y)$ (Schulman, 2020, "Approximating KL divergence"): $k_1 = -\log \rho$
 * (unbiased, high variance, can be negative), $k_2 = \frac{1}{2}(\log \rho)^2$ (biased, low variance) and
 * $k_3 = \rho - 1 - \log \rho$ (unbiased and never negative; the one GRPO adds to its loss).
 *
 * @param logpPolicy $\log \pi(y)$ of the sampled response.
 * @param logpRef $\log \pi_{\mathrm{ref}}(y)$ of the same response.
 * @returns The three estimates.
 *
 * @example Averaged over samples, k1 and k3 recover the KL
 * const p = [0.7, 0.2, 0.1]
 * const q = [0.4, 0.4, 0.2]
 * const exact = p.reduce((a, pi, i) => a + pi * Math.log(pi / q[i]), 0)
 * const mean = (k) => p.reduce((a, pi, i) => a + pi * klEstimators(Math.log(pi), Math.log(q[i]))[k], 0)
 * print('exact KL:', exact, ' E[k1]:', mean('k1'), ' E[k2]:', mean('k2'), ' E[k3]:', mean('k3'))
 */
export function klEstimators(logpPolicy: number, logpRef: number): { k1: number; k2: number; k3: number } {
  const logRatio = logpRef - logpPolicy
  return { k1: -logRatio, k2: 0.5 * logRatio * logRatio, k3: Math.expm1(logRatio) - logRatio }
}

/**
 * The unbiased estimator of pass@$k$ from $n$ samples of which $c$ are correct (Chen et al., 2021, "Evaluating large
 * language models trained on code", eq. 1): $1 - \binom{n - c}{k} / \binom{n}{k}$, the probability that at least one
 * of $k$ samples drawn without replacement from the $n$ is correct, computed as a product so it does not overflow.
 *
 * @param n The number of samples, at least $k$.
 * @param c How many of them are correct, from 0 to $n$.
 * @param k The budget $k \ge 1$.
 * @returns The estimate, between 0 and 1.
 *
 * @example Three correct out of twenty samples
 * for (const k of [1, 5, 10, 20]) print(`pass@${k} =`, passAtK(20, 3, k))
 */
export function passAtK(n: Size, c: Size, k: Size): number {
  if (!(Number.isInteger(n) && Number.isInteger(c) && Number.isInteger(k) && k >= 1 && k <= n && c >= 0 && c <= n))
    throw new DomainError('passAtK', `passAtK: need integers 1 ≤ k ≤ n and 0 ≤ c ≤ n, got n ${n}, c ${c}, k ${k}`)
  if (n - c < k) return 1
  // C(n − c, k) / C(n, k) = ∏_{i = n − c + 1}^{n} (1 − k / i).
  let ratio = 1
  for (let i = n - c + 1; i <= n; i++) ratio *= 1 - k / i
  return 1 - ratio
}

/**
 * pass@$k$ when each sample is correct with probability $p$, independently: $1 - (1 - p)^k$ (computed with `log1p` and
 * `expm1`, so it stays accurate for small $p$).
 *
 * @param p The probability that one sample is correct, from 0 to 1.
 * @param k The budget $k \ge 0$.
 * @returns The probability that at least one of $k$ samples is correct.
 *
 * @example A rare skill shows up at a large budget
 * for (const k of [1, 16, 256]) print(`pass@${k} =`, passAtKExact(0.01, k))
 */
export function passAtKExact(p: number, k: number): number {
  if (!(p >= 0 && p <= 1)) throw new DomainError('passAtKExact', `passAtKExact: p must be in [0, 1], got ${p}`)
  if (!(k >= 0)) throw new DomainError('passAtKExact', `passAtKExact: k must be non-negative, got ${k}`)
  if (p === 1) return k > 0 ? 1 : 0
  return -Math.expm1(k * Math.log1p(-p))
}
