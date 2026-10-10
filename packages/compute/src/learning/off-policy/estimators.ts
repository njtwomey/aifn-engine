/**
 * Off-policy evaluation of a target policy $\pi$ from logged bandit feedback: rounds $i = 1, \dots, n$ with a context
 * $x_i$, an action $a_i$ drawn by a logging policy $\pi_0$, its propensity $\pi_0(a_i \mid x_i)$ and the reward $r_i$.
 * The value $V(\pi) = \expect_x \sum_a \pi(a \mid x) \mu(x, a)$, with $\mu(x, a)$ the mean reward of action $a$ in
 * context $x$, is estimated by inverse propensity scoring (plain, clipped and self-normalised), the direct method (a
 * reward model $\hat q$), doubly robust (both) and switch-DR (doubly robust where the importance weight is at most
 * $\tau$, the model elsewhere). Each returns the estimate, a standard error and the importance weights it used.
 *
 * Policies and reward models are given as $n \times K$ matrices, row $i$ for round $i$ and column $a$ for action $a$:
 * the target's $\pi(a \mid x_i)$ and the model's $\hat q(x_i, a)$. The contexts themselves are never read. The
 * estimate is the mean of per-round terms, and its interval is the normal approximation
 * $\hat V \pm z_{(1 + \ell)/2} \, \mathrm{SE}$ at level $\ell$. A malformed log (lengths that disagree, an action
 * that is not an index, a propensity outside $(0, 1]$, no rounds) throws `DomainError`.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { importanceEffectiveSampleSize } from 'aifn-compute/probability/stats'

/** A dense float64 array, as `dense` reads vectors and matrices. */
type F64 = dense.F64

/** Logged bandit feedback: one entry per round. */
export interface BanditLog {
  /** The logged actions $a_i \in \{0, \dots, K - 1\}$, $n$ values. */
  readonly actions: VectorLike
  /** The observed rewards $r_i$, $n$ values. */
  readonly rewards: VectorLike
  /** The logging propensity $\pi_0(a_i \mid x_i)$ of each logged action, $n$ values in $(0, 1]$. */
  readonly propensities: VectorLike
  /** $\pi_0(a \mid x_i)$ for every action, $n \times K$; read by switch-DR only, to report the share switched. */
  readonly logging?: MatrixLike
}

/** The result of every off-policy estimator. */
export interface OffPolicyEstimate {
  /** The estimator's name. */
  readonly estimator: string
  /** The estimate $\hat V(\pi)$. */
  readonly value: number
  /**
   * The standard error: the sample standard deviation of the per-round terms over $\sqrt n$ (by the delta method for
   * SNIPS); 0 for one round.
   */
  readonly standardError: number
  /** The lower end of the normal-approximation interval $\hat V \pm z \, \mathrm{SE}$ at `level` (default 0.95). */
  readonly lower: number
  /** The upper end of the same interval. */
  readonly upper: number
  /** The per-round terms, $n$ values, whose mean is the estimate. */
  readonly terms: Tensor
  /**
   * The importance weights $w_i = \pi(a_i \mid x_i)/\pi_0(a_i \mid x_i)$ used (after clipping or switching where it
   * applies), $n$ values; all 1 for the direct method.
   */
  readonly weights: Tensor
  /**
   * Kish's effective sample size $(\sum_i w_i)^2 / \sum_i w_i^2$ of those weights ($n$ for the direct method; NaN
   * with a negative weight or when every weight is 0).
   */
  readonly effectiveSampleSize: number
}

/** Options of every estimator: `level`, the coverage of the interval `lower`, `upper` (default 0.95). */
export type EstimateOptions = { level?: number }

/** A checked log: actions `a`, rewards `r`, propensities `p0` and the number of rounds `n`. */
type Read = { a: Int32Array; r: F64; p0: F64; n: number }

/**
 * A log's arrays, checked: equal lengths, actions that are non-negative integers, propensities in $(0, 1]$ and at
 * least one round. Throws `DomainError` otherwise.
 *
 * @param log The logged rounds.
 * @param where The caller's name, for error messages.
 * @returns The actions `a` (int32), rewards `r`, propensities `p0` and the number of rounds `n`.
 */
function readLog(log: BanditLog, where: string): Read {
  const a = dense.toF64(log.actions, where)
  const r = dense.toF64(log.rewards, where)
  const p0 = dense.toF64(log.propensities, where)
  const n = a.length
  if (r.length !== n || p0.length !== n)
    throw new DomainError(where, `${where}: ${n} actions, ${r.length} rewards and ${p0.length} propensities`)
  for (let i = 0; i < n; i++) {
    if (!Number.isInteger(a[i]) || a[i] < 0) throw new DomainError(where, `${where}: action ${a[i]} at round ${i}`)
    if (!(p0[i] > 0 && p0[i] <= 1))
      throw new DomainError(where, `${where}: propensity ${p0[i]} at round ${i} is not in (0, 1]`)
  }
  if (n === 0) throw new DomainError(where, `${where}: the log is empty`)
  return { a: Int32Array.from(a), r, p0, n }
}

/**
 * A policy's (or a model's) values as a dense row-major $n \times K$ array. Throws `DomainError` when it does not have
 * one row per round.
 *
 * @param target The matrix, $n \times K$: row $i$ for round $i$, column $a$ for action $a$.
 * @param n The number of rounds.
 * @param where The caller's name, for error messages.
 * @param name What the matrix is, for the error message.
 * @returns The row-major values `pi` and the number of actions `K`.
 */
function readPolicy(target: MatrixLike, n: number, where: string, name = 'target policy'): { pi: F64; K: number } {
  const { data, m, n: K } = dense.toMatrixF64(target, where)
  if (m !== n) throw new DomainError(where, `${where}: the ${name} has ${m} rows for ${n} rounds`)
  return { pi: data, K }
}

/**
 * The importance weights $w_i = \pi(a_i \mid x_i)/\pi_0(a_i \mid x_i)$ of a log under a target policy $\pi$. Throws
 * `DomainError` for a malformed log, a policy without one row per round, or a logged action beyond its columns.
 *
 * @param log The logged actions and propensities (the rewards are checked but not used).
 * @param target The target policy, $n \times K$: row $i$ is $\pi(\cdot \mid x_i)$.
 * @returns The $n$ weights.
 *
 * @example Weights of a deterministic target
 * // Four logged rounds of two actions; the logger took action 1 with probability 0.8.
 * const log = { actions: [1, 0, 1, 0], rewards: [1, 0, 0.5, 0.2], propensities: [0.8, 0.2, 0.8, 0.2] }
 * // The target policy always takes action 1.
 * const target = [[0, 1], [0, 1], [0, 1], [0, 1]]
 * print('weights:', importanceWeights(log, target))
 */
export function importanceWeights(log: BanditLog, target: MatrixLike): Tensor {
  const where = 'importanceWeights'
  const { a, p0, n } = readLog(log, where)
  const { pi, K } = readPolicy(target, n, where)
  return dense.vec(Float64Array.from({ length: n }, (_, i) => pi[i * K + checkAction(a[i], K, where)] / p0[i]))
}

/**
 * A logged action, checked against the policy's number of actions; throws `DomainError` when it is not below it.
 *
 * @param a The logged action.
 * @param K The number of actions (columns) of the policy.
 * @param where The caller's name, for error messages.
 * @returns `a`.
 */
const checkAction = (a: number, K: number, where: string) => {
  if (a >= K) throw new DomainError(where, `${where}: action ${a} but the policy has ${K} actions`)
  return a
}

/**
 * The estimate, standard error, interval and weight summary of per-round terms (shared by the slate estimators).
 *
 * @param estimator The estimator's name, reported in the result.
 * @param terms The per-round terms, $n$ values; their mean is the estimate unless `value` is given.
 * @param weights The importance weights to report and summarise by Kish's effective sample size.
 * @param level The coverage of the normal-approximation interval, in $(0, 1)$.
 * @param value The estimate, when it is not the mean of `terms`.
 * @param seTerms The per-round terms whose sample standard deviation gives the standard error, when they are not
 *   `terms` (the delta-method terms of SNIPS).
 * @returns The estimate with its standard error, interval, terms, weights and effective sample size.
 */
export function summarise(
  estimator: string,
  terms: F64,
  weights: F64,
  level: number,
  value?: number,
  seTerms?: F64,
): OffPolicyEstimate {
  const n = terms.length
  const mean = terms.reduce((a, b) => a + b, 0) / n
  const v = value ?? mean
  const se0 = seTerms ?? terms
  const m0 = se0.reduce((a, b) => a + b, 0) / n
  const variance = n > 1 ? se0.reduce((a, b) => a + (b - m0) ** 2, 0) / (n - 1) : 0
  const standardError = Math.sqrt(variance / n)
  const z = normalQuantile(0.5 + level / 2) as number
  return {
    estimator,
    value: v,
    standardError,
    lower: v - z * standardError,
    upper: v + z * standardError,
    terms: dense.vec(terms),
    weights: dense.vec(weights),
    // Kish's ESS needs non-negative weights; the slate pseudo-inverse's may be negative.
    effectiveSampleSize:
      weights.every((w) => w >= 0) && weights.some((w) => w > 0) ? importanceEffectiveSampleSize(weights) : NaN,
  }
}

/**
 * Inverse propensity scoring (Horvitz and Thompson, 1952): $\hat V = \frac1n \sum_i w_i r_i$ with
 * $w_i = \pi(a_i \mid x_i)/\pi_0(a_i \mid x_i)$. Unbiased whenever $\pi_0 > 0$ wherever $\pi > 0$; its variance
 * grows with the weights, so small propensities make it noisy.
 *
 * @param log The logged rounds.
 * @param target The target policy, $n \times K$: row $i$ is $\pi(\cdot \mid x_i)$.
 * @param options The interval's level.
 * @returns The estimate, with terms $w_i r_i$.
 *
 * @example IPS of a deterministic target, checked by hand
 * // Four logged rounds of two actions; the logger took action 1 with probability 0.8.
 * const log = { actions: [1, 0, 1, 0], rewards: [1, 0, 0.5, 0.2], propensities: [0.8, 0.2, 0.8, 0.2] }
 * // The target policy always takes action 1.
 * const target = [[0, 1], [0, 1], [0, 1], [0, 1]]
 * // The weights are 1 / 0.8 = 1.25 on the rounds of action 1 and 0 elsewhere: (1.25 + 0.625) / 4.
 * const est = ips(log, target)
 * print('value:', est.value)
 * print('weights:', est.weights)
 * print('95% interval:', est.lower, est.upper)
 */
export function ips(log: BanditLog, target: MatrixLike, options: EstimateOptions = {}): OffPolicyEstimate {
  const where = 'ips'
  const { r, n } = readLog(log, where)
  const w = dense.data(importanceWeights(log, target))
  return summarise(
    'IPS',
    Float64Array.from({ length: n }, (_, i) => w[i] * r[i]),
    w,
    options.level ?? 0.95,
  )
}

/**
 * Clipped (capped) IPS (Bottou et al., 2013): the weights are capped at $M$, $w_i \leftarrow \min(w_i, M)$, and
 * $\hat V = \frac1n \sum_i w_i r_i$. Clipping trades variance for a downward bias (for non-negative rewards) that
 * grows as $M$ falls. Throws `DomainError` unless $M > 0$.
 *
 * @param log The logged rounds.
 * @param target The target policy, $n \times K$: row $i$ is $\pi(\cdot \mid x_i)$.
 * @param options The interval's level, and the cap.
 * @param options.clip The cap $M$ on the weights (default 10).
 * @returns The estimate, with the clipped weights.
 *
 * @example A propensity of 0.1 gives a weight of 10; a cap of 5 halves it
 * const log = { actions: [1, 1, 0, 0], rewards: [1, 1, 0, 0], propensities: [0.1, 0.5, 0.5, 0.9] }
 * const target = [[0, 1], [0, 1], [0, 1], [0, 1]]
 * const plain = ips(log, target)
 * const clipped = clippedIps(log, target, { clip: 5 })
 * print('IPS:', plain.value, 'weights', plain.weights)
 * print('clipped:', clipped.value, 'weights', clipped.weights)
 */
export function clippedIps(
  log: BanditLog,
  target: MatrixLike,
  options: EstimateOptions & { clip?: number } = {},
): OffPolicyEstimate {
  const where = 'clippedIps'
  const M = options.clip ?? 10
  if (!(M > 0)) throw new DomainError(where, `${where}: the clip must be positive`)
  const { r, n } = readLog(log, where)
  const w = dense.data(importanceWeights(log, target)).map((v) => Math.min(v, M))
  return summarise(
    'clipped IPS',
    Float64Array.from({ length: n }, (_, i) => w[i] * r[i]),
    w,
    options.level ?? 0.95,
  )
}

/**
 * Self-normalised IPS (Swaminathan and Joachims, 2015): $\hat V = \sum_i w_i r_i / \sum_i w_i$. Biased but
 * consistent, bounded by the range of the rewards, and invariant to adding a constant to every reward; its standard
 * error is by the delta method, from the terms $w_i (r_i - \hat V)/\bar w$ with $\bar w$ the mean weight. Throws
 * `DomainError` when every weight is zero.
 *
 * @param log The logged rounds.
 * @param target The target policy, $n \times K$: row $i$ is $\pi(\cdot \mid x_i)$.
 * @param options The interval's level.
 * @returns The estimate, with terms $w_i r_i / \bar w$ (whose mean is the estimate).
 *
 * @example Self-normalising divides by the total weight, not by n
 * // Four logged rounds of two actions; the logger took action 1 with probability 0.8.
 * const log = { actions: [1, 0, 1, 0], rewards: [1, 0, 0.5, 0.2], propensities: [0.8, 0.2, 0.8, 0.2] }
 * // The target policy always takes action 1.
 * const target = [[0, 1], [0, 1], [0, 1], [0, 1]]
 * // The sum of w r is 1.25 + 0.625 and the sum of w is 2.5: the mean reward of the two rounds of action 1.
 * print('IPS:', ips(log, target).value)
 * print('SNIPS:', snips(log, target).value)
 */
export function snips(log: BanditLog, target: MatrixLike, options: EstimateOptions = {}): OffPolicyEstimate {
  const where = 'snips'
  const { r, n } = readLog(log, where)
  const w = dense.data(importanceWeights(log, target))
  const total = w.reduce((a, b) => a + b, 0)
  if (!(total > 0)) throw new DomainError(where, `${where}: every importance weight is zero`)
  const mean = total / n
  const value = w.reduce((s, v, i) => s + v * r[i], 0) / total
  const terms = Float64Array.from({ length: n }, (_, i) => (w[i] * r[i]) / mean)
  const delta = Float64Array.from({ length: n }, (_, i) => (w[i] * (r[i] - value)) / mean + value)
  return summarise('SNIPS', terms, w, options.level ?? 0.95, value, delta)
}

/**
 * $\sum_a \pi(a \mid x_i) \hat q(x_i, a)$ for each round: the model's value of $\pi$ at each context.
 *
 * @param pi The target policy, row-major $n \times K$.
 * @param q The reward model, row-major $n \times K$.
 * @param n The number of rounds.
 * @param K The number of actions.
 * @returns A new array of $n$ values.
 */
function modelValues(pi: F64, q: F64, n: number, K: number): F64 {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) out[i] += pi[i * K + k] * q[i * K + k]
  return out
}

/**
 * A reward model as a dense row-major array, checked to be $n \times K$. Throws `DomainError` otherwise.
 *
 * @param model The reward model $\hat q$, $n \times K$.
 * @param n The number of rounds.
 * @param K The number of actions of the target policy.
 * @param where The caller's name, for error messages.
 * @returns The row-major values.
 */
function readModel(model: MatrixLike, n: number, K: number, where: string): F64 {
  const { data, m, n: k } = dense.toMatrixF64(model, where)
  if (m !== n || k !== K) throw new DomainError(where, `${where}: the reward model is ${m}×${k}, expected ${n}×${K}`)
  return data
}

/**
 * The direct method: $\hat V = \frac1n \sum_i \sum_a \pi(a \mid x_i) \hat q(x_i, a)$ from a reward model $\hat q$.
 * Low variance, and biased exactly as much as the model is wrong on the actions $\pi$ takes. The log's actions and
 * rewards are not used (only checked), and every reported weight is 1.
 *
 * @param log The logged rounds; only their number is used.
 * @param target The target policy, $n \times K$: row $i$ is $\pi(\cdot \mid x_i)$.
 * @param model The reward model, $n \times K$: entry $(i, a)$ is $\hat q(x_i, a)$.
 * @param options The interval's level.
 * @returns The estimate, with terms $\sum_a \pi(a \mid x_i) \hat q(x_i, a)$.
 *
 * @example The model's value of the target
 * // Four logged rounds of two actions; the logger took action 1 with probability 0.8.
 * const log = { actions: [1, 0, 1, 0], rewards: [1, 0, 0.5, 0.2], propensities: [0.8, 0.2, 0.8, 0.2] }
 * // The target policy always takes action 1.
 * const target = [[0, 1], [0, 1], [0, 1], [0, 1]]
 * // A reward model that predicts 0.1 for action 0 and 0.7 for action 1 in every round.
 * const model = [[0.1, 0.7], [0.1, 0.7], [0.1, 0.7], [0.1, 0.7]]
 * print('DM:', directMethod(log, target, model).value)
 */
export function directMethod(
  log: BanditLog,
  target: MatrixLike,
  model: MatrixLike,
  options: EstimateOptions = {},
): OffPolicyEstimate {
  const where = 'directMethod'
  const { n } = readLog(log, where)
  const { pi, K } = readPolicy(target, n, where)
  const q = readModel(model, n, K, where)
  return summarise('DM', modelValues(pi, q, n, K), new Float64Array(n).fill(1), options.level ?? 0.95)
}

/**
 * Doubly robust (Dudík, Langford and Li, 2011):
 * $\hat V = \frac1n \sum_i \big[\sum_a \pi(a \mid x_i) \hat q(x_i, a) + w_i (r_i - \hat q(x_i, a_i))\big]$: the
 * direct method corrected by IPS on its residuals. Unbiased when either the propensities or the model are right, and
 * of lower variance than IPS when the model is good.
 *
 * @param log The logged rounds.
 * @param target The target policy, $n \times K$: row $i$ is $\pi(\cdot \mid x_i)$.
 * @param model The reward model, $n \times K$: entry $(i, a)$ is $\hat q(x_i, a)$.
 * @param options The interval's level.
 * @returns The estimate, with the per-round terms in brackets above.
 *
 * @example The direct method corrected by the residuals, checked by hand
 * // Four logged rounds of two actions; the logger took action 1 with probability 0.8.
 * const log = { actions: [1, 0, 1, 0], rewards: [1, 0, 0.5, 0.2], propensities: [0.8, 0.2, 0.8, 0.2] }
 * // The target policy always takes action 1.
 * const target = [[0, 1], [0, 1], [0, 1], [0, 1]]
 * // A reward model that predicts 0.1 for action 0 and 0.7 for action 1 in every round.
 * const model = [[0.1, 0.7], [0.1, 0.7], [0.1, 0.7], [0.1, 0.7]]
 * // Residuals on the rounds of action 1: 1 - 0.7 and 0.5 - 0.7, each weighted 1.25.
 * print('DM:', directMethod(log, target, model).value)
 * print('DR:', doublyRobust(log, target, model).value)
 * print('by hand:', 0.7 + (1.25 * 0.3 - 1.25 * 0.2) / 4)
 */
export function doublyRobust(
  log: BanditLog,
  target: MatrixLike,
  model: MatrixLike,
  options: EstimateOptions = {},
): OffPolicyEstimate {
  const where = 'doublyRobust'
  const { a, r, n } = readLog(log, where)
  const { pi, K } = readPolicy(target, n, where)
  const q = readModel(model, n, K, where)
  const w = dense.data(importanceWeights(log, target))
  const base = modelValues(pi, q, n, K)
  const terms = Float64Array.from({ length: n }, (_, i) => base[i] + w[i] * (r[i] - q[i * K + a[i]]))
  return summarise('DR', terms, w, options.level ?? 0.95)
}

/**
 * Switch-DR (after Wang, Agarwal and Dudík, 2017, "Optimal and adaptive off-policy evaluation in contextual bandits",
 * ICML): a round whose importance weight $w_i$ is at most $\tau$ is estimated doubly robustly, and one whose weight is
 * larger by the model alone: $\hat V = \frac1n \sum_i t_i$ with
 * $t_i = \sum_a \pi(a \mid x_i) \hat q(x_i, a) + w_i (r_i - \hat q(x_i, a_i)) \indicator\{w_i \le \tau\}$.
 * $\tau = \infty$ is DR; $\tau = 0$ is the direct method. Needs the logging probabilities of every action
 * (`log.logging`) only to report the share of $\pi$'s mass switched: the mean over rounds of the target's mass on
 * actions with $\pi(a \mid x_i) > \tau \pi_0(a \mid x_i)$. Throws `DomainError` for a negative $\tau$.
 *
 * @param log The logged rounds, with `logging` to report the share switched.
 * @param target The target policy, $n \times K$: row $i$ is $\pi(\cdot \mid x_i)$.
 * @param model The reward model, $n \times K$: entry $(i, a)$ is $\hat q(x_i, a)$.
 * @param options The interval's level, and the threshold.
 * @param options.tau The threshold $\tau$ on the importance weight (default 10).
 * @returns The estimate, with the weights used (0 where switched to the model) and `switched`, the share of the
 *   target's mass switched (NaN without `log.logging`).
 *
 * @example A low threshold switches to the model
 * // Four logged rounds of two actions; the logger took action 1 with probability 0.8.
 * const log = {
 *   actions: [1, 0, 1, 0], rewards: [1, 0, 0.5, 0.2], propensities: [0.8, 0.2, 0.8, 0.2],
 *   logging: [[0.2, 0.8], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
 * }
 * // The target policy always takes action 1.
 * const target = [[0, 1], [0, 1], [0, 1], [0, 1]]
 * // A reward model that predicts 0.1 for action 0 and 0.7 for action 1 in every round.
 * const model = [[0.1, 0.7], [0.1, 0.7], [0.1, 0.7], [0.1, 0.7]]
 * // The weights of the target's action are 1.25: tau = 1 gives the direct method, tau = 2 gives DR.
 * const low = switchDoublyRobust(log, target, model, { tau: 1 })
 * const high = switchDoublyRobust(log, target, model, { tau: 2 })
 * print('tau = 1:', low.value, 'switched', low.switched)
 * print('tau = 2:', high.value, 'switched', high.switched)
 */
export function switchDoublyRobust(
  log: BanditLog,
  target: MatrixLike,
  model: MatrixLike,
  options: EstimateOptions & { tau?: number } = {},
): OffPolicyEstimate & { readonly switched: number } {
  const where = 'switchDoublyRobust'
  const tau = options.tau ?? 10
  if (!(tau >= 0)) throw new DomainError(where, `${where}: τ must be non-negative`)
  const { a, r, n } = readLog(log, where)
  const { pi, K } = readPolicy(target, n, where)
  const q = readModel(model, n, K, where)
  const w = dense.data(importanceWeights(log, target))
  const base = modelValues(pi, q, n, K)
  const used = w.map((v) => (v <= tau ? v : 0))
  const terms = Float64Array.from({ length: n }, (_, i) => base[i] + used[i] * (r[i] - q[i * K + a[i]]))
  // The share of the target's mass on actions whose weight exceeds τ (from every action's logging probability).
  let switched = NaN
  if (log.logging !== undefined) {
    const { pi: p0 } = readPolicy(log.logging, n, where, 'logging policy')
    let mass = 0
    for (let i = 0; i < n; i++)
      for (let k = 0; k < K; k++) if (pi[i * K + k] > 0 && pi[i * K + k] > tau * p0[i * K + k]) mass += pi[i * K + k]
    switched = mass / n
  }
  return { ...summarise('switch-DR', terms, used, options.level ?? 0.95), switched }
}
