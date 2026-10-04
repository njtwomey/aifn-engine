/**
 * Off-policy evaluation of a target policy π from logged bandit feedback: rounds i = 1 … n with a context xᵢ, an
 * action aᵢ drawn by a logging policy π₀, its propensity π₀(aᵢ | xᵢ) and the reward rᵢ. The value
 * V(π) = E_x Σₐ π(a | x) μ(x, a) is estimated by inverse propensity scoring (plain, clipped and self-normalised), the
 * direct method (a reward model q̂), doubly robust (both) and switch-DR (doubly robust where the importance weight is
 * at most τ, the model elsewhere). Each returns the estimate, a standard error and the importance weights it used.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { importanceEffectiveSampleSize } from 'aifn-compute/probability/stats'

type F64 = dense.F64

/** Logged bandit feedback: one entry per round. */
export interface BanditLog {
  /** The logged action aᵢ ∈ {0, …, K − 1} [n]. */
  readonly actions: VectorLike
  /** The observed reward rᵢ [n]. */
  readonly rewards: VectorLike
  /** The logging propensity π₀(aᵢ | xᵢ) of the logged action [n]. */
  readonly propensities: VectorLike
  /** π₀(a | xᵢ) for every action [n, K]; needed by switch-DR only. */
  readonly logging?: MatrixLike
}

/** The result of every off-policy estimator. */
export interface OffPolicyEstimate {
  /** The estimator's name. */
  readonly estimator: string
  /** The estimate V̂(π). */
  readonly value: number
  /** The standard error: the sample standard deviation of the per-round terms over √n (delta method for SNIPS). */
  readonly standardError: number
  /** The normal-approximation interval V̂ ± z SE at `level` (default 0.95). */
  readonly lower: number
  readonly upper: number
  /** The per-round terms whose mean is the estimate [n]. */
  readonly terms: Tensor
  /** The importance weights wᵢ = π(aᵢ | xᵢ)/π₀(aᵢ | xᵢ) used (after clipping where it applies) [n]. */
  readonly weights: Tensor
  /** Kish's effective sample size (Σw)²/Σw² of those weights (n for the direct method; NaN with negative weights). */
  readonly effectiveSampleSize: number
}

/** Options of every estimator. */
export type EstimateOptions = { level?: number }

type Read = { a: Int32Array; r: F64; p0: F64; n: number }

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

function readPolicy(target: MatrixLike, n: number, where: string, name = 'target policy'): { pi: F64; K: number } {
  const { data, m, n: K } = dense.toMatrixF64(target, where)
  if (m !== n) throw new DomainError(where, `${where}: the ${name} has ${m} rows for ${n} rounds`)
  return { pi: data, K }
}

/** The importance weights wᵢ = π(aᵢ | xᵢ)/π₀(aᵢ | xᵢ) of a log under a target policy π [n, K]. */
export function importanceWeights(log: BanditLog, target: MatrixLike): Tensor {
  const where = 'importanceWeights'
  const { a, p0, n } = readLog(log, where)
  const { pi, K } = readPolicy(target, n, where)
  return dense.vec(Float64Array.from({ length: n }, (_, i) => pi[i * K + checkAction(a[i], K, where)] / p0[i]))
}

const checkAction = (a: number, K: number, where: string) => {
  if (a >= K) throw new DomainError(where, `${where}: action ${a} but the policy has ${K} actions`)
  return a
}

/** The estimate, standard error, interval and weight summary of per-round terms (shared by the slate estimators). */
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
 * Inverse propensity scoring (Horvitz and Thompson, 1952): V̂ = (1/n) Σ wᵢrᵢ with wᵢ = π(aᵢ|xᵢ)/π₀(aᵢ|xᵢ). Unbiased
 * whenever π₀ > 0 wherever π > 0; its variance grows with the weights, so small propensities make it noisy.
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
 * Clipped (capped) IPS (Bottou et al., 2013): the weights are capped at M, wᵢ ← min(wᵢ, M). Clipping trades variance
 * for a downward bias (for non-negative rewards) that grows as M falls.
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
 * Self-normalised IPS (Swaminathan and Joachims, 2015): V̂ = Σ wᵢrᵢ / Σ wᵢ. Biased but consistent, bounded by the
 * range of the rewards, and invariant to adding a constant to every reward; its standard error is by the delta
 * method, from the terms wᵢ(rᵢ − V̂)/w̄.
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

/** Σₐ π(a|xᵢ) q̂(xᵢ, a) for each round: the model's value of π at each context. */
function modelValues(pi: F64, q: F64, n: number, K: number): F64 {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) out[i] += pi[i * K + k] * q[i * K + k]
  return out
}

function readModel(model: MatrixLike, n: number, K: number, where: string): F64 {
  const { data, m, n: k } = dense.toMatrixF64(model, where)
  if (m !== n || k !== K) throw new DomainError(where, `${where}: the reward model is ${m}×${k}, expected ${n}×${K}`)
  return data
}

/**
 * The direct method: V̂ = (1/n) Σᵢ Σₐ π(a|xᵢ) q̂(xᵢ, a) from a reward model q̂ [n, K]. Low variance, and biased exactly
 * as much as the model is wrong on the actions π takes. The log's actions and rewards are not used.
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
 * Doubly robust (Dudík, Langford and Li, 2011): V̂ = (1/n) Σᵢ [Σₐ π(a|xᵢ) q̂(xᵢ, a) + wᵢ(rᵢ − q̂(xᵢ, aᵢ))]: the direct
 * method corrected by IPS on its residuals. Unbiased when either the propensities or the model are right, and of
 * lower variance than IPS when the model is good.
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
 * Switch-DR (after Wang, Agarwal and Dudík, 2017): for each round, the actions whose importance weight
 * π(a|xᵢ)/π₀(a|xᵢ) is at most τ are estimated doubly robustly and the rest by the model alone:
 * V̂ = (1/n) Σᵢ [Σₐ π(a|xᵢ) q̂(xᵢ, a) + wᵢ(rᵢ − q̂(xᵢ, aᵢ)) 1{wᵢ ≤ τ}]. τ = ∞ is DR; τ = 0 is the direct method.
 * Needs the logging probabilities of every action (`log.logging`) only to report the share of π's mass switched.
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
