/**
 * Prediction with expert advice: each round t = 1, …, T the learner plays a distribution pₜ over N experts, the
 * environment reveals a loss vector ℓₜ ∈ [0, 1]ᴺ, and the learner pays ⟨pₜ, ℓₜ⟩. Hedge (exponential weights), fixed
 * share (Hedge that hands back a share α of the mass each round, to track a switching expert) and weighted majority
 * (binary predictions, multiplicative penalties), each as a step-through algorithm whose state carries the regret
 * against the best expert so far.
 */

import type { MatrixLike, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { softmax } from 'aifn-compute/numerics/special'

type F64 = dense.F64

/**
 * Where the losses come from: a fixed matrix [T, N] (an oblivious adversary or a sampled sequence), or a function of
 * the round t (1-based), the distribution the learner is about to play and the experts' cumulative losses so far (an
 * adaptive adversary, which sees pₜ before choosing ℓₜ).
 */
export type ExpertLosses = MatrixLike | ((t: number, weights: Tensor, cumulative: Tensor) => VectorLike)

/**
 * A learning rate for Hedge: a constant η > 0, `'tuned'` (η = √(8 ln N / T), which needs the horizon T) or
 * `'anytime'` (ηₜ = √(8 ln N / t) in round t, no horizon needed).
 */
export type HedgeRate = number | 'tuned' | 'anytime'

/** One state of `hedge` and `fixedShare`. */
export interface ExpertsState extends Status {
  /** The distribution pₜ₊₁ the learner plays next (uniform at t = 0). */
  weights: Tensor
  /** The distribution played in the last round, pₜ (uniform at t = 0). */
  played: Tensor
  /** The last loss vector ℓₜ (zeros at t = 0). */
  loss: Tensor
  /** Each expert's cumulative loss L_{t,i} = Σ_{s ≤ t} ℓ_{s,i}. */
  cumulative: Tensor
  /** The learner's cumulative loss L̂ₜ = Σ_{s ≤ t} ⟨pₛ, ℓₛ⟩. */
  learnerLoss: number
  /** The regret against the best expert so far, Rₜ = L̂ₜ − minᵢ L_{t,i}. */
  regret: number
  /** The best expert so far (the first of ties). */
  best: number
  /** The learning rate used in the last round (the first round's at t = 0). */
  eta: number
  done: boolean
}

/** Read the loss source: the number of experts, the horizon and a function returning ℓₜ. */
function readLosses(
  losses: ExpertLosses,
  options: { experts?: number; rounds?: number },
  where: string,
): { N: number; T: number; at: (t: number, p: F64, L: F64) => F64 } {
  if (typeof losses === 'function') {
    const N = options.experts
    const T = options.rounds ?? Infinity
    if (N === undefined || !Number.isInteger(N) || N < 1)
      throw new DomainError(where, `${where}: give the number of experts when the losses are a function`)
    return {
      N,
      T,
      at: (t, p, L) => {
        const l = dense.toF64(losses(t, dense.vec(Float64Array.from(p)), dense.vec(Float64Array.from(L))), where)
        if (l.length !== N)
          throw new DomainError(where, `${where}: round ${t} gave ${l.length} losses for ${N} experts`)
        return l
      },
    }
  }
  const { data, m, n } = dense.toMatrixF64(losses, where)
  if (n < 1) throw new DomainError(where, `${where}: no experts`)
  return { N: n, T: Math.min(m, options.rounds ?? m), at: (t) => data.subarray((t - 1) * n, t * n) }
}

/** The softmax of −η L (numerics' softmax, which shifts by the maximum so no weight underflows for every expert). */
const exponentialWeights = (L: F64, eta: number): F64 =>
  Float64Array.from(dense.data(softmax(dense.vec(L.map((v) => -eta * v)))))

const argmin = (a: ArrayLike<number>): number => {
  let k = 0
  for (let i = 1; i < a.length; i++) if (a[i] < a[k]) k = i
  return k
}

/**
 * The shared driver of the experts algorithms: plays `weights`, reads ℓₜ, books the losses and asks `next` for pₜ₊₁.
 */
function expertsGame(
  name: string,
  where: string,
  losses: ExpertLosses,
  options: { experts?: number; rounds?: number },
  rate: (t: number) => number,
  next: (played: F64, loss: F64, cumulative: F64, eta: number) => F64,
): Algorithm<void, ExpertsState> {
  const { N, T, at } = readLosses(losses, options, where)
  const uniform = new Float64Array(N).fill(1 / N)
  return {
    name,
    init: () => ({
      t: 0,
      weights: dense.vec(uniform),
      played: dense.vec(uniform),
      loss: dense.vec(new Float64Array(N)),
      cumulative: dense.vec(new Float64Array(N)),
      learnerLoss: 0,
      regret: 0,
      best: 0,
      eta: rate(1),
      done: T === 0,
    }),
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1 }
      const t = s.t + 1
      const p = dense.data(s.weights)
      const L = Float64Array.from(dense.data(s.cumulative))
      const l = at(t, p, L)
      for (let i = 0; i < N; i++) {
        if (!Number.isFinite(l[i])) throw new DomainError(where, `${where}: round ${t} has a non-finite loss`)
        L[i] += l[i]
      }
      const learnerLoss = s.learnerLoss + dense.dot(p, l)
      const best = argmin(L)
      const eta = rate(t)
      return {
        t,
        weights: dense.vec(next(p, l, L, rate(t + 1))),
        played: dense.vec(Float64Array.from(p)),
        loss: dense.vec(Float64Array.from(l)),
        cumulative: dense.vec(L),
        learnerLoss,
        regret: learnerLoss - L[best],
        best,
        eta,
        done: t >= T,
      }
    },
    done: (s) => s.done,
  }
}

/** The learning rate ηₜ of round t for a `HedgeRate`. */
function rateOf(eta: HedgeRate, N: number, T: number, where: string): (t: number) => number {
  if (typeof eta === 'number') {
    if (!(eta > 0)) throw new DomainError(where, `${where}: the learning rate must be positive`)
    return () => eta
  }
  const scale = 8 * Math.log(Math.max(N, 2))
  if (eta === 'tuned') {
    if (!Number.isFinite(T)) throw new DomainError(where, `${where}: 'tuned' needs the horizon (rounds)`)
    const tuned = Math.sqrt(scale / Math.max(T, 1))
    return () => tuned
  }
  return (t) => Math.sqrt(scale / Math.max(t, 1))
}

/** Options of `hedge`. */
export type HedgeOptions = {
  /** The learning rate (default `'tuned'` with a horizon, else `'anytime'`). */
  eta?: HedgeRate
  /** The number of experts (needed when the losses are a function). */
  experts?: number
  /** The horizon T (default: the rows of the loss matrix; unbounded for a function). */
  rounds?: number
}

/**
 * Hedge, the exponentially weighted average forecaster (Freund and Schapire, 1997): it plays
 * p_{t,i} ∝ exp(−η L_{t−1,i}). For losses in [0, 1] its regret is at most ln N / η + ηT/8, which the tuned rate
 * η = √(8 ln N / T) brings to √(T ln N / 2) (`hedgeRegretBound`). η → ∞ is follow-the-leader; η → 0 stays uniform.
 */
export function hedge(losses: ExpertLosses, options: HedgeOptions = {}): Algorithm<void, ExpertsState> {
  const where = 'hedge'
  const { N, T } = readLosses(losses, options, where)
  const rate = rateOf(options.eta ?? (Number.isFinite(T) ? 'tuned' : 'anytime'), N, T, where)
  return expertsGame('hedge', where, losses, options, rate, (_, __, L, eta) => exponentialWeights(L, eta))
}

/** Options of `fixedShare`. */
export type FixedShareOptions = HedgeOptions & {
  /** The share rate α ∈ [0, 1): the fraction of mass handed back uniformly each round (default 1/T, else 0.01). */
  alpha?: number
}

/**
 * Fixed share (Herbster and Warmuth, 1998): a Hedge update vₜ ∝ pₜ e^{−ηℓₜ} followed by pₜ₊₁ = α/N + (1 − α)vₜ, so
 * no weight falls below α/N and an expert that becomes the best takes over after about ln(N/α)/η of loss difference,
 * whatever happened before. With α ≈ m/T it tracks the best sequence of experts with m switches.
 */
export function fixedShare(losses: ExpertLosses, options: FixedShareOptions = {}): Algorithm<void, ExpertsState> {
  const where = 'fixedShare'
  const { N, T } = readLosses(losses, options, where)
  const alpha = options.alpha ?? (Number.isFinite(T) ? 1 / Math.max(T, 1) : 0.01)
  if (!(alpha >= 0 && alpha < 1)) throw new DomainError(where, `${where}: α must be in [0, 1)`)
  const rate = rateOf(options.eta ?? (Number.isFinite(T) ? 'tuned' : 'anytime'), N, T, where)
  return expertsGame('fixed-share', where, losses, options, rate, (p, l, _, eta) => {
    // vₜ ∝ pₜ e^{−ηℓₜ} = softmax(log pₜ − ηℓₜ) (a zero weight stays zero).
    const v = exponentialWeights(
      Float64Array.from(p, (pi, i) => l[i] - Math.log(pi) / eta),
      eta,
    )
    return v.map((x) => alpha / N + (1 - alpha) * x)
  })
}

// ── Weighted majority ────────────────────────────────────────────────────────────────────────────────────────────────

/** One state of `weightedMajority`. */
export interface WeightedMajorityState extends Status {
  /** The experts' normalised weights before the next round. */
  weights: Tensor
  /** The last prediction: a label (deterministic) or the probability of predicting 1 (randomised); NaN at t = 0. */
  prediction: number
  /** The last outcome (NaN at t = 0). */
  outcome: number
  /** The learner's mistakes so far (expected mistakes for the randomised version). */
  mistakes: number
  /** Each expert's mistakes so far. */
  expertMistakes: Tensor
  /** The fewest mistakes of any expert so far, m. */
  bestMistakes: number
  /**
   * The mistake bound at the current m: (ln N + m ln(1/β)) / ln(2/(1 + β)) (deterministic) or
   * (ln N + m ln(1/β)) / (1 − β) (randomised).
   */
  bound: number
  done: boolean
}

/**
 * Weighted majority (Littlestone and Warmuth, 1994): N experts predict a binary label each round (`advice` [T, N] in
 * {0, 1}), the learner predicts by a weighted vote, the outcome y is revealed (`outcomes` [T]), and every expert that
 * erred has its weight multiplied by β ∈ [0, 1) (β = 0 is the halving algorithm). The randomised version predicts 1
 * with the weight share of the experts voting 1, and its expected mistakes are Hedge's loss with η = ln(1/β).
 */
export function weightedMajority(
  advice: MatrixLike,
  outcomes: VectorLike,
  options: { beta?: number; randomised?: boolean } = {},
): Algorithm<void, WeightedMajorityState> {
  const where = 'weightedMajority'
  const { data: x, m: T, n: N } = dense.toMatrixF64(advice, where)
  const y = dense.toF64(outcomes, where)
  if (y.length !== T) throw new DomainError(where, `${where}: ${y.length} outcomes for ${T} rounds of advice`)
  const beta = options.beta ?? 0.5
  if (!(beta >= 0 && beta < 1)) throw new DomainError(where, `${where}: β must be in [0, 1)`)
  const randomised = options.randomised ?? false
  const bound = (m: number) => {
    const penalty = beta === 0 ? (m > 0 ? Infinity : 0) : m * Math.log(1 / beta)
    return (Math.log(N) + penalty) / (randomised ? 1 - beta : Math.log(2 / (1 + beta)))
  }
  return {
    name: randomised ? 'randomised-weighted-majority' : 'weighted-majority',
    init: () => ({
      t: 0,
      weights: dense.vec(new Float64Array(N).fill(1 / N)),
      prediction: NaN,
      outcome: NaN,
      mistakes: 0,
      expertMistakes: dense.vec(new Float64Array(N)),
      bestMistakes: 0,
      bound: bound(0),
      done: T === 0,
    }),
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1 }
      const t = s.t + 1
      const w = Float64Array.from(dense.data(s.weights))
      const row = x.subarray((t - 1) * N, t * N)
      let ones = 0
      for (let i = 0; i < N; i++) if (row[i] === 1) ones += w[i]
      const total = w.reduce((a, b) => a + b, 0)
      const share = total > 0 ? ones / total : 0.5
      const prediction = randomised ? share : share >= 0.5 ? 1 : 0
      const outcome = y[t - 1]
      const mistake = randomised ? (outcome === 1 ? 1 - share : share) : prediction === outcome ? 0 : 1
      const em = Float64Array.from(dense.data(s.expertMistakes))
      for (let i = 0; i < N; i++)
        if (row[i] !== outcome) {
          em[i] += 1
          w[i] *= beta
        }
      const after = w.reduce((a, b) => a + b, 0)
      // With β = 0 every expert may be eliminated (no expert is perfect): restart uniform, as the halving algorithm
      // has no rule for that case.
      const weights = after > 0 ? w.map((v) => v / after) : new Float64Array(N).fill(1 / N)
      const bestMistakes = Math.min(...em)
      return {
        t,
        weights: dense.vec(weights),
        prediction,
        outcome,
        mistakes: s.mistakes + mistake,
        expertMistakes: dense.vec(em),
        bestMistakes,
        bound: bound(bestMistakes),
        done: t >= T,
      }
    },
    done: (s) => s.done,
  }
}
