/**
 * Prediction with expert advice: each round $t = 1, \dots, T$ the learner plays a distribution $\pvec_t$ over $N$
 * experts, the environment reveals a loss vector $\boldsymbol{\ell}_t \in [0, 1]^N$, and the learner pays
 * $\langle \pvec_t, \boldsymbol{\ell}_t \rangle$. Hedge (exponential weights), fixed share (Hedge that hands back a
 * share $\alpha$ of the mass each round, to track a switching expert) and weighted majority (binary predictions,
 * multiplicative penalties), each as a step-through algorithm whose state carries the regret against the best expert
 * so far.
 *
 * Every algorithm is started with `undefined`, takes one step per round, and is done after the horizon. The losses
 * may be any finite numbers (a non-finite one throws `DomainError`), but the regret bounds quoted assume
 * $[0, 1]$. The reference is Cesa-Bianchi and Lugosi (2006), "Prediction, Learning, and Games", chapter 2.
 */

import type { MatrixLike, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { softmax } from 'aifn-compute/numerics/special'

type F64 = dense.F64

/**
 * Where the losses come from: a fixed $T \times N$ matrix, one row per round (an oblivious adversary or a sampled
 * sequence), or a function of the round $t$ (1-based), the distribution the learner is about to play and the experts'
 * cumulative losses so far, returning the $N$ losses of the round (an adaptive adversary, which sees $\pvec_t$ before
 * choosing $\boldsymbol{\ell}_t$).
 */
export type ExpertLosses = MatrixLike | ((t: number, weights: Tensor, cumulative: Tensor) => VectorLike)

/**
 * A learning rate for Hedge: a constant $\eta > 0$, `'tuned'` ($\eta = \sqrt{8 \ln N / T}$, which needs the horizon
 * $T$) or `'anytime'` ($\eta_t = \sqrt{8 \ln N / t}$ in round $t$, no horizon needed). With one expert, $\ln 2$ is
 * used for $\ln N$.
 */
export type HedgeRate = number | 'tuned' | 'anytime'

/** One state of `hedge` and `fixedShare`. */
export interface ExpertsState extends Status {
  /** The distribution $\pvec_{t+1}$ the learner plays next (uniform at $t = 0$). */
  weights: Tensor
  /** The distribution played in the last round, $\pvec_t$ (uniform at $t = 0$). */
  played: Tensor
  /** The last loss vector $\boldsymbol{\ell}_t$ (zeros at $t = 0$). */
  loss: Tensor
  /** Each expert's cumulative loss $L_{t,i} = \sum_{s \le t} \ell_{s,i}$. */
  cumulative: Tensor
  /** The learner's cumulative loss $\hat{L}_t = \sum_{s \le t} \langle \pvec_s, \boldsymbol{\ell}_s \rangle$. */
  learnerLoss: number
  /** The regret against the best expert so far, $R_t = \hat{L}_t - \min_i L_{t,i}$. */
  regret: number
  /** The index of the best expert so far (the first of ties). */
  best: number
  /** The learning rate used in the last round (the first round's at $t = 0$). */
  eta: number
  /** The horizon has been reached. */
  done: boolean
}

/**
 * Read the loss source: the number of experts, the horizon and a function returning $\boldsymbol{\ell}_t$. Throws
 * `DomainError` for a loss function without `experts`, a matrix with no columns, or (when called) a round with the
 * wrong number of losses.
 *
 * @param losses The losses, as a matrix or a function of the round.
 * @param options `experts`, the number $N$ (needed for a function), and `rounds`, the horizon (default: the matrix's
 *   rows, at most; unbounded for a function).
 * @param where The caller's name for error messages.
 * @returns `N`, the horizon `T`, and `at(t, p, L)`, the losses of round $t$ given the distribution about to be played
 *   and the cumulative losses so far (for a matrix, a view of row $t$, not to be modified).
 */
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

/**
 * The softmax of $-\eta\boldsymbol{L}$ (numerics' softmax, which shifts by the maximum so no weight underflows for
 * every expert).
 *
 * @param L The cumulative losses $\boldsymbol{L}$, one per expert.
 * @param eta The learning rate $\eta$.
 * @returns The weights $\propto \exp(-\eta L_i)$, summing to 1.
 */
const exponentialWeights = (L: F64, eta: number): F64 =>
  Float64Array.from(dense.data(softmax(dense.vec(L.map((v) => -eta * v)))))

/**
 * The index of the smallest entry (the first of ties).
 *
 * @param a The values, at least one.
 * @returns The index.
 */
const argmin = (a: ArrayLike<number>): number => {
  let k = 0
  for (let i = 1; i < a.length; i++) if (a[i] < a[k]) k = i
  return k
}

/**
 * The shared driver of the experts algorithms: plays `weights`, reads $\boldsymbol{\ell}_t$, books the losses and asks
 * `next` for $\pvec_{t+1}$. When stepped, throws `DomainError` for a non-finite loss.
 *
 * @param name The algorithm's name.
 * @param where The caller's name for error messages.
 * @param losses The losses, as a matrix or a function of the round.
 * @param options `experts` and `rounds`, as `readLosses` takes them.
 * @param rate The learning rate $\eta_t$ of round $t \ge 1$.
 * @param next The update: from the distribution played $\pvec_t$, the losses $\boldsymbol{\ell}_t$, the cumulative
 *   losses $L_t$ (round $t$ included) and the rate of the next round $\eta_{t+1}$, the distribution $\pvec_{t+1}$.
 * @returns The algorithm, started with `undefined`.
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

/**
 * The learning rate $\eta_t$ of round $t$ for a `HedgeRate`. Throws `DomainError` for a rate that is not positive,
 * or `'tuned'` without a finite horizon.
 *
 * @param eta The rate as given: a constant, `'tuned'` or `'anytime'`.
 * @param N The number of experts.
 * @param T The horizon (Infinity when unbounded).
 * @param where The caller's name for error messages.
 * @returns The rate as a function of the round $t \ge 1$.
 */
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
  /** The horizon $T$ (default: the rows of the loss matrix, and at most those; unbounded for a function). */
  rounds?: number
}

/**
 * Hedge, the exponentially weighted average forecaster (Freund and Schapire, 1997): it plays
 * $p_{t,i} \propto \exp(-\eta L_{t-1,i})$. For losses in $[0, 1]$ its regret is at most $\ln N / \eta + \eta T/8$,
 * which the tuned rate $\eta = \sqrt{8 \ln N / T}$ brings to $\sqrt{T \ln N / 2}$ (`hedgeRegretBound`).
 * $\eta \to \infty$ is follow-the-leader; $\eta \to 0$ stays uniform.
 *
 * @param losses The losses: a $T \times N$ matrix, or a function of the round (then give `experts`).
 * @param options The learning rate (default `'tuned'` with a finite horizon, else `'anytime'`), the number of experts
 *   and the horizon.
 * @returns The algorithm, started with `undefined`, one step per round.
 *
 * @example The weight moves to the best expert
 * // Three experts over six rounds; expert 1 loses least.
 * const losses = [[1, 0, 0.5], [0, 0, 1], [1, 0.5, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0.5]]
 * const s = run(hedge(losses, { eta: 1 }), undefined, 100)
 * print('weights =', s.weights)
 * print('expert losses =', s.cumulative)
 * print('learner loss =', s.learnerLoss)
 * print('regret =', s.regret)
 *
 * @example Against an adaptive adversary, within the bound
 * // Each round the adversary gives loss 1 to the expert with the most weight, and 0 to the other.
 * const adversary = (t, p) => {
 *   const w = toFlat(p)
 *   return w.map((_, i) => (i === w.indexOf(Math.max(...w)) ? 1 : 0))
 * }
 * const s = run(hedge(adversary, { experts: 2, rounds: 100 }), undefined, 1000)
 * print('expert losses =', s.cumulative)
 * print('learner loss =', s.learnerLoss)
 * print('regret =', s.regret)
 * print('bound =', hedgeRegretBound(100, 2))
 */
export function hedge(losses: ExpertLosses, options: HedgeOptions = {}): Algorithm<void, ExpertsState> {
  const where = 'hedge'
  const { N, T } = readLosses(losses, options, where)
  const rate = rateOf(options.eta ?? (Number.isFinite(T) ? 'tuned' : 'anytime'), N, T, where)
  return expertsGame('hedge', where, losses, options, rate, (_, __, L, eta) => exponentialWeights(L, eta))
}

/** Options of `fixedShare`. */
export type FixedShareOptions = HedgeOptions & {
  /**
   * The share rate $\alpha \in [0, 1)$: the fraction of mass handed back uniformly each round (default $1/T$ with a
   * finite horizon, else 0.01).
   */
  alpha?: number
}

/**
 * Fixed share (Herbster and Warmuth, 1998): a Hedge update $\vvec_t \propto \pvec_t e^{-\eta\boldsymbol{\ell}_t}$
 * followed by $\pvec_{t+1} = \alpha/N + (1 - \alpha)\vvec_t$, so no weight falls below $\alpha/N$ and an expert that
 * becomes the best takes over after about $\ln(N/\alpha)/\eta$ of loss difference, whatever happened before. With
 * $\alpha \approx m/T$ it tracks the best sequence of experts with $m$ switches. Throws `DomainError` for an
 * $\alpha$ outside $[0, 1)$.
 *
 * @param losses The losses: a $T \times N$ matrix, or a function of the round (then give `experts`).
 * @param options The options of `hedge`, with the share rate `alpha`.
 * @returns The algorithm, started with `undefined`, one step per round.
 *
 * @example Tracking an expert that changes halfway
 * // Expert 0 is right for 50 rounds, then expert 1: the best fixed expert loses 50, the best switching sequence 0.
 * const losses = Array.from({ length: 100 }, (_, t) => (t < 50 ? [0, 1] : [1, 0]))
 * const h = run(hedge(losses), undefined, 100)
 * const f = run(fixedShare(losses, { alpha: 0.02 }), undefined, 100)
 * print('hedge: learner loss =', h.learnerLoss, 'weights =', h.weights)
 * print('fixed share: learner loss =', f.learnerLoss, 'weights =', f.weights)
 * print('best with 0 and 1 switches =', bestSwitchingLoss(losses, 1))
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
  /** The last prediction: a label (deterministic) or the probability of predicting 1 (randomised); NaN at $t = 0$. */
  prediction: number
  /** The last outcome, 0 or 1 (NaN at $t = 0$). */
  outcome: number
  /** The learner's mistakes so far (expected mistakes for the randomised version). */
  mistakes: number
  /** Each expert's mistakes so far. */
  expertMistakes: Tensor
  /** The fewest mistakes of any expert so far, $m$. */
  bestMistakes: number
  /**
   * The mistake bound at the current $m$: $(\ln N + m \ln(1/\beta)) / \ln(2/(1 + \beta))$ (deterministic) or
   * $(\ln N + m \ln(1/\beta)) / (1 - \beta)$ (randomised); Infinity when $\beta = 0$ and $m > 0$.
   */
  bound: number
  /** Every round of advice has been played. */
  done: boolean
}

/**
 * Weighted majority (Littlestone and Warmuth, 1994): $N$ experts predict a binary label each round, the learner
 * predicts by a weighted vote (1 on a tie), the outcome $y$ is revealed, and every expert that erred has its weight
 * multiplied by $\beta \in [0, 1)$ ($\beta = 0$ is the halving algorithm, which restarts uniform should every expert
 * err). The randomised version predicts 1 with the weight share of the experts voting 1, and its expected mistakes are
 * Hedge's loss with $\eta = \ln(1/\beta)$. Throws `DomainError` when the outcomes and the advice differ in length,
 * or for a $\beta$ outside $[0, 1)$.
 *
 * @param advice The experts' predictions, a $T \times N$ matrix of 0 and 1, one row per round (an entry other than 1
 *   votes 0).
 * @param outcomes The $T$ true labels, 0 or 1.
 * @param options `beta`, the penalty factor $\beta$ (default 0.5), and `randomised`, whether to predict at random
 *   (default false).
 * @returns The algorithm, started with `undefined`, one step per round.
 *
 * @example Few mistakes when one expert is always right
 * // Expert 0 is always right; the others are right about half the time.
 * const advice = [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]]
 * const outcomes = [1, 0, 1, 0, 0, 1]
 * const s = run(weightedMajority(advice, outcomes), undefined, 100)
 * print('mistakes =', s.mistakes, 'bound =', s.bound)
 * print('expert mistakes =', s.expertMistakes)
 * print('weights =', s.weights)
 * const r = run(weightedMajority(advice, outcomes, { randomised: true }), undefined, 100)
 * print('randomised: expected mistakes =', r.mistakes, 'bound =', r.bound)
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
