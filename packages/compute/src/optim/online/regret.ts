/**
 * Regret bookkeeping and bounds: the regret trace of a learner against a comparator or the best expert so far, the best
 * loss of an expert sequence with at most $m$ switches (the comparator of tracking), and the regret bounds of Hedge and
 * online gradient descent that the traces are drawn against.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'

/** The result of `regretTrace`. */
export interface RegretTrace {
  /** The learner's cumulative loss $\hat{L}_t$, $T$ values. */
  learner: Tensor
  /**
   * The comparator's cumulative loss, $T$ values: the given sequence's, or $\min_i L_{t,i}$ of the best expert so
   * far.
   */
  comparator: Tensor
  /** The regret $R_t = \hat{L}_t - C_t$, with $C_t$ the comparator's cumulative loss, $T$ values. */
  regret: Tensor
  /** The average regret $R_t/t$, $T$ values. */
  average: Tensor
}

/**
 * The regret over time of a learner whose per-round losses are `learner`, against a comparator's per-round losses
 * (`{ comparator }`) or against a matrix of expert losses (`{ experts }`), in which case the comparator at $t$ is the
 * best expert on rounds $1, \dots, t$ (the regret of the game stopped at $t$). Throws `DomainError` when the lengths
 * differ.
 *
 * @param learner The learner's loss in each round, $T$ values.
 * @param against `{ comparator }`, the comparator's loss in each round ($T$ values), or `{ experts }`, the experts'
 *   losses as a $T \times N$ matrix, one row per round.
 * @returns The cumulative losses of learner and comparator, the regret and the average regret, each by round.
 *
 * @example Against the best expert so far, the regret can fall
 * // Expert 0 is perfect for two rounds, then expert 1; the learner always pays 0.5.
 * const r = regretTrace([0.5, 0.5, 0.5, 0.5], { experts: [[0, 1], [0, 1], [1, 0], [1, 0]] })
 * print('learner =', r.learner)
 * print('best expert so far =', r.comparator)
 * print('regret =', r.regret)
 * print('average regret =', r.average)
 */
export function regretTrace(
  learner: VectorLike,
  against: { comparator: VectorLike } | { experts: MatrixLike },
): RegretTrace {
  const where = 'regretTrace'
  const l = dense.toF64(learner, where)
  const T = l.length
  const c = new Float64Array(T)
  if ('experts' in against) {
    const { data, m, n } = dense.toMatrixF64(against.experts, where)
    if (m !== T) throw new DomainError(where, `${where}: ${m} rounds of expert losses for ${T} learner losses`)
    const L = new Float64Array(n)
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < n; i++) L[i] += data[t * n + i]
      c[t] = Math.min(...L)
    }
  } else {
    const u = dense.toF64(against.comparator, where)
    if (u.length !== T) throw new DomainError(where, `${where}: ${u.length} comparator losses for ${T} rounds`)
    for (let t = 0; t < T; t++) c[t] = (t > 0 ? c[t - 1] : 0) + u[t]
  }
  const cum = new Float64Array(T)
  for (let t = 0; t < T; t++) cum[t] = (t > 0 ? cum[t - 1] : 0) + l[t]
  const regret = cum.map((v, t) => v - c[t])
  return {
    learner: dense.vec(cum),
    comparator: dense.vec(c),
    regret: dense.vec(regret),
    average: dense.vec(regret.map((v, t) => v / (t + 1))),
  }
}

/**
 * The least cumulative loss of a sequence of experts that switches at most $m$ times over the rounds of `losses`, for
 * each $m = 0, \dots,$ `switches`: a dynamic programme over (round, switches used, current expert) in $O(TmN)$.
 * Entry 0 is the best single expert's loss. Throws `DomainError` unless `switches` is a non-negative integer.
 *
 * @param losses The experts' losses as a $T \times N$ matrix, one row per round.
 * @param switches The largest number of switches $m$.
 * @returns The least losses for $0, 1, \dots, m$ switches, $m + 1$ values (non-increasing).
 *
 * @example Two switches are enough for a sequence of three phases
 * // Expert 0 is best for two rounds, expert 1 for the next two, expert 0 again for the last two.
 * const losses = [[0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1]]
 * print('0, 1, 2, 3 switches =', bestSwitchingLoss(losses, 3))
 */
export function bestSwitchingLoss(losses: MatrixLike, switches: number): Tensor {
  const where = 'bestSwitchingLoss'
  const { data, m: T, n: N } = dense.toMatrixF64(losses, where)
  if (!Number.isInteger(switches) || switches < 0)
    throw new DomainError(where, `${where}: switches must be an integer ≥ 0`)
  const K = switches + 1
  // best[k][i]: least loss so far ending on expert i with exactly k switches.
  let best = Array.from({ length: K }, (_, k) =>
    Float64Array.from({ length: N }, (_, i) => (k === 0 ? (T > 0 ? data[i] : 0) : Infinity)),
  )
  for (let t = 1; t < T; t++) {
    const next = best.map(() => new Float64Array(N))
    for (let k = 0; k < K; k++) {
      const prevMin = k > 0 ? Math.min(...best[k - 1]) : Infinity
      for (let i = 0; i < N; i++) next[k][i] = Math.min(best[k][i], prevMin) + data[t * N + i]
    }
    best = next
  }
  const out = new Float64Array(K)
  let running = Infinity
  for (let k = 0; k < K; k++) {
    running = Math.min(running, Math.min(...best[k]))
    out[k] = running
  }
  return dense.vec(out)
}

/**
 * Hedge's tuned learning rate $\eta = \sqrt{8 \ln N / T}$ for $N$ experts and $T$ rounds. Throws `DomainError`
 * unless both are at least 1. (For one expert it is 0, where `hedge`'s `'tuned'` rate uses $\ln 2$ for $\ln N$.)
 *
 * @param rounds The horizon $T$.
 * @param experts The number of experts $N$.
 * @returns The rate $\eta$.
 *
 * @example The rate for ten experts over a hundred rounds
 * print('eta =', hedgeTunedRate(100, 10))
 */
export function hedgeTunedRate(rounds: number, experts: number): number {
  if (!(rounds >= 1 && experts >= 1)) throw new DomainError('hedgeTunedRate', 'hedgeTunedRate: needs T, N ≥ 1')
  return Math.sqrt((8 * Math.log(experts)) / rounds)
}

/**
 * Hedge's regret bound after $T$ rounds with $N$ experts and losses in $[0, 1]$: $\ln N/\eta + \eta T/8$ for a
 * constant rate $\eta$, or $\sqrt{T \ln N / 2}$ at the tuned rate (`eta` omitted), or
 * $2\sqrt{T \ln N / 2} + \sqrt{\ln N / 8}$ for the anytime rate (`'anytime'`). Throws `DomainError` for a constant
 * rate that is not positive.
 *
 * @param rounds The number of rounds $T$.
 * @param experts The number of experts $N$.
 * @param eta The learning rate: a constant $\eta > 0$, `'anytime'`, or omitted for the tuned rate.
 * @returns The bound on the regret against the best expert.
 *
 * @example The tuned rate minimises the bound for a constant rate
 * const eta = hedgeTunedRate(100, 10)
 * print('tuned =', hedgeRegretBound(100, 10))
 * print('at the tuned rate =', hedgeRegretBound(100, 10, eta))
 * print('at eta = 1 =', hedgeRegretBound(100, 10, 1))
 * print('anytime =', hedgeRegretBound(100, 10, 'anytime'))
 */
export function hedgeRegretBound(rounds: number, experts: number, eta?: number | 'anytime'): number {
  const lnN = Math.log(experts)
  if (eta === undefined) return Math.sqrt((rounds * lnN) / 2)
  if (eta === 'anytime') return 2 * Math.sqrt((rounds * lnN) / 2) + Math.sqrt(lnN / 8)
  if (!(eta > 0)) throw new DomainError('hedgeRegretBound', 'hedgeRegretBound: η must be positive')
  return lnN / eta + (eta * rounds) / 8
}

/**
 * The regret bound $\frac{3}{2}DG\sqrt{T}$ of online gradient descent with $\eta_t = D/(G\sqrt{t})$
 * (`ogdStepSize`). The arguments are not checked.
 *
 * @param rounds The number of rounds $T$.
 * @param diameter The diameter $D$ of the domain.
 * @param gradientBound The bound $G$ on the subgradients' norms.
 * @returns The bound on the regret against any fixed point of the domain.
 *
 * @example A hundred rounds on a domain of diameter 2 with unit gradients
 * print('bound =', ogdRegretBound(100, 2, 1))
 */
export function ogdRegretBound(rounds: number, diameter: number, gradientBound: number): number {
  return 1.5 * diameter * gradientBound * Math.sqrt(rounds)
}
