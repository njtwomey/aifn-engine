/**
 * Regret bookkeeping and bounds: the regret trace of a learner against a comparator or the best expert so far, the best
 * loss of an expert sequence with at most m switches (the comparator of tracking), and the regret bounds of Hedge and
 * online gradient descent that the traces are drawn against.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'

/** The result of `regretTrace`. */
export interface RegretTrace {
  /** The learner's cumulative loss L̂ₜ [T]. */
  learner: Tensor
  /** The comparator's cumulative loss [T]: the given sequence's, or minᵢ L_{t,i} of the best expert so far. */
  comparator: Tensor
  /** Rₜ = L̂ₜ − comparatorₜ [T]. */
  regret: Tensor
  /** Rₜ/t [T]. */
  average: Tensor
}

/**
 * The regret over time of a learner whose per-round losses are `learner` [T], against a comparator's per-round losses
 * (`{ comparator }` [T]) or against a matrix of expert losses (`{ experts }` [T, N]), in which case the comparator at t
 * is the best expert on rounds 1 … t (the regret of the game stopped at t).
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
 * The least cumulative loss of a sequence of experts that switches at most m times over the rounds of `losses` [T, N],
 * for each m = 0 … switches: a dynamic programme over (round, switches used, current expert) in O(T · m · N). Entry 0
 * is the best single expert's loss.
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

/** Hedge's tuned learning rate η = √(8 ln N / T) for N experts and T rounds. */
export function hedgeTunedRate(rounds: number, experts: number): number {
  if (!(rounds >= 1 && experts >= 1)) throw new DomainError('hedgeTunedRate', 'hedgeTunedRate: needs T, N ≥ 1')
  return Math.sqrt((8 * Math.log(experts)) / rounds)
}

/**
 * Hedge's regret bound after T rounds with N experts and losses in [0, 1]: ln N/η + ηT/8 for a constant rate η, or
 * √(T ln N / 2) at the tuned rate (η omitted), or 2√(T ln N / 2) + √(ln N / 8) for the anytime rate (`'anytime'`).
 */
export function hedgeRegretBound(rounds: number, experts: number, eta?: number | 'anytime'): number {
  const lnN = Math.log(experts)
  if (eta === undefined) return Math.sqrt((rounds * lnN) / 2)
  if (eta === 'anytime') return 2 * Math.sqrt((rounds * lnN) / 2) + Math.sqrt(lnN / 8)
  if (!(eta > 0)) throw new DomainError('hedgeRegretBound', 'hedgeRegretBound: η must be positive')
  return lnN / eta + (eta * rounds) / 8
}

/** The regret bound (3/2)DG√T of online gradient descent with ηₜ = D/(G√t). */
export function ogdRegretBound(rounds: number, diameter: number, gradientBound: number): number {
  return 1.5 * diameter * gradientBound * Math.sqrt(rounds)
}
