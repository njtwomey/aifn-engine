/**
 * Off-policy evaluation of slates: a page of $l$ slots filled from $m$ items. Slate-level IPS reweights by the
 * probability of the whole logged slate, which is tiny when the logging policy randomises over $m!/(m - l)!$
 * orderings. The pseudo-inverse estimator (Swaminathan et al., 2017, "Off-policy evaluation for slate
 * recommendation", NeurIPS) assumes the reward is additive over (slot, item) pairs, and then needs only the logging
 * policy's pairwise marginals $\Gammamat = \expect_\mu[\ones_s \ones_s^\top]$:
 * $\hat V = \frac1n \sum_i r_i \thetavec_i^\top \Gammamat^{+} \ones_{s_i}$, where $\ones_s$ is the indicator of
 * the slate's (slot, item) pairs (of length $lm$, slot $j$'s block at $jm$) and $\thetavec_i$ the target's expected
 * indicator.
 *
 * The logging policy $\mu$ is the same in every round, given as uniform over ordered slates of distinct items or as an
 * explicit list of slates with their probabilities. Target policies are deterministic: the slate shown in each round.
 * Both estimators return the `OffPolicyEstimate` of `ips` and the others.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense } from 'aifn-compute/foundation/tensor'
import { pinv } from 'aifn-compute/numerics/linalg'
import { summarise, type EstimateOptions, type OffPolicyEstimate } from './estimators'

/** A dense float64 array, as `dense` reads vectors and matrices. */
type F64 = dense.F64

/** Logged slates and their rewards. */
export interface SlateLog {
  /** The logged slates, $n \times l$: row $i$ holds the item in each slot of round $i$. */
  readonly slates: MatrixLike
  /** The reward of each logged slate, $n$ values. */
  readonly rewards: VectorLike
}

/**
 * The logging policy over slates, the same for every context: uniform over the ordered slates of $l$ distinct items
 * from $m$ (`{ kind: 'uniform', items: m }`), or an explicit list of slates ($k \times l$) with their probabilities
 * ($k$ values; a slate listed twice adds up).
 */
export type SlateLogging =
  | { readonly kind: 'uniform'; readonly items: number }
  | { readonly kind: 'list'; readonly slates: MatrixLike; readonly probabilities: VectorLike }

/**
 * A logging policy, read: the number of items `m` and slots `l`, the probability `prob` of a slate, and the pairwise
 * marginals `gamma`, $\Gammamat$ as a row-major $lm \times lm$ array.
 */
type Logging = { m: number; l: number; prob: (s: ArrayLike<number>) => number; gamma: F64 }

/**
 * A slate as a string key, its items joined by commas, for comparing and looking up slates.
 *
 * @param s The items of the slate, slot by slot.
 * @returns The key, e.g. `0,1`.
 */
const key = (s: ArrayLike<number>) => Array.from(s).join(',')

/**
 * A slate log's arrays, checked: one reward per slate and at least one round. Throws `DomainError` otherwise.
 *
 * @param log The logged slates and rewards.
 * @param where The caller's name, for error messages.
 * @returns The slates `s` (row-major $n \times l$), rewards `r`, rounds `n` and slots `l`.
 */
function readSlates(log: SlateLog, where: string): { s: F64; r: F64; n: number; l: number } {
  const { data, m: n, n: l } = dense.toMatrixF64(log.slates, where)
  const r = dense.toF64(log.rewards, where)
  if (r.length !== n) throw new DomainError(where, `${where}: ${n} slates and ${r.length} rewards`)
  if (n === 0) throw new DomainError(where, `${where}: the log is empty`)
  return { s: data, r, n, l }
}

/**
 * The indicator $\ones_s$ of a slate's (slot, item) pairs, of length $lm$: entry $jm + a$ is 1 when slot $j$ holds
 * item $a$. An item outside $0, \dots, m - 1$ would land in the next slot's block, so it throws `DomainError`.
 *
 * @param s The items of the slate, slot by slot ($l$ values).
 * @param m The number of items.
 * @param where The caller's name, for error messages.
 * @returns A new array of $lm$ zeros and ones.
 */
function indicator(s: ArrayLike<number>, m: number, where = 'slate'): F64 {
  const v = new Float64Array(s.length * m)
  for (let j = 0; j < s.length; j++) {
    if (!(Number.isInteger(s[j]) && s[j] >= 0 && s[j] < m))
      throw new DomainError(where, `${where}: item ${s[j]} in slot ${j} is not one of the logging policy's ${m} items`)
    v[j * m + s[j]] = 1
  }
  return v
}

/**
 * A logging policy's slate probabilities and pairwise marginals $\Gammamat$. For a uniform policy,
 * $\Pr(\text{slot } j = a) = 1/m$ and $\Pr(\text{slot } j = a, \text{slot } k = b) = 1/(m(m - 1))$ for $j \ne k$,
 * $a \ne b$. For a list, $\Gammamat$ sums each listed slate's probability over its pairs, and $m$ is the largest listed
 * item plus 1. Throws `DomainError` for too few uniform items, or a list whose shapes disagree with the log.
 *
 * @param logging The logging policy.
 * @param l The number of slots in the log.
 * @param where The caller's name, for error messages.
 * @returns The read policy.
 */
function readLogging(logging: SlateLogging, l: number, where: string): Logging {
  if (logging.kind === 'uniform') {
    const m = logging.items
    if (!(Number.isInteger(m) && m >= l)) throw new DomainError(where, `${where}: need at least ${l} items, got ${m}`)
    // Uniform over ordered slates of distinct items: P(slot j = a) = 1/m, P(j = a, k = b) = 1/(m(m − 1)) for j ≠ k,
    // a ≠ b, and 0 for the same item in two slots or two items in one slot.
    const d = l * m
    const gamma = new Float64Array(d * d)
    for (let j = 0; j < l; j++)
      for (let a = 0; a < m; a++)
        for (let k = 0; k < l; k++)
          for (let b = 0; b < m; b++) {
            const v = j === k ? (a === b ? 1 / m : 0) : a === b ? 0 : 1 / (m * (m - 1))
            gamma[(j * m + a) * d + k * m + b] = v
          }
    let count = 1
    for (let j = 0; j < l; j++) count *= m - j
    return {
      m,
      l,
      gamma,
      prob: (s) => (new Set(Array.from(s)).size === s.length ? 1 / count : 0),
    }
  }
  const { data, m: k, n: width } = dense.toMatrixF64(logging.slates, where)
  const p = dense.toF64(logging.probabilities, where)
  if (width !== l) throw new DomainError(where, `${where}: logging slates have ${width} slots, the log has ${l}`)
  if (p.length !== k) throw new DomainError(where, `${where}: ${k} slates and ${p.length} probabilities`)
  let m = 0
  for (const v of data) m = Math.max(m, v + 1)
  const d = l * m
  const gamma = new Float64Array(d * d)
  const table = new Map<string, number>()
  for (let c = 0; c < k; c++) {
    const s = data.subarray(c * l, (c + 1) * l)
    table.set(key(s), (table.get(key(s)) ?? 0) + p[c])
    const v = indicator(s, m)
    for (let x = 0; x < d; x++) if (v[x]) for (let y = 0; y < d; y++) if (v[y]) gamma[x * d + y] += p[c]
  }
  return { m, l, gamma, prob: (s) => table.get(key(s)) ?? 0 }
}

/**
 * The pseudo-inverse (PI) estimator of a deterministic target slate policy (Swaminathan et al., 2017):
 * $w_i = \thetavec_i^\top \Gammamat^{+} \ones_{s_i}$ and $\hat V = \frac1n \sum_i w_i r_i$, with $\Gammamat^{+}$ the
 * pseudo-inverse of the logging policy's pairwise marginals. Unbiased when the reward is additive over (slot, item)
 * pairs and the logging policy covers every pair the target uses. Weights may be negative: a logged slate that shares
 * no pair with the target still informs the estimate of the pairs it does share with other slates (and the effective
 * sample size is then NaN). Throws `DomainError` for a target of the wrong shape or an item the logging policy does not
 * have.
 *
 * @param log The logged slates and rewards.
 * @param target The target's slate in each round, $n \times l$.
 * @param logging The logging policy, the same in every round.
 * @param options The interval's level.
 * @returns The estimate, with the weights $w_i$.
 *
 * @example Every logged slate shares in the estimate
 * // Three items in two slots, logged uniformly: each of the 6 ordered slates once. The reward adds a value per
 * // (slot, item) pair, so the target slate [0, 1] is worth 0.5 + 0.4 = 0.9.
 * const slot0 = [0.5, 0.2, 0.1]
 * const slot1 = [0.3, 0.4, 0]
 * const slates = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
 * const log = { slates, rewards: slates.map(([a, b]) => slot0[a] + slot1[b]) }
 * const target = slates.map(() => [0, 1])
 * const pi = slatePseudoInverse(log, target, { kind: 'uniform', items: 3 })
 * const whole = slateIps(log, target, { kind: 'uniform', items: 3 })
 * print('PI:', pi.value, 'weights', pi.weights)
 * print('slate IPS:', whole.value, 'weights', whole.weights)
 */
export function slatePseudoInverse(
  log: SlateLog,
  target: MatrixLike,
  logging: SlateLogging,
  options: EstimateOptions = {},
): OffPolicyEstimate {
  const where = 'slatePseudoInverse'
  const { s, r, n, l } = readSlates(log, where)
  const { data: t, m: rows, n: width } = dense.toMatrixF64(target, where)
  if (rows !== n || width !== l) throw new DomainError(where, `${where}: the target is ${rows}×${width}, not ${n}×${l}`)
  const lg = readLogging(logging, l, where)
  const d = l * lg.m
  const G = dense.data(pinv(dense.mat(lg.gamma, d, d)))
  const weights = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const theta = indicator(t.subarray(i * l, (i + 1) * l), lg.m, where)
    const one = indicator(s.subarray(i * l, (i + 1) * l), lg.m, where)
    weights[i] = dense.dot(theta, dense.matVec(G, one, d, d))
  }
  const terms = weights.map((w, i) => w * r[i])
  return summarise('slate PI', terms, weights, options.level ?? 0.95)
}

/**
 * Slate-level IPS: $w_i = \indicator\{s_i = t_i\}/\mu(s_i)$, with $\mu(s_i)$ the probability of the whole logged slate
 * $s_i$ and $t_i$ the target's, and $\hat V = \frac1n \sum_i w_i r_i$. Unbiased for any reward, but the match
 * $s_i = t_i$ is rare when $\mu$ spreads over many slates, so the estimate rests on few rounds. Throws `DomainError`
 * for a target of the wrong shape, or a logged slate the logging policy gives probability 0.
 *
 * @param log The logged slates and rewards.
 * @param target The target's slate in each round, $n \times l$.
 * @param logging The logging policy, the same in every round.
 * @param options The interval's level.
 * @returns The estimate, with the weights $w_i$.
 *
 * @example Only the rounds that logged the target's slate count
 * // The logger shows [0, 1] with probability 0.75 and [1, 0] with 0.25; the target always shows [0, 1].
 * const logging = { kind: 'list', slates: [[0, 1], [1, 0]], probabilities: [0.75, 0.25] }
 * const log = { slates: [[0, 1], [1, 0], [0, 1], [1, 0]], rewards: [1, 0.5, 0, 0.5] }
 * const est = slateIps(log, [[0, 1], [0, 1], [0, 1], [0, 1]], logging)
 * print('weights:', est.weights)
 * print('value (1 / 0.75) / 4:', est.value)
 */
export function slateIps(
  log: SlateLog,
  target: MatrixLike,
  logging: SlateLogging,
  options: EstimateOptions = {},
): OffPolicyEstimate {
  const where = 'slateIps'
  const { s, r, n, l } = readSlates(log, where)
  const { data: t, m: rows, n: width } = dense.toMatrixF64(target, where)
  if (rows !== n || width !== l) throw new DomainError(where, `${where}: the target is ${rows}×${width}, not ${n}×${l}`)
  const lg = readLogging(logging, l, where)
  const weights = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const si = s.subarray(i * l, (i + 1) * l)
    const match = key(si) === key(t.subarray(i * l, (i + 1) * l))
    const p = lg.prob(si)
    if (!(p > 0))
      throw new DomainError(where, `${where}: logged slate ${key(si)} has probability 0 under the logging policy`)
    weights[i] = match ? 1 / p : 0
  }
  return summarise(
    'slate IPS',
    weights.map((w, i) => w * r[i]),
    weights,
    options.level ?? 0.95,
  )
}
