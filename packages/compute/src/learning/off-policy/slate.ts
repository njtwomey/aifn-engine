/**
 * Off-policy evaluation of slates: a page of l slots filled from m items. Slate-level IPS reweights by the probability
 * of the whole logged slate, which is tiny when the logging policy randomises over m!/(m − l)! orderings. The
 * pseudo-inverse estimator (Swaminathan et al., 2017) assumes the reward is additive over (slot, item) pairs, and then
 * needs only the logging policy's pairwise marginals Γ = E_μ[1ₛ1ₛᵀ]: V̂ = (1/n) Σᵢ rᵢ θᵢᵀ Γ⁺ 1_{sᵢ}, where 1ₛ is the
 * indicator of the slate's (slot, item) pairs and θᵢ the target's expected indicator.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense } from 'aifn-compute/foundation/tensor'
import { pinv } from 'aifn-compute/numerics/linalg'
import { summarise, type EstimateOptions, type OffPolicyEstimate } from './estimators'

type F64 = dense.F64

/** Logged slates and their rewards. */
export interface SlateLog {
  /** The logged slate: the item in each slot [n, l]. */
  readonly slates: MatrixLike
  /** The reward of each logged slate [n]. */
  readonly rewards: VectorLike
}

/**
 * The logging policy over slates, the same for every context: uniform over the ordered slates of l distinct items from
 * m (`{ kind: 'uniform', items: m }`), or an explicit list of slates [k, l] with their probabilities [k].
 */
export type SlateLogging =
  | { readonly kind: 'uniform'; readonly items: number }
  | { readonly kind: 'list'; readonly slates: MatrixLike; readonly probabilities: VectorLike }

type Logging = { m: number; l: number; prob: (s: ArrayLike<number>) => number; gamma: F64 }

const key = (s: ArrayLike<number>) => Array.from(s).join(',')

function readSlates(log: SlateLog, where: string): { s: F64; r: F64; n: number; l: number } {
  const { data, m: n, n: l } = dense.toMatrixF64(log.slates, where)
  const r = dense.toF64(log.rewards, where)
  if (r.length !== n) throw new DomainError(where, `${where}: ${n} slates and ${r.length} rewards`)
  if (n === 0) throw new DomainError(where, `${where}: the log is empty`)
  return { s: data, r, n, l }
}

/**
 * The indicator 1ₛ of a slate's (slot, item) pairs, of length l·m. An item outside 0 … m − 1 would land in the next
 * slot's block, so it throws.
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
 * The pseudo-inverse (PI) estimator of a deterministic target slate policy (`target` [n, l]: the slate it would show
 * in each round). Unbiased when the reward is additive over (slot, item) pairs and the logging policy covers every
 * pair the target uses. Weights may be negative: a logged slate that shares no pair with the target still informs the
 * estimate of the pairs it does share with other slates.
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
 * Slate-level IPS: wᵢ = 1{sᵢ = tᵢ}/μ(sᵢ), the probability of the whole logged slate. Unbiased for any reward, but
 * the match 1{sᵢ = tᵢ} is rare when μ spreads over many slates, so the estimate rests on few rounds.
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
