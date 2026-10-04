/**
 * The n-gram language model with interpolated Kneser–Ney smoothing (Kneser and Ney, 1995; in the interpolated and
 * modified forms of Chen and Goodman, 1999). The probability of token w after the context h (the last n − 1 tokens) is
 *
 *   P(w | h) = max(c(h w) − D, 0) / c(h ·) + γ(h) · P(w | h′),
 *
 * where h′ drops the oldest token of h and γ(h) = Σ_{w′ : c(h w′) > 0} min(D, c(h w′)) / c(h ·) is the mass the
 * discount freed, so every distribution sums to one. Lower orders use continuation counts N₁₊(• h′ w), the number of
 * distinct tokens seen before h′ w, instead of raw counts: a token that occurs often but only after one context gets a
 * small lower-order probability. The unigram level interpolates with the uniform distribution, so every token has
 * positive probability. Discounts are D = n₁/(n₁ + 2n₂) per order from the counts of counts (Ney, Essen and Kneser,
 * 1994), or with `modified` three discounts D₁, D₂, D₃₊ by count (Chen and Goodman, 1999).
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Estimator, Scores } from 'aifn-compute/learning/estimators'
import type { LogitsFn } from 'aifn-compute/nn/decoding'
import type { Vocabulary } from 'aifn-compute/text/vocabulary'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A corpus as token ids over a vocabulary. */
export type TokenCorpus = { readonly ids: readonly number[]; readonly vocabulary: Vocabulary }

/** Counts by context (ids joined by commas) and next token. */
type Table = Map<string, Map<number, number>>

/** The discount of a count: one value, or D₁, D₂, D₃₊. */
type Discount = readonly [number, number, number]

const keyOf = (ids: readonly number[]) => ids.join(',')

function add(table: Table, context: string, w: number, by = 1) {
  let row = table.get(context)
  if (!row) table.set(context, (row = new Map()))
  row.set(w, (row.get(w) ?? 0) + by)
}

/** Discounts from the counts of counts of a table (Chen and Goodman, 1999, eq. 26; Ney et al., 1994). */
function discountsOf(table: Table, modified: boolean): Discount {
  const n = [0, 0, 0, 0, 0]
  for (const row of table.values()) for (const c of row.values()) if (c <= 4) n[c]++
  const [, n1, n2, n3, n4] = n
  if (n1 === 0 || n2 === 0) return [0.5, 0.5, 0.5]
  const Y = n1 / (n1 + 2 * n2)
  if (!modified) return [Y, Y, Y]
  const clamp = (d: number, c: number) => (Number.isFinite(d) && d > 0 ? Math.min(d, c) : Y)
  return [
    clamp(1 - 2 * Y * (n2 / n1), 1),
    clamp(2 - 3 * Y * (n3 / n2), 2),
    clamp(n3 > 0 ? 3 - 4 * Y * (n4 / n3) : Y, 3),
  ]
}

const discountFor = (d: Discount, c: number) => (c <= 0 ? 0 : c === 1 ? d[0] : c === 2 ? d[1] : d[2])

/** A fitted Kneser–Ney language model. */
export type KneserNeyModel = Scores<Tensor> & {
  readonly kind: 'model'
  readonly name: 'kneser-ney'
  readonly order: Size
  readonly vocabularySize: Size
  readonly vocabulary: Vocabulary
  /** Discounts (D₁, D₂, D₃₊; equal unless modified) of the raw counts and of the continuation counts, by order 1…n. */
  readonly discounts: { readonly counts: readonly Discount[]; readonly continuation: readonly Discount[] }
  /** P(w | context) for every w, [V] (uses the last n − 1 tokens of the context, fewer at the start). */
  distribution(context: readonly number[]): number[]
  /** log P(· | prefix) as a logits function for `aifn-compute/nn/decoding`. */
  readonly logits: LogitsFn
  /** Per-token perplexity exp(−(1/N) Σ log P(w_t | w_{<t})) of a sequence (each token given those before it). */
  perplexity(ids: readonly number[]): number
  /** Next-token log-probabilities for contexts [N, k] of ids: [N, V]. */
  score(contexts: Tensor): Tensor
}

/** Options of `kneserNey`. */
export type KneserNeyOptions = {
  /** The n of the n-grams (default 3). */
  order?: Size
  /** Modified Kneser–Ney with three discounts (default false). */
  modified?: boolean
}

/** An interpolated Kneser–Ney n-gram language model, fitted by counting a token corpus. */
export function kneserNey(options: KneserNeyOptions = {}): Estimator<TokenCorpus, KneserNeyModel> {
  const { order = 3, modified = false } = options
  if (!(order >= 1)) throw new DomainError('kneserNey', `kneserNey: order ${order} must be at least 1`)
  return {
    name: 'kneser-ney',
    params: { order, modified },
    fit({ ids, vocabulary }) {
      const V = vocabulary.tokens.length
      // counts[m]: c(h w) with |h| = m − 1; continuation[m]: N₁₊(• h w) with |h| = m − 1, for m < n.
      const counts: Table[] = Array.from({ length: order + 1 }, () => new Map())
      for (let t = 0; t < ids.length; t++)
        for (let m = 1; m <= order && t - m + 1 >= 0; m++) add(counts[m], keyOf(ids.slice(t - m + 1, t)), ids[t])
      const continuation: Table[] = Array.from({ length: order + 1 }, () => new Map())
      for (let m = 1; m < order; m++)
        for (const [context, row] of counts[m + 1]) {
          const shorter = context.split(',').slice(1).join(',')
          for (const w of row.keys()) add(continuation[m], shorter, w)
        }
      const rawD = counts.map((t) => discountsOf(t, modified))
      const contD = continuation.map((t) => discountsOf(t, modified))
      const totals = (table: Table) =>
        new Map([...table].map(([k, row]) => [k, [...row.values()].reduce((a, b) => a + b, 0)]))
      const rawTotals = counts.map(totals)
      const contTotals = continuation.map(totals)

      /** P_m(· | h) for every token, with h the last m − 1 tokens of `context`; raw counts at the top level. */
      const level = (context: readonly number[], m: number, top: boolean): Float64Array => {
        if (m === 0) return new Float64Array(V).fill(1 / V)
        const lower = level(context, m - 1, false)
        const key = keyOf(context.slice(context.length - (m - 1)))
        const table = top ? counts[m] : continuation[m]
        const row = table.get(key)
        const total = (top ? rawTotals[m] : contTotals[m]).get(key) ?? 0
        if (!row || total === 0) return lower
        const D = top ? rawD[m] : contD[m]
        let freed = 0
        for (const c of row.values()) freed += Math.min(discountFor(D, c), c)
        const out = Float64Array.from(lower, (p) => (freed / total) * p)
        for (const [w, c] of row) out[w] += Math.max(c - discountFor(D, c), 0) / total
        return out
      }
      const distribution = (context: readonly number[]) => {
        const m = Math.min(order, context.length + 1)
        return Array.from(level(context, m, true))
      }
      const logits: LogitsFn = (prefix) => fromData(Float64Array.from(distribution(prefix), Math.log), [V])
      return {
        kind: 'model',
        name: 'kneser-ney',
        order,
        vocabularySize: V,
        vocabulary,
        discounts: { counts: rawD.slice(1), continuation: contD.slice(1, order) },
        distribution,
        logits,
        perplexity: (seq) => {
          let nll = 0
          for (let t = 0; t < seq.length; t++) nll -= Math.log(distribution(seq.slice(0, t))[seq[t]])
          return Math.exp(nll / Math.max(1, seq.length))
        },
        score: (contexts: Tensor) => {
          const [n, k] = contexts.shape.length === 1 ? [contexts.shape[0], 0] : contexts.shape
          const v = toFlat(contexts)
          const out = new Float64Array(n * V)
          for (let i = 0; i < n; i++) {
            const p = distribution(v.slice(i * k, (i + 1) * k))
            for (let w = 0; w < V; w++) out[i * V + w] = Math.log(p[w])
          }
          return fromData(out, [n, V])
        },
      }
    },
  }
}
