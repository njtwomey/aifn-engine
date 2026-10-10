/**
 * The n-gram language model with interpolated Kneser–Ney smoothing (Kneser and Ney, 1995; in the interpolated and
 * modified forms of Chen and Goodman, 1999). The probability of token $w$ after the context $h$ (the last $n - 1$
 * tokens) is
 *
 * $$P(w \mid h) = \frac{\max(c(h w) - D, 0)}{c(h\, \cdot)} + \gamma(h)\, P(w \mid h'),$$
 *
 * where $h'$ drops the oldest token of $h$ and $\gamma(h) = \sum_{w' : c(h w') > 0} \min(D, c(h w')) / c(h\, \cdot)$
 * is the mass the discount freed, so every distribution sums to one. Lower orders use continuation counts
 * $N_{1+}(\bullet\, h' w)$, the number of distinct tokens seen before $h' w$, instead of raw counts: a token that
 * occurs often but only after one context gets a small lower-order probability. The unigram level interpolates with
 * the uniform distribution $1 / V$, so every token has positive probability, and a context never seen falls back to
 * the next order down entirely. Near the start of a sequence, where fewer than $n - 1$ tokens precede $w$, the
 * highest order available uses raw counts.
 *
 * Discounts are $D = n_1 / (n_1 + 2 n_2)$ per order from the counts of counts, $n_k$ the number of n-grams seen
 * exactly $k$ times (Ney, Essen and Kneser, 1994), or with `modified` three discounts $D_1$, $D_2$, $D_{3+}$ by count
 * (Chen and Goodman, 1999). With no n-gram seen once or none seen twice, every discount of that order is $0.5$.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Estimator, Scores } from 'aifn-compute/learning/estimators'
import type { LogitsFn } from 'aifn-compute/nn/decoding'
import type { Vocabulary } from 'aifn-compute/text/vocabulary'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A corpus as token ids over a vocabulary. */
export type TokenCorpus = { readonly ids: readonly number[]; readonly vocabulary: Vocabulary }

/** Counts by context (its ids joined by commas, the empty string for none) and next token. */
type Table = Map<string, Map<number, number>>

/**
 * The discounts $D_1$, $D_2$, $D_{3+}$ of a count of one, two, and three or more; all three equal unless modified.
 */
type Discount = readonly [number, number, number]

/**
 * The key of a context in a `Table`.
 *
 * @param ids The context's token ids, oldest first.
 * @returns The ids joined by commas.
 */
const keyOf = (ids: readonly number[]) => ids.join(',')

/**
 * Add to the count of token `w` after a context, creating the context's row if it has none. Modifies `table`.
 *
 * @param table The counts to add to.
 * @param context The context's key, from `keyOf`.
 * @param w The next token's id.
 * @param by The amount to add.
 */
function add(table: Table, context: string, w: number, by = 1) {
  let row = table.get(context)
  if (!row) table.set(context, (row = new Map()))
  row.set(w, (row.get(w) ?? 0) + by)
}

/**
 * Discounts from the counts of counts of a table (Chen and Goodman, 1999, eq. 26; Ney et al., 1994):
 * $Y = n_1 / (n_1 + 2 n_2)$, and with `modified` $D_k = k - (k + 1) Y n_{k+1} / n_k$ for $k = 1, 2, 3$. A modified
 * discount that is not positive, or not finite, is replaced by $Y$; each is capped at its count. Both fall back to
 * $0.5$ when $n_1$ or $n_2$ is 0.
 *
 * @param table The counts of one order, raw or continuation.
 * @param modified Whether to give three discounts by count (modified Kneser–Ney) rather than one.
 * @returns $D_1$, $D_2$, $D_{3+}$.
 */
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

/**
 * The discount that applies to a count.
 *
 * @param d The order's discounts.
 * @param c The count of the n-gram.
 * @returns 0 for a count of 0 or less, else $D_1$, $D_2$ or $D_{3+}$ by count.
 */
const discountFor = (d: Discount, c: number) => (c <= 0 ? 0 : c === 1 ? d[0] : c === 2 ? d[1] : d[2])

/** A fitted Kneser–Ney language model. */
export type KneserNeyModel = Scores<Tensor> & {
  /** The estimator kind. */
  readonly kind: 'model'
  /** The estimator's name. */
  readonly name: 'kneser-ney'
  /** The $n$ of the n-grams. */
  readonly order: Size
  /** The vocabulary size $V$. */
  readonly vocabularySize: Size
  /** The corpus's vocabulary. */
  readonly vocabulary: Vocabulary
  /**
   * Discounts ($D_1$, $D_2$, $D_{3+}$; equal unless modified) of the raw counts by order $1, \dots, n$, and of the
   * continuation counts by order $1, \dots, n - 1$.
   */
  readonly discounts: { readonly counts: readonly Discount[]; readonly continuation: readonly Discount[] }
  /**
   * $P(w \mid \text{context})$ for every $w$, $V$ values (uses the last $n - 1$ tokens of the context, fewer at the
   * start).
   */
  distribution(context: readonly number[]): number[]
  /** $\log P(\cdot \mid \text{prefix})$ as a logits function for `aifn-compute/nn/decoding`. */
  readonly logits: LogitsFn
  /**
   * Per-token perplexity $\exp\bigl(-\frac{1}{N} \sum_t \log P(w_t \mid w_{<t})\bigr)$ of a sequence of $N$ tokens
   * (each token given those before it).
   */
  perplexity(ids: readonly number[]): number
  /**
   * Next-token log-probabilities for contexts `[N, k]` of ids: `[N, V]`. A one-dimensional tensor of $N$ ids is read as
   * $N$ one-token contexts, `[N, 1]`; any other rank throws `ShapeError`.
   */
  score(contexts: Tensor): Tensor
}

/** Options of `kneserNey`. */
export type KneserNeyOptions = {
  /** The n of the n-grams (default 3). */
  order?: Size
  /** Modified Kneser–Ney with three discounts (default false). */
  modified?: boolean
}

/**
 * An interpolated Kneser–Ney n-gram language model, fitted by counting a token corpus. Throws `DomainError` for an
 * order below 1.
 *
 * @param options The order $n$ and whether to use modified Kneser–Ney.
 * @returns The estimator; `fit` counts the corpus and returns the model.
 *
 * @example A trigram model of a repeating string
 * const corpus = charCorpus('abcabcabcabcabcabcabcabd')
 * const lm = kneserNey({ order: 3 }).fit(corpus)
 * print('alphabet:', corpus.vocabulary.tokens)
 * print('P(next | "ab"):', lm.distribution(encodeChars(corpus, 'ab')))
 * print('perplexity of "abcabc":', lm.perplexity(encodeChars(corpus, 'abcabc')))
 * print('perplexity of "acbacb":', lm.perplexity(encodeChars(corpus, 'acbacb')))
 *
 * @example Plain and modified discounts of the nursery rhymes, by order
 * const corpus = charCorpus()
 * print('plain:', kneserNey().fit(corpus).discounts.counts)
 * print('modified:', kneserNey({ modified: true }).fit(corpus).discounts.counts)
 */
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
          const r = contexts.shape.length
          if (r !== 1 && r !== 2)
            throw new ShapeError('kneserNey', `kneserNey: score expects contexts [N, k] or [N], given rank ${r}`, [
              contexts.shape,
            ])
          const [n, k] = r === 1 ? [contexts.shape[0], 1] : contexts.shape
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
