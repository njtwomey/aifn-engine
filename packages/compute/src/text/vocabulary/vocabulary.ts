/**
 * Vocabularies: the map between token strings and integer ids that every count matrix, embedding table and language
 * model indexes by. Special tokens (unknown, padding, boundaries) come first; corpus tokens follow, filtered by a
 * minimum count and capped at a maximum size.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/** Documents as token lists, or one token list. */
export type TokenDocuments = readonly (readonly string[])[] | readonly string[]

/**
 * A vocabulary as plain data: `tokens[id]` is the token with that id, `counts[id]` its count in the corpus it was built
 * from (float64 [V]; 0 for special tokens and fixed lists), `specials` the special tokens (ids 0 … S − 1) and `unknown`
 * the id that out-of-vocabulary tokens map to, or −1 when there is none.
 */
export interface Vocabulary {
  readonly kind: 'vocabulary'
  readonly tokens: readonly string[]
  readonly counts: Tensor
  readonly specials: readonly string[]
  readonly unknown: number
}

const documentsOf = (docs: TokenDocuments): readonly (readonly string[])[] =>
  docs.length > 0 && typeof docs[0] === 'string'
    ? [docs as readonly string[]]
    : (docs as readonly (readonly string[])[])

/**
 * Count tokens over documents: the distinct tokens in order of first appearance and their counts (float64 [n]), and how
 * many documents contain each (`documentCounts`, float64 [n]).
 */
export function tokenCounts(documents: TokenDocuments): {
  tokens: string[]
  counts: Tensor
  documentCounts: Tensor
} {
  const index = new Map<string, number>()
  const counts: number[] = []
  const docCounts: number[] = []
  for (const doc of documentsOf(documents)) {
    const seen = new Set<number>()
    for (const t of doc) {
      let k = index.get(t)
      if (k === undefined) {
        k = counts.length
        index.set(t, k)
        counts.push(0)
        docCounts.push(0)
      }
      counts[k]++
      if (!seen.has(k)) {
        seen.add(k)
        docCounts[k]++
      }
    }
  }
  return {
    tokens: [...index.keys()],
    counts: fromData(Float64Array.from(counts)),
    documentCounts: fromData(Float64Array.from(docCounts)),
  }
}

/** Options of {@link buildVocabulary}. */
export interface VocabularyOptions {
  /** Keep tokens seen at least this many times (default 1). */
  minCount?: number
  /** Keep at most this many corpus tokens, the most frequent (default no limit; specials are not counted). */
  maxSize?: number
  /** Special tokens, given ids 0 … S − 1 in this order (default `['<unk>']`). */
  specials?: readonly string[]
  /** The special token unknown tokens map to (default `'<unk>'` when listed in `specials`; `null` for none). */
  unknown?: string | null
  /**
   * Id order of the corpus tokens: `frequency` (default; descending count, ties alphabetical, as torchtext),
   * `alphabetical` (code-point order, as scikit-learn's `CountVectorizer`) or `appearance` (first occurrence).
   */
  order?: 'frequency' | 'alphabetical' | 'appearance'
}

const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

/** Build a vocabulary from tokenised documents. */
export function buildVocabulary(documents: TokenDocuments, options: VocabularyOptions = {}): Vocabulary {
  const { minCount = 1, maxSize = Infinity, specials = ['<unk>'], order = 'frequency' } = options
  const unknownToken = options.unknown === undefined ? (specials.includes('<unk>') ? '<unk>' : null) : options.unknown
  if (unknownToken !== null && !specials.includes(unknownToken))
    throw new DomainError('buildVocabulary', `buildVocabulary: the unknown token '${unknownToken}' is not a special`)
  const { tokens, counts } = tokenCounts(documents)
  const c = toFlat(counts)
  const special = new Set(specials)
  let kept = tokens.map((t, k) => ({ t, n: c[k] })).filter((e) => e.n >= minCount && !special.has(e.t))
  // The most frequent tokens survive the size cap whatever the id order.
  if (kept.length > maxSize) {
    kept = [...kept].sort((a, b) => b.n - a.n || byCodePoint(a.t, b.t)).slice(0, maxSize)
  }
  if (order === 'frequency') kept.sort((a, b) => b.n - a.n || byCodePoint(a.t, b.t))
  else if (order === 'alphabetical') kept.sort((a, b) => byCodePoint(a.t, b.t))
  else {
    const first = new Map(tokens.map((t, k) => [t, k]))
    kept.sort((a, b) => first.get(a.t)! - first.get(b.t)!)
  }
  return {
    kind: 'vocabulary',
    tokens: [...specials, ...kept.map((e) => e.t)],
    counts: fromData(Float64Array.from([...specials.map(() => 0), ...kept.map((e) => e.n)])),
    specials: [...specials],
    unknown: unknownToken === null ? -1 : specials.indexOf(unknownToken),
  }
}

/** A vocabulary over a fixed token list (ids in list order after the specials; counts 0). */
export function vocabularyOf(
  tokens: readonly string[],
  options: { specials?: readonly string[]; unknown?: string | null } = {},
): Vocabulary {
  const { specials = [] } = options
  const unknownToken = options.unknown ?? null
  const all = [...specials, ...tokens.filter((t) => !specials.includes(t))]
  if (new Set(all).size !== all.length) throw new DomainError('vocabularyOf', 'vocabularyOf: tokens must be distinct')
  return {
    kind: 'vocabulary',
    tokens: all,
    counts: fromData(new Float64Array(all.length)),
    specials: [...specials],
    unknown: unknownToken === null ? -1 : all.indexOf(unknownToken),
  }
}

const lookups = new WeakMap<Vocabulary, Map<string, number>>()

function lookup(v: Vocabulary): Map<string, number> {
  let m = lookups.get(v)
  if (!m) {
    m = new Map(v.tokens.map((t, k) => [t, k]))
    lookups.set(v, m)
  }
  return m
}

/** The id of a token, or −1 when it is not in the vocabulary. */
export function tokenId(vocabulary: Vocabulary, token: string): number {
  return lookup(vocabulary).get(token) ?? -1
}

/**
 * Token ids (int32 [n]). An out-of-vocabulary token maps to the unknown id; with no unknown token it is an error, or
 * dropped with `onUnknown: 'skip'`.
 */
export function encodeTokens(
  vocabulary: Vocabulary,
  tokens: readonly string[],
  options: { onUnknown?: 'error' | 'skip' } = {},
): Tensor {
  const m = lookup(vocabulary)
  const out: number[] = []
  for (const t of tokens) {
    const id = m.get(t)
    if (id !== undefined) out.push(id)
    else if (vocabulary.unknown >= 0) out.push(vocabulary.unknown)
    else if (options.onUnknown === 'skip') continue
    else
      throw new DomainError(
        'encodeTokens',
        `encodeTokens: '${t}' is not in the vocabulary and there is no unknown token`,
      )
  }
  return fromData(Int32Array.from(out))
}

/** Token strings from ids (an int tensor or a list). */
export function decodeTokens(vocabulary: Vocabulary, ids: Tensor | readonly number[]): string[] {
  const list = Array.isArray(ids) ? (ids as readonly number[]) : toFlat(ids as Tensor)
  return Array.from(list, (id) => {
    const t = vocabulary.tokens[id]
    if (t === undefined) throw new DomainError('decodeTokens', `decodeTokens: id ${id} is out of range`)
    return t
  })
}
