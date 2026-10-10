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
 * from, `specials` the special tokens (ids $0, \dots, S-1$) and `unknown` the id that out-of-vocabulary tokens map to.
 */
export interface Vocabulary {
  /** The tag `'vocabulary'`. */
  readonly kind: 'vocabulary'
  /** The token of each id: the $S$ special tokens first, then the corpus tokens ($V$ in all). */
  readonly tokens: readonly string[]
  /** The corpus count of each id (float64 [V]); 0 for special tokens and for a vocabulary over a fixed list. */
  readonly counts: Tensor
  /** The special tokens, which have ids $0, \dots, S-1$ in this order. */
  readonly specials: readonly string[]
  /** The id out-of-vocabulary tokens map to, or $-1$ when there is none. */
  readonly unknown: number
}

/**
 * Documents as a list of token lists: one token list becomes a single document.
 *
 * @param docs Token lists, or one token list (recognised by its first entry being a string).
 * @returns The documents; an empty input gives no documents.
 */
const documentsOf = (docs: TokenDocuments): readonly (readonly string[])[] =>
  docs.length > 0 && typeof docs[0] === 'string'
    ? [docs as readonly string[]]
    : (docs as readonly (readonly string[])[])

/**
 * Count tokens over documents: the distinct tokens in order of first appearance and their counts (float64 [n]), and how
 * many documents contain each (`documentCounts`, float64 [n]).
 *
 * @param documents The tokenised documents, or a single token list (one document).
 * @returns `tokens`, the $n$ distinct tokens in order of first appearance; `counts`, how often each occurs in all;
 *   `documentCounts`, in how many documents each occurs.
 *
 * @example Term and document counts
 * const c = tokenCounts([['the', 'cat', 'sat'], ['the', 'cat', 'and', 'the', 'dog']])
 * print('tokens   ', c.tokens)
 * print('counts   ', c.counts)
 * print('documents', c.documentCounts)
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
  /** Special tokens, given ids $0, \dots, S-1$ in this order (default `['<unk>']`). */
  specials?: readonly string[]
  /** The special token unknown tokens map to (default `'<unk>'` when listed in `specials`; `null` for none). */
  unknown?: string | null
  /**
   * Id order of the corpus tokens: `frequency` (default; descending count, ties alphabetical, as torchtext),
   * `alphabetical` (code-unit order, the code-point order of scikit-learn's `CountVectorizer` outside the astral
   * planes) or `appearance` (first occurrence).
   */
  order?: 'frequency' | 'alphabetical' | 'appearance'
}

/**
 * Compare two strings by UTF-16 code unit, as `<` does (the same as code-point order unless a surrogate pair is
 * involved), not by locale.
 *
 * @param a The first string.
 * @param b The second string.
 * @returns $-1$, 0 or 1 as `a` sorts before, with or after `b`.
 */
const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

/**
 * Build a vocabulary from tokenised documents, as torchtext's `build_vocab_from_iterator`. Tokens counted fewer than
 * `minCount` times are dropped; when more than `maxSize` remain, the most frequent are kept (ties broken
 * alphabetically) whatever the id order. A corpus token equal to a special is not given a second id. Throws
 * `DomainError` when `unknown` is not one of `specials`.
 *
 * @param documents The tokenised documents, or a single token list.
 * @param options The size limits, special tokens and id order; see {@link VocabularyOptions}.
 * @returns The vocabulary, specials first, with corpus counts.
 *
 * @example Specials first, then by frequency
 * const docs = [['the', 'cat', 'sat'], ['the', 'dog', 'sat'], ['a', 'cat']]
 * const v = buildVocabulary(docs, { specials: ['<unk>', '<pad>'] })
 * print('tokens', v.tokens)
 * print('counts', v.counts)
 * print('min count 2', buildVocabulary(docs, { minCount: 2 }).tokens)
 * print('alphabetical', buildVocabulary(docs, { order: 'alphabetical', specials: [] }).tokens)
 */
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

/**
 * A vocabulary over a fixed token list (ids in list order after the specials; counts 0). A token that repeats a
 * special is skipped; any other repeat throws `DomainError`.
 *
 * @param tokens The tokens, in id order.
 * @param options The special tokens and the unknown token.
 * @param options.specials The special tokens, given ids $0, \dots, S-1$ (default none).
 * @param options.unknown The token out-of-vocabulary tokens map to, or `null` for none (the default). One that is in
 *   neither list gives no unknown id.
 * @returns The vocabulary.
 *
 * @example A label set with an unknown token
 * const v = vocabularyOf(['NOUN', 'VERB', 'ADJ'], { specials: ['<unk>'], unknown: '<unk>' })
 * print('tokens', v.tokens)
 * print('unknown id', v.unknown)
 */
export function vocabularyOf(
  tokens: readonly string[],
  options: { specials?: readonly string[]; unknown?: string | null } = {},
): Vocabulary {
  const { specials = [] } = options
  if (new Set(specials).size !== specials.length)
    throw new DomainError('vocabularyOf', 'vocabularyOf: specials must be distinct')
  const unknownToken = options.unknown ?? null
  const all = [...specials, ...tokens.filter((t) => !specials.includes(t))]
  if (new Set(all).size !== all.length) throw new DomainError('vocabularyOf', 'vocabularyOf: tokens must be distinct')
  if (unknownToken !== null && !all.includes(unknownToken))
    throw new DomainError('vocabularyOf', `vocabularyOf: unknown token '${unknownToken}' is not in the vocabulary`)
  return {
    kind: 'vocabulary',
    tokens: all,
    counts: fromData(new Float64Array(all.length)),
    specials: [...specials],
    unknown: unknownToken === null ? -1 : all.indexOf(unknownToken),
  }
}

const lookups = new WeakMap<Vocabulary, Map<string, number>>()

/**
 * The token-to-id map of a vocabulary, built on first use and cached per vocabulary object.
 *
 * @param v The vocabulary; it must not be modified after its first lookup.
 * @returns A map from each token to its id.
 */
function lookup(v: Vocabulary): Map<string, number> {
  let m = lookups.get(v)
  if (!m) {
    m = new Map(v.tokens.map((t, k) => [t, k]))
    lookups.set(v, m)
  }
  return m
}

/**
 * The id of a token, or $-1$ when it is not in the vocabulary. The unknown id is not substituted.
 *
 * @param vocabulary The vocabulary to look in.
 * @param token The token, matched exactly.
 * @returns Its id, or $-1$.
 *
 * @example Known and unknown tokens
 * const v = buildVocabulary([['the', 'cat', 'sat']])
 * print('cat', tokenId(v, 'cat'), ' dog', tokenId(v, 'dog'), ' <unk>', tokenId(v, '<unk>'))
 */
export function tokenId(vocabulary: Vocabulary, token: string): number {
  return lookup(vocabulary).get(token) ?? -1
}

/**
 * Token ids (int32 [n]). An out-of-vocabulary token maps to the unknown id; with no unknown token it is an error
 * (`DomainError`), or dropped with `onUnknown: 'skip'`.
 *
 * @param vocabulary The vocabulary to encode with.
 * @param tokens The tokens, matched exactly (no case folding).
 * @param options What to do with an out-of-vocabulary token when the vocabulary has no unknown id.
 * @param options.onUnknown `'error'` (the default) throws, `'skip'` drops the token. Ignored when the vocabulary has
 *   an unknown id.
 * @returns One id per kept token, in order.
 *
 * @example Unknown tokens map to the unknown id, or are skipped
 * const v = buildVocabulary([['the', 'cat', 'sat']])
 * print('ids', encodeTokens(v, ['the', 'dog', 'sat']))
 * const bare = buildVocabulary([['the', 'cat', 'sat']], { specials: [] })
 * print('skipped', encodeTokens(bare, ['the', 'dog', 'sat'], { onUnknown: 'skip' }))
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

/**
 * Token strings from ids. Throws `DomainError` for an id with no token.
 *
 * @param vocabulary The vocabulary the ids index.
 * @param ids The ids, as a tensor (read flat) or a list of numbers.
 * @returns The token of each id, in order.
 *
 * @example A round trip
 * const v = buildVocabulary([['to', 'be', 'or', 'not', 'to', 'be']])
 * const ids = encodeTokens(v, ['not', 'to', 'be'])
 * print('ids', ids)
 * print('tokens', decodeTokens(v, ids))
 */
export function decodeTokens(vocabulary: Vocabulary, ids: Tensor | readonly number[]): string[] {
  const list = Array.isArray(ids) ? (ids as readonly number[]) : toFlat(ids as Tensor)
  return Array.from(list, (id) => {
    const t = vocabulary.tokens[id]
    if (t === undefined) throw new DomainError('decodeTokens', `decodeTokens: id ${id} is out of range`)
    return t
  })
}
