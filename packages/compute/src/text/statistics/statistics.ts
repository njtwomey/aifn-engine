/**
 * Tokeniser statistics on a corpus: fertility (tokens per word, Rust et al.'s measure of how finely a tokeniser cuts a
 * language), compression (UTF-8 bytes and characters per token), the unknown-token rate and the share of words
 * encoded without one (coverage), and the share of the vocabulary a corpus uses. Words are runs of non-space
 * characters of the original text, so every tokeniser is measured against the same denominator.
 */

import { toFlat } from 'aifn-compute/foundation/tensor'
import { encodeText, encodingTokenisation, vocabularySize, type Tokeniser } from 'aifn-compute/text/pipeline'
import type { Tokenisation } from 'aifn-compute/text/tokenise'

/** Statistics of tokenised texts. A rate whose denominator is 0 is NaN. */
export interface TokenStatistics {
  /** The number of texts. */
  readonly texts: number
  /** Runs of non-space characters in the texts. */
  readonly words: number
  /** Code points in the texts. */
  readonly characters: number
  /** UTF-8 bytes in the texts. */
  readonly bytes: number
  /** Tokens in all the tokenisations. */
  readonly tokens: number
  /** Tokens per word. */
  readonly fertility: number
  /** UTF-8 bytes per token (compression; 1 for a byte tokeniser). */
  readonly bytesPerToken: number
  /** Code points per token. */
  readonly charactersPerToken: number
  /** Tokens counted as unknown. */
  readonly unknownTokens: number
  /** Unknown tokens per token. */
  readonly unknownRate: number
  /** The share of words that no unknown token overlaps. */
  readonly wordCoverage: number
  /** The vocabulary size (NaN when not known). */
  readonly vocabularySize: number
  /** Distinct tokens (or ids, or other keys) used. */
  readonly vocabularyUsed: number
  /** `vocabularyUsed / vocabularySize` (NaN when the size is not known). */
  readonly vocabularyUsage: number
}

/** Options of {@link tokenisationStatistics}. */
export interface TokenStatisticsOptions {
  /** Which tokens are unknown, by index into each tokenisation (default none). */
  isUnknown?: (text: number, token: number) => boolean
  /** The vocabulary size, for the usage share (default unknown: the size and the share are NaN). */
  vocabularySize?: number
  /** The identity of each token for the usage count (default its string). */
  key?: (text: number, token: number) => string | number
}

const encoder = new TextEncoder()

/**
 * Statistics of tokenisations (one per text) against the words of their sources. Words are runs of non-space
 * characters of each source, so tokenisers that cut text differently are compared on the same words; a word is
 * covered when no unknown token's offsets overlap it. Characters are counted in code points, bytes in UTF-8.
 *
 * @param tokenisations One tokenisation per text, each with its source and the offsets of its tokens.
 * @param options Which tokens are unknown, what identifies a token, and the vocabulary size; see
 *   {@link TokenStatisticsOptions}.
 * @returns The counts and rates.
 *
 * @example Fertility and coverage of one hand-made tokenisation
 * const t = {
 *   kind: 'tokens',
 *   source: 'the cats sat',
 *   tokens: ['the', 'cat', '##s', 'sat'],
 *   offsets: tensor([[0, 3], [4, 7], [7, 8], [9, 12]]),
 * }
 * const s = tokenisationStatistics([t], { isUnknown: (text, token) => token === 2, vocabularySize: 10 })
 * print('words', s.words, ' tokens', s.tokens, ' fertility', s.fertility)
 * print('bytes per token', s.bytesPerToken)
 * print('unknown rate', s.unknownRate, ' word coverage', s.wordCoverage)
 * print('vocabulary usage', s.vocabularyUsage)
 */
export function tokenisationStatistics(
  tokenisations: readonly Tokenisation[],
  options: TokenStatisticsOptions = {},
): TokenStatistics {
  let words = 0
  let characters = 0
  let bytes = 0
  let tokens = 0
  let unknown = 0
  let covered = 0
  const used = new Set<string | number>()
  tokenisations.forEach((t, d) => {
    const src = t.source
    characters += [...src].length
    bytes += encoder.encode(src).length
    tokens += t.tokens.length
    const o = t.offsets.data
    const bad: [number, number][] = []
    for (let k = 0; k < t.tokens.length; k++) {
      used.add(options.key ? options.key(d, k) : t.tokens[k])
      if (options.isUnknown?.(d, k)) {
        unknown++
        bad.push([o[2 * k], o[2 * k + 1]])
      }
    }
    for (const m of src.matchAll(/\S+/gu)) {
      words++
      const s = m.index
      const e = s + m[0].length
      if (!bad.some(([a, b]) => a < e && b > s)) covered++
    }
  })
  const size = options.vocabularySize ?? NaN
  return {
    texts: tokenisations.length,
    words,
    characters,
    bytes,
    tokens,
    fertility: words > 0 ? tokens / words : NaN,
    bytesPerToken: tokens > 0 ? bytes / tokens : NaN,
    charactersPerToken: tokens > 0 ? characters / tokens : NaN,
    unknownTokens: unknown,
    unknownRate: tokens > 0 ? unknown / tokens : NaN,
    wordCoverage: words > 0 ? covered / words : NaN,
    vocabularySize: size,
    vocabularyUsed: used.size,
    vocabularyUsage: used.size / size,
  }
}

/**
 * Statistics of a tokeniser on texts: each is encoded without the template's special tokens, truncation or padding;
 * unknown tokens are those with the unknown id, and usage counts distinct ids against the vocabulary size.
 *
 * @param t The tokeniser, as `aifn-compute/text/pipeline` builds it.
 * @param texts The corpus, one text per entry.
 * @returns The statistics of the encodings.
 *
 * @example A lower-case letter tokeniser meets accented words
 * const tokens = ['<unk>', ' ', ...'abcdefghijklmnopqrstuvwxyz']
 * const vocabulary = { kind: 'vocabulary', tokens, counts: zeros([tokens.length]), specials: ['<unk>'], unknown: 0 }
 * const model = { type: 'character', vocabulary, unknown: '<unk>', byteFallback: false }
 * const letters = { kind: 'tokeniser', model, normaliser: null, preTokeniser: null, postProcessor: null,
 *   decoder: null, specials: [], truncation: null, padding: null }
 * const s = tokeniserStatistics(letters, ['the cat', 'a naïve café'])
 * print('fertility', s.fertility, ' bytes per token', s.bytesPerToken)
 * print('unknown', s.unknownTokens, ' word coverage', s.wordCoverage)
 * print('used', s.vocabularyUsed, 'of', s.vocabularySize)
 */
export function tokeniserStatistics(t: Tokeniser, texts: readonly string[]): TokenStatistics {
  const encodings = texts.map((x) => encodeText(withoutLimits(t), x, null, { addSpecialTokens: false }))
  const unk = t.model.vocabulary.unknown
  const ids = encodings.map((e) => toFlat(e.ids))
  return tokenisationStatistics(encodings.map(encodingTokenisation), {
    isUnknown: (d, k) => unk >= 0 && ids[d][k] === unk,
    key: (d, k) => ids[d][k],
    vocabularySize: vocabularySize(t),
  })
}

// Statistics read whole texts: no truncation, no padding.
/**
 * A copy of a tokeniser with truncation and padding turned off, so that whole texts are measured.
 *
 * @param t The tokeniser; it is not modified.
 * @returns The same stages with `truncation` and `padding` null.
 */
const withoutLimits = (t: Tokeniser): Tokeniser => ({ ...t, truncation: null, padding: null })
