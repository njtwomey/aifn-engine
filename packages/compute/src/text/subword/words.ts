/**
 * What the subword trainers and encoders share: the word-count table a trainer starts from, the pre-tokenisation an
 * encoder applies, and the assembly of subword tokens with offsets into the source text.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData } from 'aifn-compute/foundation/tensor'
import { tokenise, type Tokenisation, type TokenPattern } from 'aifn-compute/text/tokenise'

/** Training words: a list of words (counted), or words with their counts. */
export type WordCountsLike = readonly string[] | ReadonlyMap<string, number> | Readonly<Record<string, number>>

/**
 * Distinct words in order of first appearance, with their counts. The empty word is dropped, and so is a word whose
 * given count is 0. Throws `DomainError` when a given count is negative, NaN or infinite, or when no word is left.
 *
 * @param words The training words: a list, in which each occurrence counts once, or a `Map` or plain object from word
 *   to count.
 * @param op The caller's name, for error messages.
 * @returns `words`, the distinct words, and `counts`, the total count of each (summed when a word repeats).
 */
export function wordTable(words: WordCountsLike, op: string): { words: string[]; counts: number[] } {
  const m = new Map<string, number>()
  if (Array.isArray(words)) for (const w of words as readonly string[]) m.set(w, (m.get(w) ?? 0) + 1)
  else {
    const entries = words instanceof Map ? [...words.entries()] : Object.entries(words as Record<string, number>)
    for (const [w, c] of entries) {
      if (!(c >= 0) || !Number.isFinite(c)) throw new DomainError(op, `${op}: the count of '${w}' is not a count`)
      if (c > 0) m.set(w, (m.get(w) ?? 0) + c)
    }
  }
  m.delete('')
  if (m.size === 0) throw new DomainError(op, `${op}: no training words`)
  return { words: [...m.keys()], counts: [...m.values()] }
}

/** A piece of a word: its string and the [start, end) range of the word it covers, in UTF-16 code units. */
export interface Piece {
  /** The piece's string, as the tokeniser emits it (with any prefix or end-of-word marker). */
  token: string
  /** Where the covered range starts in the word, in UTF-16 code units. */
  start: number
  /** Where the covered range ends (exclusive); equal to `start` for a zero-width marker. */
  end: number
}

/**
 * The code points of a word as pieces with their ranges (a character outside the Basic Multilingual Plane spans two
 * code units).
 *
 * @param word The word to split.
 * @returns One piece per code point, in order.
 */
export function characterPieces(word: string): Piece[] {
  const out: Piece[] = []
  let at = 0
  for (const c of word) {
    out.push({ token: c, start: at, end: at + c.length })
    at += c.length
  }
  return out
}

/**
 * Encode text word by word: pre-tokenise with `pattern`, segment each word with `segment` (pieces with ranges relative
 * to the word) and collect the subword tokens with offsets into the text.
 *
 * @param text The text to encode.
 * @param pattern The named pre-tokeniser pattern of `tokenise` that splits the text into words.
 * @param segment The word segmenter: given one word, its pieces with ranges relative to that word.
 * @returns The subword tokens of every word in order, with their [start, end) offsets into `text` as an
 *   $n \times 2$ int32 tensor.
 */
export function encodeByWords(
  text: string,
  pattern: TokenPattern,
  segment: (word: string) => readonly Piece[],
): Tokenisation {
  const words = tokenise(text, { pattern })
  const tokens: string[] = []
  const offsets: number[] = []
  const o = words.offsets.data
  words.tokens.forEach((w, k) => {
    for (const p of segment(w)) {
      tokens.push(p.token)
      offsets.push(o[2 * k] + p.start, o[2 * k] + p.end)
    }
  })
  return { kind: 'tokens', source: text, tokens, offsets: fromData(Int32Array.from(offsets), [tokens.length, 2]) }
}

/**
 * Code-point order (as Rust's and Python's string order, so ties and alphabets sort as the reference tokenisers do).
 * JavaScript's `<` compares UTF-16 code units, which puts astral characters (surrogate pairs, U+D800…) before
 * U+E000–U+FFFF; this compares whole code points instead.
 *
 * @param a The first string.
 * @param b The second string.
 * @returns $-1$, 0 or 1 as `a` sorts before, with or after `b`; a proper prefix sorts first.
 */
export function byCodePoint(a: string, b: string): number {
  for (let i = 0; i < a.length && i < b.length;) {
    const ca = a.codePointAt(i)!
    const cb = b.codePointAt(i)!
    if (ca !== cb) return ca < cb ? -1 : 1
    i += ca > 0xffff ? 2 : 1
  }
  return Math.sign(a.length - b.length)
}
