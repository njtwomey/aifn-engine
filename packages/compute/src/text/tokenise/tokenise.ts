/**
 * Tokenisation: splitting text into tokens with their offsets, by a regular expression (words, words and punctuation,
 * white space, runs of two or more word characters as in scikit-learn, the GPT-2, cl100k, o200k and BERT
 * pre-tokenisers) or into characters (code points or grapheme clusters), and joining tokens back into text.
 *
 * Every tokeniser returns a `Tokenisation`: the token strings with their [start, end) offsets into the source in UTF-16
 * code units, so that a token can always be traced back to the text it came from.
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import { fromData } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A tokenised text: the token strings and, row by row, their offsets into `source` as an int32 tensor of shape
 * [n, 2] holding [start, end) in UTF-16 code units, so `source.slice(start, end)` is the token.
 */
export interface Tokenisation {
  /** The tag `'tokens'`. */
  readonly kind: 'tokens'
  /** The text that was tokenised. */
  readonly source: string
  /** The tokens, in order of their start offsets. */
  readonly tokens: readonly string[]
  /** Row `k` holds token `k`'s [start, end) in `source` (int32 [n, 2]). */
  readonly offsets: Tensor
}

/**
 * The named token patterns of {@link tokenise}:
 *
 * - `words`: runs of letters and digits, with inner apostrophes or hyphens ("don't", "state-of-the-art"); punctuation
 *   is dropped;
 * - `wordsAndPunctuation`: `words`, plus every other non-space character as a token of its own;
 * - `whitespace`: maximal runs of non-space characters;
 * - `alphanumeric`: runs of two or more word characters (letters, digits, underscore), scikit-learn's default
 *   `(?u)\b\w\w+\b`, so single letters are dropped;
 * - `gpt2`: the GPT-2 pre-tokeniser (Radford et al. 2019): English contractions, an optional space followed by letters,
 *   by digits or by other symbols, and white space; the tokens concatenate back to the text exactly;
 * - `cl100k`: the pre-tokeniser of tiktoken's `cl100k_base` (GPT-3.5 and GPT-4): case-insensitive contractions, letters
 *   with one optional leading non-letter, numbers in groups of at most three digits, symbols with an optional leading
 *   space and trailing newlines, and newline-aware white space; exact concatenation;
 * - `o200k`: the pre-tokeniser of tiktoken's `o200k_base` (GPT-4o): as `cl100k`, but words split at lower-to-upper case
 *   changes ("camelCase" gives "camel", "Case") and contractions attach to the word before them;
 * - `bert`: BERT's basic pre-tokeniser (Devlin et al. 2019): runs of non-space, non-punctuation characters, and every
 *   punctuation character (Unicode P, and all ASCII symbols) on its own;
 * - `wordsOrSymbols`: Hugging Face's `Whitespace` pre-tokeniser `\w+|[^\w\s]+`: runs of word characters, or runs of
 *   other non-space characters.
 */
export const TOKEN_PATTERNS = {
  words: /[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu,
  wordsAndPunctuation: /[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]/gu,
  whitespace: /\S+/gu,
  alphanumeric: /(?<![\p{L}\p{N}_])[\p{L}\p{N}_]{2,}(?![\p{L}\p{N}_])/gu,
  gpt2: /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu,
  cl100k:
    /'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu,
  o200k:
    /[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu,
  bert: /[^\s\p{P}!-/:-@[-`{-~]+|[\p{P}!-/:-@[-`{-~]/gu,
  wordsOrSymbols:
    /[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]+|[^\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}\s]+/gu,
} as const

/** A named token pattern. */
export type TokenPattern = keyof typeof TOKEN_PATTERNS

/** Options of {@link tokenise}. */
export interface TokeniseOptions {
  /** A named pattern (default `words`) or a regular expression; a pattern without the `g` flag gets it. */
  pattern?: TokenPattern | RegExp
}

/**
 * Assemble a `Tokenisation` from tokens and flat offsets.
 *
 * @param source The text the tokens came from.
 * @param tokens The tokens, in order.
 * @param offsets The offsets, two per token: start and end in UTF-16 code units, token after token.
 * @returns The tokenisation, with the offsets as an int32 tensor of shape [n, 2].
 */
function tokenisation(source: string, tokens: string[], offsets: number[]): Tokenisation {
  return { kind: 'tokens', source, tokens, offsets: fromData(Int32Array.from(offsets), [tokens.length, 2]) }
}

/**
 * Tokenise `text` by a regular expression: every non-empty match is a token, with its offsets. Text between matches is
 * dropped. Throws `DomainError` for an unknown pattern name.
 *
 * @param text The text to tokenise.
 * @param options The pattern: a name from `TOKEN_PATTERNS` (default `words`) or a regular expression.
 * @returns The tokens and their offsets into `text`.
 *
 * @example Words, words and punctuation, and scikit-learn's default
 * const text = "Don't stop: it's a state-of-the-art tokeniser!"
 * print('words              ', tokenise(text).tokens)
 * print('wordsAndPunctuation', tokenise(text, { pattern: 'wordsAndPunctuation' }).tokens)
 * print('alphanumeric       ', tokenise(text, { pattern: 'alphanumeric' }).tokens)
 *
 * @example GPT-2's pre-tokeniser keeps the spaces, so the tokens join back to the text
 * const t = tokenise("I've 2 cats  and 30 dogs", { pattern: 'gpt2' })
 * print(t.tokens.map((x) => JSON.stringify(x)).join(' '))
 * print('offsets', t.offsets)
 */
export function tokenise(text: string, options: TokeniseOptions = {}): Tokenisation {
  const p = options.pattern ?? 'words'
  const base = typeof p === 'string' ? TOKEN_PATTERNS[p] : p
  if (base === undefined) throw new DomainError('tokenise', `tokenise: unknown pattern '${String(p)}'`)
  const re = new RegExp(base.source, base.flags.includes('g') ? base.flags : base.flags + 'g')
  const tokens: string[] = []
  const offsets: number[] = []
  for (const m of text.matchAll(re)) {
    if (m[0].length === 0) continue
    tokens.push(m[0])
    offsets.push(m.index, m.index + m[0].length)
  }
  return tokenisation(text, tokens, offsets)
}

/**
 * Split on white space: maximal runs of non-space characters, as Python's `str.split()`.
 *
 * @param text The text to split.
 * @returns The runs and their offsets into `text`.
 *
 * @example Punctuation stays attached
 * print(whitespaceTokenise('  Hello,  world!\tBye.').tokens)
 */
export function whitespaceTokenise(text: string): Tokenisation {
  return tokenise(text, { pattern: 'whitespace' })
}

/** Options of {@link characterTokenise}. */
export interface CharacterOptions {
  /**
   * `codePoint` (default): one token per Unicode code point; `grapheme`: one per user-perceived character (extended
   * grapheme cluster, UAX #29, via `Intl.Segmenter`), so "é" written as e + combining accent, or a flag emoji, is one.
   */
  unit?: 'codePoint' | 'grapheme'
  /** Drop white-space characters (default false). */
  skipWhitespace?: boolean
}

/**
 * Tokenise `text` into characters, with offsets.
 *
 * @param text The text to split.
 * @param options The unit (code points or grapheme clusters) and whether white space is dropped; see
 *   {@link CharacterOptions}.
 * @returns One token per character, with its offsets into `text` (a character outside the Basic Multilingual Plane
 *   spans two code units).
 *
 * @example Code points and grapheme clusters differ on a decomposed accent
 * const text = 'café ok'
 * print('code points', characterTokenise(text, { skipWhitespace: true }).tokens)
 * print('graphemes  ', characterTokenise(text, { unit: 'grapheme', skipWhitespace: true }).tokens)
 */
export function characterTokenise(text: string, options: CharacterOptions = {}): Tokenisation {
  const { unit = 'codePoint', skipWhitespace = false } = options
  const tokens: string[] = []
  const offsets: number[] = []
  const push = (s: string, at: number) => {
    if (skipWhitespace && /^\s+$/u.test(s)) return
    tokens.push(s)
    offsets.push(at, at + s.length)
  }
  if (unit === 'grapheme') {
    for (const g of new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(text)) push(g.segment, g.index)
  } else {
    let at = 0
    for (const c of text) {
      push(c, at)
      at += c.length
    }
  }
  return tokenisation(text, tokens, offsets)
}

// No space before closing punctuation and clitics; none after opening punctuation.
/** Tokens that take no space before them when a token list is detokenised: closing punctuation and English clitics. */
const CLOSING = /^(?:[.,!?;:%)\]}»”’…]+|'s|'t|'re|'ve|'m|'ll|'d|n't)$/iu
/** Tokens that take no space after them when a token list is detokenised: opening brackets, quotes and signs. */
const OPENING = /^[([{«“$#¿¡]$/u

/**
 * Join tokens back into text. A `Tokenisation` restores its source exactly between its first and last token (the gaps
 * between tokens are copied from the source). A plain list is joined with single spaces, except before closing
 * punctuation and English clitics ("'s", "n't") and after opening brackets and quotes, so `["Hello", ",", "world",
 * "!"]` gives "Hello, world!".
 *
 * @param tokens A tokenisation, or a list of token strings.
 * @returns The text: a slice of the source for a tokenisation (from the first token's start to the last token's end),
 *   or the tokens joined by the spacing rules for a list.
 *
 * @example A list, and a tokenisation that dropped its punctuation
 * print(detokenise(['“', 'Hello', ',', 'world', '!', '”', 'She', 'did', "n't", 'say', '(', 'much', ')', '.']))
 * const t = tokenise('  Hello, world!  ')
 * print('tokens', t.tokens, '->', JSON.stringify(detokenise(t)))
 */
export function detokenise(tokens: Tokenisation | readonly string[]): string {
  if (!Array.isArray(tokens)) {
    const t = tokens as Tokenisation
    if (t.tokens.length === 0) return ''
    const o = t.offsets.data
    return t.source.slice(o[0], o[2 * t.tokens.length - 1])
  }
  let out = ''
  let previous = ''
  for (const tok of tokens as readonly string[]) {
    if (out.length > 0 && !CLOSING.test(tok) && !OPENING.test(previous)) out += ' '
    out += tok
    previous = tok
  }
  return out
}
