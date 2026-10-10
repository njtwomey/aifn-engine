/**
 * Pre-tokenisers, the second stage of a tokeniser pipeline: they cut normalised text into pre-tokens ("words") that
 * the model segments independently, so no token crosses a pre-token boundary. Some also rewrite the text: Metaspace
 * turns spaces into "▁" (SentencePiece), ByteLevel turns every character into its UTF-8 byte symbols (GPT-2). Every
 * pre-token is aligned text, so offsets survive. Each is plain data, applied by {@link applyPreTokeniser}.
 */

import { TOKEN_PATTERNS, casualTokenise, treebankTokenise, type TokenPattern } from 'aifn-compute/text/tokenise'
import { byteAlphabet } from 'aifn-compute/text/subword'
import { alignedMap, alignedPrepend, alignedReplace, alignedSlice, type AlignedText } from '../aligned'

/**
 * What a split does with the delimiters it finds (Hugging Face's `SplitDelimiterBehavior`): drop them (`removed`),
 * keep each as a pre-token of its own (`isolated`), attach it to the pre-token before or after, or keep runs of
 * them together (`contiguous`).
 */
export type SplitBehaviour = 'removed' | 'isolated' | 'mergedWithPrevious' | 'mergedWithNext' | 'contiguous'

/**
 * A pre-tokeniser stage, tagged by `type`. Of the one-line variants, `punctuation` has the `behaviour` of its
 * delimiters, `digits` whether each digit is `individual`, and `sequence` its `preTokenisers` in order; the others
 * have no fields. The factories below say what each does.
 */
export type PreTokeniser =
  | { readonly type: 'whitespace' }
  | { readonly type: 'whitespaceSplit' }
  | { readonly type: 'bert' }
  | { readonly type: 'punctuation'; readonly behaviour: SplitBehaviour }
  | { readonly type: 'digits'; readonly individual: boolean }
  | {
      readonly type: 'split'
      /** A named token pattern, or a regular-expression source. */
      readonly pattern: TokenPattern | string
      /** What happens to the delimiters. */
      readonly behaviour: SplitBehaviour
      /** When true, the matches are the pre-tokens and the gaps between them the delimiters. */
      readonly invert: boolean
    }
  | {
      readonly type: 'metaspace'
      /** What each space becomes ("▁"). */
      readonly replacement: string
      /** Which pre-tokens get the replacement put before them when they lack it. */
      readonly prependScheme: 'always' | 'first' | 'never'
      /** Whether to split before each replacement character. */
      readonly split: boolean
    }
  | {
      readonly type: 'byteLevel'
      /** Whether to put a space before text that does not start with one. */
      readonly addPrefixSpace: boolean
      /** The named pattern that splits the text before the byte mapping, or null for no split. */
      readonly pattern: TokenPattern | null
    }
  | { readonly type: 'treebank' }
  | { readonly type: 'casual' }
  | { readonly type: 'sequence'; readonly preTokenisers: readonly PreTokeniser[] }

/**
 * Runs of word characters, or of other non-space characters (Hugging Face's `Whitespace`, `\w+|[^\w\s]+`).
 *
 * @returns The pre-tokeniser.
 *
 * @example Punctuation and apostrophes split off
 * print([...preTokenCounts({ preTokeniser: whitespacePreTokeniser() }, ["Hey friend! How's it going?"]).keys()])
 */
export function whitespacePreTokeniser(): PreTokeniser {
  return { type: 'whitespace' }
}

/**
 * Split on white space only (Hugging Face's `WhitespaceSplit`).
 *
 * @returns The pre-tokeniser.
 *
 * @example Punctuation stays attached
 * print([...preTokenCounts({ preTokeniser: whitespaceSplitPreTokeniser() }, ["Hey friend! How's it going?"]).keys()])
 */
export function whitespaceSplitPreTokeniser(): PreTokeniser {
  return { type: 'whitespaceSplit' }
}

/**
 * BERT's basic pre-tokeniser: split on white space and isolate every punctuation character.
 *
 * @returns The pre-tokeniser.
 *
 * @example Every punctuation character on its own
 * print([...preTokenCounts({ preTokeniser: bertPreTokeniser() }, ["Hey friend!? How's it going"]).keys()])
 */
export function bertPreTokeniser(): PreTokeniser {
  return { type: 'bert' }
}

/**
 * Split off punctuation (Unicode P and the ASCII symbols), each character isolated by default. White space is not a
 * delimiter: it stays inside the pre-tokens.
 *
 * @param behaviour What to do with each punctuation character.
 * @returns The pre-tokeniser.
 *
 * @example Isolated, or runs kept together
 * print([...preTokenCounts({ preTokeniser: punctuationPreTokeniser() }, ['Hey, friend!?']).keys()])
 * print([...preTokenCounts({ preTokeniser: punctuationPreTokeniser('contiguous') }, ['Hey, friend!?']).keys()])
 */
export function punctuationPreTokeniser(behaviour: SplitBehaviour = 'isolated'): PreTokeniser {
  return { type: 'punctuation', behaviour }
}

/**
 * Split off digits (Unicode N): each digit its own pre-token (`individual`, as LLaMA), or runs of digits.
 *
 * @param individual True for one pre-token per digit, false for one per run of digits.
 * @returns The pre-tokeniser.
 *
 * @example A year, digit by digit or whole
 * print([...preTokenCounts({ preTokeniser: digitsPreTokeniser() }, ['in 1984']).keys()])
 * print([...preTokenCounts({ preTokeniser: digitsPreTokeniser(false) }, ['in 1984']).keys()])
 */
export function digitsPreTokeniser(individual = true): PreTokeniser {
  return { type: 'digits', individual }
}

/**
 * Split by a regular expression (a named token pattern or a source string): matches are delimiters handled by
 * `behaviour`, or with `invert` the matches are the pre-tokens and the gaps the delimiters. Empty matches are ignored.
 *
 * @param pattern A named token pattern of `tokenise` (such as `gpt2`), or a regular-expression source compiled with
 *   the `gu` flags.
 * @param options `behaviour` (default `isolated`) and `invert` (default false).
 * @returns The pre-tokeniser.
 *
 * @example Hyphens removed or merged, and the GPT-2 pattern's matches kept
 * const text = ['state-of-the-art']
 * print([...preTokenCounts({ preTokeniser: splitPreTokeniser('-', { behaviour: 'removed' }) }, text).keys()])
 * const merged = splitPreTokeniser('-', { behaviour: 'mergedWithPrevious' })
 * print([...preTokenCounts({ preTokeniser: merged }, text).keys()])
 * const gpt2 = splitPreTokeniser('gpt2', { invert: true, behaviour: 'removed' })
 * print([...preTokenCounts({ preTokeniser: gpt2 }, ["I'm here"]).keys()])
 */
export function splitPreTokeniser(
  pattern: TokenPattern | string,
  options: { behaviour?: SplitBehaviour; invert?: boolean } = {},
): PreTokeniser {
  return { type: 'split', pattern, behaviour: options.behaviour ?? 'isolated', invert: options.invert ?? false }
}

/**
 * SentencePiece's Metaspace (Kudo & Richardson 2018): every space becomes `replacement` ("▁", U+2581), a "▁" is put
 * before the first pre-token (`first`) or every one (`always`) that lacks it, and the text is split before each "▁",
 * so word-initial pieces carry the marker and decoding restores the spaces. Only U+0020 spaces are replaced.
 *
 * @param options `replacement` (default "▁"), `prependScheme` (default `always`; `never` adds none) and `split`
 *   (default true; false keeps the text whole).
 * @returns The pre-tokeniser.
 *
 * @example Split before each marker, or not
 * print([...preTokenCounts({ preTokeniser: metaspacePreTokeniser() }, ['Hey friend, hey']).keys()])
 * print([...preTokenCounts({ preTokeniser: metaspacePreTokeniser({ split: false }) }, ['Hey friend']).keys()])
 */
export function metaspacePreTokeniser(
  options: { replacement?: string; prependScheme?: 'always' | 'first' | 'never'; split?: boolean } = {},
): PreTokeniser {
  return {
    type: 'metaspace',
    replacement: options.replacement ?? '▁',
    prependScheme: options.prependScheme ?? 'always',
    split: options.split ?? true,
  }
}

/**
 * GPT-2's byte level (Radford et al. 2019): optionally a space before the text, a regular-expression split (`gpt2` by
 * default, `cl100k` or `o200k` for tiktoken's encodings, or none), then every character as its UTF-8 bytes, each
 * shown as one of 256 printable byte symbols ("Ġ" for a space). Any text, emoji included, is then made of known
 * symbols, so the model never needs an unknown token.
 *
 * @param options `addPrefixSpace` (default false) puts a space before text that does not start with one, so the first
 *   word is cut like the others; `pattern` (default `gpt2`) is the split, or null for none.
 * @returns The pre-tokeniser.
 *
 * @example Spaces become "Ġ" and "é" two byte symbols
 * print([...preTokenCounts({ preTokeniser: byteLevelPreTokeniser() }, ['Hey friend, café!']).keys()])
 * print([...preTokenCounts({ preTokeniser: byteLevelPreTokeniser({ addPrefixSpace: true }) }, ['Hey']).keys()])
 */
export function byteLevelPreTokeniser(
  options: { addPrefixSpace?: boolean; pattern?: TokenPattern | null } = {},
): PreTokeniser {
  return {
    type: 'byteLevel',
    addPrefixSpace: options.addPrefixSpace ?? false,
    pattern: options.pattern === undefined ? 'gpt2' : options.pattern,
  }
}

/**
 * Penn Treebank word tokens as pre-tokens (clitics split, punctuation separated). Text between the tokens is dropped.
 *
 * @returns The pre-tokeniser.
 *
 * @example A contraction split
 * print([...preTokenCounts({ preTokeniser: treebankPreTokeniser() }, ["They don't know."]).keys()])
 */
export function treebankPreTokeniser(): PreTokeniser {
  return { type: 'treebank' }
}

/**
 * Casual (tweet) tokens as pre-tokens: URLs, emoticons, handles, hashtags, words. Text between the tokens is dropped.
 *
 * @returns The pre-tokeniser.
 *
 * @example A handle, an emoticon, a hashtag and a URL kept whole
 * print([...preTokenCounts({ preTokeniser: casualPreTokeniser() }, ['@ann loved it :-) #nlp https://x.org']).keys()])
 */
export function casualPreTokeniser(): PreTokeniser {
  return { type: 'casual' }
}

/**
 * Pre-tokenisers applied in order, each to the pre-tokens of the one before.
 *
 * @param preTokenisers The pre-tokenisers, first applied first.
 * @returns The pre-tokeniser.
 *
 * @example White space, then digits
 * const p = preTokeniserSequence(whitespaceSplitPreTokeniser(), digitsPreTokeniser())
 * print([...preTokenCounts({ preTokeniser: p }, ['room 42b']).keys()])
 */
export function preTokeniserSequence(...preTokenisers: PreTokeniser[]): PreTokeniser {
  return { type: 'sequence', preTokenisers }
}

// ── Splitting ────────────────────────────────────────────────────────────────────────────────────────────────────────

const PUNCTUATION = /[\p{P}!-/:-@[-`{-~]/gu
const WHITESPACE = /\s+/gu

/**
 * The regular expression of a split pattern.
 *
 * @param p A named token pattern of `tokenise`, or a regular-expression source.
 * @returns The named pattern's expression, or the source compiled with the `gu` flags.
 */
const regexOf = (p: TokenPattern | string): RegExp =>
  typeof p === 'string' && p in TOKEN_PATTERNS ? TOKEN_PATTERNS[p as TokenPattern] : new RegExp(p, 'gu')

/**
 * Split one aligned text by a pattern under a delimiter behaviour (see {@link SplitBehaviour}). Empty matches are
 * ignored.
 *
 * @param a The text to split, with its alignment.
 * @param re The delimiter pattern; a copy with the `g` flag is used, so `re` itself is not advanced.
 * @param behaviour What to do with the delimiters.
 * @param invert When true, the matches are kept as pre-tokens and the text between them is the delimiter.
 * @returns The pieces, in order, each aligned to the same original.
 *
 * @example The five behaviours on "a--b-c"
 * const text = 'a--b-c'
 * // The identity alignment: code unit i of the text comes from [i, i + 1) of the original.
 * const a = { original: text, text, spans: Int32Array.from({ length: 2 * text.length }, (_, k) => (k + 1) >> 1) }
 * for (const b of ['removed', 'isolated', 'mergedWithPrevious', 'mergedWithNext', 'contiguous'])
 *   print(b, splitAligned(a, /-/g, b).map((x) => x.text))
 */
export function splitAligned(a: AlignedText, re: RegExp, behaviour: SplitBehaviour, invert = false): AlignedText[] {
  const g = new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g')
  // The text as segments [start, end, isDelimiter].
  const segments: [number, number, boolean][] = []
  let at = 0
  for (const m of a.text.matchAll(g)) {
    if (m[0].length === 0) continue
    if (m.index > at) segments.push([at, m.index, invert])
    segments.push([m.index, m.index + m[0].length, !invert])
    at = m.index + m[0].length
  }
  if (at < a.text.length) segments.push([at, a.text.length, invert])
  const out: [number, number][] = []
  // Whether each output range is a delimiter (for `contiguous`, which joins adjacent delimiters).
  const isDelimiter: boolean[] = []
  let pending: [number, number] | null = null
  const push = (r: [number, number], d: boolean) => {
    out.push(r)
    isDelimiter.push(d)
  }
  for (const [s, e, delimiter] of segments) {
    if (!delimiter) {
      if (pending) {
        push([pending[0], e], false)
        pending = null
      } else push([s, e], false)
      continue
    }
    const last = out.length - 1
    switch (behaviour) {
      case 'removed':
        break
      case 'isolated':
        push([s, e], true)
        break
      case 'mergedWithPrevious':
        if (last >= 0 && out[last][1] === s) out[last][1] = e
        else push([s, e], true)
        break
      case 'mergedWithNext':
        if (pending) push(pending, true)
        pending = [s, e]
        break
      case 'contiguous':
        if (last >= 0 && isDelimiter[last] && out[last][1] === s) out[last][1] = e
        else push([s, e], true)
        break
    }
  }
  if (pending) out.push(pending)
  return out.map(([s, e]) => alignedSlice(a, s, e))
}

/**
 * A character as GPT-2 byte symbols.
 *
 * @param c The character.
 * @returns The byte symbol of each byte of its UTF-8 encoding, joined.
 */
const byteMap = (c: string): string => {
  const table = byteAlphabet()
  let out = ''
  for (const b of new TextEncoder().encode(c)) out += table[b]
  return out
}

/**
 * Apply a pre-tokeniser to one pre-token.
 *
 * @param p The pre-tokeniser.
 * @param a The pre-token, with its alignment.
 * @param index Its position in the list being pre-tokenised; Metaspace's `first` scheme prepends only at 0.
 * @returns The pieces it is cut into (empty ones included; the caller drops them).
 */
function preTokeniseOne(p: PreTokeniser, a: AlignedText, index: number): AlignedText[] {
  switch (p.type) {
    case 'whitespace':
      return splitAligned(a, TOKEN_PATTERNS.wordsOrSymbols, 'removed', true)
    case 'whitespaceSplit':
      return splitAligned(a, WHITESPACE, 'removed')
    case 'bert':
      return splitAligned(a, WHITESPACE, 'removed').flatMap((x) => splitAligned(x, PUNCTUATION, 'isolated'))
    case 'punctuation':
      return splitAligned(a, PUNCTUATION, p.behaviour)
    case 'digits':
      return splitAligned(a, p.individual ? /\p{N}/gu : /\p{N}+/gu, 'isolated')
    case 'split':
      return splitAligned(a, regexOf(p.pattern), p.behaviour, p.invert)
    case 'metaspace': {
      let x = alignedReplace(a, / /gu, () => p.replacement)
      const prepend = p.prependScheme === 'always' || (p.prependScheme === 'first' && index === 0)
      if (prepend && x.text.length > 0 && !x.text.startsWith(p.replacement)) x = alignedPrepend(x, p.replacement)
      return p.split ? splitAligned(x, new RegExp(escape(p.replacement), 'gu'), 'mergedWithNext') : [x]
    }
    case 'byteLevel': {
      let x = a
      if (p.addPrefixSpace && x.text.length > 0 && !x.text.startsWith(' ')) x = alignedPrepend(x, ' ')
      const parts = p.pattern ? splitAligned(x, TOKEN_PATTERNS[p.pattern], 'removed', true) : [x]
      return parts.map((y) => alignedMap(y, byteMap))
    }
    case 'treebank': {
      const t = treebankTokenise(a.text)
      return Array.from({ length: t.tokens.length }, (_, k) =>
        alignedSlice(a, t.offsets.data[2 * k], t.offsets.data[2 * k + 1]),
      )
    }
    case 'casual': {
      const t = casualTokenise(a.text)
      return Array.from({ length: t.tokens.length }, (_, k) =>
        alignedSlice(a, t.offsets.data[2 * k], t.offsets.data[2 * k + 1]),
      )
    }
    case 'sequence':
      return p.preTokenisers.reduce<AlignedText[]>((parts, q) => applyPreTokeniser(q, parts), [a])
  }
}

/**
 * A string escaped for use as a literal in a regular expression.
 *
 * @param s The string.
 * @returns It with every regular-expression metacharacter backslash-escaped.
 */
const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\/-]/gu, '\\$&')

/**
 * Apply a pre-tokeniser to each pre-token of a list (the first call gets the whole normalised text as one).
 *
 * @param p The pre-tokeniser, or null to leave the list as it is.
 * @param parts The pre-tokens so far, with their alignments; not modified.
 * @returns The new pre-tokens, in order, without empty ones.
 *
 * @example BERT's pre-tokens with their start in the original
 * const text = 'Hi, you 2'
 * // The identity alignment: code unit i of the text comes from [i, i + 1) of the original.
 * const a = { original: text, text, spans: Int32Array.from({ length: 2 * text.length }, (_, k) => (k + 1) >> 1) }
 * const parts = applyPreTokeniser(bertPreTokeniser(), [a])
 * print('pre-tokens =', parts.map((x) => x.text))
 * print('starts =', parts.map((x) => x.spans[0]))
 */
export function applyPreTokeniser(p: PreTokeniser | null, parts: readonly AlignedText[]): AlignedText[] {
  if (!p) return [...parts]
  return parts.flatMap((a, k) => preTokeniseOne(p, a, k)).filter((x) => x.text.length > 0)
}
