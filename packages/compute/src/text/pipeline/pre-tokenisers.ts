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

/** A pre-tokeniser stage. */
export type PreTokeniser =
  | { readonly type: 'whitespace' }
  | { readonly type: 'whitespaceSplit' }
  | { readonly type: 'bert' }
  | { readonly type: 'punctuation'; readonly behaviour: SplitBehaviour }
  | { readonly type: 'digits'; readonly individual: boolean }
  | {
      readonly type: 'split'
      readonly pattern: TokenPattern | string
      readonly behaviour: SplitBehaviour
      readonly invert: boolean
    }
  | {
      readonly type: 'metaspace'
      readonly replacement: string
      readonly prependScheme: 'always' | 'first' | 'never'
      readonly split: boolean
    }
  | {
      readonly type: 'byteLevel'
      readonly addPrefixSpace: boolean
      readonly pattern: TokenPattern | null
    }
  | { readonly type: 'treebank' }
  | { readonly type: 'casual' }
  | { readonly type: 'sequence'; readonly preTokenisers: readonly PreTokeniser[] }

/** Runs of word characters, or of other non-space characters (Hugging Face's `Whitespace`, `\w+|[^\w\s]+`). */
export function whitespacePreTokeniser(): PreTokeniser {
  return { type: 'whitespace' }
}

/** Split on white space only (Hugging Face's `WhitespaceSplit`). */
export function whitespaceSplitPreTokeniser(): PreTokeniser {
  return { type: 'whitespaceSplit' }
}

/** BERT's basic pre-tokeniser: split on white space and isolate every punctuation character. */
export function bertPreTokeniser(): PreTokeniser {
  return { type: 'bert' }
}

/** Split off punctuation (Unicode P and the ASCII symbols), each character isolated by default. */
export function punctuationPreTokeniser(behaviour: SplitBehaviour = 'isolated'): PreTokeniser {
  return { type: 'punctuation', behaviour }
}

/** Split off digits: each digit its own pre-token (`individual`, as LLaMA), or runs of digits. */
export function digitsPreTokeniser(individual = true): PreTokeniser {
  return { type: 'digits', individual }
}

/**
 * Split by a regular expression (a named token pattern or a source string): matches are delimiters handled by
 * `behaviour`, or with `invert` the matches are the pre-tokens and the gaps the delimiters.
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
 * so word-initial pieces carry the marker and decoding restores the spaces.
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

/** Penn Treebank word tokens as pre-tokens (clitics split, punctuation separated). */
export function treebankPreTokeniser(): PreTokeniser {
  return { type: 'treebank' }
}

/** Casual (tweet) tokens as pre-tokens: URLs, emoticons, handles, hashtags, words. */
export function casualPreTokeniser(): PreTokeniser {
  return { type: 'casual' }
}

/** Pre-tokenisers applied in order, each to the pre-tokens of the one before. */
export function preTokeniserSequence(...preTokenisers: PreTokeniser[]): PreTokeniser {
  return { type: 'sequence', preTokenisers }
}

// ── Splitting ────────────────────────────────────────────────────────────────────────────────────────────────────────

const PUNCTUATION = /[\p{P}!-/:-@[-`{-~]/gu
const WHITESPACE = /\s+/gu

const regexOf = (p: TokenPattern | string): RegExp =>
  typeof p === 'string' && p in TOKEN_PATTERNS ? TOKEN_PATTERNS[p as TokenPattern] : new RegExp(p, 'gu')

/** Split one aligned text by a pattern under a delimiter behaviour (see {@link SplitBehaviour}). */
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

const byteMap = (c: string): string => {
  const table = byteAlphabet()
  let out = ''
  for (const b of new TextEncoder().encode(c)) out += table[b]
  return out
}

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

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\/-]/gu, '\\$&')

/** Apply a pre-tokeniser to each pre-token of a list (the first call gets the whole normalised text as one). */
export function applyPreTokeniser(p: PreTokeniser | null, parts: readonly AlignedText[]): AlignedText[] {
  if (!p) return [...parts]
  return parts.flatMap((a, k) => preTokeniseOne(p, a, k)).filter((x) => x.text.length > 0)
}
