/**
 * Normalisers, the first stage of a tokeniser pipeline: maps from text to text that keep, for every character they
 * produce, the range of the original it came from, so tokens found after normalisation still have offsets into the
 * text as typed. Each is plain data (`{ type, … }`), applied by {@link applyNormaliser}.
 */

import { caseFold } from 'aifn-compute/text/normalise'
import { alignedMap, alignedPrepend, alignedReplace, alignedSlice, type AlignedText } from '../aligned'

/** A Unicode normal form. */
export type NormalForm = 'NFC' | 'NFD' | 'NFKC' | 'NFKD'

/**
 * A normaliser stage, tagged by `type`: `unicode` with its normal `form`; `replace` with the regular-expression
 * source `pattern` and the replacement `content`; `prepend` with the `content` put first; `strip` with whether to strip
 * the `left` and `right` ends; `sequence` with its `normalisers` in order. The others have no fields.
 */
export type Normaliser =
  | { readonly type: 'unicode'; readonly form: NormalForm }
  | { readonly type: 'lowercase' }
  | { readonly type: 'caseFold' }
  | { readonly type: 'stripAccents' }
  | { readonly type: 'replace'; readonly pattern: string; readonly content: string }
  | { readonly type: 'prepend'; readonly content: string }
  | { readonly type: 'strip'; readonly left: boolean; readonly right: boolean }
  | { readonly type: 'collapseWhitespace' }
  | { readonly type: 'sequence'; readonly normalisers: readonly Normaliser[] }

/**
 * A Unicode normal form, applied per grapheme cluster (normalisation never composes across one).
 *
 * @param form The normal form: `NFKC` (the default) and `NFKD` also replace compatibility characters (ligatures,
 *   full-width letters, circled digits) by their plain equivalents; `NFC` composes and `NFD` decomposes accents.
 * @returns The normaliser.
 *
 * @example Compatibility characters under NFKC, and NFD splitting an accent off
 * print([...preTokenCounts({ normaliser: unicodeNormaliser() }, ['ﬁ café ①']).keys()])
 * const [nfd] = preTokenCounts({ normaliser: unicodeNormaliser('NFD') }, ['café']).keys()
 * print('length of café in NFD:', nfd.length)
 */
export function unicodeNormaliser(form: NormalForm = 'NFKC'): Normaliser {
  return { type: 'unicode', form }
}

/**
 * Lower case, per code point ("Σ" becomes "σ" everywhere, never the final "ς", as Hugging Face's `Lowercase`).
 *
 * @returns The normaliser.
 *
 * @example Greek, German and Turkish capitals
 * print([...preTokenCounts({ normaliser: lowercaseNormaliser() }, ['ΣΑΣ Straße İ']).keys()])
 */
export function lowercaseNormaliser(): Normaliser {
  return { type: 'lowercase' }
}

/**
 * Full Unicode case folding ("ß" becomes "ss"), for caseless matching.
 *
 * @returns The normaliser.
 *
 * @example "Straße" and "STRASSE" fold to the same string
 * print([...preTokenCounts({ normaliser: caseFoldNormaliser() }, ['Straße STRASSE']).keys()])
 */
export function caseFoldNormaliser(): Normaliser {
  return { type: 'caseFold' }
}

/**
 * Remove non-spacing marks (category Mn); after NFD or NFKD this strips accents.
 *
 * @returns The normaliser.
 *
 * @example Decompose first, then strip
 * const n = normaliserSequence(unicodeNormaliser('NFD'), stripAccentsNormaliser())
 * print([...preTokenCounts({ normaliser: n }, ['café naïve Ångström']).keys()])
 * print('without NFD:', [...preTokenCounts({ normaliser: stripAccentsNormaliser() }, ['café']).keys()])
 */
export function stripAccentsNormaliser(): Normaliser {
  return { type: 'stripAccents' }
}

/**
 * Replace every match of a regular expression (given by its source; Unicode, global) by `content`. The replacement's
 * characters take the range of the match they replace.
 *
 * @param pattern The regular expression's source, compiled with the `gu` flags.
 * @param content The literal replacement (`$` patterns are not expanded).
 * @returns The normaliser.
 *
 * @example LaTeX-style quotes to plain ones, and digits masked
 * print([...preTokenCounts({ normaliser: replaceNormaliser("``|''", '"') }, ["``quoted''"]).keys()])
 * print([...preTokenCounts({ normaliser: replaceNormaliser('\\d', '0') }, ['call 555-1234']).keys()])
 */
export function replaceNormaliser(pattern: string, content: string): Normaliser {
  return { type: 'replace', pattern, content }
}

/**
 * Put `content` before any non-empty text (SentencePiece's leading "▁", as Hugging Face's `Prepend`). The prefix takes
 * the range of the first character, so a token made of it alone still points at the start of the text.
 *
 * @param content The string to put first.
 * @returns The normaliser.
 *
 * @example The SentencePiece marker
 * print([...preTokenCounts({ normaliser: prependNormaliser('▁') }, ['hello world']).keys()])
 */
export function prependNormaliser(content: string): Normaliser {
  return { type: 'prepend', content }
}

/**
 * Remove white space at the start, the end or both.
 *
 * @param left Whether to remove white space at the start.
 * @param right Whether to remove white space at the end.
 * @returns The normaliser.
 *
 * @example Both ends, and the end only
 * const text = '  hi there  '
 * print(JSON.stringify([...preTokenCounts({ normaliser: stripNormaliser() }, [text]).keys()]))
 * print(JSON.stringify([...preTokenCounts({ normaliser: stripNormaliser(false, true) }, [text]).keys()]))
 */
export function stripNormaliser(left = true, right = true): Normaliser {
  return { type: 'strip', left, right }
}

/**
 * Collapse every white-space run to one space and trim.
 *
 * @returns The normaliser.
 *
 * @example Spaces, tabs and newlines
 * const n = collapseWhitespaceNormaliser()
 * print(JSON.stringify([...preTokenCounts({ normaliser: n }, ['  hi \n\t there  ']).keys()]))
 */
export function collapseWhitespaceNormaliser(): Normaliser {
  return { type: 'collapseWhitespace' }
}

/**
 * Normalisers applied in order.
 *
 * @param normalisers The normalisers, first applied first.
 * @returns The normaliser.
 *
 * @example BERT's uncased normalisation
 * const bert = normaliserSequence(unicodeNormaliser('NFD'), stripAccentsNormaliser(), lowercaseNormaliser())
 * print([...preTokenCounts({ normaliser: bert }, ['Héllo Wörld']).keys()])
 */
export function normaliserSequence(...normalisers: Normaliser[]): Normaliser {
  return { type: 'sequence', normalisers }
}

/**
 * Remove white space (`\s`, Unicode) at either end of an aligned text, keeping the ranges of what remains.
 *
 * @param a The aligned text.
 * @param left Whether to remove white space at the start.
 * @param right Whether to remove white space at the end.
 * @returns The stripped aligned text.
 */
function strip(a: AlignedText, left: boolean, right: boolean): AlignedText {
  let s = 0
  let e = a.text.length
  if (left) while (s < e && /\s/u.test(a.text[s])) s++
  if (right) while (e > s && /\s/u.test(a.text[e - 1])) e--
  return alignedSlice(a, s, e)
}

/**
 * Apply a normaliser to aligned text: every character of the result keeps the range of the original it came from.
 *
 * @param n The normaliser.
 * @param a The text to normalise, with its alignment to the original.
 * @returns The normalised text, aligned to the same original.
 *
 * @example The ligature "ﬁ" becomes two letters with the same range
 * const text = 'ﬁne'
 * // The identity alignment: code unit i of the text comes from [i, i + 1) of the original.
 * const a = { original: text, text, spans: Int32Array.from({ length: 2 * text.length }, (_, k) => (k + 1) >> 1) }
 * const b = applyNormaliser(unicodeNormaliser(), a)
 * print('text =', b.text)
 * print('ranges =', Array.from(b.text, (_, i) => [b.spans[2 * i], b.spans[2 * i + 1]]))
 */
export function applyNormaliser(n: Normaliser, a: AlignedText): AlignedText {
  switch (n.type) {
    case 'unicode':
      return alignedMap(a, (c) => c.normalize(n.form), 'grapheme')
    case 'lowercase':
      return alignedMap(a, (c) => c.toLowerCase())
    case 'caseFold':
      return alignedMap(a, caseFold)
    case 'stripAccents':
      return alignedMap(a, (c) => (/^\p{Mn}$/u.test(c) ? '' : c))
    case 'replace':
      return alignedReplace(a, new RegExp(n.pattern, 'gu'), () => n.content)
    case 'prepend':
      return a.text.length > 0 ? alignedPrepend(a, n.content) : a
    case 'strip':
      return strip(a, n.left, n.right)
    case 'collapseWhitespace':
      return strip(alignedReplace(a, /\s+/gu, ' '), true, true)
    case 'sequence':
      return n.normalisers.reduce((acc, m) => applyNormaliser(m, acc), a)
  }
}
