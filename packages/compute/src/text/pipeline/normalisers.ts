/**
 * Normalisers, the first stage of a tokeniser pipeline: maps from text to text that keep, for every character they
 * produce, the range of the original it came from, so tokens found after normalisation still have offsets into the
 * text as typed. Each is plain data (`{ type, … }`), applied by {@link applyNormaliser}.
 */

import { caseFold } from 'aifn-compute/text/normalise'
import { alignedMap, alignedPrepend, alignedReplace, alignedSlice, type AlignedText } from '../aligned'

/** A Unicode normal form. */
export type NormalForm = 'NFC' | 'NFD' | 'NFKC' | 'NFKD'

/** A normaliser stage. */
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

/** A Unicode normal form, applied per grapheme cluster (normalisation never composes across one). */
export function unicodeNormaliser(form: NormalForm = 'NFKC'): Normaliser {
  return { type: 'unicode', form }
}

/** Lower case, per code point ("Σ" → "σ" everywhere, as Hugging Face's `Lowercase`). */
export function lowercaseNormaliser(): Normaliser {
  return { type: 'lowercase' }
}

/** Full Unicode case folding ("ß" → "ss"). */
export function caseFoldNormaliser(): Normaliser {
  return { type: 'caseFold' }
}

/** Remove non-spacing marks (category Mn); after NFD or NFKD this strips accents. */
export function stripAccentsNormaliser(): Normaliser {
  return { type: 'stripAccents' }
}

/** Replace every match of a regular expression (given by its source; Unicode, global) by `content`. */
export function replaceNormaliser(pattern: string, content: string): Normaliser {
  return { type: 'replace', pattern, content }
}

/** Put `content` before any non-empty text (SentencePiece's leading "▁", as Hugging Face's `Prepend`). */
export function prependNormaliser(content: string): Normaliser {
  return { type: 'prepend', content }
}

/** Remove white space at the start, the end or both. */
export function stripNormaliser(left = true, right = true): Normaliser {
  return { type: 'strip', left, right }
}

/** Collapse every white-space run to one space and trim. */
export function collapseWhitespaceNormaliser(): Normaliser {
  return { type: 'collapseWhitespace' }
}

/** Normalisers applied in order. */
export function normaliserSequence(...normalisers: Normaliser[]): Normaliser {
  return { type: 'sequence', normalisers }
}

function strip(a: AlignedText, left: boolean, right: boolean): AlignedText {
  let s = 0
  let e = a.text.length
  if (left) while (s < e && /\s/u.test(a.text[s])) s++
  if (right) while (e > s && /\s/u.test(a.text[e - 1])) e--
  return alignedSlice(a, s, e)
}

/** Apply a normaliser to aligned text. */
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
