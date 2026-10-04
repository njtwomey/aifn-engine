/**
 * Text normalisation: Unicode normal forms (UAX #15), full case folding (Unicode CaseFolding.txt), accent stripping
 * and whitespace. Each is a map from string to string; `normalise` composes them in a fixed order.
 */

import { FOLDING_EXCEPTIONS } from './casefold'

/** A Unicode normalisation form, or `none` to leave code points as they are. */
export type UnicodeForm = 'NFC' | 'NFD' | 'NFKC' | 'NFKD' | 'none'

let folding: Map<string, string> | null = null

function foldingTable(): Map<string, string> {
  if (folding) return folding
  folding = new Map()
  for (const entry of FOLDING_EXCEPTIONS.split(' ')) {
    const [from, to] = entry.split(':')
    folding.set(
      String.fromCodePoint(parseInt(from, 16)),
      String.fromCodePoint(...to.split('.').map((h) => parseInt(h, 16))),
    )
  }
  return folding
}

/**
 * Unicode full case folding, the caseless-matching map of the Unicode standard (as Python's `str.casefold`): lower case,
 * except where folding differs, e.g. "ß" → "ss", final "ς" → "σ", "ﬁ" → "fi". Two strings match caselessly when their
 * foldings are equal. Language-specific rules (Turkish dotless i) are not applied.
 */
export function caseFold(text: string): string {
  const table = foldingTable()
  let out = ''
  for (const c of text) out += table.get(c) ?? c.toLowerCase()
  return out
}

/**
 * Remove accents: decompose (NFKD) and drop every combining mark (general category M), so "café" → "cafe" and
 * "ﬁancée" → "fiancee" (as scikit-learn's `strip_accents='unicode'`). Letters without a decomposition ("ø", "ł") stay.
 */
export function stripAccents(text: string): string {
  return text.normalize('NFKD').replace(/\p{M}+/gu, '')
}

/** Collapse every run of Unicode white space to one ASCII space and trim both ends. */
export function collapseWhitespace(text: string): string {
  return text.replace(/\s+/gu, ' ').trim()
}

/** Options of {@link normalise}. */
export interface NormaliseOptions {
  /** The Unicode normal form applied first (default NFKC, which also folds ligatures, widths and superscripts). */
  form?: UnicodeForm
  /** Apply full case folding (default true). */
  caseFold?: boolean
  /** Remove combining marks after decomposition (default false). */
  stripAccents?: boolean
  /** Collapse white space runs and trim (default true). */
  whitespace?: boolean
}

/**
 * Normalise text for matching, in the order normal form → case folding → accent stripping → white space. The default,
 * NFKC then case folding, is the `NFKC_Casefold`-style key used for caseless identifier matching (UAX #15, UTS #39).
 */
export function normalise(text: string, options: NormaliseOptions = {}): string {
  const { form = 'NFKC', caseFold: fold = true, stripAccents: strip = false, whitespace = true } = options
  let s = form === 'none' ? text : text.normalize(form)
  if (fold) s = caseFold(s)
  if (strip) s = stripAccents(s)
  // Folding can leave a non-normalised sequence (e.g. "ǰ" → "j" + U+030C); recompose for the composed forms.
  if (fold && !strip && (form === 'NFC' || form === 'NFKC')) s = s.normalize('NFC')
  if (whitespace) s = collapseWhitespace(s)
  return s
}
