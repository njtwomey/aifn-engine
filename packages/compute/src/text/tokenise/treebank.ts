/**
 * The Penn Treebank word tokeniser (Marcus, Santorini & Marcinkiewicz 1993; the sed script of Robert MacIntyre), as
 * NLTK's `TreebankWordTokenizer` applies it: a fixed sequence of regular-expression substitutions that pad
 * punctuation with spaces, turn double quotes into `` and '', split clitics ("don't" → "do n't", "they'll" → "they
 * 'll") and a handful of fused forms ("cannot" → "can not", "gonna" → "gon na"), then split on white space.
 */

import { fromData } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import type { Tokenisation } from './tokenise'

// Python's `re` is Unicode-aware for \w, \d and \b on str patterns; JavaScript's are ASCII even under `u`, so the
// patterns spell the Unicode classes out (Python's \w is a letter, a number or the underscore).
const W = '[\\p{L}\\p{N}_]'
const before = `(?<!${W})`
const after = `(?!${W})`

type Rule = [RegExp, string]

const STARTING_QUOTES: Rule[] = [
  [/^"/u, '``'],
  [/(``)/gu, ' $1 '],
  [/([ ([{<])("|'{2})/gu, '$1 `` '],
]

const PUNCTUATION: Rule[] = [
  [/([:,])([^\p{Nd}])/gu, ' $1 $2'],
  // Python's `$` also matches before a final newline.
  [/([:,])(?=\n?$)/gu, ' $1 '],
  [/\.\.\./gu, ' ... '],
  [/[;@#$%&]/gu, ' $& '],
  // The final period, with any closing brackets or quotes after it.
  [/([^.])(\.)([\])}>"']*)\s*$/gu, '$1 $2$3 '],
  [/[?!]/gu, ' $& '],
  [/([^'])' /gu, "$1 ' "],
]

const PARENS_BRACKETS: Rule = [/[\][(){}<>]/gu, ' $& ']

const CONVERT_PARENTHESES: Rule[] = [
  [/\(/gu, '-LRB-'],
  [/\)/gu, '-RRB-'],
  [/\[/gu, '-LSB-'],
  [/\]/gu, '-RSB-'],
  [/\{/gu, '-LCB-'],
  [/\}/gu, '-RCB-'],
]

const DOUBLE_DASHES: Rule = [/--/gu, ' -- ']

const ENDING_QUOTES: Rule[] = [
  [/''/gu, " '' "],
  [/"/gu, " '' "],
  [/([^' ])('[sS]|'[mM]|'[dD]|') /gu, '$1 $2 '],
  [/([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) /gu, '$1 $2 '],
]

// MacIntyre's contractions, case-insensitive.
const CONTRACTIONS2: RegExp[] = [
  ['can', 'not'],
  ['d', "'ye"],
  ['gim', 'me'],
  ['gon', 'na'],
  ['got', 'ta'],
  ['lem', 'me'],
  ['more', "'n"],
].map(([a, b]) => new RegExp(`${before}(${a})(${b})${after}`, 'giu'))
CONTRACTIONS2.push(new RegExp(`${before}(wan)(na)(?=\\s)`, 'giu'))
const CONTRACTIONS3: RegExp[] = [new RegExp(` ('t)(is)${after}`, 'giu'), new RegExp(` ('t)(was)${after}`, 'giu')]

/** Options of {@link treebankTokens} and {@link treebankTokenise}. */
export interface TreebankOptions {
  /** Replace brackets by the Treebank symbols -LRB-, -RRB-, -LSB-, -RSB-, -LCB-, -RCB- (default false). */
  convertParentheses?: boolean
}

const apply = (text: string, rules: readonly Rule[]) => rules.reduce((t, [re, s]) => t.replace(re, s), text)

/**
 * The Treebank tokens of `text`, as NLTK's `TreebankWordTokenizer().tokenize`: double quotes become `` (opening) and
 * '' (closing), so tokens are not always substrings of the text. Periods are split off only at the end of the text
 * (the tokeniser expects one sentence; "York." mid-text stays whole).
 */
export function treebankTokens(text: string, options: TreebankOptions = {}): string[] {
  let t = apply(text, STARTING_QUOTES)
  t = apply(t, PUNCTUATION)
  t = apply(t, [PARENS_BRACKETS])
  if (options.convertParentheses) t = apply(t, CONVERT_PARENTHESES)
  t = apply(t, [DOUBLE_DASHES])
  t = ` ${t} `
  t = apply(t, ENDING_QUOTES)
  for (const re of CONTRACTIONS2) t = t.replace(re, ' $1 $2 ')
  for (const re of CONTRACTIONS3) t = t.replace(re, ' $1 $2 ')
  // Python's str.split() also splits on the information separators U+001C–U+001F.
  // oxlint-disable-next-line no-control-regex -- the information separators are white space to Python
  return t.split(/[\s\x1c-\x1f]+/u).filter((x) => x.length > 0)
}

/**
 * Treebank tokens with offsets (NLTK's `span_tokenize`): each token is located in the text from where the last one
 * ended, the converted quotes `` and '' standing for the quote they replaced (", `` or ''). The tokens returned are
 * the substrings of the text, so a converted quote reads as the original quote. Bracket conversion is not applied.
 */
export function treebankTokenise(text: string): Tokenisation {
  const raw = treebankTokens(text)
  const quotes = [...text.matchAll(/``|''|"/gu)].map((m) => m[0])
  const tokens = raw.map((t) => ((t === '"' || t === '``' || t === "''") && quotes.length > 0 ? quotes.shift()! : t))
  const out: string[] = []
  const offsets: number[] = []
  let at = 0
  for (const t of tokens) {
    const s = text.indexOf(t, at)
    if (s < 0) throw new DomainError('treebankTokenise', `treebankTokenise: token ${JSON.stringify(t)} not found`)
    out.push(t)
    offsets.push(s, s + t.length)
    at = s + t.length
  }
  return { kind: 'tokens', source: text, tokens: out, offsets: fromData(Int32Array.from(offsets), [out.length, 2]) }
}
