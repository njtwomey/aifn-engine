/**
 * A tokeniser for casual text and social media (Christopher Potts's "happyfuntokenizing", as NLTK's `TweetTokenizer`):
 * one regular expression whose alternatives are tried in order, URLs, phone numbers, emoticons, HTML tags, arrows,
 * @-handles, hashtags, e-mail addresses, emoji sequences (skin tones, zero-width joins, flags), then words with inner
 * apostrophes or hyphens, numbers and fractions, ellipses and any other non-space character. HTML entities are
 * decoded first, and a run of four or more of one non-alphanumeric character is shortened to three.
 */

import { fromData } from 'aifn-compute/foundation/tensor'
import { aligned, alignedReplace, originalSpan, type AlignedText } from '../aligned'
import type { Tokenisation } from './tokenise'

// The `regex` module's Unicode \w (UTS #18): alphabetic, marks, decimal digits, connector punctuation, joiners.
const WORD = '\\p{Alphabetic}\\p{M}\\p{Nd}\\p{Pc}\\p{Join_Control}'
const W = `[${WORD}]`
// [^\W\d_]: a word character that is neither a digit nor the underscore.
const LETTER = '(?:(?!_)[\\p{Alphabetic}\\p{M}\\p{Pc}\\p{Join_Control}])'
const D = '\\p{Nd}'

const URLS =
  '(?:https?:(?:\\/{1,3}|[a-z0-9%])|[a-z0-9.\\-]+[.](?:[a-z]{2,13})\\/)' +
  '(?:[^\\s()<>{}\\[\\]]+|\\([^\\s()]*?\\([^\\s()]+\\)[^\\s()]*?\\)|\\([^\\s]+?\\))+' +
  '(?:\\([^\\s()]*?\\([^\\s()]+\\)[^\\s()]*?\\)|\\([^\\s]+?\\)|[^\\s`!()\\[\\]{};:\'".,<>?«»“”‘’])' +
  `|(?:(?<!@)[a-z0-9]+(?:[.\\-][a-z0-9]+)*[.](?:[a-z]{2,13})(?!${W})\\/?(?!@))`

const PHONE = `(?:(?:\\+?[01][ *\\-.)]*)?(?:[(]?${D}{3}[ *\\-.)]*)?${D}{3}[ *\\-.)]*${D}{4})`

const EMOTICONS = "(?:[<>]?[:;=8][\\-o*']?[)\\](\\[dDpP/:}{@|\\\\]|[)\\](\\[dDpP/:}{@|\\\\][\\-o*']?[:;=8][<>]?|<\\/?3)"

const TONE = '[\\u{1F3FB}-\\u{1F3FF}]'
const EMOJI = `.(?:${TONE}?(?:\\u200d.${TONE}?)+|${TONE})`
const FLAGS =
  '(?:[\\u{1F1E6}-\\u{1F1FF}]{2}' +
  '|\\u{1F3F4}\\u{E0067}\\u{E0062}\\u{E0065}\\u{E006E}\\u{E0067}\\u{E007F}' +
  '|\\u{1F3F4}\\u{E0067}\\u{E0062}\\u{E0073}\\u{E0063}\\u{E0074}\\u{E007F}' +
  '|\\u{1F3F4}\\u{E0067}\\u{E0062}\\u{E0077}\\u{E006C}\\u{E0073}\\u{E007F})'

const WORDS =
  `(?:${LETTER}(?:${LETTER}|['\\-_])+${LETTER})` +
  `|(?:[+\\-]?${D}+[,/.:-]${D}+[+\\-]?)` +
  `|(?:[${WORD}_]+)` +
  '|(?:\\.(?:\\s*\\.){1,})' +
  '|(?:\\S)'

/**
 * The alternatives of the tokeniser's regular expression, in the order they are tried.
 *
 * @param phone Whether the phone-number alternative is included (after URLs, before emoticons).
 * @returns The alternatives as regular-expression sources, to be joined with `|`.
 */
const PARTS = (phone: boolean) => [
  URLS,
  ...(phone ? [PHONE] : []),
  EMOTICONS,
  '<[^>\\s]+>',
  '[\\-]+>|<[\\-]+',
  `(?:@[${WORD}_]+)`,
  `(?:#+[${WORD}_]+[${WORD}'_\\-]*[${WORD}_]+)`,
  `[${WORD}.+\\-]+@[${WORD}\\-]+\\.(?:[${WORD}\\-]\\.?)+[${WORD}\\-]`,
  EMOJI,
  FLAGS,
  WORDS,
]

const WORD_RE = new RegExp(`(?:${PARTS(false).join('|')})`, 'giu')
const PHONE_WORD_RE = new RegExp(`(?:${PARTS(true).join('|')})`, 'giu')
const EMOTICON_RE = new RegExp(EMOTICONS, 'iu')
const HANG_RE = /([^a-zA-Z0-9])\1{3,}/gu
const LENGTHENING_RE = /(.)\1{2,}/gu
const ENTITY_RE = /&(#?(x?))([^&;\s]+);/gu
const HANDLES_RE = /(?<![A-Za-z0-9_!@#$%&*])@(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))/gu

// HTML 4 named entities (Python's html.entities.name2codepoint, as NLTK decodes them): name:hex code point.
const ENTITIES =
  'AElig:c6 Aacute:c1 Acirc:c2 Agrave:c0 Alpha:391 Aring:c5 Atilde:c3 Auml:c4 Beta:392 Ccedil:c7 Chi:3a7 ' +
  'Dagger:2021 Delta:394 ETH:d0 Eacute:c9 Ecirc:ca Egrave:c8 Epsilon:395 Eta:397 Euml:cb Gamma:393 Iacute:cd ' +
  'Icirc:ce Igrave:cc Iota:399 Iuml:cf Kappa:39a Lambda:39b Mu:39c Ntilde:d1 Nu:39d OElig:152 Oacute:d3 ' +
  'Ocirc:d4 Ograve:d2 Omega:3a9 Omicron:39f Oslash:d8 Otilde:d5 Ouml:d6 Phi:3a6 Pi:3a0 Prime:2033 Psi:3a8 ' +
  'Rho:3a1 Scaron:160 Sigma:3a3 THORN:de Tau:3a4 Theta:398 Uacute:da Ucirc:db Ugrave:d9 Upsilon:3a5 Uuml:dc ' +
  'Xi:39e Yacute:dd Yuml:178 Zeta:396 aacute:e1 acirc:e2 acute:b4 aelig:e6 agrave:e0 alefsym:2135 alpha:3b1 ' +
  'amp:26 and:2227 ang:2220 aring:e5 asymp:2248 atilde:e3 auml:e4 bdquo:201e beta:3b2 brvbar:a6 bull:2022 ' +
  'cap:2229 ccedil:e7 cedil:b8 cent:a2 chi:3c7 circ:2c6 clubs:2663 cong:2245 copy:a9 crarr:21b5 cup:222a ' +
  'curren:a4 dArr:21d3 dagger:2020 darr:2193 deg:b0 delta:3b4 diams:2666 divide:f7 eacute:e9 ecirc:ea egrave:e8 ' +
  'empty:2205 emsp:2003 ensp:2002 epsilon:3b5 equiv:2261 eta:3b7 eth:f0 euml:eb euro:20ac exist:2203 fnof:192 ' +
  'forall:2200 frac12:bd frac14:bc frac34:be frasl:2044 gamma:3b3 ge:2265 gt:3e hArr:21d4 harr:2194 hearts:2665 ' +
  'hellip:2026 iacute:ed icirc:ee iexcl:a1 igrave:ec image:2111 infin:221e int:222b iota:3b9 iquest:bf ' +
  'isin:2208 iuml:ef kappa:3ba lArr:21d0 lambda:3bb lang:2329 laquo:ab larr:2190 lceil:2308 ldquo:201c le:2264 ' +
  'lfloor:230a lowast:2217 loz:25ca lrm:200e lsaquo:2039 lsquo:2018 lt:3c macr:af mdash:2014 micro:b5 middot:b7 ' +
  'minus:2212 mu:3bc nabla:2207 nbsp:a0 ndash:2013 ne:2260 ni:220b not:ac notin:2209 nsub:2284 ntilde:f1 nu:3bd ' +
  'oacute:f3 ocirc:f4 oelig:153 ograve:f2 oline:203e omega:3c9 omicron:3bf oplus:2295 or:2228 ordf:aa ordm:ba ' +
  'oslash:f8 otilde:f5 otimes:2297 ouml:f6 para:b6 part:2202 permil:2030 perp:22a5 phi:3c6 pi:3c0 piv:3d6 ' +
  'plusmn:b1 pound:a3 prime:2032 prod:220f prop:221d psi:3c8 quot:22 rArr:21d2 radic:221a rang:232a raquo:bb ' +
  'rarr:2192 rceil:2309 rdquo:201d real:211c reg:ae rfloor:230b rho:3c1 rlm:200f rsaquo:203a rsquo:2019 ' +
  'sbquo:201a scaron:161 sdot:22c5 sect:a7 shy:ad sigma:3c3 sigmaf:3c2 sim:223c spades:2660 sub:2282 sube:2286 ' +
  'sum:2211 sup:2283 sup1:b9 sup2:b2 sup3:b3 supe:2287 szlig:df tau:3c4 there4:2234 theta:3b8 thetasym:3d1 ' +
  'thinsp:2009 thorn:fe tilde:2dc times:d7 trade:2122 uArr:21d1 uacute:fa uarr:2191 ucirc:fb ugrave:f9 uml:a8 ' +
  'upsih:3d2 upsilon:3c5 uuml:fc weierp:2118 xi:3be yacute:fd yen:a5 yuml:ff zeta:3b6 zwj:200d zwnj:200c'

let entities: Map<string, number> | null = null

/**
 * The text an HTML entity stands for, as NLTK decodes it: a named entity of HTML 4, or a decimal or hexadecimal
 * reference (0x80 to 0x9F read as Windows-1252). An entity that names nothing, or no valid code point, decodes to the
 * empty string.
 *
 * @param m A match of `ENTITY_RE`: group 1 is `#` or `#x` for a numeric reference (empty for a name), group 2 the `x`
 *   and group 3 the name or digits.
 * @returns The decoded character, or the empty string.
 */
function entity(m: RegExpExecArray): string {
  entities ??= new Map(
    ENTITIES.split(' ').map((e) => {
      const [k, v] = e.split(':')
      return [k, parseInt(v, 16)] as [string, number]
    }),
  )
  const body = m[3]
  let n: number | undefined
  if (m[1]) {
    n = parseInt(body, m[2] ? 16 : 10)
    const valid = m[2] ? /^[0-9a-f]+$/iu.test(body) : /^[0-9]+$/u.test(body)
    if (!valid) n = undefined
    // Numeric references 0x80–0x9F are Windows-1252 bytes, as browsers read them.
    else if (n >= 0x80 && n <= 0x9f) return new TextDecoder('windows-1252').decode(Uint8Array.of(n))
  } else n = entities.get(body)
  if (n !== undefined && n >= 0 && n <= 0x10ffff && !(n >= 0xd800 && n <= 0xdfff)) return String.fromCodePoint(n)
  // An entity that names nothing is removed.
  return ''
}

/** Options of {@link casualTokenise}. */
export interface CasualOptions {
  /** Keep case (default true); otherwise every token but an emoticon is lower-cased. */
  preserveCase?: boolean
  /** Shorten every run of three or more of one character to three ("waaaaay" becomes "waaay"; default false). */
  reduceLength?: boolean
  /** Remove @-handles (default false). */
  stripHandles?: boolean
  /** Match phone numbers as one token (default true). */
  phoneNumbers?: boolean
}

/**
 * Tokenise casual text (NLTK's `TweetTokenizer`), with offsets into the original text: a token produced from decoded
 * entities or shortened runs points at the whole stretch of text it came from.
 *
 * @param text The text to tokenise; HTML entities in it are decoded first.
 * @param options Case, length reduction, handle stripping and phone numbers; see {@link CasualOptions}.
 * @returns The tokens (decoded and, with the options, lower-cased or shortened), with offsets into `text`.
 *
 * @example NLTK's example sentence
 * print(casualTokenise('This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--').tokens)
 *
 * @example Handles stripped, lengthening reduced, case folded
 * const t = casualTokenise('@remy This is waaaaayyyy too much for you!!!!!! &amp; :-D', {
 *   stripHandles: true,
 *   reduceLength: true,
 *   preserveCase: false,
 * })
 * print(t.tokens)
 * const k = t.tokens.indexOf('&')
 * print('"&" comes from', JSON.stringify(t.source.slice(t.offsets.data[2 * k], t.offsets.data[2 * k + 1])))
 */
export function casualTokenise(text: string, options: CasualOptions = {}): Tokenisation {
  const { preserveCase = true, reduceLength = false, stripHandles = false, phoneNumbers = true } = options
  let a: AlignedText = alignedReplace(aligned(text), ENTITY_RE, entity)
  if (stripHandles) a = alignedReplace(a, HANDLES_RE, ' ')
  if (reduceLength) a = alignedReplace(a, LENGTHENING_RE, '$1$1$1')
  a = alignedReplace(a, HANG_RE, '$1$1$1')
  const re = new RegExp((phoneNumbers ? PHONE_WORD_RE : WORD_RE).source, 'giu')
  const tokens: string[] = []
  const offsets: number[] = []
  for (const m of a.text.matchAll(re)) {
    if (m[0].length === 0) continue
    const tok = preserveCase || EMOTICON_RE.test(m[0]) ? m[0] : m[0].toLowerCase()
    tokens.push(tok)
    offsets.push(...originalSpan(a, m.index, m.index + m[0].length))
  }
  return { kind: 'tokens', source: text, tokens, offsets: fromData(Int32Array.from(offsets), [tokens.length, 2]) }
}
