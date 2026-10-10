/**
 * Liang's pattern hyphenation (Liang 1983, "Word Hy-phen-a-tion by Com-put-er", Stanford PhD thesis; the algorithm of
 * TeX, Knuth 1984, "The TeXbook", appendix H). A pattern is a string of letters with a digit (a priority) at some of
 * the gaps between them, e.g. `hen5at` or `.hy3ph`, where `.` matches a word boundary. To hyphenate a word, write it as
 * `.word.`, find every pattern that occurs in it, and give each inter-letter gap the largest digit any matching
 * pattern puts there. An odd value allows a hyphen and an even value forbids one, so later, higher levels of patterns
 * add hyphens (odd) and inhibit them (even) in turn. Hyphens closer to either end than `leftMin` and `rightMin` letters
 * are never inserted (TeX's `\lefthyphenmin` = 2, `\righthyphenmin` = 3).
 *
 * Positions: a word of $n$ letters has gaps $0, \dots, n-2$, gap $i$ sitting after letter $i$ (counting from 0).
 * `hyphens` lists the gaps that get a hyphen. In the dotted word `.word.` (length $n+2$) the value slots are
 * $0, \dots, n+2$, slot $k$ lying before character $k$; gap $i$ of the word is slot $i+2$.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import type { Algorithm } from 'aifn-compute/foundation/trace'

/** A set of hyphenation patterns in TeX's notation, e.g. `['.hy3ph', 'he2n', 'hen5at']`. */
export interface HyphenationPatterns {
  /** The tag `'hyphenation-patterns'`. */
  readonly kind: 'hyphenation-patterns'
  /** The patterns, one per distinct letter string. */
  readonly patterns: readonly string[]
}

/** One parsed pattern: its letters (`.` for a boundary) and the digit in each of its `letters.length + 1` slots. */
export interface ParsedPattern {
  /** The pattern without its digits, `.` standing for a word boundary. */
  readonly letters: string
  /** The digit in each slot, slot $k$ lying before letter $k$ (0 where the pattern has none). */
  readonly values: readonly number[]
}

/** Options of `liangHyphenate` and `liangSteps`. */
export interface LiangOptions {
  /** The fewest letters before a hyphen (default 2, TeX's `\lefthyphenmin`). */
  leftMin?: number
  /** The fewest letters after a hyphen (default 3, TeX's `\righthyphenmin`). */
  rightMin?: number
}

/** A pattern that matched: the pattern as written, where its first letter sits in the dotted word, and its digits. */
export interface PatternMatch {
  /** The pattern in TeX's notation. */
  readonly pattern: string
  /** Index in `.word.` of the pattern's first character. */
  readonly at: number
  /** The pattern's letters, as in {@link ParsedPattern}. */
  readonly letters: string
  /** The pattern's digits, as in {@link ParsedPattern}: `values[k]` applies to slot `at + k` of `.word.`. */
  readonly values: readonly number[]
}

/** The result of hyphenating one word. */
export interface Hyphenation {
  /** The word, lower-cased. */
  readonly word: string
  /**
   * The value of each gap, gap $i$ after letter $i$ ($n-1$ entries): the largest digit any matching pattern puts
   * there.
   */
  readonly gaps: readonly number[]
  /** The gaps with an odd value inside the margins: a hyphen goes after these letters. */
  readonly hyphens: readonly number[]
  /** Every pattern that matched, in order of position. */
  readonly matches: readonly PatternMatch[]
}

const DIGIT = /[0-9]/

/**
 * Parse one TeX pattern: `hen5at` gives letters `henat` and values [0, 0, 0, 5, 0, 0]. Throws `DomainError` for a
 * pattern with no letters.
 *
 * @param pattern The pattern in TeX's notation: letters (and `.` for a boundary) with single digits between them.
 * @returns Its letters and the digit in each of the $\ell + 1$ slots around its $\ell$ letters.
 *
 * @example A hyphenating and a boundary pattern
 * print(parsePattern('hen5at'))
 * print(parsePattern('.hy3ph'))
 */
export function parsePattern(pattern: string): ParsedPattern {
  let letters = ''
  const values: number[] = [0]
  for (const c of pattern) {
    if (DIGIT.test(c)) values[letters.length] = Number(c)
    else {
      letters += c
      values.push(0)
    }
  }
  if (!letters) throw new DomainError('parsePattern', `parsePattern: '${pattern}' has no letters`)
  return { letters, values }
}

/**
 * Write a pattern in TeX's notation (zeros left out): `henat` with [0, 0, 0, 5, 0, 0] gives `hen5at`.
 *
 * @param letters The pattern's letters, `.` for a boundary.
 * @param values The digit of each slot, slot $k$ before letter $k$; `letters.length + 1` entries are read.
 * @returns The pattern as TeX writes it.
 *
 * @example The inverse of `parsePattern`
 * print(formatPattern('henat', [0, 0, 0, 5, 0, 0]))
 * const p = parsePattern('2io')
 * print(formatPattern(p.letters, p.values))
 */
export function formatPattern(letters: string, values: readonly number[]): string {
  let out = ''
  for (let k = 0; k <= letters.length; k++) {
    if (values[k]) out += String(values[k])
    if (k < letters.length) out += letters[k]
  }
  return out
}

/**
 * A pattern set from TeX patterns, one string per pattern or one whitespace-separated string. Patterns with the same
 * letters are merged, keeping the larger digit in each slot. Throws `DomainError` for a pattern with no letters.
 *
 * @param patterns The patterns, as a list or as one string separated by white space (as in TeX's `\patterns`).
 * @returns The set, one pattern per distinct letter string, in order of first appearance.
 *
 * @example Two patterns on the same letters merge
 * print(hyphenationPatterns('hy3ph he2n 1he2n hena4').patterns)
 */
export function hyphenationPatterns(patterns: string | readonly string[]): HyphenationPatterns {
  const list = typeof patterns === 'string' ? patterns.split(/\s+/).filter(Boolean) : patterns
  const merged = new Map<string, number[]>()
  for (const p of list) {
    const { letters, values } = parsePattern(p)
    const prev = merged.get(letters)
    merged.set(letters, prev ? prev.map((v, k) => Math.max(v, values[k])) : [...values])
  }
  return {
    kind: 'hyphenation-patterns',
    patterns: [...merged].map(([letters, values]) => formatPattern(letters, values)),
  }
}

/** The patterns of a set keyed by their letters, and the longest pattern; built once per set. */
type PatternIndex = { byLetters: Map<string, ParsedPattern & { pattern: string }>; longest: number }
const indexes = new WeakMap<HyphenationPatterns, PatternIndex>()

/**
 * The index of a pattern set, built on first use and cached per set object.
 *
 * @param set The pattern set; it must not be modified after its first use.
 * @returns Its patterns keyed by their letters, and the length of the longest.
 */
function indexOf(set: HyphenationPatterns): PatternIndex {
  let index = indexes.get(set)
  if (!index) {
    const byLetters = new Map<string, ParsedPattern & { pattern: string }>()
    let longest = 0
    for (const pattern of set.patterns) {
      const parsed = parsePattern(pattern)
      byLetters.set(parsed.letters, { ...parsed, pattern })
      longest = Math.max(longest, parsed.letters.length)
    }
    index = { byLetters, longest }
    indexes.set(set, index)
  }
  return index
}

/**
 * Apply every pattern that matches `.word.` starting at index `start` (shortest first): raise each slot it covers to
 * the pattern's digit there, in place, and return the matches. The one matching rule of `liangSteps`,
 * `liangHyphenate` and `patgenSteps`.
 *
 * @param index The pattern set's index.
 * @param dotted The word with boundary dots, `.word.`.
 * @param start The index in `dotted` where the patterns must start.
 * @param slots The value of each slot of `dotted` ($n+3$ entries); raised in place.
 * @returns The patterns that matched at `start`, shortest first.
 */
function applyAt(index: PatternIndex, dotted: string, start: number, slots: number[]): PatternMatch[] {
  const out: PatternMatch[] = []
  for (let len = 1; len <= index.longest && start + len <= dotted.length; len++) {
    const p = index.byLetters.get(dotted.slice(start, start + len))
    if (!p) continue
    p.values.forEach((v, k) => (slots[start + k] = Math.max(slots[start + k], v)))
    out.push({ pattern: p.pattern, at: start, letters: p.letters, values: p.values })
  }
  return out
}

/**
 * The slots of `.word.` under every pattern of a set, and the matches (for `liangHyphenate` and `patgenSteps`). The
 * word is used as given: no lower-casing and no check of its letters.
 *
 * @param set The pattern set.
 * @param word The word, without dots.
 * @returns `slots`, the value of each of the $n+3$ slots of `.word.` (slot $k$ before character $k$, so gap $i$ of the
 *   word is slot $i+2$), and `matches`, every pattern that matched, by position.
 *
 * @example The slots of "hyphenation"
 * const set = hyphenationPatterns('hy3ph he2n hena4 hen5at 1na n2at 1tio 2io o2n')
 * print(patternSlots(set, 'hyphenation').slots)
 */
export function patternSlots(set: HyphenationPatterns, word: string): { slots: number[]; matches: PatternMatch[] } {
  const index = indexOf(set)
  const dotted = `.${word}.`
  const slots = new Array<number>(dotted.length + 1).fill(0)
  const matches: PatternMatch[] = []
  for (let start = 0; start < dotted.length; start++) matches.push(...applyAt(index, dotted, start, slots))
  return { slots, matches }
}

/** The state of `liangSteps` after t steps: every pattern starting before index t of `.word.` has been applied. */
export interface LiangState extends Status {
  /** The word, lower-cased. */
  readonly word: string
  /** `.word.` */
  readonly dotted: string
  /** The value of each slot of the dotted word so far ($n+3$ entries). */
  readonly slots: readonly number[]
  /** The patterns matched by the last step (those starting at index $t-1$). */
  readonly fired: readonly PatternMatch[]
  /** Every pattern matched so far. */
  readonly matches: readonly PatternMatch[]
  /** True once every start index of `.word.` has been tried. */
  readonly done: boolean
}

/**
 * Lower-case a word and check that it is made of the letters a to z only, throwing `DomainError` if not.
 *
 * @param word The word to check.
 * @param caller The caller's name, for the error message.
 * @returns The word in lower case.
 */
function checkWord(word: string, caller: string): string {
  const w = word.toLowerCase()
  if (!/^[a-z]+$/.test(w)) throw new DomainError(caller, `${caller}: '${word}' is not a word of the letters a–z`)
  return w
}

/**
 * Liang's matching on one word as a traceable algorithm: step $t$ applies every pattern that starts at index $t-1$ of
 * `.word.`, raising each slot it covers to the pattern's digit there. After $n+2$ steps every pattern has been applied
 * and `liangResult` reads off the hyphens. Throws `DomainError` unless the word is made of the letters a to z (in
 * either case).
 *
 * @param set The pattern set.
 * @param word The word to hyphenate; it is lower-cased.
 * @returns The algorithm; it takes no start value.
 *
 * @example The patterns that fire as the scan moves along "hyphenation"
 * const set = hyphenationPatterns('hy3ph he2n hena4 hen5at 1na n2at 1tio 2io o2n')
 * const alg = liangSteps(set, 'hyphenation')
 * for (let t = 1; t <= 13; t++) {
 *   const s = run(alg, undefined, t)
 *   if (s.fired.length) print('step', t, s.dotted[t - 1], s.fired.map((m) => m.pattern))
 * }
 */
export function liangSteps(set: HyphenationPatterns, word: string): Algorithm<void, LiangState> {
  const w = checkWord(word, 'liangSteps')
  const dotted = `.${w}.`
  const index = indexOf(set)
  return {
    name: 'liang',
    init: () => ({
      t: 0,
      word: w,
      dotted,
      slots: new Array<number>(dotted.length + 1).fill(0),
      fired: [],
      matches: [],
      done: false,
    }),
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1, fired: [] }
      const start = s.t
      const slots = [...s.slots]
      const fired = applyAt(index, dotted, start, slots)
      return {
        ...s,
        t: s.t + 1,
        slots,
        fired,
        matches: [...s.matches, ...fired],
        done: start + 1 >= dotted.length,
      }
    },
    done: (s) => s.done,
  }
}

/**
 * The gaps and hyphens of a (finished or partial) matching state.
 *
 * @param state A state of `liangSteps`.
 * @param options The margins; see {@link LiangOptions}.
 * @returns The gaps, hyphens and matches of the patterns applied so far.
 *
 * @example Hyphens appear as the scan proceeds
 * const set = hyphenationPatterns('hy3ph he2n hena4 hen5at 1na n2at 1tio 2io o2n')
 * const alg = liangSteps(set, 'hyphenation')
 * for (const t of [3, 6, 13]) {
 *   const h = liangResult(run(alg, undefined, t))
 *   print('after', t, 'steps:', markHyphens(h.word, h.hyphens))
 * }
 */
export function liangResult(state: LiangState, options: LiangOptions = {}): Hyphenation {
  return readHyphens(state.word, state.slots, state.matches, options)
}

/**
 * Read the gaps and the hyphens from the slots of `.word.`.
 *
 * @param word The word, without dots ($n$ letters).
 * @param slots The value of each slot of `.word.`; gap $i$ is slot $i+2$.
 * @param matches The patterns that matched, passed through to the result.
 * @param options The margins.
 * @param options.leftMin The fewest letters before a hyphen.
 * @param options.rightMin The fewest letters after a hyphen.
 * @returns The hyphenation: gaps with an odd value inside the margins take a hyphen.
 */
function readHyphens(
  word: string,
  slots: readonly number[],
  matches: readonly PatternMatch[],
  { leftMin = 2, rightMin = 3 }: LiangOptions,
): Hyphenation {
  const n = word.length
  const gaps = Array.from({ length: Math.max(0, n - 1) }, (_, i) => slots[i + 2])
  const hyphens: number[] = []
  for (let i = Math.max(0, leftMin - 1); i <= n - 1 - rightMin; i++) if (gaps[i] % 2 === 1) hyphens.push(i)
  return { word, gaps, hyphens, matches }
}

/**
 * Hyphenate one word with a pattern set (see the file comment): the state `liangSteps` ends on, computed in one
 * pass. Throws `DomainError` unless the word is made of the letters a to z (in either case).
 *
 * @param set The pattern set.
 * @param word The word to hyphenate; it is lower-cased.
 * @param options The margins; see {@link LiangOptions}.
 * @returns The gaps' values, the hyphens and the patterns that matched.
 *
 * @example The hyphenation of "hyphenation"
 * const set = hyphenationPatterns('hy3ph he2n hena4 hen5at 1na n2at 1tio 2io o2n')
 * const h = liangHyphenate(set, 'hyphenation')
 * print('gaps', h.gaps)
 * print(markHyphens(h.word, h.hyphens))
 * print('leftMin 3:', markHyphens(h.word, liangHyphenate(set, 'hyphenation', { leftMin: 3 }).hyphens))
 */
export function liangHyphenate(set: HyphenationPatterns, word: string, options: LiangOptions = {}): Hyphenation {
  const w = checkWord(word, 'liangHyphenate')
  const { slots, matches } = patternSlots(set, w)
  return readHyphens(w, slots, matches, options)
}

/**
 * Insert `mark` after each letter listed in `hyphens`: "hyphenation" with [1, 5] gives "hy-phen-ation". A gap after the
 * last letter is ignored.
 *
 * @param word The word.
 * @param hyphens The gaps that take a mark, gap $i$ after letter $i$ (counting from 0).
 * @param mark The string to insert.
 * @returns The marked word.
 *
 * @example Hyphens, or soft hyphens for HTML
 * print(markHyphens('hyphenation', [1, 5]))
 * print(markHyphens('hyphenation', [1, 5], '&shy;'))
 */
export function markHyphens(word: string, hyphens: readonly number[], mark = '-'): string {
  const at = new Set(hyphens)
  let out = ''
  for (let i = 0; i < word.length; i++) out += word[i] + (at.has(i) && i < word.length - 1 ? mark : '')
  return out
}

/**
 * The word and its hyphens from a marked form: "hy-phen-ation" gives the word "hyphenation" and hyphens [1, 5]. A mark
 * before the first letter is ignored.
 *
 * @param marked The word with a mark at each hyphen.
 * @param mark The character that marks a hyphen; it is compared with each code point of `marked`.
 * @returns The word without marks, and the gaps that were marked.
 *
 * @example The inverse of `markHyphens`
 * print(parseHyphenated('hy-phen-ation'))
 * print(parseHyphenated('ta·ble', '·'))
 */
export function parseHyphenated(marked: string, mark = '-'): { word: string; hyphens: number[] } {
  let word = ''
  const hyphens: number[] = []
  for (const c of marked) {
    if (c === mark) {
      if (word.length > 0) hyphens.push(word.length - 1)
    } else word += c
  }
  return { word, hyphens }
}
