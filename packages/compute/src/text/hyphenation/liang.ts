/**
 * Liang's pattern hyphenation (Liang 1983, "Word Hy-phen-a-tion by Com-put-er", Stanford PhD thesis; the algorithm of
 * TeX, Knuth 1984, "The TeXbook", appendix H). A pattern is a string of letters with a digit (a priority) at some of
 * the gaps between them, e.g. `hen5at` or `.hy3ph`, where `.` matches a word boundary. To hyphenate a word, write it as
 * `.word.`, find every pattern that occurs in it, and give each inter-letter gap the largest digit any matching
 * pattern puts there. An odd value allows a hyphen and an even value forbids one, so later, higher levels of patterns
 * add hyphens (odd) and inhibit them (even) in turn. Hyphens closer to either end than `leftMin` and `rightMin` letters
 * are never inserted (TeX's \lefthyphenmin = 2, \righthyphenmin = 3).
 *
 * Positions: a word of n letters has gaps 0 … n − 2, gap i sitting after letter i. `hyphens` lists the gaps that get a
 * hyphen. In the dotted word `.word.` (length n + 2) the value slots are 0 … n + 2, slot k lying before character k; gap
 * i of the word is slot i + 2.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import type { Algorithm } from 'aifn-compute/foundation/trace'

/** A set of hyphenation patterns in TeX's notation, e.g. `['.hy3ph', 'he2n', 'hen5at']`. */
export interface HyphenationPatterns {
  readonly kind: 'hyphenation-patterns'
  readonly patterns: readonly string[]
}

/** One parsed pattern: its letters (with `.` for a boundary) and the digit in each of its letters.length + 1 slots. */
export interface ParsedPattern {
  readonly letters: string
  readonly values: readonly number[]
}

/** Options of `liangHyphenate` and `liangSteps`. */
export interface LiangOptions {
  /** The fewest letters before a hyphen (default 2, TeX's \lefthyphenmin). */
  leftMin?: number
  /** The fewest letters after a hyphen (default 3, TeX's \righthyphenmin). */
  rightMin?: number
}

/** A pattern that matched: the pattern as written, where its first letter sits in the dotted word, and its digits. */
export interface PatternMatch {
  readonly pattern: string
  /** Index in `.word.` of the pattern's first character. */
  readonly at: number
  readonly letters: string
  readonly values: readonly number[]
}

/** The result of hyphenating one word. */
export interface Hyphenation {
  readonly word: string
  /** The value of each gap, gap i after letter i (length n − 1): the largest digit any matching pattern puts there. */
  readonly gaps: readonly number[]
  /** The gaps with an odd value inside the margins: a hyphen goes after these letters. */
  readonly hyphens: readonly number[]
  /** Every pattern that matched, in order of position. */
  readonly matches: readonly PatternMatch[]
}

const DIGIT = /[0-9]/

/** Parse one TeX pattern: `hen5at` → letters `henat`, values [0, 0, 0, 5, 0, 0]. */
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

/** Write a pattern in TeX's notation (zeros left out): `henat`, [0, 0, 0, 5, 0, 0] → `hen5at`. */
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
 * letters are merged, keeping the larger digit in each slot.
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

/** The slots of `.word.` under every pattern of a set, and the matches (for `liangHyphenate` and `patgenSteps`). */
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
  readonly word: string
  /** `.word.` */
  readonly dotted: string
  /** The value of each slot of the dotted word so far (length n + 3). */
  readonly slots: readonly number[]
  /** The patterns matched by the last step (those starting at index t − 1). */
  readonly fired: readonly PatternMatch[]
  /** Every pattern matched so far. */
  readonly matches: readonly PatternMatch[]
  readonly done: boolean
}

function checkWord(word: string, caller: string): string {
  const w = word.toLowerCase()
  if (!/^[a-z]+$/.test(w)) throw new DomainError(caller, `${caller}: '${word}' is not a word of the letters a–z`)
  return w
}

/**
 * Liang's matching on one word as a traceable algorithm: step t applies every pattern that starts at index t − 1 of
 * `.word.`, raising each slot it covers to the pattern's digit there. After n + 2 steps every pattern has been applied
 * and `liangResult` reads off the hyphens.
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

/** The gaps and hyphens of a (finished or partial) matching state. */
export function liangResult(state: LiangState, options: LiangOptions = {}): Hyphenation {
  return readHyphens(state.word, state.slots, state.matches, options)
}

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
 * Hyphenate one word with a pattern set (see the module comment): the state `liangSteps` ends on, computed in one
 * pass.
 */
export function liangHyphenate(set: HyphenationPatterns, word: string, options: LiangOptions = {}): Hyphenation {
  const w = checkWord(word, 'liangHyphenate')
  const { slots, matches } = patternSlots(set, w)
  return readHyphens(w, slots, matches, options)
}

/** Insert `mark` (default "-") after each letter listed in `hyphens`: ("hyphenation", [1, 5]) → "hy-phen-ation". */
export function markHyphens(word: string, hyphens: readonly number[], mark = '-'): string {
  const at = new Set(hyphens)
  let out = ''
  for (let i = 0; i < word.length; i++) out += word[i] + (at.has(i) && i < word.length - 1 ? mark : '')
  return out
}

/** The word and its hyphens from a marked form: "hy-phen-ation" → { word: "hyphenation", hyphens: [1, 5] }. */
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
