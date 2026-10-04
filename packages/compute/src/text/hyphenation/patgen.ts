/**
 * Learning hyphenation patterns from a hyphenated word list, as Liang's PATGEN does (Liang 1983, ch. 4; Liang and
 * Breitenlohner, "PATGEN", 1991). Patterns are learned in levels. Level 1 (odd) adds hyphenating patterns, level 2
 * (even) inhibiting ones that undo level 1's errors, level 3 hyphenating again, and so on. A level runs one pass per
 * pattern length. In a pass, every substring of every dotted training word, with a digit at one of its slots, is a
 * candidate. At an odd level k a candidate is *good* at each gap where it would put a missing hyphen (the gap is a
 * dictionary hyphen and its current value is even) and *bad* at each gap where it would put a wrong one (not a hyphen,
 * value even). At an even level it is good where it would remove a wrong hyphen and bad where it would remove a right
 * one. A candidate is kept, with digit k, when good·goodWeight − bad·badWeight ≥ threshold. High thresholds keep
 * only patterns that are nearly always right, so the levels trade coverage against errors.
 *
 * Gaps outside the margins (`leftMin`, `rightMin`) are neither counted nor hyphenated. Each step of `patgenSteps` is
 * one pass; its state carries the patterns so far and the training counts of hyphens found (tp), wrong (fp) and
 * missed (fn).
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { formatPattern, hyphenationPatterns, patternSlots, type HyphenationPatterns, type LiangOptions } from './liang'

/** A hyphenated training word: its letters and the gaps (after letter i) that take a hyphen. */
export interface HyphenatedWord {
  readonly word: string
  readonly hyphens: readonly number[]
}

/** The selection rule of one level: keep a candidate when good·goodWeight − bad·badWeight ≥ threshold. */
export interface PatgenLevel {
  readonly goodWeight: number
  readonly badWeight: number
  readonly threshold: number
  /** The shortest and longest pattern (letters, counting `.`) of the level's passes. */
  readonly lengths: readonly [number, number]
}

/** Options of `patgenSteps`. */
export interface PatgenOptions extends LiangOptions {
  /** One entry per level (default `PATGEN_LEVELS`). */
  levels?: readonly PatgenLevel[]
  /** Stop adding patterns once the set holds this many (default no limit): a pass keeps its best candidates. */
  maxPatterns?: number
}

/**
 * Default levels for a few thousand English words, in the spirit of the parameters used for TeX's US English patterns
 * (Liang 1983, table 3): a cautious first hyphenating level, a permissive inhibiting level, then two finer ones.
 */
export const PATGEN_LEVELS: readonly PatgenLevel[] = [
  { goodWeight: 1, badWeight: 2, threshold: 6, lengths: [2, 4] },
  { goodWeight: 2, badWeight: 1, threshold: 4, lengths: [2, 5] },
  { goodWeight: 1, badWeight: 3, threshold: 3, lengths: [3, 6] },
  { goodWeight: 3, badWeight: 1, threshold: 3, lengths: [3, 6] },
  { goodWeight: 1, badWeight: 4, threshold: 2, lengths: [4, 7] },
]

/** Hyphens found (tp), wrongly inserted (fp) and missed (fn) over a word list. */
export interface HyphenCounts {
  readonly tp: number
  readonly fp: number
  readonly fn: number
}

/** What one pass did. */
export interface PatgenPass {
  readonly level: number
  readonly length: number
  /** Candidates with good > 0 that were considered. */
  readonly candidates: number
  /** Patterns kept (digit `level`), in TeX notation, best first. */
  readonly added: readonly string[]
  /** Total good and bad counts of the kept patterns. */
  readonly good: number
  readonly bad: number
}

/** The state of `patgenSteps` after t passes. */
export interface PatgenState extends Status {
  readonly patterns: HyphenationPatterns
  /** The pass the last step ran (null at step 0). */
  readonly pass: PatgenPass | null
  /** Every pass so far. */
  readonly passes: readonly PatgenPass[]
  /** The level and length of the next pass. */
  readonly next: { readonly level: number; readonly length: number } | null
  /** Training counts under the current patterns. */
  readonly counts: HyphenCounts
  readonly done: boolean
}

/** Hyphen counts of a pattern set over hyphenated words, within the margins. */
export function hyphenCounts(
  set: HyphenationPatterns,
  words: readonly HyphenatedWord[],
  options: LiangOptions = {},
): HyphenCounts {
  let tp = 0
  let fp = 0
  let fn = 0
  for (const w of words) {
    const { slots } = patternSlots(set, w.word)
    const truth = new Set(w.hyphens)
    forGaps(w.word.length, options, (i) => {
      const on = slots[i + 2] % 2 === 1
      if (on && truth.has(i)) tp++
      else if (on) fp++
      else if (truth.has(i)) fn++
    })
  }
  return { tp, fp, fn }
}

function forGaps(n: number, { leftMin = 2, rightMin = 3 }: LiangOptions, f: (gap: number) => void) {
  for (let i = Math.max(0, leftMin - 1); i <= n - 1 - rightMin; i++) f(i)
}

/** The passes of a schedule in order: each level's lengths, shortest first. */
function schedule(levels: readonly PatgenLevel[]): { level: number; length: number }[] {
  const out: { level: number; length: number }[] = []
  levels.forEach((l, k) => {
    for (let len = l.lengths[0]; len <= l.lengths[1]; len++) out.push({ level: k + 1, length: len })
  })
  return out
}

/**
 * PATGEN as a traceable algorithm (see the module comment): step 0 has no patterns; each step runs the next pass
 * (one level and pattern length) over the training words and adds the candidates its level keeps. Finishes after the
 * last pass of the last level, or when `maxPatterns` is reached.
 */
export function patgenSteps(
  words: readonly HyphenatedWord[],
  options: PatgenOptions = {},
): Algorithm<void, PatgenState> {
  const { levels = PATGEN_LEVELS, maxPatterns = Infinity } = options
  const margins: LiangOptions = { leftMin: options.leftMin ?? 2, rightMin: options.rightMin ?? 3 }
  if (levels.length === 0) throw new DomainError('patgenSteps', 'patgenSteps: needs at least one level')
  for (const w of words)
    if (!/^[a-z]+$/.test(w.word)) throw new DomainError('patgenSteps', `patgenSteps: '${w.word}' is not a–z`)
  const passes = schedule(levels)
  const truths = words.map((w) => new Set(w.hyphens))
  return {
    name: 'patgen',
    init: () => {
      const patterns = hyphenationPatterns([])
      return {
        t: 0,
        patterns,
        pass: null,
        passes: [],
        next: passes[0],
        counts: hyphenCounts(patterns, words, margins),
        done: false,
      }
    },
    step: (s) => {
      if (s.done || !s.next) return { ...s, t: s.t + 1, pass: null, done: true }
      const { level, length } = s.next
      const rule = levels[level - 1]
      const hyphenating = level % 2 === 1
      // Count each candidate (letters + the slot of its digit) where the level would fix or break a gap.
      const good = new Map<string, number>()
      const bad = new Map<string, number>()
      words.forEach((w, j) => {
        const dotted = `.${w.word}.`
        const { slots } = patternSlots(s.patterns, w.word)
        forGaps(w.word.length, margins, (i) => {
          const k = i + 2
          const on = slots[k] % 2 === 1
          const isHyphen = truths[j].has(i)
          // Odd levels act on gaps without a hyphen (value even), even levels on gaps with one.
          if (on === hyphenating) return
          const table = isHyphen === hyphenating ? good : bad
          for (let d = 0; d <= length; d++) {
            const start = k - d
            if (start < 0 || start + length > dotted.length) continue
            const key = `${dotted.slice(start, start + length)}|${d}`
            table.set(key, (table.get(key) ?? 0) + 1)
          }
        })
      })
      const room = maxPatterns - s.patterns.patterns.length
      const scored = [...good]
        .map(([key, g]) => ({ key, g, b: bad.get(key) ?? 0 }))
        .map((c) => ({ ...c, score: c.g * rule.goodWeight - c.b * rule.badWeight }))
        .filter((c) => c.score >= rule.threshold)
        .sort((a, b) => b.score - a.score || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0))
        .slice(0, Math.max(0, room))
      const added = scored.map(({ key }) => {
        const [letters, d] = key.split('|')
        const values = new Array<number>(letters.length + 1).fill(0)
        values[Number(d)] = level
        return formatPattern(letters, values)
      })
      // Merge into the set: a pattern with the same letters keeps the larger digit in each slot.
      const patterns = hyphenationPatterns([...s.patterns.patterns, ...added])
      const pass: PatgenPass = {
        level,
        length,
        candidates: good.size,
        added,
        good: scored.reduce((a, c) => a + c.g, 0),
        bad: scored.reduce((a, c) => a + c.b, 0),
      }
      const at = s.t + 1
      const full = patterns.patterns.length >= maxPatterns
      return {
        t: at,
        patterns,
        pass,
        passes: [...s.passes, pass],
        next: full ? null : (passes[at] ?? null),
        counts: hyphenCounts(patterns, words, margins),
        done: full || at >= passes.length,
      }
    },
    done: (s) => s.done,
  }
}
