/**
 * `aifn-compute/text/hyphenation`: Liang's pattern hyphenation, the algorithm of TeX, and PATGEN's learning of the
 * patterns.
 *
 * - Patterns: `hyphenationPatterns` builds a set from TeX's notation (`hen5at`, `.hy3ph`), merging patterns on the same
 *   letters; `parsePattern` and `formatPattern` convert one pattern to letters and digits and back.
 * - Hyphenating: `liangHyphenate` in one pass, or `liangSteps` (one start position of `.word.` per step) with
 *   `liangResult` to read a state; `patternSlots` gives the raw slot values. An odd value allows a hyphen, an even one
 *   forbids it.
 * - Learning: `patgenSteps` learns patterns from hyphenated words level by level, one pass (a level and a pattern
 *   length) per step, under the selection rules of `PATGEN_LEVELS` or your own; `hyphenCounts` scores a set against a
 *   word list.
 * - Marked words: `parseHyphenated` reads "hy-phen-ation", `markHyphens` writes it.
 *
 * Positions are gaps: gap $i$ sits after letter $i$ (counting from 0). Hyphens closer to the ends than `leftMin` (2)
 * and `rightMin` (3) letters are never inserted, as in TeX. Words are made of the letters a to z.
 */

export {
  formatPattern,
  hyphenationPatterns,
  liangHyphenate,
  liangResult,
  liangSteps,
  markHyphens,
  parseHyphenated,
  parsePattern,
  patternSlots,
  type Hyphenation,
  type HyphenationPatterns,
  type LiangOptions,
  type LiangState,
  type ParsedPattern,
  type PatternMatch,
} from './liang'
export {
  hyphenCounts,
  PATGEN_LEVELS,
  patgenSteps,
  type HyphenatedWord,
  type HyphenCounts,
  type PatgenLevel,
  type PatgenOptions,
  type PatgenPass,
  type PatgenState,
} from './patgen'
export { hyphenationAlgorithms, hyphenationFunctions } from './registry'
