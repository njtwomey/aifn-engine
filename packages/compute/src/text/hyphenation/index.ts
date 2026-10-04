/**
 * `aifn-compute/text/hyphenation`: Liang's pattern hyphenation, the algorithm of TeX. Patterns with odd (hyphenating) and even
 * (inhibiting) priorities are matched against a word, step by step (`liangSteps`, `liangHyphenate`), and learned
 * from a hyphenated word list level by level, one pass per step, as PATGEN does (`patgenSteps`).
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
