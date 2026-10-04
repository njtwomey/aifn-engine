/**
 * Scores of a hyphenator against dictionary points, counted over every gap between two letters of every word: a hit
 * (tp) is a dictionary point the method finds, a false alarm (fp) a hyphen the dictionary does not have, a miss (fn) a
 * dictionary point it leaves out. Precision is what matters most in typesetting, since a wrong hyphen is worse than a
 * missed one, so the F₀.₅ score (precision weighted four times as heavily as recall) is reported beside F₁.
 */

import { binaryCounts, binaryRates, fBeta } from 'aifn-compute/learning/metrics'
import type { HyphenatedWord } from 'aifn-compute/text/hyphenation'

/** Hyphen counts and rates over all gaps. */
export type HyphenScores = {
  readonly tp: number
  readonly fp: number
  readonly fn: number
  readonly tn: number
  readonly precision: number
  readonly recall: number
  readonly f1: number
  /** F₀.₅: (1 + 0.25)·P·R / (0.25·P + R). */
  readonly fHalf: number
}

/** The dictionary labels of every gap of every word, in order (1: a hyphen after that letter). */
export function gapLabels(words: readonly HyphenatedWord[]): number[] {
  const out: number[] = []
  for (const w of words) {
    const at = new Set(w.hyphens)
    for (let i = 0; i + 1 < w.word.length; i++) out.push(at.has(i) ? 1 : 0)
  }
  return out
}

/** Scores of predicted hyphens (gaps after letter i, one list per word) against the words' dictionary points. */
export function hyphenScores(
  words: readonly HyphenatedWord[],
  predicted: readonly (readonly number[])[],
): HyphenScores {
  const yTrue = gapLabels(words)
  const yPred = gapLabels(words.map((w, k) => ({ word: w.word, hyphens: predicted[k] })))
  return scoresOf(yTrue, yPred)
}

/**
 * Scores of per-gap probabilities (flattened in the order of `gapLabels`) at a threshold: a gap with probability ≥
 * `threshold` takes a hyphen.
 */
export function thresholdScores(
  labels: readonly number[],
  probabilities: ArrayLike<number>,
  threshold: number,
): HyphenScores {
  return scoresOf(
    labels,
    Array.from({ length: labels.length }, (_, i) => (probabilities[i] >= threshold ? 1 : 0)),
  )
}

function scoresOf(yTrue: readonly number[], yPred: readonly number[]): HyphenScores {
  const counts = binaryCounts([...yTrue], [...yPred], { positive: 1 })
  const rates = binaryRates(counts)
  const any = counts.tp + counts.fp > 0
  return {
    ...counts,
    precision: any ? rates.precision : NaN,
    recall: rates.recall,
    f1: rates.f1,
    fHalf: fBeta(yTrue, yPred, { beta: 0.5, average: 'binary', positive: 1, labels: [0, 1] }),
  }
}
