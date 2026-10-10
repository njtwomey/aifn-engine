/**
 * Scores of a hyphenator against dictionary points, counted over every gap between two letters of every word: a hit
 * (tp) is a dictionary point the method finds, a false alarm (fp) a hyphen the dictionary does not have, a miss (fn) a
 * dictionary point it leaves out. Precision is what matters most in typesetting, since a wrong hyphen is worse than a
 * missed one, so the $F_\beta$ score with $\beta = 0.5$, which favours precision, is reported beside $F_1$.
 *
 * A word of $n$ letters has $n - 1$ gaps, and gap $i$ is the one after letter $i$ (from 0), as in `HyphenatedWord`.
 * The gaps of a word list are flattened word by word, in the order of `gapLabels`.
 */

import { binaryCounts, binaryRates, fBeta } from 'aifn-compute/learning/metrics'
import type { HyphenatedWord } from 'aifn-compute/text/hyphenation'

/** Hyphen counts and rates over all gaps. */
export type HyphenScores = {
  /** Hits: dictionary points the method finds. */
  readonly tp: number
  /** False alarms: hyphens the dictionary does not have. */
  readonly fp: number
  /** Misses: dictionary points the method leaves out. */
  readonly fn: number
  /** Gaps that neither the dictionary nor the method hyphenates. */
  readonly tn: number
  /** $P = \mathrm{tp} / (\mathrm{tp} + \mathrm{fp})$, NaN when the method puts no hyphen anywhere. */
  readonly precision: number
  /** $R = \mathrm{tp} / (\mathrm{tp} + \mathrm{fn})$, NaN when the dictionary has no points. */
  readonly recall: number
  /** $F_1 = 2\mathrm{tp} / (2\mathrm{tp} + \mathrm{fp} + \mathrm{fn})$. */
  readonly f1: number
  /** $F_{0.5} = (1 + 0.25) P R / (0.25 P + R)$. */
  readonly fHalf: number
}

/**
 * The dictionary labels of every gap of every word, in order (1: a hyphen after that letter, 0: none).
 *
 * @param words The hyphenated words; word `w` contributes `w.word.length - 1` labels.
 * @returns The labels, word by word and gap by gap.
 *
 * @example Two words, nine gaps
 * print(gapLabels([{ word: 'hyphen', hyphens: [1] }, { word: 'table', hyphens: [1] }]))
 */
export function gapLabels(words: readonly HyphenatedWord[]): number[] {
  const out: number[] = []
  for (const w of words) {
    const at = new Set(w.hyphens)
    for (let i = 0; i + 1 < w.word.length; i++) out.push(at.has(i) ? 1 : 0)
  }
  return out
}

/**
 * Scores of predicted hyphens against the words' dictionary points, over every gap of every word.
 *
 * @param words The dictionary words, with their hyphen points.
 * @param predicted The predicted hyphens, one list per word in the order of `words`: the gaps (after letter $i$, from
 *   0) that take a hyphen.
 * @returns The counts and rates.
 *
 * @example One hit, one false alarm and one miss
 * const words = [{ word: 'hyphen', hyphens: [1] }, { word: 'table', hyphens: [1] }]
 * print(hyphenScores(words, [[1], [2]]))
 */
export function hyphenScores(
  words: readonly HyphenatedWord[],
  predicted: readonly (readonly number[])[],
): HyphenScores {
  const yTrue = gapLabels(words)
  const yPred = gapLabels(words.map((w, k) => ({ word: w.word, hyphens: predicted[k] })))
  return scoresOf(yTrue, yPred)
}

/**
 * Scores of per-gap probabilities at a threshold: a gap with probability $\ge$ `threshold` takes a hyphen.
 *
 * @param labels The dictionary label of every gap, as `gapLabels` returns them.
 * @param probabilities The probability of a hyphen at every gap, in the same order (as a tagger's test
 *   probabilities); only the first `labels.length` are read.
 * @param threshold The probability at and above which a gap is hyphenated; raising it trades recall for precision.
 * @returns The counts and rates.
 *
 * @example Precision and recall at two thresholds
 * const labels = [0, 1, 0, 0, 1]
 * const p = [0.2, 0.9, 0.6, 0.1, 0.4]
 * print('at 0.5', thresholdScores(labels, p, 0.5))
 * print('at 0.3', thresholdScores(labels, p, 0.3))
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

/**
 * The scores of two label lists.
 *
 * @param yTrue The dictionary label of every gap (1 for a hyphen).
 * @param yPred The predicted label of every gap, in the same order.
 * @returns The counts and rates, with precision NaN when nothing is predicted.
 */
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
