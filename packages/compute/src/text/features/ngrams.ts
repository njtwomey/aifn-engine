/**
 * Word and character n-grams, in scikit-learn's order: all n-grams of the smallest n first, each in order of position.
 * Character n-grams index code points; `wordBoundaries` pads each word with a space and keeps n-grams inside words
 * (scikit-learn's `char_wb`, the shingles of fastText's subword vectors, Bojanowski et al. 2017).
 */

import { DomainError } from 'aifn-compute/foundation/errors'

/** An $n$, or an inclusive range [min, max] of $n$. */
export type NgramRange = number | readonly [number, number]

/**
 * An n-gram range as [min, max], checked: throws `DomainError` unless both are integers with $1 \le \min \le \max$.
 *
 * @param n One $n$, or an inclusive range [min, max].
 * @param op The caller's name, for the error message.
 * @returns The range [min, max].
 */
function range(n: NgramRange, op: string): [number, number] {
  const [lo, hi] = typeof n === 'number' ? [n, n] : n
  if (!(Number.isInteger(lo) && Number.isInteger(hi) && lo >= 1 && hi >= lo))
    throw new DomainError(op, `${op}: n must be integers 1 ≤ min ≤ max, got [${lo}, ${hi}]`)
  return [lo, hi]
}

/**
 * The word n-grams of a token list, each joined by `joiner`, as scikit-learn's `CountVectorizer(ngram_range=...)` forms
 * them: every n-gram of the smallest $n$ first, each in order of position. Repeats are kept. Throws `DomainError` for
 * an invalid range.
 *
 * @param tokens The tokens, in order.
 * @param n One $n$, or an inclusive range [min, max] of $n$.
 * @param joiner The string placed between the tokens of an n-gram.
 * @returns The n-grams; none for an $n$ longer than the list.
 *
 * @example Unigrams and bigrams
 * print(wordNgrams(['the', 'cat', 'sat', 'on', 'the', 'mat'], [1, 2]))
 * print(wordNgrams(['new', 'york', 'city'], 3, '_'))
 */
export function wordNgrams(tokens: readonly string[], n: NgramRange, joiner = ' '): string[] {
  const [lo, hi] = range(n, 'wordNgrams')
  const out: string[] = []
  for (let k = lo; k <= hi; k++)
    for (let i = 0; i + k <= tokens.length; i++) out.push(tokens.slice(i, i + k).join(joiner))
  return out
}

/** Options of {@link characterNgrams}. */
export interface CharacterNgramOptions {
  /**
   * Keep n-grams inside words, each word padded by one space on both sides (scikit-learn's `char_wb`); a padded word
   * shorter than n gives itself once. Default false: n-grams run across the whole text (scikit-learn's `char`).
   */
  wordBoundaries?: boolean
}

/**
 * The character n-grams of a text. Runs of two or more white-space characters are first collapsed to one space, as
 * scikit-learn does. Throws `DomainError` for an invalid range.
 *
 * @param text The text; n-grams are formed over its code points.
 * @param n One $n$, or an inclusive range [min, max] of $n$.
 * @param options Whether n-grams stay inside words; see {@link CharacterNgramOptions}.
 * @returns The n-grams, smallest $n$ first, each in order of position (word by word with `wordBoundaries`).
 *
 * @example Across the text, and inside padded words (scikit-learn's `char_wb`)
 * print(characterNgrams('to be', 3))
 * print(characterNgrams('to be', 3, { wordBoundaries: true }))
 * print(characterNgrams('where', [3, 4], { wordBoundaries: true }))
 */
export function characterNgrams(text: string, n: NgramRange, options: CharacterNgramOptions = {}): string[] {
  const [lo, hi] = range(n, 'characterNgrams')
  const clean = text.replace(/\s\s+/gu, ' ')
  const out: string[] = []
  if (!options.wordBoundaries) {
    const chars = [...clean]
    for (let k = lo; k <= Math.min(hi, chars.length); k++)
      for (let i = 0; i + k <= chars.length; i++) out.push(chars.slice(i, i + k).join(''))
    return out
  }
  for (const word of clean.split(/\s+/u).filter((w) => w.length > 0)) {
    const chars = [...` ${word} `]
    for (let k = lo; k <= hi; k++) {
      let offset = 0
      out.push(chars.slice(0, k).join(''))
      while (offset + k < chars.length) {
        offset++
        out.push(chars.slice(offset, offset + k).join(''))
      }
      // A word shorter than n is counted once, not once per n.
      if (offset === 0) break
    }
  }
  return out
}
