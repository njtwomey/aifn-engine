/**
 * English hyphenation points from the Moby Hyphenator II word list (Grady Ward; Project Gutenberg etext #3204),
 * which its author placed in the public domain in January 2001. The vendored subset (`words.ts`, made by
 * `scripts/hyphenation.py`) is the 8000 single lowercase words of 4–15 letters that occur most often in the Brown
 * corpus, in rank order, each with its dictionary hyphenation points: 11401 points in all, about 79 KB. The rest of
 * the list is not kept.
 *
 * Labels are per letter: label $i$ is 1 when the dictionary puts a hyphen after letter $i$ (counting from 0), and 0
 * otherwise. Dictionary points are one convention among several acceptable ones (Moby splits by syllable, e.g. "man-y"
 * and "ver-y", which a typesetter would not use), so a "wrong" hyphen is sometimes a defensible one.
 */

import type { DatasetInfo, DatasetMeta, Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, permutation, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { parseHyphenated, type HyphenatedWord } from 'aifn-compute/text/hyphenation'
import { SOURCE_SHA256, WORDS } from './words'

/** A hyphenated word with its Brown-corpus frequency rank (0: the most frequent). */
export interface DictionaryWord extends HyphenatedWord {
  readonly rank: Size
}

/** One part of the split: words in rank order. */
export interface HyphenationPart {
  /** The part's words with their dictionary hyphens, most frequent first. */
  readonly words: readonly DictionaryWord[]
  /** Hyphenation points in all. */
  readonly points: Size
}

/**
 * The dictionary as a truth: the hyphens of any word it lists, and per-letter labels. Words outside the list have no
 * truth (`null`).
 */
export interface HyphenationTruth {
  /** The kind of truth: a lookup in the dictionary. */
  readonly kind: 'dictionary'
  /**
   * The dictionary's hyphens of a word (the letters $i$, from 0, that a hyphen follows), or null if it is not listed.
   */
  hyphens(word: string): readonly number[] | null
  /** Per-letter labels of a listed word (1: a hyphen follows the letter), or null. */
  labels(word: string): readonly number[] | null
}

/** The vendored words with a train/test split by word, and the dictionary as truth. */
export interface HyphenationData {
  /** Every kept word, in rank order. */
  readonly all: HyphenationPart
  /** The training words: those not held out, in rank order. */
  readonly train: HyphenationPart
  /** The held-out test words, in rank order. */
  readonly test: HyphenationPart
  /** The dictionary over every kept word, training and test alike. */
  readonly truth: HyphenationTruth
  /** The longest word, in letters. */
  readonly longest: Size
  /** Name, description, source and URL of the data, and `sha256`, the hash of the Moby source file. */
  readonly meta: DatasetMeta & { readonly sha256: string }
}

/** Options of `mobyHyphenation`. */
export interface MobyHyphenationOptions {
  /** Keep the `words` most frequent words (default and at most 8000). */
  words?: Size
  /** The share of words held out for testing (default 0.15). */
  testFraction?: number
}

/** The parsed word list, filled on first use by `dictionary`. */
let parsed: DictionaryWord[] | null = null

/**
 * Every vendored word, parsed once, in rank order.
 *
 * @returns The 8000 words with their hyphens and ranks; the same array on every call.
 */
function dictionary(): DictionaryWord[] {
  if (!parsed) parsed = WORDS.split('\n').map((line, rank) => ({ ...parseHyphenated(line), rank }))
  return parsed
}

/** The number of vendored words. */
export const MOBY_WORDS = 8000

/**
 * A part of the split from its words.
 *
 * @param words The part's words, in rank order; kept as given.
 * @returns The words and their total number of hyphenation points.
 */
const part = (words: DictionaryWord[]): HyphenationPart => ({
  words,
  points: words.reduce((a, w) => a + w.hyphens.length, 0),
})

/**
 * Per-letter labels of a hyphenated word: 1 at each letter a hyphen follows, 0 elsewhere.
 *
 * @param word The word and its hyphens, each the index (from 0) of the letter the hyphen follows.
 * @returns One label per letter of `word.word`.
 *
 * @example The labels of a word
 * print(hyphenLabels({ word: 'hyphenation', hyphens: [1, 5] }))
 *
 * @example The labels of the first dictionary words with a hyphen
 * const { all } = mobyHyphenation(stream(1), { words: 20 })
 * for (const w of all.words.filter((w) => w.hyphens.length > 0).slice(0, 3)) print(w.word, hyphenLabels(w))
 */
export function hyphenLabels(word: HyphenatedWord): number[] {
  const out = new Array<number>(word.word.length).fill(0)
  for (const i of word.hyphens) out[i] = 1
  return out
}

/**
 * The `words` most frequent vendored words (see the file comment), split at random by word into training words and a
 * held-out share `testFraction`, drawn from `child(s, 'split')`. Both parts keep rank order. Throws `DomainError` when
 * `words` is not an integer from 10 to `MOBY_WORDS`, or `testFraction` is not strictly between 0 and 1.
 *
 * Source: Grady Ward, Moby Hyphenator II, Project Gutenberg etext #3204, placed in the public domain by its author
 * (January 2001); words ranked by their frequency in the Brown corpus (Francis and Kučera, 1979).
 *
 * @param s The random stream the split is drawn from.
 * @param options How many words to keep and the share to hold out.
 * @returns The kept words, the train and test parts, the dictionary as truth, the longest word's length and the
 *   provenance. The test part has $\max(1, \operatorname{round}(f n))$ words, for $n$ = `words` and $f$ =
 *   `testFraction`.
 *
 * @example Sizes and the first words
 * const d = mobyHyphenation(stream(1), { words: 1000 })
 * print('train:', d.train.words.length, 'words,', d.train.points, 'points')
 * print('test:', d.test.words.length, 'words,', d.test.points, 'points  longest:', d.longest, 'letters')
 * print('first:', d.all.words.slice(0, 3).map((w) => w.word))
 * for (const w of d.all.words.filter((w) => w.hyphens.length).slice(0, 3)) print(w.word, 'hyphens after', w.hyphens)
 *
 * @example The dictionary as truth
 * const { truth } = mobyHyphenation(stream(1), { words: 1000 })
 * print('people:', truth.hyphens('people'), truth.labels('people'))
 * print('not listed:', truth.hyphens('zzzz'))
 */
export function mobyHyphenation(s: Stream, options: MobyHyphenationOptions = {}): HyphenationData {
  const { words = MOBY_WORDS, testFraction = 0.15 } = options
  if (!(Number.isInteger(words) && words >= 10 && words <= MOBY_WORDS))
    throw new DomainError('mobyHyphenation', `mobyHyphenation: words must be an integer in 10 … ${MOBY_WORDS}`)
  if (!(testFraction > 0 && testFraction < 1))
    throw new DomainError('mobyHyphenation', 'mobyHyphenation: testFraction must be in (0, 1)')
  const all = dictionary().slice(0, words)
  const order = Array.from(toFlat(permutation(child(s, 'split'), all.length)))
  const nTest = Math.max(1, Math.round(testFraction * all.length))
  const held = new Set(order.slice(0, nTest))
  const byWord = new Map(all.map((w) => [w.word, w]))
  const truth: HyphenationTruth = {
    kind: 'dictionary',
    hyphens: (word) => byWord.get(word.toLowerCase())?.hyphens ?? null,
    labels: (word) => {
      const w = byWord.get(word.toLowerCase())
      return w ? hyphenLabels(w) : null
    },
  }
  return {
    all: part(all),
    train: part(all.filter((_, i) => !held.has(i))),
    test: part(all.filter((_, i) => held.has(i))),
    truth,
    longest: all.reduce((m, w) => Math.max(m, w.word.length), 0),
    meta: {
      name: 'Moby hyphenation',
      description: `The ${words} most frequent English words of 4–15 letters (Brown corpus) with their Moby Hyphenator dictionary hyphenation points, split by word into ${words - nTest} training and ${nTest} test words.`,
      task: 'sequence',
      featureNames: ['letter'],
      labelNames: ['no hyphen after', 'hyphen after'],
      source:
        'Grady Ward, Moby Hyphenator II (public domain, 2001), Project Gutenberg etext #3204; ranked by the Brown corpus (Francis and Kučera 1979)',
      url: 'https://www.gutenberg.org/ebooks/3204',
      sha256: SOURCE_SHA256,
    },
  }
}

// ── Registration ─────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/real/hyphenation')

dataset(
  {
    key: 'mobyHyphenation',
    name: 'Moby hyphenation',
    summary:
      'Common English words with dictionary hyphenation points (Moby Hyphenator, public domain), split into train and test words.',
    task: 'sequence',
    output: 'split',
    knobs: space({
      words: int(10, MOBY_WORDS, { default: MOBY_WORDS }),
      testFraction: real(0.05, 0.5, { default: 0.15 }),
    }),
    truth: true,
    random: true,
  },
  mobyHyphenation,
)
