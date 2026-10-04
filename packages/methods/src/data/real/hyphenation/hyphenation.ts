/**
 * English hyphenation points from the Moby Hyphenator II word list (Grady Ward; Project Gutenberg etext #3204),
 * which its author placed in the public domain in January 2001. The vendored subset (`words.ts`, made by
 * `scripts/hyphenation.py`) is the 8000 single lowercase words of 4–15 letters that occur most often in the Brown
 * corpus, in rank order, each with its dictionary hyphenation points: 11401 points in all, about 79 KB. The rest of
 * the list is not kept.
 *
 * Labels are per letter: label i is 1 when the dictionary puts a hyphen after letter i. Dictionary points are one
 * convention among several acceptable ones (Moby splits by syllable, e.g. "man-y" and "ver-y", which a typesetter
 * would not use), so a "wrong" hyphen is sometimes a defensible one.
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
  readonly words: readonly DictionaryWord[]
  /** Hyphenation points in all. */
  readonly points: Size
}

/**
 * The dictionary as a truth: the hyphens of any word it lists, and per-letter labels. Words outside the list have no
 * truth (`null`).
 */
export interface HyphenationTruth {
  readonly kind: 'dictionary'
  /** The dictionary's hyphens of a word (gaps after letter i), or null if it is not listed. */
  hyphens(word: string): readonly number[] | null
  /** Per-letter labels of a listed word (1: a hyphen follows the letter), or null. */
  labels(word: string): readonly number[] | null
}

/** The vendored words with a train/test split by word, and the dictionary as truth. */
export interface HyphenationData {
  readonly all: HyphenationPart
  readonly train: HyphenationPart
  readonly test: HyphenationPart
  readonly truth: HyphenationTruth
  /** The longest word, in letters. */
  readonly longest: Size
  readonly meta: DatasetMeta & { readonly sha256: string }
}

/** Options of `mobyHyphenation`. */
export interface MobyHyphenationOptions {
  /** Keep the `words` most frequent words (default and at most 8000). */
  words?: Size
  /** The share of words held out for testing (default 0.15). */
  testFraction?: number
}

let parsed: DictionaryWord[] | null = null

/** Every vendored word, parsed once, in rank order. */
function dictionary(): DictionaryWord[] {
  if (!parsed) parsed = WORDS.split('\n').map((line, rank) => ({ ...parseHyphenated(line), rank }))
  return parsed
}

/** The number of vendored words. */
export const MOBY_WORDS = 8000

const part = (words: DictionaryWord[]): HyphenationPart => ({
  words,
  points: words.reduce((a, w) => a + w.hyphens.length, 0),
})

/** Per-letter labels of a hyphenated word: 1 at each letter a hyphen follows. */
export function hyphenLabels(word: HyphenatedWord): number[] {
  const out = new Array<number>(word.word.length).fill(0)
  for (const i of word.hyphens) out[i] = 1
  return out
}

/**
 * The `words` most frequent vendored words (see the module comment), split at random by word into training words and
 * a held-out share `testFraction`, drawn from `child(s, 'split')`. Both parts keep rank order.
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
