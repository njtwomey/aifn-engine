/**
 * WordPiece (Schuster & Nakajima 2012; Wu et al. 2016): bottom-up merges like BPE, but each step merges the pair that
 * most raises the likelihood of the corpus under a unigram model of the symbols. Pieces that continue a word carry a
 * prefix (`##`, as in BERT). Two criteria: the count ratio c_ab / (c_a c_b) of the Hugging Face reimplementation, which
 * ranks by pointwise mutual information alone, and the approximate likelihood gain c_ab log(c_ab N / (c_a c_b)) of the
 * original description. Encoding is greedy longest-match-first (Devlin et al. 2019).
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Tokenisation, TokenPattern } from 'aifn-compute/text/tokenise'
import { byCodePoint, characterPieces, encodeByWords, wordTable, type Piece, type WordCountsLike } from './words'

/** One WordPiece merge: the pair, the merged piece, the pair's count and its score under the criterion. */
export interface WordPieceMerge {
  readonly left: string
  readonly right: string
  readonly merged: string
  readonly count: number
  readonly score: number
}

/** Options of {@link wordPieceSteps} and {@link wordPiece}. */
export interface WordPieceOptions {
  /** The number of merges to learn at most (default 1000). */
  merges?: number
  /** Stop once the vocabulary (specials included) holds this many pieces (default no limit). */
  vocabularySize?: number
  /** `ratio` (default): c_ab / (c_a c_b); `likelihood`: c_ab log(c_ab N / (c_a c_b)), N the total symbol count. */
  criterion?: 'ratio' | 'likelihood'
  /** The continuation prefix (default `##`). */
  prefix?: string
  /** Special tokens listed first in the vocabulary (default `['[UNK]']`); the first is the unknown token. */
  specials?: readonly string[]
  /** The pre-tokeniser an encoder applies to new text (default `wordsAndPunctuation`, as BERT's basic tokeniser). */
  pattern?: TokenPattern
}

/** The state of WordPiece training after `t` merges. */
export interface WordPieceState extends Status {
  readonly words: readonly string[]
  /** Word counts (float64 [W]). */
  readonly wordCounts: Tensor
  readonly segmentations: readonly (readonly string[])[]
  readonly merges: readonly WordPieceMerge[]
  /** Specials, the alphabet (word-initial characters, then prefixed ones), then one piece per merge. */
  readonly vocabulary: readonly string[]
  readonly merge: WordPieceMerge | null
  /** The corpus log-likelihood under the unigram model of the current symbols, Σ_s c_s log(c_s / N), in nats. */
  readonly logLikelihood: number
  readonly done: boolean
}

/** A trained WordPiece tokeniser. */
export interface WordPieceModel {
  readonly kind: 'wordpiece'
  readonly vocabulary: readonly string[]
  readonly prefix: string
  readonly unknown: string
  readonly pattern: TokenPattern
  /** Words longer than this many characters encode as the unknown token (default 100, as BERT). */
  readonly maxCharacters: number
}

const SEP = '\u0000'

function resolve(options: WordPieceOptions) {
  return {
    maxMerges: options.merges ?? 1000,
    vocabularySize: options.vocabularySize ?? Infinity,
    criterion: options.criterion ?? 'ratio',
    prefix: options.prefix ?? '##',
    specials: options.specials ?? ['[UNK]'],
    pattern: options.pattern ?? 'wordsAndPunctuation',
  }
}

/** Symbol counts and pair counts (both weighted by word counts), pairs in order of first occurrence. */
export function wordPieceCounts(
  segmentations: readonly (readonly string[])[],
  wordCounts: Tensor | readonly number[],
): { symbols: Map<string, number>; pairs: [string, string][]; pairCounts: Tensor } {
  const c = Array.isArray(wordCounts) ? (wordCounts as readonly number[]) : toFlat(wordCounts as Tensor)
  const symbols = new Map<string, number>()
  const pairs = new Map<string, number>()
  segmentations.forEach((seg, w) => {
    for (let i = 0; i < seg.length; i++) {
      symbols.set(seg[i], (symbols.get(seg[i]) ?? 0) + c[w])
      if (i + 1 < seg.length) {
        const key = seg[i] + SEP + seg[i + 1]
        pairs.set(key, (pairs.get(key) ?? 0) + c[w])
      }
    }
  })
  return {
    symbols,
    pairs: [...pairs.keys()].map((k) => k.split(SEP) as [string, string]),
    pairCounts: fromData(Float64Array.from(pairs.values())),
  }
}

function unigramLogLikelihood(symbols: Map<string, number>): number {
  let n = 0
  for (const c of symbols.values()) n += c
  let ll = 0
  for (const c of symbols.values()) if (c > 0) ll += c * Math.log(c / n)
  return ll
}

/**
 * WordPiece training as a traceable algorithm: step 0 splits each word into its first character and prefixed
 * continuation characters (`h ##u ##g`); each step merges the pair with the highest score (ties: the pair met first),
 * dropping the right piece's prefix (`hu` + `##g` → `hug`, `##g` + `##s` → `##gs`). Stops (`done`) at the merge budget,
 * the vocabulary size, or when no pair is left.
 */
export function wordPieceSteps(words: WordCountsLike, options: WordPieceOptions = {}): Algorithm<void, WordPieceState> {
  const table = wordTable(words, 'wordPieceSteps')
  const o = resolve(options)
  const counts = fromData(Float64Array.from(table.counts))
  const stopped = (merges: number, size: number) => merges >= o.maxMerges || size >= o.vocabularySize
  return {
    name: 'wordPiece',
    init: () => {
      const segmentations = table.words.map((w) =>
        characterPieces(w).map((p, i) => (i === 0 ? p.token : o.prefix + p.token)),
      )
      const all = [...new Set(segmentations.flat())]
      const initial = all.filter((s) => !s.startsWith(o.prefix) || s === o.prefix).sort(byCodePoint)
      const continuing = all.filter((s) => s.startsWith(o.prefix) && s !== o.prefix).sort(byCodePoint)
      const vocabulary = [...o.specials, ...initial, ...continuing]
      return {
        t: 0,
        words: table.words,
        wordCounts: counts,
        segmentations,
        merges: [],
        vocabulary,
        merge: null,
        logLikelihood: unigramLogLikelihood(wordPieceCounts(segmentations, counts).symbols),
        done: stopped(0, vocabulary.length),
      }
    },
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1, merge: null }
      const { symbols, pairs, pairCounts } = wordPieceCounts(s.segmentations, s.wordCounts)
      const c = pairCounts.data
      let total = 0
      for (const v of symbols.values()) total += v
      let best = -1
      let bestScore = -Infinity
      for (let k = 0; k < pairs.length; k++) {
        const ca = symbols.get(pairs[k][0])!
        const cb = symbols.get(pairs[k][1])!
        const score = o.criterion === 'ratio' ? c[k] / (ca * cb) : c[k] * Math.log((c[k] * total) / (ca * cb))
        if (score > bestScore) [best, bestScore] = [k, score]
      }
      if (best < 0) return { ...s, t: s.t + 1, merge: null, done: true }
      const [left, right] = pairs[best]
      const merged = left + (right.startsWith(o.prefix) ? right.slice(o.prefix.length) : right)
      const merge: WordPieceMerge = { left, right, merged, count: c[best], score: bestScore }
      const segmentations = s.segmentations.map((seg) => {
        const out: string[] = []
        for (let i = 0; i < seg.length; i++) {
          if (i + 1 < seg.length && seg[i] === left && seg[i + 1] === right) {
            out.push(merged)
            i++
          } else out.push(seg[i])
        }
        return out
      })
      const vocabulary = s.vocabulary.includes(merged) ? s.vocabulary : [...s.vocabulary, merged]
      const merges = [...s.merges, merge]
      return {
        ...s,
        t: s.t + 1,
        segmentations,
        merges,
        vocabulary,
        merge,
        logLikelihood: unigramLogLikelihood(wordPieceCounts(segmentations, s.wordCounts).symbols),
        done: stopped(merges.length, vocabulary.length),
      }
    },
    done: (s) => s.done,
  }
}

/** The tokeniser of a WordPiece training state, or of a given vocabulary (e.g. BERT's). */
export function wordPieceModel(
  source: WordPieceState | readonly string[],
  options: WordPieceOptions & { maxCharacters?: number } = {},
): WordPieceModel {
  const o = resolve(options)
  return {
    kind: 'wordpiece',
    vocabulary: Array.isArray(source) ? (source as readonly string[]) : (source as WordPieceState).vocabulary,
    prefix: o.prefix,
    unknown: o.specials[0] ?? '[UNK]',
    pattern: o.pattern,
    maxCharacters: options.maxCharacters ?? 100,
  }
}

/** Train WordPiece to the end (see {@link wordPieceSteps}) and return the tokeniser. */
export function wordPiece(words: WordCountsLike, options: WordPieceOptions = {}): WordPieceModel {
  const o = resolve(options)
  return wordPieceModel(run(wordPieceSteps(words, options), undefined, Math.min(o.maxMerges, 1e6) + 1), options)
}

const sets = new WeakMap<WordPieceModel, Set<string>>()

/**
 * Segment one word greedily, longest match first: take the longest vocabulary piece that is a prefix of the rest of
 * the word (with the continuation prefix after the first piece), and repeat. If no piece matches at some point, or the
 * word is longer than `maxCharacters`, the whole word is the unknown token.
 */
export function wordPieceSegment(model: WordPieceModel, word: string): Piece[] {
  let vocab = sets.get(model)
  if (!vocab) sets.set(model, (vocab = new Set(model.vocabulary)))
  const chars = characterPieces(word)
  if (chars.length > model.maxCharacters) return [{ token: model.unknown, start: 0, end: word.length }]
  const out: Piece[] = []
  let i = 0
  while (i < chars.length) {
    let found: Piece | null = null
    for (let j = chars.length; j > i; j--) {
      const sub = word.slice(chars[i].start, chars[j - 1].end)
      const token = i === 0 ? sub : model.prefix + sub
      if (vocab.has(token)) {
        found = { token, start: chars[i].start, end: chars[j - 1].end }
        i = j
        break
      }
    }
    if (!found) return [{ token: model.unknown, start: 0, end: word.length }]
    out.push(found)
  }
  return out
}

/** Encode text with a WordPiece tokeniser: pre-tokenise by the model's pattern, then segment each word greedily. */
export function wordPieceEncode(model: WordPieceModel, text: string): Tokenisation {
  return encodeByWords(text, model.pattern, (w) => wordPieceSegment(model, w))
}
