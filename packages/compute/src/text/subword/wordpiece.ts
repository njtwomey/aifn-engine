/**
 * WordPiece (Schuster & Nakajima 2012; Wu et al. 2016): bottom-up merges like BPE, but each step merges the pair that
 * most raises the likelihood of the corpus under a unigram model of the symbols. Pieces that continue a word carry a
 * prefix (`##`, as in BERT). Two criteria, for a pair $ab$ with count $c_{ab}$, symbol counts $c_a$ and $c_b$, and
 * $N$ symbols in all: the count ratio $c_{ab} / (c_a c_b)$ of the Hugging Face reimplementation, which ranks by
 * pointwise mutual information alone, and the approximate likelihood gain $c_{ab} \log(c_{ab} N / (c_a c_b))$ of the
 * original description. Encoding is greedy longest-match-first (Devlin et al. 2019).
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Tokenisation, TokenPattern } from 'aifn-compute/text/tokenise'
import { byCodePoint, characterPieces, encodeByWords, wordTable, type Piece, type WordCountsLike } from './words'

/** One WordPiece merge: the pair, the merged piece, the pair's count and its score under the criterion. */
export interface WordPieceMerge {
  /** The left piece of the pair. */
  readonly left: string
  /** The right piece of the pair. */
  readonly right: string
  /** The new piece: `left`, then `right` without its continuation prefix. */
  readonly merged: string
  /** The pair's count over the corpus (weighted by word counts) when the merge was made. */
  readonly count: number
  /** The pair's score under the criterion, the highest of all pairs at that step. */
  readonly score: number
}

/** Options of {@link wordPieceSteps} and {@link wordPiece}. */
export interface WordPieceOptions {
  /** The number of merges to learn at most (default 1000). */
  merges?: number
  /** Stop once the vocabulary (specials included) holds this many pieces (default no limit). */
  vocabularySize?: number
  /**
   * `ratio` (default): $c_{ab} / (c_a c_b)$; `likelihood`: $c_{ab} \log(c_{ab} N / (c_a c_b))$, $N$ the total symbol
   * count.
   */
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
  /** The distinct training words, in order of first appearance. */
  readonly words: readonly string[]
  /** Word counts (float64 [W]). */
  readonly wordCounts: Tensor
  /** Each word's current segmentation into pieces, continuation pieces carrying the prefix. */
  readonly segmentations: readonly (readonly string[])[]
  /** The merges so far, in order. */
  readonly merges: readonly WordPieceMerge[]
  /** Specials, the alphabet (word-initial characters, then prefixed ones), then one piece per merge. */
  readonly vocabulary: readonly string[]
  /** The merge made by the last step (null at step 0 and when training has stopped). */
  readonly merge: WordPieceMerge | null
  /**
   * The corpus log-likelihood under the unigram model of the current symbols, $\sum_s c_s \log(c_s / N)$, in nats.
   */
  readonly logLikelihood: number
  /** True when no further merge is allowed (budget or vocabulary size reached, or no pair left). */
  readonly done: boolean
}

/** A trained WordPiece tokeniser. */
export interface WordPieceModel {
  /** Marks the value as a WordPiece tokeniser. */
  readonly kind: 'wordpiece'
  /** The pieces a word may be split into: word-initial ones bare, continuations with `prefix`. */
  readonly vocabulary: readonly string[]
  /** The continuation prefix, such as `##`. */
  readonly prefix: string
  /** The token a word becomes when it cannot be segmented. */
  readonly unknown: string
  /** The pre-tokeniser pattern that splits new text into words before segmenting. */
  readonly pattern: TokenPattern
  /** Words longer than this many characters encode as the unknown token (default 100, as BERT). */
  readonly maxCharacters: number
}

const SEP = '\u0000'

/**
 * The options of WordPiece with their defaults filled in.
 *
 * @param options The options as given.
 * @returns Every option resolved.
 */
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

/**
 * Symbol counts and pair counts (both weighted by word counts), pairs in order of first occurrence.
 *
 * @param segmentations Each word's current segmentation into pieces.
 * @param wordCounts The count of each word, in the order of `segmentations`: a float64 tensor [W] or an array.
 * @returns `symbols`, each piece's count; `pairs`, the distinct adjacent pairs as [left, right]; and `pairCounts`,
 *   the count of each pair (float64 [P]).
 *
 * @example "hug" ten times and "pug" five
 * const { symbols, pairs, pairCounts } = wordPieceCounts([['h', '##u', '##g'], ['p', '##u', '##g']], [10, 5])
 * print('symbols =', [...symbols])
 * print('pairs =', pairs.map(([a, b]) => a + ' ' + b))
 * print('pair counts =', pairCounts)
 */
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

/**
 * The log-likelihood $\sum_s c_s \log(c_s / N)$ of the corpus under a unigram model of its symbols, $N = \sum_s c_s$.
 *
 * @param symbols The count $c_s$ of each symbol $s$.
 * @returns The log-likelihood in nats (0 for no symbols).
 */
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
 * dropping the right piece's prefix (`hu` and `##g` merge to `hug`, `##g` and `##s` to `##gs`). Stops (`done`) at the
 * merge budget, the vocabulary size, or when no pair is left. Each step recounts every symbol and pair. Throws
 * `DomainError` when the words hold no non-empty word or a count is not a count.
 *
 * @param words The training words: a list (each occurrence counted) or words with their counts.
 * @param options The merge budget, vocabulary size, criterion, prefix and special tokens.
 * @returns The algorithm, to step with `run`; it takes no input.
 *
 * @example The first merges of the Hugging Face course corpus
 * const s = run(wordPieceSteps({ hug: 10, pug: 5, pun: 12, bun: 4, hugs: 5 }), undefined, 3)
 * print('merges =', s.merges.map((m) => `${m.left} + ${m.right} = ${m.merged} (score ${m.score.toFixed(4)})`))
 * print('segmentations =', s.segmentations)
 * print('log-likelihood =', s.logLikelihood)
 *
 * @example The two criteria pick different first merges
 * const words = { hug: 10, pug: 5, pun: 12, bun: 4, hugs: 5 }
 * const ratio = run(wordPieceSteps(words), undefined, 1).merge
 * const gain = run(wordPieceSteps(words, { criterion: 'likelihood' }), undefined, 1).merge
 * print('ratio:', ratio.merged, 'score', ratio.score)
 * print('likelihood:', gain.merged, 'score', gain.score)
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

/**
 * The tokeniser of a WordPiece training state, or of a given vocabulary (e.g. BERT's).
 *
 * @param source A state of `wordPieceSteps` (its vocabulary is used), or a vocabulary as a list of pieces.
 * @param options `prefix`, `pattern` and `specials` (the first special is the unknown token; `[UNK]` when the list is
 *   empty), as in training, and `maxCharacters` (default 100). The other options are ignored.
 * @returns The tokeniser.
 *
 * @example From a hand-written vocabulary
 * const model = wordPieceModel(['[UNK]', 'un', 'want', '##want', '##ed', 'runn', '##ing'])
 * print('unwanted =', wordPieceSegment(model, 'unwanted').map((p) => p.token))
 * print('running =', wordPieceSegment(model, 'running').map((p) => p.token))
 *
 * @example From a training state, after 2 and 6 merges
 * const alg = wordPieceSteps({ hug: 10, pug: 5, pun: 12, bun: 4, hugs: 5 })
 * print('step 2:', wordPieceSegment(wordPieceModel(run(alg, undefined, 2)), 'hugs').map((p) => p.token))
 * print('step 6:', wordPieceSegment(wordPieceModel(run(alg, undefined, 6)), 'hugs').map((p) => p.token))
 */
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

/**
 * Train WordPiece to the end (see {@link wordPieceSteps}) and return the tokeniser.
 *
 * @param words The training words: a list (each occurrence counted) or words with their counts.
 * @param options The merge budget, vocabulary size, criterion, prefix, special tokens and encoding pattern.
 * @returns The trained tokeniser.
 *
 * @example Six merges on the Hugging Face course corpus
 * const model = wordPiece({ hug: 10, pug: 5, pun: 12, bun: 4, hugs: 5 }, { merges: 6 })
 * print('vocabulary =', model.vocabulary)
 * print('hugs =', wordPieceSegment(model, 'hugs').map((p) => p.token))
 * print('bugs =', wordPieceSegment(model, 'bugs').map((p) => p.token))
 */
export function wordPiece(words: WordCountsLike, options: WordPieceOptions = {}): WordPieceModel {
  const o = resolve(options)
  return wordPieceModel(run(wordPieceSteps(words, options), undefined, Math.min(o.maxMerges, 1e6) + 1), options)
}

const sets = new WeakMap<WordPieceModel, Set<string>>()

/**
 * Segment one word greedily, longest match first: take the longest vocabulary piece that is a prefix of the rest of
 * the word (with the continuation prefix after the first piece), and repeat. If no piece matches at some point, or the
 * word is longer than `maxCharacters`, the whole word is the unknown token.
 *
 * @param model The tokeniser.
 * @param word One word, already pre-tokenised.
 * @returns The pieces, in order, with their ranges in `word`; or the unknown token alone, covering the whole word.
 *
 * @example Greedy longest match, and a word that fails
 * const model = wordPieceModel(['[UNK]', 'un', 'want', '##want', '##ed', '##aff', '##able'])
 * const pieces = wordPieceSegment(model, 'unwanted')
 * print('unwanted =', pieces.map((p) => p.token), 'ranges', pieces.map((p) => [p.start, p.end]))
 * print('unaffable =', wordPieceSegment(model, 'unaffable').map((p) => p.token))
 * print('affable =', wordPieceSegment(model, 'affable').map((p) => p.token))
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

/**
 * Encode text with a WordPiece tokeniser: pre-tokenise by the model's pattern, then segment each word greedily.
 *
 * @param model The tokeniser.
 * @param text The text to encode. It is not lower-cased: a pipeline does that with a normaliser.
 * @returns The tokens with their offsets into `text`.
 *
 * @example Words and punctuation
 * const model = wordPieceModel(['[UNK]', 'un', 'want', '##want', '##ed', ',', 'it', 'is'])
 * const enc = wordPieceEncode(model, 'unwanted, it is')
 * print('tokens =', enc.tokens)
 * print('offsets =', enc.offsets)
 */
export function wordPieceEncode(model: WordPieceModel, text: string): Tokenisation {
  return encodeByWords(text, model.pattern, (w) => wordPieceSegment(model, w))
}
