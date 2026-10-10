/**
 * Byte-pair encoding (Sennrich, Haddow & Birch 2016, after Gage 1994): start from every word split into characters
 * (or UTF-8 bytes) and repeatedly merge the most frequent adjacent pair of symbols, counting each pair once per
 * occurrence weighted by the word's count. The ordered merge list is the tokeniser: encoding replays it.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { createHeap, heapPop, heapPush } from 'aifn-compute/graph'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Tokenisation, TokenPattern } from 'aifn-compute/text/tokenise'
import { byteAlphabet, byteSymbols, textFromByteSymbols } from './bytes'
import { byCodePoint, characterPieces, encodeByWords, wordTable, type Piece, type WordCountsLike } from './words'

/** One merge: the left and right symbols, the merged symbol, and the pair's count when it was chosen. */
export interface BpeMerge {
  /** The left symbol of the pair. */
  readonly left: string
  /** The right symbol of the pair. */
  readonly right: string
  /** The new symbol, `left + right`. */
  readonly merged: string
  /** The pair's count over the corpus (weighted by word counts) when the merge was made. */
  readonly count: number
}

/** Options of {@link bpeSteps} and {@link bpe}. */
export interface BpeOptions {
  /** The number of merges to learn at most (default 1000). */
  merges?: number
  /** Stop once the vocabulary holds this many symbols (default no limit). */
  vocabularySize?: number
  /** Stop when the most frequent pair occurs fewer times than this (default 2: a pair seen once compresses nothing). */
  minCount?: number
  /** `character` (default): base symbols are code points; `byte`: the 256 UTF-8 byte symbols of GPT-2. */
  unit?: 'character' | 'byte'
  /** The end-of-word symbol appended to each word (default `</w>` for characters, none for bytes). */
  endOfWord?: string
  /** The pre-tokeniser an encoder applies to new text (default `words` for characters, `gpt2` for bytes). */
  pattern?: TokenPattern
  /**
   * Base symbols always in the vocabulary, seen or not (character level; e.g. the 256 byte symbols when the words are
   * already byte-level strings, as Hugging Face's `initial_alphabet`). Default: the characters of the words.
   */
  alphabet?: readonly string[]
}

/** The state of BPE training after `t` merges. */
export interface BpeState extends Status {
  /** The distinct training words, in order of first appearance. */
  readonly words: readonly string[]
  /** Their counts (float64 [W]). */
  readonly wordCounts: Tensor
  /** Each word's current segmentation into symbols. */
  readonly segmentations: readonly (readonly string[])[]
  /** The merges so far, in order. */
  readonly merges: readonly BpeMerge[]
  /** The base symbols, then one symbol per merge. */
  readonly vocabulary: readonly string[]
  /** The merge made by the last step (null at step 0 and when training has stopped). */
  readonly merge: BpeMerge | null
  /**
   * The corpus length in symbols, $\sum_w c_w \ell_w$ for word counts $c_w$ and segmentation lengths $\ell_w$ (falls
   * by the pair count at each merge).
   */
  readonly symbols: number
  /** True when no further merge is allowed (budget, vocabulary size or minimum count reached). */
  readonly done: boolean
}

/** A trained BPE tokeniser. */
export interface BpeModel {
  /** Marks the value as a BPE tokeniser. */
  readonly kind: 'bpe'
  /** The merges in the order they were learned; a merge's index is its rank. */
  readonly merges: readonly BpeMerge[]
  /** The base symbols, then one symbol per merge. */
  readonly vocabulary: readonly string[]
  /** Whether base symbols are code points (`character`) or GPT-2 byte symbols (`byte`). */
  readonly unit: 'character' | 'byte'
  /** The end-of-word symbol appended to each word before merging (empty for none). */
  readonly endOfWord: string
  /** The pre-tokeniser pattern that splits new text into words before segmenting. */
  readonly pattern: TokenPattern
}

const SEP = '\u0000'

/**
 * The symbols of a word before any merge, with their ranges in the word (the end-of-word symbol has zero width).
 *
 * @param word The word to split.
 * @param unit `character` splits into code points, `byte` into the byte symbols of its UTF-8 encoding.
 * @param endOfWord The symbol appended at the end of the word, at range [`word.length`, `word.length`); none when
 *   empty.
 * @returns The base symbols with their ranges, in order.
 */
function baseSymbols(word: string, unit: 'character' | 'byte', endOfWord: string): Piece[] {
  const pieces = unit === 'byte' ? byteSymbols(word) : characterPieces(word)
  if (endOfWord) pieces.push({ token: endOfWord, start: word.length, end: word.length })
  return pieces
}

/**
 * The count of every adjacent symbol pair, weighted by word counts, in order of first occurrence (reading the words in
 * order, each left to right): the `pairs` as [left, right] and their `counts` (float64 [P]).
 *
 * @param segmentations Each word's current segmentation into symbols.
 * @param wordCounts The count of each word, in the order of `segmentations`: a float64 tensor [W] or an array.
 * @returns `pairs`, the distinct adjacent pairs, and `counts`, the summed count of each: a pair occurring $k$ times
 *   in a word of count $c$ adds $k c$.
 *
 * @example Pairs of "low" (5 times) and "lower" (twice)
 * const { pairs, counts } = bpePairCounts([['l', 'o', 'w'], ['l', 'o', 'w', 'e', 'r']], [5, 2])
 * print('pairs =', pairs.map(([a, b]) => a + ' ' + b))
 * print('counts =', counts)
 */
export function bpePairCounts(
  segmentations: readonly (readonly string[])[],
  wordCounts: Tensor | readonly number[],
): { pairs: [string, string][]; counts: Tensor } {
  const c = Array.isArray(wordCounts) ? (wordCounts as readonly number[]) : toFlat(wordCounts as Tensor)
  const m = new Map<string, number>()
  segmentations.forEach((seg, w) => {
    for (let i = 0; i + 1 < seg.length; i++) {
      const key = seg[i] + SEP + seg[i + 1]
      m.set(key, (m.get(key) ?? 0) + c[w])
    }
  })
  return {
    pairs: [...m.keys()].map((k) => k.split(SEP) as [string, string]),
    counts: fromData(Float64Array.from(m.values())),
  }
}

/**
 * Replace every non-overlapping occurrence of `left right`, scanning left to right, by `merged`.
 *
 * @param seg The segmentation of one word; not modified.
 * @param left The left symbol of the pair.
 * @param right The right symbol of the pair.
 * @param merged The symbol that replaces each occurrence.
 * @returns The new segmentation.
 */
function mergeIn(seg: readonly string[], left: string, right: string, merged: string): string[] {
  const out: string[] = []
  for (let i = 0; i < seg.length; i++) {
    if (i + 1 < seg.length && seg[i] === left && seg[i + 1] === right) {
      out.push(merged)
      i++
    } else out.push(seg[i])
  }
  return out
}

/**
 * The options of BPE with their defaults filled in.
 *
 * @param options The options as given.
 * @returns Every option resolved: the default end-of-word symbol and pattern depend on `unit`.
 */
function resolve(options: BpeOptions) {
  const unit = options.unit ?? 'character'
  return {
    unit,
    maxMerges: options.merges ?? 1000,
    vocabularySize: options.vocabularySize ?? Infinity,
    minCount: options.minCount ?? 2,
    endOfWord: options.endOfWord ?? (unit === 'byte' ? '' : '</w>'),
    pattern: options.pattern ?? (unit === 'byte' ? 'gpt2' : 'words'),
    alphabet: options.alphabet ?? [],
  }
}

/**
 * BPE training as a traceable algorithm: step 0 holds the words split into base symbols; each step makes one merge,
 * the most frequent adjacent pair (ties: the pair met first reading the words in order of first appearance, each left
 * to right), and records it with its count. Training stops (`done`) at the merge budget, the vocabulary size, or when
 * the best pair occurs fewer than `minCount` times. Each step recounts every pair, $O(S)$ for $S$ symbols in the word
 * table: clear rather than fast, for corpora of up to a few thousand distinct words.
 *
 * The base vocabulary is the 256 byte symbols at byte level; at character level it is the characters seen together
 * with `alphabet`, in code-point order. The end-of-word symbol, if any, comes last. A step after `done` changes
 * nothing but `t`. Throws `DomainError` when the words hold no non-empty word or a count is not a count.
 *
 * @param words The training words: a list (each occurrence counted) or words with their counts.
 * @param options The merge budget and stopping rules, the base unit and end-of-word symbol.
 * @returns The algorithm, to step with `run`; it takes no input.
 *
 * @example The first three merges of "low lower lowest"
 * const s = run(bpeSteps(['low', 'lower', 'lowest']), undefined, 3)
 * print('merges =', s.merges.map((m) => `${m.left} + ${m.right} (count ${m.count})`))
 * print('segmentations =', s.segmentations)
 * print('symbols =', s.symbols)
 *
 * @example Sennrich et al.'s example: watch the corpus shrink
 * const alg = bpeSteps({ low: 5, lower: 2, newest: 6, widest: 3 })
 * for (let t = 1; t <= 4; t++) {
 *   const s = run(alg, undefined, t)
 *   print(`step ${t}: merge`, s.merge.merged, 'count', s.merge.count, ' symbols', s.symbols)
 * }
 */
export function bpeSteps(words: WordCountsLike, options: BpeOptions = {}): Algorithm<void, BpeState> {
  const table = wordTable(words, 'bpeSteps')
  const o = resolve(options)
  const counts = fromData(Float64Array.from(table.counts))
  const stopped = (merges: number, vocabulary: number) => merges >= o.maxMerges || vocabulary >= o.vocabularySize
  return {
    name: 'bpe',
    init: () => {
      const segmentations = table.words.map((w) => baseSymbols(w, o.unit, o.endOfWord).map((p) => p.token))
      const seen = new Set(segmentations.flat())
      const base =
        o.unit === 'byte'
          ? [...byteAlphabet()]
          : [...new Set([...o.alphabet, ...seen])].filter((s) => s !== o.endOfWord).sort(byCodePoint)
      if (o.endOfWord) base.push(o.endOfWord)
      const symbols = segmentations.reduce((acc, seg, w) => acc + seg.length * table.counts[w], 0)
      return {
        t: 0,
        words: table.words,
        wordCounts: counts,
        segmentations,
        merges: [],
        vocabulary: base,
        merge: null,
        symbols,
        done: stopped(0, base.length),
      }
    },
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1, merge: null }
      const { pairs, counts: pc } = bpePairCounts(s.segmentations, s.wordCounts)
      const c = pc.data
      let best = -1
      for (let k = 0; k < pairs.length; k++) if (best < 0 || c[k] > c[best]) best = k
      if (best < 0 || c[best] < o.minCount) return { ...s, t: s.t + 1, merge: null, done: true }
      const [left, right] = pairs[best]
      const merge: BpeMerge = { left, right, merged: left + right, count: c[best] }
      const segmentations = s.segmentations.map((seg) => mergeIn(seg, left, right, merge.merged))
      const merges = [...s.merges, merge]
      // A merged symbol can repeat an earlier one (two routes to one string); the vocabulary lists it once.
      const vocabulary = s.vocabulary.includes(merge.merged) ? s.vocabulary : [...s.vocabulary, merge.merged]
      return {
        ...s,
        t: s.t + 1,
        segmentations,
        merges,
        vocabulary,
        merge,
        symbols: s.symbols - merge.count,
        done: stopped(merges.length, vocabulary.length),
      }
    },
    done: (s) => s.done,
  }
}

/**
 * The tokeniser of a BPE training state (the merges learned so far).
 *
 * @param state A state of `bpeSteps`, at any step.
 * @param options The options the state was trained with: `unit`, `endOfWord` and `pattern` are read from them, so
 *   options that differ give a tokeniser that does not match its merges.
 * @returns The tokeniser, with the state's merges and vocabulary.
 *
 * @example A tokeniser from part-way through training
 * const alg = bpeSteps(['low', 'lower', 'lowest'])
 * print('after 1 merge: ', bpeSegment(bpeModel(run(alg, undefined, 1)), 'lowest').map((p) => p.token))
 * print('after 3 merges:', bpeSegment(bpeModel(run(alg, undefined, 3)), 'lowest').map((p) => p.token))
 */
export function bpeModel(state: BpeState, options: BpeOptions = {}): BpeModel {
  const o = resolve(options)
  return {
    kind: 'bpe',
    merges: state.merges,
    vocabulary: state.vocabulary,
    unit: o.unit,
    endOfWord: o.endOfWord,
    pattern: o.pattern,
  }
}

/**
 * Train BPE to the end (see {@link bpeSteps}) and return the tokeniser.
 *
 * @param words The training words: a list (each occurrence counted) or words with their counts.
 * @param options The merge budget and stopping rules, the base unit, end-of-word symbol and encoding pattern.
 * @returns The trained tokeniser.
 *
 * @example Merges learned from "low lower lowest"
 * const model = bpe(['low', 'lower', 'lowest'])
 * print('merges =', model.merges.map((m) => m.merged))
 * print('vocabulary =', model.vocabulary)
 *
 * @example Byte level: "é" is two byte symbols, merged like any others
 * const model = bpe(['café', 'cafés'], { unit: 'byte', merges: 4 })
 * print('merges =', model.merges.map((m) => m.merged))
 * print('vocabulary size =', model.vocabulary.length)
 */
export function bpe(words: WordCountsLike, options: BpeOptions = {}): BpeModel {
  const o = resolve(options)
  const final = run(bpeSteps(words, options), undefined, Math.min(o.maxMerges, 1e6) + 1)
  return bpeModel(final, options)
}

const ranks = new WeakMap<BpeModel, Map<string, number>>()

/**
 * The rank of each merge pair (its index in the merge list; the first if a pair repeats), keyed by the pair, built
 * once per model and cached.
 *
 * @param model The tokeniser whose merges are ranked.
 * @returns A map from `left` and `right` joined by a NUL character to the rank.
 */
function rankTable(model: BpeModel): Map<string, number> {
  let r = ranks.get(model)
  if (!r) {
    r = new Map()
    model.merges.forEach((m, k) => {
      const key = m.left + SEP + m.right
      if (!r!.has(key)) r!.set(key, k)
    })
    ranks.set(model, r)
  }
  return r
}

/** Options of {@link bpeSegment} and {@link bpeEncode}. */
export interface BpeSegmentOptions {
  /** Apply only the first `upTo` merges (default all). */
  upTo?: number
  /**
   * BPE-dropout (Provilkov, Emelianenko & Voita 2020): each time a merge is the next to apply, it is skipped with this
   * probability (default 0, deterministic), so one word gets different segmentations; needs `stream`.
   */
  dropout?: number
  /** The random stream of BPE-dropout. */
  stream?: Stream
}

/**
 * Segment one word with a BPE tokeniser, as pieces with their ranges in the word: start from base symbols and apply
 * merges in the order they were learned, by repeatedly merging the adjacent pair of lowest merge rank (leftmost among
 * equals), as GPT-2 and Hugging Face encode. The candidate pairs sit in a priority queue keyed by (rank, position) and
 * the symbols in a linked list, so a word of $n$ symbols costs $O(n \log n)$ rather than the $O(n^2)$ of rescanning
 * after every merge; the result equals replaying the merge list in order. With `dropout`, a popped merge is skipped
 * with that probability and retried after the next merge that succeeds (Hugging Face's BPE-dropout).
 *
 * @param model The tokeniser.
 * @param word One word, already pre-tokenised.
 * @param options `upTo`, `dropout` and `stream`; a number is taken as `upTo`. With `dropout` above 0, `stream` must
 *   be given.
 * @returns The pieces, in order, with their ranges in `word`.
 *
 * @example Segment a seen word and an unseen one
 * const model = bpe(['low', 'lower', 'lowest'])
 * const pieces = bpeSegment(model, 'lowest')
 * print('lowest =', pieces.map((p) => p.token), 'ranges', pieces.map((p) => [p.start, p.end]))
 * print('slower =', bpeSegment(model, 'slower').map((p) => p.token))
 *
 * @example Fewer merges, and BPE-dropout
 * const model = bpe(['low', 'lower', 'lowest'])
 * print('first merge only:', bpeSegment(model, 'lower', 1).map((p) => p.token))
 * const s = stream(1)
 * print('dropout 0.5:', bpeSegment(model, 'lower', { dropout: 0.5, stream: s }).map((p) => p.token))
 * print('dropout 0.5:', bpeSegment(model, 'lower', { dropout: 0.5, stream: s }).map((p) => p.token))
 * print('dropout 0.5:', bpeSegment(model, 'lower', { dropout: 0.5, stream: s }).map((p) => p.token))
 */
export function bpeSegment(model: BpeModel, word: string, options: BpeSegmentOptions | number = {}): Piece[] {
  const { upTo = Infinity, dropout = 0, stream } = typeof options === 'number' ? { upTo: options } : options
  const r = rankTable(model)
  const pieces = baseSymbols(word, model.unit, model.endOfWord)
  const n = pieces.length
  const token = pieces.map((p) => p.token)
  const start = pieces.map((p) => p.start)
  const end = pieces.map((p) => p.end)
  const prev = Array.from({ length: n }, (_, i) => i - 1)
  const next = Array.from({ length: n }, (_, i) => (i + 1 < n ? i + 1 : -1))
  const alive = new Array<boolean>(n).fill(true)
  const heap = createHeap<{ i: number; left: string; right: string; rank: number }>()
  const push = (i: number) => {
    if (i < 0 || next[i] < 0) return
    const j = next[i]
    const rank = r.get(token[i] + SEP + token[j])
    if (rank !== undefined && rank < upTo)
      heapPush(heap, { i, left: token[i], right: token[j], rank }, rank * (n + 1) + i)
  }
  for (let i = 0; i + 1 < n; i++) push(i)
  const skipped: { i: number; left: string; right: string; rank: number }[] = []
  for (;;) {
    const top = heapPop(heap)
    if (!top) break
    const { i, left, right, rank } = top.value
    const j = next[i]
    // Stale: one of the pair was merged away since this candidate was queued.
    if (!alive[i] || j < 0 || token[i] !== left || token[j] !== right) continue
    if (dropout > 0 && (uniform(stream!) as number) < dropout) {
      skipped.push(top.value)
      continue
    }
    token[i] = model.merges[rank].merged
    end[i] = end[j]
    alive[j] = false
    next[i] = next[j]
    if (next[j] >= 0) prev[next[j]] = i
    for (const s of skipped.splice(0)) heapPush(heap, s, s.rank * (n + 1) + s.i)
    if (prev[i] >= 0) push(prev[i])
    push(i)
  }
  const out: Piece[] = []
  for (let i = n > 0 ? 0 : -1; i >= 0; i = next[i]) out.push({ token: token[i], start: start[i], end: end[i] })
  return out
}

/**
 * Encode text with a BPE tokeniser: pre-tokenise by the model's pattern, then segment each word.
 *
 * @param model The tokeniser.
 * @param text The text to encode.
 * @param options `upTo`, `dropout` and `stream`, as in {@link bpeSegment}, applied to every word.
 * @returns The tokens with their offsets into `text`; an end-of-word symbol has a zero-width range at the word's end.
 *
 * @example Two words to tokens and offsets
 * const model = bpe(['low', 'lower', 'lowest'])
 * const enc = bpeEncode(model, 'lower lowest')
 * print('tokens =', enc.tokens)
 * print('offsets =', enc.offsets)
 */
export function bpeEncode(model: BpeModel, text: string, options: BpeSegmentOptions = {}): Tokenisation {
  return encodeByWords(text, model.pattern, (w) => bpeSegment(model, w, options))
}

/**
 * Text from BPE tokens: the end-of-word symbol becomes a space (character level), byte symbols are decoded as UTF-8
 * (byte level). The inverse of encoding up to white space between words.
 *
 * @param model The tokeniser that produced the tokens.
 * @param tokens The tokens, in order.
 * @returns The text. At character level, trailing white space is trimmed; with no end-of-word symbol the tokens are
 *   joined as they are.
 *
 * @example Encode and decode
 * const model = bpe(['low', 'lower', 'lowest'])
 * const { tokens } = bpeEncode(model, 'lower lowest')
 * print('tokens =', tokens)
 * print('text =', bpeDecode(model, tokens))
 *
 * @example Byte level: byte symbols back to UTF-8
 * const model = bpe(['naïve', 'naïvely'], { unit: 'byte', merges: 5 })
 * const { tokens } = bpeEncode(model, 'naïve')
 * print('tokens =', tokens)
 * print('text =', bpeDecode(model, tokens))
 */
export function bpeDecode(model: BpeModel, tokens: readonly string[]): string {
  const joined = tokens.join('')
  if (model.unit === 'byte') return textFromByteSymbols(joined)
  if (!model.endOfWord) return joined
  return joined.split(model.endOfWord).join(' ').trimEnd()
}
