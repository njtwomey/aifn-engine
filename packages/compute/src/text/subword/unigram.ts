/**
 * The unigram language-model tokeniser (Kudo 2018, "Subword regularization", §3): a vocabulary of pieces $v$ with
 * probabilities $p(v)$, a segmentation $x = (x_1, \dots, x_m)$ of a word scored by $P(x) = \prod_i p(x_i)$, the best
 * segmentation found by Viterbi on the lattice of pieces over the word's characters. Training starts from a large seed
 * vocabulary (every character and frequent substrings) and alternates EM, with expected piece counts from
 * forward–backward on each lattice, with pruning the pieces whose removal lowers the marginal likelihood least. Words
 * carry a boundary prefix (SentencePiece's "▁", Kudo & Richardson 2018).
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import type { Stream } from 'aifn-compute/foundation/random'
import { uniform } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Tokenisation, TokenPattern } from 'aifn-compute/text/tokenise'
import { byCodePoint, encodeByWords, wordTable, type Piece, type WordCountsLike } from './words'

/** A unigram tokeniser: pieces with log-probabilities (float64 [V]). */
export interface UnigramLmModel {
  /** Marks the value as a unigram tokeniser. */
  readonly kind: 'unigram-lm'
  /** The vocabulary of pieces. */
  readonly pieces: readonly string[]
  /** $\log p(v)$ of each piece, in the order of `pieces` (float64 [V]); $-\infty$ allowed. */
  readonly logProbs: Tensor
  /** Prefixed to every word before segmentation (default "▁"; empty for none). */
  readonly boundary: string
  /** The pre-tokeniser pattern that splits new text into words before segmenting. */
  readonly pattern: TokenPattern
  /** The token for a character no piece covers. */
  readonly unknown: string
}

/** Options of {@link unigramLmSteps} and {@link unigramLm}. */
export interface UnigramLmOptions {
  /** The target number of pieces (default 8000); single characters are always kept, so it may be exceeded. */
  vocabularySize?: number
  /** The longest seed piece, in characters (default 16). */
  maxPieceLength?: number
  /** Multi-character seeds kept, the most frequent (default 10 000). */
  seedSize?: number
  /** The share of pieces kept by each pruning round (default 0.75). */
  shrink?: number
  /** EM iterations per round (default 2, as SentencePiece). */
  emIterations?: number
  /** The word-boundary prefix (default "▁"). */
  boundary?: string
  /** The pre-tokeniser an encoder applies to new text (default `words`). */
  pattern?: TokenPattern
  /** The unknown token (default `<unk>`). */
  unknown?: string
}

/** A pruned piece and its loss: the drop in the corpus log marginal likelihood (nats) were it removed. */
export interface PrunedPiece {
  /** The piece removed. */
  readonly piece: string
  /** The drop in the corpus log marginal likelihood, in nats, its removal was predicted to cause. */
  readonly loss: number
}

/** The state of unigram training after `t` rounds. */
export interface UnigramLmState extends Status {
  /** The distinct training words, in order of first appearance. */
  readonly words: readonly string[]
  /** Their counts (float64 [W]). */
  readonly wordCounts: Tensor
  /** The current vocabulary: single characters (the boundary among them) first, then longer pieces. */
  readonly pieces: readonly string[]
  /** $\log p(v)$ per piece (float64 [V]). */
  readonly logProbs: Tensor
  /**
   * $\sum_w c_w \log P(w)$, the corpus log marginal likelihood (nats) under the current pieces, $c_w$ the count of
   * word $w$.
   */
  readonly logLikelihood: number
  /** The pieces removed by the last round, with their losses (smallest loss first). */
  readonly pruned: readonly PrunedPiece[]
  /** True once the vocabulary is at its target size and has been refitted. */
  readonly done: boolean
}

/**
 * The options of the unigram tokeniser with their defaults filled in.
 *
 * @param options The options as given.
 * @returns Every option resolved.
 */
function resolve(options: UnigramLmOptions) {
  return {
    vocabularySize: options.vocabularySize ?? 8000,
    maxPieceLength: options.maxPieceLength ?? 16,
    seedSize: options.seedSize ?? 10000,
    shrink: options.shrink ?? 0.75,
    emIterations: options.emIterations ?? 2,
    boundary: options.boundary ?? '▁',
    pattern: options.pattern ?? 'words',
    unknown: options.unknown ?? '<unk>',
  }
}

// ── The lattice ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** One edge of a word's lattice: piece `id` spans code points $[i, j)$. */
interface Edge {
  /** The index of the first code point covered. */
  i: number
  /** One past the index of the last code point covered. */
  j: number
  /** The piece's index in the vocabulary. */
  id: number
}

/**
 * The lattice edges of a word: one for every piece that matches a substring of its code points, in order of start,
 * then of end.
 *
 * @param chars The word's code points (boundary prefix included).
 * @param index Each piece's index in the vocabulary.
 * @param maxLength The longest piece in code points: longer substrings are not looked up.
 * @returns The edges.
 */
function lattice(chars: readonly string[], index: ReadonlyMap<string, number>, maxLength: number): Edge[] {
  const edges: Edge[] = []
  for (let i = 0; i < chars.length; i++) {
    let sub = ''
    for (let j = i + 1; j <= Math.min(chars.length, i + maxLength); j++) {
      sub += chars[j - 1]
      const id = index.get(sub)
      if (id !== undefined) edges.push({ i, j, id })
    }
  }
  return edges
}

/**
 * $\log(e^a + e^b)$ without overflow; $-\infty$ is the identity.
 *
 * @param a The first log-value.
 * @param b The second log-value.
 * @returns The log of the sum.
 */
const logAdd = (a: number, b: number) =>
  a === -Infinity ? b : b === -Infinity ? a : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)))

/**
 * Forward and backward log-sums over the lattice ($\alpha_j$: all segmentations of $[0, j)$; $\beta_i$: of $[i, n)$).
 *
 * @param n The number of code points in the word.
 * @param edges The lattice edges, in order of start (as `lattice` makes them).
 * @param lp The log-probability of each piece, by index.
 * @param skip A piece index whose edges are left out (to score the word without that piece), or $-1$ for none.
 * @returns `alpha` and `beta` ($n + 1$ log-sums each, $-\infty$ where no segmentation reaches) and `logZ`
 *   $= \alpha_n$, the log marginal probability of the word.
 */
function forwardBackward(n: number, edges: readonly Edge[], lp: ArrayLike<number>, skip = -1) {
  const alpha = new Float64Array(n + 1).fill(-Infinity)
  const beta = new Float64Array(n + 1).fill(-Infinity)
  alpha[0] = 0
  beta[n] = 0
  // Edges are generated in order of their start, so a pass sorted by end gives α and the reverse by start gives β.
  const byEnd = [...edges].sort((a, b) => a.j - b.j)
  for (const e of byEnd) if (e.id !== skip) alpha[e.j] = logAdd(alpha[e.j], alpha[e.i] + lp[e.id])
  for (let k = edges.length - 1; k >= 0; k--) {
    const e = edges[k]
    if (e.id !== skip) beta[e.i] = logAdd(beta[e.i], lp[e.id] + beta[e.j])
  }
  return { alpha, beta, logZ: alpha[n] }
}

/**
 * Each piece's index in the vocabulary (the last, should a piece repeat).
 *
 * @param pieces The vocabulary.
 * @returns A map from piece to index.
 */
const pieceIndex = (pieces: readonly string[]) => new Map(pieces.map((p, k) => [p, k]))
/**
 * The length of the longest piece, in code points (at least 1).
 *
 * @param pieces The vocabulary.
 * @returns The length.
 */
const maxLen = (pieces: readonly string[]) => pieces.reduce((m, p) => Math.max(m, [...p].length), 1)

// ── EM and pruning ───────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A training corpus as the unigram trainer holds it: each distinct word as boundary-prefixed code points, with counts.
 */
export interface UnigramLmCorpus {
  /** Each distinct word as its code points, boundary prefix first. */
  chars: string[][]
  /** The count of each word. */
  counts: number[]
}

/**
 * One EM iteration: expected piece counts by forward–backward, then $p(v) \propto$ expected count. Each expected count
 * is first raised to at least $10^{-12}$ of their total, so no piece reaches probability 0.
 *
 * @param corpus The training words and counts.
 * @param pieces The vocabulary.
 * @param lp The current log-probability of each piece; not modified.
 * @returns `lp`, the new log-probabilities, and `ll`, the corpus log marginal likelihood under the old ones.
 */
function emStep(
  corpus: UnigramLmCorpus,
  pieces: readonly string[],
  lp: Float64Array,
): { lp: Float64Array; ll: number } {
  const index = pieceIndex(pieces)
  const L = maxLen(pieces)
  const expected = new Float64Array(pieces.length)
  let ll = 0
  corpus.chars.forEach((chars, w) => {
    const edges = lattice(chars, index, L)
    const { alpha, beta, logZ } = forwardBackward(chars.length, edges, lp)
    ll += corpus.counts[w] * logZ
    for (const e of edges) expected[e.id] += corpus.counts[w] * Math.exp(alpha[e.i] + lp[e.id] + beta[e.j] - logZ)
  })
  let total = 0
  for (const x of expected) total += x
  // A floor keeps every piece (single characters above all) segmentable once EM has nearly emptied it.
  const floored = expected.map((x) => Math.max(x, 1e-12 * total))
  let z = 0
  for (const x of floored) z += x
  return { lp: floored.map((x) => Math.log(x / z)), ll }
}

/**
 * The corpus log marginal likelihood $\sum_w c_w \log P(w)$, in nats.
 *
 * @param corpus The training words and counts.
 * @param pieces The vocabulary.
 * @param lp The log-probability of each piece.
 * @returns The log-likelihood ($-\infty$ when some word cannot be segmented).
 */
function logLikelihoodOf(corpus: UnigramLmCorpus, pieces: readonly string[], lp: ArrayLike<number>): number {
  const index = pieceIndex(pieces)
  const L = maxLen(pieces)
  let ll = 0
  corpus.chars.forEach((chars, w) => {
    ll += corpus.counts[w] * forwardBackward(chars.length, lattice(chars, index, L), lp).logZ
  })
  return ll
}

/**
 * The loss of each piece: $\sum_w c_w (\log P(w) - \log P_{\setminus v}(w))$, the drop in the corpus log marginal
 * likelihood when piece $v$ is removed with the other probabilities held fixed (Kudo 2018, §3.2 step 3). Infinite
 * when some word cannot be segmented without $v$. Returns float64 [V].
 *
 * @param model The pieces and their log-probabilities: a model, or a training state.
 * @param corpus The words to score, as code points with the boundary prefix already in place, and their counts.
 * @returns The loss of each piece, in the order of `pieces`, in nats; 0 for a piece no word uses.
 *
 * @example Which pieces "hug" (10 times) and "pug" (5 times) depend on
 * const model = unigramLmModel(
 *   {
 *     pieces: ['h', 'u', 'g', 'hu', 'ug', 'p', 'pu', 'n', 'un', 'b', 'bu', 's', 'hug', 'gs', 'ugs'],
 *     probabilities: [15, 36, 20, 15, 20, 17, 17, 16, 16, 4, 4, 5, 15, 5, 5],
 *   },
 *   { boundary: '' },
 * )
 * const loss = unigramLmLosses(model, { chars: [['h', 'u', 'g'], ['p', 'u', 'g']], counts: [10, 5] })
 * print(model.pieces.map((p, k) => `${p}: ${loss.data[k].toFixed(3)}`).join(', '))
 */
export function unigramLmLosses(model: Pick<UnigramLmModel, 'pieces' | 'logProbs'>, corpus: UnigramLmCorpus): Tensor {
  const { pieces } = model
  const lp = toFlat(model.logProbs)
  const index = pieceIndex(pieces)
  const L = maxLen(pieces)
  const loss = new Float64Array(pieces.length)
  corpus.chars.forEach((chars, w) => {
    const edges = lattice(chars, index, L)
    const { logZ } = forwardBackward(chars.length, edges, lp)
    for (const id of new Set(edges.map((e) => e.id))) {
      const without = forwardBackward(chars.length, edges, lp, id).logZ
      loss[id] += corpus.counts[w] * (logZ - without)
    }
  })
  return fromData(loss)
}

/**
 * The training corpus: each word with the boundary prefixed, split into code points.
 *
 * @param words The distinct words.
 * @param counts Their counts, in the same order.
 * @param boundary The word-boundary prefix (empty for none).
 * @returns The corpus, with a copy of `counts`.
 */
function corpusOf(words: readonly string[], counts: readonly number[], boundary: string): UnigramLmCorpus {
  return { chars: words.map((w) => [...(boundary + w)]), counts: [...counts] }
}

/**
 * Seeds: every character, and the `seedSize` most frequent substrings of 2 to `maxPieceLength` characters (ties in
 * code-point order), each with probability proportional to its count in the corpus.
 *
 * @param corpus The training words and counts.
 * @param maxPieceLength The longest substring counted, in code points.
 * @param seedSize How many multi-character substrings to keep.
 * @returns `pieces`, the characters in code-point order then the substrings by falling count, and `lp`, their
 *   log-probabilities.
 */
function seeds(corpus: UnigramLmCorpus, maxPieceLength: number, seedSize: number) {
  const freq = new Map<string, number>()
  const singles = new Set<string>()
  corpus.chars.forEach((chars, w) => {
    for (let i = 0; i < chars.length; i++) {
      singles.add(chars[i])
      let sub = ''
      for (let j = i; j < Math.min(chars.length, i + maxPieceLength); j++) {
        sub += chars[j]
        freq.set(sub, (freq.get(sub) ?? 0) + corpus.counts[w])
      }
    }
  })
  const multi = [...freq.keys()]
    .filter((p) => !singles.has(p))
    .sort((a, b) => freq.get(b)! - freq.get(a)! || byCodePoint(a, b))
    .slice(0, seedSize)
  const pieces = [...[...singles].sort(byCodePoint), ...multi]
  const total = pieces.reduce((s, p) => s + freq.get(p)!, 0)
  return { pieces, lp: Float64Array.from(pieces, (p) => Math.log(freq.get(p)! / total)) }
}

/**
 * Unigram-LM training as a traceable algorithm. Step 0 holds the seed vocabulary with probabilities proportional to
 * substring frequency. Each step is one round: `emIterations` of EM, then, while the vocabulary exceeds the target,
 * removal of the multi-character pieces with the smallest loss so that $\max(T, \lfloor s V \rfloor)$ of the $V$
 * remain, for target $T$ and `shrink` $s$ (single characters are always kept, so every word stays encodable, and $T$
 * is raised to their number if it is below it). The round that reaches the target refits by EM and sets `done`.
 * Throws `DomainError` when `shrink` is not in $(0, 1)$, the words hold no non-empty word or a count is not a count.
 *
 * @param words The training words: a list (each occurrence counted) or words with their counts.
 * @param options The target size, seeding, pruning and EM settings, and the boundary prefix.
 * @returns The algorithm, to step with `run`; it takes no input.
 *
 * @example Prune 31 seeds to 12 pieces, a round per step
 * const alg = unigramLmSteps({ hug: 10, pug: 5, pun: 12, bun: 4, hugs: 5 }, { vocabularySize: 12 })
 * for (let t = 0; t <= 3; t++) {
 *   const s = run(alg, undefined, t)
 *   print(`round ${t}:`, s.pieces.length, 'pieces, log-likelihood', s.logLikelihood, s.done ? '(done)' : '')
 * }
 * print('pruned in the last round:', run(alg, undefined, 3).pruned.map((p) => p.piece))
 */
export function unigramLmSteps(words: WordCountsLike, options: UnigramLmOptions = {}): Algorithm<void, UnigramLmState> {
  const table = wordTable(words, 'unigramLmSteps')
  const o = resolve(options)
  if (!(o.shrink > 0 && o.shrink < 1))
    throw new DomainError('unigramLmSteps', 'unigramLmSteps: shrink must be in (0, 1)')
  const corpus = corpusOf(table.words, table.counts, o.boundary)
  const counts = fromData(Float64Array.from(table.counts))
  const em = (pieces: readonly string[], lp: Float64Array) => {
    let cur = { lp, ll: NaN }
    for (let k = 0; k < Math.max(1, o.emIterations); k++) cur = emStep(corpus, pieces, cur.lp)
    return { lp: cur.lp, ll: logLikelihoodOf(corpus, pieces, cur.lp) }
  }
  return {
    name: 'unigramLm',
    init: () => {
      const { pieces, lp } = seeds(corpus, o.maxPieceLength, o.seedSize)
      return {
        t: 0,
        words: table.words,
        wordCounts: counts,
        pieces,
        logProbs: fromData(lp),
        logLikelihood: logLikelihoodOf(corpus, pieces, lp),
        pruned: [],
        done: false,
      }
    },
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1, pruned: [] }
      const fitted = em(s.pieces, Float64Array.from(toFlat(s.logProbs)))
      const single = (p: string) => [...p].length === 1
      const multi = s.pieces.filter((p) => !single(p)).length
      const target = Math.max(o.vocabularySize, s.pieces.length - multi)
      if (s.pieces.length <= target) {
        return { ...s, t: s.t + 1, logProbs: fromData(fitted.lp), logLikelihood: fitted.ll, pruned: [], done: true }
      }
      const loss = toFlat(unigramLmLosses({ pieces: s.pieces, logProbs: fromData(fitted.lp) }, corpus))
      const keep = Math.max(target, Math.floor(o.shrink * s.pieces.length))
      const candidates = s.pieces
        .map((p, k) => ({ piece: p, loss: loss[k], k }))
        .filter((e) => !single(e.piece))
        .sort((a, b) => a.loss - b.loss || byCodePoint(a.piece, b.piece))
      const removed = candidates.slice(0, s.pieces.length - keep)
      const gone = new Set(removed.map((e) => e.k))
      const pieces = s.pieces.filter((_, k) => !gone.has(k))
      // Renormalise the survivors; the next round's EM refits them.
      let lp = Float64Array.from(fitted.lp.filter((_, k) => !gone.has(k)))
      let total = 0
      for (const x of lp) total += Math.exp(x)
      lp = lp.map((x) => x - Math.log(total))
      const done = pieces.length <= target
      const final = done ? em(pieces, lp) : { lp, ll: logLikelihoodOf(corpus, pieces, lp) }
      return {
        ...s,
        t: s.t + 1,
        pieces,
        logProbs: fromData(final.lp),
        logLikelihood: final.ll,
        pruned: removed.map(({ piece, loss: l }) => ({ piece, loss: l })),
        done,
      }
    },
    done: (s) => s.done,
  }
}

/**
 * A unigram tokeniser from a training state, or from pieces and their probabilities (normalised to sum to one unless
 * `normalise: false`, e.g. to reproduce a worked example with scores that do not).
 *
 * @param source A state of `unigramLmSteps` (its pieces and log-probabilities are used), or `pieces` with their
 *   `probabilities` (any non-negative weights, in the same order).
 * @param options `boundary`, `pattern` and `unknown`, as in training, and `normalise` (default true); the other
 *   options are ignored.
 * @returns The tokeniser.
 *
 * @example From hand-set weights, with no boundary prefix
 * const model = unigramLmModel(
 *   {
 *     pieces: ['h', 'u', 'g', 'hu', 'ug', 'p', 'pu', 'n', 'un', 'b', 'bu', 's', 'hug', 'gs', 'ugs'],
 *     probabilities: [15, 36, 20, 15, 20, 17, 17, 16, 16, 4, 4, 5, 15, 5, 5],
 *   },
 *   { boundary: '' },
 * )
 * print('p(u) =', Math.exp(model.logProbs.data[1]))
 * print('hugs =', unigramLmSegment(model, 'hugs').map((p) => p.token))
 *
 * @example With the default boundary "▁": word-initial pieces differ, and an unknown character
 * const model = unigramLmModel({ pieces: ['▁', 'a', 'b', '▁ab', 'ab'], probabilities: [0.1, 0.2, 0.2, 0.3, 0.2] })
 * print('ab =', unigramLmSegment(model, 'ab').map((p) => p.token))
 * print('abab =', unigramLmSegment(model, 'abab').map((p) => p.token))
 * print('abc =', unigramLmSegment(model, 'abc').map((p) => p.token))
 */
export function unigramLmModel(
  source: UnigramLmState | { pieces: readonly string[]; probabilities: Tensor | readonly number[] },
  options: UnigramLmOptions & { normalise?: boolean } = {},
): UnigramLmModel {
  const o = resolve(options)
  if ('probabilities' in source) {
    const p = Array.isArray(source.probabilities) ? source.probabilities : toFlat(source.probabilities as Tensor)
    const total = options.normalise === false ? 1 : p.reduce((a: number, b: number) => a + b, 0)
    return {
      kind: 'unigram-lm',
      pieces: source.pieces,
      logProbs: fromData(Float64Array.from(p, (x: number) => Math.log(x / total))),
      boundary: o.boundary,
      pattern: o.pattern,
      unknown: o.unknown,
    }
  }
  const s = source as UnigramLmState
  return {
    kind: 'unigram-lm',
    pieces: s.pieces,
    logProbs: s.logProbs,
    boundary: o.boundary,
    pattern: o.pattern,
    unknown: o.unknown,
  }
}

/**
 * Train the unigram tokeniser to its target size (see {@link unigramLmSteps}; at most 1000 rounds).
 *
 * @param words The training words: a list (each occurrence counted) or words with their counts.
 * @param options The target size, seeding, pruning and EM settings, the boundary prefix, pattern and unknown token.
 * @returns The trained tokeniser.
 *
 * @example Twelve pieces from five words
 * const model = unigramLm({ hug: 10, pug: 5, pun: 12, bun: 4, hugs: 5 }, { vocabularySize: 12 })
 * print('pieces =', model.pieces)
 * print('hugs =', unigramLmSegment(model, 'hugs').map((p) => p.token))
 */
export function unigramLm(words: WordCountsLike, options: UnigramLmOptions = {}): UnigramLmModel {
  return unigramLmModel(run(unigramLmSteps(words, options), undefined, 1000), options)
}

// ── Segmenting with a model ──────────────────────────────────────────────────────────────────────────────────────────

/** A word's lattice under a model, with unknown pieces added for uncovered characters. */
interface WordLattice {
  /** The word's code points, after the boundary's. */
  chars: string[]
  /** [start, end) of each code point in the original word; boundary characters are zero-width at 0. */
  spans: [number, number][]
  /** The lattice edges, in order of start, then of end. */
  edges: Edge[]
  /** The log-probability of each entry of `pieces`. */
  lp: Float64Array
  /** Pieces of the model, then one unknown piece per uncovered character. */
  pieces: string[]
}

const latticeCache = new WeakMap<UnigramLmModel, { index: Map<string, number>; L: number; lp: Float64Array }>()

/**
 * The lattice of a word under a model. A piece of probability 0 is scored at a floor 10 nats below the least probable
 * piece, and a character that no single-character piece covers gets an unknown edge at that floor.
 *
 * @param model The tokeniser; its piece index and log-probabilities are cached per model.
 * @param word The word, without the boundary prefix.
 * @returns The lattice.
 */
function wordLattice(model: UnigramLmModel, word: string): WordLattice {
  let c = latticeCache.get(model)
  if (!c) {
    c = { index: pieceIndex(model.pieces), L: maxLen(model.pieces), lp: Float64Array.from(toFlat(model.logProbs)) }
    latticeCache.set(model, c)
  }
  const chars: string[] = []
  const spans: [number, number][] = []
  for (const b of model.boundary) {
    chars.push(b)
    spans.push([0, 0])
  }
  let at = 0
  for (const ch of word) {
    chars.push(ch)
    spans.push([at, at + ch.length])
    at += ch.length
  }
  const edges = lattice(chars, c.index, c.L)
  const pieces = [...model.pieces]
  const finite = Array.from(c.lp).filter(Number.isFinite)
  const floor = (finite.length > 0 ? finite.reduce((m, v) => Math.min(m, v), Infinity) : 0) - 10
  // Pieces of probability 0 stay in the lattice at the floor, so a word never loses every segmentation.
  const lp = Array.from(c.lp, (x) => (Number.isFinite(x) ? x : floor))
  // A character no piece starts with gets an unknown edge, scored below every piece (SentencePiece's unk penalty).
  const covered = new Set(edges.filter((e) => e.j === e.i + 1).map((e) => e.i))
  for (let i = 0; i < chars.length; i++)
    if (!covered.has(i)) {
      edges.push({ i, j: i + 1, id: pieces.length })
      pieces.push(model.unknown)
      lp.push(floor)
    }
  edges.sort((a, b) => a.i - b.i || a.j - b.j)
  return { chars, spans, edges, lp: Float64Array.from(lp), pieces }
}

/**
 * The pieces of a path through a lattice, with their ranges in the original word.
 *
 * @param wl The word's lattice.
 * @param path The edges of the path, in order.
 * @returns One piece per edge.
 */
const toPieces = (wl: WordLattice, path: readonly Edge[]): Piece[] =>
  path.map((e) => ({ token: wl.pieces[e.id], start: wl.spans[e.i][0], end: wl.spans[e.j - 1][1] }))

/**
 * The most probable segmentation of a word (Viterbi on its lattice), with its log-probability.
 *
 * @param model The tokeniser.
 * @param word One word, already pre-tokenised, without the boundary prefix (the model adds it).
 * @returns `pieces`, with their ranges in `word` (the boundary has zero width at 0), and `logProb`, $\log P(x)$ of
 *   that segmentation. Among equally probable segmentations, the first found is kept.
 *
 * @example The best of the segmentations of "hugs"
 * const model = unigramLmModel(
 *   {
 *     pieces: ['h', 'u', 'g', 'hu', 'ug', 'p', 'pu', 'n', 'un', 'b', 'bu', 's', 'hug', 'gs', 'ugs'],
 *     probabilities: [15, 36, 20, 15, 20, 17, 17, 16, 16, 4, 4, 5, 15, 5, 5],
 *   },
 *   { boundary: '' },
 * )
 * const { pieces, logProb } = unigramLmViterbi(model, 'hugs')
 * print('pieces =', pieces.map((p) => p.token), 'log P =', logProb)
 * print('log p(h) + log p(ugs) =', Math.log(15 / 210) + Math.log(5 / 210))
 * print('hux =', unigramLmViterbi(model, 'hux').pieces.map((p) => p.token))
 */
export function unigramLmViterbi(model: UnigramLmModel, word: string): { pieces: Piece[]; logProb: number } {
  const wl = wordLattice(model, word)
  const n = wl.chars.length
  const best = new Float64Array(n + 1).fill(-Infinity)
  const back: (Edge | null)[] = new Array(n + 1).fill(null)
  best[0] = 0
  for (const e of [...wl.edges].sort((a, b) => a.j - b.j)) {
    const v = best[e.i] + wl.lp[e.id]
    if (v > best[e.j]) [best[e.j], back[e.j]] = [v, e]
  }
  const path: Edge[] = []
  for (let j = n; j > 0;) {
    const e = back[j]!
    path.push(e)
    j = e.i
  }
  return { pieces: toPieces(wl, path.reverse()), logProb: best[n] }
}

/**
 * Segment one word by Viterbi (the pieces with their ranges in the word).
 *
 * @param model The tokeniser.
 * @param word One word, already pre-tokenised, without the boundary prefix.
 * @returns The pieces of the most probable segmentation, with their ranges in `word`.
 *
 * @example With the boundary prefix
 * const model = unigramLmModel({ pieces: ['▁', 'a', 'b', '▁ab', 'ab'], probabilities: [0.1, 0.2, 0.2, 0.3, 0.2] })
 * const pieces = unigramLmSegment(model, 'abab')
 * print('tokens =', pieces.map((p) => p.token), 'ranges', pieces.map((p) => [p.start, p.end]))
 */
export function unigramLmSegment(model: UnigramLmModel, word: string): Piece[] {
  return unigramLmViterbi(model, word).pieces
}

/**
 * $\log \sum_x P(x)$ over every segmentation $x$ of a word: its marginal log-probability (forward algorithm).
 *
 * @param model The tokeniser.
 * @param word One word, already pre-tokenised, without the boundary prefix.
 * @returns The marginal log-probability, in nats.
 *
 * @example The sum over the three segmentations of "pug"
 * const model = unigramLmModel(
 *   {
 *     pieces: ['h', 'u', 'g', 'hu', 'ug', 'p', 'pu', 'n', 'un', 'b', 'bu', 's', 'hug', 'gs', 'ugs'],
 *     probabilities: [15, 36, 20, 15, 20, 17, 17, 16, 16, 4, 4, 5, 15, 5, 5],
 *   },
 *   { boundary: '' },
 * )
 * print('marginal =', unigramLmMarginal(model, 'pug'))
 * const all = unigramLmSegmentations(model, 'pug')
 * print('log of the sum =', Math.log(all.reduce((a, x) => a + x.probability, 0)))
 */
export function unigramLmMarginal(model: UnigramLmModel, word: string): number {
  const wl = wordLattice(model, word)
  return forwardBackward(wl.chars.length, wl.edges, wl.lp).logZ
}

/**
 * Every segmentation of a word with its probability $P(x)$ and posterior $P(x \mid w)$ for the word $w$, most probable
 * first (at most `limit`; the number of segmentations grows exponentially with length).
 *
 * @param model The tokeniser.
 * @param word One word, already pre-tokenised, without the boundary prefix.
 * @param limit The most segmentations enumerated. When there are more, the first `limit` found (in depth-first order,
 *   shorter first pieces first) are kept and sorted, which need not be the `limit` most probable.
 * @returns Each segmentation's pieces, its probability and its posterior.
 *
 * @example The three segmentations of "pug"
 * const model = unigramLmModel(
 *   {
 *     pieces: ['h', 'u', 'g', 'hu', 'ug', 'p', 'pu', 'n', 'un', 'b', 'bu', 's', 'hug', 'gs', 'ugs'],
 *     probabilities: [15, 36, 20, 15, 20, 17, 17, 16, 16, 4, 4, 5, 15, 5, 5],
 *   },
 *   { boundary: '' },
 * )
 * for (const x of unigramLmSegmentations(model, 'pug'))
 *   print(x.pieces, 'P =', x.probability, 'posterior =', x.posterior)
 */
export function unigramLmSegmentations(
  model: UnigramLmModel,
  word: string,
  limit = 1000,
): { pieces: string[]; probability: number; posterior: number }[] {
  const wl = wordLattice(model, word)
  const n = wl.chars.length
  const logZ = forwardBackward(n, wl.edges, wl.lp).logZ
  const from = new Map<number, Edge[]>()
  for (const e of wl.edges) from.set(e.i, [...(from.get(e.i) ?? []), e])
  const out: { pieces: string[]; logP: number }[] = []
  const walk = (i: number, acc: string[], logP: number) => {
    if (out.length >= limit) return
    if (i === n) {
      out.push({ pieces: acc, logP })
      return
    }
    for (const e of from.get(i) ?? []) walk(e.j, [...acc, wl.pieces[e.id]], logP + wl.lp[e.id])
  }
  walk(0, [], 0)
  return out
    .sort((a, b) => b.logP - a.logP)
    .map((x) => ({ pieces: x.pieces, probability: Math.exp(x.logP), posterior: Math.exp(x.logP - logZ) }))
}

/**
 * Sample a segmentation with probability $\propto P(x)^\alpha$ (subword regularisation, Kudo 2018 §2.2): forward
 * filtering on the lattice with log-probabilities scaled by $\alpha$, then backward sampling from the end. $\alpha = 1$
 * samples the posterior; large $\alpha$ approaches Viterbi, and small $\alpha$ approaches the uniform distribution over
 * segmentations.
 *
 * @param s The random stream; one uniform draw is taken per piece sampled.
 * @param model The tokeniser.
 * @param word One word, already pre-tokenised, without the boundary prefix.
 * @param options `alpha`, the sharpness $\alpha$ (default 1).
 * @returns The sampled pieces, with their ranges in `word`.
 *
 * @example Sharp and flat sampling of "pug"
 * const model = unigramLmModel(
 *   { pieces: ['h', 'u', 'g', 'hu', 'ug', 'p', 'pu'], probabilities: [15, 36, 20, 15, 20, 17, 17] },
 *   { boundary: '' },
 * )
 * const s = stream(0)
 * print('one draw:', unigramLmSample(s, model, 'pug').map((p) => p.token))
 * for (const alpha of [1, 0.1]) {
 *   const seen = {}
 *   for (let k = 0; k < 300; k++) {
 *     const x = unigramLmSample(s, model, 'pug', { alpha }).map((p) => p.token).join(' ')
 *     seen[x] = (seen[x] ?? 0) + 1
 *   }
 *   print(`alpha = ${alpha}, 300 draws:`, seen)
 * }
 */
export function unigramLmSample(
  s: Stream,
  model: UnigramLmModel,
  word: string,
  options: { alpha?: number } = {},
): Piece[] {
  const alpha = options.alpha ?? 1
  const wl = wordLattice(model, word)
  const lp = wl.lp.map((x) => alpha * x)
  const n = wl.chars.length
  const { alpha: fwd } = forwardBackward(n, wl.edges, lp)
  const into = new Map<number, Edge[]>()
  for (const e of wl.edges) into.set(e.j, [...(into.get(e.j) ?? []), e])
  const path: Edge[] = []
  for (let j = n; j > 0;) {
    const cands = into.get(j)!
    const w = cands.map((e) => Math.exp(fwd[e.i] + lp[e.id] - fwd[j]))
    let u = uniform(s) as number
    let k = 0
    for (; k < cands.length - 1; k++) {
      u -= w[k]
      if (u < 0) break
    }
    path.push(cands[k])
    j = cands[k].i
  }
  return toPieces(wl, path.reverse())
}

/**
 * Encode text with a unigram tokeniser: pre-tokenise by the model's pattern, then Viterbi on each word.
 *
 * @param model The tokeniser.
 * @param text The text to encode.
 * @returns The tokens with their offsets into `text`; a piece that is only the boundary has a zero-width range.
 *
 * @example Two words
 * const model = unigramLmModel({ pieces: ['▁', 'a', 'b', '▁ab', 'ab'], probabilities: [0.1, 0.2, 0.2, 0.3, 0.2] })
 * const enc = unigramLmEncode(model, 'ab abab')
 * print('tokens =', enc.tokens)
 * print('offsets =', enc.offsets)
 */
export function unigramLmEncode(model: UnigramLmModel, text: string): Tokenisation {
  return encodeByWords(text, model.pattern, (w) => unigramLmSegment(model, w))
}

/**
 * The training corpus of a state, as the loss computation reads it (boundary-prefixed characters and counts).
 *
 * @param state A state of `unigramLmSteps`.
 * @param options The options it was trained with; only `boundary` is read.
 * @returns The corpus.
 *
 * @example The corpus, and the losses of the seed pieces on it
 * const state = run(unigramLmSteps({ hug: 10, pug: 5 }), undefined, 0)
 * const corpus = unigramLmCorpus(state)
 * print('chars =', corpus.chars, 'counts =', corpus.counts)
 * const loss = unigramLmLosses(state, corpus)
 * print(state.pieces.map((p, k) => `${p}: ${loss.data[k].toFixed(2)}`).join(', '))
 */
export function unigramLmCorpus(state: UnigramLmState, options: UnigramLmOptions = {}): UnigramLmCorpus {
  return corpusOf(state.words, toFlat(state.wordCounts), resolve(options).boundary)
}
