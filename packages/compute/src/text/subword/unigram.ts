/**
 * The unigram language-model tokeniser (Kudo 2018, "Subword regularization", §3): a vocabulary of pieces with
 * probabilities p(v), a segmentation x of a word scored by P(x) = Π p(x_i), the best segmentation found by Viterbi on
 * the lattice of pieces over the word's characters. Training starts from a large seed vocabulary (every character and
 * frequent substrings) and alternates EM, with expected piece counts from forward–backward on each lattice, with
 * pruning the pieces whose removal lowers the marginal likelihood least. Words carry a boundary prefix (SentencePiece's
 * "▁", Kudo & Richardson 2018).
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
  readonly kind: 'unigram-lm'
  readonly pieces: readonly string[]
  readonly logProbs: Tensor
  /** Prefixed to every word before segmentation (default "▁"; empty for none). */
  readonly boundary: string
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
  readonly piece: string
  readonly loss: number
}

/** The state of unigram training after `t` rounds. */
export interface UnigramLmState extends Status {
  readonly words: readonly string[]
  readonly wordCounts: Tensor
  readonly pieces: readonly string[]
  /** log p(v) per piece (float64 [V]). */
  readonly logProbs: Tensor
  /** Σ_w count_w log P(w), the corpus log marginal likelihood (nats) under the current pieces. */
  readonly logLikelihood: number
  /** The pieces removed by the last round, with their losses (smallest loss first). */
  readonly pruned: readonly PrunedPiece[]
  readonly done: boolean
}

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

/** One edge of a word's lattice: piece `id` spans code points [i, j). */
interface Edge {
  i: number
  j: number
  id: number
}

/** The code points of a word and the lattice edges of the pieces that match its substrings. */
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

const logAdd = (a: number, b: number) =>
  a === -Infinity ? b : b === -Infinity ? a : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)))

/** Forward and backward log-sums over the lattice (α[j]: all segmentations of [0, j); β[i]: of [i, n)). */
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

const pieceIndex = (pieces: readonly string[]) => new Map(pieces.map((p, k) => [p, k]))
const maxLen = (pieces: readonly string[]) => pieces.reduce((m, p) => Math.max(m, [...p].length), 1)

// ── EM and pruning ───────────────────────────────────────────────────────────────────────────────────────────────────

/** A training corpus as the unigram trainer holds it: each distinct word as boundary-prefixed code points, with counts. */
export interface UnigramLmCorpus {
  chars: string[][]
  counts: number[]
}

/** One EM iteration: expected piece counts by forward–backward, then p(v) ∝ expected count. */
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
 * The loss of each piece: Σ_w count_w (log P(w) − log P_{without v}(w)), the drop in the corpus log marginal
 * likelihood when piece v is removed with the other probabilities held fixed (Kudo 2018, §3.2 step 3). Infinite when
 * some word cannot be segmented without v. Returns float64 [V].
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

function corpusOf(words: readonly string[], counts: readonly number[], boundary: string): UnigramLmCorpus {
  return { chars: words.map((w) => [...(boundary + w)]), counts: [...counts] }
}

/** Seeds: every character, and the `seedSize` most frequent substrings of 2 … maxPieceLength characters. */
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
 * removal of the multi-character pieces with the smallest loss so that max(target, ⌊shrink · V⌋) remain (single
 * characters are always kept, so every word stays encodable). The round that reaches the target refits by EM and
 * sets `done`.
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

/** Train the unigram tokeniser to its target size (see {@link unigramLmSteps}). */
export function unigramLm(words: WordCountsLike, options: UnigramLmOptions = {}): UnigramLmModel {
  return unigramLmModel(run(unigramLmSteps(words, options), undefined, 1000), options)
}

// ── Segmenting with a model ──────────────────────────────────────────────────────────────────────────────────────────

interface WordLattice {
  chars: string[]
  /** [start, end) of each code point in the original word; boundary characters are zero-width at 0. */
  spans: [number, number][]
  edges: Edge[]
  lp: Float64Array
  /** Pieces of the model, then one unknown piece per uncovered character. */
  pieces: string[]
}

const latticeCache = new WeakMap<UnigramLmModel, { index: Map<string, number>; L: number; lp: Float64Array }>()

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

const toPieces = (wl: WordLattice, path: readonly Edge[]): Piece[] =>
  path.map((e) => ({ token: wl.pieces[e.id], start: wl.spans[e.i][0], end: wl.spans[e.j - 1][1] }))

/** The most probable segmentation of a word (Viterbi on its lattice), with its log-probability. */
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

/** Segment one word by Viterbi (the pieces with their ranges in the word). */
export function unigramLmSegment(model: UnigramLmModel, word: string): Piece[] {
  return unigramLmViterbi(model, word).pieces
}

/** log Σ_x P(x) over every segmentation x of a word: its marginal log-probability (forward algorithm). */
export function unigramLmMarginal(model: UnigramLmModel, word: string): number {
  const wl = wordLattice(model, word)
  return forwardBackward(wl.chars.length, wl.edges, wl.lp).logZ
}

/**
 * Every segmentation of a word with its probability P(x) and posterior P(x | word), most probable first (at most
 * `limit`, default 1000; the number of segmentations grows exponentially with length).
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
 * Sample a segmentation with probability ∝ P(x)^α (subword regularisation, Kudo 2018 §2.2): forward filtering on the
 * lattice with log-probabilities scaled by α, then backward sampling from the end. α = 1 samples the posterior; large
 * α approaches Viterbi.
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

/** Encode text with a unigram tokeniser: pre-tokenise by the model's pattern, then Viterbi on each word. */
export function unigramLmEncode(model: UnigramLmModel, text: string): Tokenisation {
  return encodeByWords(text, model.pattern, (w) => unigramLmSegment(model, w))
}

/** The training corpus of a state, as the loss computation reads it (boundary-prefixed characters and counts). */
export function unigramLmCorpus(state: UnigramLmState, options: UnigramLmOptions = {}): UnigramLmCorpus {
  return corpusOf(state.words, toFlat(state.wordCounts), resolve(options).boundary)
}
