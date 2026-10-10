/**
 * Two neural hyphenation taggers that give each letter of a word the probability that a hyphen follows it.
 *
 * - `WindowTagger`, after NETtalk (Sejnowski and Rosenberg 1987, "Parallel networks that learn to pronounce English
 *   text", Complex Systems 1): a window of $2r + 1 = 7$ characters centred on a letter, each embedded, concatenated
 *   and passed through a one-hidden-layer MLP to one logit for "a hyphen after the centre letter". Characters beyond
 *   the word read as `.` (the boundary) and then `_` (padding).
 * - `BiRnnTagger`: the word as `.word.` padded with `_` to a fixed width, embedded, read by an LSTM in each direction
 *   (`aifn-compute/nn/layers` cells), and mapped to one logit per position. The logit at the position of letter $i$
 *   scores a hyphen after it. (A one-layer transformer tagger of the same budget learned more slowly in browser time.)
 *
 * Both are trained by binary cross-entropy on the dictionary's labels. Words are read as ids into `HYPHEN_ALPHABET`,
 * so they must be lower case a–z; any other character throws `DomainError`. Beside each tagger are its probabilities
 * for whole words and an occlusion saliency: how much each character moves a gap's logit.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { child, type Stream } from 'aifn-compute/foundation/random'
import {
  concat,
  fromData,
  permute,
  reshape,
  take,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { normalInit } from 'aifn-compute/nn/init'
import {
  childContext,
  GruCell,
  LstmCell,
  linear,
  Linear,
  Mlp,
  tap,
  unrollRecurrent,
  type CellParams,
  type Context,
  type LinearParams,
} from 'aifn-compute/nn/layers'
import type { Params } from 'aifn-compute/foundation/pytree'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The symbols the taggers read: padding `_`, the word boundary `.`, then a–z. */
export const HYPHEN_ALPHABET: readonly string[] = ['_', '.', ...'abcdefghijklmnopqrstuvwxyz']
/** The id of the padding symbol `_`. */
const PAD = 0
/** The id of the word-boundary symbol `.`. */
const BOUNDARY = 1
/**
 * The id of a letter. Throws `DomainError` for anything but a–z.
 *
 * @param c One character.
 * @returns Its index in `HYPHEN_ALPHABET` (2 to 27).
 */
const idOf = (c: string) => {
  const k = HYPHEN_ALPHABET.indexOf(c)
  if (k < 2) throw new DomainError('hyphenation taggers', `hyphenation taggers: '${c}' is not a letter a–z`)
  return k
}

/**
 * The window of $2r + 1$ symbol ids centred on letter $i$ of a word: the positions just outside the word read as `.`
 * and those further out as `_`.
 *
 * @param word The word, in a–z.
 * @param i The centre letter, from 0.
 * @param radius The radius $r$: symbols each side of the centre.
 * @returns The $2r + 1$ ids into `HYPHEN_ALPHABET`, left to right.
 *
 * @example The window of the first letter of "cat"
 * const ids = letterWindow('cat', 0, 3)
 * print(ids, ids.map((k) => HYPHEN_ALPHABET[k]).join(''))
 */
export function letterWindow(word: string, i: number, radius: Size): number[] {
  const out: number[] = []
  for (let j = i - radius; j <= i + radius; j++) {
    if (j >= 0 && j < word.length) out.push(idOf(word[j]))
    else out.push(j === -1 || j === word.length ? BOUNDARY : PAD)
  }
  return out
}

/**
 * `.word.` as ids, padded with `_` to `width`. Throws `DomainError` if the word is longer than `width - 2`.
 *
 * @param word The word, in a–z.
 * @param width The length of the row, at least the word's length plus 2.
 * @returns `width` ids into `HYPHEN_ALPHABET`: letter $i$ at position $i + 1$.
 *
 * @example "cat" in a row of eight
 * const ids = dottedIds('cat', 8)
 * print(ids, ids.map((k) => HYPHEN_ALPHABET[k]).join(''))
 */
export function dottedIds(word: string, width: Size): number[] {
  if (word.length + 2 > width)
    throw new DomainError('hyphenation taggers', `hyphenation taggers: '${word}' is longer than ${width - 2}`)
  const ids = [BOUNDARY, ...[...word].map(idOf), BOUNDARY]
  while (ids.length < width) ids.push(PAD)
  return ids
}

// ── Window MLP ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** The architecture of a `WindowTagger`. */
export type WindowTaggerConfig = {
  /** Characters each side of the centre letter (default 3: a 7-character window, as NETtalk). */
  radius?: Size
  /** Width of each character's embedding (default 8). */
  embedding?: Size
  /** Hidden units (default 32). */
  hidden?: Size
}

/**
 * Parameters of a `WindowTagger`: `embedding`, the character embedding ($V \times e$, one row per symbol of
 * `HYPHEN_ALPHABET`), and `mlp`, the MLP's layers.
 */
export type WindowTaggerParams = { embedding: Tensor; mlp: Params[] }

/** A window tagger: maps $N$ windows of ids ($N \times (2r + 1)$) to $N$ hyphen logits. */
export type WindowTagger = {
  /** Always `'window'`. */
  readonly kind: 'window'
  /** The architecture, defaults filled in. */
  readonly config: Required<WindowTaggerConfig>
  /** Fresh parameters drawn from a stream. */
  init(s: Stream): WindowTaggerParams
  /** The logits ($N$) of windows of ids ($N \times (2r + 1)$, int32); taps `embedding`, `mlp` and `logits`. */
  apply(params: WindowTaggerParams, windows: Tensor, ctx?: Context): Value
}

/**
 * A NETtalk-style window MLP (see the file comment): each window's symbols are embedded, concatenated and passed
 * through one tanh hidden layer to a logit. The embedding is drawn by `normalInit(0.5)`.
 *
 * @param config The radius, embedding width and hidden units; each left out takes its default.
 * @returns The tagger: its configuration, `init` and `apply`.
 *
 * @example An untrained tagger's logits for the first two gaps of "table"
 * const m = WindowTagger({ radius: 2 })
 * const p = m.init(stream(0))
 * print('config', m.config, 'embedding', p.embedding.shape)
 * const x = fromData(Int32Array.from([...letterWindow('table', 0, 2), ...letterWindow('table', 1, 2)]), [2, 5])
 * print('logits', m.apply(p, x))
 */
export function WindowTagger(config: WindowTaggerConfig = {}): WindowTagger {
  const c = { radius: 3, embedding: 8, hidden: 32, ...config }
  const width = 2 * c.radius + 1
  const mlp = Mlp([width * c.embedding, c.hidden, 1], { activation: 'tanh' })
  return {
    kind: 'window',
    config: c,
    init: (s) => ({
      embedding: normalInit(0.5)(child(s, 'embedding'), [HYPHEN_ALPHABET.length, c.embedding], {
        fanIn: HYPHEN_ALPHABET.length,
        fanOut: c.embedding,
      }),
      mlp: mlp.init(child(s, 'mlp')),
    }),
    apply: (p, windows, ctx) => {
      const n = windows.shape[0]
      const e = tap(childContext(ctx, 'embedding'), take(p.embedding, windows))
      const h = reshape(e, [n, width * c.embedding])
      return tap(childContext(ctx, 'logits'), reshape(mlp.apply(p.mlp, h, childContext(ctx, 'mlp')), [n]))
    },
  }
}

// ── Probabilities of one word ────────────────────────────────────────────────────────────────────────────────────────

/**
 * The logistic function.
 *
 * @param z A logit.
 * @returns $1 / (1 + e^{-z})$.
 */
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))

/**
 * The probability of a hyphen after letter $i$, for each letter of each word but the last, by a window tagger (all
 * windows in one batch).
 *
 * @param model The window tagger.
 * @param params Its parameters.
 * @param words The words, in a–z.
 * @returns One list per word, of its length minus 1 (empty for a one-letter word).
 *
 * @example An untrained tagger is unsure everywhere
 * const m = WindowTagger()
 * print(windowProbabilities(m, m.init(stream(0)), ['table', 'cat', 'a']))
 */
export function windowProbabilities(
  model: WindowTagger,
  params: WindowTaggerParams,
  words: readonly string[],
): number[][] {
  const windows: number[] = []
  const r = model.config.radius
  for (const w of words) for (let i = 0; i + 1 < w.length; i++) windows.push(...letterWindow(w, i, r))
  const n = windows.length / (2 * r + 1)
  if (n === 0) return words.map(() => [])
  const logits = toFlat(unwrap(model.apply(params, fromData(Int32Array.from(windows), [n, 2 * r + 1]))) as Tensor)
  let k = 0
  return words.map((w) => Array.from({ length: Math.max(0, w.length - 1) }, () => sigmoid(logits[k++])))
}

/**
 * Occlusion saliency of a window tagger on one word: for each letter $i$ (but the last) and each window slot $j$, the
 * drop in the hyphen logit when slot $j$ is replaced by padding (positive: that symbol pushed towards a hyphen). A
 * slot that is padding already has saliency 0.
 *
 * @param model The window tagger.
 * @param params Its parameters.
 * @param word The word, in a–z.
 * @returns $n - 1$ rows of $2r + 1$ drops, for a word of $n$ letters (none for a one-letter word).
 *
 * @example Which symbols move the two gaps of "cat"
 * const m = WindowTagger()
 * print(windowSaliency(m, m.init(stream(0)), 'cat'))
 */
export function windowSaliency(model: WindowTagger, params: WindowTaggerParams, word: string): number[][] {
  const r = model.config.radius
  const k = 2 * r + 1
  const rows: number[] = []
  for (let i = 0; i + 1 < word.length; i++) {
    const base = letterWindow(word, i, r)
    rows.push(...base)
    for (let j = 0; j < k; j++) rows.push(...base.map((v, m) => (m === j ? PAD : v)))
  }
  const n = rows.length / k
  if (n === 0) return []
  const z = toFlat(unwrap(model.apply(params, fromData(Int32Array.from(rows), [n, k]))) as Tensor)
  return Array.from({ length: word.length - 1 }, (_, i) => {
    const at = i * (k + 1)
    return Array.from({ length: k }, (_, j) => z[at] - z[at + 1 + j])
  })
}

// ── Bidirectional recurrent tagger ─────────────────────────────────────────────────────────────────────────────────────────

/** The architecture of a `BiRnnTagger`. */
export type BiRnnTaggerConfig = {
  /** Row width: `.word.` plus padding, so at least the longest word's length plus 2. */
  width: Size
  /** Symbol embedding width (default 12). */
  embedding?: Size
  /** Hidden units per direction (default 24). */
  hidden?: Size
  /** The recurrent cell (default `lstm`, with forget-gate bias 1). */
  cell?: 'lstm' | 'gru'
}

/** Parameters of a `BiRnnTagger`. */
export type BiRnnTaggerParams = {
  /** The symbol embedding ($V \times e$, one row per symbol of `HYPHEN_ALPHABET`). */
  embedding: Tensor
  /** The left-to-right cell. */
  forward: CellParams
  /** The right-to-left cell. */
  backward: CellParams
  /** The linear layer from both hidden states to one logit. */
  head: LinearParams
}

/** A bidirectional recurrent tagger: maps rows of ids ($N \times W$) to hyphen logits ($N \times W$). */
export type BiRnnTagger = {
  /** Always `'birnn'`. */
  readonly kind: 'birnn'
  /** The architecture, defaults filled in. */
  readonly config: Required<BiRnnTaggerConfig>
  /** Fresh parameters drawn from a stream. */
  init(s: Stream): BiRnnTaggerParams
  /** Logits ($N \times W$) for rows of ids ($N \times W$, int32, as `dottedIds` makes them). */
  apply(params: BiRnnTaggerParams, ids: Tensor, ctx?: Context): Value
}

/**
 * A bidirectional LSTM (or GRU) over `.word.` (Schuster and Paliwal 1997; Graves and Schmidhuber 2005): one cell reads
 * the row left to right, another right to left, and a linear layer maps both hidden states at each position to one
 * logit. With a tapping context it records `forward` and `backward` (each cell's hidden states) and `logits`.
 *
 * @param config The row width (required), embedding width, hidden units per direction and cell.
 * @returns The tagger: its configuration, `init` and `apply`.
 *
 * @example An untrained tagger's logits along ".cat."
 * const m = BiRnnTagger({ width: 7 })
 * const p = m.init(stream(0))
 * print('config', m.config)
 * print('logits', m.apply(p, fromData(Int32Array.from(dottedIds('cat', 7)), [1, 7])))
 */
export function BiRnnTagger(config: BiRnnTaggerConfig): BiRnnTagger {
  const c = { embedding: 12, hidden: 24, cell: 'lstm' as const, ...config }
  const cell = c.cell === 'gru' ? GruCell(c.embedding, c.hidden) : LstmCell(c.embedding, c.hidden, { forgetBias: 1 })
  const head = Linear(2 * c.hidden, 1)
  const reversed = Array.from({ length: c.width }, (_, i) => c.width - 1 - i)
  return {
    kind: 'birnn',
    config: c,
    init: (s) => ({
      embedding: normalInit(0.5)(child(s, 'embedding'), [HYPHEN_ALPHABET.length, c.embedding], {
        fanIn: HYPHEN_ALPHABET.length,
        fanOut: c.embedding,
      }),
      forward: cell.init(child(s, 'forward')),
      backward: cell.init(child(s, 'backward')),
      head: head.init(child(s, 'head')),
    }),
    apply: (p, ids, ctx) => {
      const [n, w] = ids.shape
      // Time first: [W, N, e].
      const xs = permute(take(p.embedding, ids), [1, 0, 2])
      const fwd = unrollRecurrent(cell, p.forward, xs, undefined, childContext(ctx, 'forward')).outputs
      const back = unrollRecurrent(cell, p.backward, take(xs, reversed), undefined, childContext(ctx, 'backward'))
      const bwd = take(back.outputs, reversed)
      const h = permute(concat([fwd, bwd], 2), [1, 0, 2])
      return tap(childContext(ctx, 'logits'), reshape(linear(h, p.head.weight, p.head.bias), [n, w]))
    },
  }
}

/**
 * The probability of a hyphen after letter $i$, for each letter of each word but the last, by a recurrent tagger (in
 * batches of 256 words). Throws `DomainError` for a word longer than the tagger's width allows.
 *
 * @param model The recurrent tagger.
 * @param params Its parameters.
 * @param words The words, in a–z.
 * @returns One list per word, of its length minus 1 (empty for a one-letter word).
 *
 * @example An untrained tagger is unsure everywhere
 * const m = BiRnnTagger({ width: 7 })
 * print(rnnProbabilities(m, m.init(stream(0)), ['table', 'cat']))
 */
export function rnnProbabilities(model: BiRnnTagger, params: BiRnnTaggerParams, words: readonly string[]): number[][] {
  const W = model.config.width
  const out: number[][] = []
  for (let start = 0; start < words.length; start += 256) {
    const chunk = words.slice(start, start + 256)
    const ids = fromData(Int32Array.from(chunk.flatMap((w) => dottedIds(w, W))), [chunk.length, W])
    const logits = toFlat(unwrap(model.apply(params, ids)) as Tensor)
    // Letter i of the word sits at position i + 1 of `.word.`.
    chunk.forEach((w, r) =>
      out.push(Array.from({ length: Math.max(0, w.length - 1) }, (_, i) => sigmoid(logits[r * W + i + 1]))),
    )
  }
  return out
}

/**
 * Occlusion saliency of a recurrent tagger on one word: for each letter $i$ (but the last) and each letter $j$ of the
 * word, the drop in the logit of a hyphen after $i$ when letter $j$ is replaced by padding (positive: that letter
 * pushed towards a hyphen).
 *
 * @param model The recurrent tagger.
 * @param params Its parameters.
 * @param word The word, in a–z, at most the tagger's width minus 2 letters.
 * @returns $n - 1$ rows of $n$ drops, for a word of $n$ letters.
 *
 * @example Which letters move the two gaps of "cat"
 * const m = BiRnnTagger({ width: 7 })
 * print(rnnSaliency(m, m.init(stream(0)), 'cat'))
 */
export function rnnSaliency(model: BiRnnTagger, params: BiRnnTaggerParams, word: string): number[][] {
  const W = model.config.width
  const base = dottedIds(word, W)
  const rows = [base, ...[...word].map((_, j) => base.map((v, k) => (k === j + 1 ? PAD : v)))]
  const z = toFlat(unwrap(model.apply(params, fromData(Int32Array.from(rows.flat()), [rows.length, W]))) as Tensor)
  return Array.from({ length: Math.max(0, word.length - 1) }, (_, i) =>
    Array.from({ length: word.length }, (_, j) => z[i + 1] - z[(j + 1) * W + i + 1]),
  )
}
