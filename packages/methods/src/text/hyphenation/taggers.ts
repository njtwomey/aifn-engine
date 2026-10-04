/**
 * Two neural hyphenation taggers that give each letter of a word the probability that a hyphen follows it.
 *
 * - `WindowTagger`, after NETtalk (Sejnowski and Rosenberg 1987, "Parallel networks that learn to pronounce English
 *   text", Complex Systems 1): a window of 2r + 1 = 7 characters centred on a letter, each embedded, concatenated and
 *   passed through a one-hidden-layer MLP to one logit for "a hyphen after the centre letter". Characters beyond the
 *   word read as `.` (the boundary) and then `_` (padding).
 * - `BiRnnTagger`: the word as `.word.` padded with `_` to a fixed width, embedded, read by an LSTM in each direction
 *   (`aifn-compute/nn/layers` cells), and mapped to one logit per position. The logit at the position of letter i scores a
 *   hyphen after it. (A one-layer transformer tagger of the same budget learned more slowly in browser time.)
 *
 * Both are trained by binary cross-entropy on the dictionary's labels.
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
const PAD = 0
const BOUNDARY = 1
const idOf = (c: string) => {
  const k = HYPHEN_ALPHABET.indexOf(c)
  if (k < 2) throw new DomainError('hyphenation taggers', `hyphenation taggers: '${c}' is not a letter a–z`)
  return k
}

/** The window of `2·radius + 1` symbol ids centred on letter i of a word. */
export function letterWindow(word: string, i: number, radius: Size): number[] {
  const out: number[] = []
  for (let j = i - radius; j <= i + radius; j++) {
    if (j >= 0 && j < word.length) out.push(idOf(word[j]))
    else out.push(j === -1 || j === word.length ? BOUNDARY : PAD)
  }
  return out
}

/** `.word.` as ids padded with `_` to `width` (at least the word's length + 2). */
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

/** Parameters of a `WindowTagger`: the character embedding [V, e] and the MLP's layers. */
export type WindowTaggerParams = { embedding: Tensor; mlp: Params[] }

/** A window tagger: windows of ids [N, 2r + 1] → hyphen logits [N]. */
export type WindowTagger = {
  readonly kind: 'window'
  readonly config: Required<WindowTaggerConfig>
  init(s: Stream): WindowTaggerParams
  apply(params: WindowTaggerParams, windows: Tensor, ctx?: Context): Value
}

/** A NETtalk-style window MLP (see the module comment). */
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

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))

/** P(hyphen after letter i) for each letter of each word but the last, by a window tagger. */
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
 * Occlusion saliency of a window tagger on one word: for each letter i (but the last) and each window slot j, the
 * drop in the hyphen logit when slot j is replaced by padding ([n − 1][2r + 1]; positive: that character pushed
 * towards a hyphen).
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
  /** Row width: `.word.` plus padding. */
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
  embedding: Tensor
  forward: CellParams
  backward: CellParams
  head: LinearParams
}

/** A bidirectional recurrent tagger: rows of ids [N, W] → hyphen logits [N, W]. */
export type BiRnnTagger = {
  readonly kind: 'birnn'
  readonly config: Required<BiRnnTaggerConfig>
  init(s: Stream): BiRnnTaggerParams
  /** Logits [N, W] for ids [N, W]. */
  apply(params: BiRnnTaggerParams, ids: Tensor, ctx?: Context): Value
}

/**
 * A bidirectional LSTM (or GRU) over `.word.` (Schuster and Paliwal 1997; Graves and Schmidhuber 2005): one cell reads
 * the row left to right, another right to left, and a linear layer maps both hidden states at each position to one
 * logit. With a tapping context it records `forward` and `backward` (each cell's hidden states) and `logits`.
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

/** P(hyphen after letter i) for each letter of each word but the last, by a recurrent tagger. */
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
 * Occlusion saliency of a recurrent tagger on one word: for each letter i (but the last) and each letter j of the
 * word, the drop in the logit of a hyphen after i when letter j is replaced by padding ([n − 1][n]).
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
