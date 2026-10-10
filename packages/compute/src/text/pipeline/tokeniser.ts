/**
 * The tokeniser pipeline (as Hugging Face `tokenizers`): normaliser, then pre-tokeniser, model and post-processor,
 * and a decoder back. A `Tokeniser` is plain data, its stages tagged objects, so it can be stored, sent to a worker and
 * rebuilt stage by stage. Encoding returns ids, tokens, offsets into the original text (through normalisation), word
 * and sequence ids, token-type ids and the attention and special-token masks; truncation keeps the overflowing
 * windows; decoding inverts the pre-tokeniser and model.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Tokenisation } from 'aifn-compute/text/tokenise'
import { tokenId } from 'aifn-compute/text/vocabulary'
import { aligned, alignedSlice, originalSpan } from '../aligned'
import { applyDecoder, byteFallbackDecoder, decoderSequence, fuseDecoder, type Decoder } from './decoders'
import { byteStage, modelSegment, type SegmentOptions, type TokeniserModel } from './models'
import { applyNormaliser, type Normaliser } from './normalisers'
import {
  addedTokens,
  pairLengths,
  templateProcessor,
  truncationWindows,
  type Padding,
  type PostProcessor,
  type Truncation,
} from './post-processors'
import { applyPreTokeniser, type PreTokeniser } from './pre-tokenisers'

/** A tokeniser pipeline. */
export interface Tokeniser {
  /** Marks the value as a tokeniser. */
  readonly kind: 'tokeniser'
  /** The first stage, or null to leave the text as typed. */
  readonly normaliser: Normaliser | null
  /** The second stage, or null to give the model the whole text as one pre-token. */
  readonly preTokeniser: PreTokeniser | null
  /** The model that segments each pre-token, with the vocabulary. */
  readonly model: TokeniserModel
  /** The template that adds special tokens, or null for none. */
  readonly postProcessor: PostProcessor | null
  /** The decoder from tokens back to text, or null to join tokens by spaces. */
  readonly decoder: Decoder | null
  /** Special tokens recognised verbatim in the input before normalisation (added tokens); they must have ids. */
  readonly specials: readonly string[]
  /** How long an encoding may be, or null for no limit. */
  readonly truncation: Truncation | null
  /** How encodings are padded, or null for none. */
  readonly padding: Padding | null
}

/** The stages of a tokeniser; only the model is required. */
export interface TokeniserParts {
  /** The normaliser (default none). */
  normaliser?: Normaliser | null
  /** The pre-tokeniser (default none). */
  preTokeniser?: PreTokeniser | null
  /** The model. */
  model: TokeniserModel
  /** The post-processor (default none). */
  postProcessor?: PostProcessor | null
  /** The decoder (default none). */
  decoder?: Decoder | null
  /** Special tokens matched verbatim in the input (default the model vocabulary's specials). */
  specials?: readonly string[]
  /** Truncation settings (default none). */
  truncation?: Truncation | null
  /** Padding settings (default none). */
  padding?: Padding | null
}

/**
 * Assemble a tokeniser from its stages (missing stages are null; specials default to the model's).
 *
 * @param parts The stages; only `model` is required.
 * @returns The tokeniser.
 *
 * @example A BERT-style pipeline turning two sentences into ids
 * const vocabulary = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', 'the', 'cat', 'sat', 'on', 'mat', '##s', '.']
 * const t = tokeniser({
 *   normaliser: lowercaseNormaliser(),
 *   preTokeniser: bertPreTokeniser(),
 *   model: wordPieceStage(vocabulary, { specials: ['[UNK]', '[CLS]', '[SEP]', '[PAD]'] }),
 *   postProcessor: bertProcessor(),
 *   decoder: wordPieceDecoder(),
 * })
 * const e = encodeText(t, 'The cat sat.', 'The mats.')
 * print('tokens =', e.tokens)
 * print('ids =', e.ids)
 * print('type ids =', e.typeIds)
 * print('decoded =', decodeIds(t, e.ids))
 */
export function tokeniser(parts: TokeniserParts): Tokeniser {
  return {
    kind: 'tokeniser',
    normaliser: parts.normaliser ?? null,
    preTokeniser: parts.preTokeniser ?? null,
    model: parts.model,
    postProcessor: parts.postProcessor ?? null,
    decoder: parts.decoder ?? null,
    specials: parts.specials ?? parts.model.vocabulary.specials,
    truncation: parts.truncation ?? null,
    padding: parts.padding ?? null,
  }
}

/** A tokeniser's stages before its model is trained (see `trainTokeniser`). */
export type UntrainedTokeniser = Omit<TokeniserParts, 'model'> & { model?: TokeniserModel }

/**
 * The tokeniser with some stages replaced.
 *
 * @param t The tokeniser; not modified.
 * @param parts The stages to replace; a stage given as null is removed.
 * @returns A new tokeniser.
 *
 * @example The same model without, then with, BERT's template
 * const model = wordPieceStage(['[UNK]', '[CLS]', '[SEP]', 'hi', '!'])
 * const t = tokeniser({ preTokeniser: bertPreTokeniser(), model })
 * print(encodeText(t, 'hi!').tokens)
 * print(encodeText(withStages(t, { postProcessor: bertProcessor() }), 'hi!').tokens)
 */
export function withStages(t: Tokeniser, parts: Partial<TokeniserParts>): Tokeniser {
  return tokeniser({ ...t, ...parts })
}

/**
 * An encoding: per token, its string, id (int32 [n]), [start, end) offsets in UTF-16 code units of the text of its
 * sequence (int32 [n, 2]; special tokens [0, 0)), the index of the pre-token it came from (`wordIds`, $-1$ for special
 * tokens and padding), the sequence it belongs to (`sequenceIds`: 0, 1 or $-1$), the token-type id, and the attention
 * mask (1 for real tokens, 0 for padding) and special-token mask (1 for the template's specials and padding; 0 for
 * specials typed in the text). `overflowing` holds the windows truncation cut off, each a full encoding.
 */
export interface Encoding {
  /** Marks the value as an encoding. */
  readonly kind: 'encoding'
  /** The input texts: one, or two for a pair. */
  readonly sources: readonly string[]
  /** The token strings. */
  readonly tokens: readonly string[]
  /** The token ids (int32 [n]). */
  readonly ids: Tensor
  /** The [start, end) offsets into the text of each token's sequence (int32 [n, 2]). */
  readonly offsets: Tensor
  /** The pre-token each token came from, counted per sequence (int32 [n]). */
  readonly wordIds: Tensor
  /** The sequence each token belongs to (int32 [n]). */
  readonly sequenceIds: Tensor
  /** The token-type ids of the template (int32 [n]). */
  readonly typeIds: Tensor
  /** 1 for real tokens, 0 for padding (int32 [n]). */
  readonly attentionMask: Tensor
  /** 1 for the template's specials and padding, else 0 (int32 [n]). */
  readonly specialTokensMask: Tensor
  /** The windows truncation cut off, each a full encoding. */
  readonly overflowing: readonly Encoding[]
}

/** An encoding under construction: plain arrays, one entry per token (two for `offsets`). */
interface Draft {
  /** The token strings. */
  tokens: string[]
  /** The token ids. */
  ids: number[]
  /** The offsets, start and end per token, flat. */
  offsets: number[]
  /** The pre-token index of each token, or $-1$. */
  wordIds: number[]
  /** The sequence of each token, or $-1$. */
  sequenceIds: number[]
  /** The token-type ids. */
  typeIds: number[]
  /** The attention mask. */
  attention: number[]
  /** The special-token mask. */
  special: number[]
}

/**
 * A draft with no tokens.
 *
 * @returns The draft.
 */
const emptyDraft = (): Draft => ({
  tokens: [],
  ids: [],
  offsets: [],
  wordIds: [],
  sequenceIds: [],
  typeIds: [],
  attention: [],
  special: [],
})

/**
 * The tokens [s, e) of a draft.
 *
 * @param d The draft; not modified.
 * @param s The first token kept.
 * @param e One past the last token kept.
 * @returns A new draft.
 */
function sliceDraft(d: Draft, s: number, e: number): Draft {
  return {
    tokens: d.tokens.slice(s, e),
    ids: d.ids.slice(s, e),
    offsets: d.offsets.slice(2 * s, 2 * e),
    wordIds: d.wordIds.slice(s, e),
    sequenceIds: d.sequenceIds.slice(s, e),
    typeIds: d.typeIds.slice(s, e),
    attention: d.attention.slice(s, e),
    special: d.special.slice(s, e),
  }
}

/**
 * Append a draft's tokens to another, optionally relabelling their type and sequence ids.
 *
 * @param to The draft appended to; modified.
 * @param d The draft whose tokens are appended.
 * @param typeId The type id to give them; left out, they keep their own.
 * @param sequenceId The sequence id to give them; left out, they keep their own.
 */
function append(to: Draft, d: Draft, typeId?: number, sequenceId?: number): void {
  to.tokens.push(...d.tokens)
  to.ids.push(...d.ids)
  to.offsets.push(...d.offsets)
  to.wordIds.push(...d.wordIds)
  to.sequenceIds.push(...(sequenceId === undefined ? d.sequenceIds : d.ids.map(() => sequenceId)))
  to.typeIds.push(...(typeId === undefined ? d.typeIds : d.ids.map(() => typeId)))
  to.attention.push(...d.attention)
  to.special.push(...d.special)
}

/**
 * An int32 tensor of numbers.
 *
 * @param x The numbers.
 * @param shape The shape (default a vector).
 * @returns The tensor.
 */
const int32 = (x: readonly number[], shape?: number[]) => fromData(Int32Array.from(x), shape)

/**
 * An encoding from a draft.
 *
 * @param d The draft.
 * @param sources The input texts.
 * @param overflowing The encodings of the windows truncation cut off (default none).
 * @returns The encoding, its arrays as int32 tensors.
 */
function finish(d: Draft, sources: readonly string[], overflowing: readonly Encoding[] = []): Encoding {
  return {
    kind: 'encoding',
    sources,
    tokens: d.tokens,
    ids: int32(d.ids),
    offsets: int32(d.offsets, [d.tokens.length, 2]),
    wordIds: int32(d.wordIds),
    sequenceIds: int32(d.sequenceIds),
    typeIds: int32(d.typeIds),
    attentionMask: int32(d.attention),
    specialTokensMask: int32(d.special),
    overflowing,
  }
}

/**
 * A draft from an encoding, to pad it further.
 *
 * @param e The encoding.
 * @returns Its tokens as plain arrays (overflowing windows left out).
 */
function draftOf(e: Encoding): Draft {
  const a = (t: Tensor) => Array.from(toFlat(t))
  return {
    tokens: [...e.tokens],
    ids: a(e.ids),
    offsets: a(e.offsets),
    wordIds: a(e.wordIds),
    sequenceIds: a(e.sequenceIds),
    typeIds: a(e.typeIds),
    attention: a(e.attentionMask),
    special: a(e.specialTokensMask),
  }
}

/**
 * A string escaped for use as a literal in a regular expression.
 *
 * @param s The string.
 * @returns It with every regular-expression metacharacter backslash-escaped.
 */
const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\/-]/gu, '\\$&')

/** Options of {@link encodeText} and {@link encodeBatch}. */
export interface EncodeOptions extends SegmentOptions {
  /** Add the post-processor's special tokens (default true). */
  addSpecialTokens?: boolean
}

/**
 * The splits to keep when truncation limits a sequence to `max` tokens, as `tokenizers` (0.22 and later) does before
 * truncating: splits (pre-tokens, and specials typed in the text) are taken from the kept end until they hold at least
 * `max` tokens, and the rest are dropped, so overflowing windows never reach beyond them. A special split counts but
 * never ends the run; the next pre-token does. Returns the kept range [first, last) of split indices.
 *
 * @param sizes The number of tokens of each split, in order.
 * @param special Whether each split is a special token typed in the text.
 * @param max The maximum length the splits must reach.
 * @param direction `right` keeps the start, `left` the end.
 * @returns The kept range of split indices; every split when they hold fewer than `max` tokens.
 */
function limitSplits(
  sizes: readonly number[],
  special: readonly boolean[],
  max: number,
  direction: 'right' | 'left',
): [number, number] {
  let total = 0
  if (direction === 'right') {
    for (let i = 0; i < sizes.length; i++) {
      total += sizes[i]!
      if (!special[i] && total >= max) return [0, i + 1]
    }
    return [0, sizes.length]
  }
  for (let i = sizes.length - 1; i >= 0; i--) {
    total += sizes[i]!
    if (!special[i] && total >= max) return [i, sizes.length]
  }
  return [0, sizes.length]
}

/**
 * The tokens of one text through normaliser, pre-tokeniser and model, before post-processing. With `limit`, only the
 * splits {@link limitSplits} keeps are returned, their word ids counted from the first kept split.
 *
 * @param t The tokeniser.
 * @param text The text of one sequence.
 * @param options `stream`, `dropout` and `upTo`, passed to the model.
 * @param limit The maximum length and direction to keep splits for, or null to keep every split.
 * @returns The tokens, with sequence and type ids 0.
 */
function encodeTemplateRows(
  t: Tokeniser,
  text: string,
  options: EncodeOptions,
  limit: { max: number; direction: 'right' | 'left' } | null = null,
): Draft {
  const d = emptyDraft()
  // Token index where each split (one word id each) starts, and whether it is a special typed in the text.
  const starts: number[] = []
  const special: boolean[] = []
  const whole = aligned(text)
  // Special tokens typed in the text are matched verbatim first; the stretches between them go through the pipeline.
  const specials = [...t.specials].filter((s) => s.length > 0).sort((a, b) => b.length - a.length)
  const re = specials.length > 0 ? new RegExp(specials.map(escapeRe).join('|'), 'gu') : null
  const stretches: { s: number; e: number; special: string | null }[] = []
  let at = 0
  if (re)
    for (const m of text.matchAll(re)) {
      if (m.index > at) stretches.push({ s: at, e: m.index, special: null })
      stretches.push({ s: m.index, e: m.index + m[0].length, special: m[0] })
      at = m.index + m[0].length
    }
  if (at < text.length) stretches.push({ s: at, e: text.length, special: null })
  let word = 0
  for (const st of stretches) {
    if (st.special !== null) {
      starts.push(d.ids.length)
      special.push(true)
      d.tokens.push(st.special)
      d.ids.push(tokenId(t.model.vocabulary, st.special))
      d.offsets.push(st.s, st.e)
      d.wordIds.push(word++)
      d.sequenceIds.push(0)
      d.typeIds.push(0)
      d.attention.push(1)
      d.special.push(0) // as tokenizers: the mask marks what the template and padding add, not specials in the text
      continue
    }
    let a = alignedSlice(whole, st.s, st.e)
    if (t.normaliser) a = applyNormaliser(t.normaliser, a)
    for (const part of applyPreTokeniser(t.preTokeniser, [a])) {
      starts.push(d.ids.length)
      special.push(false)
      for (const m of modelSegment(t.model, part.text, options)) {
        const [s, e] = originalSpan(part, m.start, m.end)
        d.tokens.push(m.token)
        d.ids.push(m.id)
        d.offsets.push(s, e)
        d.wordIds.push(word)
        d.sequenceIds.push(0)
        d.typeIds.push(0)
        d.attention.push(1)
        d.special.push(0)
      }
      word++
    }
  }
  if (!limit) return d
  starts.push(d.ids.length)
  const sizes = special.map((_, i) => starts[i + 1]! - starts[i]!)
  const [first, last] = limitSplits(sizes, special, limit.max, limit.direction)
  const kept = sliceDraft(d, starts[first]!, starts[last]!)
  kept.wordIds = kept.wordIds.map((w) => w - first)
  return kept
}

/**
 * Put one or two sequences into the tokeniser's template. Throws `DomainError` when a special token of the template
 * has no id.
 *
 * @param t The tokeniser.
 * @param a The tokens of the first sequence.
 * @param b The tokens of the second, or null.
 * @param add Whether to add the template's special tokens; without them, or with no post-processor, the sequences are
 *   joined with type ids 0 and 1.
 * @returns The joined draft.
 */
function template(t: Tokeniser, a: Draft, b: Draft | null, add: boolean): Draft {
  const out = emptyDraft()
  if (!t.postProcessor || !add) {
    append(out, a, 0, 0)
    if (b) append(out, b, 1, 1)
    return out
  }
  for (const item of b ? t.postProcessor.pair : t.postProcessor.single) {
    if ('sequence' in item) {
      if (item.sequence === 'A') append(out, a, item.typeId, 0)
      else if (b) append(out, b, item.typeId, 1)
      continue
    }
    const id = tokenId(t.model.vocabulary, item.special)
    if (id < 0) throw new DomainError('encode', `encode: the template's special token '${item.special}' has no id`)
    out.tokens.push(item.special)
    out.ids.push(id)
    out.offsets.push(0, 0)
    out.wordIds.push(-1)
    out.sequenceIds.push(-1)
    out.typeIds.push(item.typeId)
    out.attention.push(1)
    out.special.push(1)
  }
  return out
}

/**
 * Pad a draft to a length with the padding token (attention 0, special 1, word and sequence ids $-1$).
 *
 * @param d The draft; not modified.
 * @param length The length to reach; a draft already as long is returned as it is.
 * @param p The padding settings, for the token, id, type id and side.
 * @returns The padded draft.
 */
function pad(d: Draft, length: number, p: Padding): Draft {
  const n = length - d.ids.length
  if (n <= 0) return d
  const fill = emptyDraft()
  for (let k = 0; k < n; k++) {
    fill.tokens.push(p.token)
    fill.ids.push(p.id)
    fill.offsets.push(0, 0)
    fill.wordIds.push(-1)
    fill.sequenceIds.push(-1)
    fill.typeIds.push(p.typeId)
    fill.attention.push(0)
    fill.special.push(1)
  }
  const out = emptyDraft()
  if (p.direction === 'left') {
    append(out, fill)
    append(out, d)
  } else {
    append(out, d)
    append(out, fill)
  }
  return out
}

/**
 * The length to pad to.
 *
 * @param n The length of the longest encoding (used when the settings give no length).
 * @param p The padding settings.
 * @returns The fixed length, or `n`, rounded up to a multiple of `multipleOf` when it is set.
 */
const padTarget = (n: number, p: Padding) => {
  const base = p.length ?? n
  return p.multipleOf && base % p.multipleOf !== 0 ? base + p.multipleOf - (base % p.multipleOf) : base
}

/**
 * Encode a text or a pair, as {@link encodeText} describes. Throws `DomainError` when one sequence of a pair alone
 * leaves no room for the other under truncation.
 *
 * @param t The tokeniser.
 * @param text The first text.
 * @param pair The second text, or null.
 * @param options Whether to add special tokens, and the model's segment options.
 * @returns The encoding, padded to a fixed length when the tokeniser says so.
 */
function encodeOne(t: Tokeniser, text: string, pair: string | null, options: EncodeOptions): Encoding {
  const add = options.addSpecialTokens ?? true
  const tr = t.truncation
  // Every sequence is cut to its first (or last) maxLength tokens' worth of splits, except the first under onlySecond.
  const limit = (first: boolean) =>
    tr && (tr.strategy !== 'onlySecond' || !first) ? { max: tr.maxLength, direction: tr.direction } : null
  const a = encodeTemplateRows(t, text, options, limit(true))
  const b = pair === null ? null : encodeTemplateRows(t, pair, options, limit(false))
  const sources = pair === null ? [text] : [text, pair]
  let windowsA: [number, number][] = [[0, a.ids.length]]
  let windowsB: [number, number][] = b ? [[0, b.ids.length]] : []
  if (tr) {
    const room = tr.maxLength - (add ? addedTokens(t.postProcessor, b !== null) : 0)
    if (!b) windowsA = truncationWindows(a.ids.length, room, tr.stride, tr.direction)
    else {
      let [na, nb] = [a.ids.length, b.ids.length]
      if (tr.strategy === 'longestFirst') [na, nb] = pairLengths(na, nb, room)
      else if (tr.strategy === 'onlyFirst') na = Math.min(na, room - nb)
      else nb = Math.min(nb, room - na)
      if (na < 0 || nb < 0)
        throw new DomainError('encode', 'encode: the other sequence alone is longer than the maximum length')
      windowsA = truncationWindows(a.ids.length, na, tr.stride, tr.direction)
      windowsB = truncationWindows(b.ids.length, nb, tr.stride, tr.direction)
    }
  }
  const build = (wa: [number, number], wb: [number, number] | null) =>
    template(t, sliceDraft(a, wa[0], wa[1]), b && wb ? sliceDraft(b, wb[0], wb[1]) : null, add)
  const main = build(windowsA[0], b ? windowsB[0] : null)
  const p = t.padding
  const padded = (d: Draft) => (p && p.length !== null ? pad(d, padTarget(d.ids.length, p), p) : d)
  const one = (wa: [number, number], wb: [number, number] | null, nested: Encoding[] = []) =>
    finish(padded(build(wa, wb)), sources, nested)
  // Overflow as Hugging Face nests it: each later window of A with B's first (holding A's window with B's later ones),
  // then with each later window of B; then A's first window with each later window of B (holding A's later windows
  // with it). A single sequence's later windows are simply listed.
  const overflow: Encoding[] = []
  const [a0, ...aRest] = windowsA
  if (!b) for (const w of aRest) overflow.push(one(w, null))
  else {
    const [b0, ...bRest] = windowsB
    for (const wa of aRest) {
      overflow.push(
        one(
          wa,
          b0,
          bRest.map((wb) => one(wa, wb)),
        ),
      )
      for (const wb of bRest) overflow.push(one(wa, wb))
    }
    for (const wb of bRest)
      overflow.push(
        one(
          a0,
          wb,
          aRest.map((wa) => one(wa, wb)),
        ),
      )
  }
  return finish(padded(main), sources, overflow)
}

/**
 * Encode a text, or a pair of texts, with a tokeniser: specials typed in the text, then normalisation,
 * pre-tokenisation and the model per pre-token, then truncation, the template and padding to a fixed length (padding
 * to the longest needs a batch: {@link encodeBatch}). With `stream`, a BPE model with dropout samples its merges.
 * Throws `DomainError` when a token has no id and there is no unknown token, or a template special has no id.
 *
 * @param t The tokeniser.
 * @param text The text, or the first of a pair.
 * @param pair The second text of a pair, or null for one text.
 * @param options `addSpecialTokens` (default true), and `stream`, `dropout` and `upTo` for the model.
 * @returns The encoding, with any overflowing windows.
 *
 * @example Offsets, word ids and sequence ids of a pair
 * const vocabulary = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', 'the', 'cat', 'sat', 'on', 'mat', '##s', '.']
 * const t = tokeniser({
 *   normaliser: lowercaseNormaliser(),
 *   preTokeniser: bertPreTokeniser(),
 *   model: wordPieceStage(vocabulary, { specials: ['[UNK]', '[CLS]', '[SEP]', '[PAD]'] }),
 *   postProcessor: bertProcessor(),
 *   decoder: wordPieceDecoder(),
 * })
 * const e = encodeText(t, 'The cat sat.', 'The mats.')
 * print('tokens =', e.tokens)
 * print('offsets =', e.offsets)
 * print('word ids =', e.wordIds)
 * print('sequence ids =', e.sequenceIds)
 * print('special tokens mask =', e.specialTokensMask)
 *
 * @example Without the template's special tokens
 * const vocabulary = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', 'the', 'cat', 'sat', 'on', 'mat', '##s', '.']
 * const t = tokeniser({
 *   normaliser: lowercaseNormaliser(),
 *   preTokeniser: bertPreTokeniser(),
 *   model: wordPieceStage(vocabulary, { specials: ['[UNK]', '[CLS]', '[SEP]', '[PAD]'] }),
 *   postProcessor: bertProcessor(),
 *   decoder: wordPieceDecoder(),
 * })
 * print(encodeText(t, 'The cat sat.', null, { addSpecialTokens: false }).tokens)
 */
export function encodeText(
  t: Tokeniser,
  text: string,
  pair: string | null = null,
  options: EncodeOptions = {},
): Encoding {
  return encodeOne(t, text, pair, options)
}

/**
 * Encode several texts (or [text, pair] pairs), padded to the longest when the tokeniser pads to the longest.
 *
 * @param t The tokeniser.
 * @param inputs The texts, each a string or a [text, pair] pair.
 * @param options As for {@link encodeText}.
 * @returns One encoding per input; with padding to the longest, every encoding and overflowing window has the length
 *   of the longest encoding, rounded up to `multipleOf`.
 *
 * @example Two sentences padded to the same length
 * const vocabulary = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', 'the', 'cat', 'sat', 'on', 'mat', '##s', '.']
 * const t = tokeniser({
 *   normaliser: lowercaseNormaliser(),
 *   preTokeniser: bertPreTokeniser(),
 *   model: wordPieceStage(vocabulary, { specials: ['[UNK]', '[CLS]', '[SEP]', '[PAD]'] }),
 *   postProcessor: bertProcessor(),
 *   padding: padding('[PAD]', 3),
 * })
 * for (const e of encodeBatch(t, ['The cat sat.', 'Cats!'])) print(e.tokens, e.attentionMask)
 */
export function encodeBatch(
  t: Tokeniser,
  inputs: readonly (string | readonly [string, string])[],
  options: EncodeOptions = {},
): Encoding[] {
  const out = inputs.map((x) =>
    typeof x === 'string' ? encodeOne(t, x, null, options) : encodeOne(t, x[0], x[1], options),
  )
  const p = t.padding
  if (!p || p.length !== null) return out
  const longest = Math.max(0, ...out.map((e) => e.tokens.length))
  const target = padTarget(longest, p)
  return out.map((e) =>
    finish(
      pad(draftOf(e), target, p),
      e.sources,
      e.overflowing.map((o) => finish(pad(draftOf(o), target, p), o.sources)),
    ),
  )
}

/** Options of {@link decodeIds}. */
export interface DecodeOptions {
  /** Leave out special tokens (default true). */
  skipSpecialTokens?: boolean
}

/**
 * Text from ids (an int tensor or a list) through the tokeniser's decoder. Throws `DomainError` for an id outside the
 * vocabulary.
 *
 * @param t The tokeniser.
 * @param ids The token ids, in order.
 * @param options `skipSpecialTokens` (default true) leaves out the tokeniser's and the vocabulary's special tokens.
 * @returns The text.
 *
 * @example With and without the special tokens
 * const vocabulary = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', 'the', 'cat', 'sat', 'on', 'mat', '##s', '.']
 * const t = tokeniser({
 *   normaliser: lowercaseNormaliser(),
 *   preTokeniser: bertPreTokeniser(),
 *   model: wordPieceStage(vocabulary, { specials: ['[UNK]', '[CLS]', '[SEP]', '[PAD]'] }),
 *   postProcessor: bertProcessor(),
 *   decoder: wordPieceDecoder(),
 * })
 * const { ids } = encodeText(t, 'The cat sat.')
 * print(decodeIds(t, ids))
 * print(decodeIds(t, ids, { skipSpecialTokens: false }))
 */
export function decodeIds(t: Tokeniser, ids: Tensor | readonly number[], options: DecodeOptions = {}): string {
  const skip = options.skipSpecialTokens ?? true
  const v = t.model.vocabulary
  const special = new Set([...t.specials, ...v.specials])
  const list = Array.isArray(ids) ? (ids as readonly number[]) : Array.from(toFlat(ids as Tensor))
  const tokens: string[] = []
  for (const id of list) {
    const tok = v.tokens[id]
    if (tok === undefined) throw new DomainError('decode', `decode: id ${id} is out of range`)
    if (skip && special.has(tok)) continue
    tokens.push(tok)
  }
  return applyDecoder(t.decoder, tokens)
}

/**
 * The vocabulary size of a tokeniser (special tokens included).
 *
 * @param t The tokeniser.
 * @returns The number of tokens with ids.
 *
 * @example The byte tokeniser: 3 specials and 256 bytes
 * print(vocabularySize(byteTokeniser()))
 */
export function vocabularySize(t: Tokeniser): number {
  return t.model.vocabulary.tokens.length
}

/**
 * An encoding's first sequence as a `Tokenisation` (its tokens with offsets into the first text), leaving out the
 * template's special tokens and padding, so the views and statistics of tokenisations apply.
 *
 * @param e The encoding.
 * @returns The tokens of sequence 0 with their offsets into `e.sources[0]`.
 *
 * @example The first sentence of a pair, without [CLS] and [SEP]
 * const vocabulary = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', 'the', 'cat', 'sat', 'on', 'mat', '##s', '.']
 * const t = tokeniser({
 *   normaliser: lowercaseNormaliser(),
 *   preTokeniser: bertPreTokeniser(),
 *   model: wordPieceStage(vocabulary, { specials: ['[UNK]', '[CLS]', '[SEP]', '[PAD]'] }),
 *   postProcessor: bertProcessor(),
 *   decoder: wordPieceDecoder(),
 * })
 * print(encodingTokenisation(encodeText(t, 'The cat sat.', 'The mats.')))
 */
export function encodingTokenisation(e: Encoding): Tokenisation {
  const seq = e.sequenceIds.data
  const o = e.offsets.data
  const tokens: string[] = []
  const offsets: number[] = []
  e.tokens.forEach((tok, k) => {
    if (seq[k] !== 0) return
    tokens.push(tok)
    offsets.push(o[2 * k], o[2 * k + 1])
  })
  return { kind: 'tokens', source: e.sources[0], tokens, offsets: int32(offsets, [tokens.length, 2]) }
}

/**
 * The pre-tokens of texts after normalisation and pre-tokenisation, with their counts: the word table a subword
 * trainer starts from, so the model is trained on exactly what it will be asked to segment.
 *
 * @param t The stages to apply: `normaliser` and `preTokeniser`, either left out or null to skip it. A tokeniser, or
 *   its parts before training, will do.
 * @param texts The texts.
 * @returns Each distinct pre-token with its count, in order of first appearance.
 *
 * @example Lower-cased words of two sentences
 * const counts = preTokenCounts({ normaliser: lowercaseNormaliser(), preTokeniser: whitespacePreTokeniser() }, [
 *   'The cat sat.',
 *   'the cat!',
 * ])
 * print([...counts])
 */
export function preTokenCounts(
  t: { readonly normaliser?: Normaliser | null; readonly preTokeniser?: PreTokeniser | null },
  texts: readonly string[],
): Map<string, number> {
  const counts = new Map<string, number>()
  for (const text of texts) {
    let a = aligned(text)
    if (t.normaliser) a = applyNormaliser(t.normaliser, a)
    for (const p of applyPreTokeniser(t.preTokeniser ?? null, [a])) counts.set(p.text, (counts.get(p.text) ?? 0) + 1)
  }
  return counts
}

/**
 * A pure byte tokeniser (ByT5, Xue et al. 2022): no normaliser, no pre-tokeniser, one token per UTF-8 byte with ids
 * 3 to 258 after `<pad>`, `</s>` and `<unk>`, `</s>` appended, and decoding that rebuilds the bytes exactly.
 *
 * @param options `eos`: false leaves out the appended `</s>` (default true).
 * @returns The tokeniser.
 *
 * @example Bytes in, the same text out
 * const t = byteTokeniser()
 * const e = encodeText(t, 'hé!')
 * print('tokens =', e.tokens)
 * print('ids =', e.ids)
 * print('decoded =', decodeIds(t, e.ids))
 */
export function byteTokeniser(options: { eos?: boolean } = {}): Tokeniser {
  return tokeniser({
    model: byteStage(),
    postProcessor: options.eos === false ? null : templateProcessor('$A </s>', '$A </s> $B:1 </s>:1'),
    decoder: decoderSequence(byteFallbackDecoder(), fuseDecoder()),
  })
}
