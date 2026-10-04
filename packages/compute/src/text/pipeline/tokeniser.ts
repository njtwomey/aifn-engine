/**
 * The tokeniser pipeline (as Hugging Face `tokenizers`): normaliser → pre-tokeniser → model → post-processor, and a
 * decoder back. A `Tokeniser` is plain data, its stages tagged objects, so it can be stored, sent to a worker and
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
  readonly kind: 'tokeniser'
  readonly normaliser: Normaliser | null
  readonly preTokeniser: PreTokeniser | null
  readonly model: TokeniserModel
  readonly postProcessor: PostProcessor | null
  readonly decoder: Decoder | null
  /** Special tokens recognised verbatim in the input before normalisation (added tokens); they must have ids. */
  readonly specials: readonly string[]
  readonly truncation: Truncation | null
  readonly padding: Padding | null
}

/** The stages of a tokeniser; only the model is required. */
export interface TokeniserParts {
  normaliser?: Normaliser | null
  preTokeniser?: PreTokeniser | null
  model: TokeniserModel
  postProcessor?: PostProcessor | null
  decoder?: Decoder | null
  specials?: readonly string[]
  truncation?: Truncation | null
  padding?: Padding | null
}

/** Assemble a tokeniser from its stages (missing stages are null; specials default to the model's). */
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

/** The tokeniser with some stages replaced. */
export function withStages(t: Tokeniser, parts: Partial<TokeniserParts>): Tokeniser {
  return tokeniser({ ...t, ...parts })
}

/**
 * An encoding: per token, its string, id (int32 [n]), [start, end) offsets in UTF-16 code units of the text of its
 * sequence (int32 [n, 2]; special tokens [0, 0)), the index of the pre-token it came from (`wordIds`, −1 for special
 * tokens and padding), the sequence it belongs to (`sequenceIds`: 0, 1 or −1), the token-type id, and the attention
 * mask (1 for real tokens, 0 for padding) and special-token mask (1 for the template's specials and padding; 0 for specials typed in the text). `overflowing`
 * holds the windows truncation cut off, each a full encoding.
 */
export interface Encoding {
  readonly kind: 'encoding'
  /** The input texts: one, or two for a pair. */
  readonly sources: readonly string[]
  readonly tokens: readonly string[]
  readonly ids: Tensor
  readonly offsets: Tensor
  readonly wordIds: Tensor
  readonly sequenceIds: Tensor
  readonly typeIds: Tensor
  readonly attentionMask: Tensor
  readonly specialTokensMask: Tensor
  readonly overflowing: readonly Encoding[]
}

/** An encoding under construction: plain arrays. */
interface Draft {
  tokens: string[]
  ids: number[]
  offsets: number[]
  wordIds: number[]
  sequenceIds: number[]
  typeIds: number[]
  attention: number[]
  special: number[]
}

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

const int32 = (x: readonly number[], shape?: number[]) => fromData(Int32Array.from(x), shape)

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

const padTarget = (n: number, p: Padding) => {
  const base = p.length ?? n
  return p.multipleOf && base % p.multipleOf !== 0 ? base + p.multipleOf - (base % p.multipleOf) : base
}

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
 */
export function encodeText(
  t: Tokeniser,
  text: string,
  pair: string | null = null,
  options: EncodeOptions = {},
): Encoding {
  return encodeOne(t, text, pair, options)
}

/** Encode several texts (or [text, pair] pairs), padded to the longest when the tokeniser pads to the longest. */
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

/** Text from ids (an int tensor or a list) through the tokeniser's decoder. */
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

/** The vocabulary size of a tokeniser (special tokens included). */
export function vocabularySize(t: Tokeniser): number {
  return t.model.vocabulary.tokens.length
}

/**
 * An encoding's first sequence as a `Tokenisation` (its tokens with offsets into the first text), leaving out the
 * template's special tokens and padding, so the views and statistics of tokenisations apply.
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
 * 3 … 258 after `<pad>`, `</s>` and `<unk>`, `</s>` appended, and decoding that rebuilds the bytes exactly.
 */
export function byteTokeniser(options: { eos?: boolean } = {}): Tokeniser {
  return tokeniser({
    model: byteStage(),
    postProcessor: options.eos === false ? null : templateProcessor('$A </s>', '$A </s> $B:1 </s>:1'),
    decoder: decoderSequence(byteFallbackDecoder(), fuseDecoder()),
  })
}
