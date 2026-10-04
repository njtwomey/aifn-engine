/**
 * Models, the third stage of a tokeniser pipeline: each segments one pre-token into vocabulary tokens with ids and
 * ranges. BPE (with byte fallback, BPE-dropout and an optional end-of-word symbol), WordPiece, the unigram language
 * model (with byte fallback), a word-level vocabulary, characters, and raw UTF-8 bytes (ByT5). A token outside the
 * vocabulary becomes the unknown token, or with byte fallback its UTF-8 bytes as `<0xNN>` tokens (SentencePiece,
 * LLaMA), so no text is lost.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import type { Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  bpeSegment,
  unigramLmModel,
  unigramLmViterbi,
  wordPieceModel,
  wordPieceSegment,
  type BpeMerge,
  type BpeModel,
  type BpeState,
  type Piece,
  type UnigramLmModel,
  type UnigramLmState,
  type WordPieceModel,
  type WordPieceState,
} from 'aifn-compute/text/subword'
import { tokenId, type Vocabulary } from 'aifn-compute/text/vocabulary'

/** A token of one pre-token: its string, id and [start, end) range in the pre-token's text. */
export interface ModelToken {
  token: string
  id: number
  start: number
  end: number
}

interface Common {
  /** Token ↔ id, special tokens included. */
  readonly vocabulary: Vocabulary
  /** The unknown token, or null for none (an uncovered character is then an error). */
  readonly unknown: string | null
}

/** A BPE model stage. */
export interface BpeStage extends Common {
  readonly type: 'bpe'
  readonly merges: readonly BpeMerge[]
  readonly byteFallback: boolean
  readonly fuseUnknown: boolean
  /** A symbol appended to every pre-token before merging (Sennrich's "</w>"; empty for none). */
  readonly endOfWord: string
  /** BPE-dropout probability, used when encoding with a random stream (default 0). */
  readonly dropout: number
}

/** A WordPiece model stage. */
export interface WordPieceStage extends Common {
  readonly type: 'wordPiece'
  readonly prefix: string
  readonly maxCharacters: number
}

/** A unigram language-model stage: pieces with log-probabilities (float64 [P]). */
export interface UnigramStage extends Common {
  readonly type: 'unigram'
  readonly pieces: readonly string[]
  readonly logProbs: Tensor
  readonly byteFallback: boolean
  readonly fuseUnknown: boolean
}

/** A word-level stage: each pre-token is one token, or the unknown token. */
export interface WordLevelStage extends Common {
  readonly type: 'wordLevel'
}

/** A character stage: one token per code point, unknown or byte fallback for unseen characters. */
export interface CharacterStage extends Common {
  readonly type: 'character'
  readonly byteFallback: boolean
}

/** A byte stage (ByT5): one token per UTF-8 byte, ids after the special tokens. */
export interface ByteStage extends Common {
  readonly type: 'byte'
}

/** A model stage. */
export type TokeniserModel = BpeStage | WordPieceStage | UnigramStage | WordLevelStage | CharacterStage | ByteStage

// ── Vocabularies and byte tokens ─────────────────────────────────────────────────────────────────────────────────────

/** The byte token of a byte value, SentencePiece's `<0xNN>` (upper-case hexadecimal). */
export const byteToken = (b: number): string => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`

/** The 256 byte tokens `<0x00>` … `<0xFF>`. */
export const BYTE_TOKENS: readonly string[] = Array.from({ length: 256 }, (_, b) => byteToken(b))

const BYTE_RE = /^<0x([0-9A-F]{2})>$/u

/** The byte value of a byte token, or −1. */
export const byteOfToken = (t: string): number => {
  const m = BYTE_RE.exec(t)
  return m ? parseInt(m[1], 16) : -1
}

/**
 * A vocabulary whose ids are given: a token list (id = position) or a token → id record (ids must be 0 … V − 1).
 * `specials` marks special tokens; `unknown` names the unknown token (or null).
 */
export function vocabularyWithIds(
  tokens: readonly string[] | Readonly<Record<string, number>>,
  options: { specials?: readonly string[]; unknown?: string | null } = {},
): Vocabulary {
  let list: string[]
  if (Array.isArray(tokens)) list = [...(tokens as readonly string[])]
  else {
    const entries = Object.entries(tokens as Record<string, number>)
    list = new Array<string>(entries.length)
    for (const [t, id] of entries) {
      if (!(id >= 0 && id < entries.length) || list[id] !== undefined)
        throw new DomainError(
          'vocabularyWithIds',
          `vocabularyWithIds: ids must be 0 … ${entries.length - 1}, once each`,
        )
      list[id] = t
    }
  }
  if (new Set(list).size !== list.length)
    throw new DomainError('vocabularyWithIds', 'vocabularyWithIds: tokens must be distinct')
  const specials = options.specials ?? []
  const unknown = options.unknown ?? null
  return {
    kind: 'vocabulary',
    tokens: list,
    counts: fromData(new Float64Array(list.length)),
    specials: [...specials],
    unknown: unknown === null ? -1 : list.indexOf(unknown),
  }
}

const dedupe = (xs: readonly string[]) => [...new Set(xs)]

// ── Constructors ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options shared by the model constructors. */
export interface ModelOptions {
  /** Special tokens, given the first ids (when the vocabulary is built here). */
  specials?: readonly string[]
  /** The unknown token (default the model's usual one, or null). */
  unknown?: string | null
  /** Encode characters outside the vocabulary as `<0xNN>` byte tokens (BPE, unigram, character). */
  byteFallback?: boolean
  /** Join consecutive unknown tokens into one. */
  fuseUnknown?: boolean
}

/**
 * A BPE stage from merges and a vocabulary (a token list, or a token → id record as in a Hugging Face
 * `tokenizer.json`), or from a training state (vocabulary: specials, then the 256 byte tokens with byte fallback,
 * then the alphabet and one symbol per merge).
 */
export function bpeStage(
  source:
    | BpeState
    | {
        merges: readonly (readonly [string, string] | BpeMerge)[]
        vocabulary: readonly string[] | Record<string, number>
      },
  options: ModelOptions & { endOfWord?: string; dropout?: number } = {},
): BpeStage {
  const specials = options.specials ?? []
  const byteFallback = options.byteFallback ?? false
  const unknown = options.unknown === undefined ? null : options.unknown
  const merges: BpeMerge[] = source.merges.map((m) =>
    Array.isArray(m) ? { left: m[0], right: m[1], merged: m[0] + m[1], count: 0 } : (m as BpeMerge),
  )
  const isState = 'segmentations' in source
  const vocabulary = isState
    ? vocabularyWithIds(
        dedupe([...specials, ...(byteFallback ? BYTE_TOKENS : []), ...(source as BpeState).vocabulary]),
        {
          specials,
          unknown,
        },
      )
    : vocabularyWithIds(source.vocabulary as readonly string[] | Record<string, number>, { specials, unknown })
  return {
    type: 'bpe',
    vocabulary,
    unknown,
    merges,
    byteFallback,
    fuseUnknown: options.fuseUnknown ?? false,
    endOfWord: options.endOfWord ?? '',
    dropout: options.dropout ?? 0,
  }
}

/** A WordPiece stage from a vocabulary or a training state (whose vocabulary already lists its specials). */
export function wordPieceStage(
  source: WordPieceState | readonly string[] | Record<string, number>,
  options: ModelOptions & { prefix?: string; maxCharacters?: number } = {},
): WordPieceStage {
  const unknown = options.unknown === undefined ? '[UNK]' : options.unknown
  const tokens = Array.isArray(source)
    ? (source as readonly string[])
    : 'segmentations' in (source as object)
      ? (source as WordPieceState).vocabulary
      : (source as Record<string, number>)
  return {
    type: 'wordPiece',
    vocabulary: vocabularyWithIds(tokens, { specials: options.specials ?? (unknown ? [unknown] : []), unknown }),
    unknown,
    prefix: options.prefix ?? '##',
    maxCharacters: options.maxCharacters ?? 100,
  }
}

/**
 * A unigram stage from pieces with log-probabilities (Hugging Face's `[piece, score]` list, in id order; specials and
 * byte tokens in it are kept out of the lattice) or from a training state (vocabulary: specials, byte tokens with
 * byte fallback, then the pieces).
 */
export function unigramStage(
  source: UnigramLmState | readonly (readonly [string, number])[],
  options: ModelOptions = {},
): UnigramStage {
  const unknown = options.unknown === undefined ? '<unk>' : options.unknown
  const specials = options.specials ?? (unknown ? [unknown] : [])
  const byteFallback = options.byteFallback ?? false
  let tokens: string[]
  let pieces: string[]
  let lp: number[]
  if (Array.isArray(source)) {
    const list = source as readonly (readonly [string, number])[]
    tokens = list.map(([p]) => p)
    const kept = list.filter(([p]) => !specials.includes(p) && byteOfToken(p) < 0)
    pieces = kept.map(([p]) => p)
    lp = kept.map(([, s]) => s)
  } else {
    const s = source as UnigramLmState
    pieces = [...s.pieces]
    lp = Array.from(toFlat(s.logProbs))
    tokens = dedupe([...specials, ...(byteFallback ? BYTE_TOKENS : []), ...pieces])
  }
  return {
    type: 'unigram',
    vocabulary: vocabularyWithIds(tokens, { specials, unknown }),
    unknown,
    pieces,
    logProbs: fromData(Float64Array.from(lp)),
    byteFallback,
    fuseUnknown: options.fuseUnknown ?? true,
  }
}

/** A word-level stage from a vocabulary (list or record). */
export function wordLevelStage(
  vocabulary: readonly string[] | Record<string, number>,
  options: ModelOptions = {},
): WordLevelStage {
  const unknown = options.unknown === undefined ? '[UNK]' : options.unknown
  return {
    type: 'wordLevel',
    vocabulary: vocabularyWithIds(vocabulary, { specials: options.specials ?? (unknown ? [unknown] : []), unknown }),
    unknown,
  }
}

/** A character stage over a list of characters (specials and, with byte fallback, byte tokens first). */
export function characterStage(characters: readonly string[], options: ModelOptions = {}): CharacterStage {
  const unknown = options.unknown === undefined ? '<unk>' : options.unknown
  const specials = options.specials ?? (unknown ? [unknown] : [])
  const byteFallback = options.byteFallback ?? false
  return {
    type: 'character',
    vocabulary: vocabularyWithIds(dedupe([...specials, ...(byteFallback ? BYTE_TOKENS : []), ...characters]), {
      specials,
      unknown,
    }),
    unknown,
    byteFallback,
  }
}

/**
 * The byte stage of ByT5 (Xue et al. 2022): the specials (default `<pad>`, `</s>`, `<unk>`, ids 0–2), then the 256
 * bytes. Byte tokens print as the character for printable ASCII (0x20–0x7E) and as `<0xNN>` otherwise.
 */
export function byteStage(options: { specials?: readonly string[] } = {}): ByteStage {
  const specials = options.specials ?? ['<pad>', '</s>', '<unk>']
  return {
    type: 'byte',
    vocabulary: vocabularyWithIds([...specials, ...BYTE_SYMBOLS], { specials, unknown: null }),
    unknown: null,
  }
}

/** The byte stage's 256 token strings. */
const BYTE_SYMBOLS: readonly string[] = Array.from({ length: 256 }, (_, b) =>
  b >= 0x20 && b <= 0x7e ? String.fromCharCode(b) : byteToken(b),
)

// ── Segmenting ───────────────────────────────────────────────────────────────────────────────────────────────────────

const encoder = new TextEncoder()

/** Options of {@link modelSegment}. */
export interface SegmentOptions {
  /** The random stream of BPE-dropout (dropout applies only when given). */
  stream?: Stream
  /** Override the stage's dropout probability. */
  dropout?: number
  /** BPE: apply only the first `upTo` merges. */
  upTo?: number
}

const bpeCache = new WeakMap<BpeStage, BpeModel>()
const unigramCache = new WeakMap<UnigramStage, UnigramLmModel>()
const wordPieceCache = new WeakMap<WordPieceStage, WordPieceModel>()

/**
 * Pieces to tokens with ids: a piece outside the vocabulary becomes its byte tokens (byte fallback, when they are all
 * in the vocabulary) or the unknown token (`text: true` keeps the piece's own text as the token string, with the
 * unknown id, as Hugging Face's unigram does); consecutive unknowns fuse when asked.
 */
function withIds(
  model: TokeniserModel,
  word: string,
  pieces: readonly Piece[],
  o: { byteFallback: boolean; fuse: boolean; unknownText: boolean; isUnknown?: (p: Piece) => boolean },
): ModelToken[] {
  const v = model.vocabulary
  const out: ModelToken[] = []
  let lastUnknown = false
  for (const p of pieces) {
    const id = o.isUnknown?.(p) ? -1 : tokenId(v, p.token)
    if (id >= 0) {
      out.push({ token: p.token, id, start: p.start, end: p.end })
      lastUnknown = false
      continue
    }
    const text = word.slice(p.start, p.end)
    if (o.byteFallback) {
      const ids = Array.from(encoder.encode(text), (b) => tokenId(v, byteToken(b)))
      if (ids.every((x) => x >= 0)) {
        ids.forEach((x) => out.push({ token: v.tokens[x], id: x, start: p.start, end: p.end }))
        lastUnknown = false
        continue
      }
    }
    if (v.unknown < 0)
      throw new DomainError(
        'encode',
        `encode: ${JSON.stringify(text)} is not in the vocabulary and there is no unknown token`,
      )
    const last = out[out.length - 1]
    if (o.fuse && lastUnknown && last.end === p.start) {
      last.end = p.end
      if (o.unknownText) last.token = word.slice(last.start, last.end)
    } else out.push({ token: o.unknownText ? text : v.tokens[v.unknown], id: v.unknown, start: p.start, end: p.end })
    lastUnknown = true
  }
  return out
}

const codePoints = (word: string): Piece[] => {
  const out: Piece[] = []
  let at = 0
  for (const c of word) {
    out.push({ token: c, start: at, end: at + c.length })
    at += c.length
  }
  return out
}

/** Segment one pre-token with a model stage: tokens with ids and ranges in the pre-token. */
export function modelSegment(model: TokeniserModel, word: string, options: SegmentOptions = {}): ModelToken[] {
  switch (model.type) {
    case 'bpe': {
      let m = bpeCache.get(model)
      if (!m) {
        m = {
          kind: 'bpe',
          merges: model.merges,
          vocabulary: model.vocabulary.tokens,
          unit: 'character',
          endOfWord: model.endOfWord,
          pattern: 'whitespace',
        }
        bpeCache.set(model, m)
      }
      const dropout = options.stream ? (options.dropout ?? model.dropout) : 0
      const pieces = bpeSegment(m, word, { upTo: options.upTo, dropout, stream: options.stream })
      return withIds(model, word, pieces, {
        byteFallback: model.byteFallback,
        fuse: model.fuseUnknown,
        unknownText: false,
      })
    }
    case 'wordPiece': {
      let m = wordPieceCache.get(model)
      if (!m) {
        m = wordPieceModel(model.vocabulary.tokens, {
          prefix: model.prefix,
          specials: model.unknown ? [model.unknown] : [],
          maxCharacters: model.maxCharacters,
        })
        wordPieceCache.set(model, m)
      }
      return withIds(model, word, wordPieceSegment(m, word), { byteFallback: false, fuse: false, unknownText: false })
    }
    case 'unigram': {
      let m = unigramCache.get(model)
      if (!m) {
        m = unigramLmModel(
          { pieces: model.pieces, probabilities: Array.from(toFlat(model.logProbs), Math.exp) },
          { boundary: '', unknown: '\u0000unk', normalise: false },
        )
        unigramCache.set(model, m)
      }
      const pieces = unigramLmViterbi(m, word).pieces
      return withIds(model, word, pieces, {
        byteFallback: model.byteFallback,
        fuse: model.fuseUnknown,
        unknownText: true,
        isUnknown: (p) => p.token === '\u0000unk',
      })
    }
    case 'wordLevel':
      return withIds(model, word, [{ token: word, start: 0, end: word.length }], {
        byteFallback: false,
        fuse: false,
        unknownText: false,
      })
    case 'character':
      return withIds(model, word, codePoints(word), {
        byteFallback: model.byteFallback,
        fuse: false,
        unknownText: false,
      })
    case 'byte': {
      const out: ModelToken[] = []
      for (const p of codePoints(word))
        for (const b of encoder.encode(p.token)) {
          const id = tokenId(model.vocabulary, BYTE_SYMBOLS[b])
          out.push({ token: BYTE_SYMBOLS[b], id, start: p.start, end: p.end })
        }
      return out
    }
  }
}
