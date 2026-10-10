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
  /** The token's string. */
  token: string
  /** Its id in the model's vocabulary. */
  id: number
  /** Where its range starts in the pre-token, in UTF-16 code units. */
  start: number
  /** Where its range ends (exclusive). */
  end: number
}

/** The fields every model stage has. */
interface Common {
  /** The tokens and their ids, special tokens included. */
  readonly vocabulary: Vocabulary
  /** The unknown token, or null for none (an uncovered character is then an error). */
  readonly unknown: string | null
}

/** A BPE model stage. */
export interface BpeStage extends Common {
  /** Marks a BPE stage. */
  readonly type: 'bpe'
  /** The merges in rank order. */
  readonly merges: readonly BpeMerge[]
  /** Whether a symbol outside the vocabulary becomes its `<0xNN>` byte tokens. */
  readonly byteFallback: boolean
  /** Whether consecutive unknown tokens are joined into one. */
  readonly fuseUnknown: boolean
  /** A symbol appended to every pre-token before merging (Sennrich's "</w>"; empty for none). */
  readonly endOfWord: string
  /** BPE-dropout probability, used when encoding with a random stream (default 0). */
  readonly dropout: number
}

/** A WordPiece model stage. */
export interface WordPieceStage extends Common {
  /** Marks a WordPiece stage. */
  readonly type: 'wordPiece'
  /** The continuation prefix, such as `##`. */
  readonly prefix: string
  /** Pre-tokens longer than this many characters become the unknown token. */
  readonly maxCharacters: number
}

/** A unigram language-model stage: pieces with log-probabilities (float64 [P]). */
export interface UnigramStage extends Common {
  /** Marks a unigram stage. */
  readonly type: 'unigram'
  /** The pieces of the lattice (no special or byte tokens). */
  readonly pieces: readonly string[]
  /** The log-probability, or score, of each piece. */
  readonly logProbs: Tensor
  /** Whether an uncovered character becomes its `<0xNN>` byte tokens. */
  readonly byteFallback: boolean
  /** Whether consecutive unknown tokens are joined into one. */
  readonly fuseUnknown: boolean
}

/** A word-level stage: each pre-token is one token, or the unknown token. */
export interface WordLevelStage extends Common {
  /** Marks a word-level stage. */
  readonly type: 'wordLevel'
}

/** A character stage: one token per code point, unknown or byte fallback for unseen characters. */
export interface CharacterStage extends Common {
  /** Marks a character stage. */
  readonly type: 'character'
  /** Whether an unseen character becomes its `<0xNN>` byte tokens. */
  readonly byteFallback: boolean
}

/** A byte stage (ByT5): one token per UTF-8 byte, ids after the special tokens. */
export interface ByteStage extends Common {
  /** Marks a byte stage. */
  readonly type: 'byte'
}

/** A model stage. */
export type TokeniserModel = BpeStage | WordPieceStage | UnigramStage | WordLevelStage | CharacterStage | ByteStage

// ── Vocabularies and byte tokens ─────────────────────────────────────────────────────────────────────────────────────

/**
 * The byte token of a byte value, SentencePiece's `<0xNN>` (upper-case hexadecimal).
 *
 * @param b The byte value, 0 to 255.
 * @returns The token.
 *
 * @example Three bytes
 * print(byteToken(0), byteToken(10), byteToken(255))
 */
export const byteToken = (b: number): string => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`

/** The 256 byte tokens `<0x00>` … `<0xFF>`. */
export const BYTE_TOKENS: readonly string[] = Array.from({ length: 256 }, (_, b) => byteToken(b))

const BYTE_RE = /^<0x([0-9A-F]{2})>$/u

/**
 * The byte value of a byte token, or $-1$.
 *
 * @param t A token string.
 * @returns The byte value when `t` is exactly `<0xNN>` with upper-case hexadecimal digits, otherwise $-1$.
 *
 * @example Only upper-case byte tokens are recognised
 * print(byteOfToken('<0x0A>'), byteOfToken('<0x0a>'), byteOfToken('a'))
 * print(byteOfToken(byteToken(200)))
 */
export const byteOfToken = (t: string): number => {
  const m = BYTE_RE.exec(t)
  return m ? parseInt(m[1], 16) : -1
}

/**
 * A vocabulary whose ids are given: a token list (the id is the position) or a record from token to id (ids must be
 * $0, \dots, V - 1$, each once). `specials` marks special tokens; `unknown` names the unknown token (or null). Throws
 * `DomainError` when the ids are not that range or a token repeats.
 *
 * @param tokens The tokens in id order, or a record from token to id (as the `vocab` of a Hugging Face
 *   `tokenizer.json`).
 * @param options `specials`, the tokens to mark as special (default none), and `unknown`, the unknown token (default
 *   null, none; a token not in the list also gives none).
 * @returns The vocabulary, with zero counts.
 *
 * @example From a record of ids
 * const v = vocabularyWithIds({ '[UNK]': 0, hello: 2, world: 1 }, { specials: ['[UNK]'], unknown: '[UNK]' })
 * print('tokens =', v.tokens, 'unknown id =', v.unknown)
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

/**
 * The distinct strings of a list, in order of first appearance.
 *
 * @param xs The strings.
 * @returns Them without repeats.
 */
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
 * A BPE stage from merges and a vocabulary (a token list, or a record from token to id as in a Hugging Face
 * `tokenizer.json`), or from a training state (vocabulary: specials, then the 256 byte tokens with byte fallback,
 * then the alphabet and one symbol per merge).
 *
 * @param source A state of `bpeSteps`, or `merges` (as pairs, or merges with counts) in rank order with a
 *   `vocabulary` that must hold every merged symbol a word can reach.
 * @param options `specials`, `unknown` (default null: a symbol outside the vocabulary is then an error),
 *   `byteFallback` and `fuseUnknown` (default false), `endOfWord` (default none) and `dropout` (default 0).
 * @returns The stage.
 *
 * @example Two merges, and an unknown character
 * const model = bpeStage(
 *   { merges: [['l', 'o'], ['lo', 'w']], vocabulary: ['<unk>', 'l', 'o', 'w', 'e', 'r', 'lo', 'low'] },
 *   { unknown: '<unk>' },
 * )
 * print('lower =', modelSegment(model, 'lower').map((t) => [t.token, t.id]))
 * print('lowx =', modelSegment(model, 'lowx').map((t) => [t.token, t.id]))
 *
 * @example Byte fallback instead of an unknown token
 * const vocabulary = ['h', 'i', 'hi', ...BYTE_TOKENS]
 * const model = bpeStage({ merges: [['h', 'i']], vocabulary }, { byteFallback: true })
 * print(modelSegment(model, 'hié').map((t) => t.token))
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

/**
 * A WordPiece stage from a vocabulary or a training state (whose vocabulary already lists its specials).
 *
 * @param source A state of `wordPieceSteps`, the tokens in id order, or a record from token to id.
 * @param options `unknown` (default `[UNK]`), `specials` (default the unknown token alone), `prefix` (default `##`)
 *   and `maxCharacters` (default 100); `byteFallback` and `fuseUnknown` are ignored.
 * @returns The stage.
 *
 * @example A word-initial piece and continuations, and a word with no match for its end
 * const model = wordPieceStage(['[UNK]', 'un', 'want', '##want', '##ed'])
 * print('unwanted =', modelSegment(model, 'unwanted').map((t) => [t.token, t.id]))
 * print('wanted =', modelSegment(model, 'wanted').map((t) => [t.token, t.id]))
 * print('wants =', modelSegment(model, 'wants').map((t) => [t.token, t.id]))
 */
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
 *
 * @param source A state of `unigramLmSteps`, or `[piece, log-probability]` pairs in id order.
 * @param options `unknown` (default `<unk>`), `specials` (default the unknown token alone), `byteFallback` (default
 *   false; from a list, its byte tokens must be in it) and `fuseUnknown` (default true).
 * @returns The stage.
 *
 * @example An unknown character keeps its own text, with the unknown id
 * const model = unigramStage([
 *   ['<unk>', 0], ['▁', -2], ['h', -3], ['u', -3], ['g', -3], ['hug', -2], ['s', -3], ['▁hug', -1.5],
 * ])
 * print('▁hugs =', modelSegment(model, '▁hugs').map((t) => [t.token, t.id]))
 * print('▁hux =', modelSegment(model, '▁hux').map((t) => [t.token, t.id]))
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

/**
 * A word-level stage from a vocabulary (list or record).
 *
 * @param vocabulary The words in id order, or a record from word to id.
 * @param options `unknown` (default `[UNK]`) and `specials` (default the unknown token alone); the others are ignored.
 * @returns The stage.
 *
 * @example A known word, an unknown one
 * const model = wordLevelStage(['[UNK]', 'the', 'cat', 'sat'])
 * print(['the', 'dog', 'sat'].map((w) => modelSegment(model, w)[0].token))
 */
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

/**
 * A character stage over a list of characters (specials and, with byte fallback, byte tokens first).
 *
 * @param characters The characters of the vocabulary; repeats are dropped.
 * @param options `unknown` (default `<unk>`), `specials` (default the unknown token alone) and `byteFallback`
 *   (default false); `fuseUnknown` is ignored.
 * @returns The stage.
 *
 * @example Byte fallback for an unseen character
 * const model = characterStage(['a', 'b', 'c'], { byteFallback: true })
 * print('cab =', modelSegment(model, 'cab').map((t) => [t.token, t.id]))
 * print('aé =', modelSegment(model, 'aé').map((t) => [t.token, t.id]))
 */
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
 *
 * @param options `specials`, the special tokens before the bytes; byte $b$ has id $b$ plus their number.
 * @returns The stage.
 *
 * @example One token per byte, ids offset by 3
 * print(modelSegment(byteStage(), 'hé').map((t) => [t.token, t.id]))
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
 * unknown id, as Hugging Face's unigram does); consecutive unknowns fuse when asked. Throws `DomainError` when a
 * piece must be unknown and the vocabulary has no unknown token.
 *
 * @param model The stage, whose vocabulary gives the ids.
 * @param word The pre-token the pieces were cut from, for the text of a piece that falls back.
 * @param pieces The pieces, with ranges in `word`.
 * @param o `byteFallback`, `fuse` (join adjacent unknowns), `unknownText` (keep the piece's text as the unknown
 *   token's string) and `isUnknown`, which marks a piece as unknown whatever its string.
 * @returns The tokens with ids and ranges; byte tokens of one piece share its range.
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

/**
 * The code points of a word as pieces with their ranges.
 *
 * @param word The word.
 * @returns One piece per code point.
 */
const codePoints = (word: string): Piece[] => {
  const out: Piece[] = []
  let at = 0
  for (const c of word) {
    out.push({ token: c, start: at, end: at + c.length })
    at += c.length
  }
  return out
}

/**
 * Segment one pre-token with a model stage: tokens with ids and ranges in the pre-token. Throws `DomainError` when a
 * piece is outside the vocabulary and there is no unknown token or byte fallback for it.
 *
 * @param model The stage.
 * @param word One pre-token.
 * @param options For BPE: `upTo`, and `stream` with an optional `dropout` (the stage's own when left out). Dropout
 *   applies only when `stream` is given. Other stages ignore them.
 * @returns The tokens, in order.
 *
 * @example BPE, deterministic and with dropout
 * const model = bpeStage(
 *   { merges: [['l', 'o'], ['lo', 'w'], ['e', 'r']], vocabulary: ['l', 'o', 'w', 'e', 'r', 'lo', 'low', 'er'] },
 *   { dropout: 0.5 },
 * )
 * print('lower =', modelSegment(model, 'lower'))
 * print('first merge only:', modelSegment(model, 'lower', { upTo: 1 }).map((t) => t.token))
 * const s = stream(3)
 * print('with a stream:', modelSegment(model, 'lower', { stream: s }).map((t) => t.token))
 * print('with a stream:', modelSegment(model, 'lower', { stream: s }).map((t) => t.token))
 */
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
