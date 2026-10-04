/**
 * Tokeniser pipelines against Hugging Face `tokenizers` 0.22+ (fixture `text/pipeline`): the pre-tokenisers on
 * crafted text, and six pipelines trained there, rebuilt here from their vocabularies and merges, compared token by
 * token: ids, tokens, offsets (Hugging Face counts code points, converted to UTF-16 here), word ids, type ids, masks,
 * decoded text, truncation with stride and overflow, and padding.
 */
import { describe, expect, it } from 'vitest'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { aligned, originalSpan } from 'aifn-compute/text'
import { byteAlphabet } from 'aifn-compute/text/subword'
import {
  applyPreTokeniser,
  bertPreTokeniser,
  bpeStage,
  byteFallbackDecoder,
  byteLevelDecoder,
  byteLevelPreTokeniser,
  decodeIds,
  decoderSequence,
  digitsPreTokeniser,
  encodeText,
  encodeBatch,
  fuseDecoder,
  lowercaseNormaliser,
  metaspaceDecoder,
  metaspacePreTokeniser,
  normaliserSequence,
  padding,
  punctuationPreTokeniser,
  replaceDecoder,
  stripAccentsNormaliser,
  stripDecoder,
  templateProcessor,
  tokeniser,
  truncation,
  unicodeNormaliser,
  unigramStage,
  whitespacePreTokeniser,
  whitespaceSplitPreTokeniser,
  withStages,
  wordLevelStage,
  wordPieceDecoder,
  wordPieceStage,
  type Encoding,
  type PreTokeniser,
  type Tokeniser,
} from 'aifn-compute/text/pipeline'
import { fixture } from '../../fixtures'

type HfEncoding = {
  ids: number[]
  tokens: string[]
  offsets: [number, number][]
  wordIds: number[]
  typeIds: number[]
  specialTokensMask: number[]
  attentionMask: number[]
  overflowing: HfEncoding[]
}
type Model = {
  vocab: Record<string, number> | [string, number][]
  merges?: [string, string][]
  unk_token?: string
}
type Pipeline = {
  model: Model
  encode: (HfEncoding & { text: string; decoded: string })[]
  pairs?: (HfEncoding & { a: string; b: string })[]
}
type Fx = {
  preTokenisers: ({ text: string } & Record<string, [string, [number, number]][]>)[]
  byteAlphabet: string[]
  pipelines: Record<'bpe' | 'byteLevelBpe' | 'wordPiece' | 'unigram' | 'sentencePieceBpe' | 'wordLevel', Pipeline> & {
    truncation: {
      single: (HfEncoding & { text: string })[]
      pairs: (HfEncoding & { a: string; b: string })[]
      batch: HfEncoding[]
      left: HfEncoding[]
      variants: (HfEncoding & {
        maxLength: number
        stride: number
        strategy: 'longest_first' | 'only_first' | 'only_second'
        direction: 'right' | 'left'
        a: string
        b: string | null
      })[]
    }
  }
}
const fx = fixture<Fx>('text/pipeline')
const P = fx.pipelines

/** Code-point offset → UTF-16 offset in `text`. */
const utf16 = (text: string, cp: number) => [...text].slice(0, cp).join('').length

const PRE: Record<string, PreTokeniser> = {
  whitespace: whitespacePreTokeniser(),
  whitespaceSplit: whitespaceSplitPreTokeniser(),
  bert: bertPreTokeniser(),
  punctuation: punctuationPreTokeniser(),
  digits: digitsPreTokeniser(true),
  digitRuns: digitsPreTokeniser(false),
  metaspace: metaspacePreTokeniser({ prependScheme: 'always' }),
  metaspaceFirst: metaspacePreTokeniser({ prependScheme: 'first' }),
  metaspaceNever: metaspacePreTokeniser({ prependScheme: 'never' }),
  byteLevel: byteLevelPreTokeniser(),
  byteLevelPrefix: byteLevelPreTokeniser({ addPrefixSpace: true }),
}

describe('pre-tokenisers', () => {
  it('byte alphabet is the GPT-2 byte-to-unicode table', () => {
    expect([...byteAlphabet()].sort()).toEqual([...fx.byteAlphabet].sort())
  })
  for (const [name, p] of Object.entries(PRE))
    it.each(fx.preTokenisers.map((c) => [c.text, c] as const))(`${name} matches tokenizers on %j`, (_, c) => {
      const parts = applyPreTokeniser(p, [aligned(c.text)])
      const ours = parts.map((x) => [x.text, originalSpan(x, 0, x.text.length)])
      expect(ours).toEqual(c[name].map(([s, [a, b]]) => [s, [utf16(c.text, a), utf16(c.text, b)]]))
    })
})

// ── The pipelines ────────────────────────────────────────────────────────────────────────────────────────────────────

type Name = 'bpe' | 'byteLevelBpe' | 'wordPiece' | 'unigram' | 'sentencePieceBpe' | 'wordLevel'
const PIPELINES: Record<Name, () => Tokeniser> = {
  bpe: () =>
    tokeniser({
      preTokeniser: whitespacePreTokeniser(),
      model: bpeStage(
        { merges: P.bpe.model.merges!, vocabulary: P.bpe.model.vocab as Record<string, number> },
        { specials: ['[UNK]'], unknown: '[UNK]' },
      ),
    }),
  byteLevelBpe: () =>
    tokeniser({
      preTokeniser: byteLevelPreTokeniser(),
      model: bpeStage(
        { merges: P.byteLevelBpe.model.merges!, vocabulary: P.byteLevelBpe.model.vocab as Record<string, number> },
        { specials: ['<|endoftext|>'] },
      ),
      decoder: byteLevelDecoder(),
    }),
  wordPiece: () =>
    tokeniser({
      normaliser: normaliserSequence(unicodeNormaliser('NFD'), lowercaseNormaliser(), stripAccentsNormaliser()),
      preTokeniser: bertPreTokeniser(),
      model: wordPieceStage(P.wordPiece.model.vocab as Record<string, number>, {
        specials: ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
        unknown: '[UNK]',
      }),
      postProcessor: templateProcessor('[CLS] $A [SEP]', '[CLS] $A [SEP] $B:1 [SEP]:1'),
      decoder: wordPieceDecoder(),
    }),
  unigram: () =>
    tokeniser({
      normaliser: unicodeNormaliser('NFKC'),
      preTokeniser: metaspacePreTokeniser(),
      model: unigramStage(P.unigram.model.vocab as [string, number][], { unknown: '<unk>' }),
      decoder: metaspaceDecoder(),
    }),
  sentencePieceBpe: () =>
    tokeniser({
      preTokeniser: metaspacePreTokeniser({ prependScheme: 'first' }),
      model: bpeStage(
        {
          merges: P.sentencePieceBpe.model.merges!,
          vocabulary: P.sentencePieceBpe.model.vocab as Record<string, number>,
        },
        { specials: ['<unk>', '<s>', '</s>'], unknown: '<unk>', byteFallback: true, fuseUnknown: true },
      ),
      postProcessor: templateProcessor('<s> $A', '<s> $A $B'),
      decoder: decoderSequence(replaceDecoder('▁', ' '), byteFallbackDecoder(), fuseDecoder(), stripDecoder(' ', 1, 0)),
    }),
  wordLevel: () =>
    tokeniser({
      preTokeniser: whitespacePreTokeniser(),
      model: wordLevelStage(P.wordLevel.model.vocab as Record<string, number>, { unknown: '[UNK]' }),
    }),
}

/** An encoding in the fixture's terms: offsets converted to code points of the sequence they index. */
function asHf(e: Encoding): Omit<HfEncoding, 'overflowing'> & { overflowing: unknown[] } {
  const seq = toFlat(e.sequenceIds)
  const cp = (k: number, x: number) => (seq[k] < 0 ? x : [...e.sources[seq[k]].slice(0, x)].length)
  return {
    ids: Array.from(toFlat(e.ids)),
    tokens: [...e.tokens],
    offsets: toRows(e.offsets).map(([s, t], k) => [cp(k, s), cp(k, t)] as [number, number]),
    wordIds: Array.from(toFlat(e.wordIds)),
    typeIds: Array.from(toFlat(e.typeIds)),
    specialTokensMask: Array.from(toFlat(e.specialTokensMask)),
    attentionMask: Array.from(toFlat(e.attentionMask)),
    overflowing: e.overflowing.map(asHf),
  }
}

type Compared = Omit<HfEncoding, 'overflowing'> & { overflowing: Compared[] }
const strip = ({
  ids,
  tokens,
  offsets,
  wordIds,
  typeIds,
  specialTokensMask,
  attentionMask,
  overflowing,
}: HfEncoding): Compared => ({
  ids,
  tokens,
  offsets,
  wordIds,
  typeIds,
  specialTokensMask,
  attentionMask,
  overflowing: overflowing.map(strip),
})

describe('pipelines', () => {
  for (const [name, make] of Object.entries(PIPELINES)) {
    const t = make()
    const cases = P[name as Name]
    it.each(cases.encode.map((c) => [c.text, c] as const))(`${name} encodes %j as tokenizers does`, (_, c) => {
      const e = encodeText(t, c.text)
      expect(asHf(e)).toEqual(strip(c))
      expect(decodeIds(t, e.ids)).toBe(c.decoded)
    })
    if (cases.pairs)
      it.each(cases.pairs.map((c) => [`${c.a} | ${c.b}`, c] as const))(`${name} encodes the pair %j`, (_, c) => {
        expect(asHf(encodeText(t, c.a, c.b))).toEqual(strip(c))
      })
  }
})

describe('truncation and padding', () => {
  const wp = PIPELINES.wordPiece()
  const T = P.truncation
  it.each(T.single.map((c) => [c.text, c] as const))('cuts %j into overlapping windows (10, stride 3)', (_, c) => {
    const t = withStages(wp, { truncation: truncation(10, { stride: 3 }) })
    expect(asHf(encodeText(t, c.text))).toEqual(strip(c))
  })
  it('truncates a pair longest-first (16, stride 2), overflow in tokenizers order', () => {
    const t = withStages(wp, { truncation: truncation(16, { stride: 2 }) })
    for (const c of T.pairs) expect(asHf(encodeText(t, c.a, c.b))).toEqual(strip(c))
  })
  const STRATEGY = { longest_first: 'longestFirst', only_first: 'onlyFirst', only_second: 'onlySecond' } as const
  it.each(T.variants.map((c) => [`${c.strategy} ${c.direction} ${c.maxLength}/${c.stride}: ${c.a}`, c] as const))(
    'tokenises only the splits truncation can reach, then cuts (%s)',
    (_, c) => {
      const options = { stride: c.stride, strategy: STRATEGY[c.strategy], direction: c.direction }
      const t = withStages(wp, { truncation: truncation(c.maxLength, options) })
      expect(asHf(encodeText(t, c.a, c.b))).toEqual(strip(c))
    },
  )
  it('pads a batch to its longest, and to a fixed length on the left', () => {
    const texts = P.wordPiece.encode.map((c) => c.text)
    const right = withStages(wp, { padding: padding('[PAD]', 0) })
    expect(encodeBatch(right, texts).map(asHf)).toEqual(T.batch.map(strip))
    const left = withStages(wp, { padding: padding('[PAD]', 0, { length: 16, direction: 'left' }) })
    expect(encodeBatch(left, texts.slice(0, 2)).map(asHf)).toEqual(T.left.map(strip))
  })
})
