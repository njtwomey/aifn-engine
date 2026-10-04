/**
 * Laws of the tokeniser pipeline: the priority-queue BPE encoder equals the naive merge loop; BPE-dropout is seeded and
 * reduces to BPE at p = 0 and to characters at p = 1; byte-level, byte and byte-fallback pipelines round-trip
 * arbitrary Unicode exactly; offsets survive normalisation; training through the pipeline meets its vocabulary size.
 */
import { describe, expect, it } from 'vitest'
import { stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { bpe, bpeSegment, type BpeModel } from 'aifn-compute/text/subword'
import {
  byteLevelDecoder,
  byteLevelPreTokeniser,
  byteTokeniser,
  decodeIds,
  decoderSequence,
  digitsPreTokeniser,
  encodeText,
  lowercaseNormaliser,
  metaspaceDecoder,
  metaspacePreTokeniser,
  byteFallbackDecoder,
  fuseDecoder,
  normaliserSequence,
  preTokeniserSequence,
  trainTokeniser,
  trainedModel,
  trainingSteps,
  unicodeNormaliser,
  vocabularySize,
  whitespacePreTokeniser,
  type Trainer,
} from 'aifn-compute/text/pipeline'

const CORPUS = [
  'the quick brown fox jumps over the lazy dog',
  'a tokeniser splits text into tokens and tokens become ids',
  'lower lowest newer newest wider widest',
  'byte pair encoding merges the most frequent pair again and again',
  'numbers like 1234 and 56 are split into digits',
]

/** The textbook loop: repeatedly merge every occurrence of the lowest-ranked adjacent pair, left to right. */
function naiveSegment(model: BpeModel, word: string): string[] {
  const rank = new Map(model.merges.map((m, k) => [m.left + '\u0000' + m.right, k] as [string, number]).reverse())
  let seg = [...word, ...(model.endOfWord ? [model.endOfWord] : [])]
  for (;;) {
    let best = Infinity
    for (let i = 0; i + 1 < seg.length; i++) best = Math.min(best, rank.get(seg[i] + '\u0000' + seg[i + 1]) ?? Infinity)
    if (best === Infinity) return seg
    const { left, right, merged } = model.merges[best]
    const out: string[] = []
    for (let i = 0; i < seg.length; i++)
      if (i + 1 < seg.length && seg[i] === left && seg[i + 1] === right) {
        out.push(merged)
        i++
      } else out.push(seg[i])
    seg = out
  }
}

function randomWord(s: ReturnType<typeof stream>, alphabet: string): string {
  const n = 1 + Math.floor((uniform(s) as number) * 14)
  let w = ''
  for (let k = 0; k < n; k++) w += alphabet[Math.floor((uniform(s) as number) * alphabet.length)]
  return w
}

describe('BPE encoding with a priority queue', () => {
  const model = bpe(CORPUS.join(' ').split(' '), { merges: 80, minCount: 1 })
  it('equals the naive merge loop on 2000 random words (repeated letters, unseen characters)', () => {
    const s = stream('bpe-heap')
    for (let k = 0; k < 2000; k++) {
      const w = randomWord(s, 'aaeeesttlowrdnpq!')
      expect(bpeSegment(model, w).map((p) => p.token)).toEqual(naiveSegment(model, w))
    }
  })
  it('keeps ranges that tile the word', () => {
    const pieces = bpeSegment(model, 'newestlowest')
    expect(pieces.map((p) => 'newestlowest'.slice(p.start, p.end)).join('')).toBe('newestlowest')
  })
})

describe('BPE-dropout', () => {
  const model = bpe(CORPUS.join(' ').split(' '), { merges: 80, minCount: 1 })
  const word = 'newest'
  it('is BPE at p = 0 and characters at p = 1', () => {
    const tokens = (p: number) => bpeSegment(model, word, { dropout: p, stream: stream(1) }).map((x) => x.token)
    expect(tokens(0)).toEqual(bpeSegment(model, word).map((x) => x.token))
    expect(tokens(1)).toEqual([...word, '</w>'])
  })
  it('is reproducible from its seed and samples several segmentations at p = 0.3, each spelling the word', () => {
    const sample = (seed: number) =>
      bpeSegment(model, word, { dropout: 0.3, stream: stream(seed) })
        .map((x) => x.token)
        .join(' ')
    expect(sample(7)).toBe(sample(7))
    const seen = new Set(Array.from({ length: 60 }, (_, k) => sample(k)))
    expect(seen.size).toBeGreaterThan(3)
    for (const s of seen) expect(s.replaceAll(' ', '').replace('</w>', '')).toBe(word)
  })
})

/** Random valid Unicode: ASCII, Latin-1, combining marks, CJK, emoji with skin tones and joiners, controls. */
function randomText(s: ReturnType<typeof stream>): string {
  const pools = [
    [0x20, 0x7e],
    [0xa0, 0xff],
    [0x300, 0x36f],
    [0x4e00, 0x4e80],
    [0x1f600, 0x1f64f],
    [0x1f3fb, 0x1f3ff],
    [0x200d, 0x200d],
    [0x0, 0x1f],
    [0x10000, 0x10ffff],
  ]
  let out = ''
  const n = Math.floor((uniform(s) as number) * 40)
  for (let k = 0; k < n; k++) {
    const [a, b] = pools[Math.floor((uniform(s) as number) * pools.length)]
    let c = a + Math.floor((uniform(s) as number) * (b - a + 1))
    if (c >= 0xd800 && c <= 0xdfff) c = 0x41
    out += String.fromCodePoint(c)
  }
  return out
}

describe('round trips', () => {
  const byteLevel = trainTokeniser({ preTokeniser: byteLevelPreTokeniser(), decoder: byteLevelDecoder() }, CORPUS, {
    type: 'bpe',
    vocabularySize: 320,
    byteAlphabet: true,
    minCount: 1,
  })
  const sentencePiece = trainTokeniser(
    {
      preTokeniser: metaspacePreTokeniser({ prependScheme: 'first' }),
      decoder: decoderSequence(metaspaceDecoder({ prependScheme: 'first' }), byteFallbackDecoder(), fuseDecoder()),
    },
    CORPUS,
    { type: 'bpe', vocabularySize: 400, byteFallback: true, specials: ['<unk>', '<s>', '</s>'] },
  )
  const bytes = byteTokeniser()
  it('byte-level BPE, the byte tokeniser and SentencePiece-style byte fallback decode what they encode', () => {
    const s = stream('round-trip')
    for (let k = 0; k < 300; k++) {
      const text = randomText(s)
      expect(decodeIds(byteLevel, encodeText(byteLevel, text).ids)).toBe(text)
      expect(decodeIds(bytes, encodeText(bytes, text).ids)).toBe(text)
      // Metaspace drops one leading space on decoding and reads "▁" as a space.
      const sp = text.replace(/^ +/u, '').replaceAll('▁', '')
      expect(decodeIds(sentencePiece, encodeText(sentencePiece, sp).ids)).toBe(sp)
    }
  })
  it('the byte tokeniser has 259 ids: three specials, then byte b at b + 3, and ends with </s>', () => {
    const e = encodeText(bytes, 'hé')
    expect(vocabularySize(bytes)).toBe(259)
    expect(Array.from(toFlat(e.ids))).toEqual([0x68 + 3, 0xc3 + 3, 0xa9 + 3, 1])
    expect(toRows(e.offsets)).toEqual([
      [0, 1],
      [1, 2],
      [1, 2],
      [0, 0],
    ])
  })
  it('byte fallback spells an unseen character as <0xNN> tokens with the character’s offsets', () => {
    const e = encodeText(sentencePiece, 'a 🙂')
    const k = e.tokens.indexOf('<0xF0>')
    expect(e.tokens.slice(k, k + 4)).toEqual(['<0xF0>', '<0x9F>', '<0x99>', '<0x82>'])
    expect(toRows(e.offsets).slice(k, k + 4)).toEqual(Array(4).fill([2, 4]))
  })
})

describe('offsets through normalisation', () => {
  it('point into the original text after NFKC and lower case', () => {
    const t = trainTokeniser(
      {
        normaliser: normaliserSequence(unicodeNormaliser('NFKC'), lowercaseNormaliser()),
        preTokeniser: whitespacePreTokeniser(),
      },
      ['full fine cafe café'],
      { type: 'wordLevel' },
    )
    const text = 'Ｆｕｌｌ ﬁne CAFÉ'
    const e = encodeText(t, text)
    expect(e.tokens).toEqual(['full', 'fine', 'café'])
    expect(toRows(e.offsets).map(([s, x]) => text.slice(s, x))).toEqual(['Ｆｕｌｌ', 'ﬁne', 'CAFÉ'])
  })
  it('splits digits one by one (LLaMA) inside a sequence of pre-tokenisers', () => {
    const t = trainTokeniser(
      { preTokeniser: preTokeniserSequence(whitespacePreTokeniser(), digitsPreTokeniser(true)) },
      ['in 2024 we'],
      { type: 'character' },
    )
    expect(encodeText(t, 'x 2024').tokens).toEqual(['<unk>', '2', '0', '2', '4'])
  })
})

describe('training through the pipeline', () => {
  const parts = { preTokeniser: whitespacePreTokeniser() }
  it.each<Extract<Trainer, { vocabularySize: number }>>([
    { type: 'bpe', vocabularySize: 60, specials: ['[UNK]'], unknown: '[UNK]', minCount: 1 },
    { type: 'wordPiece', vocabularySize: 70, specials: ['[UNK]', '[CLS]', '[SEP]'] },
    { type: 'unigram', vocabularySize: 50 },
  ])('$type reaches its vocabulary size and its step-through model encodes', (trainer) => {
    const t = trainTokeniser(parts, CORPUS, trainer)
    expect(vocabularySize(t)).toBeLessThanOrEqual(trainer.vocabularySize)
    expect(vocabularySize(t)).toBeGreaterThan(trainer.vocabularySize * 0.6)
    const state = run(trainingSteps(parts, CORPUS, trainer), undefined, 3)
    const model = trainedModel(trainer, state)
    expect(encodeText({ ...t, model }, 'the newest tokens').tokens.length).toBeGreaterThan(2)
  })
})
