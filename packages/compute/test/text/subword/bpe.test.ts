/**
 * Byte-pair encoding against Sennrich et al. (2016)'s corpus, as worked in the byte-pair-encoding note: the first ten
 * merges with their counts, the corpus after them, and the encoding of the unseen word "lowest".
 */
import { describe, expect, it } from 'vitest'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  bpe,
  bpeDecode,
  bpeEncode,
  bpeModel,
  bpePairCounts,
  bpeSegment,
  bpeSteps,
  byteAlphabet,
  byteSymbols,
  textFromByteSymbols,
} from 'aifn-compute/text/subword'

const SENNRICH = new Map([
  ['low', 5],
  ['lower', 2],
  ['newest', 6],
  ['widest', 3],
])

describe('bpeSteps', () => {
  it('learns the first ten merges of the worked example, ties to the pair met first', () => {
    const s = run(bpeSteps(SENNRICH, { merges: 10 }), undefined, 20)
    expect(s.merges.map((m) => [m.left, m.right, m.count])).toEqual([
      ['e', 's', 9],
      ['es', 't', 9],
      ['est', '</w>', 9],
      ['l', 'o', 7],
      ['lo', 'w', 7],
      ['n', 'e', 6],
      ['ne', 'w', 6],
      ['new', 'est</w>', 6],
      ['low', '</w>', 5],
      ['w', 'i', 3],
    ])
    expect(s.segmentations).toEqual([['low</w>'], ['low', 'e', 'r', '</w>'], ['newest</w>'], ['wi', 'd', 'est</w>']])
    expect(s.done).toBe(true)
    expect(s.t).toBe(10)
  })
  it('counts the corpus in symbols: 95 before, 28 tokens across the distinct words after ten merges', () => {
    expect(run(bpeSteps(SENNRICH), undefined, 0).symbols).toBe(95)
    const tr = trace(bpeSteps(SENNRICH, { merges: 10 }), undefined, 20)
    const final = tr.final
    expect(final.symbols).toBe(28) // 5·1 + 2·4 + 6·1 + 3·3
    // each step's symbol count falls by its merge count
    expect(final.vocabulary.length).toBe(11 + 10)
  })
  it('starts from step 0 with base symbols and no merge', () => {
    const s0 = run(bpeSteps(SENNRICH), undefined, 0)
    expect(s0.merge).toBeNull()
    expect(s0.vocabulary).toEqual(['d', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'w', '</w>'])
    expect(toFlat(s0.wordCounts)).toEqual([5, 2, 6, 3])
  })
  it('stops when the best pair is rarer than minCount', () => {
    const s = run(bpeSteps(SENNRICH, { minCount: 6 }), undefined, 100)
    expect(s.merges.at(-1)!.count).toBe(6)
    expect(s.done).toBe(true)
  })
  it('sorts the base alphabet by code point, astral characters last (review: UTF-16 order put 😀 before ～)', () => {
    const s0 = run(bpeSteps(['\u{1F600}', '\uFF5E', 'a'], { merges: 0 }), undefined, 1)
    expect(s0.vocabulary.slice(0, 3)).toEqual(['a', '\uFF5E', '\u{1F600}'])
  })

  it('counts pairs in order of first occurrence', () => {
    const { pairs, counts } = bpePairCounts([['a', 'b', 'a', 'b']], [2])
    expect(pairs).toEqual([
      ['a', 'b'],
      ['b', 'a'],
    ])
    expect(toFlat(counts)).toEqual([4, 2])
  })
})

describe('bpe encoding', () => {
  const model = bpe(SENNRICH, { merges: 10 })
  it('encodes "lowest" as low + est</w>, replaying the merges in order', () => {
    expect(bpeSegment(model, 'lowest').map((p) => p.token)).toEqual(['low', 'est</w>'])
    expect(bpeSegment(model, 'lowest', 1).map((p) => p.token)).toEqual(['l', 'o', 'w', 'es', 't', '</w>'])
    expect(bpeSegment(model, 'lowest', 3).map((p) => p.token)).toEqual(['l', 'o', 'w', 'est</w>'])
  })
  it('gives offsets into the text and decodes back', () => {
    const t = bpeEncode(model, 'lowest newer')
    expect(t.tokens).toEqual(['low', 'est</w>', 'new', 'e', 'r', '</w>'])
    expect(toRows(t.offsets)).toEqual([
      [0, 3],
      [3, 6],
      [7, 10],
      [10, 11],
      [11, 12],
      [12, 12],
    ])
    expect(bpeDecode(model, t.tokens)).toBe('lowest newer')
  })
  it('makes a model of any training state', () => {
    const s = run(bpeSteps(SENNRICH), undefined, 2)
    expect(bpeModel(s).merges).toHaveLength(2)
  })
})

describe('byte-level BPE', () => {
  it('maps the 256 bytes to printable symbols as GPT-2', () => {
    const a = byteAlphabet()
    expect(a).toHaveLength(256)
    expect(new Set(a).size).toBe(256)
    expect(a[32]).toBe('Ġ')
    expect(a[10]).toBe('Ċ')
    expect(a[65]).toBe('A')
  })
  it('round-trips multi-byte text and keeps the character range of each byte', () => {
    const sym = byteSymbols('né')
    expect(sym.map((s) => [s.start, s.end])).toEqual([
      [0, 1],
      [1, 2],
      [1, 2],
    ])
    expect(textFromByteSymbols(sym.map((s) => s.token).join(''))).toBe('né')
  })
  it('trains on bytes and decodes exactly with the GPT-2 pre-tokeniser', () => {
    const text = 'the cat and the hat, the café'
    const words = text.match(/ ?\p{L}+| ?[^\s\p{L}]+/gu)!
    const model = bpe(words, { unit: 'byte', merges: 20 })
    expect(model.vocabulary.length).toBeGreaterThan(256)
    const t = bpeEncode(model, text)
    expect(bpeDecode(model, t.tokens)).toBe(text)
    expect(t.tokens).toContain('Ġthe')
  })
})
