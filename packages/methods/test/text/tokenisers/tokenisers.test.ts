/** The tokeniser suite: every member trains on the corpus, encodes and decodes, and the subword sizes are met. */
import { describe, expect, it } from 'vitest'
import { decodeIds, encodeText, vocabularySize } from 'aifn-compute/text/pipeline'
import { tokeniserStatistics } from 'aifn-compute/text/statistics'
import { TOKENISER_CORPUS, TOKENISER_KINDS, tokeniserSuite } from 'aifn-methods/text/tokenisers'

describe('tokeniserSuite', () => {
  const suite = tokeniserSuite(TOKENISER_CORPUS, { vocabularySize: 300 })
  const text = "The newest tokenisers don't split 2024 the same way: café 🙂"
  it.each(TOKENISER_KINDS)('%s encodes with offsets into the text', (kind) => {
    const e = encodeText(suite[kind], text)
    expect(e.tokens.length).toBeGreaterThan(5)
    const o = e.offsets.data
    for (let k = 0; k < e.tokens.length; k++) expect(o[2 * k + 1]).toBeLessThanOrEqual(text.length)
  })
  it('WordPiece maps unknown words to [UNK], never to another special', () => {
    const e = encodeText(suite.wordPiece, 'zzz qqq')
    expect(e.tokens.filter((t) => t.startsWith('[') && t !== '[CLS]' && t !== '[SEP]')).toEqual(
      e.tokens.filter((t) => t === '[UNK]'),
    )
    expect(suite.wordPiece.model.vocabulary.tokens[suite.wordPiece.model.vocabulary.unknown]).toBe('[UNK]')
  })
  it('the lossless tokenisers decode exactly', () => {
    for (const k of ['byte', 'byteLevelBpe', 'sentencePiece', 'character'] as const)
      expect(decodeIds(suite[k], encodeText(suite[k], text).ids)).toBe(text)
  })
  it('subword models meet their vocabulary sizes, byte models with their 256 bytes on top', () => {
    expect(vocabularySize(suite.bpe)).toBeLessThanOrEqual(300)
    expect(vocabularySize(suite.byteLevelBpe)).toBeLessThanOrEqual(556)
    expect(vocabularySize(suite.wordPiece)).toBeLessThanOrEqual(300)
  })
  it('fertility falls from bytes to characters to subwords', () => {
    const f = (k: (typeof TOKENISER_KINDS)[number]) => tokeniserStatistics(suite[k], TOKENISER_CORPUS).fertility
    expect(f('byte')).toBeGreaterThan(f('character'))
    expect(f('character')).toBeGreaterThan(f('bpe'))
    expect(f('bpe')).toBeGreaterThan(1)
  })
})
