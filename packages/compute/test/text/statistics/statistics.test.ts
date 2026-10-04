/** Tokeniser statistics on small cases worked by hand. */
import { describe, expect, it } from 'vitest'
import { byteTokeniser, trainTokeniser, whitespacePreTokeniser } from 'aifn-compute/text/pipeline'
import { tokenisationStatistics, tokeniserStatistics } from 'aifn-compute/text/statistics'
import { characterTokenise, whitespaceTokenise } from 'aifn-compute/text/tokenise'

describe('tokenisationStatistics', () => {
  it('counts words, bytes, characters and tokens', () => {
    const s = tokenisationStatistics([whitespaceTokenise('naïve café'), whitespaceTokenise('ok')])
    expect([s.texts, s.words, s.characters, s.bytes, s.tokens]).toEqual([2, 3, 12, 14, 3])
    expect(s.fertility).toBe(1)
    expect(s.bytesPerToken).toBeCloseTo(14 / 3)
    const c = tokenisationStatistics([characterTokenise('ab cd', { skipWhitespace: true })])
    expect([c.fertility, c.charactersPerToken]).toEqual([2, 5 / 4])
  })
})

describe('tokeniserStatistics', () => {
  it('a byte tokeniser has one byte per token and fertility bytes per word', () => {
    const s = tokeniserStatistics(byteTokeniser(), ['héllo wörld'])
    expect(s.bytesPerToken).toBe(1)
    expect(s.fertility).toBe(13 / 2)
    expect(s.vocabularyUsed).toBe(10)
    expect(s.vocabularyUsage).toBeCloseTo(10 / 259)
  })
  it('a word-level vocabulary: unknown rate and word coverage', () => {
    const t = trainTokeniser({ preTokeniser: whitespacePreTokeniser() }, ['the cat sat'], { type: 'wordLevel' })
    const s = tokeniserStatistics(t, ['the dog sat', 'a cat'])
    expect([s.tokens, s.unknownTokens, s.unknownRate, s.wordCoverage]).toEqual([5, 2, 2 / 5, 3 / 5])
    expect([s.vocabularySize, s.vocabularyUsed]).toEqual([4, 4])
  })
})
