/** `aifn-compute/text/vocabulary`: counts, special tokens, filtering, id orders, encoding and decoding. */
import { describe, expect, it } from 'vitest'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  buildVocabulary,
  decodeTokens,
  encodeTokens,
  tokenCounts,
  tokenId,
  vocabularyOf,
} from 'aifn-compute/text/vocabulary'

const DOCS = [
  ['the', 'cat', 'sat', 'on', 'the', 'mat'],
  ['the', 'dog', 'sat', 'on', 'the', 'log'],
  ['the', 'cat', 'chased', 'the', 'dog'],
]

describe('tokenCounts', () => {
  it('counts tokens and documents in order of first appearance', () => {
    const c = tokenCounts(DOCS)
    expect(c.tokens.slice(0, 4)).toEqual(['the', 'cat', 'sat', 'on'])
    expect(toFlat(c.counts).slice(0, 4)).toEqual([6, 2, 2, 2])
    expect(toFlat(c.documentCounts).slice(0, 4)).toEqual([3, 2, 2, 2])
    expect(tokenCounts(['a', 'b', 'a']).tokens).toEqual(['a', 'b'])
  })
})

describe('buildVocabulary', () => {
  it('puts specials first, then tokens by descending count, ties alphabetical', () => {
    const v = buildVocabulary(DOCS, { specials: ['<pad>', '<unk>'] })
    expect(v.tokens).toEqual(['<pad>', '<unk>', 'the', 'cat', 'dog', 'on', 'sat', 'chased', 'log', 'mat'])
    expect(v.unknown).toBe(1)
    expect(toFlat(v.counts).slice(0, 4)).toEqual([0, 0, 6, 2])
  })
  it('filters by minimum count and caps the size by frequency', () => {
    expect(buildVocabulary(DOCS, { minCount: 2 }).tokens).toEqual(['<unk>', 'the', 'cat', 'dog', 'on', 'sat'])
    expect(buildVocabulary(DOCS, { maxSize: 2, specials: [] }).tokens).toEqual(['the', 'cat'])
    const alpha = buildVocabulary(DOCS, { maxSize: 3, specials: [], order: 'alphabetical' })
    expect(alpha.tokens).toEqual(['cat', 'dog', 'the'])
    expect(buildVocabulary(DOCS, { specials: [], order: 'appearance' }).tokens.slice(0, 3)).toEqual([
      'the',
      'cat',
      'sat',
    ])
    expect(buildVocabulary(DOCS, { specials: [] }).unknown).toBe(-1)
  })
  it('rejects an unknown token that is not special', () => {
    expect(() => buildVocabulary(DOCS, { specials: [], unknown: '<unk>' })).toThrow(DomainError)
  })
})

describe('encodeTokens and decodeTokens', () => {
  const v = buildVocabulary(DOCS)
  it('maps unknown tokens to the unknown id and round-trips known ones', () => {
    const ids = encodeTokens(v, ['the', 'zebra', 'cat'])
    expect(ids.dtype).toBe('int32')
    expect(toFlat(ids)).toEqual([1, 0, 2])
    expect(decodeTokens(v, ids)).toEqual(['the', '<unk>', 'cat'])
    expect(tokenId(v, 'zebra')).toBe(-1)
  })
  it('throws or skips without an unknown token', () => {
    const fixed = vocabularyOf(['a', 'b'])
    expect(() => encodeTokens(fixed, ['c'])).toThrow(DomainError)
    expect(toFlat(encodeTokens(fixed, ['b', 'c', 'a'], { onUnknown: 'skip' }))).toEqual([1, 0])
    expect(() => decodeTokens(fixed, [5])).toThrow(DomainError)
    expect(() => vocabularyOf(['a', 'a'])).toThrow(DomainError)
  })
})
