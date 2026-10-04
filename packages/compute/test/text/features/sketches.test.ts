/**
 * One-hot tokens, shingles and Jaccard similarity, MinHash (against datasketch's estimates on the same shingle sets)
 * and LSH banding laws.
 */
import { describe, expect, it } from 'vitest'
import { pairwiseDistances } from 'aifn-compute/numerics/linalg'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import {
  characterShingles,
  jaccardSimilarity,
  minHashSignature,
  minHashSignatures,
  minHashSimilarity,
  minHashStandardError,
  oneHotTokens,
  wordShingles,
} from 'aifn-compute/text/features'
import { lshBands, lshCandidates, lshProbability, lshThreshold } from 'aifn-compute/numerics/neighbours'
import { buildVocabulary } from 'aifn-compute/text/vocabulary'
import { fixture } from '../../fixtures'

const F = fixture('text/representations') as {
  minhash: { pairs: { a: string; b: string; jaccard: number; datasketch: number }[] }
}

describe('oneHotTokens', () => {
  it('puts one 1 per position, and every pair of distinct words at distance √2', () => {
    const tokens = ['the', 'cat', 'sat', 'the', 'mat']
    const v = buildVocabulary([tokens], { specials: [] })
    const M = toRows(oneHotTokens(tokens, v))
    expect(M).toHaveLength(4)
    for (let j = 0; j < 5; j++) expect(M.reduce((s, r) => s + r[j], 0)).toBe(1)
    const D = toRows(pairwiseDistances(oneHotTokens(v.tokens, v)))
    D.forEach((r, i) => r.forEach((d, j) => expect(d).toBeCloseTo(i === j ? 0 : Math.SQRT2, 14)))
    expect(() => oneHotTokens(['dog'], v)).toThrow()
    expect(toFlat(oneHotTokens(['dog'], v, { onUnknown: 'zero' }))).toEqual([0, 0, 0, 0])
  })
})

describe('shingles and Jaccard', () => {
  it('are sets, in code-point order, with collapsed white space', () => {
    expect(characterShingles('abab', 2)).toEqual(['ab', 'ba'])
    expect(characterShingles('a  b', 3)).toEqual(['a b'])
    expect(characterShingles('ab', 5)).toEqual(['ab'])
    expect(wordShingles(['a', 'rose', 'is', 'a', 'rose'], 2)).toEqual(['a rose', 'is a', 'rose is'])
  })
  it('Jaccard: |A ∩ B| / |A ∪ B|, matching the fixture', () => {
    expect(jaccardSimilarity(['a', 'b', 'c'], ['b', 'c', 'd'])).toBe(0.5)
    expect(jaccardSimilarity([], [])).toBe(1)
    for (const p of F.minhash.pairs)
      expect(jaccardSimilarity(characterShingles(p.a, 4), characterShingles(p.b, 4))).toBeCloseTo(p.jaccard, 14)
  })
})

describe('MinHash', () => {
  it('estimates Jaccard within 3 standard errors, as datasketch does', () => {
    for (const p of F.minhash.pairs) {
      const [a, b] = [characterShingles(p.a, 4), characterShingles(p.b, 4)]
      const est = minHashSimilarity(minHashSignature(a, { hashes: 256 }), minHashSignature(b, { hashes: 256 }))
      const se = Math.max(minHashStandardError(p.jaccard, 256), 1 / 256)
      expect(Math.abs(est - p.jaccard)).toBeLessThan(3 * se + 1e-12)
      expect(Math.abs(p.datasketch - p.jaccard)).toBeLessThan(3 * se + 1e-12)
    }
  })
  it('is unbiased: the mean estimate over seeds approaches J', () => {
    const [a, b] = [characterShingles('the cook bakes the bread', 3), characterShingles('the cook bakes the cake', 3)]
    const J = jaccardSimilarity(a, b)
    let mean = 0
    for (let seed = 0; seed < 40; seed++)
      mean +=
        minHashSimilarity(minHashSignature(a, { hashes: 64, seed }), minHashSignature(b, { hashes: 64, seed })) / 40
    expect(Math.abs(mean - J)).toBeLessThan(3 * minHashStandardError(J, 64 * 40))
  })
  it('a longer signature extends a shorter one; equal sets agree everywhere', () => {
    const s = characterShingles('some buses stop near the market', 4)
    expect(toFlat(minHashSignature(s, { hashes: 200 })).slice(0, 50)).toEqual(
      toFlat(minHashSignature(s, { hashes: 50 })),
    )
    const sig = minHashSignature(s, { hashes: 32 })
    expect(minHashSimilarity(sig, minHashSignature([...s].reverse(), { hashes: 32 }))).toBe(1)
    expect(toFlat(minHashSignature([], { hashes: 3 }))).toEqual([2 ** 32, 2 ** 32, 2 ** 32])
  })
})

describe('LSH banding', () => {
  it('S-curve 1 − (1 − sʳ)ᵇ and threshold (1/b)^{1/r}', () => {
    const banding = { bands: 20, rows: 5 }
    expect(lshProbability(0, banding)).toBe(0)
    expect(lshProbability(1, banding)).toBe(1)
    expect(lshProbability(0.8, banding)).toBeCloseTo(1 - (1 - 0.8 ** 5) ** 20, 14)
    expect(lshThreshold(banding)).toBeCloseTo(0.05 ** 0.2, 14)
  })
  it('finds exactly the pairs that share a band, and bands key by index', () => {
    const sets = [
      characterShingles('the cat watches the grass', 4),
      characterShingles('the cat watches the grass', 4),
      characterShingles('a train pulls the timber', 4),
    ]
    const sig = minHashSignatures(sets, { hashes: 32 })
    const banding = { bands: 8, rows: 4 }
    const cands = lshCandidates(sig, banding)
    expect(cands.find((c) => c.i === 0 && c.j === 1)?.bands).toBe(8)
    const rows = toRows(sig)
    for (const c of cands) {
      const shared = lshBands(rows[c.i], banding).filter((k, b) => k === lshBands(rows[c.j], banding)[b]).length
      expect(shared).toBe(c.bands)
    }
    expect(lshBands(rows[0], banding)[3].startsWith('3:')).toBe(true)
    expect(() => lshBands(rows[0], { bands: 9, rows: 4 })).toThrow()
  })
})
