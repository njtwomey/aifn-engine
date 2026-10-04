/**
 * `aifn-compute/text/cooccurrence` against the worked example of the co-occurrence and PMI note: thirteen sentences, a window
 * of two, 154 pairs; cat's PMIs with "the" and "slept", with and without context smoothing; and the cosine
 * similarities of count, PPMI and smoothed-PPMI rows.
 */
import { describe, expect, it } from 'vitest'
import { matmul, toFlat, toRows, transpose } from 'aifn-compute/foundation/tensor'
import { cooccurrence, pmi, ppmi, wordVectors } from 'aifn-compute/text/cooccurrence'
import { tokenId } from 'aifn-compute/text/vocabulary'

const SENTENCES = [
  'the cat is a pet',
  'the dog is a pet',
  'we fed the cat',
  'we fed the dog',
  'the vet saw the cat',
  'the vet saw the dog',
  'the cat chased the mouse',
  'the dog chased the cat',
  'the mouse ate the cheese',
  'the mouse ate the grain',
  'we ate the cheese',
  'the cat slept',
  'the dog slept',
].map((s) => s.split(' '))

const C = cooccurrence(SENTENCES, { window: 2 })
const id = (w: string) => tokenId(C.words, w)
const rows = toRows(C.matrix)
const cosine = (a: number[], b: number[]) =>
  a.reduce((s, x, k) => s + x * b[k], 0) / Math.hypot(...a) / Math.hypot(...b)

describe('cooccurrence', () => {
  it('counts 154 pairs over 16 word types, 14 of them for cat', () => {
    expect(C.words.tokens).toHaveLength(16)
    expect(toFlat(C.matrix).reduce((s, x) => s + x, 0)).toBe(154)
    expect(rows[id('cat')].reduce((s, x) => s + x, 0)).toBe(14)
    expect(rows[id('cat')][id('the')]).toBe(7)
    expect(rows[id('cat')][id('chased')]).toBe(2)
    expect(rows.map((r) => r[id('the')]).reduce((s, x) => s + x, 0)).toBe(47)
  })
  it('is symmetric with both sides, and weights by distance on request', () => {
    const m = toRows(C.matrix)
    m.forEach((r, i) => r.forEach((x, j) => expect(x).toBe(m[j][i])))
    const h = toRows(cooccurrence([['a', 'b', 'c']], { window: 2, weighting: 'harmonic' }).matrix)
    const v = cooccurrence([['a', 'b', 'c']], { window: 2 }).words
    expect(h[tokenId(v, 'a')][tokenId(v, 'c')]).toBe(0.5)
    const right = toRows(cooccurrence([['a', 'b']], { rightOnly: true }).matrix)
    expect(right.flat().reduce((s, x) => s + x, 0)).toBe(1)
  })
})

describe('pmi and ppmi', () => {
  it('gives PMI(cat, the) = 0.712 and PMI(cat, slept) = 1.459 bits', () => {
    const p = toRows(pmi(C.matrix, { base: 2 }))
    expect(p[id('cat')][id('the')]).toBeCloseTo(Math.log2((7 * 154) / (14 * 47)), 12)
    expect(+p[id('cat')][id('the')].toFixed(3)).toBe(0.712)
    expect(+p[id('cat')][id('slept')].toFixed(3)).toBe(1.459)
    expect(p[id('cat')][id('cheese')]).toBe(-Infinity)
  })
  it('with context smoothing α = 0.75 raises "the" to 1.195 bits and lowers "slept" to 1.054', () => {
    const p = toRows(pmi(C.matrix, { base: 2, alpha: 0.75 }))
    expect(+p[id('cat')][id('the')].toFixed(3)).toBe(1.195)
    expect(+p[id('cat')][id('slept')].toFixed(3)).toBe(1.054)
  })
  it('reproduces the cosine similarities of counts, PPMI and smoothed PPMI', () => {
    const tables = [toRows(C.matrix), toRows(ppmi(C.matrix)), toRows(ppmi(C.matrix, { alpha: 0.75 }))]
    const sims = (a: string, b: string) => tables.map((t) => +cosine(t[id(a)], t[id(b)]).toFixed(3))
    expect(sims('cat', 'dog')).toEqual([0.993, 0.939, 0.919])
    expect(sims('cat', 'mouse')).toEqual([0.887, 0.409, 0.555])
    expect(sims('mouse', 'cheese')).toEqual([0.904, 0.823, 0.862])
    expect(sims('cat', 'cheese')).toEqual([0.65, 0.068, 0.208])
  })
  it('clips at zero and shifts by log k', () => {
    const p = toFlat(pmi(C.matrix))
    const pp = toFlat(ppmi(C.matrix))
    const sp = toFlat(ppmi(C.matrix, { shift: 5 }))
    p.forEach((x, k) => {
      expect(pp[k]).toBeCloseTo(Math.max(x, 0), 14)
      expect(sp[k]).toBeCloseTo(Math.max(x - Math.log(5), 0), 14)
    })
  })
})

describe('wordVectors', () => {
  it('has Gram matrix M Mᵀ at full rank with p = 1, and keeps the leading directions', () => {
    const M = ppmi(C.matrix)
    const W = wordVectors(M, 16, { power: 1 })
    const a = toRows(matmul(W, transpose(W)))
    const b = toRows(matmul(M, transpose(M)))
    a.forEach((r, i) => r.forEach((x, j) => expect(x).toBeCloseTo(b[i][j], 8)))
    const W2 = wordVectors(M, 2)
    expect(W2.shape).toEqual([16, 2])
    expect(() => wordVectors(M, 17)).toThrow()
  })
})
