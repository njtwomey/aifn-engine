/**
 * `aifn-compute/text/representations` and the truncated SVD against scikit-learn (CountVectorizer, TruncatedSVD) and numpy
 * (SVD, 3CosAdd), with the laws of LSA, cosine neighbours and random indexing.
 */
import { describe, expect, it } from 'vitest'
import { fromData, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { truncatedSvd } from 'aifn-compute/text/cooccurrence'
import {
  analogy,
  cosineMap,
  cosineSimilarities,
  indexVector,
  lsa,
  nearestByCosine,
  randomIndexing,
  termDocumentMatrix,
  termTermMatrix,
  weightMatrix,
} from 'aifn-compute/text/representations'
import { tokenise } from 'aifn-compute/text/tokenise'
import { tokenId, vocabularyOf } from 'aifn-compute/text/vocabulary'
import { fixture } from '../../fixtures'

type Truncated = { matrix: number[][]; k: number; singularValues: number[]; energy: number[]; lsaRows: number[][] }
const F = fixture('text/representations') as {
  docs: string[]
  vocabulary: string[]
  termDocument: number[][]
  truncated: Record<string, Truncated>
  analogies: { vectors: number[][]; cases: { abc: number[]; answers: number[]; cosines: number[] }[] }
}
const docs = F.docs.map((d) => tokenise(d).tokens)

describe('termDocumentMatrix', () => {
  it("equals CountVectorizer's counts, transposed", () => {
    const td = termDocumentMatrix(docs, { terms: vocabularyOf(F.vocabulary) })
    expect(toRows(td.counts)).toEqual(F.termDocument)
    expect(td.columns).toEqual(['d1', 'd2', 'd3', 'd4', 'd5', 'd6'])
  })
  it('weights: binary, log, unit TF-IDF columns, PPMI non-negative', () => {
    const td = termDocumentMatrix(docs)
    const c = toFlat(td.counts)
    expect(toFlat(weightMatrix(td.counts, 'binary'))).toEqual(Array.from(c, (x) => (x > 0 ? 1 : 0)))
    toFlat(weightMatrix(td.counts, 'log')).forEach((x, i) => expect(x).toBeCloseTo(Math.log1p(c[i]), 14))
    const t = toRows(termDocumentMatrix(docs, { weighting: 'tfidf' }).matrix)
    for (let d = 0; d < 6; d++) expect(Math.hypot(...t.map((r) => r[d]))).toBeCloseTo(1, 12)
    // With binary tf, "the" (in every document) weighs least in each.
    const tb = toRows(termDocumentMatrix(docs, { weighting: 'tfidf', tfidf: { tf: 'binary' } }).matrix)
    const the = tokenId(td.terms, 'the')
    for (let d = 0; d < 6; d++) tb.forEach((r) => r[d] > 0 && expect(r[d]).toBeGreaterThanOrEqual(tb[the][d] - 1e-12))
    expect(Math.min(...toFlat(termDocumentMatrix(docs, { weighting: 'ppmi' }).matrix))).toBe(0)
  })
})

describe('termTermMatrix', () => {
  it('is symmetric with both sides, and a left window is the transpose of a right one', () => {
    const both = toRows(termTermMatrix(docs, { window: 2 }).matrix)
    both.forEach((r, i) => r.forEach((x, j) => expect(x).toBe(both[j][i])))
    const left = toRows(termTermMatrix(docs, { left: 2, right: 0 }).matrix)
    const right = toRows(termTermMatrix(docs, { left: 0, right: 2 }).matrix)
    left.forEach((r, i) => r.forEach((x, j) => expect(x).toBe(right[j][i])))
    left.forEach((r, i) => r.forEach((x, j) => expect(x + right[i][j]).toBe(both[i][j])))
  })
  it('weighs by distance as HAL: L − d + 1', () => {
    const m = termTermMatrix([['a', 'b', 'c', 'd']], { window: 3, left: 0, right: 3, distance: 'hal' })
    const id = (w: string) => tokenId(m.terms, w)
    const r = toRows(m.counts)
    expect([r[id('a')][id('b')], r[id('a')][id('c')], r[id('a')][id('d')]]).toEqual([3, 2, 1])
  })
})

describe('truncatedSvd', () => {
  for (const [name, f] of Object.entries(F.truncated))
    it(`matches TruncatedSVD and numpy (${name}, ${f.matrix.length} × ${f.matrix[0].length}, k = ${f.k})`, () => {
      const t = truncatedSvd(f.matrix, f.k)
      toFlat(t.S).forEach((s, i) => expect(s).toBeCloseTo(f.singularValues[i], 8))
      toFlat(t.energy).forEach((e, i) => expect(e).toBeCloseTo(f.energy[i], 10))
      // U Σ up to sign per column; U and V orthonormal.
      toRows(lsa(f.matrix, f.k).rows).forEach((r, i) =>
        r.forEach((x, j) => expect(Math.abs(x)).toBeCloseTo(f.lsaRows[i][j], 6)),
      )
      const U = toRows(t.U)
      for (let a = 0; a < f.k; a++)
        for (let b = 0; b < f.k; b++) expect(U.reduce((s, r) => s + r[a] * r[b], 0)).toBeCloseTo(a === b ? 1 : 0, 8)
    })
  it('is the full SVD at full rank: U Σ Vᵀ rebuilds the matrix', () => {
    const f = F.truncated.small
    const { U, S, V } = truncatedSvd(f.matrix, 9)
    const [u, s, v] = [toRows(U), toFlat(S), toRows(V)]
    f.matrix.forEach((r, i) =>
      r.forEach((x, j) => expect(u[i].reduce((acc, ui, c) => acc + ui * s[c] * v[j][c], 0)).toBeCloseTo(x, 10)),
    )
  })
})

describe('cosine neighbours and analogies', () => {
  const vocab = vocabularyOf(F.analogies.vectors.map((_, i) => `w${i}`))
  it('3CosAdd equals numpy', () => {
    for (const c of F.analogies.cases) {
      const [a, b, cc] = c.abc.map((i) => `w${i}`)
      const got = analogy(F.analogies.vectors, vocab, a, b, cc, { count: 3 })
      expect(got.map((x) => x.index)).toEqual(c.answers)
      got.forEach((x, i) => expect(x.cosine).toBeCloseTo(c.cosines[i], 12))
    }
  })
  it('ranks by cosine, excludes the query, agrees with the similarity matrix', () => {
    const S = toRows(cosineSimilarities(F.analogies.vectors))
    S.forEach((r, i) => expect(r[i]).toBeCloseTo(1, 14))
    const nn = nearestByCosine(F.analogies.vectors, 3, { count: 11 })
    expect(nn.map((x) => x.index)).not.toContain(3)
    nn.forEach((x) => expect(x.cosine).toBeCloseTo(S[3][x.index], 14))
    for (let i = 1; i < nn.length; i++) expect(nn[i].cosine).toBeLessThanOrEqual(nn[i - 1].cosine)
    expect(() => analogy(F.analogies.vectors, vocab, 'w0', 'nope', 'w1')).toThrow()
  })
  it('one-hot geometry: every pair at cosine 0, and the cosine map ignores row length', () => {
    const I = fromData(
      Float64Array.from({ length: 25 }, (_, k) => (k % 6 === 0 ? 1 : 0)),
      [5, 5],
    )
    toRows(cosineSimilarities(I)).forEach((r, i) => r.forEach((x, j) => expect(x).toBe(i === j ? 1 : 0)))
    const scaled = F.analogies.vectors.map((r, i) => r.map((x) => x * (i + 1)))
    const a = toRows(cosineMap(F.analogies.vectors))
    const b = toRows(cosineMap(scaled))
    a.forEach((r, i) => r.forEach((x, j) => expect(Math.abs(x)).toBeCloseTo(Math.abs(b[i][j]), 10)))
  })
})

describe('randomIndexing', () => {
  it('index vectors are seeded, sparse and ternary', () => {
    const v = toFlat(indexVector('cat', { dimensions: 32, nonZeros: 6, seed: 3 }))
    expect(v.filter((x) => x === 1)).toHaveLength(3)
    expect(v.filter((x) => x === -1)).toHaveLength(3)
    expect(toFlat(indexVector('cat', { dimensions: 32, nonZeros: 6, seed: 3 }))).toEqual(v)
    expect(toFlat(indexVector('dog', { dimensions: 32, nonZeros: 6, seed: 3 }))).not.toEqual(v)
  })
  it('is the co-occurrence matrix times the index vectors', () => {
    const ri = randomIndexing(docs, { dimensions: 16, window: 2 })
    const C = toRows(termTermMatrix(docs, { terms: ri.words, window: 2 }).counts)
    const R = toRows(ri.index)
    toRows(ri.vectors).forEach((row, w) =>
      row.forEach((x, j) =>
        expect(x).toBeCloseTo(
          C[w].reduce((s, c, i) => s + c * R[i][j], 0),
          12,
        ),
      ),
    )
  })
})
