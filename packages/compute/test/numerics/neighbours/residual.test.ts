/**
 * Residual vector quantisation: the distortion falls level by level and matches its codes; encoding is the greedy
 * nearest-codeword walk; decoding sums codewords over leading levels; one level is k-means; the code prefix tree's
 * counts, spans and structure.
 */
import { describe, expect, it } from 'vitest'
import { child, normal, stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { codePrefixTree, residualQuantiser, rqDecode, rqEncode, trainCodebook } from 'aifn-compute/numerics/neighbours'

const data = normal(stream(11), 0, 1, { shape: [500, 3] })
const rq = residualQuantiser(data, { levels: 4, codewords: 8, stream: stream(12) })

describe('residualQuantiser', () => {
  it('has D codebooks of K codewords, and a distortion that never rises from one level to the next', () => {
    expect(rq.codebooks.shape).toEqual([4, 8, 3])
    const d = Array.from(rq.distortionByLevel)
    for (let l = 1; l < d.length; l++) expect(d[l]).toBeLessThanOrEqual(d[l - 1] + 1e-12)
    expect(d[3]).toBeLessThan(d[0] / 3)
    expect(rq.distortion).toBe(d[3])
  })

  it('reports the distortion its own codes give on the training rows', () => {
    const codes = rqEncode(rq, data)
    const X = toRows(data)
    for (const l of [1, 2, 4]) {
      const R = toRows(rqDecode(rq, codes, l))
      const mse = X.reduce((a, x, i) => a + x.reduce((b, v, j) => b + (v - R[i][j]) ** 2, 0), 0) / X.length
      expect(mse).toBeCloseTo(rq.distortionByLevel[l - 1], 9)
    }
  })

  it('with one level is the k-means codebook', () => {
    // Level 0 seeds its k-means from child(stream, 'level', 0).
    const one = residualQuantiser(data, { levels: 1, codewords: 8, stream: stream(12) })
    const direct = trainCodebook(data, 8, { stream: child(stream(12), 'level', 0) })
    expect(one.distortion).toBeCloseTo(direct.inertia / 500, 9)
    expect(Array.from(toFlat(one.codebooks))).toEqual(Array.from(toFlat(direct.centroids)))
  })

  it('rejects bad levels and codewords', () => {
    expect(() => residualQuantiser(data, { levels: 0, codewords: 4, stream: stream(1) })).toThrow(/levels/)
    expect(() => residualQuantiser(data, { levels: 2, codewords: 0, stream: stream(1) })).toThrow(/codewords/)
  })
})

describe('rqEncode and rqDecode', () => {
  it('encode greedily: each level picks the codeword nearest to the residual so far', () => {
    const point = [[0.3, -1.2, 0.8]]
    const code = Array.from(toFlat(rqEncode(rq, point)))
    const B = toFlat(rq.codebooks)
    let r = point[0].slice()
    for (let l = 0; l < 4; l++) {
      const dist = (k: number) => r.reduce((a, v, j) => a + (v - B[(l * 8 + k) * 3 + j]) ** 2, 0)
      const best = Array.from({ length: 8 }, (_, k) => k).reduce((a, k) => (dist(k) < dist(a) ? k : a), 0)
      expect(code[l]).toBe(best)
      r = r.map((v, j) => v - B[(l * 8 + best) * 3 + j])
    }
  })

  it('decode sums codewords over the leading levels; zero levels is the origin', () => {
    const B = toFlat(rq.codebooks)
    const sumOf = (levels: number) =>
      [0, 1, 2].map((j) => [3, 1, 4, 1].slice(0, levels).reduce((a, k, l) => a + B[(l * 8 + k) * 3 + j], 0))
    expect(Array.from(toFlat(rqDecode(rq, [[3, 1, 4, 1]], 0)))).toEqual([0, 0, 0])
    for (const l of [1, 3, 4]) {
      const got = toFlat(rqDecode(rq, [[3, 1, 4, 1]], l))
      sumOf(l).forEach((v, j) => expect(got[j]).toBeCloseTo(v, 12))
    }
    expect(() => rqDecode(rq, [[8, 0, 0, 0]])).toThrow(/codeword/)
    expect(() => rqDecode(rq, [[0, 0]])).toThrow(/levels/)
  })
})

describe('codePrefixTree', () => {
  const codes = [
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
    [0, 2],
    [1, 0],
  ]
  const tree = codePrefixTree(codes)

  it('has a root of every row, then the occupied prefixes breadth first, children in code order', () => {
    expect(tree.map((n) => n.prefix)).toEqual([[], [0], [1], [0, 1], [0, 2], [1, 0]])
    expect(tree.map((n) => n.count)).toEqual([6, 3, 3, 2, 1, 3])
    expect(tree[0].children).toEqual([1, 2])
    expect(tree[1].children).toEqual([3, 4])
    expect(tree[5].parent).toBe(2)
  })

  it('gives each node the span of its rows in lexicographic order, nested in its parent’s', () => {
    expect(tree.map((n) => n.offset)).toEqual([0, 0, 3, 0, 2, 3])
    for (const n of tree.slice(1)) {
      const p = tree[n.parent]
      expect(n.offset).toBeGreaterThanOrEqual(p.offset)
      expect(n.offset + n.count).toBeLessThanOrEqual(p.offset + p.count)
    }
    // Each level's counts add up to every row.
    for (const d of [1, 2]) expect(tree.filter((n) => n.depth === d).reduce((a, n) => a + n.count, 0)).toBe(6)
  })

  it('rejects codes that are not non-negative integers', () => {
    expect(() => codePrefixTree([[0, -1]])).toThrow(/non-negative/)
  })
})
