/** aifn-methods/unsupervised/embedding/graph: node2vec walks, their co-occurrence counts and skip-gram on them. */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { fromEdges } from 'aifn-compute/graph'
import {
  node2vecTransition,
  node2vecWalks,
  skipGram,
  walkCooccurrence,
} from 'aifn-methods/unsupervised/embedding/graph'

const flat = (t: Parameters<typeof toFlat>[0]) => Array.from(toFlat(t))
// Node 2 has neighbours 1, 0 and 3; node 0 is a neighbour of 1, node 3 is not.
const kite = fromEdges(
  4,
  [
    [0, 1],
    [1, 2],
    [0, 2],
    [2, 3],
  ],
  { directed: false },
)

describe('node2vecTransition', () => {
  it('with p = q = 1 is proportional to the edge weights', () => {
    const g = fromEdges(
      3,
      [
        [0, 1, 1],
        [0, 2, 3],
      ],
      { directed: false },
    )
    const t = node2vecTransition(g, null, 0, 1, 1)
    expect(t.to).toEqual([1, 2])
    expect(t.probability).toEqual([0.25, 0.75])
    expect(node2vecTransition(g, 1, 0, 1, 1).probability).toEqual([0.25, 0.75])
  })

  it('weights the return by 1/p, a common neighbour by 1 and an outward move by 1/q', () => {
    // At 2 from 1: back to 1 (1/p), to 0 (a neighbour of 1, weight 1), out to 3 (1/q).
    const p = 0.5
    const q = 4
    const t = node2vecTransition(kite, 1, 2, p, q)
    const w = [1 / p, 1, 1 / q]
    const total = w.reduce((a, b) => a + b, 0)
    const byNode = new Map(t.to.map((x, k) => [x, t.probability[k]]))
    expect(byNode.get(1)).toBeCloseTo(w[0] / total, 15)
    expect(byNode.get(0)).toBeCloseTo(w[1] / total, 15)
    expect(byNode.get(3)).toBeCloseTo(w[2] / total, 15)
    expect(t.probability.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 15)
  })

  it('reports no move, and throws for zero total weight instead of returning NaN', () => {
    const isolated = fromEdges(2, [], { directed: false })
    expect(node2vecTransition(isolated, null, 0, 1, 1)).toEqual({ to: [], probability: [] })
    const zero = fromEdges(2, [[0, 1, 0]], { directed: false })
    expect(() => node2vecTransition(zero, null, 0, 1, 1)).toThrow(/zero weight/)
    expect(() => node2vecTransition(kite, null, 0, 0, 1)).toThrow(/positive/)
  })
})

describe('node2vecWalks', () => {
  it('starts one walk per node per pass, moves along edges, and is deterministic in its stream', () => {
    const walks = node2vecWalks(stream(3), kite, { walksPerNode: 3, length: 12, p: 0.5, q: 2 })
    expect(walks.length).toBe(12)
    for (let r = 0; r < 3; r++)
      expect(
        walks
          .slice(r * 4, r * 4 + 4)
          .map((w) => w[0])
          .sort(),
      ).toEqual([0, 1, 2, 3])
    const edges = new Set(['0-1', '1-2', '0-2', '2-3'])
    for (const w of walks) {
      expect(w.length).toBe(12)
      for (let j = 1; j < w.length; j++) {
        const key = `${Math.min(w[j - 1], w[j])}-${Math.max(w[j - 1], w[j])}`
        expect(edges.has(key), key).toBe(true)
      }
    }
    expect(node2vecWalks(stream(3), kite, { walksPerNode: 3, length: 12, p: 0.5, q: 2 })).toEqual(walks)
  })

  it('returns to the previous node at the rate 1/p predicts (node 3 has the single neighbour 2)', () => {
    // Steps out of 2 that arrived from 3: back to 3 with weight 1/p, to 0 or 1 with weight 1/q each.
    const p = 0.25
    const q = 1
    const walks = node2vecWalks(stream(5), kite, { walksPerNode: 20, length: 200, p, q })
    let back = 0
    let total = 0
    for (const w of walks)
      for (let j = 2; j < w.length; j++)
        if (w[j - 2] === 3 && w[j - 1] === 2) {
          total++
          if (w[j] === 3) back++
        }
    expect(total).toBeGreaterThan(500)
    expect(Math.abs(back / total - 1 / p / (1 / p + 2 / q))).toBeLessThan(0.05)
  })

  it('stops a walk at a node with no neighbours', () => {
    const g = fromEdges(3, [[0, 1]], { directed: true })
    const walks = node2vecWalks(stream(1), g, { walksPerNode: 1, length: 5 })
    const from0 = walks.find((w) => w[0] === 0)!
    expect(from0).toEqual([0, 1])
    expect(walks.find((w) => w[0] === 2)).toEqual([2])
  })
})

describe('walkCooccurrence', () => {
  it('counts each pair within the window both ways; the total is the number of skip-gram pairs', () => {
    expect(flat(walkCooccurrence([[0, 1, 2, 1]], 3, 1))).toEqual([0, 1, 0, 1, 0, 2, 0, 2, 0])
    const walks = node2vecWalks(stream(2), kite, { walksPerNode: 2, length: 10 })
    const total = flat(walkCooccurrence(walks, 4, 3)).reduce((a, b) => a + b, 0)
    expect(total).toBe(skipGram(stream(4), walks, 4, { window: 3, dim: 2, negatives: 0 }).pairs)
  })
})

describe('skipGram', () => {
  it('is deterministic, lowers the loss, and approaches the shifted PMI log #(w, c)|D| / (k #(w) #(c))', () => {
    const walks = node2vecWalks(stream(7), kite, { walksPerNode: 20, length: 40 })
    const k = 2
    const fit = skipGram(stream(8), walks, 4, { dim: 4, window: 2, negatives: k, epochs: 8, rate: 0.05 })
    expect(fit.loss.at(-1)!).toBeLessThan(fit.loss[0])
    const again = skipGram(stream(8), walks, 4, { dim: 4, window: 2, negatives: k, epochs: 8, rate: 0.05 })
    expect(flat(again.input)).toEqual(flat(fit.input))
    // Levy and Goldberg (2014): at the optimum of SGNS, u_c . v_w = PMI(w, c) - log k, with the unigram noise.
    const counts = flat(walkCooccurrence(walks, 4, 2))
    const row = [0, 1, 2, 3].map((i) => counts.slice(i * 4, i * 4 + 4).reduce((a, b) => a + b, 0))
    const total = row.reduce((a, b) => a + b, 0)
    const input = flat(fit.input)
    const output = flat(fit.output)
    // The ordering of the inner products follows the ordering of the shifted PMI for the strongest and weakest pairs.
    const dot = (w: number, c: number) => [0, 1, 2, 3].reduce((a, j) => a + input[w * 4 + j] * output[c * 4 + j], 0)
    const pmi = (w: number, c: number) => Math.log((counts[w * 4 + c] * total) / (k * row[w] * row[c]))
    const pairs = [
      [3, 2],
      [0, 1],
      [3, 0],
    ].filter(([w, c]) => counts[w * 4 + c] > 0)
    const byPmi = [...pairs].sort((a, b) => pmi(b[0], b[1]) - pmi(a[0], a[1]))
    expect(dot(byPmi[0][0], byPmi[0][1])).toBeGreaterThan(dot(byPmi.at(-1)![0], byPmi.at(-1)![1]))
  })

  it('refuses a bad dimension', () => {
    expect(() => skipGram(stream(1), [[0, 1]], 2, { dim: 0 })).toThrow(/dim/)
  })
})
