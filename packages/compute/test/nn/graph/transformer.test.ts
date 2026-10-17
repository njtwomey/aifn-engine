/** Graphormer's spatial bias (aifn-compute/nn/graph), against the site's self-check (depth-long-range). */
import { describe, expect, it } from 'vitest'
import { graphormerAttention, spatialBias } from 'aifn-compute/nn/graph'
import { hopDistances } from 'aifn-compute/graph/traversal'
import { fromEdges } from 'aifn-compute/graph'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'

/** The triangle 0-1-2 with a tail 2-3. */
const g = fromEdges(
  4,
  [
    [0, 1],
    [0, 2],
    [1, 2],
    [2, 3],
  ],
  { directed: false },
)
const w = tensor([[0.5]])
const head = { query: w, key: w, value: w }
const h = tensor([[1], [2], [3], [4]])

describe('graphormerAttention', () => {
  it('row 0 without a bias, and with the table (0, 0.5, -1)', () => {
    const d = hopDistances(g)
    const row0 = (table: number[]) =>
      (toRows(graphormerAttention(h, head, spatialBias(d, table)).weights as Tensor) as number[][])[0]
    const plain = row0([0])
    ;[0.165, 0.212, 0.273, 0.35].forEach((v, k) => expect(plain[k]).toBeCloseTo(v, 3))
    const biased = row0([0, 0.5, -1])
    ;[0.151, 0.32, 0.411, 0.118].forEach((v, k) => expect(biased[k]).toBeCloseTo(v, 3))
  })
})

describe('spatialBias', () => {
  it('clips long distances to the last entry and marks unreachable pairs', () => {
    const b = toFlat(spatialBias([[0, 1, 5, Infinity]], [0, 1, 2]))
    expect(Array.from(b)).toEqual([0, 1, 2, -Infinity])
    expect(Array.from(toFlat(spatialBias([[Infinity]], [0], -9)))).toEqual([-9])
  })
  it('throws for an empty table or a distance that is not a hop count', () => {
    expect(() => spatialBias([[0]], [])).toThrow()
    expect(() => spatialBias([[0.5]], [0])).toThrow()
  })
})
