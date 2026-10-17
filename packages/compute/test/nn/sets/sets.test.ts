/** Set pooling (aifn-compute/nn/sets), against the site's self-check (set-networks). */
import { describe, expect, it } from 'vitest'
import { criticalSet, padSets, setPool } from 'aifn-compute/nn/sets'
import { grad } from 'aifn-compute/foundation/autodiff'
import { sum, tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

const rows = (v: Value) => toRows(v as Tensor) as number[][]
const SETS = [
  [
    [1, 2],
    [2, 3],
    [4, 4],
  ],
  [[-3, -1]],
]

describe('padSets', () => {
  it('shape [2, 3, 2], mask [[1, 1, 1], [1, 0, 0]]', () => {
    const p = padSets(SETS)
    expect(p.x.shape).toEqual([2, 3, 2])
    expect(rows(p.mask)).toEqual([
      [1, 1, 1],
      [1, 0, 0],
    ])
    expect(p.sizes).toEqual([3, 1])
  })
  it('throws for elements of different lengths', () => {
    expect(() => padSets([[[1, 2], [3]]])).toThrow()
  })
})

describe('setPool', () => {
  // The site's check: sets of sizes 3 and 1.
  const sets = [
    [
      [1, 5],
      [4, 2],
      [2, 2],
    ],
    [[-3, -1]],
  ]
  const { x, mask } = padSets(sets)
  it('sum [[7, 9], [-3, -1]], mean [[2.333333, 3], [-3, -1]], max [[4, 5], [-3, -1]]', () => {
    expect(rows(setPool(x, mask, 'sum'))).toEqual([
      [7, 9],
      [-3, -1],
    ])
    const mean = rows(setPool(x, mask, 'mean'))
    ;[7 / 3, 3, -3, -1].forEach((v, k) => expect(mean[Math.floor(k / 2)][k % 2]).toBeCloseTo(v, 12))
    expect(rows(setPool(x, mask, 'max'))).toEqual([
      [4, 5],
      [-3, -1],
    ])
  })
  it('every pool is invariant to the order of the elements', () => {
    const shuffled = padSets([[sets[0][2], sets[0][0], sets[0][1]], sets[1]])
    for (const p of ['sum', 'mean', 'max'] as const) {
      const a = rows(setPool(shuffled.x, shuffled.mask, p)).flat()
      rows(setPool(x, mask, p))
        .flat()
        .forEach((v, k) => expect(a[k]).toBeCloseTo(v, 12))
    }
  })
  it('padding never wins the maximum, even below the padding value', () => {
    const neg = padSets([[[-5]], [[-7], [-9]]])
    expect(Array.from(toFlat(setPool(neg.x, neg.mask, 'max') as Tensor))).toEqual([-5, -7])
  })
  it('the maximum passes the gradient to the winner only', () => {
    const g = grad((h: Value) => sum(setPool(h, mask, 'max')))(x) as Tensor
    // Channel 0's maximum is element 1 of set 0, channel 1's element 0; set 1's only element wins both.
    expect(Array.from(toFlat(g))).toEqual([0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
  })
  it('throws for an empty set', () => {
    const p = padSets([[[1]], []])
    expect(() => setPool(p.x, p.mask, 'mean')).toThrow()
  })
})

describe('criticalSet', () => {
  it('[[1, 5], [4, 2], [2, 2]]: elements 0 and 1', () => {
    expect(
      Array.from(
        toFlat(
          criticalSet(
            tensor([
              [1, 5],
              [4, 2],
              [2, 2],
            ]),
          ),
        ),
      ),
    ).toEqual([0, 1])
  })
})
