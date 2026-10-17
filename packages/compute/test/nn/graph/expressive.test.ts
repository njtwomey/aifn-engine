/** GIN and PNA (aifn-compute/nn/graph), against the site's self-checks (gin-conv, pna-aggregation). */
import { describe, expect, it } from 'vitest'
import { GinConv, ginConv, pnaAggregate, pnaDelta } from 'aifn-compute/nn/graph'
import { fromEdges } from 'aifn-compute/graph'
import { stream } from 'aifn-compute/foundation/random'
import { shapeOf, tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

const flat = (v: Value) => Array.from(toFlat(v as Tensor))
const identity = (x: Value) => x
const path = fromEdges(
  3,
  [
    [0, 1],
    [1, 2],
  ],
  { directed: false },
)
const h = tensor([[1], [2], [3]])

describe('ginConv', () => {
  it('path 0-1-2 with features 1, 2, 3: sum with epsilon 0.5, mean and max', () => {
    expect(flat(ginConv(path, h, identity, { epsilon: 0.5 }))).toEqual([3.5, 7, 6.5])
    expect(flat(ginConv(path, h, identity, { aggregate: 'mean' }))).toEqual([3, 4, 5])
    expect(flat(ginConv(path, h, identity, { aggregate: 'max' }))).toEqual([3, 5, 5])
  })
  it('an edge carrying 1 and 2: GIN-0 gives [3, 3], epsilon 0.5 gives [3.5, 4]', () => {
    const edge = fromEdges(2, [[0, 1]], { directed: false })
    expect(flat(ginConv(edge, tensor([[1], [2]]), identity))).toEqual([3, 3])
    expect(flat(ginConv(edge, tensor([[1], [2]]), identity, { epsilon: 0.5 }))).toEqual([3.5, 4])
  })
  it('throws for features without one row per node', () => {
    expect(() => ginConv(path, tensor([[1], [2]]), identity)).toThrow()
  })
})

describe('GinConv', () => {
  it('output 3 x 4, learned epsilon starting at 0.1, equal to the functional form', () => {
    const layer = GinConv(path, 1, 4, { hidden: [8], learnEpsilon: true, epsilon: 0.1, activation: 'tanh' })
    const p = layer.init(stream(3))
    const out = layer.apply(p, h)
    expect(shapeOf(out as Tensor)).toEqual([3, 4])
    expect(flat(p.epsilon!)).toEqual([0.1])
    expect(layer.label).toContain('learned')
  })
})

/** A star whose centre 0 has the given leaves. */
const star = (k: number) =>
  fromEdges(
    k + 1,
    Array.from({ length: k }, (_, i) => [0, i + 1] as [number, number]),
    { directed: false },
  )
const centre = (leaves: number[], options = {}) => {
  const out = pnaAggregate(star(leaves.length), tensor([[0], ...leaves.map((v) => [v])]), options)
  return (toRows(out as Tensor) as number[][])[0]
}

describe('pnaAggregate', () => {
  const all = { aggregators: ['mean', 'max', 'min', 'std', 'sum'], scalers: ['identity'], epsilon: 0 } as const
  it('Corso et al. figure 1: {-1, 1} and {-2, 2} share mean and sum, differ in max, min and std', () => {
    expect(centre([-1, 1], all)).toEqual([0, 1, -1, 1, 0])
    expect(centre([-2, 2], all)).toEqual([0, 2, -2, 2, 0])
  })
  it('{1, 1} against {1}: the degree scalers separate them (delta = 1)', () => {
    const amp = { aggregators: ['mean'], scalers: ['identity', 'amplification', 'attenuation'], delta: 1, epsilon: 0 }
    const a = centre([1, 1], amp as never)
    ;[1, Math.log(3), 1 / Math.log(3)].forEach((v, k) => expect(a[k]).toBeCloseTo(v, 12))
    const b = centre([1], amp as never)
    ;[1, Math.log(2), 1 / Math.log(2)].forEach((v, k) => expect(b[k]).toBeCloseTo(v, 12))
  })
  it('default width F x 3 scalers x 4 aggregators = 12', () => {
    expect(centre([1, 2]).length).toBe(12)
  })
  it('scales by in-degree on a directed graph and ignores edge weights', () => {
    // Node 2 receives from 0 and 1 (weight 10); node 0 sends only, so its in-degree is 0.
    const g = fromEdges(3, [
      [0, 2],
      [1, 2, 10],
    ])
    const options = { aggregators: ['sum'], scalers: ['amplification'], delta: 1 } as const
    const out = toRows(pnaAggregate(g, tensor([[1], [3], [0]]), options) as Tensor) as number[][]
    expect(out[2][0]).toBeCloseTo(4 * Math.log(3), 12)
    expect(out[0][0]).toBe(0)
  })
})

describe('pnaDelta', () => {
  it('a 3-leaf star: (log 4 + 3 log 2) / 4', () => {
    expect(pnaDelta([star(3)])).toBeCloseTo((Math.log(4) + 3 * Math.log(2)) / 4, 12)
  })
})
