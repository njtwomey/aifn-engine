/** Message passing on a graph (aifn-compute/graph/propagation): sum, mean and max aggregation, and its gradient. */
import { describe, expect, it } from 'vitest'
import { messageEdges, propagate } from 'aifn-compute/graph/propagation'
import { fromEdges } from 'aifn-compute/graph'
import { grad, vmap } from 'aifn-compute/foundation/autodiff'
import { sum, tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

// 0 → 2 (w 2), 1 → 2 (w 1), 2 → 0 (w 1).
const g = fromEdges(3, [
  [0, 2, 2],
  [1, 2, 1],
  [2, 0, 1],
])
const h = tensor([
  [1, 10],
  [2, 20],
  [3, 30],
])

describe('propagate', () => {
  it('sum, mean and max of weighted source features; a node with no incoming edge gets zeros', () => {
    expect(toRows(propagate(g, h) as Tensor)).toEqual([
      [3, 30],
      [0, 0],
      [4, 40],
    ])
    expect(toRows(propagate(g, h, { aggregate: 'mean' }) as Tensor)).toEqual([
      [3, 30],
      [0, 0],
      [2, 20],
    ])
    expect(toRows(propagate(g, h, { aggregate: 'max' }) as Tensor)).toEqual([
      [3, 30],
      [0, 0],
      [2, 20],
    ])
    expect(toRows(propagate(g, h, { selfLoops: true }) as Tensor)[1]).toEqual([2, 20])
  })
  it('messageEdges lists both directions of an undirected edge', () => {
    const u = fromEdges(2, [[0, 1]], { directed: false })
    const m = messageEdges(u)
    expect(Array.from(m.source)).toEqual([0, 1])
    expect(Array.from(m.destination)).toEqual([1, 0])
  })
  it('is differentiable in the features, and batches with vmap', () => {
    // ∂/∂h Σ propagate(h) = the weighted out-degree of each node, per feature.
    const gr = grad((x: Value) => sum(propagate(g, x)))(h) as Tensor
    expect(toRows(gr)).toEqual([
      [2, 2],
      [1, 1],
      [1, 1],
    ])
    const batch = tensor([toRows(h), toRows(h).map((r) => r.map((v) => -v))])
    const out = vmap((x: Value) => propagate(g, x))(batch) as Tensor
    expect(toFlat(out).slice(0, 6)).toEqual(toFlat(propagate(g, h) as Tensor))
  })
})
