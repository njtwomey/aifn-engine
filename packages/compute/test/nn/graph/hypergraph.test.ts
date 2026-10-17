/** Hypergraph message passing (aifn-compute/nn/graph), against the site's self-check (hypergraph-conv). */
import { describe, expect, it } from 'vitest'
import { hgnnConv, hypergraphMeanPass } from 'aifn-compute/nn/graph'
import { cliqueExpansion } from 'aifn-compute/graph/structures'
import { propagate } from 'aifn-compute/graph/propagation'
import { grad } from 'aifn-compute/foundation/autodiff'
import { sum, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

/** The site's coauthor hypergraphs: six authors with features 1 to 6. */
const PAPERS = [
  [0, 1, 2],
  [2, 3],
  [3, 4, 5],
]
const SPLIT = [
  [0, 1],
  [1, 2],
  [0, 2],
  [2, 3],
  [3, 4, 5],
]
const x = tensor([[1], [2], [3], [4], [5], [6]])
const flat = (v: Value) => Array.from(toFlat(v as Tensor))
const close = (got: number[], want: number[]) => want.forEach((w, i) => expect(got[i]).toBeCloseTo(w, 12))

describe('hypergraphMeanPass', () => {
  it('hyperedge means (2, 3.5, 5), node means (2, 2, 2.75, 4.25, 5, 5)', () => {
    const pass = hypergraphMeanPass(x, PAPERS)
    close(flat(pass.edges), [2, 3.5, 5])
    close(flat(pass.nodes), [2, 2, 2.75, 4.25, 5, 5])
  })
  it('the split hypergraph: node means (1.75, 2, 8/3, 4.25, 5, 5)', () => {
    close(flat(hypergraphMeanPass(x, SPLIT).nodes), [1.75, 2, 8 / 3, 4.25, 5, 5])
  })
  it('the clique expansion with self-loops averages to (2, 2, 2.5, 4.5, 5, 5)', () => {
    const mean = propagate(cliqueExpansion(6, PAPERS), x, { aggregate: 'mean', selfLoops: true })
    close(flat(mean), [2, 2, 2.5, 4.5, 5, 5])
  })
})

describe('hgnnConv', () => {
  it('with Theta = 1: (1.7071, 1.7071, 2.9571, 5.0094, 4.6095, 4.6095), as computed by hand', () => {
    const s2 = Math.SQRT2
    const e1 = (1 + 2 + 3 / s2) / 3
    const e2 = (3 / s2 + 4 / s2) / 2
    const e3 = (4 / s2 + 5 + 6) / 3
    const y = flat(hgnnConv(x, PAPERS, tensor([[1]])))
    close(y, [e1, e1, (e1 + e2) / s2, (e2 + e3) / s2, e3, e3])
    ;[1.7071, 1.7071, 2.9571, 5.0094, 4.6095, 4.6095].forEach((v, i) => expect(y[i]).toBeCloseTo(v, 4))
  })
  it('is differentiable in the weight, and rejects an empty hyperedge', () => {
    const g = grad((theta: Value) => sum(hgnnConv(x, PAPERS, theta)))(tensor([[1]])) as Tensor
    expect(Number.isFinite(toFlat(g)[0])).toBe(true)
    expect(() => hgnnConv(x, [[0, 1], []], tensor([[1]]))).toThrow()
  })
})
