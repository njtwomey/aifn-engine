/** Readout and pooling (aifn-compute/nn/graph), against the site's self-check (graph-pooling). */
import { describe, expect, it } from 'vitest'
import { attentionPool, batchGraphs, diffPool, globalPool, topKPool } from 'aifn-compute/nn/graph'
import { fromEdges } from 'aifn-compute/graph'
import { tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

/** The entries of a value; a full reduction (the losses) is a plain number. */
const flat = (v: Value) => (typeof v === 'number' ? [v] : Array.from(toFlat(v as Tensor)))
const close = (got: number[], want: number[], digits = 4) =>
  want.forEach((w, i) => expect(got[i]).toBeCloseTo(w, digits))

/** The note's worked example: the triangle 0-1-2 with a tail 2-3, embeddings (6, 6, 10, 7). */
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
const h = tensor([[6], [6], [10], [7]])
const one = Int32Array.from([0, 0, 0, 0])

describe('readout', () => {
  it('sum 29, mean 7.25, max 10', () => {
    expect(flat(globalPool(h, one, 1))).toEqual([29])
    expect(flat(globalPool(h, one, 1, 'mean'))).toEqual([7.25])
    expect(flat(globalPool(h, one, 1, 'max'))).toEqual([10])
  })
  it('attention weights (0.2175, 0.2175, 0.3245, 0.2404), readout 7.5385', () => {
    const att = attentionPool(h, tensor([[0.6], [0.6], [1.0], [0.7]]), one, 1)
    close(flat(att.weights), [0.2175, 0.2175, 0.3245, 0.2404])
    close(flat(att.pooled), [7.5385])
  })
})

describe('batchGraphs', () => {
  it('two copies: 8 nodes, 8 edges, batch [0, 0, 0, 0, 1, 1, 1, 1]; sums 29 and 29', () => {
    const twice = batchGraphs([g, g])
    expect([twice.graph.nodes, twice.graph.edges.length]).toEqual([8, 8])
    expect(Array.from(twice.batch)).toEqual([0, 0, 0, 0, 1, 1, 1, 1])
    expect(twice.offsets).toEqual([0, 4])
    const h2 = tensor([[6], [6], [10], [7], [6], [6], [10], [7]])
    const joined = Int32Array.from([0, 0, 0, 0, 0, 0, 0, 0])
    const modes = ['sum', 'mean', 'max'] as const
    expect(modes.map((m) => flat(globalPool(h2, joined, 1, m))[0])).toEqual([58, 7.25, 10])
    expect(flat(globalPool(h2, twice.batch, 2))).toEqual([29, 29])
  })
  it('keeps direction, weights and labels; throws on a mix of directed and undirected graphs', () => {
    const a = fromEdges(2, [[0, 1, 3]], { labels: ['a', 'b'] })
    const b = fromEdges(1, [], { labels: ['c'] })
    const batch = batchGraphs([a, b])
    expect(batch.graph.directed).toBe(true)
    expect(batch.graph.edges[0].weight).toBe(3)
    expect(batch.graph.labels).toEqual(['a', 'b', 'c'])
    expect(() => batchGraphs([a, g])).toThrow()
  })
})

describe('diffPool', () => {
  // Logits whose row softmax is (1, 0), (1, 0), (0.5, 0.5), (0, 1) up to e^-40.
  const logits = tensor([
    [40, 0],
    [40, 0],
    [0, 0],
    [0, 40],
  ])
  const A = tensor([
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0],
  ])
  it('S^T A S = [[4, 1.5], [1.5, 1]], S^T Z = (17, 12), link loss 4.75 / 16, entropy ln 2 / 4', () => {
    const d = diffPool(A, h, logits)
    close(flat(d.adjacency), [4, 1.5, 1.5, 1], 6)
    close(flat(d.features), [17, 12])
    close(flat(d.linkLoss), [0.2969])
    close(flat(d.entropy), [Math.log(2) / 4])
  })
  it('S^T A S is invariant under relabelling the nodes', () => {
    const perm = [3, 0, 2, 1]
    const rows = (t: Tensor) => toRows(t) as number[][]
    const Ap = tensor(perm.map((i) => perm.map((j) => rows(A)[i][j])))
    const Lp = tensor(perm.map((i) => rows(logits)[i]))
    const hp = tensor(perm.map((i) => rows(h)[i]))
    close(flat(diffPool(Ap, hp, Lp).adjacency), flat(diffPool(A, h, logits).adjacency), 10)
  })
})

describe('topKPool', () => {
  it('scores (0.2, -0.5, 0.9, 0.4) keep nodes 2 and 3, gate their rows by tanh, keep the edge between them', () => {
    const tk = topKPool(g, tensor([[0.2], [-0.5], [0.9], [0.4]]), tensor([[1]]), 2)
    expect(Array.from(toFlat(tk.kept))).toEqual([2, 3])
    close(flat(tk.features), [0.9 * Math.tanh(0.9), 0.4 * Math.tanh(0.4)], 12)
    expect(tk.graph.edges.map((e) => [e.from, e.to])).toEqual([[0, 1]])
    expect(tk.graph.labels).toEqual(['2', '3'])
  })
  it('throws for k outside 1 to V', () => {
    expect(() => topKPool(g, h, tensor([[1]]), 5)).toThrow()
  })
})
