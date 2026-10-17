/** Computation graphs (aifn-compute/nn/graph), against the site's self-check (computation-graph). */
import { describe, expect, it } from 'vitest'
import { computationGraph, sampledComputationGraph } from 'aifn-compute/nn/graph'
import { barabasiAlbertGraph } from 'aifn-compute/graph/structures'
import { fromEdges } from 'aifn-compute/graph'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

/** The triangle 0-1-2 with a tail 2-3 (the message-passing note's graph). */
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

describe('computationGraph', () => {
  it('from node 3, two layers: touched [1, 2, 4]', () => {
    const c = computationGraph(g, [3], 2)
    expect(c.touched).toEqual([1, 2, 4])
    expect(Array.from(toFlat(c.nodes)).sort()).toEqual([0, 1, 2, 3])
  })
  it('follows messages backwards on a directed graph', () => {
    const d = fromEdges(3, [
      [0, 1],
      [1, 2],
    ])
    expect(computationGraph(d, [2], 2).touched).toEqual([1, 2, 3])
    expect(computationGraph(d, [0], 2).touched).toEqual([1, 1, 1])
  })
})

describe('sampledComputationGraph', () => {
  it('a fan-out at least the largest degree keeps every neighbour', () => {
    expect(sampledComputationGraph(g, [3], [3, 3], stream(0)).touched).toEqual([1, 2, 4])
  })
  it('one neighbour per node: node 3 keeps 2', () => {
    expect(sampledComputationGraph(g, [3], [1, 1], stream(0)).touched.slice(0, 2)).toEqual([1, 2])
  })
  it('on a Barabasi-Albert graph, sampling needs fewer nodes than the full neighbourhood', () => {
    const ba = barabasiAlbertGraph(stream(7), 300, 3)
    const targets = Array.from({ length: 16 }, (_, i) => 100 + i)
    const full = computationGraph(ba, targets, 3)
    const sampled = sampledComputationGraph(ba, targets, [2, 2, 2], stream(9))
    expect(sampled.samples.length).toBe(3)
    expect(sampled.touched[3]).toBeLessThan(full.touched[3])
    expect(sampled.touched[0]).toBe(16)
  })
})
