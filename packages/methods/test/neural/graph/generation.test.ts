/** Graph sequences (aifn-methods/neural/graph), against the site's self-check (graph-sequences). */
import { describe, expect, it } from 'vitest'
import { adjacencySequence, sequenceBandwidth } from 'aifn-methods/neural/graph'
import { cycleGraph, gridGraph } from 'aifn-compute/graph/structures'
import { breadthFirstSearch } from 'aifn-compute/graph/traversal'
import { permutation, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'

/** The site's breadth-first ordering: from the root, then the other nodes as later roots. */
const bfs = (g: Graph, root = 0) => {
  const rest = Array.from({ length: g.nodes }, (_, v) => v).filter((v) => v !== root)
  return Array.from(toFlat(breadthFirstSearch(g, [root, ...rest]).order))
}
const total = (rows: number[][]) => rows.flat().reduce((a, b) => a + b, 0)

describe('adjacencySequence', () => {
  it('the 4-cycle from root 0: order [0, 1, 3, 2], rows [[], [1], [1, 0], [0, 1, 1]], bandwidth 2', () => {
    const c4 = cycleGraph(4)
    const order = bfs(c4)
    expect(order).toEqual([0, 1, 3, 2])
    const rows = adjacencySequence(c4, order)
    expect(rows).toEqual([[], [1], [1, 0], [0, 1, 1]])
    expect(sequenceBandwidth(rows)).toBe(2)
  })
  it('the 3 x 4 grid: every edge counted once under BFS and random orderings, BFS bandwidth at most 4', () => {
    const grid = gridGraph(3, 4)
    const random = permutation(stream(1), 12)
    expect(total(adjacencySequence(grid, bfs(grid)))).toBe(17)
    expect(total(adjacencySequence(grid, random))).toBe(17)
    expect(sequenceBandwidth(adjacencySequence(grid, bfs(grid)))).toBeLessThanOrEqual(4)
  })
  it('throws for an ordering that is not a permutation of the nodes', () => {
    expect(() => adjacencySequence(cycleGraph(4), [0, 1, 1, 2])).toThrow()
    expect(() => adjacencySequence(cycleGraph(4), [0, 1, 2])).toThrow()
  })
})
