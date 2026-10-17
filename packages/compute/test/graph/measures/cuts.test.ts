/** The Cheeger constant (aifn-compute/graph/measures), against brute force and the site's self-check. */
import { describe, expect, it } from 'vitest'
import { cheegerConstant } from 'aifn-compute/graph/measures'
import { barbellGraph, chainGraph, completeGraph, erdosRenyiGraph } from 'aifn-compute/graph/structures'
import { adjacencyMatrix } from 'aifn-compute/graph/matrices'
import { fromEdges, type Graph } from 'aifn-compute/graph'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

/** The site candidate's enumeration: every mask without the last node, in increasing order, first minimum kept. */
function bruteForce(g: Graph): { value: number; set: number[] } {
  const n = g.nodes
  const a = toFlat(adjacencyMatrix(g))
  const d = Array.from({ length: n }, (_, i) => a.slice(i * n, i * n + n).reduce((s, v) => s + v, 0))
  const total = d.reduce((s, v) => s + v, 0)
  let best = Infinity
  let bestMask = 0
  for (let mask = 1; mask < (1 << n) - 1; mask++) {
    if (mask & (1 << (n - 1))) continue
    let cut = 0
    let vol = 0
    for (let i = 0; i < n; i++)
      if (mask & (1 << i)) {
        vol += d[i]
        for (let j = 0; j < n; j++) if (!(mask & (1 << j))) cut += a[i * n + j]
      }
    const denom = Math.min(vol, total - vol)
    const h = denom > 0 ? cut / denom : Infinity
    if (h < best) {
      best = h
      bestMask = mask
    }
  }
  return { value: best, set: Array.from({ length: n }, (_, i) => i).filter((i) => bestMask & (1 << i)) }
}

describe('cheegerConstant', () => {
  it('barbell(5, 3): 1/23 (site check)', () => {
    expect(cheegerConstant(barbellGraph(5, 3)).value).toBeCloseTo(1 / 23, 12)
  })
  it('matches brute force, value and set, on small graphs', () => {
    for (const g of [chainGraph(4), completeGraph(5), barbellGraph(3, 1), erdosRenyiGraph(stream(2), 9, 0.4)]) {
      const mine = cheegerConstant(g)
      const want = bruteForce(g)
      expect(mine.value).toBeCloseTo(want.value, 12)
      expect(Array.from(toFlat(mine.set))).toEqual(want.set)
    }
  })
  it('is 0 for a disconnected graph', () => {
    expect(
      cheegerConstant(
        fromEdges(
          4,
          [
            [0, 1],
            [2, 3],
          ],
          { directed: false },
        ),
      ).value,
    ).toBe(0)
  })
  it('throws for a directed graph or too many nodes', () => {
    expect(() => cheegerConstant(fromEdges(3, [[0, 1]]))).toThrow()
    expect(() => cheegerConstant(chainGraph(23))).toThrow()
  })
})
